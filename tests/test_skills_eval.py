import io
import json
import os
import shlex
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cli import _normalise_eval_argv, _normalise_policy_argv, _normalise_skills_argv, build_parser
from agent_trace.models import EventType, TraceEvent
from agent_trace.skills_eval import (
    APPROVE_ALL_FLAGS,
    PRESETS,
    EvalCase,
    EvalInputError,
    agent_env,
    approve_all_flags,
    case_invocation,
    classify_run,
    cmd_skills_eval,
    find_session_dirs,
    hook_command_prefix,
    load_cases,
    parse_template,
    render_command,
    score_trace_dir,
    skill_matches,
    write_hook_configs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
T0 = 1_700_000_000.0
SKILL = "release-notes"


def _event(event_type, offset, parent_id="", **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=T0 + offset, parent_id=parent_id, data=data)


def _write_session(store: Path, sid: str, events: list[TraceEvent], *, meta: bool = True) -> Path:
    session_dir = store / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.ndjson").write_text("".join(event.to_json() + "\n" for event in events))
    if meta:
        (session_dir / "meta.json").write_text(json.dumps({"session_id": sid, "agent_name": "github-copilot"}))
    return session_dir


def _skill_tool_session(cwd: str, skill: str = SKILL, *, start: bool = True) -> list[TraceEvent]:
    events = [_event(EventType.SESSION_START, 0, cwd=cwd)] if start else []
    call = _event(EventType.TOOL_CALL, 2, tool_name="skill", arguments={"skill": skill})
    events += [
        _event(EventType.USER_PROMPT, 1, prompt="Draft the release notes"),
        call,
        _event(EventType.TOOL_RESULT, 3, parent_id=call.event_id, tool_name="skill", result="loaded"),
    ]
    return events


def _plain_session(cwd: str) -> list[TraceEvent]:
    return [
        _event(EventType.SESSION_START, 0, cwd=cwd),
        _event(EventType.USER_PROMPT, 1, prompt="Fix the typo"),
        _event(EventType.TOOL_CALL, 2, tool_name="view", arguments={"path": "CHANGELOG.md"}),
    ]


def _write_skill(repo: Path, name: str = SKILL, *, disable_model_invocation: bool = False) -> Path:
    skill_dir = repo / ".github" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    flag = "disable-model-invocation: true\n" if disable_model_invocation else ""
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: test skill\n{flag}---\n# {name}\n")
    return skill_dir / "SKILL.md"


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    args = build_parser().parse_args(_normalise_skills_argv(argv))
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cmd_skills_eval(args)
    return code, out.getvalue(), err.getvalue()


class _TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_cases(self, cases, name="cases.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(cases))
        return path


class LoadCasesTests(_TempDirTest):
    def test_accepts_list_and_object_forms(self):
        (self.root / "fixture").mkdir()
        case = {"id": "a", "skill": SKILL, "prompt": "Draft notes", "should_trigger": True,
                "fixture_dir": "fixture", "runs": 2, "note": "paraphrase"}
        listed = load_cases(self.write_cases([case], "list.json"))
        wrapped = load_cases(self.write_cases({"cases": [case]}, "wrapped.json"))

        self.assertEqual(listed, wrapped)
        self.assertEqual(listed[0].runs, 2)
        self.assertEqual(listed[0].fixture_dir, str((self.root / "fixture").resolve()))
        self.assertEqual(listed[0].note, "paraphrase")

    def test_reports_every_invalid_field(self):
        cases = [
            {"id": "ok", "skill": SKILL, "prompt": "p", "should_trigger": True},
            {"id": "ok", "skill": SKILL, "prompt": "p", "should_trigger": False},
            {"id": "../escape", "skill": SKILL, "prompt": "p", "should_trigger": True},
            {"id": "b", "skill": "", "prompt": " ", "should_trigger": "yes"},
            {"id": "c", "skill": SKILL, "prompt": "p", "should_trigger": True, "runs": 0},
            {"id": "d", "skill": SKILL, "prompt": "p", "should_trigger": True, "runs": True},
            {"id": "e", "skill": SKILL, "prompt": "p", "should_trigger": True, "query": "x"},
            "not an object",
        ]
        with self.assertRaises(EvalInputError) as ctx:
            load_cases(self.write_cases(cases))
        message = str(ctx.exception)
        for expected in (
            "case 2 (ok): duplicate id",
            "case 3 (../escape): id must be",
            "case 4 (b): skill must be",
            "case 4 (b): prompt must be",
            "case 4 (b): should_trigger must be true or false",
            "case 5 (c): runs must be a positive integer",
            "case 6 (d): runs must be a positive integer",
            "case 7 (e): unknown field(s): query",
            "case 8: must be an object",
        ):
            self.assertIn(expected, message)

    def test_rejects_bad_files(self):
        bad_json = self.root / "bad.json"
        bad_json.write_text("[{")
        for path in (bad_json, self.write_cases([], "empty.json"), self.write_cases({"x": 1}, "obj.json"),
                     self.root / "missing.json"):
            with self.subTest(path=path.name), self.assertRaises(EvalInputError):
                load_cases(path)

    def test_example_cases_file_is_valid(self):
        cases = load_cases(REPO_ROOT / "examples" / "skills-eval" / "cases.json")

        self.assertTrue(any(case.should_trigger for case in cases))
        self.assertTrue(any(not case.should_trigger for case in cases))
        for case in cases:
            self.assertTrue(Path(case.fixture_dir).is_dir(), case.id)
            self.assertEqual(case.explicit_invocation(), "", case.id)
        fixture = Path(cases[0].fixture_dir)
        self.assertTrue(any(fixture.glob(f".*/skills/{cases[0].skill}/SKILL.md")))

    def test_explicit_invocation_is_flagged(self):
        self.assertEqual(EvalCase("a", SKILL, f"/{SKILL} for v2", True).explicit_invocation(), "slash_command")
        self.assertEqual(EvalCase("a", SKILL, f"use ${SKILL} now", True).explicit_invocation(), "dollar_mention")
        self.assertEqual(EvalCase("a", SKILL, "draft release notes", True).explicit_invocation(), "")


class TemplateTests(_TempDirTest):
    def test_prompt_is_one_argv_token_and_never_shell_parsed(self):
        prompt = "notes; $(touch pwned) `id` 'quoted' \"double\" {workspace}"
        argv = render_command(parse_template("agent -p {prompt} --dir={workspace}"),
                              {"prompt": prompt, "workspace": "/tmp/w s"})

        self.assertEqual(argv, ["agent", "-p", prompt, "--dir=/tmp/w s"])

    def test_template_validation(self):
        for template, expected in (
            ("agent --flag", "must contain {prompt}"),
            ("agent {prompt} {home}", "unknown placeholder(s) in agent command: {home}"),
            ("{workspace}/agent {prompt}", "cannot be a placeholder"),
            ("", "empty"),
            ("agent 'unterminated {prompt}", "cannot parse"),
        ):
            with self.subTest(template=template), self.assertRaisesRegex(EvalInputError, ".*"):
                try:
                    parse_template(template)
                except EvalInputError as exc:
                    self.assertIn(expected, str(exc))
                    raise

    def test_presets_never_approve_every_tool(self):
        for name, template in PRESETS.items():
            with self.subTest(name=name):
                tokens = parse_template(template)
                self.assertEqual(approve_all_flags(tokens), [])
                self.assertFalse(any(token.startswith("--allow") for token in tokens))

    def test_detects_approve_all_flags(self):
        found = approve_all_flags(["copilot", "-p", "x", "--yolo", "--permission-mode", "bypassPermissions",
                                   "--allow-all-tools=true"])
        self.assertEqual(found, ["--yolo", "--allow-all-tools=true", "--permission-mode bypassPermissions"])
        self.assertIn("--dangerously-skip-permissions", APPROVE_ALL_FLAGS)

    def test_hook_configs_call_this_checkout(self):
        paths = write_hook_configs(self.root)
        prefix = hook_command_prefix()
        plugin = json.loads((Path(paths["hooks_plugin_dir"]) / "plugin.json").read_text())
        copilot = json.loads((Path(paths["hooks_plugin_dir"]) / "hooks.json").read_text())
        claude = json.loads(Path(paths["claude_settings"]).read_text())

        self.assertEqual(plugin["hooks"], "hooks.json")
        self.assertEqual(copilot["hooks"]["preToolUse"][0]["command"], f"{prefix} --provider copilot pre-tool")
        self.assertEqual(claude["hooks"]["PreToolUse"][0]["hooks"][0]["command"], f"{prefix} pre-tool")
        self.assertNotIn("agent-strace hook", json.dumps(copilot) + json.dumps(claude))
        self.assertIn(shlex.quote(sys.executable), prefix)

    def test_agent_env_scrubs_inherited_overrides(self):
        leaked = {
            "COPILOT_ALLOW_ALL": "1",
            "CLAUDECODE": "1",
            "AGENT_STRACE_ENDPOINT": "http://collector",
            "AGENT_STRACE_WORKSPACE": "other",
            "AGENT_TRACE_COPILOT_SESSION_ID": "abc",
            "AGENT_TRACE_DIR": "/elsewhere",
        }
        with mock.patch.dict(os.environ, leaked):
            env = agent_env(self.root / "traces")

        for key in leaked:
            if key != "AGENT_TRACE_DIR":
                self.assertNotIn(key, env)
        self.assertEqual(env["AGENT_TRACE_DIR"], str(self.root / "traces"))


class ScoreTraceDirTests(_TempDirTest):
    def setUp(self):
        super().setUp()
        self.repo = self.root / "repo"
        self.skill_file = _write_skill(self.repo)
        self.store = self.root / "traces"

    def score(self, skill=SKILL):
        return score_trace_dir(self.store, skill, workspace=str(self.repo), user_dirs=())

    def test_skill_tool_load(self):
        _write_session(self.store, "s1", _skill_tool_session(str(self.repo)))
        result = self.score()

        self.assertTrue(result["loaded"])
        self.assertEqual(result["methods"], ["skill_tool"])
        self.assertEqual(result["load_sessions"], [{"session": "s1", "has_meta": True, "has_session_start": True}])

    def test_skill_file_read_load(self):
        _write_session(self.store, "s1", [
            _event(EventType.SESSION_START, 0, cwd=str(self.repo)),
            _event(EventType.TOOL_CALL, 1, tool_name="view", arguments={"path": str(self.skill_file)}),
        ])
        result = self.score()

        self.assertTrue(result["loaded"])
        self.assertEqual(result["methods"], ["file_read"])

    def test_load_in_subagent_dir_without_meta_or_session_start(self):
        # Shape of a Copilot CLI subagent recorded before metadata was created lazily.
        _write_session(self.store, "parent", _plain_session(str(self.repo)))
        _write_session(self.store, "child", _skill_tool_session(str(self.repo), start=False), meta=False)
        result = self.score()

        self.assertTrue(result["loaded"])
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["sessions_without_meta"], 1)
        self.assertEqual(result["sessions_without_session_start"], 1)
        self.assertEqual(result["load_sessions"], [{"session": "child", "has_meta": False, "has_session_start": False}])

    def test_user_invoked_skill_is_not_a_model_load(self):
        _write_session(self.store, "s1", [
            _event(EventType.SESSION_START, 0, cwd=str(self.repo)),
            _event(EventType.USER_PROMPT, 1, prompt=f"/{SKILL} for v2"),
        ])
        result = self.score()

        self.assertFalse(result["loaded"])
        self.assertTrue(result["user_invoked"])

    def test_plugin_qualified_name_matches(self):
        _write_session(self.store, "s1", _skill_tool_session(str(self.repo), skill=f"docs-plugin:{SKILL}"))

        self.assertTrue(self.score()["loaded"])
        self.assertTrue(skill_matches(f"docs-plugin:{SKILL}", SKILL))
        self.assertTrue(skill_matches(SKILL, f"docs-plugin:{SKILL}"))
        self.assertFalse(skill_matches("release-notes-lite", SKILL))

    def test_other_skill_is_reported_separately(self):
        _write_session(self.store, "s1", _skill_tool_session(str(self.repo), skill="changelog-lint"))
        result = self.score()

        self.assertFalse(result["loaded"])
        self.assertEqual(result["other_skills_loaded"], ["changelog-lint"])

    def test_workspace_scoped_layout_and_bad_lines(self):
        session_dir = _write_session(self.store / "workspaces" / "ws1", "s1", _skill_tool_session(str(self.repo)))
        with open(session_dir / "events.ndjson", "a") as handle:
            handle.write("not json\n{\"event_type\": \"nope\"}\n")
        (self.store / "notes").mkdir(parents=True)

        self.assertEqual(find_session_dirs(self.store), [session_dir])
        self.assertTrue(self.score()["loaded"])

    def test_classify_run(self):
        self.assertEqual(classify_run({"loaded": True, "exit_code": 1, "events": 3})["status"], "loaded")
        self.assertEqual(classify_run({"loaded": False, "exit_code": 0, "events": 3})["status"], "not_loaded")
        self.assertEqual(classify_run({"loaded": False, "exit_code": None, "events": 3})["status"], "not_loaded")
        failed = classify_run({"loaded": False, "exit_code": 1, "events": 3, "error": "agent exited with status 1"})
        self.assertEqual((failed["status"], failed["unscored_reason"]), ("unscored", "agent exited with status 1"))
        empty = classify_run({"loaded": False, "exit_code": 0, "events": 0})
        self.assertEqual(empty["status"], "unscored")
        self.assertIn("no trace events", empty["unscored_reason"])


class FromTracesTests(_TempDirTest):
    def setUp(self):
        super().setUp()
        self.repo = self.root / "repo"
        _write_skill(self.repo)
        self.recorded = self.root / "recorded"
        cwd = str(self.repo)
        _write_session(self.recorded / "draft-notes" / "run-1", "s1", _skill_tool_session(cwd))
        _write_session(self.recorded / "draft-notes" / "run-2", "s2", _plain_session(cwd))
        _write_session(self.recorded / "draft-notes" / "run-2", "sub", _skill_tool_session(cwd, start=False),
                       meta=False)
        _write_session(self.recorded / "draft-notes" / "run-3", "s3", _plain_session(cwd))
        failed = self.recorded / "draft-notes" / "run-10"
        _write_session(failed, "s4", _plain_session(cwd))
        (failed / "run.json").write_text(json.dumps({"exit_code": 1, "error": "agent exited with status 1"}))
        _write_session(self.recorded / "fix-typo" / "run-1", "s5", _plain_session(cwd))
        _write_session(self.recorded / "fix-typo" / "run-2", "s6", _skill_tool_session(cwd))
        _write_session(self.recorded / "stray-case" / "run-1", "s7", _plain_session(cwd))
        self.cases = self.write_cases([
            {"id": "draft-notes", "skill": SKILL, "prompt": "Draft release notes", "should_trigger": True},
            {"id": "fix-typo", "skill": SKILL, "prompt": "Fix the changelog typo", "should_trigger": False},
            {"id": "not-recorded", "skill": SKILL, "prompt": "Summarise changes", "should_trigger": True},
        ])
        self._home = mock.patch.dict(os.environ, {"HOME": str(self.root / "home")})
        self._home.start()

    def tearDown(self):
        self._home.stop()
        super().tearDown()

    def run_eval(self, *extra):
        return _run_cli(["skills", "eval", str(self.cases), "--from-traces", str(self.recorded), *extra])

    def test_json_report_counts(self):
        code, out, err = self.run_eval("--format", "json")
        report = json.loads(out)
        cases = {case["id"]: case for case in report["cases"]}

        self.assertEqual(code, 0, err)
        self.assertEqual(report["schema_version"], 1)
        self.assertTrue(report["experimental"])
        self.assertEqual(report["mode"], "from_traces")
        self.assertNotIn("score", report)
        self.assertEqual([run["run"] for run in cases["draft-notes"]["run_results"]],
                         ["run-1", "run-2", "run-3", "run-10"])
        draft = cases["draft-notes"]
        self.assertEqual((draft["runs"], draft["scored"], draft["loaded"], draft["not_loaded"], draft["unscored"]),
                         (4, 3, 2, 1, 1))
        self.assertEqual(draft["trigger_rate"], 0.6667)
        self.assertEqual(draft["run_results"][1]["load_sessions"][0]["has_session_start"], False)
        self.assertEqual(cases["fix-typo"]["loaded"], 1)
        self.assertEqual(cases["not-recorded"]["runs"], 0)
        self.assertIsNone(cases["not-recorded"]["trigger_rate"])
        self.assertEqual(report["totals"], {
            "should_trigger": {"cases": 2, "runs": 4, "loaded": 2, "missed": 1, "unscored": 1},
            "should_not_trigger": {"cases": 1, "runs": 2, "loaded": 1, "clean": 1, "unscored": 0},
        })
        self.assertTrue(any("not-recorded: no recorded runs" in w for w in report["warnings"]))
        self.assertTrue(any("stray-case" in w for w in report["warnings"]))

    def test_text_report(self):
        code, out, _ = self.run_eval()

        self.assertEqual(code, 0)
        self.assertIn("Skill trigger eval (experimental)", out)
        self.assertIn("draft-notes", out)
        self.assertIn("67% (2/3)", out)
        self.assertIn("should trigger:     4 runs, 2 loaded, 1 missed, 1 unscored", out)
        self.assertIn("should not trigger: 2 runs, 1 loaded, 1 clean, 0 unscored", out)
        self.assertIn("draft-notes run-10: unscored, agent exited with status 1", out)
        self.assertIn("draft-notes run-2: loaded only in a session without session start", out)
        self.assertNotRegex(out.lower(), r"\bscore\b")

    def test_exit_gates_are_opt_in(self):
        self.assertEqual(self.run_eval("--fail-on-miss")[0], 1)
        self.assertEqual(self.run_eval("--fail-on-false-trigger")[0], 1)
        self.assertEqual(self.run_eval("--case", "fix-typo", "--fail-on-miss")[0], 0)
        code, _, err = self.run_eval("--case", "fix-typo", "--fail-on-false-trigger")
        self.assertEqual(code, 1)
        self.assertIn("fix-typo: should not trigger; 1 loaded", err)

    def test_clean_results_pass_both_gates(self):
        code, _, err = self.run_eval("--case", "draft-notes", "--case", "fix-typo",
                                     "--fail-on-miss", "--fail-on-false-trigger")
        self.assertEqual(code, 1)
        clean = self.write_cases([
            {"id": "fix-typo", "skill": "changelog-lint", "prompt": "Fix the typo", "should_trigger": False},
        ], "clean.json")
        code, out, err = _run_cli(["skills", "eval", str(clean), "--from-traces", str(self.recorded),
                                   "--fail-on-miss", "--fail-on-false-trigger"])
        self.assertEqual(code, 0, err + out)

    def test_input_errors_exit_2(self):
        self.assertEqual(self.run_eval("--case", "nope")[0], 2)
        self.assertEqual(_run_cli(["skills", "eval", str(self.cases), "--from-traces",
                                   str(self.root / "absent")])[0], 2)
        code, out, _ = self.run_eval("--runs", "5", "--format", "json")
        self.assertEqual(code, 0)
        self.assertIn("--runs is ignored with --from-traces", json.loads(out)["warnings"])


class UserOnlySkillTests(_TempDirTest):
    def setUp(self):
        super().setUp()
        self.repo = self.root / "repo"
        _write_skill(self.repo)
        _write_skill(self.repo, "interview-me", disable_model_invocation=True)
        self.recorded = self.root / "recorded"
        cwd = str(self.repo)
        for run in ("run-1", "run-2"):
            _write_session(self.recorded / "interview" / run, f"i-{run}", _plain_session(cwd))
            _write_session(self.recorded / "quiet" / run, f"q-{run}", _plain_session(cwd))
        _write_session(self.recorded / "draft-notes" / "run-1", "d1", _skill_tool_session(cwd))
        self.cases = self.write_cases([
            {"id": "interview", "skill": "interview-me", "prompt": "Interview me about this plan", "should_trigger": True},
            {"id": "quiet", "skill": "interview-me", "prompt": "Fix the typo", "should_trigger": False},
            {"id": "draft-notes", "skill": SKILL, "prompt": "Draft release notes", "should_trigger": True},
        ])
        self._home = mock.patch.dict(os.environ, {"HOME": str(self.root / "home")})
        self._home.start()

    def tearDown(self):
        self._home.stop()
        super().tearDown()

    def run_eval(self, *extra):
        return _run_cli(["skills", "eval", str(self.cases), "--from-traces", str(self.recorded),
                         "--workspace", str(self.repo), *extra])

    def test_case_invocation_lookup(self):
        fixture_case = EvalCase(id="a", skill="interview-me", prompt="x", should_trigger=True,
                                fixture_dir=str(self.repo))
        plugin_case = EvalCase(id="b", skill="plugin:interview-me", prompt="x", should_trigger=True)
        missing_case = EvalCase(id="c", skill="nope", prompt="x", should_trigger=True)

        self.assertEqual(case_invocation(fixture_case, user_dirs=()),
                         {"model_invocable": False, "user_invocable": True})
        self.assertEqual(case_invocation(plugin_case, workspace=str(self.repo), user_dirs=()),
                         {"model_invocable": False, "user_invocable": True})
        self.assertEqual(case_invocation(missing_case, workspace=str(self.repo), user_dirs=()),
                         {"model_invocable": None, "user_invocable": None})
        self.assertEqual(case_invocation(plugin_case, extra_dirs=[str(self.repo / ".github" / "skills")],
                                         user_dirs=())["model_invocable"], False)

    def test_json_marks_user_only_cases_without_changing_counts(self):
        code, out, err = self.run_eval("--format", "json")
        report = json.loads(out)
        cases = {case["id"]: case for case in report["cases"]}

        self.assertEqual(code, 0, err)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual((cases["interview"]["model_invocable"], cases["interview"]["user_invocable"]), (False, True))
        self.assertEqual((cases["interview"]["loaded"], cases["interview"]["not_loaded"]), (0, 2))
        self.assertEqual(cases["interview"]["trigger_rate"], 0.0)
        self.assertIs(cases["draft-notes"]["model_invocable"], True)
        self.assertEqual(report["totals"], {
            "should_trigger": {"cases": 2, "runs": 3, "loaded": 1, "missed": 2, "unscored": 0},
            "should_not_trigger": {"cases": 1, "runs": 2, "loaded": 0, "clean": 2, "unscored": 0},
        })
        self.assertNotIn("score", report)

    def test_text_notes_and_gate_message(self):
        code, out, _ = self.run_eval()

        self.assertEqual(code, 0)
        self.assertIn("user-only skill", out)
        self.assertIn("interview: interview-me has disable-model-invocation: true; the model cannot invoke it, "
                      "so a not-loaded run is not a trigger miss", out)
        self.assertNotIn("draft-notes: " + SKILL + " has disable-model-invocation", out)
        self.assertNotRegex(out.lower(), r"\bscore\b")
        code, _, err = self.run_eval("--fail-on-miss")
        self.assertEqual(code, 1)
        self.assertIn("interview: should trigger; 2 missed, 0 unscored of 2 runs "
                      "(user-only skill: disable-model-invocation: true)", err)

    def test_unknown_definition_is_null(self):
        code, out, _ = _run_cli(["skills", "eval", str(self.cases), "--from-traces", str(self.recorded),
                                 "--format", "json"])
        cases = {case["id"]: case for case in json.loads(out)["cases"]}

        self.assertEqual(code, 0)
        self.assertIsNone(cases["interview"]["model_invocable"])
        self.assertIsNone(cases["interview"]["user_invocable"])


class CliRoutingTests(unittest.TestCase):
    def parse(self, argv):
        return build_parser().parse_args(
            _normalise_eval_argv(_normalise_skills_argv(_normalise_policy_argv(argv)))
        )

    def test_skills_eval_routes_to_eval_parser(self):
        args = self.parse(["--trace-dir", "x", "skills", "eval", "cases.json", "--agent", "copilot"])
        self.assertEqual(args.command, "skills-eval")
        self.assertEqual((args.cases_file, args.agent, args.runs), ("cases.json", "copilot", None))

    def test_skills_session_and_eval_commands_unchanged(self):
        self.assertEqual(self.parse(["skills", "abc123"]).session_id, "abc123")
        self.assertIsNone(self.parse(["skills"]).session_id)
        self.assertEqual(self.parse(["eval", "abc123", "--baseline", "b"]).command, "eval")

    def test_source_is_required_and_exclusive(self):
        for argv in (["skills", "eval", "c.json"],
                     ["skills", "eval", "c.json", "--agent", "claude", "--from-traces", "d"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(argv)


FAKE_AGENT = textwrap.dedent(r'''
    import json, os, subprocess, sys, time, uuid
    from pathlib import Path

    if len(sys.argv) != 4:
        sys.exit(8)
    prompt, plugin_dir, workspace = sys.argv[1:]
    cwd = os.getcwd()
    if os.path.realpath(workspace) != os.path.realpath(cwd):
        sys.exit(8)
    if os.environ.get("COPILOT_ALLOW_ALL") or os.environ.get("CLAUDECODE"):
        sys.exit(9)
    if "SLEEP" in prompt:
        time.sleep(60)
    if "FAIL" in prompt:
        sys.stderr.write("starting\nnot logged in\n")
        sys.exit(3)
    if "SILENT" in prompt:
        sys.exit(0)

    hooks = json.loads((Path(plugin_dir) / "hooks.json").read_text())["hooks"]

    def fire(name, **payload):
        payload.update(sessionId=sid, cwd=cwd)
        for hook in hooks.get(name, []):
            subprocess.run(hook["command"], shell=True, input=json.dumps(payload), text=True, check=True)

    sid = uuid.uuid4().hex
    if "SUBAGENT" not in prompt:
        fire("sessionStart", source="new")
    fire("userPromptSubmitted", prompt=prompt)
    if "READ" in prompt:
        path = os.path.join(cwd, ".github", "skills", "release-notes", "SKILL.md")
        if not os.path.isfile(path):
            sys.exit(7)
        fire("preToolUse", toolName="view", toolArgs={"path": path})
    elif "LOAD" in prompt:
        fire("preToolUse", toolName="skill", toolArgs={"skill": "release-notes"})
        fire("postToolUse", toolName="skill", toolArgs={"skill": "release-notes"},
             toolResult={"resultType": "success", "textResultForLlm": "ok"})
    else:
        fire("preToolUse", toolName="view", toolArgs={"path": "CHANGELOG.md"})
    fire("agentStop", stopReason="end_turn")
''')


@unittest.skipUnless(os.name == "posix", "live runner is exercised on POSIX")
class LiveRunTests(_TempDirTest):
    def setUp(self):
        super().setUp()
        self.agent = self.root / "fake_agent.py"
        self.agent.write_text(FAKE_AGENT)
        fixture = self.root / "fixture"
        _write_skill(fixture)
        self.template = (f"{shlex.quote(sys.executable)} {shlex.quote(str(self.agent))} "
                         "{prompt} {hooks_plugin_dir} {workspace}")

    def test_live_runs_capture_score_and_round_trip(self):
        tricky = "LOAD the notes; $(touch pwned) `touch pwned2` 'q'"
        cases = self.write_cases([
            {"id": "loads", "skill": SKILL, "prompt": tricky, "should_trigger": True, "runs": 2},
            {"id": "subagent", "skill": SKILL, "prompt": "SUBAGENT LOAD", "should_trigger": True},
            {"id": "reads", "skill": SKILL, "prompt": "READ", "should_trigger": True, "fixture_dir": "fixture"},
            {"id": "clean", "skill": SKILL, "prompt": "fix the typo", "should_trigger": False},
            {"id": "fails", "skill": SKILL, "prompt": "FAIL", "should_trigger": True},
            {"id": "silent", "skill": SKILL, "prompt": "SILENT", "should_trigger": False},
        ])
        saved = self.root / "saved"
        leaked = {"COPILOT_ALLOW_ALL": "1", "CLAUDECODE": "1", "AGENT_STRACE_ENDPOINT": "http://127.0.0.1:9"}
        with mock.patch.dict(os.environ, leaked):
            code, out, err = _run_cli(["skills", "eval", str(cases), "--agent-cmd", self.template, "--runs", "1",
                                       "--save-traces", str(saved), "--format", "json"])
        live = json.loads(out)
        by_id = {case["id"]: case for case in live["cases"]}

        self.assertEqual(code, 0, err)
        self.assertEqual(live["mode"], "live")
        self.assertEqual(live["agent"]["name"], "custom")
        self.assertIn("[1/7] loads run 1 ... loaded", err)
        self.assertEqual((by_id["loads"]["runs"], by_id["loads"]["loaded"]), (2, 2))
        self.assertEqual(by_id["reads"]["run_results"][0]["methods"], ["file_read"])
        subagent_run = by_id["subagent"]["run_results"][0]
        self.assertEqual(subagent_run["status"], "loaded")
        self.assertEqual(subagent_run["sessions_without_session_start"], 1)
        # Hooks now create metadata for sessions that never fire a session start.
        self.assertEqual(subagent_run["sessions_without_meta"], 0)
        self.assertEqual(by_id["clean"]["run_results"][0]["status"], "not_loaded")
        failed = by_id["fails"]["run_results"][0]
        self.assertEqual((failed["status"], failed["exit_code"]), ("unscored", 3))
        self.assertIn("agent exited with status 3: not logged in", failed["unscored_reason"])
        self.assertIn("no trace events", by_id["silent"]["run_results"][0]["unscored_reason"])
        self.assertEqual(live["totals"], {
            "should_trigger": {"cases": 4, "runs": 5, "loaded": 4, "missed": 0, "unscored": 1},
            "should_not_trigger": {"cases": 2, "runs": 2, "loaded": 0, "clean": 1, "unscored": 1},
        })

        saved_events = [
            TraceEvent.from_json(line)
            for path in sorted((saved / "loads").glob("run-*/*/events.ndjson"))
            for line in path.read_text().splitlines()
        ]
        prompts = [event.data["prompt"] for event in saved_events if event.event_type == EventType.USER_PROMPT]
        self.assertEqual(prompts, [tricky, tricky])
        self.assertEqual(json.loads((saved / "fails" / "run-1" / "run.json").read_text())["exit_code"], 3)

        code, out, err = _run_cli(["skills", "eval", str(cases), "--from-traces", str(saved), "--format", "json"])
        replay = json.loads(out)
        self.assertEqual(code, 0, err)
        self.assertEqual(replay["totals"], live["totals"])

    def test_timeout_kills_the_agent(self):
        cases = self.write_cases([{"id": "slow", "skill": SKILL, "prompt": "SLEEP", "should_trigger": True}])
        started = time.monotonic()
        code, out, err = _run_cli(["skills", "eval", str(cases), "--agent-cmd", self.template, "--runs", "1",
                                   "--timeout", "1", "--format", "json", "--fail-on-miss"])
        run = json.loads(out)["cases"][0]["run_results"][0]

        self.assertLess(time.monotonic() - started, 15)
        self.assertEqual(code, 1)
        self.assertTrue(run["timed_out"])
        self.assertEqual(run["status"], "unscored")
        self.assertIn("timed out after 1s", run["unscored_reason"])

    def test_live_input_errors_and_warnings(self):
        cases = self.write_cases([{"id": "a", "skill": SKILL, "prompt": "SILENT", "should_trigger": False}])
        missing = self.write_cases([{"id": "a", "skill": SKILL, "prompt": "x", "should_trigger": True,
                                     "fixture_dir": "nope"}], "missing.json")
        (self.root / "full").mkdir()
        (self.root / "full" / "x").write_text("x")

        self.assertEqual(_run_cli(["skills", "eval", str(cases), "--agent-cmd", "no-such-agent-xyz {prompt}"])[0], 2)
        self.assertEqual(_run_cli(["skills", "eval", str(cases), "--agent-cmd", "agent -p"])[0], 2)
        self.assertEqual(_run_cli(["skills", "eval", str(missing), "--agent-cmd", self.template])[0], 2)
        self.assertEqual(_run_cli(["skills", "eval", str(cases), "--agent-cmd", self.template,
                                   "--save-traces", str(self.root / "full")])[0], 2)
        code, _, err = _run_cli(["skills", "eval", str(cases), "--agent-cmd",
                                 f"{shlex.quote(sys.executable)} -c pass {{prompt}} --yolo", "--runs", "1"])
        self.assertEqual(code, 0)
        self.assertIn("approves every tool call (--yolo)", err)


if __name__ == "__main__":
    unittest.main()
