import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cli import build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.review import is_test_command
from agent_trace.skills import (
    aggregate_reports,
    analyze_session,
    cmd_skills,
    command_signatures,
    discover_skills,
    event_arguments,
    extract_skill_commands,
    format_aggregate_text,
    format_session_text,
)
from agent_trace.store import TraceStore

BODY_SENTINEL = "SKILLBODYSENTINEL7731"
PROMPT_SENTINEL = "PROMPTSENTINEL4402"
T0 = 1_700_000_000.0


def _write_skill(base: Path, name: str, body: str, *, frontmatter_name: str | None = None,
                 resources: dict[str, str] | None = None) -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    header = f"---\nname: {frontmatter_name}\ndescription: test\n---\n" if frontmatter_name else ""
    (skill_dir / "SKILL.md").write_text(header + body)
    for rel, text in (resources or {}).items():
        path = skill_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return skill_dir


DEPLOY_BODY = f"""# Deploy check

{BODY_SENTINEL} prose that must never be printed.

```bash
$ make deploy-check --target {BODY_SENTINEL}
./scripts/check.sh
```

See reference.md for details.
"""


class _Events:
    """Builds a time-ordered synthetic event stream."""

    def __init__(self, cwd: str | None = None):
        self.events: list[TraceEvent] = []
        self.t = T0
        if cwd is not None:
            self.add(EventType.SESSION_START, cwd=cwd)

    def add(self, event_type, parent_id="", **data) -> TraceEvent:
        self.t += 1
        event = TraceEvent(event_type=event_type, timestamp=self.t, parent_id=parent_id, data=data)
        self.events.append(event)
        return event

    def tool(self, name, arguments=None, *, key="arguments", result=None, error=False, **extra):
        event = self.add(EventType.TOOL_CALL, tool_name=name, **{key: arguments if arguments is not None else {}}, **extra)
        if error:
            self.add(EventType.ERROR, parent_id=event.event_id, error="failed")
        elif result is not None:
            self.add(EventType.TOOL_RESULT, parent_id=event.event_id, tool_name=name, **result)
        return event

    def bash(self, command, **kwargs):
        return self.tool("bash", {"command": command}, **kwargs)

    def prompt(self, text):
        return self.add(EventType.USER_PROMPT, prompt=text)


class SkillsTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.skills_root = self.repo / ".github" / "skills"
        self.deploy_dir = _write_skill(
            self.skills_root,
            "deploy-check",
            DEPLOY_BODY,
            resources={"reference.md": "ref", "scripts/check.sh": "#!/bin/sh\n"},
        )

    def analyze(self, events, **kwargs):
        kwargs.setdefault("user_dirs", ())
        return analyze_session(events, session_id="s1", **kwargs)

    def loaded(self, report, name):
        for entry in report["loaded"]:
            if entry["name"] == name:
                return entry
        self.fail(f"{name} not loaded: {[e['name'] for e in report['loaded']]}")


class TestDiscovery(SkillsTestBase):
    def test_frontmatter_name_fallback_and_source(self):
        _write_skill(self.repo / ".claude" / "skills", "dir-name", "plain body\n")
        _write_skill(self.repo / ".agents" / "skills", "folder", "body\n", frontmatter_name="pretty-name")

        catalog = discover_skills(self.repo, user_dirs=())

        self.assertIn("deploy-check", catalog.skills)
        self.assertIn("dir-name", catalog.skills)
        self.assertIn("pretty-name", catalog.skills)
        self.assertNotIn("folder", catalog.skills)
        self.assertEqual(catalog.skills["deploy-check"].source, "repo")
        self.assertEqual(catalog.skills["deploy-check"].source_dir, ".github/skills")
        self.assertEqual(catalog.skills["pretty-name"].source_dir, ".agents/skills")

    def test_root_skill_md_and_depth_limit(self):
        extra = self.tmp / "extra"
        extra.mkdir()
        (extra / "SKILL.md").write_text("---\nname: rooty\n---\nbody\n")
        _write_skill(self.skills_root / "group", "too-deep", "body\n")

        catalog = discover_skills(self.repo, [str(extra)], user_dirs=())

        self.assertIn("rooty", catalog.skills)
        self.assertEqual(catalog.skills["rooty"].source, "extra")
        self.assertNotIn("too-deep", catalog.skills)

    def test_missing_directories_are_reported_not_fatal(self):
        empty = self.tmp / "empty-repo"
        empty.mkdir()
        catalog = discover_skills(empty, [str(self.tmp / "nope")], user_dirs=(str(self.tmp / "no-user"),))

        self.assertEqual(catalog.skills, {})
        self.assertTrue(catalog.scanned)
        self.assertFalse(any(item.exists for item in catalog.scanned))

    def test_duplicate_name_keeps_repo_precedence(self):
        user = self.tmp / "user-skills"
        _write_skill(user, "deploy-check", "user copy\n")

        catalog = discover_skills(self.repo, user_dirs=(str(user),))

        self.assertEqual(catalog.skills["deploy-check"].source, "repo")
        self.assertEqual(len(catalog.also_in["deploy-check"]), 1)

    def test_definition_commands_and_resources(self):
        catalog = discover_skills(self.repo, user_dirs=())
        definition = catalog.skills["deploy-check"]

        self.assertIn("make deploy-check", definition.commands)
        self.assertIn("reference.md", definition.resources)
        self.assertIn(os.path.join("scripts", "check.sh"), definition.resources)

    def test_extract_skill_commands_fences(self):
        text = (
            "```bash\n$ FOO=1 sudo -E npm run build\ngh api repos/x/y \\\n  --jq .z\n```\n"
            "```python\nimport os\n```\n"
            "~~~sh\ngit --no-pager log -1\n~~~\n"
            "```text\n> pytest -q tests/\n```\n"
            "```\nunlabelled fences are ignored\n```\n"
        )
        signatures = extract_skill_commands(text)

        self.assertEqual(set(signatures), {"npm run", "gh api", "git log", "pytest -q"})

    def test_command_signatures_split_segments(self):
        signatures = command_signatures("cd /tmp && FOO=bar git status -s | grep x; echo done")
        self.assertIn("git status", signatures)
        self.assertIn("grep x", signatures)
        self.assertNotIn("echo done", signatures)
        self.assertNotIn("cd /tmp", signatures)

    def test_command_signatures_multiline_and_comments(self):
        for command in ("make a\n\nmake b", "make a;\nmake b", "make a &&\nmake b", "make a |\nmake b",
                        "(make a);make b", "make a # note\nmake b", "make a 2>&1 | make b"):
            with self.subTest(command=command):
                self.assertEqual(command_signatures(command), ["make a", "make b"])
        self.assertEqual(command_signatures("git log --format=%H#%s -1 && make b"), ["git log", "make b"])
        self.assertEqual(command_signatures("make 'a # b'\nmake c"), ["make a # b", "make c"])
        self.assertEqual(
            extract_skill_commands("```bash\nnpm ci  # install deps\n\nnpm test\n```\n"),
            ("npm ci", "npm test"),
        )

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores permissions")
    def test_unreadable_skill_dir_is_not_fatal(self):
        locked = _write_skill(self.skills_root, "locked", "```bash\nmake locked\n```\n")
        os.chmod(locked, 0)
        self.addCleanup(os.chmod, locked, 0o755)

        catalog = discover_skills(self.repo, user_dirs=())

        self.assertIn("deploy-check", catalog.skills)
        self.assertNotIn("locked", catalog.skills)


class TestLoads(SkillsTestBase):
    def test_copilot_skill_tool_with_string_arguments(self):
        ev = _Events(str(self.repo))
        call = ev.tool("skill", "{'skill': 'deploy-check'}", result={"result": "Skill loaded"})

        report = self.analyze(ev.events)
        entry = self.loaded(report, "deploy-check")
        load = entry["loads"][0]

        self.assertEqual(event_arguments(call), {"skill": "deploy-check"})
        self.assertEqual(load["method"], "skill_tool")
        self.assertEqual(load["invoked_by"], "model")
        self.assertTrue(load["load_succeeded"])
        self.assertEqual(load["event_id"], call.event_id)
        self.assertTrue(entry["definition_found"])
        self.assertEqual(report["workspace_source"], "session_start")

    def test_claude_skill_tool_newer_and_older_shapes(self):
        for key in ("skill", "command"):
            with self.subTest(key=key):
                ev = _Events(str(self.repo))
                ev.tool("Skill", {key: "plugin:deploy-check"}, key="tool_input")
                load = self.loaded(self.analyze(ev.events), "deploy-check")["loads"][0]
                self.assertEqual(load["method"], "skill_tool")
                self.assertIsNone(load["load_succeeded"])

    def test_claude_slash_prompt_is_user_load(self):
        ev = _Events(str(self.repo))
        ev.prompt("/deploy-check please")
        ev.prompt("/deploy-checker is not a skill")
        ev.prompt("/unknown-skill too")

        report = self.analyze(ev.events)
        entry = self.loaded(report, "deploy-check")

        self.assertEqual(len(report["loaded"]), 1)
        self.assertEqual(len(entry["loads"]), 1)
        self.assertEqual(entry["loads"][0]["method"], "user_prompt")
        self.assertEqual(entry["loads"][0]["invoked_by"], "user")
        self.assertEqual(entry["loads"][0]["detail"], "slash_command")
        self.assertIsNone(entry["loads"][0]["load_succeeded"])

    def test_codex_dollar_mention(self):
        ev = _Events(str(self.repo))
        ev.prompt("use $deploy-check for this, cost is $5")

        load = self.loaded(self.analyze(ev.events), "deploy-check")["loads"][0]
        self.assertEqual(load["detail"], "dollar_mention")
        self.assertEqual(load["invoked_by"], "user")

    def test_gemini_activate_skill(self):
        ev = _Events(str(self.repo))
        ev.tool("activate_skill", {"name": "deploy-check"}, result={"success": True})

        load = self.loaded(self.analyze(ev.events), "deploy-check")["loads"][0]
        self.assertEqual(load["method"], "skill_tool")
        self.assertEqual(load["detail"], "activate_skill")
        self.assertTrue(load["load_succeeded"])

    def test_file_read_loads(self):
        skill_md = str(self.deploy_dir / "SKILL.md")
        cases = [
            ("read_tool", lambda ev: ev.tool("Read", {"file_path": skill_md})),
            ("read_tool", lambda ev: ev.tool("view", {"path": ".github/skills/deploy-check/SKILL.md"})),
            ("read_tool", lambda ev: ev.add(EventType.FILE_READ, path=skill_md)),
            ("shell_read", lambda ev: ev.bash("cd .github/skills && cat deploy-check/SKILL.md")),
        ]
        for detail, emit in cases:
            with self.subTest(detail=detail):
                ev = _Events(str(self.repo))
                emit(ev)
                load = self.loaded(self.analyze(ev.events), "deploy-check")["loads"][0]
                self.assertEqual(load["method"], "file_read")
                self.assertEqual(load["detail"], detail)

    def test_editing_skill_md_is_not_a_load(self):
        ev = _Events(str(self.repo))
        ev.tool("edit", {"path": str(self.deploy_dir / "SKILL.md")})
        ev.bash("sed -i 's/a/b/' .github/skills/deploy-check/SKILL.md")

        self.assertEqual(self.analyze(ev.events)["loaded"], [])

    def test_multiline_shell_reads_are_loads(self):
        for command in ("git status\n\ncat .github/skills/deploy-check/SKILL.md",
                        "git status &&\ncat .github/skills/deploy-check/SKILL.md",
                        "ls .github/skills  # list\ncat .github/skills/deploy-check/SKILL.md"):
            with self.subTest(command=command):
                ev = _Events(str(self.repo))
                ev.bash(command)
                self.loaded(self.analyze(ev.events), "deploy-check")

    def test_similar_filenames_are_not_loads(self):
        ev = _Events(str(self.repo))
        ev.tool("view", {"path": "docs/WRITING_A_SKILL.md"})
        ev.bash("cat guides/NEW_SKILL.md")

        self.assertEqual(self.analyze(ev.events)["loaded"], [])

    def test_unknown_skill_is_still_a_load(self):
        ev = _Events(str(self.repo))
        ev.tool("skill", {"skill": "app-bundled"}, result={"result": "ok"})

        entry = self.loaded(self.analyze(ev.events), "app-bundled")
        self.assertFalse(entry["definition_found"])
        self.assertEqual(entry["follow"]["status"], "not_measurable")
        self.assertEqual(entry["follow"]["reason"], "definition_not_found")

    def test_failed_load_from_linked_error(self):
        ev = _Events(str(self.repo))
        ev.tool("skill", {"skill": "deploy-check"}, error=True)

        load = self.loaded(self.analyze(ev.events), "deploy-check")["loads"][0]
        self.assertFalse(load["load_succeeded"])

    def test_loaded_before_first_edit_and_timing(self):
        ev = _Events(str(self.repo))
        ev.tool("skill", {"skill": "deploy-check"})
        ev.tool("edit", {"path": "src/app.py"})
        ev.prompt("/deploy-check again")

        loads = self.loaded(self.analyze(ev.events), "deploy-check")["loads"]
        self.assertTrue(loads[0]["loaded_before_first_edit"])
        self.assertFalse(loads[1]["loaded_before_first_edit"])
        self.assertAlmostEqual(loads[0]["seconds_since_start"], 1.0)

        ev = _Events(str(self.repo))
        ev.tool("skill", {"skill": "deploy-check"})
        load = self.loaded(self.analyze(ev.events), "deploy-check")["loads"][0]
        self.assertIsNone(load["loaded_before_first_edit"])

    def test_workspace_fallback_without_session_start(self):
        ev = _Events()
        ev.tool("skill", {"skill": "deploy-check"})

        report = self.analyze(ev.events, workspace=str(self.repo))
        self.assertEqual(report["workspace_source"], "--workspace")
        self.assertTrue(self.loaded(report, "deploy-check")["definition_found"])


class TestFollowSignals(SkillsTestBase):
    def _follow(self, *emitters, before=()):
        ev = _Events(str(self.repo))
        for emit in before:
            emit(ev)
        ev.tool("skill", {"skill": "deploy-check"})
        for emit in emitters:
            emit(ev)
        return self.loaded(self.analyze(ev.events), "deploy-check")["follow"]

    def test_resource_read(self):
        follow = self._follow(lambda ev: ev.tool("view", {"path": str(self.deploy_dir / "reference.md")}))

        self.assertEqual(follow["status"], "signals_observed")
        self.assertEqual(follow["signals"]["resource_read"], {"count": 1, "paths": ["reference.md"]})

    def test_script_run(self):
        follow = self._follow(
            lambda ev: ev.bash("bash .github/skills/deploy-check/scripts/check.sh --fast"),
            lambda ev: ev.bash(f"cd {self.deploy_dir} && ./scripts/check.sh"),
        )

        self.assertEqual(follow["signals"]["script_run"]["count"], 2)
        self.assertEqual(follow["signals"]["script_run"]["paths"], [os.path.join("scripts", "check.sh")])

    def test_command_match_reports_only_signatures(self):
        follow = self._follow(lambda ev: ev.bash("CI=1 make deploy-check --target prod"))

        match = follow["signals"]["command_match"]
        self.assertEqual(match["count"], 1)
        self.assertEqual(match["signatures"], ["make deploy-check"])
        self.assertEqual(match["by_signature"], {"make deploy-check": 1})
        self.assertTrue(follow["first_signal_before_next_prompt"])
        self.assertEqual(follow["seconds_to_first_signal"], 1.0)

    def test_signal_after_later_prompt_is_labelled(self):
        follow = self._follow(lambda ev: ev.prompt("next task"), lambda ev: ev.bash("make deploy-check"))
        self.assertFalse(follow["first_signal_before_next_prompt"])

    def test_no_signal_and_pre_load_activity_ignored(self):
        follow = self._follow(
            lambda ev: ev.bash("ls -la"),
            before=(lambda ev: ev.bash("make deploy-check"),),
        )

        self.assertEqual(follow["status"], "no_signal")
        self.assertIsNone(follow["seconds_to_first_signal"])
        self.assertNotIn("violated", json.dumps(follow))
        self.assertNotIn("ignored", json.dumps(follow))

    def test_not_measurable_without_commands_or_resources(self):
        _write_skill(self.skills_root, "prose-only", "Be careful and kind.\n```python\nprint(1)\n```\n")
        ev = _Events(str(self.repo))
        ev.tool("skill", {"skill": "prose-only"})
        ev.bash("make deploy-check")

        follow = self.loaded(self.analyze(ev.events), "prose-only")["follow"]
        self.assertEqual(follow["status"], "not_measurable")
        self.assertEqual(follow["reason"], "definition_has_no_commands_or_resources")

    def test_cd_then_bare_resource_names(self):
        follow = self._follow(
            lambda ev: ev.bash("cd .github/skills/deploy-check && cat reference.md"),
            lambda ev: ev.bash("cd .github/skills/deploy-check/scripts && bash check.sh"),
        )

        self.assertEqual(follow["signals"]["resource_read"]["paths"], ["reference.md"])
        self.assertEqual(follow["signals"]["script_run"]["paths"], [os.path.join("scripts", "check.sh")])

    def test_bare_tokens_must_name_known_resources(self):
        follow = self._follow(
            lambda ev: ev.bash("cd .github/skills/deploy-check && grep -rn reference ."),
            lambda ev: ev.bash("cat reference.md"),
        )

        self.assertEqual(follow["status"], "no_signal")


class TestRobustness(SkillsTestBase):
    def test_malformed_event_fields_do_not_crash(self):
        events = [
            TraceEvent(event_type=EventType.SESSION_START, timestamp="1700000000", data={"cwd": str(self.repo)}),
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=T0 + 1, data="not a dict"),
            TraceEvent(event_type=EventType.USER_PROMPT, timestamp=T0 + 2, data=None),
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=T0 + 3, data=[]),
            TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=T0 + 4, parent_id=["x"], event_id={"a": 1}, data={}),
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp="not a time", data={}),
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=T0 + 5,
                       data={"tool_name": "skill", "arguments": {"skill": "deploy-check"}}),
        ]

        report = self.analyze(events)

        self.assertEqual(report["events_skipped_malformed"], 1)
        self.assertEqual(report["workspace_source"], "session_start")
        self.loaded(report, "deploy-check")

    def test_deeply_nested_and_nul_arguments_do_not_crash(self):
        ev = _Events(str(self.repo))
        ev.tool("bash", '{"command":' + "[" * 50000)
        ev.tool("view", {"path": "x\x00/SKILL.md"})
        ev.bash("cat ~x\x00y/SKILL.md; cd /tmp\x00; cat a/b")
        ev.tool("skill", {"skill": "deploy-check"})
        ev.tool("view", {"path": "y\x00/reference.md"})

        report = self.analyze(ev.events)

        self.assertEqual([entry["name"] for entry in report["loaded"]], ["deploy-check"])


class TestOutcome(SkillsTestBase):
    def test_tests_after_load_use_linked_results(self):
        ev = _Events(str(self.repo))
        ev.bash("pytest -q", result={"exit_code": 0})
        ev.tool("skill", {"skill": "deploy-check"})
        ev.bash("npm test", result={"exit_code": 0})
        last = ev.bash("python -m pytest tests/", result={"exit_code": 1, "success": False})
        ev.add(EventType.ERROR, error="unrelated")
        ev.add(EventType.SESSION_END)

        report = self.analyze(ev.events)
        outcome = self.loaded(report, "deploy-check")["outcome_after_load"]

        self.assertEqual(outcome["tests_run"], 2)
        self.assertEqual(outcome["last_test"]["event_id"], last.event_id)
        self.assertEqual(outcome["last_test"]["linked_result"], "tool_result")
        self.assertEqual(outcome["last_test"]["exit_code"], 1)
        self.assertIs(outcome["last_test"]["success"], False)
        self.assertEqual(outcome["errors"], 1)
        self.assertTrue(outcome["session_end_recorded"])
        self.assertTrue(report["session_facts"]["tests_run"])

    def test_test_linked_to_error_and_no_session_end(self):
        ev = _Events(str(self.repo))
        ev.tool("skill", {"skill": "deploy-check"})
        ev.bash("go test ./...", error=True)

        outcome = self.loaded(self.analyze(ev.events), "deploy-check")["outcome_after_load"]
        self.assertEqual(outcome["last_test"]["linked_result"], "error")
        self.assertIsNone(outcome["last_test"]["exit_code"])
        self.assertEqual(outcome["errors"], 1)
        self.assertFalse(outcome["session_end_recorded"])

    def test_is_test_command_is_public(self):
        self.assertTrue(is_test_command("python -m pytest -q"))
        self.assertTrue(is_test_command("npm test"))
        self.assertFalse(is_test_command("git status"))


class TestAggregate(SkillsTestBase):
    def setUp(self):
        super().setUp()
        _write_skill(self.skills_root, "never-used", "```sh\nmake never\n```\n")

    def _reports(self, loaded_sessions, total):
        reports = []
        for index in range(total):
            ev = _Events(str(self.repo))
            if index < loaded_sessions:
                ev.tool("skill", {"skill": "deploy-check"})
                ev.bash("make deploy-check")
                ev.bash("pytest", result={"exit_code": 0})
            ev.add(EventType.SESSION_END)
            reports.append(analyze_session(ev.events, session_id=f"s{index}", user_dirs=()))
        return reports

    def _skill(self, report, name):
        return next(skill for skill in report["skills"] if skill["name"] == name)

    def test_never_loaded_and_counts(self):
        report = aggregate_reports(self._reports(2, 5), min_sessions=5)
        deploy = self._skill(report, "deploy-check")

        self.assertEqual(report["never_loaded"], [{"name": "never-used", "sessions_available": 5}])
        self.assertEqual(deploy["sessions_available"], 5)
        self.assertEqual(deploy["sessions_loaded"], 2)
        self.assertEqual(deploy["load_rate"], {"count": 2, "n": 5})
        self.assertEqual(deploy["invocations"], {"user": 0, "model": 2})
        self.assertEqual(deploy["follow_signals"], {"count": 2, "n": 2})
        self.assertEqual(deploy["follow_signal_kinds"]["command_match"], {"count": 2, "n": 2})
        self.assertEqual(deploy["tests_after_load"], {"count": 2, "n": 2})
        self.assertEqual(deploy["median_seconds_to_load"], 1.0)
        self.assertNotIn("score", json.dumps(report))
        self.assertNotIn("grade", json.dumps(report))

    def test_never_loaded_respects_min_sessions(self):
        report = aggregate_reports(self._reports(0, 3), min_sessions=5)
        self.assertEqual(report["never_loaded"], [])

    def test_comparison_suppressed_below_min_n(self):
        report = aggregate_reports(self._reports(2, 8), min_sessions=3)
        comparison = self._skill(report, "deploy-check")["comparison"]

        self.assertTrue(comparison["suppressed"])
        self.assertNotIn("tests_run", comparison)
        out = io.StringIO()
        format_aggregate_text(report, out)
        self.assertIn("suppressed", out.getvalue())

    def test_comparison_present_at_min_n_and_labelled(self):
        report = aggregate_reports(self._reports(3, 6), min_sessions=3)
        comparison = self._skill(report, "deploy-check")["comparison"]

        self.assertFalse(comparison["suppressed"])
        self.assertIn("correlational", comparison["label"])
        self.assertEqual(comparison["tests_run"], {
            "loaded": {"count": 3, "n": 3},
            "available_not_loaded": {"count": 0, "n": 3},
        })


class TestCli(SkillsTestBase):
    def setUp(self):
        super().setUp()
        home = self.tmp / "home"
        home.mkdir()
        patcher = mock.patch.dict(os.environ, {"HOME": str(home), "AGENT_STRACE_TELEMETRY": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.trace_dir = self.tmp / ".agent-traces"
        self.store = TraceStore(self.trace_dir)

    def _session(self, sid, events, started_at):
        meta = SessionMeta(agent_name="claude-code")
        meta.session_id = sid
        meta.started_at = started_at
        self.store.create_session(meta)
        for event in events:
            self.store.append_event(sid, event)
        return sid

    def _run(self, *argv):
        args = build_parser().parse_args(["--trace-dir", str(self.trace_dir), "skills", *argv])
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = cmd_skills(args)
        return code, out.getvalue()

    def _seed(self):
        now = time.time()
        ev = _Events(str(self.repo))
        for event in ev.events:
            event.timestamp = now - 60
        ev.t = now - 50
        ev.prompt(f"/deploy-check {PROMPT_SENTINEL}")
        ev.bash(f"make deploy-check --target {BODY_SENTINEL}")
        ev.tool("Read", {"file_path": str(self.deploy_dir / "SKILL.md")}, key="tool_input")
        return self._session("cli00001", ev.events, now - 60)

    def test_session_json_smoke(self):
        sid = self._seed()
        code, output = self._run(sid, "--format", "json")

        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(report["mode"], "session")
        self.assertEqual(report["availability_basis"], "filesystem_now")
        entry = report["loaded"][0]
        self.assertEqual(entry["name"], "deploy-check")
        self.assertEqual([load["method"] for load in entry["loads"]], ["user_prompt", "file_read"])
        self.assertEqual(entry["follow"]["signals"]["command_match"]["signatures"], ["make deploy-check"])
        self.assertTrue(all(not item["load_observable"] for item in report["context_files"]))

    def test_aggregate_json_smoke(self):
        self._seed()
        code, output = self._run("--since", "1d", "--min-sessions", "1", "--format", "json")

        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(report["mode"], "aggregate")
        self.assertEqual(report["sessions"], 1)
        self.assertEqual(report["min_sessions"], 1)

    def test_unknown_session_and_bad_min_sessions(self):
        code, _ = self._run("doesnotexist")
        self.assertEqual(code, 1)
        code, _ = self._run("--min-sessions", "0")
        self.assertEqual(code, 2)

    def test_aggregate_skips_unreadable_session(self):
        self._seed()
        bad = self._session("bad00001", _Events(str(self.repo)).events, time.time() - 30)
        events_file = next(self.trace_dir.rglob(f"*{bad}*/events.ndjson"))
        with events_file.open("a") as handle:
            handle.write('{"event_type": "tool_call", "unexpected_field": 1}\n')

        code, output = self._run("--since", "1d", "--min-sessions", "1", "--format", "json")

        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(report["sessions"], 1)
        self.assertEqual(report["sessions_skipped_unreadable"], 1)

    def test_no_content_leak(self):
        sid = self._seed()
        outputs = [
            self._run(sid)[1],
            self._run(sid, "--format", "json")[1],
            self._run("--since", "1d", "--min-sessions", "1")[1],
            self._run("--since", "1d", "--min-sessions", "1", "--format", "json")[1],
        ]
        for output in outputs:
            self.assertIn("deploy-check", output)
            self.assertNotIn(BODY_SENTINEL, output)
            self.assertNotIn(PROMPT_SENTINEL, output)
            self.assertNotIn("prose that must never be printed", output)
            self.assertNotIn(str(self.deploy_dir), output)

    def test_text_formatter_mentions_limits(self):
        ev = _Events(str(self.repo))
        ev.tool("skill", {"skill": "app-bundled"})
        out = io.StringIO()
        format_session_text(self.analyze(ev.events), out)
        text = out.getvalue()

        self.assertIn("not found in scanned directories", text)
        self.assertIn("Not measurable here", text)
        self.assertNotIn("score", text.lower())


if __name__ == "__main__":
    unittest.main()
