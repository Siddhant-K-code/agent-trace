"""Tests for ``skills --from-copilot`` and ``skills --from-codex``.

All transcripts here are synthetic.  Sentinels stand in for prompt text,
model output, reasoning, and file bodies, none of which may be stored or
printed.
"""

import io
import json
import os
import shutil
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent_trace.cli import build_parser
from agent_trace.skills import cmd_skills, format_session_text, load_codex_history, load_copilot_history
from agent_trace.transcript_import import skill_tokens

PROMPT = "PROMPT_SENTINEL_9f2c"
REASONING = "REASONING_SENTINEL_51ab"
BODY = "BODY_SENTINEL_77de"
SYSTEM = "SYSTEM_SENTINEL_0c4e"
SENTINELS = (PROMPT, REASONING, BODY, SYSTEM)


def _iso(offset: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.123456789Z", time.gmtime(time.time() - 600 + offset))


def _write(path: Path, records: list, tail: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records) + tail)
    return path


class TranscriptTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        home = self.tmp / "home"
        home.mkdir()
        patcher = mock.patch.dict(os.environ, {"HOME": str(home), "AGENT_STRACE_TELEMETRY": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("COPILOT_HOME", "CODEX_HOME"):
            if name in os.environ:
                patcher = mock.patch.dict(os.environ)
                patcher.start()
                self.addCleanup(patcher.stop)
                del os.environ[name]
        self.repo = self.tmp / "repo"
        self.skills_root = self.repo / ".github" / "skills"
        deploy = self.skills_root / "deploy-check"
        deploy.mkdir(parents=True)
        (deploy / "SKILL.md").write_text(
            f"# Deploy check\n\n{BODY}\n\n```bash\n$ make deploy-check --target prod\n```\n"
        )
        self.deploy_file = deploy / "SKILL.md"
        self.trace_dir = self.tmp / ".agent-traces"

    def run_cli(self, *argv):
        args = build_parser().parse_args(["--trace-dir", str(self.trace_dir), "skills", *argv])
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cmd_skills(args)
        return code, out.getvalue(), err.getvalue()

    def assert_private(self, text: str):
        for sentinel in SENTINELS:
            self.assertNotIn(sentinel, text)

    def skill(self, report, name):
        return next(item for item in report["skills"] if item["name"] == name)


class TestCopilot(TranscriptTestBase):
    def _session(self, home: Path, sid: str, records: list, tail: str = "") -> Path:
        return _write(home / "session-state" / sid / "events.jsonl", records, tail)

    def _recorded_session(self, home: Path) -> Path:
        listing = "".join(
            f"<skill><name>{name}</name><description>d</description><location>{where}</location></skill>"
            for name, where in (("deploy-check", "project"), ("release-notes", "personal-copilot"),
                                ("never-used", "plugin"))
        )
        records = [
            {"type": "session.start", "timestamp": _iso(0),
             "data": {"startTime": _iso(0), "context": {"cwd": str(self.repo)}}},
            {"type": "system.message", "timestamp": _iso(1),
             "data": {"content": f"{SYSTEM}\n<available_skills>{listing}</available_skills>"}},
            {"type": "user.message", "timestamp": _iso(2), "data": {"content": f"/release-notes {PROMPT}"}},
            {"type": "assistant.message", "timestamp": _iso(3),
             "data": {"content": REASONING, "reasoningText": REASONING}},
            {"type": "tool.execution_start", "timestamp": _iso(4),
             "data": {"toolCallId": "t1", "toolName": "skill", "arguments": {"skill": "deploy-check"}}},
            {"type": "tool.execution_complete", "timestamp": _iso(5),
             "data": {"toolCallId": "t1", "success": True, "result": {"content": BODY}}},
            {"type": "skill.invoked", "timestamp": _iso(5),
             "data": {"name": "deploy-check", "path": str(self.deploy_file), "source": "project",
                      "trigger": "agent-invoked", "content": BODY}},
            {"type": "tool.execution_start", "timestamp": _iso(6),
             "data": {"toolCallId": "t2", "toolName": "bash",
                      "arguments": {"command": "make deploy-check --target prod", "description": PROMPT}}},
            {"type": "tool.execution_complete", "timestamp": _iso(7),
             "data": {"toolCallId": "t2", "success": True, "shellExecution": {"exitCode": 0}}},
            {"type": "tool.execution_start", "timestamp": _iso(8),
             "data": {"toolCallId": "t3", "toolName": "create",
                      "arguments": {"path": str(self.repo / "x.py"), "file_text": BODY}}},
            {"type": "a.future.event", "timestamp": _iso(9), "data": {"anything": [1, 2]}},
            "not an object",
            {"type": "session.shutdown", "timestamp": _iso(10), "data": {}},
        ]
        return self._session(home, "aaaa1111-0000-4000-8000-000000000001", records,
                             tail='{"type": "user.message", "data": {"content": "trunc')

    def test_reads_recorded_availability_and_loads(self):
        home = self.tmp / "copilot"
        self._recorded_session(home)
        self._session(home, "bbbb2222-0000-4000-8000-000000000002", [
            {"type": "session.start", "timestamp": _iso(20), "data": {"context": {"cwd": str(self.repo)}}},
            {"type": "user.message", "timestamp": _iso(21), "data": {"content": PROMPT}},
        ])
        (home / "session-state" / "no-events-here").mkdir(parents=True)
        (home / "session-state" / "no-events-here" / "workspace.yaml").write_text("x: 1\n")

        code, out, err = self.run_cli("--from-copilot", "--copilot-dir", str(home),
                                      "--min-sessions", "1", "--format", "json")

        self.assertEqual(code, 0, err)
        self.assertIn("Read 2 GitHub Copilot session(s)", err)
        self.assertIn("1 recorded their available skills", err)
        report = json.loads(out)
        self.assertEqual(report["availability_basis"], "mixed")
        self.assertEqual(report["availability_basis_sessions"], {"recorded": 1, "filesystem_now": 1})
        deploy = self.skill(report, "deploy-check")
        self.assertEqual((deploy["sessions_available"], deploy["sessions_loaded"]), (2, 1))
        # The skill tool call and its skill.invoked event are one load.
        self.assertEqual(deploy["invocations"], {"user": 0, "model": 1})
        self.assertEqual(deploy["follow_signals"], {"count": 1, "n": 1})
        notes = self.skill(report, "release-notes")
        self.assertEqual(notes["invocations"], {"user": 1, "model": 0})
        self.assertFalse(notes["definition_found"])
        self.assertEqual([item["name"] for item in report["never_loaded"]], ["never-used"])
        self.assertFalse(self.trace_dir.exists())
        self.assert_private(out + err)

    def test_session_text_names_recorded_basis(self):
        home = self.tmp / "copilot"
        self._recorded_session(home)

        code, out, err = self.run_cli("aaaa1111", "--from-copilot", "--copilot-dir", str(home))

        self.assertEqual(code, 0, err)
        self.assertIn("Availability is the skill list this session recorded.", out)
        self.assertNotIn("filesystem now, not session time", out)
        self.assert_private(out + err)

    def test_temporary_store_keeps_no_prompt_or_output_text(self):
        home = self.tmp / "copilot"
        self._recorded_session(home)

        store, counts = load_copilot_history(str(home), str(self.tmp / "imported"))

        self.assertEqual(counts["recorded"], 1)
        events = store.load_events(store.list_sessions()[0].session_id)
        dumped = json.dumps([event.data for event in events])
        self.assert_private(dumped)
        self.assertIn("/release-notes", dumped)

    def test_unpaired_and_failed_skill_loads(self):
        home = self.tmp / "copilot"
        self._session(home, "cccc3333-0000-4000-8000-000000000003", [
            {"type": "session.start", "timestamp": _iso(0), "data": {"context": {"cwd": str(self.repo)}}},
            {"type": "skill.invoked", "timestamp": _iso(1),
             "data": {"name": "deploy-check", "path": str(self.deploy_file), "trigger": "agent-invoked"}},
            {"type": "tool.execution_start", "timestamp": _iso(2),
             "data": {"toolCallId": "t9", "toolName": "skill", "arguments": {"skill": "missing-skill"}}},
            {"type": "tool.execution_complete", "timestamp": _iso(3), "data": {"toolCallId": "t9", "success": False}},
        ])

        code, out, err = self.run_cli("cccc3333", "--from-copilot", "--copilot-dir", str(home), "--format", "json")

        self.assertEqual(code, 0, err)
        loaded = {entry["name"]: entry for entry in json.loads(out)["loaded"]}
        self.assertEqual(loaded["deploy-check"]["loads"][0]["method"], "skill_tool")
        self.assertTrue(loaded["deploy-check"]["loads"][0]["load_succeeded"])
        self.assertFalse(loaded["missing-skill"]["loads"][0]["load_succeeded"])

    def test_defaults_to_copilot_home(self):
        home = self.tmp / "custom-copilot"
        self._recorded_session(home)

        with mock.patch.dict(os.environ, {"COPILOT_HOME": str(home)}):
            code, out, _ = self.run_cli("--from-copilot", "--format", "json")

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["sessions"], 1)

    def test_missing_dir_fails_with_hint(self):
        code, out, err = self.run_cli("--from-copilot", "--copilot-dir", str(self.tmp / "nope"))

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("No GitHub Copilot sessions found", err)
        self.assertIn("--copilot-dir", err)


class TestCodex(TranscriptTestBase):
    SESSION_ID = "019a0000-0000-7000-8000-000000000001"

    def _rollout(self, home: Path, name: str, records: list, tail: str = "", sub: str = "sessions/2026/01/02"):
        return _write(home / sub / f"rollout-{name}.jsonl", records, tail)

    def _meta(self, sid: str, offset: float = 0, **extra) -> dict:
        payload = {"id": sid, "timestamp": _iso(offset), "cwd": str(self.repo), "source": "vscode",
                   "thread_source": "user", **extra}
        return {"type": "session_meta", "timestamp": _iso(offset), "payload": payload}

    def _item(self, offset: float, **item) -> dict:
        return {"type": "event_msg", "timestamp": _iso(offset), "payload": {"type": "item_completed", "item": item}}

    def _recorded_rollout(self, home: Path) -> Path:
        instructions = (
            "<skills_instructions>\n## Skills\n### Skill roots\n"
            f"- `r0` = `{self.skills_root}`\n"
            "### Available skills\n"
            "- deploy-check: Run deploy checks. (file: r0/deploy-check/SKILL.md)\n"
            "- release-notes: Draft notes. (file: r0/release-notes/SKILL.md)\n"
            "- never-used: Never needed. (file: r0/never-used/SKILL.md)\n"
            "- remote-tool: Remote. (cloud package: remote-tool@1)\n"
            "</skills_instructions>"
        )
        records = [
            self._meta(self.SESSION_ID),
            {"type": "response_item", "timestamp": _iso(1), "payload": {
                "type": "message", "role": "developer",
                "content": [{"type": "input_text", "text": f"{SYSTEM}\n{instructions}"}]}},
            {"type": "turn_context", "timestamp": _iso(2), "payload": {"cwd": str(self.repo)}},
            self._item(3, type="UserMessage", content=[{"type": "text", "text": f"use $release-notes {PROMPT}"}]),
            {"type": "response_item", "timestamp": _iso(4), "payload": {
                "type": "reasoning", "summary": [{"type": "summary_text", "text": REASONING}]}},
            self._item(5, type="CommandExecution", cwd=str(self.repo), exit_code=0, status="completed",
                       command=["/bin/zsh", "-lc", f"sed -n '1,80p' {self.deploy_file}"],
                       aggregated_output=BODY),
            self._item(6, type="CommandExecution", cwd=str(self.repo), exit_code=0, status="completed",
                       command=["/bin/zsh", "-lc", "make deploy-check --target prod"]),
            self._item(7, type="FileChange", status="completed", changes={"x.py": {"content": BODY}}),
            self._item(8, type="AgentMessage", text=REASONING),
            {"type": "turn_context", "timestamp": _iso(9), "payload": {}},
            {"type": "response_item", "timestamp": _iso(10), "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text",
                             "text": f"<skill>\n<name>release-notes</name>\n<path>/x/SKILL.md</path>\n{BODY}\n</skill>"}]}},
            {"type": "future_record", "payload": {"x": 1}},
        ]
        return self._rollout(home, "2026-01-02T00-00-00-" + self.SESSION_ID, records, tail='{"type": "event_m')

    def test_reads_recorded_availability_and_loads(self):
        home = self.tmp / "codex"
        self._recorded_rollout(home)
        self._rollout(home, "not-codex", [{"type": "message", "role": "user", "content": PROMPT}])
        self._rollout(home, "review", [
            self._meta("019a0000-0000-7000-8000-000000000002", thread_source="guardian_review"),
        ])
        self._rollout(home, "archived", [
            self._meta("019a0000-0000-7000-8000-000000000003", 30),
            self._item(31, type="UserMessage", content=[{"type": "text", "text": PROMPT}]),
        ], sub="archived_sessions")

        code, out, err = self.run_cli("--from-codex", "--codex-dir", str(home),
                                      "--min-sessions", "1", "--format", "json")

        self.assertEqual(code, 0, err)
        self.assertIn("Read 2 Codex session(s)", err)
        self.assertIn("skipped 1 file(s) that are not Codex rollouts, 1 approval review thread(s)", err)
        self.assertIn("1 recorded their available skills", err)
        report = json.loads(out)
        self.assertEqual(report["availability_basis_sessions"], {"recorded": 1, "filesystem_now": 1})
        deploy = self.skill(report, "deploy-check")
        self.assertEqual(deploy["invocations"], {"user": 0, "model": 1})
        self.assertTrue(deploy["definition_found"])
        self.assertEqual(deploy["follow_signals"], {"count": 1, "n": 1})
        notes = self.skill(report, "release-notes")
        self.assertEqual((notes["sessions_loaded"], notes["invocations"]), (1, {"user": 2, "model": 0}))
        self.assertEqual(self.skill(report, "remote-tool")["sessions_available"], 1)
        never = [item["name"] for item in report["never_loaded"]]
        self.assertIn("never-used", never)
        self.assertNotIn("deploy-check", never)
        self.assertFalse(self.trace_dir.exists())
        self.assert_private(out + err)

    def test_temporary_store_keeps_no_prompt_or_output_text(self):
        home = self.tmp / "codex"
        self._recorded_rollout(home)

        store, counts = load_codex_history(str(home), str(self.tmp / "imported"))

        self.assertEqual(counts["recorded"], 1)
        events = store.load_events(self.SESSION_ID)
        self.assert_private(json.dumps([event.data for event in events]))

    def test_session_text_reports_recorded_basis(self):
        home = self.tmp / "codex"
        self._recorded_rollout(home)

        code, out, err = self.run_cli(self.SESSION_ID[:12], "--from-codex", "--codex-dir", str(home))

        self.assertEqual(code, 0, err)
        self.assertIn("Availability is the skill list this session recorded.", out)
        self.assert_private(out + err)

    def test_only_skipped_files_fails_with_counts(self):
        home = self.tmp / "codex"
        self._rollout(home, "not-codex", [{"type": "message"}])

        code, out, err = self.run_cli("--from-codex", "--codex-dir", str(home))

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("No Codex sessions found", err)
        self.assertIn("1 file(s) that are not Codex rollouts", err)
        self.assertIn("--codex-dir", err)

    def test_defaults_to_codex_home(self):
        home = self.tmp / "custom-codex"
        self._recorded_rollout(home)

        with mock.patch.dict(os.environ, {"CODEX_HOME": str(home)}):
            code, out, _ = self.run_cli("--from-codex", "--format", "json")

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["sessions"], 1)


class TestCli(TranscriptTestBase):
    def test_from_flags_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["skills", "--from-claude", "--from-codex"])

    def test_empty_store_hint_lists_every_reader(self):
        code, out, _ = self.run_cli()

        self.assertEqual(code, 0)
        for flag in ("--from-claude", "--from-copilot", "--from-codex"):
            self.assertIn(f"agent-strace skills {flag}", out)

    def test_skill_tokens_keep_only_mentions(self):
        self.assertEqual(skill_tokens(f"/deploy-check {PROMPT} $a-b and $c, cost $5"), "/deploy-check $a-b $c")
        self.assertEqual(skill_tokens(f"{PROMPT} /not-leading"), "")


if __name__ == "__main__":
    unittest.main()
