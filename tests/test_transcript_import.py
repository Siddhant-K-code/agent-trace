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

try:
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

from agent_trace.cli import build_parser
from agent_trace.skills import (
    cmd_skills,
    format_session_text,
    is_skills_eval_workspace,
    load_codex_history,
    load_copilot_history,
)
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

        store, counts, _ = load_copilot_history(str(home), str(self.tmp / "imported"))

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

    def test_resume_resets_skill_load_timing(self):
        home = self.tmp / "copilot"
        self._session(home, "cccc3333-0000-4000-8000-000000000004", [
            {"type": "session.start", "timestamp": _iso(0),
             "data": {"startTime": _iso(0), "context": {"cwd": str(self.repo)}}},
            self._listing(("deploy-check", "project")),
            {"type": "session.resume", "timestamp": _iso(100),
             "data": {"resumeTime": _iso(100), "context": {"cwd": str(self.repo)}}},
            {"type": "skill.invoked", "timestamp": _iso(102),
             "data": {"name": "deploy-check", "path": str(self.deploy_file),
                      "source": "project", "trigger": "agent-invoked"}},
        ])

        code, out, err = self.run_cli(
            "--from-copilot",
            "--copilot-dir",
            str(home),
            "--min-sessions",
            "1",
            "--format",
            "json",
        )

        self.assertEqual(code, 0, err)
        deploy = self.skill(json.loads(out), "deploy-check")
        self.assertAlmostEqual(deploy["median_seconds_to_load"], 2.0, places=3)

        store, _, _ = load_copilot_history(str(home), str(self.tmp / "imported-resume"))
        starts = [
            event
            for event in store.load_events(store.list_sessions()[0].session_id)
            if event.event_type.value == "session_start"
        ]
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[1].data["source"], "resume")

    def test_resume_without_start_keeps_workspace(self):
        home = self.tmp / "copilot"
        self._session(home, "cccc3333-0000-4000-8000-000000000005", [
            {"type": "session.resume", "timestamp": _iso(100),
             "data": {"resumeTime": _iso(100), "context": {"cwd": str(self.repo)}}},
            {"type": "skill.invoked", "timestamp": _iso(102),
             "data": {"name": "deploy-check", "path": str(self.deploy_file),
                      "source": "project", "trigger": "agent-invoked"}},
        ])

        code, out, err = self.run_cli(
            "--from-copilot",
            "--copilot-dir",
            str(home),
            "--min-sessions",
            "1",
            "--format",
            "json",
        )

        self.assertEqual(code, 0, err)
        report = json.loads(out)
        self.assertEqual(report["sessions"], 1)
        self.assertEqual(self.skill(report, "deploy-check")["median_seconds_to_load"], 2.0)

    def _skill_file(self, skill_dir: Path, command: str) -> Path:
        skill_dir.mkdir(parents=True)
        path = skill_dir / "SKILL.md"
        path.write_text(f"# Skill\n\n{BODY}\n\n```bash\n$ {command}\n```\n")
        return path

    def _listing(self, *skills) -> dict:
        listing = "".join(f"<skill><name>{name}</name><description>d</description><location>{where}</location>"
                          "</skill>" for name, where in skills)
        return {"type": "system.message", "timestamp": _iso(1),
                "data": {"content": f"{SYSTEM}\n<available_skills>{listing}</available_skills>"}}

    def _start(self, cwd: Path, offset: float = 0) -> dict:
        return {"type": "session.start", "timestamp": _iso(offset), "data": {"context": {"cwd": str(cwd)}}}

    def _bash(self, call_id: str, offset: float, command: str) -> list:
        return [
            {"type": "tool.execution_start", "timestamp": _iso(offset),
             "data": {"toolCallId": call_id, "toolName": "bash", "arguments": {"command": command}}},
            {"type": "tool.execution_complete", "timestamp": _iso(offset + 1),
             "data": {"toolCallId": call_id, "success": True, "shellExecution": {"exitCode": 0}}},
        ]

    def test_stale_builtin_path_uses_newest_installed_version(self):
        home = self.tmp / "copilot"
        pkg = self.tmp / "cache" / "pkg" / "test-platform"
        self._skill_file(pkg / "1.0.9" / "builtin" / "pr-media", "make media-old")
        self._skill_file(pkg / "1.0.10" / "builtin" / "pr-media", "make media-new")
        stale = pkg / "1.0.1" / "builtin" / "pr-media" / "SKILL.md"
        self._session(home, "eeee5555-0000-4000-8000-000000000005", [
            self._start(self.repo),
            self._listing(("pr-media", "builtin")),
            {"type": "skill.invoked", "timestamp": _iso(2),
             "data": {"name": "pr-media", "path": str(stale), "source": "builtin", "trigger": "agent-invoked"}},
            *self._bash("t1", 3, "make media-new"),
        ])

        code, out, err = self.run_cli("--from-copilot", "--copilot-dir", str(home),
                                      "--min-sessions", "1", "--format", "json")

        self.assertEqual(code, 0, err)
        media = self.skill(json.loads(out), "pr-media")
        self.assertTrue(media["definition_found"])
        # 1.0.10 sorts after 1.0.9, and only its command was run.
        self.assertEqual(media["follow_signals"], {"count": 1, "n": 1})

    def test_skill_listed_by_name_finds_definition_in_root_of_other_loads(self):
        home = self.tmp / "copilot"
        app = self.tmp / "app" / "app-skills"
        canvas = self._skill_file(app / "canvas", "make canvas-open")
        self._skill_file(app / "latex", "make latex-render")
        self._session(home, "eeee5555-0000-4000-8000-000000000001", [
            self._start(self.repo),
            self._listing(("canvas", "custom"), ("latex", "custom")),
            {"type": "skill.invoked", "timestamp": _iso(2),
             "data": {"name": "canvas", "path": str(canvas), "source": "custom", "trigger": "agent-invoked"}},
        ])
        self._session(home, "eeee5555-0000-4000-8000-000000000002", [
            self._start(self.repo, 10),
            self._listing(("latex", "custom")),
            {"type": "tool.execution_start", "timestamp": _iso(12),
             "data": {"toolCallId": "t1", "toolName": "skill", "arguments": {"skill": "latex"}}},
            {"type": "tool.execution_complete", "timestamp": _iso(13), "data": {"toolCallId": "t1", "success": True}},
            *self._bash("t2", 14, "make latex-render"),
        ])
        self._session(home, "eeee5555-0000-4000-8000-000000000003", [self._start(self.repo, 20)])

        code, out, err = self.run_cli("--from-copilot", "--copilot-dir", str(home),
                                      "--min-sessions", "1", "--format", "json")

        self.assertEqual(code, 0, err)
        report = json.loads(out)
        latex = self.skill(report, "latex")
        self.assertTrue(latex["definition_found"])
        self.assertEqual(latex["follow_signals"], {"count": 1, "n": 1})
        # The session that recorded no skill list sees the same directory.
        self.assertEqual(latex["sessions_available"], 3)
        self.assertEqual(self.skill(report, "canvas")["sessions_available"], 2)

    def test_sourced_user_messages_are_not_prompts(self):
        home = self.tmp / "copilot"
        self._session(home, "eeee5555-0000-4000-8000-000000000004", [
            self._start(self.repo),
            self._listing(("deploy-check", "project"), ("release-notes", "personal-copilot")),
            {"type": "user.message", "timestamp": _iso(2), "data": {
                "source": "skill-release-notes",
                "content": f'<skill-context name="release-notes">\nrun $deploy-check {BODY}\n</skill-context>'}},
            {"type": "user.message", "timestamp": _iso(3), "data": {"source": "system", "content": "/deploy-check"}},
            {"type": "user.message", "timestamp": _iso(4),
             "data": {"source": "autopilot", "content": "$deploy-check", "isAutopilotContinuation": True}},
            {"type": "user.message", "timestamp": _iso(5), "data": {
                "source": "agent-0f0e0d0c-0000-4000-8000-000000000000", "content": f"/deploy-check {PROMPT}"}},
            {"type": "user.message", "timestamp": _iso(6), "data": {"content": f"/release-notes {PROMPT}"}},
        ])

        code, out, err = self.run_cli("--from-copilot", "--copilot-dir", str(home),
                                      "--min-sessions", "1", "--format", "json")

        self.assertEqual(code, 0, err)
        report = json.loads(out)
        self.assertEqual(self.skill(report, "release-notes")["invocations"], {"user": 1, "model": 0})
        self.assertIn("deploy-check", [item["name"] for item in report["never_loaded"]])
        self.assert_private(out + err)

    def test_skills_eval_sessions_are_left_out(self):
        home = self.tmp / "copilot"
        self._recorded_session(home)
        workspace = self.tmp / "agent-strace-skills-eval-notes-case-k3x9_q2a" / "workspace"
        self._session(home, "eeee5555-0000-4000-8000-000000000006", [
            self._start(workspace, 30),
            self._listing(("release-notes", "project")),
            {"type": "skill.invoked", "timestamp": _iso(32),
             "data": {"name": "release-notes", "path": str(workspace / ".github/skills/release-notes/SKILL.md"),
                      "source": "project", "trigger": "agent-invoked"}},
        ])

        code, out, err = self.run_cli("--from-copilot", "--copilot-dir", str(home),
                                      "--min-sessions", "1", "--format", "json")
        _, text, _ = self.run_cli("--from-copilot", "--copilot-dir", str(home), "--min-sessions", "1")

        self.assertEqual(code, 0, err)
        self.assertIn("Excluded 1 session(s) that agent-strace skills eval ran in its temporary workspaces.", err)
        report = json.loads(out)
        self.assertEqual((report["sessions"], report["sessions_total"]), (1, 2))
        self.assertEqual(report["sessions_excluded"]["skills_eval"], 1)
        notes = self.skill(report, "release-notes")
        self.assertEqual((notes["sessions_available"], notes["invocations"]), (1, {"user": 1, "model": 0}))
        self.assertIn("Skills across 1 of 2 sessions since ", text)
        self.assertIn(" (1 from agent-strace skills eval). ", text)

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

    def test_skills_eval_sessions_are_left_out(self):
        home = self.tmp / "codex"
        self._recorded_rollout(home)
        workspace = self.tmp / "agent-strace-skills-eval-notes-k3x9_q2a" / "workspace"
        self._rollout(home, "eval", [
            self._meta("019a0000-0000-7000-8000-000000000004", 30, cwd=str(workspace)),
            self._item(31, type="UserMessage", content=[{"type": "text", "text": "use $release-notes"}]),
        ])

        code, out, err = self.run_cli("--from-codex", "--codex-dir", str(home),
                                      "--min-sessions", "1", "--format", "json")

        self.assertEqual(code, 0, err)
        self.assertIn("Excluded 1 session(s) that agent-strace skills eval ran", err)
        report = json.loads(out)
        self.assertEqual(report["sessions_excluded"]["skills_eval"], 1)
        self.assertEqual(self.skill(report, "release-notes")["sessions_loaded"], 1)

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

    FORK_ID = "019a0000-0000-7000-8000-0000000000f0"

    def _parent_and_fork(self, home: Path, **fork_meta) -> None:
        """Write a parent that loads deploy-check three ways and a fork that copies it."""
        instructions = (
            "<skills_instructions>\n## Skills\n### Skill roots\n"
            f"- `r0` = `{self.skills_root}`\n"
            "### Available skills\n"
            "- deploy-check: Run deploy checks. (file: r0/deploy-check/SKILL.md)\n"
            "</skills_instructions>"
        )
        parent = [
            self._meta(self.SESSION_ID),
            {"type": "response_item", "timestamp": _iso(1), "payload": {
                "type": "message", "role": "developer", "content": [{"type": "input_text", "text": instructions}]}},
            {"type": "turn_context", "timestamp": _iso(2), "payload": {}},
            {"type": "response_item", "timestamp": _iso(30), "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": f"<skill>\n<name>deploy-check</name>\n{BODY}\n</skill>"}]}},
            {"type": "response_item", "timestamp": _iso(31), "payload": {
                "type": "function_call", "namespace": "skills", "name": "read",
                "arguments": json.dumps({"package": "deploy-check"})}},
            self._item(32, type="CommandExecution", exit_code=0, status="completed",
                       command=["/bin/zsh", "-lc", f"cat {self.deploy_file}"]),
        ]
        self._rollout(home, "parent", parent)
        copied = [dict(record, timestamp=_iso(100)) for record in parent]
        own = [
            {"type": "turn_context", "timestamp": _iso(101), "payload": {}},
            self._item(102, type="UserMessage", content=[{"type": "text", "text": PROMPT}]),
        ]
        meta = {"forked_from_id": self.SESSION_ID, "subagent_history_start_ordinal": 1 + len(copied),
                "thread_source": "subagent", **fork_meta}
        meta = {key: value for key, value in meta.items() if value is not None}
        self._rollout(home, "fork", [self._meta(self.FORK_ID, 100, **meta), *copied, *own])

    def test_subagent_fork_skips_copied_parent_loads(self):
        home = self.tmp / "codex"
        self._parent_and_fork(home)

        code, out, err = self.run_cli("--from-codex", "--codex-dir", str(home),
                                      "--min-sessions", "1", "--format", "json")

        self.assertEqual(code, 0, err)
        report = json.loads(out)
        self.assertEqual(report["availability_basis_sessions"], {"recorded": 2})
        deploy = self.skill(report, "deploy-check")
        self.assertEqual(deploy["sessions_available"], 2)
        self.assertEqual(deploy["sessions_loaded"], 1)
        self.assertEqual(deploy["invocations"], {"user": 1, "model": 2})
        self.assertEqual(deploy["median_seconds_to_load"], 30.0)
        store, _ = load_codex_history(str(home), str(self.tmp / "imported"))
        fork_events = store.load_events(self.FORK_ID)
        self.assertEqual([event.event_type.value for event in fork_events], ["session_start", "user_prompt"])
        self.assertEqual([item["name"] for item in fork_events[0].data["available_skills"]], ["deploy-check"])
        self.assert_private(out + err)

    def test_subagent_fork_with_unusable_ordinal_counts_every_record(self):
        for name, extra in (("missing", {"subagent_history_start_ordinal": None}),
                            ("string", {"subagent_history_start_ordinal": "7"}),
                            ("bool", {"subagent_history_start_ordinal": True}),
                            ("negative", {"subagent_history_start_ordinal": -1}),
                            ("past-end", {"subagent_history_start_ordinal": 99}),
                            ("not-a-fork", {"forked_from_id": None})):
            with self.subTest(name):
                home = self.tmp / f"codex-{name}"
                self._parent_and_fork(home, **extra)

                store, _ = load_codex_history(str(home), str(self.tmp / f"imported-{name}"))

                kinds = [event.event_type.value for event in store.load_events(self.FORK_ID)]
                self.assertIn("tool_call", kinds)
                self.assertEqual(kinds.count("user_prompt"), 2)


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

    def test_skills_eval_workspace_detection(self):
        for path in ("/t/agent-strace-skills-eval-notes-k3x9_q2a/workspace",
                     "/t/agent-strace-skills-eval-a.b-abcdefgh/workspace/sub/dir",
                     "C:\\T\\agent-strace-skills-eval-notes-k3x9_q2a\\workspace"):
            self.assertTrue(is_skills_eval_workspace(path), path)
        for path in ("/t/agent-strace-skills-eval-notes-k3x9_q2a/traces",
                     "/t/agent-strace-skills-eval-notes-k3x9_q2a",
                     "/t/agent-strace-skills-eval-notes/workspace",
                     "/t/probe/workspace", ""):
            self.assertFalse(is_skills_eval_workspace(path), path)


if __name__ == "__main__":
    unittest.main()
