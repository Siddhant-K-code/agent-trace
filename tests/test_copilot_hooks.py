"""Tests for GitHub Copilot hooks integration."""

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.cli import cmd_setup
from agent_trace.hooks import _read_active_session, hook_main
from agent_trace.models import EventType
from agent_trace.store import TraceStore


class TestCopilotHooks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_COPILOT_SESSION_ID", None)
        os.environ.pop("AGENT_TRACE_REDACT", None)

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_COPILOT_SESSION_ID", None)

    def test_copilot_hook_main_normalizes_camel_case_payloads(self):
        start_payload = json.dumps({
            "sessionId": "copilotsession123456",
            "source": "startup",
            "model": "copilot",
        })
        with patch.object(sys, "stdin", io.StringIO(start_payload)):
            hook_main(["--provider", "copilot", "session-start"])

        session_id = _read_active_session(provider="copilot")

        prompt_payload = json.dumps({
            "sessionId": "copilotsession123456",
            "turnId": "turn_1",
            "initialPrompt": "Run the checks",
        })
        with patch.object(sys, "stdin", io.StringIO(prompt_payload)):
            hook_main(["--provider", "copilot", "user-prompt"])

        tool_payload = json.dumps({
            "sessionId": "copilotsession123456",
            "turnId": "turn_1",
            "toolUseId": "tool_1",
            "toolName": "terminal",
            "toolArgs": {"command": "pytest"},
        })
        with patch.object(sys, "stdin", io.StringIO(tool_payload)):
            hook_main(["--provider", "copilot", "pre-tool"])

        result_payload = json.dumps({
            "sessionId": "copilotsession123456",
            "toolUseId": "tool_1",
            "toolName": "terminal",
            "toolResult": {"exit_code": 0, "output": "ok"},
        })
        with patch.object(sys, "stdin", io.StringIO(result_payload)):
            hook_main(["--provider", "copilot", "post-tool"])

        stop_payload = json.dumps({
            "sessionId": "copilotsession123456",
            "stopReason": "end_turn",
            "transcriptPath": "/tmp/copilot-transcript.jsonl",
        })
        with patch.object(sys, "stdin", io.StringIO(stop_payload)):
            hook_main(["--provider", "copilot", "agentStop"])

        store = TraceStore(self.tmpdir)
        meta = store.load_meta(session_id)
        events = store.load_events(session_id)
        prompts = [event for event in events if event.event_type == EventType.USER_PROMPT]
        calls = [event for event in events if event.event_type == EventType.TOOL_CALL]
        results = [event for event in events if event.event_type == EventType.TOOL_RESULT]
        stops = [
            event for event in events
            if event.event_type == EventType.ASSISTANT_RESPONSE and event.data.get("hook_event") == "stop"
        ]

        self.assertEqual(meta.agent_name, "github-copilot")
        self.assertEqual(events[0].data["provider"], "copilot")
        self.assertEqual(prompts[0].data["prompt"], "Run the checks")
        self.assertEqual(calls[0].data["tool_name"], "terminal")
        self.assertEqual(calls[0].data["arguments"]["command"], "pytest")
        self.assertEqual(results[0].parent_id, calls[0].event_id)
        self.assertEqual(stops[0].data["stop_reason"], "end_turn")
        self.assertEqual(stops[0].data["transcript_path"], "/tmp/copilot-transcript.jsonl")

    def test_copilot_hook_main_records_official_tool_result_payloads(self):
        with patch.object(sys, "stdin", io.StringIO(json.dumps({
            "sessionId": "copilotofficial123",
            "source": "startup",
        }))):
            hook_main(["--provider", "copilot", "session-start"])

        with patch.object(sys, "stdin", io.StringIO(json.dumps({
            "sessionId": "copilotofficial123",
            "toolName": "bash",
            "toolArgs": {"command": "echo ok"},
        }))):
            hook_main(["--provider", "copilot", "pre-tool"])

        with patch.object(sys, "stdin", io.StringIO(json.dumps({
            "sessionId": "copilotofficial123",
            "toolName": "bash",
            "toolArgs": {"command": "echo ok"},
            "toolResult": {
                "resultType": "success",
                "textResultForLlm": "ok",
            },
        }))):
            hook_main(["--provider", "copilot", "post-tool"])

        session_id = _read_active_session(provider="copilot")
        events = TraceStore(self.tmpdir).load_events(session_id)
        results = [event for event in events if event.event_type == EventType.TOOL_RESULT]

        self.assertEqual(results[0].data["result"], "ok")

    def test_copilot_hook_main_records_official_failure_payload(self):
        with patch.object(sys, "stdin", io.StringIO(json.dumps({
            "sessionId": "copilotfailure123",
            "source": "startup",
        }))):
            hook_main(["--provider", "copilot", "session-start"])

        with patch.object(sys, "stdin", io.StringIO(json.dumps({
            "sessionId": "copilotfailure123",
            "toolName": "bash",
            "toolArgs": {"command": "false"},
        }))):
            hook_main(["--provider", "copilot", "pre-tool"])

        with patch.object(sys, "stdin", io.StringIO(json.dumps({
            "sessionId": "copilotfailure123",
            "toolName": "bash",
            "toolArgs": {"command": "false"},
            "error": "Command exited with status 1",
        }))):
            hook_main(["--provider", "copilot", "post-tool-failure"])

        session_id = _read_active_session(provider="copilot")
        events = TraceStore(self.tmpdir).load_events(session_id)
        errors = [event for event in events if event.event_type == EventType.ERROR]

        self.assertEqual(errors[0].data["error"], "Command exited with status 1")

    def test_copilot_resume_preserves_existing_session_metadata(self):
        session_payload = {
            "sessionId": "copilotresume1234",
            "source": "startup",
        }
        with patch.object(sys, "stdin", io.StringIO(json.dumps(session_payload))):
            hook_main(["--provider", "copilot", "session-start"])

        store = TraceStore(self.tmpdir)
        session_id = _read_active_session(provider="copilot")
        meta = store.load_meta(session_id)
        started_at = meta.started_at
        meta.ended_at = started_at + 4
        meta.total_duration_ms = 4000
        meta.tool_calls = 3
        meta.errors = 1
        store.update_meta(meta)

        session_payload["source"] = "resume"
        with patch.object(sys, "stdin", io.StringIO(json.dumps(session_payload))):
            hook_main(["--provider", "copilot", "session-start"])

        resumed = store.load_meta(session_id)
        events = store.load_events(session_id)
        starts = [event for event in events if event.event_type == EventType.SESSION_START]

        self.assertEqual(resumed.started_at, started_at)
        self.assertIsNone(resumed.ended_at)
        self.assertEqual(resumed.total_duration_ms, 4000)
        self.assertEqual(resumed.tool_calls, 3)
        self.assertEqual(resumed.errors, 1)
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[-1].data["source"], "resume")


class TestHookPayloadStoreDirectory(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.process_cwd = self.root / "installed-plugin"
        self.payload_cwd = self.root / "repository"
        self.process_cwd.mkdir()
        self.payload_cwd.mkdir()
        self.original_cwd = Path.cwd()
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_CLAUDE_SESSION_ID", None)
        os.environ.pop("AGENT_TRACE_COPILOT_SESSION_ID", None)
        os.chdir(self.process_cwd)

    def tearDown(self):
        os.chdir(self.original_cwd)
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_CLAUDE_SESSION_ID", None)
        os.environ.pop("AGENT_TRACE_COPILOT_SESSION_ID", None)
        self.tempdir.cleanup()

    def _dispatch_start(self, provider, payload):
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))):
            hook_main(["--provider", provider, "session-start"])

    def test_hook_main_uses_payload_cwd_for_camel_and_snake_payloads(self):
        copilot_session = "copilot-cwd-session"
        claude_session = "claude-cwd-session"

        self._dispatch_start("copilot", {
            "sessionId": copilot_session,
            "cwd": str(self.payload_cwd),
            "source": "startup",
        })
        self._dispatch_start("claude", {
            "session_id": claude_session,
            "cwd": str(self.payload_cwd),
            "source": "startup",
        })

        store = TraceStore(self.payload_cwd / ".agent-traces")
        self.assertIsNotNone(store.load_meta(copilot_session[:16]))
        self.assertIsNotNone(store.load_meta(claude_session[:16]))
        self.assertFalse((self.process_cwd / ".agent-traces").exists())

    def test_explicit_trace_dir_takes_precedence_over_payload_cwd(self):
        explicit_dir = self.root / "explicit-traces"
        os.environ["AGENT_TRACE_DIR"] = str(explicit_dir)
        session = "explicit-dir-session"

        self._dispatch_start("copilot", {
            "sessionId": session,
            "cwd": str(self.payload_cwd),
            "source": "startup",
        })

        self.assertIsNotNone(TraceStore(explicit_dir).load_meta(session[:16]))
        self.assertFalse((self.payload_cwd / ".agent-traces").exists())
        self.assertFalse((self.process_cwd / ".agent-traces").exists())

    def test_invalid_payload_cwd_falls_back_without_using_invalid_path(self):
        non_directory = self.root / "not-a-directory"
        non_directory.write_text("not a directory")
        relative_directory = self.process_cwd / "relative-directory"
        relative_directory.mkdir()
        invalid_cwds = (
            ("non-string", [str(self.payload_cwd)]),
            ("non-directory", str(non_directory)),
            ("relative", relative_directory.name),
        )

        for index, (case, cwd) in enumerate(invalid_cwds):
            session = f"invalid-cwd-{index}-session"
            with self.subTest(case=case):
                self._dispatch_start("copilot", {
                    "sessionId": session,
                    "cwd": cwd,
                    "source": "startup",
                })
                self.assertIsNotNone(
                    TraceStore(self.process_cwd / ".agent-traces").load_meta(
                        session[:16]
                    )
                )

        self.assertFalse((non_directory / ".agent-traces").exists())
        self.assertFalse((relative_directory / ".agent-traces").exists())


class TestCopilotSetup(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["COPILOT_HOME"] = self.tmpdir

    def tearDown(self):
        os.environ.pop("COPILOT_HOME", None)

    def test_setup_cli_copilot_writes_user_hooks_file(self):
        args = argparse.Namespace(
            redact=False,
            no_redact=False,
            global_config=False,
            cli="copilot",
        )

        out = io.StringIO()
        err = io.StringIO()
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            cmd_setup(args)

        hooks_path = Path(self.tmpdir) / "hooks" / "agent-strace.json"
        hooks = json.loads(hooks_path.read_text())
        printed_hooks = json.loads(out.getvalue())

        self.assertEqual(printed_hooks, hooks)
        self.assertEqual(hooks["version"], 1)
        self.assertIn("sessionStart", hooks["hooks"])
        self.assertIn("postToolUseFailure", hooks["hooks"])
        self.assertIn("agentStop", hooks["hooks"])
        self.assertIn("sessionEnd", hooks["hooks"])
        self.assertEqual(
            hooks["hooks"]["userPromptSubmitted"][0]["command"],
            "agent-strace hook --provider copilot user-prompt",
        )
        self.assertEqual(
            hooks["hooks"]["agentStop"][0]["command"],
            "agent-strace hook --provider copilot stop",
        )
        self.assertIn("GitHub Copilot hooks config", err.getvalue())


if __name__ == "__main__":
    unittest.main()
