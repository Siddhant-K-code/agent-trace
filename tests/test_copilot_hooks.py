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
        self.assertIn("SessionStart", hooks["hooks"])
        self.assertIn("PostToolUseFailure", hooks["hooks"])
        self.assertIn("Stop", hooks["hooks"])
        self.assertIn("SessionEnd", hooks["hooks"])
        self.assertEqual(
            hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"],
            "agent-strace hook --provider copilot user-prompt",
        )
        self.assertEqual(
            hooks["hooks"]["Stop"][0]["hooks"][0]["command"],
            "agent-strace hook --provider copilot stop",
        )
        self.assertIn("GitHub Copilot hooks config", err.getvalue())


if __name__ == "__main__":
    unittest.main()
