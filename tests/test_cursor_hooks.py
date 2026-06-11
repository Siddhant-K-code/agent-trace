"""Tests for Cursor hooks integration."""

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
from agent_trace.hooks import (
    _read_active_session,
    handle_file_write,
    handle_post_tool,
    handle_pre_tool,
    handle_session_start,
    handle_stop,
    handle_user_prompt,
    hook_main,
)
from agent_trace.models import EventType
from agent_trace.store import TraceStore


class TestCursorHooks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_CURSOR_SESSION_ID", None)
        os.environ.pop("AGENT_TRACE_REDACT", None)

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_CURSOR_SESSION_ID", None)

    def test_cursor_session_start_creates_session(self):
        handle_session_start({
            "session_id": "cursorsession123456789",
            "source": "startup",
            "model": "cursor-agent",
            "cwd": "/work/repo",
        }, provider="cursor")

        session_id = _read_active_session(provider="cursor")
        store = TraceStore(self.tmpdir)
        meta = store.load_meta(session_id)
        events = store.load_events(session_id)

        self.assertEqual(meta.agent_name, "cursor-agent")
        self.assertIn("cursor-agent", meta.command)
        self.assertEqual(events[0].event_type, EventType.SESSION_START)
        self.assertEqual(events[0].data["provider"], "cursor")
        self.assertEqual(events[0].data["mode"], "cursor-agent-hooks")

    def test_cursor_shell_execution_is_linked(self):
        handle_session_start({"session_id": "cursorshell123456", "source": "startup"}, provider="cursor")
        session_id = _read_active_session(provider="cursor")

        handle_pre_tool({
            "session_id": "cursorshell123456",
            "tool_name": "shell",
            "tool_input": {"command": "pytest tests/"},
        }, provider="cursor")
        handle_post_tool({
            "session_id": "cursorshell123456",
            "tool_name": "shell",
            "tool_response": {"exit_code": 0, "output": "ok"},
        }, provider="cursor")

        events = TraceStore(self.tmpdir).load_events(session_id)
        calls = [event for event in events if event.event_type == EventType.TOOL_CALL]
        results = [event for event in events if event.event_type == EventType.TOOL_RESULT]
        self.assertEqual(calls[0].data["arguments"]["command"], "pytest tests/")
        self.assertEqual(results[0].parent_id, calls[0].event_id)

    def test_cursor_prompt_response_and_file_edit(self):
        handle_session_start({"session_id": "cursorprompt12345", "source": "startup"}, provider="cursor")
        session_id = _read_active_session(provider="cursor")

        handle_user_prompt({
            "session_id": "cursorprompt12345",
            "prompt": "Refactor parser",
        }, provider="cursor")
        handle_file_write({
            "session_id": "cursorprompt12345",
            "file_path": "src/parser.py",
            "diff": "+new line",
        }, provider="cursor")
        handle_stop({
            "session_id": "cursorprompt12345",
            "last_assistant_message": "Refactor complete.",
        }, provider="cursor")

        events = TraceStore(self.tmpdir).load_events(session_id)
        prompts = [event for event in events if event.event_type == EventType.USER_PROMPT]
        writes = [event for event in events if event.event_type == EventType.FILE_WRITE]
        responses = [event for event in events if event.event_type == EventType.ASSISTANT_RESPONSE]
        self.assertEqual(prompts[0].data["prompt"], "Refactor parser")
        self.assertEqual(writes[0].data["path"], "src/parser.py")
        self.assertIn("Refactor", responses[0].data["text"])

    def test_hook_main_accepts_cursor_aliases(self):
        handle_session_start({"session_id": "cursoralias12345", "source": "startup"}, provider="cursor")
        session_id = _read_active_session(provider="cursor")

        shell_payload = json.dumps({
            "session_id": "cursoralias12345",
            "command": "python -m pytest",
        })
        with patch.object(sys, "stdin", io.StringIO(shell_payload)):
            hook_main(["--provider", "cursor", "before-shell-execution"])

        result_payload = json.dumps({
            "session_id": "cursoralias12345",
            "command": "python -m pytest",
            "tool_response": {"exit_code": 1, "output": "failed"},
        })
        with patch.object(sys, "stdin", io.StringIO(result_payload)):
            hook_main(["--provider", "cursor", "after-shell-execution"])

        payload = json.dumps({
            "session_id": "cursoralias12345",
            "file_path": "src/app.py",
            "diff": "+x",
        })
        with patch.object(sys, "stdin", io.StringIO(payload)):
            hook_main(["--provider", "cursor", "after-file-edit"])

        events = TraceStore(self.tmpdir).load_events(session_id)
        calls = [event for event in events if event.event_type == EventType.TOOL_CALL]
        errors = [event for event in events if event.event_type == EventType.ERROR]
        writes = [event for event in events if event.event_type == EventType.FILE_WRITE]
        self.assertEqual(calls[0].data["tool_name"], "shell")
        self.assertEqual(calls[0].data["arguments"]["command"], "python -m pytest")
        self.assertEqual(errors[0].parent_id, calls[0].event_id)
        self.assertEqual(writes[0].data["path"], "src/app.py")


class TestCursorSetup(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["CURSOR_CONFIG_DIR"] = self.tmpdir
        self.codex_dir = tempfile.mkdtemp()
        os.environ["CODEX_CONFIG_DIR"] = self.codex_dir
        self.gemini_dir = tempfile.mkdtemp()
        os.environ["GEMINI_CONFIG_DIR"] = self.gemini_dir
        self.copilot_home = tempfile.mkdtemp()
        os.environ["COPILOT_HOME"] = self.copilot_home

    def tearDown(self):
        os.environ.pop("CURSOR_CONFIG_DIR", None)
        os.environ.pop("CODEX_CONFIG_DIR", None)
        os.environ.pop("GEMINI_CONFIG_DIR", None)
        os.environ.pop("COPILOT_HOME", None)

    def test_setup_cli_cursor_writes_hooks_json(self):
        args = argparse.Namespace(
            redact=False,
            no_redact=False,
            global_config=False,
            cli="cursor",
        )

        out = io.StringIO()
        err = io.StringIO()
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            cmd_setup(args)

        hooks_path = Path(self.tmpdir) / "hooks.json"
        hooks = json.loads(hooks_path.read_text())
        printed_hooks = json.loads(out.getvalue())

        self.assertEqual(hooks["version"], 1)
        self.assertIn("beforeSubmitPrompt", hooks["hooks"])
        self.assertIn("beforeShellExecution", hooks["hooks"])
        self.assertIn("afterFileEdit", hooks["hooks"])
        self.assertIn("stop", hooks["hooks"])
        self.assertEqual(
            hooks["hooks"]["afterFileEdit"][0]["command"],
            "agent-strace hook --provider cursor after-file-edit",
        )
        self.assertEqual(
            hooks["hooks"]["stop"][0]["command"],
            "agent-strace hook --provider cursor stop",
        )
        self.assertEqual(printed_hooks, hooks)
        self.assertIn("Cursor hooks config", err.getvalue())

    def test_setup_cli_all_includes_cursor(self):
        args = argparse.Namespace(
            redact=False,
            no_redact=False,
            global_config=False,
            cli="all",
        )

        out = io.StringIO()
        err = io.StringIO()
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            cmd_setup(args)

        self.assertTrue((Path(self.tmpdir) / "hooks.json").exists())
        self.assertTrue((Path(self.copilot_home) / "hooks" / "agent-strace.json").exists())
        self.assertIn("agent-strace hook --provider codex user-prompt", out.getvalue())
        self.assertIn("Cursor hooks config", err.getvalue())


if __name__ == "__main__":
    unittest.main()
