"""Tests for OpenAI Codex hooks integration."""

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
    handle_post_tool,
    handle_pre_tool,
    handle_session_start,
    handle_stop,
    handle_user_prompt,
    hook_main,
)
from agent_trace.models import EventType
from agent_trace.store import TraceStore


class TestCodexHooks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_CODEX_SESSION_ID", None)
        os.environ.pop("AGENT_TRACE_REDACT", None)

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_CODEX_SESSION_ID", None)

    def test_session_start_creates_codex_session(self):
        handle_session_start({
            "session_id": "codexsession123456789",
            "source": "startup",
            "model": "gpt-5-codex",
            "cwd": "/work/repo",
            "permission_mode": "default",
        }, provider="codex")

        session_id = _read_active_session(provider="codex")
        store = TraceStore(self.tmpdir)
        meta = store.load_meta(session_id)
        events = store.load_events(session_id)

        self.assertEqual(meta.agent_name, "openai-codex")
        self.assertIn("openai-codex", meta.command)
        self.assertEqual(events[0].event_type, EventType.SESSION_START)
        self.assertEqual(events[0].data["mode"], "openai-codex-hooks")
        self.assertEqual(events[0].data["provider"], "codex")
        self.assertEqual(events[0].data["cwd"], "/work/repo")

    def test_codex_tool_call_and_result_are_linked_by_tool_use_id(self):
        handle_session_start({"session_id": "codextool1234567", "source": "startup"}, provider="codex")
        session_id = _read_active_session(provider="codex")

        handle_pre_tool({
            "session_id": "codextool1234567",
            "tool_name": "Bash",
            "tool_use_id": "call_1",
            "tool_input": {"command": "pytest tests/"},
            "turn_id": "turn_1",
        }, provider="codex")
        handle_post_tool({
            "session_id": "codextool1234567",
            "tool_name": "Bash",
            "tool_use_id": "call_1",
            "tool_response": {"exit_code": 0, "output": "ok"},
        }, provider="codex")

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        results = [e for e in events if e.event_type == EventType.TOOL_RESULT]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].data["tool_use_id"], "call_1")
        self.assertEqual(calls[0].data["turn_id"], "turn_1")
        self.assertEqual(results[0].parent_id, calls[0].event_id)
        self.assertIn("exit_code", results[0].data["result"])

    def test_codex_user_prompt_and_stop_payloads(self):
        handle_session_start({"session_id": "codexprompt12345", "source": "startup"}, provider="codex")
        session_id = _read_active_session(provider="codex")

        handle_user_prompt({
            "session_id": "codexprompt12345",
            "prompt": "Fix the flaky test",
            "turn_id": "turn_2",
        }, provider="codex")
        handle_stop({
            "session_id": "codexprompt12345",
            "last_assistant_message": "Fixed the flaky test.",
            "turn_id": "turn_2",
        }, provider="codex")

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)

        prompts = [e for e in events if e.event_type == EventType.USER_PROMPT]
        responses = [e for e in events if e.event_type == EventType.ASSISTANT_RESPONSE]
        self.assertEqual(prompts[0].data["prompt"], "Fix the flaky test")
        self.assertEqual(prompts[0].data["turn_id"], "turn_2")
        self.assertIn("Fixed", responses[0].data["text"])

    def test_codex_nonzero_tool_response_records_error(self):
        handle_session_start({"session_id": "codexfail1234567", "source": "startup"}, provider="codex")
        session_id = _read_active_session(provider="codex")

        handle_pre_tool({
            "session_id": "codexfail1234567",
            "tool_name": "Bash",
            "tool_use_id": "call_fail",
            "tool_input": {"command": "false"},
        }, provider="codex")
        handle_post_tool({
            "session_id": "codexfail1234567",
            "tool_name": "Bash",
            "tool_use_id": "call_fail",
            "tool_response": {"exit_code": 1, "output": "failed"},
        }, provider="codex")

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        meta = store.load_meta(session_id)

        self.assertEqual(len(errors), 1)
        self.assertIn("failed", errors[0].data["error"])
        self.assertEqual(meta.errors, 1)

    def test_hook_main_accepts_codex_provider(self):
        payload = json.dumps({
            "session_id": "codexmain1234567",
            "source": "startup",
            "model": "gpt-5-codex",
        })
        with patch.object(sys, "stdin", io.StringIO(payload)):
            hook_main(["--provider", "codex", "session-start"])

        store = TraceStore(self.tmpdir)
        meta = store.load_meta("codexmain1234567")
        self.assertEqual(meta.agent_name, "openai-codex")

    def test_hook_main_requires_event_after_provider(self):
        err = io.StringIO()
        with patch.object(sys, "stderr", err):
            with self.assertRaises(SystemExit) as raised:
                hook_main(["--provider", "codex"])

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Usage:", err.getvalue())


class TestCodexSetup(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["CODEX_CONFIG_DIR"] = self.tmpdir
        self.gemini_dir = tempfile.mkdtemp()
        os.environ["GEMINI_CONFIG_DIR"] = self.gemini_dir
        self.copilot_home = tempfile.mkdtemp()
        os.environ["COPILOT_HOME"] = self.copilot_home

    def tearDown(self):
        os.environ.pop("CODEX_CONFIG_DIR", None)
        os.environ.pop("GEMINI_CONFIG_DIR", None)
        os.environ.pop("COPILOT_HOME", None)

    def test_setup_cli_codex_outputs_hooks_json(self):
        args = argparse.Namespace(
            redact=False,
            no_redact=False,
            global_config=False,
            cli="codex",
        )

        out = io.StringIO()
        err = io.StringIO()
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            cmd_setup(args)

        config = json.loads(out.getvalue())
        written_config = json.loads((Path(self.tmpdir) / "hooks.json").read_text())
        err_text = err.getvalue()
        self.assertEqual(written_config, config)
        self.assertIn("Wrote OpenAI Codex hooks config", err_text)
        self.assertIn("~/.codex/hooks.json", err_text)
        self.assertIn("Codex hook checklist", err_text)
        self.assertIn("~/.codex/hooks/hooks.json", err_text)
        self.assertIn("[features].hooks = false", err_text)
        self.assertIn("SessionStart", config["hooks"])
        self.assertEqual(
            config["hooks"]["PreToolUse"][0]["hooks"][0]["command"],
            "agent-strace hook --provider codex pre-tool",
        )
        self.assertEqual(
            config["hooks"]["PostToolUse"][0]["hooks"][0]["command"],
            "agent-strace hook --provider codex post-tool",
        )

    def test_setup_cli_all_outputs_claude_and_codex_sections(self):
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

        text = out.getvalue()
        self.assertIn("agent-strace hook user-prompt", text)
        self.assertIn("agent-strace hook --provider codex user-prompt", text)
        self.assertTrue((Path(self.tmpdir) / "hooks.json").exists())
        self.assertTrue((Path(self.copilot_home) / "hooks" / "agent-strace.json").exists())
        self.assertIn("~/.claude/settings.json", err.getvalue())
        self.assertIn("~/.codex/hooks.json", err.getvalue())
        self.assertIn("GitHub Copilot hooks config", err.getvalue())
        self.assertIn("Codex hook checklist", err.getvalue())


if __name__ == "__main__":
    unittest.main()
