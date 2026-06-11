"""Tests for Gemini CLI hooks integration."""

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


class TestGeminiHooks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_GEMINI_SESSION_ID", None)
        os.environ.pop("AGENT_TRACE_REDACT", None)

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_GEMINI_SESSION_ID", None)

    def _hook(self, event, payload):
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))):
            hook_main(["--provider", "gemini", event])

    def test_gemini_session_start_creates_session(self):
        self._hook("session-start", {
            "session_id": "geminisession123456789",
            "hook_event_name": "SessionStart",
            "source": "startup",
            "cwd": "/work/repo",
        })

        session_id = _read_active_session(provider="gemini")
        store = TraceStore(self.tmpdir)
        meta = store.load_meta(session_id)
        events = store.load_events(session_id)

        self.assertEqual(meta.agent_name, "gemini-cli")
        self.assertIn("gemini-cli", meta.command)
        self.assertEqual(events[0].event_type, EventType.SESSION_START)
        self.assertEqual(events[0].data["provider"], "gemini")
        self.assertEqual(events[0].data["mode"], "gemini-cli-hooks")
        self.assertEqual(events[0].data["cwd"], "/work/repo")

    def test_gemini_tool_call_and_result_are_linked(self):
        self._hook("session-start", {"session_id": "geminitool123456", "source": "startup"})
        session_id = _read_active_session(provider="gemini")

        self._hook("pre-tool", {
            "session_id": "geminitool123456",
            "hook_event_name": "BeforeTool",
            "tool_name": "run_shell_command",
            "tool_input": {"command": "pytest tests/"},
        })
        self._hook("post-tool", {
            "session_id": "geminitool123456",
            "hook_event_name": "AfterTool",
            "tool_name": "run_shell_command",
            "tool_input": {"command": "pytest tests/"},
            "tool_response": {"llmContent": "ok", "returnDisplay": "ok"},
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        results = [e for e in events if e.event_type == EventType.TOOL_RESULT]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].data["tool_name"], "run_shell_command")
        self.assertEqual(results[0].parent_id, calls[0].event_id)
        self.assertIn("llmContent", results[0].data["result"])

    def test_gemini_tool_error_increments_meta_errors(self):
        self._hook("session-start", {"session_id": "geminifail123456", "source": "startup"})
        session_id = _read_active_session(provider="gemini")

        self._hook("pre-tool", {
            "session_id": "geminifail123456",
            "tool_name": "run_shell_command",
            "tool_input": {"command": "false"},
        })
        self._hook("post-tool", {
            "session_id": "geminifail123456",
            "tool_name": "run_shell_command",
            "tool_response": {"error": {"message": "command failed"}},
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        meta = store.load_meta(session_id)

        self.assertEqual(len(errors), 1)
        self.assertIn("command failed", errors[0].data["error"])
        self.assertEqual(meta.errors, 1)

    def test_gemini_prompt_response_and_session_end(self):
        self._hook("session-start", {"session_id": "geminiprompt123", "source": "startup"})
        session_id = _read_active_session(provider="gemini")

        self._hook("before-agent", {
            "session_id": "geminiprompt123",
            "hook_event_name": "BeforeAgent",
            "prompt": "Fix the issue",
        })
        self._hook("after-agent", {
            "session_id": "geminiprompt123",
            "hook_event_name": "AfterAgent",
            "prompt": "Fix the issue",
            "prompt_response": "Done.",
        })
        self._hook("session-end", {
            "session_id": "geminiprompt123",
            "hook_event_name": "SessionEnd",
            "reason": "exit",
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        meta = store.load_meta(session_id)

        self.assertEqual([e.event_type for e in events], [
            EventType.SESSION_START,
            EventType.USER_PROMPT,
            EventType.ASSISTANT_RESPONSE,
            EventType.SESSION_END,
        ])
        self.assertEqual(events[1].data["prompt"], "Fix the issue")
        self.assertEqual(events[2].data["text"], "Done.")
        self.assertIsNotNone(meta.ended_at)

    def test_legacy_gemini_aliases_are_supported(self):
        self._hook("session-start", {"session_id": "geminialias1234", "source": "startup"})
        session_id = _read_active_session(provider="gemini")

        self._hook("before-tool-call", {
            "session_id": "geminialias1234",
            "tool_name": "read_file",
            "tool_input": {"path": "README.md"},
        })
        self._hook("after-tool-call", {
            "session_id": "geminialias1234",
            "tool_name": "read_file",
            "tool_response": {"llmContent": "readme"},
        })
        self._hook("before-prompt", {
            "session_id": "geminialias1234",
            "prompt": "Summarize",
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        self.assertEqual([e.event_type for e in events[1:]], [
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.USER_PROMPT,
        ])


class TestGeminiSetup(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["GEMINI_CONFIG_DIR"] = self.tmpdir
        self.codex_dir = tempfile.mkdtemp()
        os.environ["CODEX_CONFIG_DIR"] = self.codex_dir
        self.cursor_dir = tempfile.mkdtemp()
        os.environ["CURSOR_CONFIG_DIR"] = self.cursor_dir
        self.copilot_home = tempfile.mkdtemp()
        os.environ["COPILOT_HOME"] = self.copilot_home

    def tearDown(self):
        os.environ.pop("GEMINI_CONFIG_DIR", None)
        os.environ.pop("CODEX_CONFIG_DIR", None)
        os.environ.pop("CURSOR_CONFIG_DIR", None)
        os.environ.pop("COPILOT_HOME", None)

    def test_setup_cli_gemini_writes_extension_files(self):
        args = argparse.Namespace(
            redact=False,
            no_redact=False,
            global_config=False,
            cli="gemini",
        )

        out = io.StringIO()
        err = io.StringIO()
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            cmd_setup(args)

        extension_dir = Path(self.tmpdir) / "extensions" / "agent-strace"
        manifest = json.loads((extension_dir / "gemini-extension.json").read_text())
        hooks = json.loads((extension_dir / "hooks" / "hooks.json").read_text())
        printed_hooks = json.loads(out.getvalue())

        self.assertEqual(manifest["name"], "agent-strace")
        self.assertIn("BeforeTool", hooks["hooks"])
        self.assertIn("AfterTool", hooks["hooks"])
        self.assertIn("BeforeAgent", hooks["hooks"])
        self.assertIn("AfterAgent", hooks["hooks"])
        self.assertIn("SessionEnd", hooks["hooks"])
        self.assertEqual(
            hooks["hooks"]["BeforeTool"][0]["hooks"][0]["command"],
            "agent-strace hook --provider gemini pre-tool",
        )
        self.assertEqual(printed_hooks, hooks)
        self.assertIn("gemini-extension.json", err.getvalue())

    def test_setup_cli_all_includes_gemini_extension(self):
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

        extension_dir = Path(self.tmpdir) / "extensions" / "agent-strace"
        self.assertTrue((extension_dir / "gemini-extension.json").exists())
        self.assertTrue((extension_dir / "hooks" / "hooks.json").exists())
        self.assertTrue((Path(self.codex_dir) / "hooks.json").exists())
        self.assertTrue((Path(self.cursor_dir) / "hooks.json").exists())
        self.assertTrue((Path(self.copilot_home) / "hooks" / "agent-strace.json").exists())
        self.assertIn("agent-strace hook --provider codex user-prompt", out.getvalue())
        self.assertIn("Gemini CLI extension", err.getvalue())
        self.assertIn("GitHub Copilot hooks config", err.getvalue())


if __name__ == "__main__":
    unittest.main()
