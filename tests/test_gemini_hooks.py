"""Tests for Gemini CLI hooks integration."""

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.cli import _gemini_config_dir, cmd_setup
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
        self.codex_dir = tempfile.mkdtemp()
        self.cursor_dir = tempfile.mkdtemp()
        self.copilot_home = tempfile.mkdtemp()
        env = patch.dict(os.environ, {
            "GEMINI_CLI_HOME": self.tmpdir,
            "CODEX_HOME": self.codex_dir,
            "CURSOR_CONFIG_DIR": self.cursor_dir,
            "COPILOT_HOME": self.copilot_home,
        })
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("CODEX_CONFIG_DIR", None)
        os.environ.pop("GEMINI_CONFIG_DIR", None)

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

        extension_dir = Path(self.tmpdir) / ".gemini" / "extensions" / "agent-strace"
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

        extension_dir = Path(self.tmpdir) / ".gemini" / "extensions" / "agent-strace"
        self.assertTrue((extension_dir / "gemini-extension.json").exists())
        self.assertTrue((extension_dir / "hooks" / "hooks.json").exists())
        self.assertTrue((Path(self.codex_dir) / "hooks.json").exists())
        self.assertTrue((Path(self.cursor_dir) / "hooks.json").exists())
        self.assertTrue((Path(self.copilot_home) / "hooks" / "agent-strace.json").exists())
        self.assertIn("agent-strace hook --provider codex user-prompt", out.getvalue())
        self.assertIn("Gemini CLI extension", err.getvalue())
        self.assertIn("GitHub Copilot hooks config", err.getvalue())


class TestGeminiHomeResolution(unittest.TestCase):
    """Setup installs where Gemini CLI loads extensions: GEMINI_CLI_HOME, else home."""

    _ARGS = {"redact": False, "no_redact": False, "global_config": False, "verify": False}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.home = self.root / "home"
        self.default_dir = self.home / ".gemini"
        self.cli_home = self.root / "gemini-home"
        self.legacy_dir = self.root / "legacy"
        self.other_dirs = [self.root / name for name in ("codex", "cursor", "copilot")]
        for path in (self.home, self.legacy_dir, self.root / "cwd", *self.other_dirs):
            path.mkdir()
        env = patch.dict(os.environ, {
            "HOME": str(self.home),
            "USERPROFILE": str(self.home),
            "CODEX_HOME": str(self.other_dirs[0]),
            "CURSOR_CONFIG_DIR": str(self.other_dirs[1]),
            "COPILOT_HOME": str(self.other_dirs[2]),
        })
        env.start()
        self.addCleanup(env.stop)
        for name in ("GEMINI_CLI_HOME", "GEMINI_CONFIG_DIR", "CODEX_CONFIG_DIR", "CLAUDE_CONFIG_DIR"):
            os.environ.pop(name, None)
        old_cwd = os.getcwd()
        os.chdir(self.root / "cwd")
        self.addCleanup(os.chdir, old_cwd)
        # A relative GEMINI_CLI_HOME starts from the real current directory.
        self.cwd = Path(os.getcwd())

    def _run(self, env, cli="gemini"):
        args = argparse.Namespace(cli=cli, **self._ARGS)
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env), patch.object(sys, "stdout", out), patch.object(
            sys, "stderr", err
        ):
            status = cmd_setup(args)
        return status, out.getvalue(), err.getvalue()

    def _cases(self):
        return [
            ({}, self.default_dir),
            ({"GEMINI_CLI_HOME": ""}, self.default_dir),
            ({"GEMINI_CLI_HOME": str(self.cli_home)}, self.cli_home / ".gemini"),
            ({"GEMINI_CLI_HOME": "rel/home"}, self.cwd / "rel" / "home" / ".gemini"),
            # Gemini CLI does not expand "~".
            ({"GEMINI_CLI_HOME": "~/gh"}, self.cwd / "~" / "gh" / ".gemini"),
        ]

    def test_resolution_matches_gemini_cli(self):
        legacy = str(self.legacy_dir)
        cases = self._cases() + [
            ({"GEMINI_CONFIG_DIR": legacy}, self.default_dir),
            ({"GEMINI_CONFIG_DIR": legacy, "GEMINI_CLI_HOME": ""}, self.default_dir),
            (
                {"GEMINI_CONFIG_DIR": legacy, "GEMINI_CLI_HOME": str(self.cli_home)},
                self.cli_home / ".gemini",
            ),
        ]
        for env, expected in cases:
            with self.subTest(env=env), patch.dict(os.environ, env):
                self.assertEqual(_gemini_config_dir(), expected)

    def test_setup_writes_where_gemini_cli_loads_extensions(self):
        for env, gemini_dir in self._cases():
            for cli in ("gemini", "all"):
                with self.subTest(env=env, cli=cli):
                    self.assertFalse(gemini_dir.exists())

                    status, out, err = self._run(env, cli=cli)

                    extension_dir = gemini_dir / "extensions" / "agent-strace"
                    manifest_path = extension_dir / "gemini-extension.json"
                    hooks_path = extension_dir / "hooks" / "hooks.json"
                    hooks = json.loads(hooks_path.read_text())
                    self.assertIsNone(status)
                    self.assertEqual(json.loads(manifest_path.read_text())["name"], "agent-strace")
                    self.assertEqual(
                        hooks["hooks"]["BeforeTool"][0]["hooks"][0]["command"],
                        "agent-strace hook --provider gemini pre-tool",
                    )
                    if cli == "gemini":
                        self.assertEqual(json.loads(out), hooks)
                    self.assertIn(f"Wrote Gemini CLI extension manifest: {manifest_path}\n", err)
                    self.assertIn(f"Wrote Gemini CLI hooks config: {hooks_path}\n", err)
                    self.assertNotIn("GEMINI_CONFIG_DIR", err)
                    # Nothing lands in the home directory unless Gemini CLI reads from it.
                    expected_home = [self.default_dir] if gemini_dir == self.default_dir else []
                    self.assertEqual(list(self.home.iterdir()), expected_home)
                    shutil.rmtree(gemini_dir)

    def test_missing_gemini_cli_home_is_created(self):
        cli_home = self.root / "missing" / "gemini-home"

        status, _, err = self._run({"GEMINI_CLI_HOME": str(cli_home)})

        hooks_path = cli_home / ".gemini" / "extensions" / "agent-strace" / "hooks" / "hooks.json"
        self.assertIsNone(status)
        self.assertTrue(hooks_path.is_file())
        self.assertIn(f"Wrote Gemini CLI hooks config: {hooks_path}\n", err)

    def test_note_when_gemini_config_dir_is_ignored(self):
        cli_home = str(self.cli_home)
        for env, gemini_dir in (
            ({"GEMINI_CONFIG_DIR": str(self.legacy_dir)}, self.default_dir),
            ({"GEMINI_CONFIG_DIR": str(self.legacy_dir), "GEMINI_CLI_HOME": cli_home}, self.cli_home / ".gemini"),
            # GEMINI_CONFIG_DIR named the .gemini directory, GEMINI_CLI_HOME names its parent.
            ({"GEMINI_CONFIG_DIR": cli_home, "GEMINI_CLI_HOME": cli_home}, self.cli_home / ".gemini"),
        ):
            for cli in ("gemini", "all"):
                with self.subTest(env=env, cli=cli):
                    legacy = Path(env["GEMINI_CONFIG_DIR"])

                    status, _, err = self._run(env, cli=cli)

                    self.assertIsNone(status)
                    self.assertTrue(
                        (gemini_dir / "extensions" / "agent-strace" / "gemini-extension.json").is_file()
                    )
                    self.assertFalse((legacy / "extensions").exists())
                    self.assertEqual(err.count("GEMINI_CONFIG_DIR"), 1)
                    self.assertIn(
                        f"Note: ignoring GEMINI_CONFIG_DIR={legacy}. Gemini CLI does not read it; "
                        "it reads GEMINI_CLI_HOME, else your home directory. Older agent-strace "
                        "versions may have installed the extension under "
                        f"{legacy / 'extensions' / 'agent-strace'}, which Gemini CLI does not load.\n",
                        err,
                    )
                    self.assertLess(err.index("Note: ignoring"), err.index("Wrote Gemini CLI"))
                    shutil.rmtree(gemini_dir)

    def test_no_note_when_gemini_config_dir_is_the_gemini_dir(self):
        for env in (
            {"GEMINI_CONFIG_DIR": "~/.gemini"},
            {"GEMINI_CONFIG_DIR": f"{self.default_dir}{os.sep}"},
            {"GEMINI_CONFIG_DIR": str(self.cli_home / ".gemini"), "GEMINI_CLI_HOME": str(self.cli_home)},
            {"GEMINI_CONFIG_DIR": ""},
        ):
            with self.subTest(env=env):
                status, _, err = self._run(env)

                self.assertIsNone(status)
                self.assertNotIn("GEMINI_CONFIG_DIR", err)


if __name__ == "__main__":
    unittest.main()
