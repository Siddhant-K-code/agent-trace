"""Tests for OpenAI Codex hooks integration."""

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace import __version__
from agent_trace.cli import _codex_config_dir, _codex_hooks_config, cmd_setup
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
        self.gemini_dir = tempfile.mkdtemp()
        self.cursor_dir = tempfile.mkdtemp()
        self.copilot_home = tempfile.mkdtemp()
        env = patch.dict(os.environ, {
            "CODEX_HOME": self.tmpdir,
            "GEMINI_CONFIG_DIR": self.gemini_dir,
            "CURSOR_CONFIG_DIR": self.cursor_dir,
            "COPILOT_HOME": self.copilot_home,
        })
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("CODEX_CONFIG_DIR", None)
        os.environ.pop("CLAUDE_CONFIG_DIR", None)

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
        self.assertFalse((Path(self.tmpdir) / "hooks.json.agent-strace.bak").exists())
        self.assertIn("No existing OpenAI Codex hooks config to back up", err_text)
        self.assertIn("Wrote OpenAI Codex hooks config", err_text)
        self.assertIn(str(Path(self.tmpdir) / "hooks.json"), err_text)
        self.assertIn("Codex hook checklist", err_text)
        self.assertIn(str(Path(self.tmpdir) / "hooks" / "hooks.json"), err_text)
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

    def test_setup_cli_codex_checklist_respects_codex_home(self):
        with tempfile.TemporaryDirectory() as home:
            args = argparse.Namespace(
                redact=False, no_redact=False, global_config=False, cli="codex"
            )
            out = io.StringIO()
            err = io.StringIO()
            with patch.dict(os.environ, {"CODEX_CONFIG_DIR": "", "CODEX_HOME": home}):
                with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
                    cmd_setup(args)
            text = err.getvalue()
            hooks_path = Path(home) / "hooks.json"
            self.assertTrue(hooks_path.exists())
            self.assertIn(f"Put the JSON at {hooks_path}", text)
            self.assertIn(str(Path(home) / "hooks" / "hooks.json"), text)
            self.assertIn(str(Path(home) / "config.toml"), text)
            self.assertNotIn("~/.codex", text)

    def test_setup_cli_codex_backs_up_existing_file_bytes(self):
        hooks_path = Path(self.tmpdir) / "hooks.json"
        backup_path = Path(self.tmpdir) / "hooks.json.agent-strace.bak"
        original_bytes = b'{"custom": "hooks"}\r\n\xff'
        hooks_path.write_bytes(original_bytes)
        args = argparse.Namespace(
            redact=False,
            no_redact=False,
            global_config=False,
            cli="codex",
        )

        err = io.StringIO()
        with patch.object(sys, "stdout", io.StringIO()), patch.object(sys, "stderr", err):
            cmd_setup(args)

        self.assertEqual(backup_path.read_bytes(), original_bytes)
        self.assertIn(
            f"Backed up existing OpenAI Codex hooks config: {backup_path}",
            err.getvalue(),
        )
        self.assertIn(f"cp {backup_path} {hooks_path}", err.getvalue())

    def test_setup_cli_codex_without_existing_file_does_not_create_backup(self):
        hooks_path = Path(self.tmpdir) / "hooks.json"
        backup_path = Path(self.tmpdir) / "hooks.json.agent-strace.bak"
        args = argparse.Namespace(
            redact=False,
            no_redact=False,
            global_config=False,
            cli="codex",
        )

        err = io.StringIO()
        with patch.object(sys, "stdout", io.StringIO()), patch.object(sys, "stderr", err):
            cmd_setup(args)

        self.assertTrue(hooks_path.exists())
        self.assertFalse(backup_path.exists())
        self.assertIn(
            f"No existing OpenAI Codex hooks config to back up at {hooks_path}",
            err.getvalue(),
        )

    def test_repeated_setup_preserves_original_codex_backup(self):
        hooks_path = Path(self.tmpdir) / "hooks.json"
        backup_path = Path(self.tmpdir) / "hooks.json.agent-strace.bak"
        original_bytes = b'{"original": true}\n'
        hooks_path.write_bytes(original_bytes)
        args = argparse.Namespace(
            redact=False,
            no_redact=False,
            global_config=False,
            cli="codex",
        )

        with patch.object(sys, "stdout", io.StringIO()), patch.object(
            sys, "stderr", io.StringIO()
        ):
            cmd_setup(args)
        hooks_path.write_bytes(b'{"changed_after_first_setup": true}\n')

        err = io.StringIO()
        with patch.object(sys, "stdout", io.StringIO()), patch.object(sys, "stderr", err):
            cmd_setup(args)

        self.assertEqual(backup_path.read_bytes(), original_bytes)
        self.assertIn(
            f"Preserved existing OpenAI Codex hooks backup (not overwritten): {backup_path}",
            err.getvalue(),
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
        self.assertIn(str(Path(self.tmpdir) / "hooks.json"), err.getvalue())
        self.assertIn("GitHub Copilot hooks config", err.getvalue())
        self.assertIn("Codex hook checklist", err.getvalue())


class TestCodexHomeResolution(unittest.TestCase):
    """Setup and --verify use the directory Codex reads: CODEX_HOME, else ~/.codex."""

    _ARGS = {"redact": False, "no_redact": False, "global_config": False}
    _BAD_HOME = (
        "CODEX_HOME is set to {}, which is not an existing directory. Codex rejects "
        "that value. Point CODEX_HOME at an existing directory or unset it.\n"
    )

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.home = self.root / "home"
        self.default_dir = self.home / ".codex"
        self.codex_home = self.root / "codex-home"
        self.legacy_dir = self.root / "legacy"
        self.other_dirs = [self.root / name for name in ("gemini", "cursor", "copilot")]
        for path in (self.home, self.codex_home, self.legacy_dir, *self.other_dirs):
            path.mkdir()
        env = patch.dict(os.environ, {
            "HOME": str(self.home),
            "GEMINI_CONFIG_DIR": str(self.other_dirs[0]),
            "CURSOR_CONFIG_DIR": str(self.other_dirs[1]),
            "COPILOT_HOME": str(self.other_dirs[2]),
        })
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("CODEX_HOME", None)
        os.environ.pop("CODEX_CONFIG_DIR", None)

    def _run(self, env, cli="codex", verify=False):
        args = argparse.Namespace(cli=cli, verify=verify, **self._ARGS)
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env), patch.object(sys, "stdout", out), patch.object(
            sys, "stderr", err
        ):
            status = cmd_setup(args)
        return status, out.getvalue(), err.getvalue()

    def _verify(self, env):
        probe = subprocess.CompletedProcess(
            args=["/bin/agent-strace", "--version"],
            returncode=0,
            stdout=f"agent-strace {__version__}\n",
            stderr="",
        )
        with patch(
            "agent_trace.setup_verify.shutil.which", return_value="/bin/agent-strace"
        ), patch("agent_trace.setup_verify.subprocess.run", return_value=probe):
            return self._run(env, verify=True)

    def _hooks_json(self):
        return json.dumps(_codex_hooks_config(argparse.Namespace(**self._ARGS)))

    def test_resolution_matches_codex(self):
        legacy, codex_home = str(self.legacy_dir), str(self.codex_home)
        cases = [
            ({}, self.default_dir),
            ({"CODEX_HOME": ""}, self.default_dir),
            ({"CODEX_HOME": codex_home}, self.codex_home),
            ({"CODEX_CONFIG_DIR": legacy}, self.default_dir),
            ({"CODEX_CONFIG_DIR": legacy, "CODEX_HOME": ""}, self.default_dir),
            ({"CODEX_CONFIG_DIR": legacy, "CODEX_HOME": codex_home}, self.codex_home),
            ({"CODEX_HOME": "~/codex"}, Path("~/codex")),  # Codex does not expand "~".
        ]
        for env, expected in cases:
            with self.subTest(env=env), patch.dict(os.environ, env):
                self.assertEqual(_codex_config_dir(), expected)

    def test_setup_ignores_codex_config_dir_set_alone(self):
        status, out, err = self._run({"CODEX_CONFIG_DIR": str(self.legacy_dir)})

        self.assertIsNone(status)
        written = json.loads((self.default_dir / "hooks.json").read_text())
        self.assertEqual(written, json.loads(out))
        self.assertEqual(list(self.legacy_dir.iterdir()), [])
        self.assertEqual(err.count("CODEX_CONFIG_DIR"), 1)
        self.assertIn(
            f"Note: ignoring CODEX_CONFIG_DIR={self.legacy_dir}. Codex does not read it",
            err,
        )
        self.assertIn(
            f"may have written {self.legacy_dir / 'hooks.json'}, which Codex does not load",
            err,
        )
        self.assertIn(f"Put the JSON at {self.default_dir / 'hooks.json'}", err)

    def test_setup_prefers_codex_home_when_both_are_set(self):
        status, _, err = self._run({
            "CODEX_CONFIG_DIR": str(self.legacy_dir),
            "CODEX_HOME": str(self.codex_home),
        })

        self.assertIsNone(status)
        self.assertTrue((self.codex_home / "hooks.json").is_file())
        self.assertEqual(list(self.legacy_dir.iterdir()), [])
        self.assertFalse(self.default_dir.exists())
        self.assertEqual(err.count("Note: ignoring CODEX_CONFIG_DIR"), 1)
        self.assertIn(
            f"Wrote OpenAI Codex hooks config: {self.codex_home / 'hooks.json'}", err
        )
        self.assertIn(f"In {self.codex_home / 'config.toml'}", err)

    def test_empty_codex_home_uses_and_creates_default_dir(self):
        self.assertFalse(self.default_dir.exists())

        status, _, err = self._run({"CODEX_HOME": ""})

        self.assertIsNone(status)
        self.assertTrue((self.default_dir / "hooks.json").is_file())
        self.assertIn(
            f"Wrote OpenAI Codex hooks config: {self.default_dir / 'hooks.json'}", err
        )
        self.assertNotIn("CODEX_CONFIG_DIR", err)

    def test_no_note_when_codex_config_dir_is_the_codex_dir(self):
        for env in (
            {"CODEX_CONFIG_DIR": f"{self.codex_home}/", "CODEX_HOME": str(self.codex_home)},
            {"CODEX_CONFIG_DIR": "~/.codex"},
        ):
            with self.subTest(env=env):
                status, _, err = self._run(env)
                self.assertIsNone(status)
                self.assertNotIn("CODEX_CONFIG_DIR", err)

    def test_setup_rejects_codex_home_that_is_not_a_directory_without_writing(self):
        missing = self.root / "missing"
        not_a_dir = self.root / "file"
        not_a_dir.write_text("x")
        for value in (missing, not_a_dir):
            for cli in ("codex", "all"):
                with self.subTest(value=value.name, cli=cli):
                    status, out, err = self._run({"CODEX_HOME": str(value)}, cli=cli)

                    self.assertEqual(status, 1)
                    self.assertEqual(out, "")
                    self.assertEqual(err, "Setup failed: " + self._BAD_HOME.format(value))
                    self.assertFalse(missing.exists())
                    self.assertEqual(not_a_dir.read_text(), "x")
                    self.assertFalse(self.default_dir.exists())
                    for path in self.other_dirs:
                        self.assertEqual(list(path.iterdir()), [])

    def test_verify_checks_codex_home_not_codex_config_dir(self):
        # Older versions verified this file although Codex never loads it.
        (self.legacy_dir / "hooks.json").write_text(self._hooks_json())

        status, out, err = self._verify({"CODEX_CONFIG_DIR": str(self.legacy_dir)})

        self.assertEqual(status, 1)
        self.assertEqual(out, "")
        self.assertIn(f"Missing Codex hooks config: {self.default_dir / 'hooks.json'}", err)
        self.assertEqual(err.count("Note: ignoring CODEX_CONFIG_DIR"), 1)
        self.assertFalse(self.default_dir.exists())

        (self.codex_home / "hooks.json").write_text(self._hooks_json())
        status, out, err = self._verify({
            "CODEX_CONFIG_DIR": str(self.legacy_dir),
            "CODEX_HOME": str(self.codex_home),
        })

        self.assertEqual(status, 0)
        self.assertIn(f"hooks:      {self.codex_home / 'hooks.json'} (", out)
        self.assertIn(f"config:     {self.codex_home / 'config.toml'} (", out)
        self.assertEqual(err.count("Note: ignoring CODEX_CONFIG_DIR"), 1)

    def test_verify_rejects_codex_home_that_is_not_a_directory(self):
        missing = self.root / "missing"

        status, out, err = self._verify({"CODEX_HOME": str(missing)})

        self.assertEqual(status, 1)
        self.assertEqual(out, "")
        self.assertEqual(err, "Setup verification failed: " + self._BAD_HOME.format(missing))
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
