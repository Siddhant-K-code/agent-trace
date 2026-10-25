"""AGENT_TRACE_DIR resolution shared by hook writers and CLI readers."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_trace import cli, hooks
from agent_trace.cli import build_parser, main
from agent_trace.hooks import (
    _PROVIDER_ENV,
    handle_post_tool,
    handle_pre_tool,
    handle_session_end,
    handle_session_start,
    handle_stop,
    handle_user_prompt,
)
from agent_trace.integrations import _base as integrations_base
from agent_trace.models import SessionMeta
from agent_trace.store import DEFAULT_TRACE_DIR, TRACE_DIR_ENV, TraceStore, default_trace_dir

_ISOLATED_ENV = (
    TRACE_DIR_ENV,
    *_PROVIDER_ENV.values(),
    "AGENT_TRACE_REDACT",
    "AGENT_TRACE_NO_REDACT",
    "AGENT_STRACE_NO_REDACT",
    "AGENT_STRACE_TENANT_ID",
    "AGENT_STRACE_WORKSPACE",
    "AGENT_STRACE_ENDPOINT",
)


def _hook_session(session_id: str) -> None:
    """Drive the real Claude hook handlers through one tool call."""
    handle_session_start({"session_id": session_id, "source": "startup"}, provider="claude")
    handle_user_prompt({"session_id": session_id, "prompt": "list files"}, provider="claude")
    handle_pre_tool({
        "session_id": session_id,
        "tool_name": "Bash",
        "tool_use_id": "tool-1",
        "tool_input": {"command": "ls"},
    }, provider="claude")
    handle_post_tool({
        "session_id": session_id,
        "tool_name": "Bash",
        "tool_use_id": "tool-1",
        "tool_response": {"output": "ok"},
    }, provider="claude")
    handle_stop({"session_id": session_id, "last_assistant_message": "done"}, provider="claude")
    handle_session_end({"session_id": session_id}, provider="claude")


class _IsolatedEnv(unittest.TestCase):
    """Clear AGENT_TRACE_DIR and related env, and run inside a fresh cwd."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.workdir = self.root / "work"
        self.workdir.mkdir()

        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for name in _ISOLATED_ENV:
            os.environ.pop(name, None)

        previous_cwd = os.getcwd()
        os.chdir(self.workdir)
        self.addCleanup(os.chdir, previous_cwd)

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        # list_sessions/replay_session bind ``out=sys.stdout`` at import time,
        # so pass the redirected stream explicitly.
        def current_stdout(func):
            return lambda *args, **kwargs: func(*args, out=sys.stdout, **kwargs)

        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["agent-strace", *argv]), \
                patch("agent_trace.cli._product_telemetry.capture"), \
                patch("agent_trace.cli._product_telemetry.maybe_show_telemetry_notice"), \
                patch("agent_trace.cli.list_sessions", current_stdout(cli.list_sessions)), \
                patch("agent_trace.cli.replay_session", current_stdout(cli.replay_session)), \
                redirect_stdout(out), redirect_stderr(err), \
                self.assertRaises(SystemExit) as raised:
            main()
        return raised.exception.code, out.getvalue(), err.getvalue()

    @staticmethod
    def make_session(trace_dir: str | Path, name: str) -> str:
        meta = SessionMeta(agent_name=name)
        TraceStore(trace_dir).create_session(meta)
        return meta.session_id


class TestDefaultTraceDir(_IsolatedEnv):
    def test_unset_uses_default(self):
        self.assertEqual(default_trace_dir(), DEFAULT_TRACE_DIR)

    def test_set_value_wins(self):
        os.environ[TRACE_DIR_ENV] = str(self.root / "shared")
        self.assertEqual(default_trace_dir(), str(self.root / "shared"))

    def test_empty_value_is_treated_as_unset(self):
        os.environ[TRACE_DIR_ENV] = ""
        self.assertEqual(default_trace_dir(), DEFAULT_TRACE_DIR)

    def test_relative_value_is_returned_unchanged(self):
        os.environ[TRACE_DIR_ENV] = "relative/traces"
        self.assertEqual(default_trace_dir(), "relative/traces")


class TestCliParserDefault(_IsolatedEnv):
    def test_default_is_read_when_the_parser_is_built(self):
        self.assertEqual(build_parser().parse_args(["list"]).trace_dir, DEFAULT_TRACE_DIR)
        os.environ[TRACE_DIR_ENV] = str(self.root / "first")
        self.assertEqual(build_parser().parse_args(["list"]).trace_dir, str(self.root / "first"))
        os.environ[TRACE_DIR_ENV] = str(self.root / "second")
        self.assertEqual(build_parser().parse_args(["list"]).trace_dir, str(self.root / "second"))

    def test_read_commands_use_env_dir(self):
        env_dir = str(self.root / "shared")
        os.environ[TRACE_DIR_ENV] = env_dir
        for argv in (["list"], ["replay"], ["inspect", "s"], ["health"], ["stats"],
                     ["mcp"], ["dashboard"], ["watch"]):
            with self.subTest(argv=argv):
                self.assertEqual(build_parser().parse_args(argv).trace_dir, env_dir)

    def test_explicit_trace_dir_overrides_env(self):
        os.environ[TRACE_DIR_ENV] = str(self.root / "shared")
        args = build_parser().parse_args(["--trace-dir", "explicit", "list"])
        self.assertEqual(args.trace_dir, "explicit")
        args = build_parser().parse_args(["--trace-dir=explicit", "health"])
        self.assertEqual(args.trace_dir, "explicit")

    def test_empty_env_falls_back_to_default(self):
        os.environ[TRACE_DIR_ENV] = ""
        self.assertEqual(build_parser().parse_args(["list"]).trace_dir, DEFAULT_TRACE_DIR)

    def test_help_documents_env_precedence(self):
        action = next(
            action for action in build_parser()._actions if action.dest == "trace_dir"
        )
        self.assertIn("$AGENT_TRACE_DIR", action.help)
        self.assertIn(".agent-traces", action.help)


class TestCliReadsHookStore(_IsolatedEnv):
    def setUp(self):
        super().setUp()
        self.env_dir = self.root / "shared" / "traces"
        os.environ[TRACE_DIR_ENV] = str(self.env_dir)

    def test_list_shows_sessions_hooks_wrote_to_env_dir(self):
        _hook_session("envlist000000001")
        self.assertTrue((self.env_dir / "envlist000000001" / "meta.json").is_file())
        self.assertFalse((self.workdir / DEFAULT_TRACE_DIR).exists())

        code, out, err = self.run_cli("list")

        self.assertEqual((code, err), (0, ""))
        self.assertIn("envlist000000001"[:12], out)
        self.assertNotIn("No traces found", out)

    def test_health_reads_env_dir(self):
        _hook_session("envhealth0000001")

        code, out, _ = self.run_cli("health", "envhealth0000001")

        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("Evidence health: "))

    def test_replay_reads_env_dir(self):
        _hook_session("envreplay0000001")

        code, out, err = self.run_cli("replay", "envreplay0000001")

        self.assertEqual(code, 0)
        self.assertNotIn("not found", (out + err).lower())
        self.assertIn("Bash", out)

    def test_explicit_trace_dir_overrides_env_for_list(self):
        _hook_session("envonly000000001")
        explicit = self.root / "explicit"
        explicit_id = self.make_session(explicit, "explicit-agent")

        code, out, _ = self.run_cli("--trace-dir", str(explicit), "list")

        self.assertEqual(code, 0)
        self.assertIn(explicit_id[:12], out)
        self.assertNotIn("envonly000000001"[:12], out)

    def test_empty_env_reads_dot_agent_traces(self):
        os.environ[TRACE_DIR_ENV] = ""
        local_id = self.make_session(DEFAULT_TRACE_DIR, "local-agent")

        code, out, _ = self.run_cli("list")

        self.assertEqual(code, 0)
        self.assertIn(local_id[:12], out)

    def test_relative_env_resolves_against_cwd_for_writer_and_reader(self):
        os.environ[TRACE_DIR_ENV] = "relative/traces"
        _hook_session("envrelative00001")
        self.assertTrue(
            (self.workdir / "relative" / "traces" / "envrelative00001" / "meta.json").is_file()
        )

        code, out, _ = self.run_cli("list")

        self.assertEqual(code, 0)
        self.assertIn("envrelative00001"[:12], out)


class TestWriterParity(_IsolatedEnv):
    def test_hooks_and_integrations_match_cli_when_env_is_set(self):
        os.environ[TRACE_DIR_ENV] = str(self.root / "shared")
        cli_dir = build_parser().parse_args(["list"]).trace_dir
        self.assertEqual(hooks._get_store_dir(), cli_dir)
        self.assertEqual(integrations_base._get_store().base_dir, Path(cli_dir))

    def test_empty_env_is_unset_for_hooks_and_integrations(self):
        os.environ[TRACE_DIR_ENV] = ""
        self.assertEqual(hooks._get_store_dir(), DEFAULT_TRACE_DIR)
        self.assertEqual(integrations_base._get_store().base_dir, Path(DEFAULT_TRACE_DIR))

        payload_dir = str(self.root / "payload" / DEFAULT_TRACE_DIR)
        token = hooks._HOOK_STORE_DIR.set(payload_dir)
        try:
            self.assertEqual(hooks._get_store_dir(), payload_dir)
        finally:
            hooks._HOOK_STORE_DIR.reset(token)

    def test_set_env_still_beats_hook_payload_dir(self):
        os.environ[TRACE_DIR_ENV] = str(self.root / "shared")
        token = hooks._HOOK_STORE_DIR.set(str(self.root / "payload" / DEFAULT_TRACE_DIR))
        try:
            self.assertEqual(hooks._get_store_dir(), str(self.root / "shared"))
        finally:
            hooks._HOOK_STORE_DIR.reset(token)

    def test_empty_env_hook_writes_stay_out_of_the_working_directory_root(self):
        os.environ[TRACE_DIR_ENV] = ""
        _hook_session("envempty00000001")

        self.assertTrue(
            (self.workdir / DEFAULT_TRACE_DIR / "envempty00000001" / "meta.json").is_file()
        )
        self.assertFalse((self.workdir / "envempty00000001").exists())
        self.assertFalse((self.workdir / ".store.lock").exists())


if __name__ == "__main__":
    unittest.main()
