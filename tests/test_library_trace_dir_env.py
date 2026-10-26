"""AGENT_TRACE_DIR resolution for the Python library entry points.

The CLI, hooks and integrations already honor ``$AGENT_TRACE_DIR``. These
tests cover the library defaults (``start_session``, ``import_jsonl``, the
workspace helpers) and the CLI fallbacks that used to hard-code
``.agent-traces``.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import agent_trace
from agent_trace.cli import main
from agent_trace.compliance_report import ComplianceReportError, cmd_compliance_report
from agent_trace.decorator import end_session, start_session, trace_tool
from agent_trace.eval.dataset import cmd_dataset
from agent_trace.hooks import _PROVIDER_ENV
from agent_trace.jsonl_import import import_jsonl
from agent_trace.models import EventType
from agent_trace.store import (
    DEFAULT_TRACE_DIR,
    TRACE_DIR_ENV,
    TraceStore,
    default_trace_dir,
    resolve_trace_dir,
)
from agent_trace.workspace import (
    create_workspace,
    delete_workspace,
    list_workspaces,
    workspace_session_count,
)

_ISOLATED_ENV = (
    TRACE_DIR_ENV,
    *_PROVIDER_ENV.values(),
    "AGENT_TRACE_REDACT",
    "AGENT_TRACE_NO_REDACT",
    "AGENT_STRACE_NO_REDACT",
    "AGENT_STRACE_TENANT_ID",
    "AGENT_STRACE_WORKSPACE",
    "AGENT_STRACE_ENDPOINT",
    "AGENT_STRACE_STORAGE",
)


@trace_tool
def _lookup(query: str) -> str:
    return f"result for {query}"


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
        self.addCleanup(end_session)

    @property
    def cwd_store(self) -> Path:
        return self.workdir / DEFAULT_TRACE_DIR

    @staticmethod
    def run_session(**kwargs) -> str:
        session_id = start_session(name="lib-agent", **kwargs)
        _lookup("auth")
        meta = end_session()
        assert meta is not None and meta.session_id == session_id
        return session_id

    def assert_session_in(self, trace_dir: Path | str, session_id: str) -> None:
        store = TraceStore(trace_dir)
        self.assertTrue(store.session_exists(session_id), f"{session_id} not in {trace_dir}")
        meta = store.load_meta(session_id)
        self.assertEqual(meta.agent_name, "lib-agent")
        self.assertEqual(meta.tool_calls, 1)
        types = [e.event_type for e in store.load_events(session_id)]
        self.assertEqual(types[0], EventType.SESSION_START)
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertEqual(types[-1], EventType.SESSION_END)


class TestResolveTraceDir(_IsolatedEnv):
    def test_none_with_env_unset_uses_default(self):
        self.assertEqual(resolve_trace_dir(None), DEFAULT_TRACE_DIR)

    def test_none_with_env_set_uses_env(self):
        os.environ[TRACE_DIR_ENV] = str(self.root / "env-store")
        self.assertEqual(resolve_trace_dir(None), str(self.root / "env-store"))

    def test_none_with_empty_env_uses_default(self):
        os.environ[TRACE_DIR_ENV] = ""
        self.assertEqual(resolve_trace_dir(None), DEFAULT_TRACE_DIR)

    def test_explicit_value_wins_over_env(self):
        os.environ[TRACE_DIR_ENV] = str(self.root / "env-store")
        self.assertEqual(resolve_trace_dir("explicit"), "explicit")
        self.assertEqual(resolve_trace_dir(""), "")
        explicit = self.root / "explicit"
        self.assertIs(resolve_trace_dir(explicit), explicit)


class TestStartSession(_IsolatedEnv):
    def test_default_is_resolved_at_call_time(self):
        default = inspect.signature(start_session).parameters["trace_dir"].default
        self.assertIsNone(default)

    def test_env_unset_writes_dot_agent_traces(self):
        session_id = self.run_session()
        self.assert_session_in(self.cwd_store, session_id)

    def test_empty_env_writes_dot_agent_traces(self):
        os.environ[TRACE_DIR_ENV] = ""
        session_id = self.run_session()
        self.assert_session_in(self.cwd_store, session_id)

    def test_env_set_writes_env_dir_readable_by_default_store(self):
        env_dir = self.root / "env-store"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        session_id = self.run_session()

        self.assert_session_in(default_trace_dir(), session_id)
        self.assertEqual(
            [m.session_id for m in TraceStore(default_trace_dir()).list_sessions()],
            [session_id],
        )
        self.assertFalse(self.cwd_store.exists())

    def test_explicit_trace_dir_overrides_env(self):
        env_dir = self.root / "env-store"
        explicit = self.root / "explicit"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        session_id = self.run_session(trace_dir=str(explicit))

        self.assert_session_in(explicit, session_id)
        self.assertFalse(env_dir.exists())
        self.assertFalse(self.cwd_store.exists())

    def test_env_set_after_import_applies_on_each_call(self):
        # agent_trace.decorator was imported at module load, before any env.
        first, second = self.root / "first", self.root / "second"
        os.environ[TRACE_DIR_ENV] = str(first)
        first_id = self.run_session()
        os.environ[TRACE_DIR_ENV] = str(second)
        second_id = self.run_session()

        self.assert_session_in(first, first_id)
        self.assert_session_in(second, second_id)
        self.assertFalse(TraceStore(first).session_exists(second_id))

    def test_relative_env_resolves_against_cwd(self):
        os.environ[TRACE_DIR_ENV] = "rel/store"
        session_id = self.run_session()

        self.assert_session_in(self.workdir / "rel" / "store", session_id)
        self.assert_session_in(default_trace_dir(), session_id)
        self.assertFalse(self.cwd_store.exists())

    def test_top_level_package_api_honors_env(self):
        env_dir = self.root / "env-store"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        session_id = agent_trace.start_session(name="lib-agent")
        _lookup("auth")
        agent_trace.end_session()

        self.assert_session_in(env_dir, session_id)


class TestImportJsonl(_IsolatedEnv):
    def write_log(self) -> Path:
        path = self.root / "session.jsonl"
        entries = [
            {
                "type": "user",
                "sessionId": "claude-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "message": {"role": "user", "content": "Hello"},
                "uuid": "u1",
            },
            {
                "type": "assistant",
                "sessionId": "claude-1",
                "timestamp": "2025-01-01T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi"}],
                },
                "uuid": "a1",
                "parentUuid": "u1",
            },
        ]
        path.write_text("".join(json.dumps(e) + "\n" for e in entries))
        return path

    def test_default_import_lands_in_env_dir(self):
        env_dir = self.root / "env-store"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        session_id = import_jsonl(self.write_log())

        self.assertTrue(TraceStore(default_trace_dir()).session_exists(session_id))
        self.assertFalse(self.cwd_store.exists())

    def test_default_import_without_env_uses_dot_agent_traces(self):
        session_id = import_jsonl(self.write_log())
        self.assertTrue(TraceStore(self.cwd_store).session_exists(session_id))

    def test_explicit_trace_dir_overrides_env(self):
        env_dir, explicit = self.root / "env-store", self.root / "explicit"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        session_id = import_jsonl(self.write_log(), trace_dir=str(explicit))

        self.assertTrue(TraceStore(explicit).session_exists(session_id))
        self.assertFalse(env_dir.exists())

    def test_explicit_store_overrides_env(self):
        env_dir, explicit = self.root / "env-store", self.root / "explicit"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        session_id = import_jsonl(self.write_log(), store=TraceStore(explicit))

        self.assertTrue(TraceStore(explicit).session_exists(session_id))
        self.assertFalse(env_dir.exists())


class TestWorkspaceHelpers(_IsolatedEnv):
    def test_helpers_use_env_dir_by_default(self):
        env_dir = self.root / "env-store"
        os.environ[TRACE_DIR_ENV] = str(env_dir)

        path = create_workspace("staging")
        self.assertEqual(path, env_dir / "workspaces" / "staging")
        self.assertEqual(list_workspaces(), ["staging"])
        self.assertEqual(workspace_session_count("staging"), 0)
        self.assertTrue(delete_workspace("staging"))
        self.assertEqual(list_workspaces(), [])
        self.assertFalse(self.cwd_store.exists())

    def test_helpers_use_dot_agent_traces_without_env(self):
        path = create_workspace("staging")
        self.assertEqual(path.resolve(), self.cwd_store / "workspaces" / "staging")
        self.assertEqual(list_workspaces(), ["staging"])

    def test_explicit_trace_dir_overrides_env(self):
        env_dir, explicit = self.root / "env-store", self.root / "explicit"
        os.environ[TRACE_DIR_ENV] = str(env_dir)

        create_workspace("staging", str(explicit))
        self.assertEqual(list_workspaces(str(explicit)), ["staging"])
        self.assertEqual(list_workspaces(), [])
        self.assertTrue(delete_workspace("staging", str(explicit)))


class TestCliFallbacks(_IsolatedEnv):
    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["agent-strace", *argv]), \
                patch("agent_trace.cli._product_telemetry.capture"), \
                patch("agent_trace.cli._product_telemetry.maybe_show_telemetry_notice"), \
                redirect_stdout(out), redirect_stderr(err), \
                self.assertRaises(SystemExit) as raised:
            main()
        return raised.exception.code, out.getvalue(), err.getvalue()

    def write_config(self) -> str:
        cfg = self.root / ".agent-strace.yaml"
        cfg.write_text("workspaces:\n  - name: prod\n")
        return str(cfg)

    def test_dataset_auto_without_trace_dir_uses_env(self):
        env_dir = self.root / "env-store"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        args = argparse.Namespace(
            dataset_command="auto",
            dataset=str(self.root / "ds.jsonl"),
            filter="has-errors",
            since="7d",
            label="",
        )
        with patch("agent_trace.eval.dataset.auto_populate", return_value=0) as auto, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(cmd_dataset(args), 0)
        self.assertEqual(auto.call_args.args[0].storage_root, env_dir)

    def test_dataset_auto_explicit_trace_dir_wins(self):
        os.environ[TRACE_DIR_ENV] = str(self.root / "env-store")
        explicit = self.root / "explicit"
        args = argparse.Namespace(
            dataset_command="auto",
            dataset=str(self.root / "ds.jsonl"),
            filter="has-errors",
            since="7d",
            label="",
            trace_dir=str(explicit),
        )
        with patch("agent_trace.eval.dataset.auto_populate", return_value=0) as auto, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(cmd_dataset(args), 0)
        self.assertEqual(auto.call_args.args[0].storage_root, explicit)

    def test_compliance_report_without_trace_dir_uses_env(self):
        env_dir = self.root / "env-store"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        seen: list = []

        def stop(base_dir):
            seen.append(base_dir)
            raise ComplianceReportError("stop")

        args = argparse.Namespace(framework="soc2", format="json", output=None)
        with patch("agent_trace.compliance_report.enumerate_trace_stores", side_effect=stop), \
                redirect_stderr(io.StringIO()):
            self.assertEqual(cmd_compliance_report(args), 1)
        self.assertEqual(seen, [str(env_dir)])

    def test_rbac_uses_env_dir(self):
        env_dir = self.root / "env-store"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        code, out, _ = self.run_cli(
            "rbac", "assign", "--user", "alice@example.com", "--role", "admin",
        )
        self.assertEqual(code, 0)
        self.assertTrue((env_dir / "rbac.json").exists())
        self.assertFalse(self.cwd_store.exists())

        code, out, _ = self.run_cli("rbac", "list")
        self.assertEqual(code, 0)
        self.assertIn("alice@example.com", out)

    def test_rbac_global_trace_dir_overrides_env(self):
        env_dir, explicit = self.root / "env-store", self.root / "explicit"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        code, _, _ = self.run_cli(
            "--trace-dir", str(explicit),
            "rbac", "assign", "--user", "alice@example.com", "--role", "admin",
        )
        self.assertEqual(code, 0)
        self.assertTrue((explicit / "rbac.json").exists())
        self.assertFalse(env_dir.exists())

    def test_rbac_without_env_uses_dot_agent_traces(self):
        code, _, _ = self.run_cli(
            "rbac", "assign", "--user", "alice@example.com", "--role", "admin",
        )
        self.assertEqual(code, 0)
        self.assertTrue((self.cwd_store / "rbac.json").exists())

    def test_apply_uses_env_dir_and_workspace_list_agrees(self):
        env_dir = self.root / "env-store"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        cfg = self.write_config()

        code, out, _ = self.run_cli("apply", "--config", cfg)
        self.assertEqual(code, 0, out)
        self.assertTrue((env_dir / "workspaces" / "prod").is_dir())
        self.assertFalse(self.cwd_store.exists())

        code, out, _ = self.run_cli("workspace", "list")
        self.assertEqual(code, 0)
        self.assertIn("prod", out)

        code, out, _ = self.run_cli("config-diff", "--config", cfg)
        self.assertEqual(code, 0)
        self.assertIn("No drift detected.", out)

    def test_apply_global_trace_dir_overrides_env(self):
        env_dir, explicit = self.root / "env-store", self.root / "explicit"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        code, out, _ = self.run_cli(
            "--trace-dir", str(explicit), "apply", "--config", self.write_config(),
        )
        self.assertEqual(code, 0, out)
        self.assertTrue((explicit / "workspaces" / "prod").is_dir())
        self.assertFalse(env_dir.exists())

    def test_apply_dir_flag_wins(self):
        env_dir = self.root / "env-store"
        explicit, flag_dir = self.root / "explicit", self.root / "flag-dir"
        os.environ[TRACE_DIR_ENV] = str(env_dir)
        cfg = self.write_config()
        code, out, _ = self.run_cli(
            "--trace-dir", str(explicit), "apply", "--config", cfg, "--dir", str(flag_dir),
        )
        self.assertEqual(code, 0, out)
        self.assertTrue((flag_dir / "workspaces" / "prod").is_dir())
        self.assertFalse(env_dir.exists())
        self.assertFalse(explicit.exists())

        code, out, _ = self.run_cli("config-diff", "--config", cfg)
        self.assertEqual(code, 0)
        self.assertIn("+ workspace: prod", out)


if __name__ == "__main__":
    unittest.main()
