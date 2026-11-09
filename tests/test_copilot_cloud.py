"""Tests for the Copilot cloud agent repository hook template."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

from agent_trace import __version__
from agent_trace.cli import _copilot_hooks_config, build_parser, cmd_setup
from agent_trace.copilot_cloud import (
    CLOUD_HOOK_EVENTS,
    HOOKS_RELATIVE_PATH,
    SCRIPT_RELATIVE_PATH,
    find_repo_root,
    hook_script,
    hooks_config,
    setup_steps_workflow,
    write_repository_files,
)
from agent_trace.models import EventType
from agent_trace.server import _make_handler
from agent_trace.store import TraceStore

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
BASH = shutil.which("bash")

SESSION_ID = "cloudsession1234567890"
PAYLOADS = (
    ("session-start", {"sessionId": SESSION_ID, "source": "new", "cwd": "/workspace"}),
    ("user-prompt", {"sessionId": SESSION_ID, "prompt": "Fix the failing test"}),
    ("pre-tool", {
        "sessionId": SESSION_ID,
        "toolName": "bash",
        "toolArgs": {"command": "pytest -q"},
    }),
    ("post-tool", {
        "sessionId": SESSION_ID,
        "toolName": "bash",
        "toolResult": {"resultType": "success", "textResultForLlm": "1 passed"},
    }),
    ("stop", {"sessionId": SESSION_ID, "stopReason": "end_turn"}),
    ("session-end", {"sessionId": SESSION_ID, "reason": "complete"}),
)


def _setup_args(**overrides) -> argparse.Namespace:
    values = {
        "redact": False,
        "no_redact": False,
        "global_config": False,
        "cli": "copilot-cloud",
        "verify": False,
        "live": False,
        "session": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_shim(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestCopilotCloudHooksConfig(unittest.TestCase):
    def test_registers_every_copilot_hook_as_a_bash_entry(self):
        config = hooks_config()

        self.assertEqual(config["version"], 1)
        self.assertEqual(
            list(config["hooks"]),
            [copilot_event for copilot_event, _, _ in CLOUD_HOOK_EVENTS],
        )
        for copilot_event, hook_event, timeout in CLOUD_HOOK_EVENTS:
            entries = config["hooks"][copilot_event]
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry["type"], "command")
            self.assertEqual(entry["bash"], f"bash .github/hooks/agent-strace-cloud.sh {hook_event}")
            self.assertEqual(entry["cwd"], ".")
            self.assertEqual(entry["timeoutSec"], timeout)
            self.assertNotIn("powershell", entry)
            self.assertNotIn("env", entry)

    def test_cloud_events_match_local_copilot_hooks(self):
        local = _copilot_hooks_config(_setup_args(cli="copilot"))["hooks"]
        cloud = hooks_config()["hooks"]

        self.assertEqual(set(cloud), set(local))
        for copilot_event, hook_event, _ in CLOUD_HOOK_EVENTS:
            self.assertEqual(
                local[copilot_event][0]["command"],
                f"agent-strace hook --provider copilot {hook_event}",
            )

    def test_redaction_flags_become_hook_env(self):
        config = hooks_config({"AGENT_TRACE_NO_REDACT": "1"})
        for entries in config["hooks"].values():
            self.assertEqual(entries[0]["env"], {"AGENT_TRACE_NO_REDACT": "1"})

    def test_install_hooks_get_time_for_one_install(self):
        timeouts = {event: timeout for event, _, timeout in CLOUD_HOOK_EVENTS}
        self.assertGreaterEqual(timeouts["sessionStart"], 60)
        self.assertGreaterEqual(timeouts["userPromptSubmitted"], 60)


class TestCopilotCloudSetupSteps(unittest.TestCase):
    def test_workflow_uses_the_required_job_and_pins_the_version(self):
        workflow = setup_steps_workflow()

        self.assertIn("\njobs:\n  copilot-setup-steps:\n", workflow)
        self.assertEqual(workflow.count("copilot-setup-steps:"), 1)
        self.assertIn(f'"agent-strace=={__version__}"', workflow)
        self.assertIn("--target /tmp/agent-strace-pkg", workflow)
        self.assertIn("contents: read", workflow)
        self.assertNotIn("\t", workflow)
        self.assertNotIn("\u2014", workflow)

    def test_workflow_job_uses_only_allowed_keys(self):
        allowed = {"steps", "permissions", "runs-on", "services", "snapshot", "timeout-minutes"}
        lines = setup_steps_workflow().splitlines()
        job_index = lines.index("  copilot-setup-steps:")
        keys = {
            line.strip().split(":", 1)[0]
            for line in lines[job_index + 1:]
            if line.startswith("    ") and not line.startswith("     ")
        }
        self.assertTrue(keys)
        self.assertLessEqual(keys, allowed)


@unittest.skipIf(BASH is None, "bash is required")
class TestCopilotCloudHookScript(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.repo = self.root / "workspace"
        (self.repo / ".git").mkdir(parents=True)
        write_repository_files(self.repo)
        self.trace_dir = self.root / "traces"
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.install_dir = self.root / "pkg"

    def _env(self, **extra) -> dict[str, str]:
        env = {
            "PATH": f"{self.bin_dir}:/usr/bin:/bin",
            "HOME": str(self.root),
            "AGENT_TRACE_DIR": str(self.trace_dir),
            "AGENT_STRACE_INSTALL_DIR": str(self.install_dir),
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        }
        env.update(extra)
        return env

    def _run(self, event: str, payload: dict, env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [BASH, str(SCRIPT_RELATIVE_PATH), event],
            cwd=self.repo,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

    def _agent_strace_shim(self) -> None:
        _write_shim(
            self.bin_dir,
            "agent-strace",
            f'PYTHONPATH="{SRC_DIR}" exec "{sys.executable}" -m agent_trace.cli "$@"\n',
        )

    def _python3_shim(self, log: Path | None = None, fail_pip: bool = False) -> None:
        body = ""
        if log is not None:
            body += f'echo "$@" >> "{log}"\n'
        if fail_pip:
            body += 'case "$*" in *"-m pip"*) exit 1;; esac\n'
        body += f'exec "{sys.executable}" "$@"\n'
        _write_shim(self.bin_dir, "python3", body)

    def _recorded_events(self, store_dir: Path):
        store = TraceStore(store_dir)
        sessions = store.list_sessions()
        self.assertEqual(len(sessions), 1)
        return sessions[0], store.load_events(sessions[0].session_id)

    def test_generated_files_are_written_and_executable(self):
        script = self.repo / SCRIPT_RELATIVE_PATH
        hooks = json.loads((self.repo / HOOKS_RELATIVE_PATH).read_text())

        self.assertEqual(hooks, hooks_config())
        self.assertEqual(script.read_text(), hook_script())
        self.assertTrue(os.access(script, os.X_OK))
        self.assertIn(f'version="{__version__}"', script.read_text())
        self.assertNotIn("\u2014", script.read_text())
        syntax = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_records_hook_events_with_agent_strace_on_path(self):
        self._agent_strace_shim()
        env = self._env(AGENT_STRACE_INSTALL="0")
        for event, payload in PAYLOADS:
            result = self._run(event, payload, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")

        meta, events = self._recorded_events(self.trace_dir)
        types = [event.event_type for event in events]
        self.assertEqual(meta.agent_name, "github-copilot")
        self.assertEqual(types[0], EventType.SESSION_START)
        self.assertIn(EventType.USER_PROMPT, types)
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertIn(EventType.TOOL_RESULT, types)
        self.assertEqual(types[-1], EventType.SESSION_END)
        call = next(event for event in events if event.event_type == EventType.TOOL_CALL)
        self.assertEqual(call.data["arguments"]["command"], "pytest -q")

    def test_runs_from_install_dir_when_not_on_path(self):
        self._python3_shim()
        self.install_dir.mkdir()
        (self.install_dir / "agent_trace").symlink_to(SRC_DIR / "agent_trace")
        env = self._env()
        for event, payload in PAYLOADS[:2]:
            result = self._run(event, payload, env)
            self.assertEqual(result.returncode, 0, result.stderr)

        _, events = self._recorded_events(self.trace_dir)
        self.assertEqual(
            [event.event_type for event in events][:2],
            [EventType.SESSION_START, EventType.USER_PROMPT],
        )
        self.assertFalse(Path(f"{self.install_dir}.lock").exists())

    def test_attempts_one_pinned_install_and_never_fails_the_hook(self):
        log = self.root / "python3.log"
        self._python3_shim(log=log, fail_pip=True)
        env = self._env()
        for event, payload in PAYLOADS:
            result = self._run(event, payload, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")

        calls = log.read_text().splitlines()
        self.assertEqual(len(calls), 1)
        self.assertIn("-m pip install", calls[0])
        self.assertIn(f"--target {self.install_dir}", calls[0])
        self.assertIn(f"agent-strace=={__version__}", calls[0])
        self.assertFalse(self.trace_dir.exists())

    def test_install_can_be_disabled(self):
        log = self.root / "python3.log"
        self._python3_shim(log=log)
        result = self._run("pre-tool", PAYLOADS[2][1], self._env(AGENT_STRACE_INSTALL="0"))

        self.assertEqual(result.returncode, 0)
        self.assertFalse(log.exists())
        self.assertFalse(Path(f"{self.install_dir}.lock").exists())

    def test_hook_failure_still_exits_zero(self):
        _write_shim(self.bin_dir, "agent-strace", "cat >/dev/null; exit 1\n")
        result = self._run("pre-tool", PAYLOADS[2][1], self._env(AGENT_STRACE_INSTALL="0"))

        self.assertEqual(result.returncode, 0)

    def test_session_start_warns_when_no_endpoint_is_set(self):
        self._agent_strace_shim()
        result = self._run("session-start", PAYLOADS[0][1], self._env(AGENT_STRACE_INSTALL="0"))

        self.assertEqual(result.returncode, 0)
        self.assertIn("AGENT_STRACE_ENDPOINT is not set", result.stderr)

    def test_streams_events_to_an_authenticated_collector(self):
        self._agent_strace_shim()
        collector_store = TraceStore(self.root / "collector")
        auth_key = "ast_cloudtestkey"
        server = HTTPServer(
            ("127.0.0.1", 0),
            _make_handler(collector_store, threading.Lock(), auth_key=auth_key),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        env = self._env(
            AGENT_STRACE_INSTALL="0",
            AGENT_STRACE_ENDPOINT=endpoint,
            AGENT_STRACE_AUTH_KEY=auth_key,
        )
        for event, payload in PAYLOADS:
            result = self._run(event, payload, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("AGENT_STRACE_ENDPOINT is not set", result.stderr)

        meta, events = self._recorded_events(self.root / "collector")
        types = [event.event_type for event in events]
        self.assertEqual(meta.agent_name, "github-copilot")
        self.assertIn(EventType.SESSION_START, types)
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertIn(EventType.SESSION_END, types)
        local = TraceStore(self.trace_dir)
        for session in local.list_sessions():
            self.assertEqual(local.load_events(session.session_id), [])


class TestCopilotCloudSetupCommand(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo, True)
        (self.repo / ".git").mkdir()
        self.nested = self.repo / "src" / "pkg"
        self.nested.mkdir(parents=True)
        self.previous_cwd = os.getcwd()
        self.addCleanup(os.chdir, self.previous_cwd)

    def _setup(self, **overrides) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            code = cmd_setup(_setup_args(**overrides))
        return code, out.getvalue(), err.getvalue()

    def test_writes_files_at_the_repository_root(self):
        os.chdir(self.nested)
        code, out, err = self._setup()

        self.assertEqual(code, 0)
        hooks_path = self.repo / HOOKS_RELATIVE_PATH
        script_path = self.repo / SCRIPT_RELATIVE_PATH
        self.assertEqual(json.loads(hooks_path.read_text()), json.loads(out))
        self.assertTrue(os.access(script_path, os.X_OK))
        self.assertFalse((self.nested / ".github").exists())
        self.assertFalse((self.repo / ".github" / "workflows").exists())
        self.assertIn("default branch", err)
        self.assertIn("AGENT_STRACE_ENDPOINT", err)
        self.assertIn("firewall", err)
        self.assertIn("not supported", err)
        self.assertIn(setup_steps_workflow(), err)
        self.assertNotIn("\u2014", out + err)

    def test_no_redact_is_passed_through_hook_env(self):
        os.chdir(self.repo)
        code, out, _ = self._setup(no_redact=True)

        self.assertEqual(code, 0)
        for entries in json.loads(out)["hooks"].values():
            self.assertEqual(entries[0]["env"], {"AGENT_TRACE_NO_REDACT": "1"})

    def test_verify_is_rejected(self):
        os.chdir(self.repo)
        code, _, err = self._setup(verify=True)

        self.assertEqual(code, 2)
        self.assertIn("supports only --cli codex or --cli claude", err)
        self.assertFalse((self.repo / ".github").exists())

    def test_find_repo_root_falls_back_to_start(self):
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, outside, True)
        self.assertEqual(find_repo_root(self.nested), self.repo.resolve())
        if not any((parent / ".git").exists() for parent in outside.resolve().parents):
            self.assertEqual(find_repo_root(outside), outside.resolve())

    def test_parser_accepts_copilot_cloud(self):
        args = build_parser().parse_args(["setup", "--cli", "copilot-cloud"])
        self.assertEqual(args.cli, "copilot-cloud")


if __name__ == "__main__":
    unittest.main()
