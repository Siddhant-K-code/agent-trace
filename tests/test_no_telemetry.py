"""The test suite never sends product telemetry; see tests/_no_telemetry.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry

from agent_trace import telemetry

SRC = Path(__file__).resolve().parent.parent / "src"
AGENT_STRACE = [sys.executable, "-m", "agent_trace.entry"]
SESSION = {"session_id": "s1"}

# Each case runs agent-strace in a child and names the event it would send.
CASES = {
    "cli": ([(["list"], "")], telemetry.CLI_COMMAND_COMPLETED),
    "hook": (
        [
            (["hook", "session-start"], json.dumps(SESSION)),
            (["hook", "session-end"], json.dumps({**SESSION, "reason": "exit"})),
        ],
        telemetry.SESSION_COMPLETED,
    ),
}


class _Collector(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.received.append(json.loads(body))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


class TestNoTelemetryFromTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        home = self.root / "home"
        home.mkdir()
        config = self.root / "telemetry.json"
        # The notice was shown, so the disclosure gate does not stop a send.
        config.write_text(json.dumps({"notice_shown": True}))

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
        server.received = []
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.received = server.received

        # Built from scratch like the leaking tests' envs: no CI, no opt-out.
        self.env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "USERPROFILE": str(home),
            "PYTHONPATH": str(SRC),
            "AGENT_TRACE_DIR": str(self.root / "traces"),
            "AGENT_STRACE_TELEMETRY_CONFIG": str(config),
            "AGENT_STRACE_TELEMETRY_HOST": f"http://127.0.0.1:{server.server_port}",
        }
        if "SYSTEMROOT" in os.environ:
            self.env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

    def run_steps(self, steps, env):
        self.received.clear()
        for argv, stdin in steps:
            kwargs = {} if env is None else {"env": env}
            proc = subprocess.run(
                AGENT_STRACE + argv, input=stdin, capture_output=True, text=True,
                cwd=self.root, timeout=60, **kwargs,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
        return [event["event"] for event in self.received]

    def test_an_explicit_opt_in_still_sends_to_its_own_host(self):
        # Without this, the checks below could pass because nothing captures.
        for name, (steps, event) in CASES.items():
            with self.subTest(name):
                env = {**self.env, "AGENT_STRACE_TELEMETRY": "1"}
                self.assertEqual(self.run_steps(steps, env), [event])

    def test_a_child_with_a_scratch_env_sends_nothing(self):
        for name, (steps, _) in CASES.items():
            with self.subTest(name):
                self.assertEqual(self.run_steps(steps, self.env), [])

    def test_a_child_inheriting_a_cleared_environ_sends_nothing(self):
        for name, (steps, _) in CASES.items():
            with self.subTest(name), patch.dict(os.environ, self.env, clear=True):
                self.assertEqual(self.run_steps(steps, None), [])

    def test_an_opt_in_without_a_host_is_pointed_at_a_closed_port(self):
        env = {**self.env, "AGENT_STRACE_TELEMETRY": "1"}
        del env["AGENT_STRACE_TELEMETRY_HOST"]
        proc = subprocess.run(
            [sys.executable, "-c", "import os; print(os.environ['AGENT_STRACE_TELEMETRY_HOST'])"],
            env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.stdout.strip(), _no_telemetry.CLOSED_HOST)

    def test_in_process_capture_cannot_reach_posthog(self):
        guard = urllib.request.urlopen
        self.assertTrue(getattr(guard, _no_telemetry._MARK, False))
        network = Mock()
        env = {key: self.env[key] for key in ("HOME", "USERPROFILE", "AGENT_STRACE_TELEMETRY_CONFIG")}
        with patch.dict(os.environ, env, clear=True), patch.object(guard, "__wrapped__", network):
            self.assertTrue(telemetry.telemetry_enabled())
            self.assertTrue(telemetry.disclosure_satisfied())
            self.assertFalse(telemetry.capture(telemetry.CLI_COMMAND_COMPLETED, {"command": "list"}))
            network.assert_not_called()
            guard("http://127.0.0.1:9/not-posthog")
        network.assert_called_once()


if __name__ == "__main__":
    unittest.main()
