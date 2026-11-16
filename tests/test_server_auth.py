"""Tests for API key authentication on the collector server (Issue #130)."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent_trace.cli import build_parser
from agent_trace.server import (
    KEY_PREFIX,
    _auth_headers,
    _make_handler,
    cmd_server,
    generate_api_key,
    run_server,
    send_event_to_endpoint,
    send_session_meta_to_endpoint,
)
from agent_trace.store import TraceStore
from agent_trace.models import EventType, SessionMeta, TraceEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> TraceStore:
    return TraceStore(Path(tempfile.mkdtemp()))


def _start_server(store: TraceStore, auth_key: str = "", port: int = 0):
    """Start a test server on an ephemeral port. Returns (server, port)."""
    from http.server import HTTPServer
    lock = threading.Lock()
    handler = _make_handler(store, lock, auth_key=auth_key)
    server = HTTPServer(("127.0.0.1", port), handler)
    actual_port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, actual_port


def _post(url: str, body: bytes, content_type: str, headers: dict | None = None) -> tuple[int, bytes]:
    h = {"Content-Type": content_type, **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _get(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _make_event(session_id: str = "test-session-001") -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        timestamp=1_000_000.0,
        session_id=session_id,
        data={"tool_name": "Bash", "arguments": {"command": "ls"}},
    )


# ---------------------------------------------------------------------------
# generate_api_key
# ---------------------------------------------------------------------------

class TestGenerateApiKey(unittest.TestCase):

    def test_starts_with_prefix(self):
        key = generate_api_key()
        self.assertTrue(key.startswith(KEY_PREFIX))

    def test_length(self):
        # ast_ (4) + 32 hex chars = 36
        key = generate_api_key()
        self.assertEqual(len(key), 36)

    def test_hex_suffix(self):
        key = generate_api_key()
        suffix = key[len(KEY_PREFIX):]
        self.assertTrue(all(c in "0123456789abcdef" for c in suffix))

    def test_unique(self):
        keys = {generate_api_key() for _ in range(20)}
        self.assertEqual(len(keys), 20)

    def test_prefix_constant(self):
        self.assertEqual(KEY_PREFIX, "ast_")


# ---------------------------------------------------------------------------
# _auth_headers
# ---------------------------------------------------------------------------

class TestAuthHeaders(unittest.TestCase):

    def test_empty_when_no_env(self):
        os.environ.pop("AGENT_STRACE_AUTH_KEY", None)
        self.assertEqual(_auth_headers(), {})

    def test_returns_bearer_when_set(self):
        os.environ["AGENT_STRACE_AUTH_KEY"] = "ast_abc123"
        try:
            h = _auth_headers()
            self.assertEqual(h.get("Authorization"), "Bearer ast_abc123")
        finally:
            del os.environ["AGENT_STRACE_AUTH_KEY"]

    def test_empty_string_env_gives_no_header(self):
        os.environ["AGENT_STRACE_AUTH_KEY"] = ""
        try:
            self.assertEqual(_auth_headers(), {})
        finally:
            del os.environ["AGENT_STRACE_AUTH_KEY"]


# ---------------------------------------------------------------------------
# Server: no-auth mode (existing behaviour unchanged)
# ---------------------------------------------------------------------------

class TestServerNoAuth(unittest.TestCase):

    def setUp(self):
        self.store = _make_store()
        self.server, self.port = _start_server(self.store, auth_key="")
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_health_no_auth_required(self):
        status, body = _get(f"{self.base}/health")
        self.assertEqual(status, 200)

    def test_post_events_no_auth_required(self):
        event = _make_event()
        status, _ = _post(
            f"{self.base}/events",
            (event.to_json() + "\n").encode(),
            "application/x-ndjson",
        )
        self.assertEqual(status, 200)

    def test_get_sessions_no_auth_required(self):
        status, _ = _get(f"{self.base}/sessions")
        self.assertEqual(status, 200)

    def test_no_auth_header_still_works(self):
        status, _ = _get(f"{self.base}/health")
        self.assertNotEqual(status, 401)


# ---------------------------------------------------------------------------
# Server: auth mode — valid key accepted
# ---------------------------------------------------------------------------

class TestServerAuthValid(unittest.TestCase):

    KEY = "ast_deadbeef1234567890abcdef123456"

    def setUp(self):
        self.store = _make_store()
        self.server, self.port = _start_server(self.store, auth_key=self.KEY)
        self.base = f"http://127.0.0.1:{self.port}"
        self.auth = {"Authorization": f"Bearer {self.KEY}"}

    def tearDown(self):
        self.server.shutdown()

    def test_health_with_valid_key(self):
        status, _ = _get(f"{self.base}/health", headers=self.auth)
        self.assertEqual(status, 200)

    def test_post_events_with_valid_key(self):
        event = _make_event()
        status, _ = _post(
            f"{self.base}/events",
            (event.to_json() + "\n").encode(),
            "application/x-ndjson",
            headers=self.auth,
        )
        self.assertEqual(status, 200)

    def test_get_sessions_with_valid_key(self):
        status, _ = _get(f"{self.base}/sessions", headers=self.auth)
        self.assertEqual(status, 200)

    def test_post_sessions_with_valid_key(self):
        meta = SessionMeta(agent_name="test")
        status, _ = _post(
            f"{self.base}/sessions",
            meta.to_json().encode(),
            "application/json",
            headers=self.auth,
        )
        self.assertEqual(status, 200)


# ---------------------------------------------------------------------------
# Server: auth mode — missing key rejected
# ---------------------------------------------------------------------------

class TestServerAuthMissing(unittest.TestCase):

    KEY = "ast_cafebabe1234567890abcdef123456"

    def setUp(self):
        self.store = _make_store()
        self.server, self.port = _start_server(self.store, auth_key=self.KEY)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_health_without_key_returns_401(self):
        status, _ = _get(f"{self.base}/health")
        self.assertEqual(status, 401)

    def test_post_events_without_key_returns_401(self):
        event = _make_event()
        status, _ = _post(
            f"{self.base}/events",
            (event.to_json() + "\n").encode(),
            "application/x-ndjson",
        )
        self.assertEqual(status, 401)

    def test_get_sessions_without_key_returns_401(self):
        status, _ = _get(f"{self.base}/sessions")
        self.assertEqual(status, 401)

    def test_401_body_is_json(self):
        status, body = _get(f"{self.base}/health")
        self.assertEqual(status, 401)
        data = json.loads(body)
        self.assertIn("error", data)


# ---------------------------------------------------------------------------
# Server: auth mode — wrong key rejected
# ---------------------------------------------------------------------------

class TestServerAuthWrongKey(unittest.TestCase):

    KEY = "ast_rightkey1234567890abcdef1234"

    def setUp(self):
        self.store = _make_store()
        self.server, self.port = _start_server(self.store, auth_key=self.KEY)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_wrong_key_returns_401(self):
        status, _ = _get(
            f"{self.base}/health",
            headers={"Authorization": "Bearer ast_wrongkey000000000000000000000"},
        )
        self.assertEqual(status, 401)

    def test_partial_key_returns_401(self):
        status, _ = _get(
            f"{self.base}/health",
            headers={"Authorization": f"Bearer {self.KEY[:-1]}"},
        )
        self.assertEqual(status, 401)

    def test_no_bearer_prefix_returns_401(self):
        status, _ = _get(
            f"{self.base}/health",
            headers={"Authorization": self.KEY},
        )
        self.assertEqual(status, 401)

    def test_empty_bearer_returns_401(self):
        status, _ = _get(
            f"{self.base}/health",
            headers={"Authorization": "Bearer "},
        )
        self.assertEqual(status, 401)


# ---------------------------------------------------------------------------
# cmd_server keygen
# ---------------------------------------------------------------------------

class TestCmdServerKeygen(unittest.TestCase):

    def _run_keygen(self) -> str:
        args = argparse.Namespace(
            server_subcommand="keygen",
            port=4317,
            storage=None,
            host="0.0.0.0",
            auth_key=None,
            trace_dir=Path(tempfile.mkdtemp()),
        )
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            result = cmd_server(args)
        finally:
            sys.stdout = old
        return buf.getvalue().strip(), result

    def test_keygen_returns_0(self):
        _, rc = self._run_keygen()
        self.assertEqual(rc, 0)

    def test_keygen_prints_key(self):
        key, _ = self._run_keygen()
        self.assertTrue(key.startswith(KEY_PREFIX))

    def test_keygen_key_correct_length(self):
        key, _ = self._run_keygen()
        self.assertEqual(len(key), 36)

    def test_keygen_produces_unique_keys(self):
        keys = {self._run_keygen()[0] for _ in range(5)}
        self.assertEqual(len(keys), 5)


# ---------------------------------------------------------------------------
# cmd_server refuses the SSO flags: the server has no SSO
# ---------------------------------------------------------------------------

_SECRET = "csecret-4f1d9a"
_SSO_ARGS = {
    "--auth": ["--auth", "oidc"],
    "--oidc-issuer": ["--oidc-issuer", "https://idp.invalid"],
    "--oidc-client-id": ["--oidc-client-id", "cid"],
    "--oidc-client-secret": ["--oidc-client-secret", _SECRET],
    "--enforce-sso": ["--enforce-sso"],
}
_ALL_SSO_ARGS = [arg for argv in _SSO_ARGS.values() for arg in argv]


def _refusal(flags: str) -> str:
    return (
        "Error: agent-strace server has no SSO, so it refuses to start with "
        f"{flags}; to require authentication, use --auth-key instead\n"
    )


class TestCmdServerRefusesSsoFlags(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.storage = Path(tmp.name) / "traces"

    def _run(self, *argv):
        args = build_parser().parse_args(
            ["server", "--storage", str(self.storage), *argv]
        )
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ), \
                mock.patch("agent_trace.server.run_server") as run, \
                redirect_stdout(out), redirect_stderr(err):
            os.environ.pop("AGENT_STRACE_AUTH_KEY", None)
            rc = cmd_server(args)
        return rc, out.getvalue(), err.getvalue(), run

    def _assert_refused(self, argv, flags):
        rc, out, err, run = self._run(*argv)
        self.assertEqual(rc, 1)
        self.assertNotIn(_SECRET, err)
        self.assertEqual(err, _refusal(flags))
        self.assertEqual(out, "")
        run.assert_not_called()
        self.assertFalse(self.storage.exists())

    def test_each_flag_alone_is_refused(self):
        for flag, argv in _SSO_ARGS.items():
            with self.subTest(flag=flag):
                self._assert_refused(argv, flag)

    def test_all_five_flags_are_refused_and_named(self):
        self._assert_refused(
            _ALL_SSO_ARGS,
            "--auth, --oidc-issuer, --oidc-client-id, --oidc-client-secret, "
            "--enforce-sso",
        )

    def test_an_empty_oidc_value_is_refused(self):
        for flag in ("--oidc-issuer", "--oidc-client-id", "--oidc-client-secret"):
            with self.subTest(flag=flag):
                self._assert_refused([flag, ""], flag)

    def test_auth_key_does_not_lift_the_refusal(self):
        for flag, argv in _SSO_ARGS.items():
            with self.subTest(flag=flag):
                self._assert_refused([*argv, "--auth-key", "ast_probekey"], flag)

    def test_the_refusal_comes_before_the_host_check(self):
        self._assert_refused(
            ["--auth", "oidc", "--dashboard", "--host", "0.0.0.0"], "--auth"
        )

    def test_keygen_still_prints_a_key_with_the_flags(self):
        rc, out, err, run = self._run(*_ALL_SSO_ARGS, "keygen")
        self.assertEqual(rc, 0)
        self.assertRegex(out, r"^ast_[0-9a-f]{32}\n$")
        self.assertEqual(err, "")
        run.assert_not_called()

    def test_auth_key_alone_still_starts_the_server(self):
        rc, _, err, run = self._run("--auth-key", "ast_probekey")
        self.assertEqual(rc, 0)
        self.assertEqual(err, "")
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["auth_key"], "ast_probekey")

    def test_a_namespace_without_the_sso_attributes_still_starts(self):
        args = argparse.Namespace(
            server_subcommand=None, port=0, storage=str(self.storage),
            host=None, dashboard=False, auth_key="ast_probekey",
        )
        with mock.patch("agent_trace.server.run_server") as run:
            self.assertEqual(cmd_server(args), 0)
        run.assert_called_once()

    def test_help_does_not_list_the_sso_flags(self):
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(["server", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = out.getvalue()
        self.assertIn("--auth-key", text)
        for flag in _SSO_ARGS:
            with self.subTest(flag=flag):
                self.assertNotRegex(text, re.escape(flag) + r"(?![\w-])")


# ---------------------------------------------------------------------------
# AGENT_STRACE_AUTH_KEY env var injected into client send functions
# ---------------------------------------------------------------------------

class TestClientAuthInjection(unittest.TestCase):

    KEY = "ast_clienttest1234567890abcdef12"

    def setUp(self):
        self.store = _make_store()
        self.server, self.port = _start_server(self.store, auth_key=self.KEY)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        os.environ.pop("AGENT_STRACE_AUTH_KEY", None)

    def test_send_event_fails_without_env_key(self):
        os.environ.pop("AGENT_STRACE_AUTH_KEY", None)
        event = _make_event()
        result = send_event_to_endpoint(event, self.base)
        self.assertFalse(result)

    def test_send_event_succeeds_with_env_key(self):
        os.environ["AGENT_STRACE_AUTH_KEY"] = self.KEY
        event = _make_event()
        result = send_event_to_endpoint(event, self.base)
        self.assertTrue(result)

    def test_send_session_meta_fails_without_env_key(self):
        os.environ.pop("AGENT_STRACE_AUTH_KEY", None)
        meta = SessionMeta(agent_name="test")
        result = send_session_meta_to_endpoint(meta, self.base)
        self.assertFalse(result)

    def test_send_session_meta_succeeds_with_env_key(self):
        os.environ["AGENT_STRACE_AUTH_KEY"] = self.KEY
        meta = SessionMeta(agent_name="test")
        result = send_session_meta_to_endpoint(meta, self.base)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
