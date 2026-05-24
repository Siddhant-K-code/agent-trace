"""Tests for API key authentication on the collector server (Issue #130)."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

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
