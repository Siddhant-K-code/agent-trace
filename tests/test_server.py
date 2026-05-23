"""Tests for the server-side event collector (issue #101)."""

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.server import (
    CollectorHandler,
    _make_handler,
    send_event_to_endpoint,
    send_session_meta_to_endpoint,
)
from agent_trace.store import TraceStore
from http.server import HTTPServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> tuple[TraceStore, str]:
    tmpdir = tempfile.mkdtemp()
    return TraceStore(tmpdir), tmpdir


def _start_test_server(store: TraceStore) -> tuple[HTTPServer, int, threading.Thread]:
    """Start a test server on a random port. Returns (server, port, thread)."""
    lock = threading.Lock()
    handler = _make_handler(store, lock)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, thread


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url: str, body: bytes, content_type: str = "application/json") -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.store, _ = _make_store()
        self.server, self.port, _ = _start_test_server(self.store)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_health_returns_200(self):
        status, body = _get(f"{self.base}/health")
        self.assertEqual(status, 200)

    def test_health_returns_ok(self):
        status, body = _get(f"{self.base}/health")
        data = json.loads(body)
        self.assertEqual(data["status"], "ok")

    def test_health_includes_session_count(self):
        status, body = _get(f"{self.base}/health")
        data = json.loads(body)
        self.assertIn("sessions", data)


# ---------------------------------------------------------------------------
# POST /events
# ---------------------------------------------------------------------------

class TestPostEvents(unittest.TestCase):
    def setUp(self):
        self.store, _ = _make_store()
        self.server, self.port, _ = _start_test_server(self.store)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def _make_event_ndjson(self, session_id: str, n: int = 1) -> bytes:
        lines = []
        for i in range(n):
            ev = TraceEvent(
                event_type=EventType.TOOL_CALL,
                session_id=session_id,
                data={"tool_name": "Bash", "arguments": {"command": f"echo {i}"}},
            )
            lines.append(ev.to_json())
        return "\n".join(lines).encode("utf-8")

    def test_post_events_returns_200(self):
        sid = "test-session-001"
        body = self._make_event_ndjson(sid, 2)
        status, resp = _post(f"{self.base}/events", body, "application/x-ndjson")
        self.assertIn(status, (200, 202))

    def test_post_events_accepted_count(self):
        sid = "test-session-002"
        body = self._make_event_ndjson(sid, 3)
        status, resp = _post(f"{self.base}/events", body, "application/x-ndjson")
        data = json.loads(resp)
        self.assertEqual(data["accepted"], 3)

    def test_post_events_stored_on_disk(self):
        sid = "test-session-003"
        body = self._make_event_ndjson(sid, 2)
        _post(f"{self.base}/events", body, "application/x-ndjson")
        time.sleep(0.05)
        events = self.store.load_events(sid)
        self.assertEqual(len(events), 2)

    def test_post_events_auto_creates_session(self):
        sid = "auto-created-session"
        body = self._make_event_ndjson(sid, 1)
        _post(f"{self.base}/events", body, "application/x-ndjson")
        time.sleep(0.05)
        self.assertTrue(self.store.session_exists(sid))

    def test_post_events_empty_body_returns_200(self):
        status, resp = _post(f"{self.base}/events", b"", "application/x-ndjson")
        self.assertIn(status, (200, 202))

    def test_post_events_invalid_json_counted_as_error(self):
        body = b"not valid json\n"
        status, resp = _post(f"{self.base}/events", body, "application/x-ndjson")
        data = json.loads(resp)
        self.assertGreater(data["errors"], 0)


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------

class TestPostSessions(unittest.TestCase):
    def setUp(self):
        self.store, _ = _make_store()
        self.server, self.port, _ = _start_test_server(self.store)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_post_session_creates_session(self):
        meta = SessionMeta(agent_name="test-agent")
        body = meta.to_json().encode("utf-8")
        status, resp = _post(f"{self.base}/sessions", body)
        self.assertEqual(status, 200)
        time.sleep(0.05)
        self.assertTrue(self.store.session_exists(meta.session_id))

    def test_post_session_missing_session_id_returns_400(self):
        body = json.dumps({"agent_name": "test"}).encode("utf-8")
        status, resp = _post(f"{self.base}/sessions", body)
        self.assertEqual(status, 400)


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------

class TestGetSessions(unittest.TestCase):
    def setUp(self):
        self.store, _ = _make_store()
        self.server, self.port, _ = _start_test_server(self.store)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_get_sessions_empty(self):
        status, body = _get(f"{self.base}/sessions")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data, [])

    def test_get_sessions_returns_created_sessions(self):
        meta = SessionMeta(agent_name="agent-a")
        self.store.create_session(meta)
        status, body = _get(f"{self.base}/sessions")
        data = json.loads(body)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["session_id"], meta.session_id)


# ---------------------------------------------------------------------------
# GET /sessions/<id>/events
# ---------------------------------------------------------------------------

class TestGetSessionEvents(unittest.TestCase):
    def setUp(self):
        self.store, _ = _make_store()
        self.server, self.port, _ = _start_test_server(self.store)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_get_events_returns_ndjson(self):
        meta = SessionMeta()
        self.store.create_session(meta)
        ev = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "Bash"},
        )
        self.store.append_event(meta.session_id, ev)
        status, body = _get(f"{self.base}/sessions/{meta.session_id}/events")
        self.assertEqual(status, 200)
        lines = body.decode("utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["event_type"], "tool_call")

    def test_get_events_unknown_session_returns_404(self):
        status, body = _get(f"{self.base}/sessions/nonexistent/events")
        self.assertEqual(status, 404)


# ---------------------------------------------------------------------------
# send_event_to_endpoint helper
# ---------------------------------------------------------------------------

class TestSendEventToEndpoint(unittest.TestCase):
    def setUp(self):
        self.store, _ = _make_store()
        self.server, self.port, _ = _start_test_server(self.store)
        self.endpoint = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()

    def test_send_event_returns_true_on_success(self):
        ev = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id="remote-session-1",
            data={"tool_name": "Bash"},
        )
        result = send_event_to_endpoint(ev, self.endpoint)
        self.assertTrue(result)

    def test_send_event_to_bad_endpoint_returns_false(self):
        ev = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id="x",
            data={},
        )
        result = send_event_to_endpoint(ev, "http://127.0.0.1:1")
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

class TestServerCLIRegistered(unittest.TestCase):
    def test_server_in_help(self):
        from agent_trace.cli import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "--help"]
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                main()
            except SystemExit:
                pass
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        self.assertIn("server", output)


if __name__ == "__main__":
    unittest.main()
