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
from unittest import mock
from urllib.parse import quote
from pathlib import Path

from agent_trace.audit import verify_chain
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.server import (
    CollectorHandler,
    _make_handler,
    resolve_server_host,
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


class TestDashboardBinding(unittest.TestCase):
    def test_dashboard_defaults_to_loopback(self):
        self.assertEqual(
            resolve_server_host(None, dashboard=True, auth_key=""),
            "127.0.0.1",
        )

    def test_collector_without_dashboard_keeps_remote_default(self):
        self.assertEqual(
            resolve_server_host(None, dashboard=False, auth_key=""),
            "0.0.0.0",
        )

    def test_unauthenticated_remote_dashboard_is_rejected(self):
        for host in ("0.0.0.0", "::", "192.0.2.10", "dashboard.example"):
            with self.subTest(host=host):
                with self.assertRaisesRegex(
                    ValueError,
                    "unauthenticated dashboard access is limited to loopback",
                ):
                    resolve_server_host(host, dashboard=True, auth_key="")

    def test_authenticated_remote_dashboard_is_allowed(self):
        self.assertEqual(
            resolve_server_host("0.0.0.0", dashboard=True, auth_key="ast_test"),
            "0.0.0.0",
        )

    def test_loopback_dashboard_does_not_require_auth(self):
        for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost"):
            with self.subTest(host=host):
                self.assertEqual(
                    resolve_server_host(host, dashboard=True, auth_key=""),
                    host,
                )


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


_SECRET = "SECRET-DATA-VALUE"
_NO_DATA = object()


def _event_record(session_id: str, event_id: str, data: object = _NO_DATA) -> dict:
    record = {
        "event_type": "tool_call",
        "timestamp": 1760000000.0,
        "event_id": event_id,
        "session_id": session_id,
    }
    if data is not _NO_DATA:
        record["data"] = data
    return record


class TestPostEventsNonObjectData(unittest.TestCase):
    """POST /events refuses an event whose data is present but not an object."""

    def setUp(self):
        self.store, _ = _make_store()
        self.server, self.port, _ = _start_test_server(self.store)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _post_records(self, *records: dict) -> tuple[int, dict, str]:
        """POST one NDJSON line per record. Returns status, body, and server stderr."""
        body = "".join(json.dumps(record) + "\n" for record in records)
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            status, resp = _post(
                f"{self.base}/events", body.encode("utf-8"), "application/x-ndjson"
            )
        return status, json.loads(resp), stderr.getvalue()

    def _stored(self, session_id: str) -> list[dict]:
        path = self.store.base_dir / session_id / "events.ndjson"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_each_non_object_data_type_is_refused(self):
        for name, data, phrase in (
            ("array", [_SECRET], "an array"),
            ("string", _SECRET, "a string"),
            ("empty-string", "", "a string"),
            ("number", 42, "a number"),
            ("zero", 0, "a number"),
            ("float", 4.5, "a number"),
            ("true", True, "a boolean"),
            ("false", False, "a boolean"),
            ("null", None, "null"),
        ):
            with self.subTest(data=name):
                session_id = f"refused-{name}"
                status, resp, stderr = self._post_records(
                    _event_record(session_id, "evt-1", data)
                )
                self.assertEqual(status, 202)
                self.assertEqual(resp, {"accepted": 0, "errors": 1})
                self.assertFalse(self.store.session_exists(session_id))
                self.assertIn(
                    "[server] Refused one tool_call event because its data is "
                    f"{phrase}, not an object.\n",
                    stderr,
                )

    def test_object_events_around_a_refused_event_are_stored_in_order(self):
        session_id = "mixed-batch"
        status, resp, _ = self._post_records(
            _event_record(session_id, "evt-1", {"tool_name": "first"}),
            _event_record(session_id, "evt-2", [_SECRET]),
            _event_record(session_id, "evt-3", {"tool_name": "third"}),
        )
        self.assertEqual(status, 202)
        self.assertEqual(resp, {"accepted": 2, "errors": 1})
        stored = self._stored(session_id)
        self.assertEqual([event["event_id"] for event in stored], ["evt-1", "evt-3"])
        self.assertEqual(
            [event["data"] for event in stored],
            [{"tool_name": "first"}, {"tool_name": "third"}],
        )
        # The server computes prev_hash from what it stored, so the chain holds.
        self.assertTrue(verify_chain(self.store, session_id).ok)

    def test_event_without_data_is_stored_with_an_empty_object(self):
        session_id = "no-data-key"
        status, resp, stderr = self._post_records(_event_record(session_id, "evt-1"))
        self.assertEqual(status, 200)
        self.assertEqual(resp, {"accepted": 1, "errors": 0})
        self.assertEqual([event["data"] for event in self._stored(session_id)], [{}])
        self.assertNotIn("Refused", stderr)

    def test_object_with_a_wrong_typed_field_is_stored(self):
        session_id = "wrong-typed-field"
        status, resp, _ = self._post_records(
            _event_record(session_id, "evt-1", {"tool_name": 5})
        )
        self.assertEqual(status, 200)
        self.assertEqual(resp, {"accepted": 1, "errors": 0})
        self.assertEqual(
            [event["data"] for event in self._stored(session_id)], [{"tool_name": 5}]
        )

    def test_batch_of_only_refused_events_creates_no_session(self):
        session_id = "only-refused"
        status, resp, _ = self._post_records(
            _event_record(session_id, "evt-1", [_SECRET]),
            _event_record(session_id, "evt-2", _SECRET),
        )
        self.assertEqual(status, 202)
        self.assertEqual(resp, {"accepted": 0, "errors": 2})
        self.assertFalse((self.store.base_dir / session_id).exists())
        self.assertEqual(self.store.list_sessions(), [])

    def test_refusal_lines_have_no_data_values_or_event_ids(self):
        _, _, stderr = self._post_records(
            _event_record("leak-check", "SECRET-EVENT-ID-1", [_SECRET, {"key": _SECRET}]),
            _event_record("leak-check", "SECRET-EVENT-ID-2", _SECRET),
        )
        self.assertNotIn(_SECRET, stderr)
        self.assertNotIn("SECRET-EVENT-ID", stderr)
        self.assertEqual(
            [line for line in stderr.splitlines() if "Refused" in line],
            [
                "[server] Refused one tool_call event because its data is an array, "
                "not an object.",
                "[server] Refused one tool_call event because its data is a string, "
                "not an object.",
            ],
        )


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

    def test_post_session_from_newer_client_with_unknown_field(self):
        # A client one release ahead may send fields this collector does not
        # know. The session is still created and the field is kept.
        meta = json.loads(SessionMeta(session_id="newer-client-001").to_json())
        meta["field_from_a_future_release"] = "kept"
        status, _ = _post(f"{self.base}/sessions", json.dumps(meta).encode("utf-8"))
        self.assertEqual(status, 200)
        stored = self.store.base_dir / "newer-client-001" / "meta.json"
        self.assertEqual(json.loads(stored.read_text())["field_from_a_future_release"], "kept")

    def test_existing_tenant_cannot_be_cleared_or_changed(self):
        meta = SessionMeta(session_id="tenant-session", tenant_id="tenant-a")
        self.store.create_session(meta)
        for tenant_id in ("", "tenant-b"):
            body = json.dumps({
                "session_id": meta.session_id, "tenant_id": tenant_id,
            }).encode("utf-8")
            status, _ = _post(f"{self.base}/sessions", body)
            self.assertEqual(status, 409)
        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "tenant-a")

    def test_first_tenant_assignment_retags_existing_events(self):
        meta = SessionMeta(session_id="legacy-tenant-session", tenant_id="")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=meta.session_id,
            data={"prompt": "hello"},
            tenant_id="",
        ))
        body = json.dumps({
            "session_id": meta.session_id, "tenant_id": "tenant-a",
        }).encode("utf-8")

        status, _ = _post(f"{self.base}/sessions", body)

        self.assertEqual(status, 200)
        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "tenant-a")
        raw = json.loads(
            (self.store._session_dir(meta.session_id) / "events.ndjson").read_text()
        )
        self.assertEqual(raw["tenant_id"], "tenant-a")

    def test_path_traversal_session_id_is_rejected(self):
        status, _ = _post(
            f"{self.base}/sessions",
            json.dumps({"session_id": "../escape"}).encode("utf-8"),
        )
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

    def test_get_reads_url_decoded_safe_legacy_id_but_post_rejects_it(self):
        legacy_id = "Legacy café session " + "x" * 130
        session_dir = self.store.base_dir / legacy_id
        session_dir.mkdir()
        meta = SessionMeta(session_id=legacy_id)
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=legacy_id,
            data={"tool_name": "Read"},
        )
        (session_dir / "meta.json").write_text(meta.to_json())
        (session_dir / "events.ndjson").write_text(event.to_json() + "\n")
        encoded = quote(legacy_id, safe="")

        status, body = _get(f"{self.base}/sessions/{encoded}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["session_id"], legacy_id)
        status, body = _get(f"{self.base}/sessions/{encoded}/events")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["session_id"], legacy_id)

        status, _ = _post(
            f"{self.base}/sessions", meta.to_json().encode("utf-8")
        )
        self.assertEqual(status, 400)

    def test_get_rejects_url_encoded_path_separator(self):
        encoded = quote("../escape", safe="")
        status, _ = _get(f"{self.base}/sessions/{encoded}/events")
        self.assertEqual(status, 400)

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
