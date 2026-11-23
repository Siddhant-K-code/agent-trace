"""Tests for the server-side event collector (issue #101)."""

import io
import hashlib
import http.client
import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock
from urllib.parse import quote
from pathlib import Path

from agent_trace.audit import verify_chain
from agent_trace.cli import build_parser, cmd_export
from agent_trace.collector_client import CollectorClient
from agent_trace.hooks import handle_session_end, handle_session_start
from agent_trace.integrations._base import emit
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.server import (
    CollectorHandler,
    _make_handler,
    cmd_server,
    resolve_server_host,
    send_event_to_endpoint,
    send_session_meta_to_endpoint,
)
from agent_trace.store import TraceStore
from http import HTTPStatus
from http.server import HTTPServer

try:
    from _redirect_servers import ALL_REDIRECTS, REDIRECT_NOTE, RedirectServers
    from test_fileutil import RAW_SEPARATORS
    from test_store import (
        OTHER_SESSION, OTHER_TENANT, event_line, name_other_session_in_meta, write_session_lines,
    )
except ImportError:
    from tests._redirect_servers import ALL_REDIRECTS, REDIRECT_NOTE, RedirectServers
    from tests.test_fileutil import RAW_SEPARATORS
    from tests.test_store import (
        OTHER_SESSION, OTHER_TENANT, event_line, name_other_session_in_meta, write_session_lines,
    )

try:
    import _no_telemetry  # noqa: F401
except ImportError:
    from tests import _no_telemetry  # noqa: F401


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
        with e:
            return e.code, e.read()


def _post(
    url: str, body: bytes, content_type: str = "application/json",
    headers: dict | None = None,
) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": content_type, **(headers or {})},
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

    def test_post_events_keeps_a_raw_line_separator_inside_a_value(self):
        # JS JSON.stringify and serde_json leave these characters raw.
        for separator in RAW_SEPARATORS:
            with self.subTest(separator=hex(ord(separator))):
                sid = f"raw-separator-{ord(separator):04x}"
                records = [
                    {"event_type": "tool_call", "session_id": sid, "timestamp": 1.0,
                     "data": {"tool_name": "Bash",
                              "arguments": {"command": f"echo a{separator}b"}}},
                    {"event_type": "tool_result", "session_id": sid, "timestamp": 2.0,
                     "data": {"tool_name": "Bash", "result": "ok"}},
                ]
                body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
                status, resp = _post(
                    f"{self.base}/events", body.encode("utf-8"), "application/x-ndjson"
                )
                self.assertEqual((status, json.loads(resp)), (200, {"accepted": 2, "errors": 0}))
                events = self.store.load_events(sid)
                self.assertEqual(
                    [event.event_type for event in events],
                    [EventType.TOOL_CALL, EventType.TOOL_RESULT],
                )
                self.assertEqual(events[0].data["arguments"]["command"], f"echo a{separator}b")


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


_GOOD_TIMESTAMP = 1767225600.5
_BIG_INT = int("1" + "0" * 400)
# Not a Unix time in seconds. Each is valid JSON that json.dumps can write.
_BAD_TIMESTAMPS = (
    ("iso-text", "2026-01-01T00:00:00Z"),
    ("null", None),
    ("true", True),
    ("object", {"key": _SECRET}),
    ("nan", float("nan")),
    ("infinity", float("inf")),
    ("negative", -5),
    ("1e300", 1e300),
    ("milliseconds", 1767225600000),
    ("400-digit-int", _BIG_INT),
)
_TIMESTAMP_REFUSAL = (
    "[server] Refused one tool_call event because its timestamp is not a Unix "
    "time in seconds."
)


def _timed_record(session_id: str, event_id: str, timestamp: object) -> dict:
    record = _event_record(session_id, event_id, {"tool_name": event_id})
    record["timestamp"] = timestamp
    return record


def _without_access_log(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if not line.startswith('[server] "')]


class TestPostEventsNonObjectData(unittest.TestCase):
    """POST /events refuses an event whose data is present but not an object.

    It also refuses an event whose timestamp is not a Unix time in seconds.
    """

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

    def test_each_unusable_timestamp_is_refused(self):
        for name, timestamp in _BAD_TIMESTAMPS:
            with self.subTest(timestamp=name):
                session_id = f"bad-time-{name}"
                status, resp, stderr = self._post_records(
                    _timed_record(session_id, "evt-1", timestamp)
                )
                self.assertEqual(status, 202)
                self.assertEqual(resp, {"accepted": 0, "errors": 1})
                self.assertFalse((self.store.base_dir / session_id).exists())
                lines = _without_access_log(stderr)
                self.assertEqual(lines, [_TIMESTAMP_REFUSAL])
                self.assertNotRegex(lines[0], r"\d")
                for shown in (json.dumps(timestamp), str(timestamp), _SECRET):
                    self.assertNotIn(shown.lower(), lines[0].lower())

    def test_good_or_absent_timestamp_is_stored(self):
        absent = _timed_record("absent-time", "evt-1", None)
        del absent["timestamp"]
        before = time.time()
        for record in (_timed_record("good-time", "evt-1", _GOOD_TIMESTAMP), absent):
            with self.subTest(session=record["session_id"]):
                status, resp, stderr = self._post_records(record)
                self.assertEqual(status, 200)
                self.assertEqual(resp, {"accepted": 1, "errors": 0})
                self.assertEqual(_without_access_log(stderr), [])
        self.assertEqual(
            [event["timestamp"] for event in self._stored("good-time")], [_GOOD_TIMESTAMP]
        )
        [stored] = self._stored("absent-time")
        self.assertGreaterEqual(stored["timestamp"], before)
        self.assertLessEqual(stored["timestamp"], time.time())

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

    def test_events_around_a_refused_timestamp_are_stored_in_order(self):
        session_id = "mixed-times"
        status, resp, _ = self._post_records(
            _timed_record(session_id, "evt-1", _GOOD_TIMESTAMP),
            _timed_record(session_id, "evt-2", 1767225600000),
            _timed_record(session_id, "evt-3", _GOOD_TIMESTAMP + 1),
        )
        self.assertEqual(status, 202)
        self.assertEqual(resp, {"accepted": 2, "errors": 1})
        stored = self._stored(session_id)
        self.assertEqual([event["event_id"] for event in stored], ["evt-1", "evt-3"])
        self.assertEqual(
            [event["timestamp"] for event in stored], [_GOOD_TIMESTAMP, _GOOD_TIMESTAMP + 1]
        )
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

    def test_batch_of_only_refused_timestamps_creates_no_session(self):
        session_id = "only-bad-times"
        status, resp, _ = self._post_records(
            _timed_record(session_id, "evt-1", -5),
            _timed_record(session_id, "evt-2", None),
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

    def test_first_tenant_assignment_rejects_non_string(self):
        meta = SessionMeta(session_id="legacy-invalid-tenant", tenant_id="")
        self.store.create_session(meta)
        body = json.dumps({
            "session_id": meta.session_id,
            "tenant_id": ["tenant-a"],
        }).encode("utf-8")

        status, response = _post(f"{self.base}/sessions", body)

        self.assertEqual(status, 400)
        self.assertIn(b"tenant ID must be a string", response)
        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "")

    def test_path_traversal_session_id_is_rejected(self):
        status, _ = _post(
            f"{self.base}/sessions",
            json.dumps({"session_id": "../escape"}).encode("utf-8"),
        )
        self.assertEqual(status, 400)


def _store_files(store: TraceStore) -> dict[str, bytes]:
    """Map each file under the store to its bytes."""
    base = Path(store.base_dir)
    return {
        path.relative_to(base).as_posix(): path.read_bytes()
        for path in base.rglob("*")
        if path.is_file()
    }


def _nested_lists(depth: int) -> str:
    return "[" * depth + "]" * depth


def _nested_objects(depth: int) -> str:
    return '{"a": ' * (depth - 1) + "{}" + "}" * (depth - 1)


def _body(session_id: object, *fields: str) -> bytes:
    """Build a POST /sessions body from the ID and raw '"key": value' texts."""
    return ("{" + ", ".join([f'"session_id": {json.dumps(session_id)}', *fields]) + "}").encode()


# Each case is (label, raw '"key": value' JSON text, the 400 error). The text
# stays raw so NaN, 1e400 and 400-digit numbers reach the server as written.
_REFUSED_FIELDS = [
    ("tool_calls text", '"tool_calls": "lots"', "tool_calls must be a non-negative integer"),
    ("tool_calls negative", '"tool_calls": -1', "tool_calls must be a non-negative integer"),
    ("depth bool", '"depth": true', "depth must be a non-negative integer"),
    ("ended_at text", '"ended_at": "soon"', "ended_at must be a number"),
    ("attribution text", '"attribution": "me"', "attribution must be an object"),
    ("started_at text", '"started_at": "x"', "started_at must be a number"),
    ("started_at bool", '"started_at": true', "started_at must be a number"),
    ("started_at NaN", '"started_at": NaN', "non-finite JSON number 'NaN' is not allowed"),
    ("started_at 1e400", '"started_at": 1e400', "non-finite JSON number 'Infinity' is not allowed"),
    (
        "started_at 400 digits", '"started_at": ' + "9" * 400,
        "started_at must be a finite non-negative number",
    ),
    (
        "total_duration_ms 400 digits", '"total_duration_ms": ' + "9" * 400,
        "total_duration_ms must be a finite non-negative number",
    ),
    ("started_at negative", '"started_at": -5', "started_at must be a finite non-negative number"),
    ("started_at zero", '"started_at": 0', "started_at is required"),
]

# A create keeps unknown keys, so these are refused there. An update ignores them.
_REFUSED_UNKNOWN_KEYS = [
    ("unknown key NaN", '"zzz": NaN', "non-finite JSON number 'NaN' is not allowed"),
    (
        "unknown key 64 nested lists", '"zzz": ' + _nested_lists(64),
        "collector JSON nesting exceeds the limit",
    ),
    (
        "unknown key 64 nested objects", '"zzz": ' + _nested_objects(64),
        "collector JSON nesting exceeds the limit",
    ),
]

# A time sent in milliseconds. CollectorClient reads it, since it is a finite
# number, but datetime.fromtimestamp refuses it, so the store's own commands
# cannot use it.
_MILLISECONDS = "1780000000000"


class TestPostSessionsStoresReadableMeta(unittest.TestCase):
    """POST /sessions stores only metadata that CollectorClient can read, with
    times that list, stats and the tenant commands can use."""

    SESSION = "exist00000000001"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name, use_workspace_env=False)
        server, port, _ = _start_test_server(self.store)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{port}/sessions"
        status, _ = self.post(_body(self.SESSION, '"started_at": 1780000000.0', '"zzz": 1'))
        self.assertEqual(status, 200)
        # An event, so a tenant tag would show as a rewritten events file.
        self.store.append_event(self.SESSION, TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=self.SESSION,
            data={"prompt": "hi"},
        ))

    def post(self, body: bytes) -> tuple[int, dict]:
        status, raw = _post(self.url, body)
        return status, json.loads(raw)

    def test_update_ignores_keys_that_are_not_meta_fields(self):
        before = _store_files(self.store)
        cases = [
            ("method name", '"to_json": "x"'),
            ("unknown-field store", '"_unknown_fields": {"zzz": 1, "tenant_id": "t2"}'),
        ] + [(label, field) for label, field, _ in _REFUSED_UNKNOWN_KEYS]
        for label, field in cases:
            with self.subTest(label):
                status, reply = self.post(_body(self.SESSION, field))
                self.assertEqual(
                    (status, reply), (200, {"session_id": self.SESSION, "status": "ok"})
                )
                self.assertEqual(_store_files(self.store), before)
        self.assertEqual(self.store.list_sessions(tenant_id="t2"), [])

    def test_create_refuses_the_reserved_unknown_fields_key(self):
        before = _store_files(self.store)
        status, reply = self.post(_body("new0000000000001", '"_unknown_fields": {"zzz": 1}'))
        self.assertEqual((status, reply), (400, {"error": "_unknown_fields is a reserved key"}))
        self.assertEqual(_store_files(self.store), before)

    def test_create_refuses_meta_the_client_cannot_read(self):
        before = _store_files(self.store)
        for label, field, error in _REFUSED_FIELDS + _REFUSED_UNKNOWN_KEYS:
            with self.subTest(label):
                status, reply = self.post(_body("new0000000000001", field))
                self.assertEqual((status, reply), (400, {"error": error}))
                self.assertEqual(_store_files(self.store), before)

    def test_update_refuses_meta_the_client_cannot_read(self):
        before = _store_files(self.store)
        for label, field, error in _REFUSED_FIELDS:
            with self.subTest(label):
                status, reply = self.post(_body(self.SESSION, field))
                self.assertEqual((status, reply), (400, {"error": error}))
                self.assertEqual(_store_files(self.store), before)

    def test_nesting_one_level_under_the_limit_is_stored(self):
        for session_id, value in (
            ("deep-lists000000", _nested_lists(63)),
            ("deep-objects0000", _nested_objects(63)),
        ):
            with self.subTest(session_id):
                status, _ = self.post(_body(session_id, f'"zzz": {value}'))
                self.assertEqual(status, 200)
                meta_file = self.store._session_dir(session_id) / "meta.json"
                self.assertEqual(json.loads(meta_file.read_text())["zzz"], json.loads(value))

    def test_refused_update_does_not_tag_the_session(self):
        before = _store_files(self.store)
        status, reply = self.post(
            _body(self.SESSION, '"tenant_id": "t2"', '"tool_calls": "lots"')
        )
        self.assertEqual(
            (status, reply), (400, {"error": "tool_calls must be a non-negative integer"})
        )
        self.assertEqual(_store_files(self.store), before)
        self.assertEqual(self.store.list_sessions(tenant_id="t2"), [])

    def test_create_refuses_a_time_out_of_range(self):
        before = _store_files(self.store)
        for field in ("started_at", "ended_at"):
            with self.subTest(field):
                status, reply = self.post(
                    _body("new0000000000001", f'"{field}": {_MILLISECONDS}')
                )
                self.assertEqual(
                    (status, reply), (400, {"error": f"{field} must be a Unix time in seconds"})
                )
                self.assertFalse(self.store._session_dir("new0000000000001").exists())
                self.assertEqual(_store_files(self.store), before)

    def test_update_refuses_a_time_out_of_range(self):
        before = _store_files(self.store)
        for field in ("started_at", "ended_at"):
            for tenant_field in ([], ['"tenant_id": "t2"']):
                with self.subTest(field=field, new_tenant=bool(tenant_field)):
                    status, reply = self.post(
                        _body(self.SESSION, *tenant_field, f'"{field}": {_MILLISECONDS}')
                    )
                    self.assertEqual(
                        (status, reply), (400, {"error": f"{field} must be a Unix time in seconds"})
                    )
                    self.assertEqual(_store_files(self.store), before)
        self.assertEqual(self.store.list_sessions(tenant_id="t2"), [])

    def test_usable_times_and_a_null_ended_at_are_stored(self):
        times = ('"started_at": 1780000000.5', '"ended_at": 1780000060.5')
        cases = [
            ("create", "new0000000000001", *times),
            ("create ended_at null", "new0000000000002", '"ended_at": null'),
            ("update", self.SESSION, *times),
            ("update ended_at null", self.SESSION, '"ended_at": null'),
        ]
        for label, session_id, *fields in cases:
            with self.subTest(label):
                status, reply = self.post(_body(session_id, *fields))
                self.assertEqual(
                    (status, reply), (200, {"session_id": session_id, "status": "ok"})
                )
                meta_file = self.store._session_dir(session_id) / "meta.json"
                stored = json.loads(meta_file.read_text())
                # A null ended_at is stored as no ended_at.
                for key, value in json.loads("{" + ", ".join(fields) + "}").items():
                    self.assertEqual(stored.get(key), value)

    def test_numeric_session_id_is_stored_as_text(self):
        status, reply = self.post(b'{"session_id": 12345}')
        self.assertEqual((status, reply), (200, {"session_id": "12345", "status": "ok"}))
        meta_file = self.store._session_dir("12345") / "meta.json"
        self.assertEqual(json.loads(meta_file.read_text())["session_id"], "12345")


class TestPostSessionsKeepsTheCollectorReadable(unittest.TestCase):
    """After each POST /sessions, CollectorClient.list_sessions reads every session."""

    KEY = "ast_deadbeef1234567890abcdef123456"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = TraceStore(tmp.name, use_workspace_env=False)
        server = HTTPServer(
            ("127.0.0.1", 0), _make_handler(store, threading.Lock(), auth_key=self.KEY)
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def test_client_reads_every_session_the_server_accepted(self):
        client = CollectorClient(self.base, self.KEY)
        plain = "plain00000000001"
        refused = _REFUSED_FIELDS + _REFUSED_UNKNOWN_KEYS
        cases = [("plain create", _body(plain, '"started_at": 1780000000.0'))]
        cases += [
            (f"create {label}", _body(f"new{index:013d}", field))
            for index, (label, field, _) in enumerate(refused)
        ]
        cases += [(f"update {label}", _body(plain, field)) for label, field, _ in refused]
        cases += [
            ("update method name", _body(plain, '"to_json": "x"')),
            ("update unknown-field store", _body(plain, '"_unknown_fields": {"zzz": 1}')),
            ("create 63 nested lists", _body("deep-lists000000", f'"zzz": {_nested_lists(63)}')),
            ("create 63 nested objects", _body("deep-objects0000", f'"zzz": {_nested_objects(63)}')),
            ("create _unknown_fields", _body("reserved00000001", '"_unknown_fields": {"zzz": 1}')),
            ("create numeric ID", b'{"session_id": 12345}'),
        ]
        accepted = set()
        for label, body in cases:
            with self.subTest(label):
                status, raw = _post(
                    f"{self.base}/sessions", body,
                    headers={"Authorization": f"Bearer {self.KEY}"},
                )
                if status == 200:
                    accepted.add(json.loads(raw)["session_id"])
                self.assertEqual({meta.session_id for meta in client.list_sessions()}, accepted)
        self.assertEqual(
            accepted, {plain, "deep-lists000000", "deep-objects0000", "12345"}
        )


class TestHookSessionMetaRoundTrip(unittest.TestCase):
    """The metas the session hooks send pass the check and are stored as sent."""

    def test_start_and_end_metas_are_stored_as_sent(self):
        server_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(server_tmp.cleanup)
        local_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(local_tmp.cleanup)
        server_store = TraceStore(server_tmp.name, redact=False, use_workspace_env=False)
        server, port, _ = _start_test_server(server_store)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        sent = []

        def send_and_record(meta, endpoint):
            body = meta.to_json()
            sent.append((body, send_session_meta_to_endpoint(meta, endpoint)))
            return sent[-1][1]

        hook_session = "a1b2c3d4e5f60718293a"
        meta_file = server_store._session_dir(hook_session[:16]) / "meta.json"
        env = {
            "AGENT_TRACE_DIR": local_tmp.name,
            "AGENT_STRACE_ENDPOINT": f"http://127.0.0.1:{port}",
        }
        with mock.patch.dict(os.environ, env), mock.patch(
            "agent_trace.server.send_session_meta_to_endpoint", side_effect=send_and_record
        ):
            for name in (
                "AGENT_STRACE_AUTH_KEY", "AGENT_STRACE_TENANT_ID", "AGENT_STRACE_WORKSPACE",
                "AGENT_TRACE_REDACT", "AGENT_TRACE_NO_REDACT", "AGENT_STRACE_NO_REDACT",
            ):
                os.environ.pop(name, None)
            handle_session_start({"session_id": hook_session, "source": "startup"})
            self.assertEqual(len(sent), 1)
            self.assertTrue(sent[0][1])
            self.assertEqual(json.loads(meta_file.read_text()), json.loads(sent[0][0]))

            handle_session_end({"session_id": hook_session})
            self.assertEqual(len(sent), 2)
            self.assertTrue(sent[1][1])
            stored = json.loads(meta_file.read_text())
            self.assertEqual(stored, json.loads(sent[1][0]))
            self.assertIn("ended_at", stored)


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


class TestGetRefusedSession(unittest.TestCase):
    """GET answers 400 with the refusal for a session the store refuses to read."""

    def setUp(self):
        self.store, _ = _make_store()
        self.server, self.port, _ = _start_test_server(self.store)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_meta_and_events_answer_400_with_one_message(self):
        good = event_line()
        meta_names_other = write_session_lines(self.store, [good])
        name_other_session_in_meta(self.store, meta_names_other)
        event_session = write_session_lines(
            self.store, [good, event_line(session_id=OTHER_SESSION)]
        )
        event_tenant = write_session_lines(
            self.store, [event_line(tenant_id=OTHER_TENANT)], tenant_id="team-a"
        )
        unreadable = write_session_lines(self.store, [good, "{"])

        def refused(session_id, reason):
            return f"cannot read session {session_id}: {reason}"

        corrupt_meta = (
            f"session {meta_names_other} has a corrupt meta.json: "
            "metadata session ID does not match its directory"
        )
        cases = [
            (f"/sessions/{meta_names_other}", corrupt_meta),
            (f"/sessions/{meta_names_other}/events", corrupt_meta),
            (f"/sessions/{event_session}/events",
             refused(event_session, "event session ID does not match its directory")),
            (f"/sessions/{event_tenant}/events",
             refused(event_tenant, "event tenant does not match session metadata")),
            (f"/sessions/{unreadable}/events",
             refused(unreadable, "line 2 of events.ndjson is not valid JSON")),
        ]
        for path, message in cases:
            with self.subTest(path=path):
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    status, body = _get(self.base + path)
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body), {"error": message})
                self.assertNotIn(OTHER_SESSION, body.decode("utf-8"))
                self.assertNotIn(OTHER_TENANT, body.decode("utf-8"))
                self.assertEqual(_without_access_log(stderr.getvalue()), [])


# ---------------------------------------------------------------------------
# GET routes on a session whose meta.json nests too deep
# ---------------------------------------------------------------------------

class TestGetCorruptSessionMeta(unittest.TestCase):
    """A meta.json too deep to read answers 400. Session lists skip it."""

    ERROR = {
        "error": "session bbbb2222 has a corrupt meta.json: metadata nests more than 128 levels"
    }
    REFUSED = [
        "/sessions/bbbb2222",
        "/sessions/bbbb2222/events",
        "/api/sessions/bbbb2222/events",
        "/api/sessions/bbbb2222/review",
        "/api/sessions/bbbb2222/compare?other=aaaa1111",
        "/api/sessions/aaaa1111/compare?other=bbbb2222",
        "/api/sessions/bbbb2222/share?mode=minimized",
        "/api/sessions/bbbb2222/otlp",
    ]

    def serve(self, levels: int) -> str:
        """Serve the dashboard for good session aaaa1111 and bbbb2222 nested ``levels`` deep."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = TraceStore(tmp.name, use_workspace_env=False)
        store.create_session(SessionMeta(session_id="aaaa1111", started_at=1.0))
        store.create_session(SessionMeta(session_id="bbbb2222", started_at=2.0))
        (store.base_dir / "bbbb2222" / "meta.json").write_text(
            '{"session_id": "bbbb2222", "agent_name": ' + _nested_objects(levels) + "}"
        )
        server = HTTPServer(
            ("127.0.0.1", 0), _make_handler(store, threading.Lock(), dashboard=True)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_session_routes_answer_400_and_lists_skip_it(self):
        # json.loads raises RecursionError at 1,000,000 levels on every
        # Python. 3,000 levels parses on 3.12 and later, but copying it for
        # the response ran out of stack and dropped the connection.
        for levels in (1_000_000, 3_000):
            base = self.serve(levels)
            for path in self.REFUSED:
                with self.subTest(levels=levels, path=path):
                    status, body = _get(base + path)
                    self.assertEqual((status, json.loads(body)), (400, self.ERROR))
            for path in ("/sessions", "/api/sessions"):
                with self.subTest(levels=levels, path=path):
                    status, body = _get(base + path)
                    self.assertEqual(status, 200)
                    self.assertEqual(
                        [session["session_id"] for session in json.loads(body)], ["aaaa1111"]
                    )

    def test_a_prefix_of_the_session_answers_400_like_its_full_id(self):
        for levels in (1_000_000, 3_000):
            base = self.serve(levels)
            for path in [path.replace("bbbb2222", "bbbb") for path in self.REFUSED]:
                with self.subTest(levels=levels, path=path):
                    stderr = io.StringIO()
                    with mock.patch("sys.stderr", stderr):
                        status, body = _get(base + path)
                    self.assertEqual((status, json.loads(body)), (400, self.ERROR))
                    self.assertEqual(_without_access_log(stderr.getvalue()), [])
            for path, expected in (
                ("/sessions/aaaa", 200),
                ("/sessions/cccc", 404),
                ("/sessions/cccc/events", 404),
            ):
                with self.subTest(levels=levels, path=path):
                    self.assertEqual(_get(base + path)[0], expected)


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


class TestSendersRefuseRedirects(unittest.TestCase):
    """A followed redirect re-sent the POST as a GET with no body and the key."""

    def setUp(self):
        self.servers = RedirectServers(self)
        environ = mock.patch.dict(os.environ, {"AGENT_STRACE_AUTH_KEY": "collector-key"})
        environ.start()
        self.addCleanup(environ.stop)

    def _check(self, send, message):
        self.assertTrue(send(self.servers.b.url))
        for status in ALL_REDIRECTS:
            with self.subTest(status=status):
                self.servers.redirect(status)
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    self.assertFalse(send(self.servers.url))
                detail = f"HTTP Error {status}: {HTTPStatus(status).phrase}{REDIRECT_NOTE}"
                self.assertEqual(stderr.getvalue(), f"[agent-strace] {message}: {detail}\n")
                self.assertEqual(self.servers.a.methods(), ["POST"])
                self.assertEqual(self.servers.b.received(), [])

    def test_event(self):
        ev = TraceEvent(event_type=EventType.TOOL_CALL, session_id="s1", data={"tool_name": "Bash"})
        self._check(lambda url: send_event_to_endpoint(ev, url), "remote send failed")

    def test_session_meta(self):
        meta = SessionMeta(session_id="s1", agent_name="probe")
        self._check(
            lambda url: send_session_meta_to_endpoint(meta, url),
            "remote session meta send failed",
        )


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


# ---------------------------------------------------------------------------
# Requests from web pages: the Host and Origin rules
# ---------------------------------------------------------------------------

_HOST_REFUSED = json.dumps({
    "error": (
        "Host not allowed: use localhost or an IP address, "
        "or start the server with --allow-host NAME or --auth-key"
    )
}).encode()
_ORIGIN_REFUSED = json.dumps({
    "error": "POST with an Origin header refused: browser requests are not accepted"
}).encode()
_RULE_KEY = "ast_" + "5" * 32
_SEED = "seedsession001"
_MARKER = "marker-7f3a"
_READ_PATHS = (
    "/",
    "/api/sessions",
    "/sessions",
    "/health",
    f"/session/{_SEED}",
    f"/sessions/{_SEED}/events",
    f"/api/sessions/{_SEED}/share?mode=minimized",
    f"/api/sessions/{_SEED}/otlp",
)


def _forged(port, method, path, *, hosts=(), origin=None, key=None, body=None):
    """Send one request with exactly these Host headers, none by default."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for host in hosts:
            conn.putheader("Host", host)
        if origin is not None:
            conn.putheader("Origin", origin)
        if key is not None:
            conn.putheader("Authorization", f"Bearer {key}")
        if body is not None:
            # What a no-cors fetch from a web page can send without a preflight.
            conn.putheader("Content-Type", "text/plain")
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def _session_body(session_id):
    return json.dumps({"session_id": session_id, "agent_name": "probe"}).encode()


def _event_body(session_id):
    event = TraceEvent(
        event_type=EventType.TOOL_CALL,
        session_id=session_id,
        data={"tool_name": "Bash"},
    )
    return event.to_json().encode() + b"\n"


def _seed(store):
    store.create_session(SessionMeta(session_id=_SEED, started_at=1, ended_at=2))
    for event_id, event_type, timestamp in (
        ("start", EventType.SESSION_START, 1),
        ("end", EventType.SESSION_END, 2),
    ):
        store.append_event(_SEED, TraceEvent(
            event_type=event_type,
            timestamp=timestamp,
            session_id=_SEED,
            event_id=event_id,
        ))


class _RuleServerCase(unittest.TestCase):
    """A dashboard server on 127.0.0.1 with one stored session."""

    auth_key = ""
    # Set to make the handler see another bound address without binding it.
    fake_bound_address = ""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name)
        _seed(self.store)
        handler = _make_handler(
            self.store, threading.Lock(), auth_key=self.auth_key, dashboard=True
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        self.port = server.server_address[1]
        if self.fake_bound_address:
            server.server_address = (self.fake_bound_address, self.port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

    def request(self, method, path, **kwargs):
        return _forged(self.port, method, path, **kwargs)

    def stored(self):
        return sorted(meta.session_id for meta in self.store.list_sessions())


class TestHostRuleOnLoopbackWithoutKey(_RuleServerCase):
    def allowed(self):
        port = self.port
        return (
            ["localhost"],
            [f"LOCALHOST:{port}"],
            [f"127.0.0.1:{port}"],
            ["127.0.0.2"],
            [f"[::1]:{port}"],
            [f"192.0.2.10:{port}"],
            [f"localhost:{port + 1}"],
            # Whitespace around a field value is not part of it.
            [f"localhost:{port} \t"],
            [],
        )

    def refused(self):
        port = self.port
        return (
            ["attacker.test"],
            [f"attacker.test:{port}"],
            ["localhost."],
            [f"localhost.:{port}"],
            [f"foo.localhost:{port}"],
            [""],
            ["localhost:abc"],
            ["localhost:"],
            ["[::1"],
            [f"127.0.0.1:{port}", "attacker.test"],
        )

    def test_allowed_hosts_read_every_route(self):
        for hosts in self.allowed():
            for path in _READ_PATHS:
                with self.subTest(hosts=hosts, path=path):
                    status, _ = self.request("GET", path, hosts=hosts)
                    self.assertEqual(status, 200)

    def test_refused_hosts_get_403_on_every_route(self):
        for hosts in self.refused():
            for path in (*_READ_PATHS, "/no-such-route"):
                with self.subTest(hosts=hosts, path=path):
                    reply = self.request("GET", path, hosts=hosts)
                    self.assertEqual(reply, (403, _HOST_REFUSED))

    def test_post_without_origin_is_stored_whatever_the_host(self):
        # Hook senders in a container reach a loopback collector by another
        # name, such as host.docker.internal.
        hosts = (f"host.docker.internal:{self.port}", f"attacker.test:{self.port}")
        for index, host in enumerate(hosts):
            with self.subTest(host=host):
                meta_session = f"meta{index}"
                status, _ = self.request(
                    "POST", "/sessions", hosts=[host], body=_session_body(meta_session)
                )
                self.assertEqual(status, 200)
                events_session = f"events{index}"
                status, _ = self.request(
                    "POST", "/events", hosts=[host], body=_event_body(events_session)
                )
                self.assertEqual(status, 200)
                self.assertTrue(self.store.session_exists(meta_session))
                self.assertTrue(self.store.session_exists(events_session))


class TestHostRuleOffOnNonLoopbackBind(_RuleServerCase):
    fake_bound_address = "0.0.0.0"

    def test_any_host_reads(self):
        for path in _READ_PATHS:
            with self.subTest(path=path):
                status, _ = self.request("GET", path, hosts=["attacker.test"])
                self.assertEqual(status, 200)


class TestHostRuleOffWithKey(_RuleServerCase):
    auth_key = _RULE_KEY

    def test_any_host_reads_with_the_bearer(self):
        for path in _READ_PATHS:
            with self.subTest(path=path):
                status, _ = self.request(
                    "GET", path, hosts=["attacker.test"], key=_RULE_KEY
                )
                self.assertEqual(status, 200)

    def test_any_host_without_the_bearer_gets_401(self):
        status, _ = self.request("GET", "/sessions", hosts=["attacker.test"])
        self.assertEqual(status, 401)


class _OriginRuleTests:
    """Every POST that carries Origin is refused before anything is stored."""

    def bearer(self):
        return self.auth_key or None

    def test_post_with_origin_is_refused_and_stores_nothing(self):
        port = self.port
        cases = (
            ([f"127.0.0.1:{port}"], "https://attacker.test"),
            ([f"127.0.0.1:{port}"], f"http://127.0.0.1:{port}"),
            ([f"attacker.test:{port}"], f"http://attacker.test:{port}"),
            ([f"127.0.0.1:{port}"], "null"),
            ([f"127.0.0.1:{port}"], ""),
        )
        for hosts, origin in cases:
            for path, body in (
                ("/sessions", _session_body("fromweb")),
                ("/events", _event_body("fromweb")),
            ):
                with self.subTest(hosts=hosts, origin=origin, path=path):
                    reply = self.request(
                        "POST", path, hosts=hosts, origin=origin,
                        key=self.bearer(), body=body,
                    )
                    self.assertEqual(reply, (403, _ORIGIN_REFUSED))
                    self.assertEqual(self.stored(), [_SEED])

    def test_post_with_origin_to_an_unknown_path_is_refused(self):
        reply = self.request(
            "POST", "/no-such-route", hosts=[f"127.0.0.1:{self.port}"],
            origin="https://attacker.test", key=self.bearer(), body=b"{}",
        )
        self.assertEqual(reply, (403, _ORIGIN_REFUSED))

    def test_origin_is_refused_before_the_body_is_read(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.putrequest("POST", "/sessions", skip_host=True)
            conn.putheader("Host", f"127.0.0.1:{self.port}")
            conn.putheader("Origin", "https://attacker.test")
            if self.bearer():
                conn.putheader("Authorization", f"Bearer {self.bearer()}")
            conn.putheader("Content-Length", "100000")
            conn.endheaders()
            response = conn.getresponse()
            self.assertEqual((response.status, response.read()), (403, _ORIGIN_REFUSED))
        finally:
            conn.close()

    def test_post_without_origin_is_stored(self):
        for path, body in (
            ("/sessions", _session_body("fromclient")),
            ("/events", _event_body("fromclient")),
        ):
            with self.subTest(path=path):
                status, _ = self.request(
                    "POST", path, hosts=[f"127.0.0.1:{self.port}"],
                    key=self.bearer(), body=body,
                )
                self.assertEqual(status, 200)
        self.assertEqual(self.stored(), ["fromclient", _SEED])


class TestOriginRuleOnLoopbackWithoutKey(_OriginRuleTests, _RuleServerCase):
    pass


class TestOriginRuleOnNonLoopbackBind(_OriginRuleTests, _RuleServerCase):
    fake_bound_address = "0.0.0.0"


class TestOriginRuleWithKey(_OriginRuleTests, _RuleServerCase):
    auth_key = _RULE_KEY

    def test_origin_without_the_bearer_gets_403_not_401(self):
        reply = self.request(
            "POST", "/sessions", hosts=[f"127.0.0.1:{self.port}"],
            origin="https://attacker.test", body=_session_body("fromweb"),
        )
        self.assertEqual(reply, (403, _ORIGIN_REFUSED))
        self.assertEqual(self.stored(), [_SEED])


class TestRefusalsDoNotEchoHeaders(_RuleServerCase):
    def test_marker_is_in_neither_the_reply_nor_stderr(self):
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            host_reply = self.request(
                "GET", "/", hosts=[f"{_MARKER}.attacker.test:{self.port}"]
            )
            origin_reply = self.request(
                "POST", "/sessions", hosts=[f"127.0.0.1:{self.port}"],
                origin=f"https://{_MARKER}.attacker.test",
                body=_session_body("fromweb"),
            )
        self.assertEqual(host_reply, (403, _HOST_REFUSED))
        self.assertEqual(origin_reply, (403, _ORIGIN_REFUSED))
        # The access log still records both requests.
        self.assertIn('"GET / HTTP/1.1" 403', stderr.getvalue())
        self.assertIn('"POST /sessions HTTP/1.1" 403', stderr.getvalue())
        self.assertNotIn(_MARKER, stderr.getvalue())


class TestShippedSendersStillDeliver(_RuleServerCase):
    def test_hook_senders_and_emit(self):
        endpoint = f"http://127.0.0.1:{self.port}"
        self.assertTrue(send_session_meta_to_endpoint(
            SessionMeta(session_id="hookmeta", agent_name="probe"), endpoint
        ))
        self.assertTrue(send_event_to_endpoint(TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id="hookevent",
            data={"tool_name": "Bash"},
        ), endpoint))
        with tempfile.TemporaryDirectory() as local, \
                mock.patch.dict(os.environ, {"AGENT_STRACE_ENDPOINT": endpoint}):
            emit(EventType.TOOL_CALL, "emitted", TraceStore(local), tool_name="Bash")
        self.assertEqual(self.stored(), ["emitted", "hookevent", "hookmeta", _SEED])


class TestAllowHostFlag(unittest.TestCase):
    """--allow-host through the parser, cmd_server and run_server."""

    def _start(self, *argv):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        args = build_parser().parse_args(
            ["server", "--port", "0", "--storage", tmp.name, *argv]
        )
        servers = []
        started = threading.Event()

        class RecordingServer(HTTPServer):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                servers.append(self)
                started.set()

        for patcher in (
            mock.patch("agent_trace.server.HTTPServer", RecordingServer),
            mock.patch.dict(os.environ),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("AGENT_STRACE_AUTH_KEY", None)
        thread = threading.Thread(target=cmd_server, args=(args,), daemon=True)
        thread.start()
        self.assertTrue(started.wait(5))
        server = servers[0]
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        _seed(TraceStore(tmp.name))
        return server.server_address[1]

    def _get(self, port, host, path="/api/sessions"):
        return _forged(port, "GET", path, hosts=[host])

    def test_allowed_names_match_whole_and_ignore_case_and_port(self):
        port = self._start(
            "--dashboard",
            "--allow-host", "dash.example.test",
            "--allow-host", "Second.Example.TEST",
        )
        for host in (
            f"dash.example.test:{port}",
            "DASH.example.test",
            f"second.example.test:{port + 1}",
            f"localhost:{port}",
            f"127.0.0.1:{port}",
        ):
            with self.subTest(host=host):
                self.assertEqual(self._get(port, host)[0], 200)
        for host in (
            "evildash.example.test",
            f"www.dash.example.test:{port}",
            f"dash.example.test.attacker.test:{port}",
            f"attacker.test:{port}",
        ):
            with self.subTest(host=host):
                self.assertEqual(self._get(port, host), (403, _HOST_REFUSED))

    def test_loopback_comes_from_the_bound_socket(self):
        # --host localhost is a name, but the socket is bound to 127.0.0.1.
        port = self._start("--host", "localhost", "--dashboard")
        self.assertEqual(self._get(port, f"localhost:{port}")[0], 200)
        self.assertEqual(
            self._get(port, f"attacker.test:{port}"), (403, _HOST_REFUSED)
        )

    def test_collector_without_dashboard_on_loopback(self):
        port = self._start("--host", "127.0.0.1")
        self.assertEqual(self._get(port, f"localhost:{port}", "/sessions")[0], 200)
        self.assertEqual(
            self._get(port, f"attacker.test:{port}", "/sessions"),
            (403, _HOST_REFUSED),
        )

    def test_values_that_are_not_bare_names_exit_2(self):
        for value in (
            "https://dash.example.test/",
            "dash.example.test:80",
            "",
            "dash.example.test/path",
            "*.example.test",
            "dash..example.test",
            "dash.example.test.",
        ):
            with self.subTest(value=value):
                err = io.StringIO()
                with redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
                    build_parser().parse_args(["server", "--allow-host", value])
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn("argument --allow-host: invalid host name", err.getvalue())


def _refuse_constant(token):
    raise ValueError(f"not JSON: {token}")


def _strict_loads(text):
    """Parse text as JSON that has no bare NaN, Infinity or -Infinity."""
    return json.loads(text, parse_constant=_refuse_constant)


class TestEventsAreStrictJson(unittest.TestCase):
    """The collector stores and serves NaN and infinity as null, as JSON.stringify writes them."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = tmp.name
        self.store = TraceStore(tmp.name, redact=False, use_workspace_env=False)
        server = HTTPServer(
            ("127.0.0.1", 0), _make_handler(self.store, threading.Lock(), dashboard=True)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def served_events(self, session_id):
        """Return the events from both GET routes, parsed strictly."""
        status, body = _get(f"{self.base}/sessions/{session_id}/events")
        self.assertEqual(status, 200)
        ndjson = [_strict_loads(line) for line in body.decode("utf-8").splitlines()]
        status, body = _get(f"{self.base}/api/sessions/{session_id}/events")
        self.assertEqual(status, 200)
        self.assertEqual(_strict_loads(body.decode("utf-8")), ndjson)
        return ndjson

    def stored_lines(self, session_id):
        return (self.store.base_dir / session_id / "events.ndjson").read_text(
            encoding="utf-8"
        ).splitlines()

    def export(self, session_id, fmt):
        args = build_parser().parse_args(
            ["--trace-dir", self.trace_dir, "export", session_id, "--format", fmt]
        )
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            self.assertEqual(cmd_export(args), 0)
        return out.getvalue()

    def test_posted_nan_and_infinity_are_stored_and_served_as_null(self):
        line = (
            '{"event_type":"tool_call","timestamp":1700000000.0,"event_id":"p1",'
            '"session_id":"srvnan01","duration_ms":Infinity,'
            '"data":{"tool_name":"Bash","arguments":{"v":NaN,"w":[1,-Infinity]}}}'
        )
        status, body = _post(f"{self.base}/events", line.encode(), "application/x-ndjson")
        self.assertEqual((status, json.loads(body)), (200, {"accepted": 1, "errors": 0}))

        (stored,) = [_strict_loads(text) for text in self.stored_lines("srvnan01")]
        self.assertNotIn("duration_ms", stored)
        self.assertEqual(stored["data"]["arguments"], {"v": None, "w": [1, None]})
        self.assertEqual(self.served_events("srvnan01"), [stored])

    def test_an_old_session_with_bare_tokens_loads_verifies_and_is_served_strict(self):
        session_id = "oldnan01"
        self.store.create_session(SessionMeta(session_id=session_id, started_at=1.0))
        records = [
            {"event_type": "session_start", "timestamp": 1.0, "event_id": "e1",
             "session_id": session_id, "data": {"cwd": "/w"}},
            {"event_type": "tool_call", "timestamp": 2.0, "event_id": "e2",
             "session_id": session_id, "duration_ms": float("inf"),
             "data": {"tool_name": "Bash", "arguments": {"x": float("nan"), "y": float("-inf")}}},
            {"event_type": "tool_result", "timestamp": 3.0, "event_id": "e3",
             "session_id": session_id, "parent_id": "e2",
             "data": {"tool_name": "Bash", "result": "ok", "score": float("nan")}},
        ]
        lines = []
        for record in records:
            if lines:
                record = {**record, "prev_hash": hashlib.sha256(lines[-1].encode()).hexdigest()}
            # Written as releases before strict JSON wrote it.
            lines.append(json.dumps(record, separators=(",", ":")))
        old_text = "\n".join(lines) + "\n"
        for token in ("NaN", "Infinity", "-Infinity"):
            self.assertIn(token, old_text)
        events_path = self.store.base_dir / session_id / "events.ndjson"
        events_path.write_text(old_text, encoding="utf-8")

        loaded = self.store.load_events(session_id)
        self.assertTrue(math.isnan(loaded[1].data["arguments"]["x"]))
        self.assertTrue(verify_chain(self.store, session_id).ok)

        exported = _strict_loads(self.export(session_id, "json"))
        self.assertEqual(
            [_strict_loads(text) for text in self.export(session_id, "ndjson").splitlines()],
            exported,
        )
        self.assertEqual(exported[1]["data"]["arguments"], {"x": None, "y": None})
        self.assertNotIn("duration_ms", exported[1])
        self.assertEqual(self.served_events(session_id), exported)

        self.store.append_event(session_id, TraceEvent(
            event_type=EventType.SESSION_END, timestamp=4.0, data={"v": float("nan")},
        ))
        self.assertTrue(verify_chain(self.store, session_id).ok)
        text = events_path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(old_text))
        self.assertEqual(_strict_loads(text[len(old_text):])["data"], {"v": None})


if __name__ == "__main__":
    unittest.main()
