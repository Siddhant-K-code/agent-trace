"""Tests for the server-side event collector (issue #101)."""

import io
import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from contextlib import redirect_stderr
from unittest import mock
from urllib.parse import quote
from pathlib import Path

from agent_trace.audit import verify_chain
from agent_trace.cli import build_parser
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
except ImportError:
    from tests._redirect_servers import ALL_REDIRECTS, REDIRECT_NOTE, RedirectServers


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


if __name__ == "__main__":
    unittest.main()
