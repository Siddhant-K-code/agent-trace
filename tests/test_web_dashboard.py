"""Tests for the web dashboard module and server --dashboard integration."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer

sys.path.insert(0, "src")

from agent_trace.web_dashboard import (
    render_sessions_page,
    render_detail_page,
    render_cost_page,
    render_violations_page,
    render_health_page,
    api_sessions,
    api_session_events,
    api_session_review,
    api_session_compare,
    api_minimized_review_share,
    api_session_otlp,
)
from agent_trace.server import _make_handler
from agent_trace.store import TraceStore
from agent_trace.models import SessionMeta, TraceEvent, EventType
from agent_trace.event_data import NonObjectEventDataError


def _start_server(tmp_dir, dashboard=True):
    store = TraceStore(tmp_dir)
    lock = threading.Lock()
    handler = _make_handler(store, lock, auth_key="", dashboard=dashboard)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port, store


def _append_raw_event(store, session_id, data):
    line = json.dumps({
        "event_type": "tool_call",
        "timestamp": 2,
        "event_id": "bad-data",
        "session_id": session_id,
        "data": data,
    })
    path = store.base_dir / session_id / "events.ndjson"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


class TestRenderPages(unittest.TestCase):
    def test_sessions_page_contains_nav(self):
        html = render_sessions_page()
        self.assertIn("<nav>", html)
        self.assertIn('href="/"', html)
        self.assertIn('href="/cost"', html)
        self.assertIn('href="/violations"', html)
        self.assertIn('href="/health"', html)

    def test_sessions_page_active_nav(self):
        self.assertIn('class="active"', render_sessions_page())

    def test_detail_page_contains_session_id(self):
        html = render_detail_page("abc123def456")
        self.assertIn("abc123def4", html)
        self.assertIn("Local evidence review", html)
        self.assertIn("Save minimized share", html)
        self.assertIn("Download OTLP JSON", html)
        self.assertIn("Compare sessions", html)
        self.assertIn("J/K or arrows move", html)

    def test_cost_page_title(self):
        html = render_cost_page()
        self.assertIn("Cost", html)
        self.assertIn("By Team", html)

    def test_violations_page_title(self):
        self.assertIn("Violations", render_violations_page())

    def test_health_page_title(self):
        self.assertIn("Health", render_health_page())

    def test_all_pages_are_valid_html(self):
        for html in [render_sessions_page(), render_detail_page("t"),
                     render_cost_page(), render_violations_page(), render_health_page()]:
            self.assertTrue(html.startswith("<!DOCTYPE html>"))
            self.assertIn("</html>", html)
            self.assertIn("<script>", html)


class TestApiHelpers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_api_sessions_empty(self):
        self.assertEqual(json.loads(api_sessions(TraceStore(self._tmp))), [])

    def test_api_sessions_returns_list(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(session_id="sess001", agent_name="agent"))
        data = json.loads(api_sessions(store))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["session_id"], "sess001")

    def test_api_session_events_not_found(self):
        self.assertIsNone(api_session_events(TraceStore(self._tmp), "nope"))

    def test_api_session_events_returns_events(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(session_id="s2"))
        store.append_event("s2", TraceEvent(event_type=EventType.TOOL_CALL,
                                            session_id="s2", data={"tool_name": "bash"}))
        data = json.loads(api_session_events(store, "s2"))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["event_type"], "tool_call")

    def test_api_session_events_prefix_lookup(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(session_id="abcdef123456"))
        store.append_event("abcdef123456",
                           TraceEvent(event_type=EventType.SESSION_START,
                                      session_id="abcdef123456"))
        data = json.loads(api_session_events(store, "abcdef"))
        self.assertEqual(len(data), 1)

    def test_api_session_review_returns_projected_evidence(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(
            session_id="review001",
            started_at=1,
            ended_at=3,
            agent_name="review-agent",
        ))
        store.append_event("review001", TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=1,
            session_id="review001",
            event_id="start",
            data={"provider": "synthetic", "capture_method": "fixture"},
        ))
        store.append_event("review001", TraceEvent(
            event_type=EventType.TOOL_RESULT,
            timestamp=2,
            session_id="review001",
            event_id="orphan",
            parent_id="missing",
            data={"output": "private output"},
        ))
        store.append_event("review001", TraceEvent(
            event_type=EventType.SESSION_END,
            timestamp=3,
            session_id="review001",
            event_id="end",
        ))

        payload = json.loads(api_session_review(store, "review001"))

        self.assertEqual(payload["session"]["provider"], "synthetic")
        self.assertEqual(payload["items"][1]["relationship"], "orphaned")
        self.assertIn("gaps", payload["items"][1]["categories"])

    def test_minimized_share_and_otlp_are_local_artifacts(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(
            session_id="artifact001",
            started_at=1,
            ended_at=2,
        ))
        store.append_event("artifact001", TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=1,
            session_id="artifact001",
            event_id="start",
            data={"provider": "synthetic", "secret": "do-not-share"},
        ))
        store.append_event("artifact001", TraceEvent(
            event_type=EventType.SESSION_END,
            timestamp=2,
            session_id="artifact001",
            event_id="end",
        ))

        share = api_minimized_review_share(store, "artifact001")
        self.assertIn("Privacy-minimized evidence review", share)
        self.assertNotIn("do-not-share", share)
        self.assertNotIn("synthetic", share)
        otlp = json.loads(api_session_otlp(store, "artifact001"))
        self.assertIn("resourceSpans", otlp)

    def test_api_session_compare_reuses_existing_report(self):
        store = TraceStore(self._tmp)
        for session_id, ended_at in (("compare-a", 2), ("compare-b", 3)):
            store.create_session(SessionMeta(
                session_id=session_id,
                started_at=1,
                ended_at=ended_at,
            ))
            store.append_event(session_id, TraceEvent(
                event_type=EventType.SESSION_START,
                timestamp=1,
                session_id=session_id,
                event_id=f"{session_id}-start",
            ))
            store.append_event(session_id, TraceEvent(
                event_type=EventType.SESSION_END,
                timestamp=ended_at,
                session_id=session_id,
                event_id=f"{session_id}-end",
            ))

        report = json.loads(api_session_compare(store, "compare-a", "compare-b"))

        self.assertEqual(report["session_a"], "compare-a")
        self.assertEqual(report["session_b"], "compare-b")
        self.assertIn("decision_divergence", report)

    def test_review_helpers_reject_non_object_event_data(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(session_id="invalid-review", started_at=1))
        _append_raw_event(store, "invalid-review", ["not", "an", "object"])

        for operation in (api_session_review, api_minimized_review_share):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(
                    NonObjectEventDataError,
                    "session invalid-review event data must be an object",
                ):
                    operation(store, "invalid-review")


class TestServerDashboardRoutes(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_root_returns_html(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            self.assertEqual(r.status, 200)
            self.assertIn("text/html", r.headers.get("Content-Type", ""))
            self.assertIn("<!DOCTYPE html>", r.read().decode())
        finally:
            server.shutdown()
            server.server_close()

    def test_cost_route(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/cost")
            self.assertEqual(r.status, 200)
            self.assertIn("Cost", r.read().decode())
        finally:
            server.shutdown()
            server.server_close()

    def test_violations_route(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/violations")
            self.assertEqual(r.status, 200)
        finally:
            server.shutdown()
            server.server_close()

    def test_health_route_returns_html_when_dashboard(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
            self.assertEqual(r.status, 200)
            self.assertIn("text/html", r.headers.get("Content-Type", ""))
        finally:
            server.shutdown()
            server.server_close()

    def test_session_detail_route(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/session/abc123")
            self.assertEqual(r.status, 200)
            self.assertIn("<!DOCTYPE html>", r.read().decode())
        finally:
            server.shutdown()
            server.server_close()

    def test_api_sessions_route(self):
        server, port, store = _start_server(self._tmp)
        try:
            store.create_session(SessionMeta(session_id="live001", agent_name="a"))
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sessions")
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertTrue(any(s["session_id"] == "live001" for s in data))
        finally:
            server.shutdown()
            server.server_close()

    def test_api_session_events_route(self):
        server, port, store = _start_server(self._tmp)
        try:
            store.create_session(SessionMeta(session_id="live002"))
            store.append_event("live002",
                               TraceEvent(event_type=EventType.TOOL_CALL,
                                          session_id="live002",
                                          data={"tool_name": "read_file"}))
            r = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/sessions/live002/events")
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["event_type"], "tool_call")
        finally:
            server.shutdown()
            server.server_close()

    def test_api_session_events_404(self):
        server, port, _ = _start_server(self._tmp)
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/sessions/nosuch/events")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()

    def test_review_download_routes(self):
        server, port, store = _start_server(self._tmp)
        try:
            store.create_session(SessionMeta(
                session_id="downloads001",
                started_at=1,
                ended_at=2,
            ))
            store.append_event("downloads001", TraceEvent(
                event_type=EventType.SESSION_START,
                timestamp=1,
                session_id="downloads001",
                event_id="start",
            ))
            store.append_event("downloads001", TraceEvent(
                event_type=EventType.SESSION_END,
                timestamp=2,
                session_id="downloads001",
                event_id="end",
            ))

            review = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/sessions/downloads001/review"
            )
            self.assertEqual(review.status, 200)
            self.assertEqual(json.loads(review.read())["session"]["session_id"], "downloads001")

            share = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/sessions/downloads001/share?mode=minimized"
            )
            self.assertIn("attachment", share.headers.get("Content-Disposition", ""))
            self.assertIn("Privacy-minimized evidence review", share.read().decode())

            otlp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/sessions/downloads001/otlp"
            )
            self.assertIn("attachment", otlp.headers.get("Content-Disposition", ""))
            self.assertIn("resourceSpans", json.loads(otlp.read()))
        finally:
            server.shutdown()
            server.server_close()

    def test_review_routes_return_400_for_non_object_event_data(self):
        server, port, store = _start_server(self._tmp)
        try:
            store.create_session(SessionMeta(
                session_id="invalid-review-route",
                started_at=1,
            ))
            _append_raw_event(
                store,
                "invalid-review-route",
                "not-an-object",
            )
            for suffix in ("review", "share?mode=minimized"):
                with self.subTest(suffix=suffix):
                    with self.assertRaises(urllib.error.HTTPError) as context:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/api/sessions/"
                            f"invalid-review-route/{suffix}"
                        )
                    self.assertEqual(context.exception.code, 400)
                    payload = json.loads(context.exception.read())
                    self.assertEqual(
                        payload["error"],
                        "session invalid-review-route event data must be an object: "
                        "event 1 of 1 (tool_call) has string data",
                    )
        finally:
            server.shutdown()
            server.server_close()

    def test_dashboard_disabled_root_falls_through(self):
        server, port, _ = _start_server(self._tmp, dashboard=False)
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()

    def test_health_json_when_dashboard_disabled(self):
        server, port, _ = _start_server(self._tmp, dashboard=False)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
            self.assertEqual(r.status, 200)
            self.assertIn("application/json", r.headers.get("Content-Type", ""))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
