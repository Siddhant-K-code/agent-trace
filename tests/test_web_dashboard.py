"""Tests for the web dashboard module and server --dashboard integration."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import HTTPServer

import pytest

from agent_trace.web_dashboard import (
    render_sessions_page,
    render_detail_page,
    render_cost_page,
    render_violations_page,
    render_health_page,
    api_sessions,
    api_session_events,
)
from agent_trace.server import _make_handler, run_server
from agent_trace.store import TraceStore
from agent_trace.models import SessionMeta, TraceEvent, EventType


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

class TestRenderPages:
    def test_sessions_page_contains_nav(self):
        html = render_sessions_page()
        assert "<nav>" in html
        assert 'href="/"' in html
        assert 'href="/cost"' in html
        assert 'href="/violations"' in html
        assert 'href="/health"' in html

    def test_sessions_page_active_nav(self):
        html = render_sessions_page()
        # Sessions link should be active
        assert 'class="active"' in html

    def test_detail_page_contains_session_id(self):
        html = render_detail_page("abc123def456")
        assert "abc123def4" in html  # first 10 chars appear in title

    def test_cost_page_title(self):
        html = render_cost_page()
        assert "Cost" in html
        assert "By Team" in html

    def test_violations_page_title(self):
        html = render_violations_page()
        assert "Violations" in html

    def test_health_page_title(self):
        html = render_health_page()
        assert "Health" in html

    def test_all_pages_are_valid_html(self):
        pages = [
            render_sessions_page(),
            render_detail_page("test123"),
            render_cost_page(),
            render_violations_page(),
            render_health_page(),
        ]
        for html in pages:
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html
            assert "<script>" in html


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

class TestApiHelpers:
    def test_api_sessions_empty(self, tmp_path):
        store = TraceStore(str(tmp_path))
        result = api_sessions(store)
        data = json.loads(result)
        assert data == []

    def test_api_sessions_returns_list(self, tmp_path):
        store = TraceStore(str(tmp_path))
        meta = SessionMeta(session_id="sess001", agent_name="test-agent")
        store.create_session(meta)
        result = api_sessions(store)
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["session_id"] == "sess001"
        assert data[0]["agent_name"] == "test-agent"

    def test_api_session_events_not_found(self, tmp_path):
        store = TraceStore(str(tmp_path))
        result = api_session_events(store, "nonexistent")
        assert result is None

    def test_api_session_events_returns_events(self, tmp_path):
        store = TraceStore(str(tmp_path))
        meta = SessionMeta(session_id="sess002")
        store.create_session(meta)
        event = TraceEvent(event_type=EventType.TOOL_CALL, session_id="sess002",
                           data={"tool_name": "bash"})
        store.append_event("sess002", event)
        result = api_session_events(store, "sess002")
        assert result is not None
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["event_type"] == "tool_call"

    def test_api_session_events_prefix_lookup(self, tmp_path):
        store = TraceStore(str(tmp_path))
        meta = SessionMeta(session_id="abcdef123456")
        store.create_session(meta)
        event = TraceEvent(event_type=EventType.SESSION_START, session_id="abcdef123456")
        store.append_event("abcdef123456", event)
        # prefix lookup
        result = api_session_events(store, "abcdef")
        assert result is not None
        data = json.loads(result)
        assert len(data) == 1


# ---------------------------------------------------------------------------
# Live server integration
# ---------------------------------------------------------------------------

def _start_test_server(tmp_path, dashboard=True):
    """Start a test server on a random port, return (server, port, thread)."""
    store = TraceStore(str(tmp_path))
    lock = threading.Lock()
    handler = _make_handler(store, lock, auth_key="", dashboard=dashboard)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port, store


class TestServerDashboardRoutes:
    def test_root_returns_html(self, tmp_path):
        server, port, _ = _start_test_server(tmp_path)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            assert r.status == 200
            ct = r.headers.get("Content-Type", "")
            assert "text/html" in ct
            body = r.read().decode()
            assert "<!DOCTYPE html>" in body
        finally:
            server.shutdown()

    def test_cost_route(self, tmp_path):
        server, port, _ = _start_test_server(tmp_path)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/cost")
            assert r.status == 200
            body = r.read().decode()
            assert "Cost" in body
        finally:
            server.shutdown()

    def test_violations_route(self, tmp_path):
        server, port, _ = _start_test_server(tmp_path)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/violations")
            assert r.status == 200
        finally:
            server.shutdown()

    def test_health_route_returns_html_when_dashboard(self, tmp_path):
        server, port, _ = _start_test_server(tmp_path)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
            assert r.status == 200
            ct = r.headers.get("Content-Type", "")
            assert "text/html" in ct
        finally:
            server.shutdown()

    def test_session_detail_route(self, tmp_path):
        server, port, _ = _start_test_server(tmp_path)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/session/abc123")
            assert r.status == 200
            body = r.read().decode()
            assert "<!DOCTYPE html>" in body
        finally:
            server.shutdown()

    def test_api_sessions_route(self, tmp_path):
        server, port, store = _start_test_server(tmp_path)
        try:
            meta = SessionMeta(session_id="live001", agent_name="live-agent")
            store.create_session(meta)
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sessions")
            assert r.status == 200
            data = json.loads(r.read())
            assert any(s["session_id"] == "live001" for s in data)
        finally:
            server.shutdown()

    def test_api_session_events_route(self, tmp_path):
        server, port, store = _start_test_server(tmp_path)
        try:
            meta = SessionMeta(session_id="live002")
            store.create_session(meta)
            ev = TraceEvent(event_type=EventType.TOOL_CALL, session_id="live002",
                            data={"tool_name": "read_file"})
            store.append_event("live002", ev)
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sessions/live002/events")
            assert r.status == 200
            data = json.loads(r.read())
            assert len(data) == 1
            assert data[0]["event_type"] == "tool_call"
        finally:
            server.shutdown()

    def test_api_session_events_404(self, tmp_path):
        server, port, _ = _start_test_server(tmp_path)
        try:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sessions/nosuchsession/events")
                assert False, "expected 404"
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            server.shutdown()

    def test_dashboard_disabled_root_falls_through(self, tmp_path):
        """When --dashboard is not set, / returns 404 (not a dashboard page)."""
        server, port, _ = _start_test_server(tmp_path, dashboard=False)
        try:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/")
                assert False, "expected 404"
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            server.shutdown()

    def test_health_json_when_dashboard_disabled(self, tmp_path):
        """When --dashboard is not set, /health returns JSON (original behaviour)."""
        server, port, _ = _start_test_server(tmp_path, dashboard=False)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
            assert r.status == 200
            ct = r.headers.get("Content-Type", "")
            assert "application/json" in ct
        finally:
            server.shutdown()
