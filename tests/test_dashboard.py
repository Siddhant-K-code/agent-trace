"""Tests for multi-session dashboard (issue #23)."""

import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.dashboard import (
    DashboardReport,
    SessionSummary,
    build_dashboard,
    format_dashboard,
    render_html_dashboard,
)
from agent_trace.models import SessionMeta
from agent_trace.store import TraceStore


def _make_store_with_sessions(n: int) -> TraceStore:
    tmpdir = tempfile.mkdtemp()
    store = TraceStore(tmpdir)
    for i in range(n):
        meta = SessionMeta(
            agent_name="test-agent",
            tool_calls=i * 3,
            llm_requests=i + 1,
            errors=1 if i % 3 == 0 else 0,
            total_tokens=i * 1000,
            total_duration_ms=i * 5000,
        )
        store.create_session(meta)
    return store


class TestBuildDashboard(unittest.TestCase):
    def test_empty_store(self):
        tmpdir = tempfile.mkdtemp()
        store = TraceStore(tmpdir)
        report = build_dashboard(store)
        self.assertEqual(len(report.summaries), 0)
        self.assertEqual(report.total_cost, 0.0)

    def test_builds_from_sessions(self):
        store = _make_store_with_sessions(5)
        report = build_dashboard(store)
        self.assertEqual(len(report.summaries), 5)
        self.assertGreaterEqual(report.total_tool_calls, 0)

    def test_limit_respected(self):
        store = _make_store_with_sessions(10)
        report = build_dashboard(store, limit=3)
        self.assertLessEqual(len(report.summaries), 3)

    def test_agent_filter(self):
        store = _make_store_with_sessions(5)
        report = build_dashboard(store, agent_filter="test-agent")
        self.assertEqual(len(report.summaries), 5)

    def test_agent_filter_no_match(self):
        store = _make_store_with_sessions(5)
        report = build_dashboard(store, agent_filter="nonexistent")
        self.assertEqual(len(report.summaries), 0)

    def test_success_rate(self):
        store = _make_store_with_sessions(3)
        report = build_dashboard(store)
        self.assertGreaterEqual(report.success_rate, 0.0)
        self.assertLessEqual(report.success_rate, 1.0)


class TestFormatDashboard(unittest.TestCase):
    def test_format_no_crash(self):
        store = _make_store_with_sessions(3)
        report = build_dashboard(store)
        buf = io.StringIO()
        format_dashboard(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Dashboard", output)
        self.assertIn("sessions", output)

    def test_format_empty(self):
        report = DashboardReport(
            summaries=[], total_cost=0, total_tokens=0,
            total_tool_calls=0, total_errors=0,
            avg_duration_s=0, success_rate=0,
        )
        buf = io.StringIO()
        format_dashboard(report, out=buf)
        self.assertIn("0 session", buf.getvalue())


class TestRenderHtmlDashboard(unittest.TestCase):
    def test_renders_html(self):
        store = _make_store_with_sessions(3)
        report = build_dashboard(store)
        html = render_html_dashboard(report)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("dashboard", html)
        self.assertIn("Sessions", html)

    def test_html_contains_session_rows(self):
        store = _make_store_with_sessions(2)
        report = build_dashboard(store)
        html = render_html_dashboard(report)
        self.assertIn("<tr", html)


if __name__ == "__main__":
    unittest.main()
