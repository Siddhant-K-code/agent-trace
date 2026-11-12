"""Tests for multi-session dashboard (issue #23)."""

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cost import estimate_cost
from agent_trace.dashboard import (
    DashboardReport,
    SessionSummary,
    build_dashboard,
    format_dashboard,
    render_html_dashboard,
)
from agent_trace.models import SessionMeta
from agent_trace.store import TraceStore

try:
    from test_cost import _codex_events
except ImportError:  # pragma: no cover - package-style test discovery
    from tests.test_cost import _codex_events


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


def _codex_session(store: TraceStore, prompt_data=None, started_at: float = 1000.0) -> str:
    """A session whose model requests record usage with cache reads.

    meta.total_tokens is 1M, which the old flat-rate estimate priced at $3.
    """
    meta = SessionMeta(
        agent_name="codex", started_at=started_at, tool_calls=1, llm_requests=3,
        errors=0, total_tokens=1_000_000, total_duration_ms=9000,
    )
    store.create_session(meta)
    events = _codex_events(meta.session_id)
    if prompt_data is not None:
        events[0].data = prompt_data
    for event in events:
        store.append_event(meta.session_id, event)
    return meta.session_id


class TestDashboardCost(unittest.TestCase):
    """Session costs are the `agent-strace cost` totals."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name)

    def _render(self, report):
        out = io.StringIO()
        format_dashboard(report, out=out)
        return out.getvalue(), render_html_dashboard(report)

    def test_cost_matches_cost_command(self):
        sid = _codex_session(self.store)
        want = estimate_cost(self.store, sid).total_cost
        [summary] = build_dashboard(self.store).summaries
        self.assertEqual(summary.estimated_cost, want)
        self.assertNotAlmostEqual(want, 3.0)

    def test_unpriceable_session_is_not_priced_at_zero(self):
        priced = _codex_session(self.store, started_at=1000.0)
        _codex_session(self.store, prompt_data=["list the files"], started_at=2000.0)
        report = build_dashboard(self.store)
        unpriced, ok = report.summaries
        self.assertIsNone(unpriced.estimated_cost)
        self.assertEqual(report.unpriced_sessions, 1)
        self.assertEqual(report.total_cost, estimate_cost(self.store, priced).total_cost)

        text, page = self._render(report)
        self.assertIn(f"Total cost:    ~${report.total_cost:.4f} (1 not priced)\n", text)
        self.assertIn(f"     n/a  ", text)
        self.assertIn(f"${ok.estimated_cost:>7.4f}", text)
        self.assertIn("<td>n/a</td>", page)
        self.assertIn(f"${report.total_cost:.4f} (1 not priced)</div>", page)
        self.assertNotIn("$0.0000", text + page)

    def test_unreadable_events_are_not_priced(self):
        sid = _codex_session(self.store)
        with open(self.store.base_dir / sid / "events.ndjson", "a") as f:
            f.write("not json\n")
        [summary] = build_dashboard(self.store).summaries
        self.assertIsNone(summary.estimated_cost)

    def test_no_priced_session_renders(self):
        for i in range(3):
            _codex_session(self.store, prompt_data=["x"], started_at=1000.0 + i)
        report = build_dashboard(self.store)
        self.assertEqual(report.total_cost, 0.0)
        self.assertEqual(report.unpriced_sessions, 3)
        text, page = self._render(report)
        self.assertIn("(3 not priced)", text)
        self.assertNotIn("Cost trend", text)
        self.assertIn('<polyline points=""/>', page)

    def test_pricing_bug_is_not_hidden(self):
        _codex_session(self.store)
        with patch("agent_trace.dashboard.session_cost", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                build_dashboard(self.store)

    def test_other_columns_still_come_from_meta(self):
        sid = _codex_session(self.store)
        meta = self.store.load_meta(sid)
        report = build_dashboard(self.store)
        [summary] = report.summaries
        self.assertEqual(
            (summary.tool_calls, summary.llm_requests, summary.errors,
             summary.total_tokens, summary.duration_s, summary.succeeded),
            (meta.tool_calls, meta.llm_requests, meta.errors,
             meta.total_tokens, meta.total_duration_ms / 1000, True),
        )
        text, page = self._render(report)
        self.assertNotIn("not priced", text + page)
        self.assertNotIn("n/a", text + page)
        self.assertIn("  Total tokens:  1,000,000\n", text)


if __name__ == "__main__":
    unittest.main()
