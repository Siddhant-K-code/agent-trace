"""Tests for agent-strace budget-report (Issue #115)."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.budget_report import (
    BudgetReport,
    SessionSpend,
    build_report,
    format_report_json,
    format_report_markdown,
    format_report_text,
    cmd_budget_report,
    _parse_date,
    _read_postmortem,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> tuple[TraceStore, str]:
    tmp = tempfile.mkdtemp()
    return TraceStore(Path(tmp)), tmp


def _add_session(store: TraceStore, started_at: float, agent_name: str = "test",
                 tool_calls: list[str] | None = None,
                 with_postmortem: bool = False,
                 budget: float | None = None) -> str:
    meta = SessionMeta(agent_name=agent_name, command="test")
    meta.started_at = started_at
    sp = store.create_session(meta)
    sid = sp.name
    # Fix the started_at in meta.json (create_session uses default_factory)
    meta2 = store.load_meta(sid)
    meta2.started_at = started_at
    store.update_meta(meta2)

    # Add some events
    events = [
        TraceEvent(event_type=EventType.SESSION_START, timestamp=started_at,
                   session_id=sid, data={}),
    ]
    for tool in (tool_calls or ["Read", "Write"]):
        events.append(TraceEvent(event_type=EventType.TOOL_CALL, timestamp=started_at + 1,
                                 session_id=sid, data={"tool_name": tool}))
    events.append(TraceEvent(event_type=EventType.SESSION_END, timestamp=started_at + 60,
                             session_id=sid, data={}))
    for e in events:
        store.append_event(sid, e)

    if with_postmortem:
        pm = {
            "session_id": sid,
            "reason": "budget_exceeded",
            "elapsed_seconds": 60.0,
            "cost_at_death": 4.50,
            "max_cost_dollars": budget or 5.0,
        }
        (store._session_dir(sid) / "watchdog-postmortem.json").write_text(json.dumps(pm))

    return sid


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------

class TestParseDate(unittest.TestCase):
    def test_iso_date(self):
        ts = _parse_date("2026-01-01")
        self.assertIsInstance(ts, float)
        self.assertGreater(ts, 0)

    def test_days_duration(self):
        before = time.time()
        ts = _parse_date("7d")
        after = time.time()
        self.assertAlmostEqual(ts, before - 7 * 86400, delta=5)

    def test_hours_duration(self):
        ts = _parse_date("24h")
        self.assertAlmostEqual(ts, time.time() - 24 * 3600, delta=5)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _parse_date("not-a-date")


# ---------------------------------------------------------------------------
# _read_postmortem
# ---------------------------------------------------------------------------

class TestReadPostmortem(unittest.TestCase):
    def test_returns_none_when_absent(self):
        store, _ = _make_store()
        meta = SessionMeta(agent_name="t", command="t")
        sp = store.create_session(meta)
        sid = sp.name
        self.assertIsNone(_read_postmortem(store, sid))

    def test_returns_dict_when_present(self):
        store, _ = _make_store()
        meta = SessionMeta(agent_name="t", command="t")
        sp = store.create_session(meta)
        sid = sp.name
        pm = {"reason": "budget_exceeded", "cost_at_death": 3.0}
        (store._session_dir(sid) / "watchdog-postmortem.json").write_text(json.dumps(pm))
        result = _read_postmortem(store, sid)
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "budget_exceeded")

    def test_returns_none_on_corrupt_json(self):
        store, _ = _make_store()
        meta = SessionMeta(agent_name="t", command="t")
        sp = store.create_session(meta)
        sid = sp.name
        (store._session_dir(sid) / "watchdog-postmortem.json").write_text("not json{{{")
        self.assertIsNone(_read_postmortem(store, sid))


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

class TestBuildReport(unittest.TestCase):
    def test_empty_store(self):
        store, _ = _make_store()
        now = time.time()
        report = build_report(store, now - 7 * 86400, now)
        self.assertEqual(report.session_count, 0)
        self.assertEqual(report.total_cost, 0.0)

    def test_sessions_in_window_included(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3 * 86400)  # 3 days ago — in window
        _add_session(store, now - 10 * 86400)  # 10 days ago — outside window
        report = build_report(store, now - 7 * 86400, now)
        self.assertEqual(report.session_count, 1)

    def test_sessions_outside_window_excluded(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 10 * 86400)
        report = build_report(store, now - 7 * 86400, now)
        self.assertEqual(report.session_count, 0)

    def test_prior_window_populated(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3 * 86400)   # current window
        _add_session(store, now - 10 * 86400)  # prior window (7–14 days ago)
        report = build_report(store, now - 7 * 86400, now, include_prior=True)
        self.assertEqual(report.session_count, 1)
        self.assertEqual(report.prior_session_count, 1)

    def test_watchdog_terminated_detected(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 1 * 86400, with_postmortem=True, budget=5.0)
        report = build_report(store, now - 7 * 86400, now)
        self.assertEqual(len(report.watchdog_terminated_sessions), 1)

    def test_watchdog_savings_calculated(self):
        store, _ = _make_store()
        now = time.time()
        # Post-mortem says cost_at_death=4.50, budget=5.0 → savings=0.50
        _add_session(store, now - 1 * 86400, with_postmortem=True, budget=5.0)
        report = build_report(store, now - 7 * 86400, now)
        # Savings = budget - actual_cost (actual cost from estimate_cost, not pm)
        # Just verify it's non-negative
        self.assertGreaterEqual(report.watchdog_savings, 0.0)

    def test_tool_totals_aggregated(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 1 * 86400, tool_calls=["Bash", "Bash", "Read"])
        report = build_report(store, now - 7 * 86400, now)
        # Tool totals should have Bash and Read
        totals = report.tool_totals
        self.assertIn("Bash", totals)
        self.assertIn("Read", totals)

    def test_top_sessions_sorted_by_cost(self):
        store, _ = _make_store()
        now = time.time()
        for i in range(6):
            _add_session(store, now - i * 3600)
        report = build_report(store, now - 7 * 86400, now)
        top = report.top_sessions
        self.assertLessEqual(len(top), 5)
        # Verify descending order
        for i in range(len(top) - 1):
            self.assertGreaterEqual(top[i].cost, top[i + 1].cost)

    def test_multiple_sessions_cost_summed(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 1 * 86400)
        _add_session(store, now - 2 * 86400)
        _add_session(store, now - 3 * 86400)
        report = build_report(store, now - 7 * 86400, now)
        self.assertEqual(report.session_count, 3)
        self.assertAlmostEqual(report.total_cost, sum(s.cost for s in report.sessions))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatReportText(unittest.TestCase):
    def _make_report(self, n_sessions=2, with_watchdog=False):
        store, _ = _make_store()
        now = time.time()
        for i in range(n_sessions):
            # Use (i+1)*3600 so all sessions are strictly before window_end=now
            _add_session(store, now - (i + 1) * 3600,
                         with_postmortem=(with_watchdog and i == 0), budget=5.0)
        return build_report(store, now - 7 * 86400, now)

    def test_empty_report(self):
        import io
        store, _ = _make_store()
        now = time.time()
        report = build_report(store, now - 7 * 86400, now)
        out = io.StringIO()
        format_report_text(report, out)
        self.assertIn("No sessions", out.getvalue())

    def test_contains_total_spend(self):
        import io
        report = self._make_report(2)
        out = io.StringIO()
        format_report_text(report, out)
        self.assertIn("Total spend", out.getvalue())

    def test_contains_session_count(self):
        import io
        report = self._make_report(3)
        out = io.StringIO()
        format_report_text(report, out)
        self.assertIn("Sessions:", out.getvalue())

    def test_watchdog_line_shown(self):
        import io
        report = self._make_report(2, with_watchdog=True)
        out = io.StringIO()
        format_report_text(report, out)
        self.assertIn("watchdog", out.getvalue())

    def test_week_over_week_delta_shown(self):
        import io
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3 * 86400)   # current
        _add_session(store, now - 10 * 86400)  # prior
        report = build_report(store, now - 7 * 86400, now)
        out = io.StringIO()
        format_report_text(report, out)
        text = out.getvalue()
        # Delta indicator should appear
        self.assertTrue("↑" in text or "↓" in text or "≈" in text)


class TestFormatReportMarkdown(unittest.TestCase):
    def test_markdown_headers(self):
        import io
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 1 * 86400)
        report = build_report(store, now - 7 * 86400, now)
        out = io.StringIO()
        format_report_markdown(report, out)
        text = out.getvalue()
        self.assertIn("## Budget Report", text)
        self.assertIn("### Top", text)

    def test_markdown_table_format(self):
        import io
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 1 * 86400)
        report = build_report(store, now - 7 * 86400, now)
        out = io.StringIO()
        format_report_markdown(report, out)
        text = out.getvalue()
        self.assertIn("|", text)  # table rows

    def test_markdown_empty(self):
        import io
        store, _ = _make_store()
        now = time.time()
        report = build_report(store, now - 7 * 86400, now)
        out = io.StringIO()
        format_report_markdown(report, out)
        self.assertIn("No sessions", out.getvalue())


class TestFormatReportJson(unittest.TestCase):
    def test_json_structure(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 1 * 86400)
        report = build_report(store, now - 7 * 86400, now)
        data = json.loads(format_report_json(report))
        self.assertIn("total_cost", data)
        self.assertIn("session_count", data)
        self.assertIn("top_sessions", data)
        self.assertIn("tool_totals", data)
        self.assertIn("watchdog_savings", data)

    def test_json_empty_report(self):
        store, _ = _make_store()
        now = time.time()
        report = build_report(store, now - 7 * 86400, now)
        data = json.loads(format_report_json(report))
        self.assertEqual(data["session_count"], 0)
        self.assertEqual(data["total_cost"], 0.0)


# ---------------------------------------------------------------------------
# CLI: cmd_budget_report
# ---------------------------------------------------------------------------

class TestCmdBudgetReport(unittest.TestCase):
    def _args(self, store_dir, fmt="text", since=None, until=None):
        import argparse
        args = argparse.Namespace()
        args.trace_dir = store_dir
        args.format = fmt
        args.since = since
        args.until = until
        args.endpoint = None
        return args

    def test_returns_0_on_empty_store(self):
        store, _ = _make_store()
        args = self._args(store.base_dir)
        result = cmd_budget_report(args)
        self.assertEqual(result, 0)

    def test_returns_0_with_sessions(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 1 * 86400)
        args = self._args(store.base_dir)
        result = cmd_budget_report(args)
        self.assertEqual(result, 0)

    def test_json_format(self):
        import io
        from unittest.mock import patch
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 1 * 86400)
        args = self._args(store.base_dir, fmt="json")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cmd_budget_report(args)
        data = json.loads(captured.getvalue())
        self.assertIn("total_cost", data)

    def test_markdown_format(self):
        import io
        from unittest.mock import patch
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 1 * 86400)
        args = self._args(store.base_dir, fmt="markdown")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cmd_budget_report(args)
        self.assertIn("## Budget Report", captured.getvalue())

    def test_custom_since_until(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3 * 86400)
        args = self._args(store.base_dir, since="7d", until="1d")
        result = cmd_budget_report(args)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
