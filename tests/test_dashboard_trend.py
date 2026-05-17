"""Tests for dashboard --trend: TrendReport, SVG rendering, annotation storage."""

import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.dashboard import (
    TrendAnnotation,
    TrendPoint,
    TrendReport,
    _annotation_x_fracs,
    _svg_sparkline,
    build_trend_report,
    format_trend_terminal,
    load_annotations,
    render_html_trend,
    save_annotation,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp: str) -> TraceStore:
    return TraceStore(tmp)


def _add_session(
    store: TraceStore,
    session_id: str,
    events: list[TraceEvent] | None = None,
    started_at: float | None = None,
    errors: int = 0,
    tool_calls: int = 0,
) -> None:
    ts = started_at or time.time()
    meta = SessionMeta(
        session_id=session_id,
        started_at=ts,
        ended_at=ts + 60,
        errors=errors,
        tool_calls=tool_calls,
        total_duration_ms=60_000,
        total_tokens=1000,
    )
    store.create_session(meta)
    for ev in (events or []):
        store.append_event(session_id, ev)


def _write_eval_json(store: TraceStore, session_id: str, results: list[dict]) -> None:
    path = store.base_dir / session_id / "eval.json"
    path.write_text(json.dumps({"results": results}))


# ---------------------------------------------------------------------------
# SVG sparkline
# ---------------------------------------------------------------------------

class TestSvgSparkline(unittest.TestCase):
    def test_returns_svg_element(self):
        svg = _svg_sparkline([1.0, 2.0, 3.0])
        self.assertIn("<svg", svg)
        self.assertIn("polyline", svg)

    def test_single_value_no_crash(self):
        svg = _svg_sparkline([1.0])
        self.assertIn("<svg", svg)

    def test_empty_no_crash(self):
        svg = _svg_sparkline([])
        self.assertIn("<svg", svg)

    def test_annotations_rendered(self):
        svg = _svg_sparkline([1.0, 2.0, 3.0], annotations=[(0.5, "model upgrade")])
        self.assertIn("model upgrade", svg)
        self.assertIn("line", svg)

    def test_custom_color(self):
        svg = _svg_sparkline([1.0, 2.0], color="#ff0000")
        self.assertIn("#ff0000", svg)


# ---------------------------------------------------------------------------
# Annotation storage
# ---------------------------------------------------------------------------

class TestAnnotations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def test_save_and_load(self):
        save_annotation(self.store, TrendAnnotation(date="2026-05-10", note="Added retry policy"))
        save_annotation(self.store, TrendAnnotation(date="2026-05-12", note="Model upgrade"))
        anns = load_annotations(self.store)
        self.assertEqual(len(anns), 2)
        self.assertEqual(anns[0].date, "2026-05-10")
        self.assertEqual(anns[1].note, "Model upgrade")

    def test_load_empty(self):
        anns = load_annotations(self.store)
        self.assertEqual(anns, [])

    def test_annotation_x_fracs(self):
        now = time.time()
        pts = [
            TrendPoint("s1", now - 86400 * 10, 0, 0, 0, 0),
            TrendPoint("s2", now, 0, 0, 0, 0),
        ]
        from datetime import datetime, timezone
        mid_date = datetime.fromtimestamp(now - 86400 * 5, tz=timezone.utc).strftime("%Y-%m-%d")
        anns = [TrendAnnotation(date=mid_date, note="test")]
        fracs = _annotation_x_fracs(pts, anns)
        self.assertEqual(len(fracs), 1)
        frac, label = fracs[0]
        self.assertGreater(frac, 0.0)
        self.assertLess(frac, 1.0)
        self.assertEqual(label, "test")


# ---------------------------------------------------------------------------
# Trend report builder
# ---------------------------------------------------------------------------

class TestBuildTrendReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def test_empty_store(self):
        report = build_trend_report(self.store)
        self.assertEqual(report.points, [])
        self.assertEqual(report.judge_names, [])

    def test_sessions_included(self):
        now = time.time()
        _add_session(self.store, "s1", started_at=now - 100)
        _add_session(self.store, "s2", started_at=now - 50)
        report = build_trend_report(self.store)
        self.assertEqual(len(report.points), 2)

    def test_since_filter(self):
        now = time.time()
        _add_session(self.store, "old", started_at=now - 86400 * 10)
        _add_session(self.store, "new", started_at=now - 100)
        report = build_trend_report(self.store, since_days=1)
        ids = [p.session_id for p in report.points]
        self.assertIn("new", ids)
        self.assertNotIn("old", ids)

    def test_eval_scores_loaded(self):
        now = time.time()
        _add_session(self.store, "s_eval", started_at=now - 100)
        _write_eval_json(self.store, "s_eval", [
            {"scorer": "no_errors", "score": 1.0, "threshold": 1.0, "passed": True},
        ])
        report = build_trend_report(self.store)
        pt = next(p for p in report.points if p.session_id == "s_eval")
        self.assertEqual(len(pt.eval_scores), 1)
        self.assertEqual(pt.eval_scores[0].judge, "no_errors")
        self.assertIn("no_errors", report.judge_names)

    def test_error_rate_computed(self):
        now = time.time()
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=now, data={"tool_name": "Bash"}),
            TraceEvent(event_type=EventType.ERROR, timestamp=now + 1, data={"message": "fail"}),
        ]
        _add_session(self.store, "s_err", events=events, started_at=now - 100)
        report = build_trend_report(self.store)
        pt = next(p for p in report.points if p.session_id == "s_err")
        self.assertGreater(pt.error_rate, 0.0)

    def test_retry_rate_computed(self):
        now = time.time()
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=now, data={"tool_name": "Bash"}),
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=now + 1, data={"tool_name": "Bash"}),
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=now + 2, data={"tool_name": "Bash"}),
        ]
        _add_session(self.store, "s_retry", events=events, started_at=now - 100)
        report = build_trend_report(self.store)
        pt = next(p for p in report.points if p.session_id == "s_retry")
        self.assertGreater(pt.retry_rate, 0.0)

    def test_points_ordered_oldest_first(self):
        now = time.time()
        _add_session(self.store, "early", started_at=now - 200)
        _add_session(self.store, "late", started_at=now - 50)
        report = build_trend_report(self.store)
        self.assertLess(report.points[0].started_at, report.points[-1].started_at)


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

class TestFormatTrendTerminal(unittest.TestCase):
    def _make_report(self, n: int = 6) -> TrendReport:
        now = time.time()
        pts = [
            TrendPoint(
                session_id=f"s{i}",
                started_at=now - (n - i) * 100,
                error_rate=0.1 * i,
                retry_rate=0.05,
                cost=0.01 * i,
                duration_s=60.0,
            )
            for i in range(n)
        ]
        return TrendReport(points=pts, annotations=[], judge_names=[])

    def test_no_crash_empty(self):
        buf = io.StringIO()
        format_trend_terminal(TrendReport(points=[], annotations=[], judge_names=[]), out=buf)
        self.assertIn("No sessions", buf.getvalue())

    def test_behavioral_section_present(self):
        buf = io.StringIO()
        format_trend_terminal(self._make_report(), out=buf)
        output = buf.getvalue()
        self.assertIn("Behavioral", output)
        self.assertIn("Error rate", output)

    def test_annotations_shown(self):
        report = self._make_report()
        report.annotations = [TrendAnnotation(date="2026-05-10", note="config change")]
        buf = io.StringIO()
        format_trend_terminal(report, out=buf)
        self.assertIn("config change", buf.getvalue())

    def test_eval_scores_shown(self):
        from agent_trace.dashboard import EvalScorePoint
        now = time.time()
        pts = [
            TrendPoint(
                session_id="s1",
                started_at=now - 100,
                error_rate=0.0,
                retry_rate=0.0,
                cost=0.01,
                duration_s=60.0,
                eval_scores=[EvalScorePoint(judge="no_errors", score=1.0, passed=True)],
            ),
        ]
        report = TrendReport(points=pts, annotations=[], judge_names=["no_errors"])
        buf = io.StringIO()
        format_trend_terminal(report, out=buf)
        self.assertIn("no_errors", buf.getvalue())


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

class TestRenderHtmlTrend(unittest.TestCase):
    def _make_report(self) -> TrendReport:
        from agent_trace.dashboard import EvalScorePoint
        now = time.time()
        pts = [
            TrendPoint(
                session_id=f"s{i}",
                started_at=now - (5 - i) * 100,
                error_rate=0.1,
                retry_rate=0.05,
                cost=0.02,
                duration_s=60.0,
                eval_scores=[EvalScorePoint(judge="no_errors", score=float(i % 2), passed=bool(i % 2))],
            )
            for i in range(5)
        ]
        return TrendReport(
            points=pts,
            annotations=[TrendAnnotation(date="2026-05-10", note="model upgrade")],
            judge_names=["no_errors"],
        )

    def test_returns_html(self):
        html_out = render_html_trend(self._make_report())
        self.assertIn("<!DOCTYPE html>", html_out)
        self.assertIn("<svg", html_out)

    def test_judge_section_present(self):
        html_out = render_html_trend(self._make_report())
        self.assertIn("no_errors", html_out)

    def test_annotation_in_html(self):
        html_out = render_html_trend(self._make_report())
        self.assertIn("model upgrade", html_out)

    def test_no_external_resources(self):
        html_out = render_html_trend(self._make_report())
        self.assertNotIn("cdn.", html_out)
        self.assertNotIn("googleapis", html_out)
        self.assertNotIn("cloudflare", html_out)
        self.assertNotIn("<script src", html_out)

    def test_empty_report_no_crash(self):
        report = TrendReport(points=[], annotations=[], judge_names=[])
        html_out = render_html_trend(report)
        self.assertIn("<!DOCTYPE html>", html_out)

    def test_self_contained_no_link_tags(self):
        html_out = render_html_trend(self._make_report())
        self.assertNotIn('<link rel="stylesheet"', html_out)


if __name__ == "__main__":
    unittest.main()
