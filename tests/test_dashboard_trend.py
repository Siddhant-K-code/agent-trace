"""Tests for dashboard --trend: TrendReport, SVG rendering, annotation storage."""

import io
import json
import tempfile
import time
import unittest
from html import escape as html_escape
from pathlib import Path

from agent_trace.dashboard import (
    EvalScorePoint,
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
from agent_trace.cost import estimate_cost
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    from test_cost import _codex_events
except ImportError:  # pragma: no cover - package-style test discovery
    from tests.test_cost import _codex_events

try:
    import _escapes
except ImportError:  # pragma: no cover - package-style test discovery
    from tests import _escapes

try:
    from test_fileutil import RAW_SEPARATORS
except ImportError:  # pragma: no cover - package-style test discovery
    from tests.test_fileutil import RAW_SEPARATORS

# Markup that ends the cell and runs script if it goes into the page raw.
HTML_PAYLOAD = "</td><script>alert(1)</script><img src=x onerror=alert(1)>"


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


def _trend(judge="plain", date="2026-05-10", note="plain", n=2) -> TrendReport:
    """A report built directly: *n* sessions, each scored by *judge*, and one annotation.

    With one session the terminal prints the judge with "(no eval scores)",
    and with two it prints the judge with a pass rate.
    """
    points = [
        TrendPoint(
            session_id=f"s{i}", started_at=1_700_000_000.0 + i * 100, error_rate=0.0,
            retry_rate=0.0, cost=0.01, duration_s=60.0,
            eval_scores=[EvalScorePoint(judge=judge, score=1.0, passed=True)],
        )
        for i in range(n)
    ]
    return TrendReport(
        points=points,
        annotations=[TrendAnnotation(date=date, note=note)],
        judge_names=[judge],
    )


def _format_trend(report: TrendReport) -> str:
    buf = io.StringIO()
    format_trend_terminal(report, out=buf)
    return buf.getvalue()


# (label, kind, build): build(value) returns a report with value in the place
# the label names. A judge name comes from eval.json and an annotation from
# annotations.jsonl, so each can be any JSON value. The kinds are in
# tests/_escapes.py.
TREND_TERMINAL_CASES = [
    ("judge with no scores", "text", lambda v: _trend(judge=v, n=1)),
    ("judge with a pass rate", "text", lambda v: _trend(judge=v)),
    ("annotation date", "text", lambda v: _trend(date=v)),
    ("annotation note", "free", lambda v: _trend(note=v)),
]


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

    def test_load_keeps_a_raw_line_separator_inside_a_note(self):
        notes = [f"a{separator}b" for separator in RAW_SEPARATORS]
        path = self.store.base_dir / "annotations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(
            (json.dumps({"date": "2026-05-10", "note": note}, ensure_ascii=False) + "\n")
            .encode("utf-8")
            for note in notes
        ))

        self.assertEqual([annotation.note for annotation in load_annotations(self.store)], notes)

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

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, TREND_TERMINAL_CASES, _format_trend)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, TREND_TERMINAL_CASES, _format_trend)

    def test_whitespace_in_free_text_shows_as_one_space(self):
        _escapes.check_whitespace_in_free_text(self, TREND_TERMINAL_CASES, _format_trend)

    def test_whitespace_in_judge_name_shows_as_escapes(self):
        output = _format_trend(_trend(judge=_escapes.SPACES))
        self.assertIn(r"one\ttwo  three\u3000four", output)

    def test_sgr_code_shows_as_text(self):
        for label, _, build in TREND_TERMINAL_CASES:
            with self.subTest(label):
                output = _format_trend(build("a\x1b[8mhidden"))
                self.assertIn(r"a\x1b[8mhidden", output)
                self.assertNotIn("\x1b", output)

    def test_plain_lines_print_as_before(self):
        plain = _escapes.PLAIN
        self.assertIn(f"  {plain:<30} (no eval scores)\n", _format_trend(_trend(judge=plain, n=1)))
        self.assertIn(f"  {plain:<30} 100% → 100%  → +0pp\n", _format_trend(_trend(judge=plain)))
        self.assertIn(f"  {plain}  {plain}\n", _format_trend(_trend(date=plain, note=plain)))


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

    def test_judge_name_in_session_table_is_escaped(self):
        page = render_html_trend(_trend(judge=HTML_PAYLOAD, n=1))
        self.assertNotIn(HTML_PAYLOAD, page)
        # Once in the chart label, which was already escaped, and once in the table.
        self.assertEqual(page.count(html_escape(HTML_PAYLOAD)), 2)

    def test_judge_names_in_session_table_render_as_before(self):
        page = render_html_trend(_trend(judge=_escapes.PLAIN, n=1))
        self.assertIn(f"<td>{_escapes.PLAIN}=1.00</td>", page)
        # The chart label needs a str judge, so the number goes in the table only.
        report = _trend(judge=7, n=1)
        report.judge_names = []
        self.assertIn("<td>7=1.00</td>", render_html_trend(report))


class TestTrendCost(unittest.TestCase):
    """Trend costs are the `agent-strace cost` totals."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = _make_store(tmp.name)
        self.now = time.time()

    def _codex(self, sid: str, offset: float, prompt_data=None) -> None:
        events = _codex_events(sid)
        if prompt_data is not None:
            events[0].data = prompt_data
        # meta.total_tokens is 1000, which the old flat-rate estimate priced.
        _add_session(self.store, sid, events, started_at=self.now - offset, tool_calls=1)

    def test_cost_matches_cost_command(self):
        self._codex("a", 200)
        [point] = build_trend_report(self.store).points
        self.assertEqual(point.cost, estimate_cost(self.store, "a").total_cost)
        self.assertNotAlmostEqual(point.cost, 1000 / 1_000_000 * 3.0)
        page = render_html_trend(build_trend_report(self.store))
        self.assertIn(f"<td>${point.cost:.4f}</td>", page)
        self.assertNotIn("not priced", page)

    def test_unpriceable_session_is_not_priced_at_zero(self):
        self._codex("a", 300)
        self._codex("b", 200, prompt_data=["list the files"])
        self._codex("c", 100)
        report = build_trend_report(self.store)
        priced, unpriced, _ = report.points
        self.assertIsNone(unpriced.cost)
        self.assertEqual(priced.cost, estimate_cost(self.store, "a").total_cost)
        # Error and retry rates still come from the events.
        self.assertEqual((unpriced.error_rate, unpriced.retry_rate),
                         (priced.error_rate, priced.retry_rate))

        buf = io.StringIO()
        format_trend_terminal(report, out=buf)
        text = buf.getvalue()
        self.assertIn(f"  {'Sessions not priced':<30} 1\n", text)
        self.assertIn(f"{priced.cost:.3f} → {priced.cost:.3f}", text)

        page = render_html_trend(report)
        self.assertIn("<td>n/a</td>", page)
        self.assertIn('<span class="badge">1 not priced</span>', page)
        self.assertNotIn("$0.0000", page)

    def test_unreadable_events_are_not_priced(self):
        self._codex("a", 200)
        with open(self.store.base_dir / "a" / "events.ndjson", "a") as f:
            f.write("not json\n")
        [point] = build_trend_report(self.store).points
        self.assertIsNone(point.cost)

    def test_all_priced_terminal_has_no_unpriced_line(self):
        self._codex("a", 300)
        self._codex("b", 200)
        buf = io.StringIO()
        format_trend_terminal(build_trend_report(self.store), out=buf)
        self.assertIn("Avg cost/session ($)", buf.getvalue())
        self.assertNotIn("not priced", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
