"""Tests for HTML share generation and postmortem analysis."""

import io
import tempfile
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.share import render_html, _event_summary, _esc
from agent_trace.postmortem import (
    analyze_session,
    format_postmortem,
    render_postmortem_html,
    _load_agents_md,
    _detect_agents_md_violations,
    _find_root_cause,
    _build_timeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: EventType, ts: float, session_id: str = "s1", **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)


def _make_store(events: list[TraceEvent], session_id: str = "s1") -> tuple[TraceStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = TraceStore(tmp.name)
    meta = SessionMeta(
        session_id=session_id,
        started_at=events[0].timestamp if events else 0.0,
        total_duration_ms=(events[-1].timestamp - events[0].timestamp) * 1000 if len(events) > 1 else 0,
    )
    store.create_session(meta)
    for e in events:
        store.append_event(session_id, e)
    store.update_meta(meta)
    return store, tmp


def _basic_events(session_id: str = "s1") -> list[TraceEvent]:
    return [
        _make_event(EventType.SESSION_START, 0.0, session_id),
        _make_event(EventType.USER_PROMPT, 1.0, session_id, prompt="run the tests"),
        _make_event(EventType.TOOL_CALL, 2.0, session_id,
                    tool_name="Bash", arguments={"command": "pytest"}),
        _make_event(EventType.TOOL_RESULT, 3.0, session_id, result="5 passed"),
        _make_event(EventType.ASSISTANT_RESPONSE, 4.0, session_id, text="All tests passed."),
        _make_event(EventType.SESSION_END, 5.0, session_id, exit_code=0),
    ]


def _failed_events(session_id: str = "s1") -> list[TraceEvent]:
    return [
        _make_event(EventType.SESSION_START, 0.0, session_id),
        _make_event(EventType.USER_PROMPT, 1.0, session_id, prompt="install deps"),
        _make_event(EventType.TOOL_CALL, 2.0, session_id,
                    tool_name="Bash", arguments={"command": "pip install psycopg2"}),
        _make_event(EventType.ERROR, 3.0, session_id, message="missing libpq-dev"),
        _make_event(EventType.TOOL_CALL, 4.0, session_id,
                    tool_name="Bash", arguments={"command": "pip install psycopg2"}),
        _make_event(EventType.ERROR, 5.0, session_id, message="missing libpq-dev"),
        _make_event(EventType.SESSION_END, 6.0, session_id, exit_code=1),
    ]


# ---------------------------------------------------------------------------
# share.py tests
# ---------------------------------------------------------------------------

class TestEscaping(unittest.TestCase):
    def test_esc_html_entities(self):
        self.assertEqual(_esc("<script>"), "&lt;script&gt;")
        self.assertEqual(_esc('"hello"'), "&quot;hello&quot;")
        self.assertEqual(_esc("a & b"), "a &amp; b")


class TestEventSummary(unittest.TestCase):
    def test_tool_call_bash(self):
        e = _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Bash", arguments={"command": "ls -la"})
        summary = _event_summary(e)
        self.assertIn("ls -la", summary)

    def test_tool_call_read(self):
        e = _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Read", arguments={"file_path": "src/main.py"})
        summary = _event_summary(e)
        self.assertIn("src/main.py", summary)

    def test_user_prompt(self):
        e = _make_event(EventType.USER_PROMPT, 0.0, prompt="fix the bug")
        self.assertIn("fix the bug", _event_summary(e))

    def test_error_event(self):
        e = _make_event(EventType.ERROR, 0.0, message="something went wrong")
        self.assertIn("something went wrong", _event_summary(e))

    def test_session_end(self):
        e = _make_event(EventType.SESSION_END, 0.0, exit_code=0)
        self.assertIn("exit=0", _event_summary(e))


class TestRenderHtml(unittest.TestCase):
    def setUp(self):
        events = _basic_events("html1")
        self.store, self.tmp = _make_store(events, "html1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_string(self):
        html = render_html(self.store, "html1")
        self.assertIsInstance(html, str)

    def test_valid_html_structure(self):
        html = render_html(self.store, "html1")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<html", html)
        self.assertIn("</html>", html)
        self.assertIn("<head>", html)
        self.assertIn("</head>", html)
        self.assertIn("<body>", html)
        self.assertIn("</body>", html)

    def test_no_external_urls(self):
        html = render_html(self.store, "html1")
        # Should not reference any CDN or external resource
        self.assertNotIn("cdn.jsdelivr.net", html)
        self.assertNotIn("unpkg.com", html)
        self.assertNotIn("googleapis.com", html)
        self.assertNotIn('src="http', html)
        self.assertNotIn('href="http', html)

    def test_session_id_in_output(self):
        html = render_html(self.store, "html1")
        self.assertIn("html1", html)

    def test_css_inlined(self):
        html = render_html(self.store, "html1")
        self.assertIn("<style>", html)

    def test_js_inlined(self):
        html = render_html(self.store, "html1")
        self.assertIn("<script>", html)

    def test_phase_section_present(self):
        html = render_html(self.store, "html1")
        self.assertIn("Phase", html)

    def test_failed_session_shows_failed(self):
        events = _failed_events("html2")
        store2, tmp2 = _make_store(events, "html2")
        try:
            html = render_html(store2, "html2")
            self.assertIn("FAILED", html)
        finally:
            tmp2.cleanup()

    def test_postmortem_html_included_when_provided(self):
        html = render_html(self.store, "html1", postmortem_html="<div>POSTMORTEM</div>")
        self.assertIn("POSTMORTEM", html)


# ---------------------------------------------------------------------------
# postmortem.py tests
# ---------------------------------------------------------------------------

class TestFindRootCause(unittest.TestCase):
    def test_finds_first_error(self):
        events = _failed_events()
        base_ts = events[0].timestamp
        idx, desc, offset = _find_root_cause(events, base_ts)
        self.assertGreater(idx, 0)
        self.assertIn("missing libpq-dev", desc)
        self.assertAlmostEqual(offset, 3.0, places=1)

    def test_no_error_returns_minus_one(self):
        events = _basic_events()
        base_ts = events[0].timestamp
        idx, desc, offset = _find_root_cause(events, base_ts)
        self.assertEqual(idx, -1)


class TestBuildTimeline(unittest.TestCase):
    def test_timeline_has_entries(self):
        events = _basic_events()
        base_ts = events[0].timestamp
        timeline = _build_timeline(events, base_ts, root_cause_idx=-1)
        self.assertGreater(len(timeline), 0)

    def test_root_cause_flagged(self):
        events = _failed_events()
        base_ts = events[0].timestamp
        idx, _, _ = _find_root_cause(events, base_ts)
        timeline = _build_timeline(events, base_ts, root_cause_idx=idx)
        root_entries = [e for e in timeline if e.is_root_cause]
        self.assertEqual(len(root_entries), 1)

    def test_retry_flagged(self):
        events = _failed_events()
        base_ts = events[0].timestamp
        timeline = _build_timeline(events, base_ts, root_cause_idx=-1)
        retries = [e for e in timeline if e.is_retry]
        self.assertGreater(len(retries), 0)


class TestAnalyzeSession(unittest.TestCase):
    def setUp(self):
        events = _failed_events("pm1")
        self.store, self.tmp = _make_store(events, "pm1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_report(self):
        from agent_trace.postmortem import PostmortemReport
        report = analyze_session(self.store, "pm1")
        self.assertIsInstance(report, PostmortemReport)

    def test_failed_session_detected(self):
        report = analyze_session(self.store, "pm1")
        self.assertTrue(report.failed)

    def test_root_cause_populated(self):
        report = analyze_session(self.store, "pm1")
        self.assertIn("libpq-dev", report.root_cause)

    def test_recommendations_non_empty(self):
        report = analyze_session(self.store, "pm1")
        self.assertGreater(len(report.recommendations), 0)

    def test_wasted_seconds_positive(self):
        report = analyze_session(self.store, "pm1")
        self.assertGreater(report.wasted_seconds, 0)


class TestFormatPostmortem(unittest.TestCase):
    def setUp(self):
        events = _failed_events("pm2")
        self.store, self.tmp = _make_store(events, "pm2")

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_contains_session_id(self):
        report = analyze_session(self.store, "pm2")
        buf = io.StringIO()
        format_postmortem(report, out=buf)
        self.assertIn("pm2", buf.getvalue())

    def test_output_contains_root_cause(self):
        report = analyze_session(self.store, "pm2")
        buf = io.StringIO()
        format_postmortem(report, out=buf)
        self.assertIn("Root cause", buf.getvalue())

    def test_output_contains_recommendations(self):
        report = analyze_session(self.store, "pm2")
        buf = io.StringIO()
        format_postmortem(report, out=buf)
        self.assertIn("Recommendations", buf.getvalue())


class TestRenderPostmortemHtml(unittest.TestCase):
    def setUp(self):
        events = _failed_events("pm3")
        self.store, self.tmp = _make_store(events, "pm3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_html_for_failed_session(self):
        report = analyze_session(self.store, "pm3")
        html = render_postmortem_html(report)
        self.assertIn("<div", html)
        self.assertIn("Postmortem", html)

    def test_returns_empty_for_ok_session(self):
        events = _basic_events("pm4")
        store2, tmp2 = _make_store(events, "pm4")
        try:
            report = analyze_session(store2, "pm4")
            html = render_postmortem_html(report)
            self.assertEqual(html, "")
        finally:
            tmp2.cleanup()


class TestAgentsMdViolations(unittest.TestCase):
    def test_detects_pip_when_uv_required(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Bash", arguments={"command": "pip install requests"}),
        ]
        agents_md_lines = ["use uv, not pip for package installation"]
        violations = _detect_agents_md_violations(events, agents_md_lines)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("pip" in v for v in violations))

    def test_no_violation_when_correct_tool_used(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Bash", arguments={"command": "uv install requests"}),
        ]
        agents_md_lines = ["use uv, not pip"]
        violations = _detect_agents_md_violations(events, agents_md_lines)
        self.assertEqual(len(violations), 0)

    def test_empty_agents_md_no_violations(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Bash", arguments={"command": "pip install x"}),
        ]
        violations = _detect_agents_md_violations(events, [])
        self.assertEqual(len(violations), 0)


if __name__ == "__main__":
    unittest.main()
