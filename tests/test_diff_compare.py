"""Tests for issue #47: enhanced diff --compare."""

from __future__ import annotations

import io
import time
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


def _make_store(tmp_dir: str) -> TraceStore:
    import os
    path = os.path.join(tmp_dir, "traces")
    return TraceStore(path)


def _make_two_sessions(store: TraceStore):
    for name in ("session-a", "session-b"):
        meta = SessionMeta(agent_name=name)
        store.create_session(meta)
        sid = meta.session_id
        for event in [
            TraceEvent(event_type=EventType.SESSION_START, session_id=sid, data={}),
            TraceEvent(
                event_type=EventType.TOOL_CALL, session_id=sid,
                data={"tool_name": "read", "arguments": {"file_path": "src/main.py"}},
            ),
            TraceEvent(
                event_type=EventType.TOOL_CALL, session_id=sid,
                data={"tool_name": "write", "arguments": {"file_path": "src/main.py"}},
            ),
            TraceEvent(event_type=EventType.SESSION_END, session_id=sid, data={}),
        ]:
            store.append_event(sid, event)
    sessions = store.list_sessions()
    return sessions[1].session_id, sessions[0].session_id


class TestCompareReport(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_compare_sessions_returns_report(self):
        from agent_trace.diff import compare_sessions
        store = _make_store(self._tmp)
        sid_a, sid_b = _make_two_sessions(store)
        report = compare_sessions(store, sid_a, sid_b)
        self.assertEqual(report.session_a, sid_a)
        self.assertEqual(report.session_b, sid_b)
        self.assertIsInstance(report.cost_a, float)
        self.assertIsInstance(report.tool_calls_a, int)
        self.assertIsInstance(report.verdict, str)

    def test_format_compare_output(self):
        from agent_trace.diff import compare_sessions, format_compare
        store = _make_store(self._tmp)
        sid_a, sid_b = _make_two_sessions(store)
        report = compare_sessions(store, sid_a, sid_b)
        buf = io.StringIO()
        format_compare(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Session Comparison", output)
        self.assertIn("Duration", output)
        self.assertIn("Verdict", output)

    def test_format_compare_has_change_column(self):
        from agent_trace.diff import compare_sessions, format_compare
        store = _make_store(self._tmp)
        sid_a, sid_b = _make_two_sessions(store)
        report = compare_sessions(store, sid_a, sid_b)
        buf = io.StringIO()
        format_compare(report, out=buf)
        # Header must include "change" column
        self.assertIn("change", buf.getvalue())


class TestCompareHelpers(unittest.TestCase):
    def test_redundant_reads_counted(self):
        from agent_trace.diff import _count_redundant_reads
        events = [
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={"tool_name": "read", "arguments": {"file_path": "src/foo.py"}},
            ),
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={"tool_name": "read", "arguments": {"file_path": "src/foo.py"}},
            ),
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={"tool_name": "read", "arguments": {"file_path": "src/bar.py"}},
            ),
        ]
        self.assertEqual(_count_redundant_reads(events), 1)

    def test_no_redundant_reads_unique_files(self):
        from agent_trace.diff import _count_redundant_reads
        events = [
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={"tool_name": "read", "arguments": {"file_path": "a.py"}},
            ),
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={"tool_name": "read", "arguments": {"file_path": "b.py"}},
            ),
        ]
        self.assertEqual(_count_redundant_reads(events), 0)

    def test_context_resets_counted(self):
        from agent_trace.diff import _count_context_resets
        now = time.time()
        events = [
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=now, data={}),
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=now + 200, data={}),
        ]
        self.assertEqual(_count_context_resets(events), 1)

    def test_no_context_resets_within_window(self):
        from agent_trace.diff import _count_context_resets
        now = time.time()
        events = [
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=now, data={}),
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=now + 30, data={}),
        ]
        self.assertEqual(_count_context_resets(events), 0)

    def test_cli_has_compare_flag(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["diff", "abc", "def", "--compare"])
        self.assertTrue(args.compare)


def _report(**fields):
    from agent_trace.diff import CompareReport
    values = dict(
        session_a="sessa005", session_b="sessb005", label_a="plain", label_b="plain",
        duration_a=1.0, duration_b=2.0, cost_a=0.0, cost_b=0.0,
        tool_calls_a=0, tool_calls_b=0, redundant_reads_a=0, redundant_reads_b=0,
        context_resets_a=0, context_resets_b=0, files_modified_a=0, files_modified_b=0,
        errors_a=0, errors_b=0, divergence_points=[(1, "plain", "plain")],
        verdict="inconclusive",
    )
    values.update(fields)
    return CompareReport(**values)


def _compare_text(report):
    from agent_trace.diff import format_compare
    buf = io.StringIO()
    format_compare(report, out=buf)
    return buf.getvalue()


def _divergence_lines(report):
    block = _compare_text(report).split("Approach divergence:\n", 1)[1]
    return block.split("─" * 65, 1)[0]


# (label, kind, build): build(value) returns a report with value in the place
# the label names. The kinds are in tests/_escapes.py. A label shows cut at
# 18 characters in the header and at 10 in each divergence line, so it takes
# the short kind. The header cases have no divergence line, and
# _divergence_lines reads only those lines, so each place is checked on its
# own. The change column divides by the A count, so text in A raises there.
# Files modified has no change column, so it covers the A column. Only a
# hand-built report has text in it, because compare_sessions counts the files.
FORMAT_COMPARE_CASES = [
    ("label A in the header", "short", lambda v: _report(label_a=v, divergence_points=[])),
    ("label B in the header", "short", lambda v: _report(label_b=v, divergence_points=[])),
    ("tool_calls B", "text", lambda v: _report(tool_calls_b=v)),
    ("errors B", "text", lambda v: _report(errors_b=v)),
    ("files_modified A", "text", lambda v: _report(files_modified_a=v)),
    ("files_modified B", "text", lambda v: _report(files_modified_b=v)),
    ("divergence A", "text", lambda v: _report(divergence_points=[(1, v, "plain")])),
    ("divergence B", "text", lambda v: _report(divergence_points=[(1, "plain", v)])),
    ("verdict", "text", lambda v: _report(verdict=v)),
]

DIVERGENCE_LINE_CASES = [
    ("label A in a divergence line", "short", lambda v: _report(label_a=v)),
    ("label B in a divergence line", "short", lambda v: _report(label_b=v)),
]

# (label, cut, build): the formatter keeps cut characters of the value the
# label names. A label shows cut at 18 in the header and at 10 in each
# divergence line, which _divergence_lines checks on their own.
FORMAT_COMPARE_CUTS = [
    ("label A in the header", 18, lambda v: _report(label_a=v)),
    ("label B in the header", 18, lambda v: _report(label_b=v)),
    ("divergence A", 50, lambda v: _report(divergence_points=[(1, v, "plain")])),
    ("divergence B", 50, lambda v: _report(divergence_points=[(1, "plain", v)])),
]

DIVERGENCE_LINE_CUTS = [
    ("label A in a divergence line", 10, lambda v: _report(label_a=v)),
    ("label B in a divergence line", 10, lambda v: _report(label_b=v)),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """An agent name or a path in a trace can hold escape codes and line
    breaks. The comparison shows them as text."""

    VALUES = [
        ("whole output", FORMAT_COMPARE_CASES, _compare_text),
        ("divergence lines", DIVERGENCE_LINE_CASES, _divergence_lines),
    ]
    CUTS = [
        ("whole output", FORMAT_COMPARE_CUTS, _compare_text),
        ("divergence lines", DIVERGENCE_LINE_CUTS, _divergence_lines),
    ]

    def test_each_value_shows_escapes_as_text(self):
        for name, cases, render in self.VALUES:
            with self.subTest(name):
                _escapes.check_each_value(self, cases, render)

    def test_plain_text_prints_unchanged(self):
        for name, cases, render in self.VALUES:
            with self.subTest(name):
                _escapes.check_plain_text_unchanged(self, cases, render)

    def test_value_as_long_as_the_cut_shows_in_full(self):
        for name, cuts, render in self.CUTS:
            with self.subTest(name):
                _escapes.check_each_cut(self, cuts, render)

    def test_sgr_code_shows_as_text(self):
        output = _compare_text(_report(divergence_points=[(1, "a\x1b[8mhidden", "plain")]))
        self.assertIn(r"a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)
