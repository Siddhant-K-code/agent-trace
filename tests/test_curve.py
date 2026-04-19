"""Tests for issue #50: personal cost curve."""

from __future__ import annotations

import io
import os
import tempfile
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store(tmp_dir: str) -> TraceStore:
    return TraceStore(os.path.join(tmp_dir, "traces"))


def _populate_sessions(store: TraceStore, n: int = 25) -> None:
    task_types = [
        ("write unit tests for auth module", "pytest"),
        ("fix bug in login flow", "debug"),
        ("refactor database layer", "refactor"),
    ]
    for i in range(n):
        name, cmd = task_types[i % len(task_types)]
        meta = SessionMeta(agent_name=name, command=cmd)
        store.create_session(meta)
        sid = meta.session_id
        store.append_event(sid, TraceEvent(
            event_type=EventType.LLM_REQUEST,
            session_id=sid,
            data={"model": "claude-opus-4-6", "input_tokens": 500},
        ))


class TestCurveClassification(unittest.TestCase):
    def test_classify_unit_tests(self):
        from agent_trace.curve import _classify_session
        self.assertEqual(_classify_session("write unit tests", "pytest"), "Unit test writing")

    def test_classify_bug_debugging(self):
        from agent_trace.curve import _classify_session
        self.assertEqual(_classify_session("fix bug in auth", ""), "Bug debugging")

    def test_classify_refactoring(self):
        from agent_trace.curve import _classify_session
        self.assertEqual(_classify_session("refactor database", ""), "Code refactoring")

    def test_classify_architecture(self):
        from agent_trace.curve import _classify_session
        self.assertEqual(_classify_session("system design for payments", ""), "Architecture")

    def test_classify_fallback(self):
        from agent_trace.curve import _classify_session
        self.assertEqual(_classify_session("", ""), "General / other")


class TestCurveAnalysis(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_analyse_curve_returns_report(self):
        from agent_trace.curve import analyse_curve
        store = _make_store(self._tmp)
        _populate_sessions(store, 25)
        report = analyse_curve(store, min_sessions=20)
        self.assertEqual(report.session_count, 25)
        self.assertFalse(report.insufficient_data)
        self.assertGreater(len(report.stats), 0)

    def test_insufficient_data_flag(self):
        from agent_trace.curve import analyse_curve
        store = _make_store(self._tmp)
        _populate_sessions(store, 5)
        report = analyse_curve(store, min_sessions=20)
        self.assertTrue(report.insufficient_data)

    def test_stats_have_valid_verdict(self):
        from agent_trace.curve import analyse_curve
        store = _make_store(self._tmp)
        _populate_sessions(store, 25)
        report = analyse_curve(store, min_sessions=20)
        valid_verdicts = {"efficient", "over sweet spot", "do this yourself"}
        for stat in report.stats:
            self.assertIn(stat.verdict, valid_verdicts)

    def test_format_curve_output(self):
        from agent_trace.curve import analyse_curve, format_curve
        store = _make_store(self._tmp)
        _populate_sessions(store, 25)
        report = analyse_curve(store, min_sessions=20)
        buf = io.StringIO()
        format_curve(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Cost Curve", output)
        self.assertIn("Sweet spot", output)

    def test_export_csv_header(self):
        from agent_trace.curve import analyse_curve, export_curve_csv
        store = _make_store(self._tmp)
        _populate_sessions(store, 25)
        report = analyse_curve(store, min_sessions=20)
        buf = io.StringIO()
        export_curve_csv(report, out=buf)
        lines = buf.getvalue().strip().splitlines()
        self.assertTrue(lines[0].startswith("task_type"))
        self.assertGreater(len(lines), 1)

    def test_empty_store(self):
        from agent_trace.curve import analyse_curve
        store = _make_store(self._tmp)
        report = analyse_curve(store)
        self.assertEqual(report.session_count, 0)
        self.assertTrue(report.insufficient_data)

    def test_cli_has_curve_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["curve", "--min-sessions", "10", "--export", "csv"])
        self.assertEqual(args.min_sessions, 10)
        self.assertEqual(args.export, "csv")
