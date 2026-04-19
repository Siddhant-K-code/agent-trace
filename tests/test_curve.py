"""Tests for issue #50: personal cost curve."""

from __future__ import annotations

import io

import pytest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store(tmp_path) -> TraceStore:
    return TraceStore(str(tmp_path / "traces"))


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


class TestCurve:
    def test_analyse_curve_returns_report(self, tmp_path):
        from agent_trace.curve import analyse_curve
        store = _make_store(tmp_path)
        _populate_sessions(store, 25)
        report = analyse_curve(store, min_sessions=20)
        assert report.session_count == 25
        assert not report.insufficient_data
        assert len(report.stats) > 0

    def test_insufficient_data_flag(self, tmp_path):
        from agent_trace.curve import analyse_curve
        store = _make_store(tmp_path)
        _populate_sessions(store, 5)
        report = analyse_curve(store, min_sessions=20)
        assert report.insufficient_data

    def test_classify_session(self):
        from agent_trace.curve import _classify_session
        assert _classify_session("write unit tests", "pytest") == "Unit test writing"
        assert _classify_session("fix bug in auth", "") == "Bug debugging"
        assert _classify_session("refactor database", "") == "Code refactoring"
        assert _classify_session("", "") == "General / other"

    def test_classify_architecture(self):
        from agent_trace.curve import _classify_session
        assert _classify_session("system design for payments", "") == "Architecture"

    def test_format_curve_output(self, tmp_path):
        from agent_trace.curve import analyse_curve, format_curve
        store = _make_store(tmp_path)
        _populate_sessions(store, 25)
        report = analyse_curve(store, min_sessions=20)
        buf = io.StringIO()
        format_curve(report, out=buf)
        output = buf.getvalue()
        assert "Cost Curve" in output
        assert "Sweet spot" in output
        assert "Verdict" in output

    def test_export_csv(self, tmp_path):
        from agent_trace.curve import analyse_curve, export_curve_csv
        store = _make_store(tmp_path)
        _populate_sessions(store, 25)
        report = analyse_curve(store, min_sessions=20)
        buf = io.StringIO()
        export_curve_csv(report, out=buf)
        lines = buf.getvalue().strip().splitlines()
        assert lines[0].startswith("task_type")
        assert len(lines) > 1

    def test_empty_store(self, tmp_path):
        from agent_trace.curve import analyse_curve
        store = _make_store(tmp_path)
        report = analyse_curve(store)
        assert report.session_count == 0
        assert report.insufficient_data

    def test_stats_have_verdict(self, tmp_path):
        from agent_trace.curve import analyse_curve
        store = _make_store(tmp_path)
        _populate_sessions(store, 25)
        report = analyse_curve(store, min_sessions=20)
        for stat in report.stats:
            assert stat.verdict in ("efficient", "over sweet spot", "do this yourself")
            assert stat.verdict_icon in ("✅", "⚠️ ", "❌")

    def test_cli_has_curve_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["curve", "--min-sessions", "10", "--export", "csv"])
        assert args.min_sessions == 10
        assert args.export == "csv"
