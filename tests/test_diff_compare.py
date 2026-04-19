"""Tests for issue #47: enhanced diff --compare."""

from __future__ import annotations

import io
import time

import pytest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store(tmp_path) -> TraceStore:
    return TraceStore(str(tmp_path / "traces"))


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
    return sessions[1].session_id, sessions[0].session_id  # oldest first


class TestCompare:
    def test_compare_sessions_returns_report(self, tmp_path):
        from agent_trace.diff import compare_sessions
        store = _make_store(tmp_path)
        sid_a, sid_b = _make_two_sessions(store)
        report = compare_sessions(store, sid_a, sid_b)
        assert report.session_a == sid_a
        assert report.session_b == sid_b
        assert isinstance(report.cost_a, float)
        assert isinstance(report.tool_calls_a, int)
        assert isinstance(report.verdict, str)

    def test_format_compare_output(self, tmp_path):
        from agent_trace.diff import compare_sessions, format_compare
        store = _make_store(tmp_path)
        sid_a, sid_b = _make_two_sessions(store)
        report = compare_sessions(store, sid_a, sid_b)
        buf = io.StringIO()
        format_compare(report, out=buf)
        output = buf.getvalue()
        assert "Session Comparison" in output
        assert "Duration" in output
        assert "Verdict" in output

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
        assert _count_redundant_reads(events) == 1

    def test_context_resets_counted(self):
        from agent_trace.diff import _count_context_resets
        now = time.time()
        events = [
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=now, data={}),
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=now + 200, data={}),
        ]
        assert _count_context_resets(events) == 1

    def test_no_context_resets_within_window(self):
        from agent_trace.diff import _count_context_resets
        now = time.time()
        events = [
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=now, data={}),
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=now + 30, data={}),
        ]
        assert _count_context_resets(events) == 0

    def test_cli_has_compare_flag(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["diff", "abc", "def", "--compare"])
        assert args.compare is True
