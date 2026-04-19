"""Tests for issue #41: agent standup report."""

from __future__ import annotations

import io
import os
import tempfile
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store(tmp_dir: str) -> TraceStore:
    return TraceStore(os.path.join(tmp_dir, "traces"))


def _make_session(store: TraceStore, events_data: list[dict]) -> str:
    meta = SessionMeta(agent_name="test-agent")
    store.create_session(meta)
    sid = meta.session_id
    for ed in events_data:
        store.append_event(sid, TraceEvent(
            event_type=ed["event_type"],
            session_id=sid,
            data=ed.get("data", {}),
        ))
    return sid


class TestStandupExtraction(unittest.TestCase):
    def test_extract_new_deps_npm(self):
        from agent_trace.standup import _extract_new_deps
        events = [TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "bash", "arguments": {"command": "npm install jsonwebtoken"}},
        )]
        deps = _extract_new_deps(events)
        self.assertIn("jsonwebtoken", deps)

    def test_extract_new_deps_pip(self):
        from agent_trace.standup import _extract_new_deps
        events = [TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "bash", "arguments": {"command": "pip install requests"}},
        )]
        deps = _extract_new_deps(events)
        self.assertIn("requests", deps)

    def test_extract_uncertainties_todo(self):
        from agent_trace.standup import _extract_uncertainties
        events = [TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "write", "arguments": {
                "file_path": "src/auth.py",
                "new_str": "# TODO: handle edge case X\ndef login(): pass",
            }},
        )]
        signals = _extract_uncertainties(events)
        self.assertTrue(any("TODO" in s.description for s in signals))

    def test_extract_approaches_no_retries(self):
        from agent_trace.standup import _extract_approaches
        events = [
            TraceEvent(event_type=EventType.TOOL_RESULT,
                       data={"content": "success", "is_error": False}),
        ]
        approaches = _extract_approaches(events)
        self.assertEqual(len(approaches), 1)
        self.assertFalse(approaches[0].abandoned)

    def test_extract_approaches_with_retries(self):
        from agent_trace.standup import _extract_approaches
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL,
                       data={"tool_name": "bash", "arguments": {}}),
            TraceEvent(event_type=EventType.TOOL_RESULT,
                       data={"content": "error: build failed", "is_error": True}),
            TraceEvent(event_type=EventType.TOOL_RESULT,
                       data={"content": "error: still failing", "is_error": True}),
            TraceEvent(event_type=EventType.TOOL_RESULT,
                       data={"content": "success", "is_error": False}),
        ]
        approaches = _extract_approaches(events)
        abandoned = [a for a in approaches if a.abandoned]
        self.assertGreater(len(abandoned), 0)


class TestStandupReport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_analyse_standup_returns_report(self):
        from agent_trace.standup import analyse_standup
        store = _make_store(self._tmp)
        sid = _make_session(store, [
            {"event_type": EventType.SESSION_START},
            {"event_type": EventType.TOOL_CALL,
             "data": {"tool_name": "read", "arguments": {"file_path": "src/auth.py"}}},
            {"event_type": EventType.TOOL_CALL,
             "data": {"tool_name": "write", "arguments": {"file_path": "src/auth.py",
                                                           "new_str": "def login(): pass"}}},
            {"event_type": EventType.SESSION_END},
        ])
        report = analyse_standup(store, sid)
        self.assertEqual(report.session_id, sid)
        self.assertGreaterEqual(report.files_read, 0)
        self.assertGreaterEqual(report.files_modified, 0)
        self.assertIsInstance(report.cost_usd, float)

    def test_files_modified_tracked(self):
        from agent_trace.standup import analyse_standup
        store = _make_store(self._tmp)
        sid = _make_session(store, [
            {"event_type": EventType.TOOL_CALL,
             "data": {"tool_name": "write", "arguments": {"file_path": "src/a.py", "new_str": "x"}}},
            {"event_type": EventType.TOOL_CALL,
             "data": {"tool_name": "write", "arguments": {"file_path": "src/b.py", "new_str": "y"}}},
        ])
        report = analyse_standup(store, sid)
        self.assertEqual(report.files_modified, 2)
        self.assertIn("src/a.py", report.files_modified_list)

    def test_format_standup_output(self):
        from agent_trace.standup import analyse_standup, format_standup
        store = _make_store(self._tmp)
        sid = _make_session(store, [
            {"event_type": EventType.SESSION_START},
            {"event_type": EventType.SESSION_END},
        ])
        report = analyse_standup(store, sid)
        buf = io.StringIO()
        format_standup(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Session:", output)
        self.assertIn("What the agent did", output)
        self.assertIn("Stats:", output)

    def test_cli_has_standup_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["standup", "--no-llm"])
        self.assertTrue(args.no_llm)


