"""Tests for issue #43: on-call readiness report."""

from __future__ import annotations

import io
import os
import tempfile
import time
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store(tmp_dir: str) -> TraceStore:
    return TraceStore(os.path.join(tmp_dir, "traces"))


def _make_session_with_writes(store: TraceStore, paths: list[str]) -> str:
    meta = SessionMeta(agent_name="test-agent")
    store.create_session(meta)
    sid = meta.session_id
    for path in paths:
        store.append_event(sid, TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=sid,
            data={"tool_name": "write", "arguments": {"file_path": path}},
        ))
    return sid


class TestOncallAnalysis(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_analyse_oncall_returns_report(self):
        from agent_trace.oncall import analyse_oncall
        store = _make_store(self._tmp)
        _make_session_with_writes(store, ["src/auth.py", "src/db.py"])
        report = analyse_oncall(store, rotation_start="2099-01-01")
        self.assertIsInstance(report.unread_files, list)
        self.assertIsInstance(report.total_reading_minutes, float)
        self.assertEqual(report.rotation_start, "2099-01-01")

    def test_days_until_rotation_future(self):
        from agent_trace.oncall import analyse_oncall
        store = _make_store(self._tmp)
        report = analyse_oncall(store, rotation_start="2099-12-31")
        self.assertGreater(report.days_until_rotation, 0)

    def test_days_until_rotation_past(self):
        from agent_trace.oncall import analyse_oncall
        store = _make_store(self._tmp)
        report = analyse_oncall(store, rotation_start="2000-01-01")
        self.assertEqual(report.days_until_rotation, 0)

    def test_agent_modified_files_detected(self):
        from agent_trace.oncall import analyse_oncall
        store = _make_store(self._tmp)
        _make_session_with_writes(store, ["src/auth.py", "src/payments.py"])
        report = analyse_oncall(store, rotation_start="2099-01-01")
        paths = [f.path for f in report.unread_files]
        self.assertIn("src/auth.py", paths)
        self.assertIn("src/payments.py", paths)

    def test_scope_filter_applied(self):
        from agent_trace.oncall import analyse_oncall
        store = _make_store(self._tmp)
        _make_session_with_writes(store, ["src/auth.py", "tests/test_auth.py"])
        report = analyse_oncall(store, rotation_start="2099-01-01", scope_glob="src/**")
        paths = [f.path for f in report.unread_files]
        self.assertIn("src/auth.py", paths)
        self.assertNotIn("tests/test_auth.py", paths)

    def test_format_oncall_output(self):
        from agent_trace.oncall import analyse_oncall, format_oncall
        store = _make_store(self._tmp)
        _make_session_with_writes(store, ["src/auth.py"])
        report = analyse_oncall(store, rotation_start="2099-01-01")
        buf = io.StringIO()
        format_oncall(report, out=buf)
        output = buf.getvalue()
        self.assertIn("On-Call Readiness Report", output)
        self.assertIn("2099-01-01", output)

    def test_empty_store_no_files(self):
        from agent_trace.oncall import analyse_oncall
        store = _make_store(self._tmp)
        report = analyse_oncall(store, rotation_start="2099-01-01")
        self.assertEqual(report.unread_files, [])

    def test_cli_has_oncall_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["oncall", "--rotation-start", "2099-01-01", "--scope", "src/**"])
        self.assertEqual(args.rotation_start, "2099-01-01")
        self.assertEqual(args.scope, "src/**")


