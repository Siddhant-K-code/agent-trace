"""Tests for semantic session diff (issue #28)."""

import os
import sys
import tempfile
import unittest
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.diff import (
    SemanticDiffReport,
    format_semantic_diff,
    semantic_diff,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_session(store, tool_calls=2, errors=0, tokens=1000, files_read=None, files_written=None, cmds=None):
    meta = SessionMeta(
        agent_name="test",
        tool_calls=tool_calls,
        errors=errors,
        total_tokens=tokens,
        total_duration_ms=5000,
    )
    store.create_session(meta)

    for path in (files_read or []):
        e = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "read", "arguments": {"file_path": path}},
        )
        store.append_event(meta.session_id, e)

    for path in (files_written or []):
        e = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "write", "arguments": {"file_path": path}},
        )
        store.append_event(meta.session_id, e)

    for cmd in (cmds or []):
        e = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "bash", "arguments": {"command": cmd}},
        )
        store.append_event(meta.session_id, e)

    return meta.session_id


class TestSemanticDiff(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_basic_diff(self):
        sid_a = _make_session(self.store, errors=0, tokens=1000)
        sid_b = _make_session(self.store, errors=2, tokens=2000)
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertEqual(report.errors_a, 0)
        self.assertEqual(report.errors_b, 2)

    def test_verdict_b_better(self):
        sid_a = _make_session(self.store, errors=3, tokens=5000)
        sid_b = _make_session(self.store, errors=0, tokens=1000)
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertEqual(report.verdict, "B is better")

    def test_verdict_a_better(self):
        sid_a = _make_session(self.store, errors=0, tokens=1000)
        sid_b = _make_session(self.store, errors=3, tokens=5000)
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertEqual(report.verdict, "A is better")

    def test_file_sets(self):
        sid_a = _make_session(self.store, files_read=["src/a.py", "src/b.py"])
        sid_b = _make_session(self.store, files_read=["src/b.py", "src/c.py"])
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertIn("src/b.py", report.files_read_both)
        self.assertIn("src/a.py", report.files_read_a_only)
        self.assertIn("src/c.py", report.files_read_b_only)

    def test_command_sets(self):
        sid_a = _make_session(self.store, cmds=["pytest", "make build"])
        sid_b = _make_session(self.store, cmds=["pytest", "make test"])
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertIn("pytest", report.cmds_both)
        self.assertIn("make build", report.cmds_a_only)
        self.assertIn("make test", report.cmds_b_only)

    def test_identical_sessions_inconclusive(self):
        sid_a = _make_session(self.store, errors=0, tokens=1000)
        sid_b = _make_session(self.store, errors=0, tokens=1000)
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertEqual(report.verdict, "inconclusive")


class TestFormatSemanticDiff(unittest.TestCase):
    def test_format_no_crash(self):
        tmpdir = tempfile.mkdtemp()
        store = TraceStore(tmpdir)
        sid_a = _make_session(store, errors=1, tokens=2000)
        sid_b = _make_session(store, errors=0, tokens=1000)
        report = semantic_diff(store, sid_a, sid_b)
        buf = io.StringIO()
        format_semantic_diff(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Semantic diff", output)
        self.assertIn("Verdict", output)

    def test_format_shows_verdict(self):
        tmpdir = tempfile.mkdtemp()
        store = TraceStore(tmpdir)
        sid_a = _make_session(store, errors=5)
        sid_b = _make_session(store, errors=0)
        report = semantic_diff(store, sid_a, sid_b)
        buf = io.StringIO()
        format_semantic_diff(report, out=buf)
        self.assertIn("B is better", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
