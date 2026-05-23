"""Tests for replay performance improvements: --limit flag and progress indicator."""

import io
import os
import sys
import tempfile
import time
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.replay import replay_session, _LARGE_SESSION_THRESHOLD
from agent_trace.store import TraceStore


def _make_store_with_events(n: int) -> tuple[TraceStore, str]:
    """Create a temp store with n tool_call events and return (store, session_id)."""
    tmpdir = tempfile.mkdtemp()
    store = TraceStore(tmpdir)
    meta = SessionMeta()
    store.create_session(meta)
    for i in range(n):
        ev = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            timestamp=time.time() + i,
            data={"tool_name": "Bash", "arguments": {"command": f"echo {i}"}},
        )
        store.append_event(meta.session_id, ev)
    return store, meta.session_id


class TestReplayLimit(unittest.TestCase):
    def test_limit_caps_output(self):
        store, sid = _make_store_with_events(50)
        out = io.StringIO()
        replay_session(store, sid, out=out, limit=10)
        output = out.getvalue()
        # Should mention the limit
        self.assertIn("10", output)
        # Should not contain all 50 echo commands
        echo_count = output.count("echo ")
        self.assertLessEqual(echo_count, 10)

    def test_limit_none_shows_all(self):
        store, sid = _make_store_with_events(20)
        out = io.StringIO()
        replay_session(store, sid, out=out, limit=None)
        output = out.getvalue()
        echo_count = output.count("echo ")
        self.assertEqual(echo_count, 20)

    def test_limit_larger_than_events_shows_all(self):
        store, sid = _make_store_with_events(10)
        out = io.StringIO()
        replay_session(store, sid, out=out, limit=100)
        output = out.getvalue()
        echo_count = output.count("echo ")
        self.assertEqual(echo_count, 10)
        # No truncation notice when limit >= total
        self.assertNotIn("more events not shown", output)

    def test_truncation_notice_shown(self):
        store, sid = _make_store_with_events(30)
        out = io.StringIO()
        replay_session(store, sid, out=out, limit=5)
        output = out.getvalue()
        self.assertIn("more events not shown", output)
        self.assertIn("25", output)  # 30 - 5 = 25 truncated

    def test_no_truncation_notice_when_not_truncated(self):
        store, sid = _make_store_with_events(5)
        out = io.StringIO()
        replay_session(store, sid, out=out, limit=10)
        output = out.getvalue()
        self.assertNotIn("more events not shown", output)

    def test_limit_zero_treated_as_no_limit(self):
        # limit=0 is treated as "no limit" (same as None) — 0 is not a useful cap
        store, sid = _make_store_with_events(10)
        out = io.StringIO()
        replay_session(store, sid, out=out, limit=0)
        output = out.getvalue()
        echo_count = output.count("echo ")
        self.assertEqual(echo_count, 10)

    def test_events_are_chronological(self):
        """First N events shown, not last N."""
        store, sid = _make_store_with_events(20)
        out = io.StringIO()
        replay_session(store, sid, out=out, limit=3)
        output = out.getvalue()
        # echo 0, 1, 2 should appear; echo 19 should not
        self.assertIn("echo 0", output)
        self.assertIn("echo 1", output)
        self.assertIn("echo 2", output)
        self.assertNotIn("echo 19", output)


class TestReplayProgressIndicator(unittest.TestCase):
    def test_progress_written_to_stderr_for_large_session(self):
        n = _LARGE_SESSION_THRESHOLD + 10
        store, sid = _make_store_with_events(n)
        out = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = err = io.StringIO()
        try:
            replay_session(store, sid, out=out)
        finally:
            sys.stderr = old_stderr
        self.assertIn("Loading", err.getvalue())
        self.assertIn(str(n), err.getvalue())

    def test_no_progress_for_small_session(self):
        store, sid = _make_store_with_events(5)
        out = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = err = io.StringIO()
        try:
            replay_session(store, sid, out=out)
        finally:
            sys.stderr = old_stderr
        self.assertEqual(err.getvalue(), "")

    def test_progress_mentions_limit_when_truncated(self):
        n = _LARGE_SESSION_THRESHOLD + 50
        store, sid = _make_store_with_events(n)
        out = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = err = io.StringIO()
        try:
            replay_session(store, sid, out=out, limit=10)
        finally:
            sys.stderr = old_stderr
        self.assertIn("showing first 10", err.getvalue())


class TestReplayCLILimitFlag(unittest.TestCase):
    def test_limit_flag_registered(self):
        from agent_trace.cli import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "replay", "--help"]
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                main()
            except SystemExit:
                pass
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        self.assertIn("--limit", output)


if __name__ == "__main__":
    unittest.main()
