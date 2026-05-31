"""Tests for replay --diff dual-session HTML viewer (issue #139)."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.replay import replay_to_html_diff
from agent_trace.store import TraceStore


def _make_session(store: TraceStore, tool_names: list[str]) -> str:
    meta = SessionMeta(agent_name="test")
    meta.started_at = time.time() - 60
    meta.ended_at = time.time()
    store.create_session(meta)
    for name in tool_names:
        store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": name},
        ))
    store.update_meta(meta)
    return meta.session_id


class TestReplayDiff(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_returns_html_string(self):
        sid_a = _make_session(self.store, ["Bash", "Read"])
        sid_b = _make_session(self.store, ["Bash", "Write"])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        self.assertIsInstance(html, str)
        self.assertIn("<!DOCTYPE html>", html)

    def test_both_session_ids_in_output(self):
        sid_a = _make_session(self.store, ["Bash"])
        sid_b = _make_session(self.store, ["Read"])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        self.assertIn(sid_a[:16], html)
        self.assertIn(sid_b[:16], html)

    def test_event_counts_in_output(self):
        sid_a = _make_session(self.store, ["Bash", "Read", "Write"])
        sid_b = _make_session(self.store, ["Bash"])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        # A has 3 events — the count should appear in the stats section
        self.assertIn("<b>3</b>", html)

    def test_writes_to_output_path(self):
        sid_a = _make_session(self.store, ["Bash"])
        sid_b = _make_session(self.store, ["Read"])
        out = os.path.join(self.tmpdir, "diff.html")
        replay_to_html_diff(self.store, sid_a, sid_b, output_path=out)
        self.assertTrue(os.path.exists(out))
        content = open(out).read()
        self.assertIn("<!DOCTYPE html>", content)

    def test_only_a_ids_in_output(self):
        sid_a = _make_session(self.store, ["UniqueToolA"])
        sid_b = _make_session(self.store, ["UniqueToolB"])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        # JS variables onlyA / onlyB must be present
        self.assertIn("onlyA", html)
        self.assertIn("onlyB", html)

    def test_empty_sessions_dont_crash(self):
        sid_a = _make_session(self.store, [])
        sid_b = _make_session(self.store, [])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        self.assertIn("<!DOCTYPE html>", html)


if __name__ == "__main__":
    unittest.main()
