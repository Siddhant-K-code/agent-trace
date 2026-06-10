"""Tests for trace storage."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


class TestTraceStore(unittest.TestCase):
    def setUp(self):
        os.environ.pop("AGENT_TRACE_NO_REDACT", None)
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_NO_REDACT", None)

    def test_create_and_load_session(self):
        meta = SessionMeta(agent_name="test-agent")
        self.store.create_session(meta)

        loaded = self.store.load_meta(meta.session_id)
        self.assertEqual(loaded.agent_name, "test-agent")
        self.assertEqual(loaded.session_id, meta.session_id)

    def test_append_and_load_events(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        e1 = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "read_file"},
        )
        e2 = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            session_id=meta.session_id,
            parent_id=e1.event_id,
            duration_ms=42.5,
            data={"content_preview": "hello world"},
        )

        self.store.append_event(meta.session_id, e1)
        self.store.append_event(meta.session_id, e2)

        events = self.store.load_events(meta.session_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, EventType.TOOL_CALL)
        self.assertEqual(events[1].parent_id, e1.event_id)
        self.assertAlmostEqual(events[1].duration_ms, 42.5)

    def test_list_sessions(self):
        m1 = SessionMeta(agent_name="first")
        m2 = SessionMeta(agent_name="second")
        self.store.create_session(m1)
        self.store.create_session(m2)

        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_list_sessions_sorted_newest_first_by_started_at(self):
        old = SessionMeta(session_id="zz-old", started_at=1.0)
        new = SessionMeta(session_id="aa-new", started_at=2.0)
        self.store.create_session(old)
        self.store.create_session(new)

        sessions = self.store.list_sessions()
        self.assertEqual(
            [(m.session_id, m.started_at) for m in sessions],
            [("aa-new", 2.0), ("zz-old", 1.0)],
        )

    def test_list_sessions_uses_session_id_tiebreaker(self):
        lower = SessionMeta(session_id="aa-same", started_at=1.0)
        higher = SessionMeta(session_id="zz-same", started_at=1.0)
        self.store.create_session(lower)
        self.store.create_session(higher)

        sessions = self.store.list_sessions()
        self.assertEqual([m.session_id for m in sessions], ["zz-same", "aa-same"])

    def test_list_sessions_skips_malformed_metadata(self):
        valid = SessionMeta(session_id="valid", started_at=1.0)
        self.store.create_session(valid)

        malformed_dir = os.path.join(self.tmpdir, "malformed")
        os.makedirs(malformed_dir)
        with open(os.path.join(malformed_dir, "meta.json"), "w") as f:
            f.write("{not json")
        with open(os.path.join(self.tmpdir, "loose-file"), "w") as f:
            f.write("ignored")

        sessions = self.store.list_sessions()
        self.assertEqual([m.session_id for m in sessions], ["valid"])

    def test_get_latest_session_returns_newest_meta(self):
        old = SessionMeta(session_id="zz-old", started_at=1.0)
        new = SessionMeta(session_id="aa-new", started_at=2.0)
        self.store.create_session(old)
        self.store.create_session(new)

        latest = self.store.get_latest_session()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.session_id, "aa-new")
        self.assertEqual(latest.started_at, 2.0)

    def test_get_latest_session_id_uses_started_at_not_session_id(self):
        old = SessionMeta(session_id="zz-old", started_at=1.0)
        new = SessionMeta(session_id="aa-new", started_at=2.0)
        self.store.create_session(old)
        self.store.create_session(new)

        self.assertEqual(self.store.get_latest_session_id(), "aa-new")

    def test_find_session_by_prefix(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        prefix = meta.session_id[:6]
        found = self.store.find_session(prefix)
        self.assertEqual(found, meta.session_id)

    def test_find_session_not_found(self):
        found = self.store.find_session("nonexistent")
        self.assertIsNone(found)

    def test_session_exists(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        self.assertTrue(self.store.session_exists(meta.session_id))
        self.assertFalse(self.store.session_exists("fake"))

    def test_update_meta(self):
        meta = SessionMeta(agent_name="test")
        self.store.create_session(meta)

        meta.tool_calls = 10
        meta.errors = 2
        self.store.update_meta(meta)

        loaded = self.store.load_meta(meta.session_id)
        self.assertEqual(loaded.tool_calls, 10)
        self.assertEqual(loaded.errors, 2)

    def test_empty_events(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        events = self.store.load_events(meta.session_id)
        self.assertEqual(events, [])

    def test_append_event_redacts_secrets_by_default(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={
                "tool_name": "http_request",
                "arguments": {
                    "headers": {
                        "Authorization": "Bearer sk-abc123def456ghi789jkl012mno345pqr678",
                    },
                },
            },
        )

        self.store.append_event(meta.session_id, event)

        events = self.store.load_events(meta.session_id)
        self.assertTrue(events[0].redacted)
        self.assertNotIn("sk-abc123", str(events[0].data))
        self.assertIn("[REDACTED:", str(events[0].data))

    def test_append_event_allows_redaction_opt_out(self):
        store = TraceStore(self.tmpdir, redact=False)
        meta = SessionMeta()
        store.create_session(meta)

        event = TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=meta.session_id,
            data={"prompt": "use sk-abc123def456ghi789jkl012mno345pqr678"},
        )
        store.append_event(meta.session_id, event)

        events = store.load_events(meta.session_id)
        self.assertFalse(events[0].redacted)
        self.assertIn("sk-abc123", str(events[0].data))

    def test_append_event_respects_no_redact_env(self):
        os.environ["AGENT_TRACE_NO_REDACT"] = "1"
        store = TraceStore(self.tmpdir)
        meta = SessionMeta()
        store.create_session(meta)

        event = TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=meta.session_id,
            data={"prompt": "use sk-abc123def456ghi789jkl012mno345pqr678"},
        )
        store.append_event(meta.session_id, event)

        events = store.load_events(meta.session_id)
        self.assertFalse(events[0].redacted)
        self.assertIn("sk-abc123", str(events[0].data))

    def test_create_session_redacts_metadata_by_default(self):
        meta = SessionMeta(
            agent_name="test-agent",
            command="run --api-key sk-abc123def456ghi789jkl012mno345pqr678",
        )
        self.store.create_session(meta)

        loaded = self.store.load_meta(meta.session_id)
        self.assertTrue(loaded.redacted)
        self.assertNotIn("sk-abc123", loaded.command)


if __name__ == "__main__":
    unittest.main()
