"""Tests for trace data model."""

import json
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent


class TestTraceEvent(unittest.TestCase):
    def test_create_tool_call_event(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id="abc123",
            data={"tool_name": "read_file", "arguments": {"path": "/tmp/test.py"}},
        )
        self.assertEqual(event.event_type, EventType.TOOL_CALL)
        self.assertEqual(event.data["tool_name"], "read_file")
        self.assertIsNotNone(event.event_id)
        self.assertIsNotNone(event.timestamp)

    def test_serialize_roundtrip(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id="abc123",
            data={"tool_name": "search", "arguments": {"query": "test"}},
        )
        json_str = event.to_json()
        parsed = TraceEvent.from_json(json_str)
        self.assertEqual(parsed.event_type, EventType.TOOL_CALL)
        self.assertEqual(parsed.data["tool_name"], "search")
        self.assertEqual(parsed.session_id, "abc123")

    def test_json_omits_empty_fields(self):
        event = TraceEvent(
            event_type=EventType.ERROR,
            data={"message": "something broke"},
        )
        json_str = event.to_json()
        d = json.loads(json_str)
        self.assertNotIn("parent_id", d)
        self.assertNotIn("duration_ms", d)
        self.assertNotIn("session_id", d)

    def test_all_event_types_valid(self):
        for et in EventType:
            event = TraceEvent(event_type=et, data={})
            json_str = event.to_json()
            parsed = TraceEvent.from_json(json_str)
            self.assertEqual(parsed.event_type, et)


class TestSessionMeta(unittest.TestCase):
    def test_create_session(self):
        meta = SessionMeta(agent_name="test-agent")
        self.assertIsNotNone(meta.session_id)
        self.assertEqual(meta.agent_name, "test-agent")
        self.assertEqual(meta.tool_calls, 0)

    def test_serialize_roundtrip(self):
        meta = SessionMeta(
            agent_name="test",
            tool_calls=5,
            llm_requests=3,
            errors=1,
        )
        json_str = meta.to_json()
        parsed = SessionMeta.from_json(json_str)
        self.assertEqual(parsed.agent_name, "test")
        self.assertEqual(parsed.tool_calls, 5)
        self.assertEqual(parsed.llm_requests, 3)
        self.assertEqual(parsed.errors, 1)


class TestForwardCompatibleReaders(unittest.TestCase):
    """Fields added by newer writers must not break older readers."""

    def test_session_meta_ignores_unknown_fields(self):
        text = json.dumps({
            "session_id": "fwdcompat0000001",
            "agent_name": "claude-code",
            "field_from_a_future_release": {"nested": True},
        })
        meta = SessionMeta.from_json(text)
        self.assertEqual(meta.session_id, "fwdcompat0000001")
        self.assertFalse(hasattr(meta, "field_from_a_future_release"))

    def test_trace_event_ignores_unknown_fields(self):
        line = json.dumps({
            "event_type": "user_prompt",
            "event_id": "evt1",
            "data": {"prompt": "hi"},
            "field_from_a_future_release": "x",
        })
        event = TraceEvent.from_json(line)
        self.assertEqual(event.event_id, "evt1")
        self.assertEqual(event.data, {"prompt": "hi"})

    def test_rewrite_keeps_unknown_fields(self):
        # update_meta and tag_session re-serialise loaded records; a reader
        # that does not know a field must not erase it.
        meta = SessionMeta.from_json(json.dumps({
            "session_id": "fwdcompat0000002",
            "field_from_a_future_release": "kept",
        }))
        meta.tool_calls = 3
        written = json.loads(meta.to_json())
        self.assertEqual(written["field_from_a_future_release"], "kept")
        self.assertEqual(written["tool_calls"], 3)

        event = TraceEvent.from_json(json.dumps({
            "event_type": "user_prompt",
            "field_from_a_future_release": "kept",
        }))
        self.assertEqual(json.loads(event.to_json())["field_from_a_future_release"], "kept")

    def test_records_without_unknown_fields_serialise_unchanged(self):
        meta = SessionMeta.from_json(json.dumps({"session_id": "fwdcompat0000003"}))
        self.assertNotIn("_unknown_fields", json.loads(meta.to_json()))

    def test_empty_conversation_id_is_not_written(self):
        # v0.106.0 and older readers reject the key, so sessions without a
        # native ID must not carry it.
        meta = SessionMeta(session_id="fwdcompat0000004", conversation_id="")
        self.assertNotIn("conversation_id", json.loads(meta.to_json()))
        meta.conversation_id = "native-session-id"
        self.assertEqual(json.loads(meta.to_json())["conversation_id"], "native-session-id")


if __name__ == "__main__":
    unittest.main()
