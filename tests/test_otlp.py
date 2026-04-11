"""Tests for OTLP exporter."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import (
    _make_attributes,
    _to_span_id,
    _to_trace_id,
    _ts_to_nanos,
    session_to_otlp,
)


class TestHelpers(unittest.TestCase):
    def test_trace_id_is_32_hex(self):
        tid = _to_trace_id("abc123")
        self.assertEqual(len(tid), 32)
        int(tid, 16)  # should not raise

    def test_span_id_is_16_hex(self):
        sid = _to_span_id("event-1")
        self.assertEqual(len(sid), 16)
        int(sid, 16)

    def test_trace_id_deterministic(self):
        self.assertEqual(_to_trace_id("session-1"), _to_trace_id("session-1"))

    def test_different_sessions_different_ids(self):
        self.assertNotEqual(_to_trace_id("a"), _to_trace_id("b"))

    def test_ts_to_nanos(self):
        result = _ts_to_nanos(1.5)
        self.assertEqual(result, "1500000000")

    def test_make_attributes_string(self):
        attrs = _make_attributes({"key": "value"})
        self.assertEqual(len(attrs), 1)
        self.assertEqual(attrs[0]["key"], "key")
        self.assertEqual(attrs[0]["value"]["stringValue"], "value")

    def test_make_attributes_int(self):
        attrs = _make_attributes({"count": 42})
        self.assertEqual(attrs[0]["value"]["intValue"], "42")

    def test_make_attributes_float(self):
        attrs = _make_attributes({"ratio": 3.14})
        self.assertAlmostEqual(attrs[0]["value"]["doubleValue"], 3.14)

    def test_make_attributes_bool(self):
        attrs = _make_attributes({"enabled": True})
        self.assertEqual(attrs[0]["value"]["boolValue"], True)

    def test_make_attributes_dict(self):
        attrs = _make_attributes({"config": {"a": 1}})
        self.assertIn("a", attrs[0]["value"]["stringValue"])

    def test_make_attributes_list(self):
        attrs = _make_attributes({"items": [1, 2, 3]})
        self.assertIn("1", attrs[0]["value"]["stringValue"])


class TestSessionToOTLP(unittest.TestCase):
    def _make_session(self):
        meta = SessionMeta(
            agent_name="claude-code",
            command="test",
            tool_calls=3,
            errors=1,
        )
        events = [
            TraceEvent(
                event_type=EventType.SESSION_START,
                session_id=meta.session_id,
                data={"mode": "claude-code-hooks"},
            ),
            TraceEvent(
                event_type=EventType.USER_PROMPT,
                session_id=meta.session_id,
                data={"prompt": "Fix the bug"},
            ),
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                session_id=meta.session_id,
                data={"tool_name": "Bash", "arguments": {"command": "npm test"}},
            ),
        ]
        # tool_result linked to tool_call
        call_id = events[2].event_id
        events.append(TraceEvent(
            event_type=EventType.TOOL_RESULT,
            session_id=meta.session_id,
            data={"tool_name": "Bash", "result": "All tests passed"},
            parent_id=call_id,
            duration_ms=150,
        ))
        events.append(TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "Read", "arguments": {"file_path": "/src/main.py"}},
        ))
        # error linked to tool_call
        read_id = events[4].event_id
        events.append(TraceEvent(
            event_type=EventType.ERROR,
            session_id=meta.session_id,
            data={"tool_name": "Read", "error": "File not found"},
            parent_id=read_id,
            duration_ms=5,
        ))
        events.append(TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE,
            session_id=meta.session_id,
            data={"text": "I fixed the bug."},
        ))
        events.append(TraceEvent(
            event_type=EventType.SESSION_END,
            session_id=meta.session_id,
            data={},
        ))
        return meta, events

    def test_returns_valid_structure(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)

        self.assertIn("resourceSpans", payload)
        self.assertEqual(len(payload["resourceSpans"]), 1)

        rs = payload["resourceSpans"][0]
        self.assertIn("resource", rs)
        self.assertIn("scopeSpans", rs)
        self.assertEqual(len(rs["scopeSpans"]), 1)

        ss = rs["scopeSpans"][0]
        self.assertIn("spans", ss)
        self.assertGreater(len(ss["spans"]), 0)

    def test_root_span_exists(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        root = spans[0]
        self.assertIn("agent-session", root["name"])
        self.assertNotIn("parentSpanId", root)

    def test_tool_call_becomes_child_span(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        root_id = spans[0]["spanId"]
        child_spans = [s for s in spans[1:] if s.get("parentSpanId") == root_id]
        self.assertGreater(len(child_spans), 0)

        bash_span = [s for s in child_spans if "bash" in s["name"].lower()]
        self.assertEqual(len(bash_span), 1)

    def test_error_span_has_error_status(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        # Root span also has error status; filter to child spans only
        root_id = spans[0]["spanId"]
        error_children = [s for s in spans[1:] if s.get("status", {}).get("code") == 2]
        self.assertEqual(len(error_children), 1)
        self.assertEqual(error_children[0]["name"], "Read")

    def test_error_span_has_exception_event(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        error_children = [s for s in spans[1:] if s.get("status", {}).get("code") == 2]
        self.assertEqual(len(error_children[0].get("events", [])), 1)
        self.assertEqual(error_children[0]["events"][0]["name"], "exception")

    def test_user_prompt_is_root_event(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        root = spans[0]
        prompt_events = [e for e in root.get("events", []) if e["name"] == "user_prompt"]
        self.assertEqual(len(prompt_events), 1)

    def test_assistant_response_is_root_event(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        root = spans[0]
        resp_events = [e for e in root.get("events", []) if e["name"] == "assistant_response"]
        self.assertEqual(len(resp_events), 1)

    def test_trace_id_consistent(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        trace_ids = {s["traceId"] for s in spans}
        self.assertEqual(len(trace_ids), 1)

    def test_service_name_in_resource(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events, service_name="my-agent")

        attrs = payload["resourceSpans"][0]["resource"]["attributes"]
        service_attr = [a for a in attrs if a["key"] == "service.name"]
        self.assertEqual(len(service_attr), 1)
        self.assertEqual(service_attr[0]["value"]["stringValue"], "my-agent")

    def test_tool_input_in_span_attributes(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        bash_span = [s for s in spans if "bash" in s["name"].lower()][0]
        attr_keys = [a["key"] for a in bash_span["attributes"]]
        self.assertIn("tool.input.command", attr_keys)

    def test_span_has_duration(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        bash_span = [s for s in spans if "bash" in s["name"].lower()][0]
        start = int(bash_span["startTimeUnixNano"])
        end = int(bash_span["endTimeUnixNano"])
        self.assertGreater(end, start)

    def test_empty_session(self):
        meta = SessionMeta(agent_name="test")
        payload = session_to_otlp(meta, [])
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        # Should still have root span
        self.assertEqual(len(spans), 1)


if __name__ == "__main__":
    unittest.main()
