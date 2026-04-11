"""Tests for OTLP improvements: span hierarchy and semantic conventions (issue #25)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import (
    _SEMCONV_SYSTEM,
    _SEMCONV_TOOL_NAME,
    _SEMCONV_OP,
    session_to_otlp,
    tree_to_otlp,
)
from agent_trace.store import TraceStore


def _make_session(store, agent_name="claude-code", events=None):
    meta = SessionMeta(agent_name=agent_name)
    store.create_session(meta)
    for e in (events or []):
        e.session_id = meta.session_id
        store.append_event(meta.session_id, e)
    return meta


def _get_spans(payload):
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attr_value(attrs, key):
    for a in attrs:
        if a["key"] == key:
            v = a["value"]
            return v.get("stringValue") or v.get("intValue") or v.get("boolValue")
    return None


class TestSessionToOtlpSemanticConventions(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_gen_ai_system_anthropic(self):
        meta = _make_session(self.store, agent_name="claude-code")
        payload = session_to_otlp(meta, [])
        spans = _get_spans(payload)
        root = spans[0]
        system = _attr_value(root["attributes"], _SEMCONV_SYSTEM)
        self.assertEqual(system, "anthropic")

    def test_gen_ai_system_openai(self):
        meta = _make_session(self.store, agent_name="gpt-4o-agent")
        payload = session_to_otlp(meta, [])
        spans = _get_spans(payload)
        system = _attr_value(spans[0]["attributes"], _SEMCONV_SYSTEM)
        self.assertEqual(system, "openai")

    def test_gen_ai_system_unknown(self):
        meta = _make_session(self.store, agent_name="my-custom-agent")
        payload = session_to_otlp(meta, [])
        spans = _get_spans(payload)
        system = _attr_value(spans[0]["attributes"], _SEMCONV_SYSTEM)
        self.assertEqual(system, "unknown")

    def test_tool_span_uses_semconv_tool_name(self):
        call = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "bash", "arguments": {"command": "ls"}},
        )
        result = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            parent_id=call.event_id,
            data={"tool_name": "bash", "result": "file.txt"},
        )
        meta = _make_session(self.store, events=[call, result])
        payload = session_to_otlp(meta, [call, result])
        spans = _get_spans(payload)
        tool_spans = [s for s in spans if s["name"].startswith("tool/")]
        self.assertTrue(len(tool_spans) > 0)
        tool_name = _attr_value(tool_spans[0]["attributes"], _SEMCONV_TOOL_NAME)
        self.assertEqual(tool_name, "bash")

    def test_tool_span_kind_is_client(self):
        call = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "read", "arguments": {}},
        )
        result = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            parent_id=call.event_id,
            data={"tool_name": "read", "result": "content"},
        )
        meta = _make_session(self.store, events=[call, result])
        payload = session_to_otlp(meta, [call, result])
        spans = _get_spans(payload)
        tool_spans = [s for s in spans if s["name"].startswith("tool/")]
        self.assertTrue(len(tool_spans) > 0)
        self.assertEqual(tool_spans[0]["kind"], 3)  # SPAN_KIND_CLIENT

    def test_parent_span_id_set_for_subagent(self):
        meta = _make_session(self.store)
        payload = session_to_otlp(
            meta, [],
            parent_span_id="abcdef1234567890",
            parent_trace_id="a" * 32,
        )
        spans = _get_spans(payload)
        root = spans[0]
        self.assertEqual(root.get("parentSpanId"), "abcdef1234567890")
        self.assertEqual(root["traceId"], "a" * 32)

    def test_no_parent_span_id_for_root(self):
        meta = _make_session(self.store)
        payload = session_to_otlp(meta, [])
        spans = _get_spans(payload)
        root = spans[0]
        self.assertNotIn("parentSpanId", root)


if __name__ == "__main__":
    unittest.main()
