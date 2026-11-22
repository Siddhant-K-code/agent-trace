"""Tests for OTLP improvements: span hierarchy and semantic conventions (issue #25)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace import __version__
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import (
    _SEMCONV_SYSTEM,
    _SEMCONV_TOOL_NAME,
    _SEMCONV_OP,
    _to_trace_id,
    node_tree_to_otlp,
    session_to_otlp,
    session_to_otlp_genai,
    tree_to_otlp,
    tree_to_otlp_genai,
)
from agent_trace.store import TraceStore
from agent_trace.subagent import SessionNode


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

    def test_service_version_is_the_agent_trace_version(self):
        meta = _make_session(self.store)
        for payload in (
            session_to_otlp(meta, []),
            tree_to_otlp(self.store, meta.session_id),
            session_to_otlp_genai(meta, []),
            tree_to_otlp_genai(self.store, meta.session_id),
        ):
            resource = payload["resourceSpans"][0]["resource"]["attributes"]
            self.assertEqual(_attr_value(resource, "service.version"), __version__)


class TestSubagentTreeExport(unittest.TestCase):
    """tree_to_otlp and tree_to_otlp_genai export a session and its subagents
    as one trace, each subagent's root under the span that spawned it (#552)."""

    TREES = (("classic", tree_to_otlp, "tool/Task"), ("genai", tree_to_otlp_genai, "execute_tool Task"))

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name)
        self.parent = SessionMeta(agent_name="claude-code", started_at=1000.0, ended_at=1100.0)
        self.store.create_session(self.parent)
        for event in (
            TraceEvent(event_type=EventType.USER_PROMPT, timestamp=1001.0, data={"prompt": "Review it"}),
            TraceEvent(event_type=EventType.TOOL_CALL, event_id="spawn1", timestamp=1002.0,
                       data={"tool_name": "Task", "arguments": {"prompt": "read the diff"}}),
            TraceEvent(event_type=EventType.TOOL_RESULT, parent_id="spawn1", timestamp=1050.0,
                       data={"result": "looks fine"}),
        ):
            event.session_id = self.parent.session_id
            self.store.append_event(self.parent.session_id, event)
        self.child = SessionMeta(
            agent_name="claude-code", started_at=1003.0, ended_at=1049.0,
            parent_session_id=self.parent.session_id, parent_event_id="spawn1", depth=1,
        )
        self.store.create_session(self.child)
        for event in (
            TraceEvent(event_type=EventType.TOOL_CALL, event_id="read1", timestamp=1004.0,
                       data={"tool_name": "Read", "arguments": {"file_path": "a.py"}}),
            TraceEvent(event_type=EventType.TOOL_RESULT, parent_id="read1", timestamp=1005.0,
                       data={"result": "print(1)"}),
        ):
            event.session_id = self.child.session_id
            self.store.append_event(self.child.session_id, event)

    def _set_parent_event_id(self, value):
        path = os.path.join(self.store.base_dir, self.child.session_id, "meta.json")
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["parent_event_id"] = value
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _roots(self, spans):
        return [s for s in spans if s["name"].startswith(("agent-session", "invoke_agent"))]

    def test_one_trace_with_the_subagent_under_its_spawning_span(self):
        for name, export, spawn_name in self.TREES:
            with self.subTest(export=name):
                payload = export(self.store, self.parent.session_id)
                spans = _get_spans(payload)
                self.assertEqual({s["traceId"] for s in spans}, {_to_trace_id(self.parent.session_id)})
                span_ids = [s["spanId"] for s in spans]
                self.assertEqual(len(set(span_ids)), len(span_ids))
                self.assertTrue(all(s.get("parentSpanId", span_ids[0]) in span_ids for s in spans))
                parent_root, child_root = self._roots(spans)
                self.assertNotIn("parentSpanId", parent_root)
                (spawn,) = [s for s in spans if s["name"] == spawn_name]
                self.assertEqual(child_root["parentSpanId"], spawn["spanId"])
                self.assertEqual(spawn["parentSpanId"], parent_root["spanId"])
                resource = payload["resourceSpans"][0]["resource"]["attributes"]
                self.assertEqual(_attr_value(resource, "agent.session_id"), self.parent.session_id)

    def test_scope_is_the_single_session_scope(self):
        for name, export, convert in (
            ("classic", tree_to_otlp, session_to_otlp),
            ("genai", tree_to_otlp_genai, session_to_otlp_genai),
        ):
            with self.subTest(export=name):
                tree = export(self.store, self.parent.session_id)["resourceSpans"][0]
                single = convert(self.parent, [])["resourceSpans"][0]
                self.assertEqual(tree["scopeSpans"][0]["scope"], single["scopeSpans"][0]["scope"])
                self.assertEqual(tree["resource"], single["resource"])

    def test_unusable_parent_event_id_parents_to_the_parent_root(self):
        # A value that is not a non-empty string, or that names no span of
        # the parent, is read as missing. Classic tree_to_otlp used to emit a
        # parentSpanId no span had, or raise on a list or an object.
        for value in (["spawn1"], {"id": "spawn1"}, 5, True, "", "missing", "read1"):
            self._set_parent_event_id(value)
            for name, export, _ in self.TREES:
                with self.subTest(value=value, export=name):
                    spans = _get_spans(export(self.store, self.parent.session_id))
                    parent_root, child_root = self._roots(spans)
                    self.assertEqual(child_root["parentSpanId"], parent_root["spanId"])

    def test_session_in_a_parent_link_cycle_is_exported_once(self):
        parent = SessionNode(meta=self.parent, events=self.store.load_events(self.parent.session_id))
        child = SessionNode(meta=self.child, events=self.store.load_events(self.child.session_id))
        parent.children.append(child)
        child.children.append(parent)
        for genai in (False, True):
            with self.subTest(genai=genai):
                spans = _get_spans(node_tree_to_otlp(parent, genai=genai))
                self.assertEqual(len(self._roots(spans)), 2)
                span_ids = [s["spanId"] for s in spans]
                self.assertEqual(len(set(span_ids)), len(span_ids))


if __name__ == "__main__":
    unittest.main()
