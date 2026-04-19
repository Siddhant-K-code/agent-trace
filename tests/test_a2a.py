"""Tests for issue #45: A2A protocol support."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store(tmp_dir: str) -> TraceStore:
    return TraceStore(os.path.join(tmp_dir, "traces"))


class TestA2ADetection(unittest.TestCase):
    def test_is_a2a_request_by_path(self):
        from agent_trace.a2a import is_a2a_request
        self.assertTrue(is_a2a_request("POST", "/a2a/tasks/send", {}, b"{}"))

    def test_is_a2a_request_by_header(self):
        from agent_trace.a2a import is_a2a_request
        self.assertTrue(is_a2a_request("POST", "/api/v1", {"x-a2a-session": "abc"}, b"{}"))

    def test_is_a2a_request_by_body_task(self):
        from agent_trace.a2a import is_a2a_request
        body = json.dumps({"task": "review this code"}).encode()
        self.assertTrue(is_a2a_request("POST", "/api/v1", {}, body))

    def test_is_a2a_request_jsonrpc(self):
        from agent_trace.a2a import is_a2a_request
        body = json.dumps({"jsonrpc": "2.0", "method": "tasks/send", "id": 1}).encode()
        self.assertTrue(is_a2a_request("POST", "/api/v1", {}, body))

    def test_not_a2a_regular_request(self):
        from agent_trace.a2a import is_a2a_request
        body = json.dumps({"query": "SELECT * FROM users"}).encode()
        self.assertFalse(is_a2a_request("GET", "/api/users", {}, body))

    def test_not_a2a_empty_body(self):
        from agent_trace.a2a import is_a2a_request
        self.assertFalse(is_a2a_request("GET", "/health", {}, b""))


class TestA2AEventCreation(unittest.TestCase):
    def test_make_a2a_event(self):
        from agent_trace.a2a import make_a2a_event, A2A_CALL
        event = make_a2a_event(
            session_id="sess123",
            agent_id="code-reviewer",
            agent_url="http://reviewer:8080",
            task="review auth.ts",
            response={"issues": []},
            duration_ms=420.0,
            cost_usd=0.08,
        )
        self.assertEqual(event.event_type, EventType.TOOL_CALL)
        self.assertEqual(event.data["event_subtype"], A2A_CALL)
        self.assertEqual(event.data["agent_id"], "code-reviewer")
        self.assertEqual(event.data["task"], "review auth.ts")
        self.assertEqual(event.data["duration_ms"], 420.0)

    def test_a2a_call_event_from_trace_event(self):
        from agent_trace.a2a import make_a2a_event, A2ACallEvent
        event = make_a2a_event(
            session_id="sess123",
            agent_id="tester",
            agent_url="http://tester:9000",
            task="run tests",
            response={"passed": 42},
            sub_session_id="sub456",
        )
        call = A2ACallEvent.from_trace_event(event)
        self.assertIsNotNone(call)
        self.assertEqual(call.agent_id, "tester")
        self.assertEqual(call.sub_session_id, "sub456")

    def test_a2a_call_event_from_non_a2a_event(self):
        from agent_trace.a2a import A2ACallEvent
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "read_file", "arguments": {}},
        )
        self.assertIsNone(A2ACallEvent.from_trace_event(event))


class TestA2ATree(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_session(self, store, agent_name="root"):
        meta = SessionMeta(agent_name=agent_name)
        store.create_session(meta)
        return meta.session_id

    def test_build_tree_single_session(self):
        from agent_trace.a2a import build_a2a_tree
        store = _make_store(self._tmp)
        sid = self._make_session(store, "orchestrator")
        store.append_event(sid, TraceEvent(
            event_type=EventType.SESSION_START, session_id=sid, data={}
        ))
        report = build_a2a_tree(store, sid)
        self.assertEqual(report.root.session_id, sid)
        self.assertEqual(report.total_agents, 1)
        self.assertEqual(report.max_depth, 0)

    def test_build_tree_with_linked_child(self):
        from agent_trace.a2a import build_a2a_tree, make_a2a_event
        store = _make_store(self._tmp)
        parent_sid = self._make_session(store, "orchestrator")
        child_sid = self._make_session(store, "reviewer")

        # Link child to parent
        child_meta = store.load_meta(child_sid)
        child_meta.parent_session_id = parent_sid
        store.update_meta(child_meta)

        store.append_event(parent_sid, make_a2a_event(
            session_id=parent_sid,
            agent_id="reviewer",
            agent_url="http://reviewer",
            task="review code",
            response={},
            sub_session_id=child_sid,
        ))

        report = build_a2a_tree(store, parent_sid)
        self.assertGreaterEqual(report.total_agents, 1)

    def test_format_a2a_tree(self):
        from agent_trace.a2a import build_a2a_tree, format_a2a_tree
        store = _make_store(self._tmp)
        sid = self._make_session(store, "orchestrator")
        report = build_a2a_tree(store, sid)
        buf = io.StringIO()
        format_a2a_tree(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Agent Call Graph", output)
        self.assertIn("orchestrator", output)

    def test_otlp_spans_generated(self):
        from agent_trace.a2a import build_a2a_tree, a2a_calls_to_otlp_spans
        store = _make_store(self._tmp)
        sid = self._make_session(store, "orchestrator")
        report = build_a2a_tree(store, sid)
        spans = a2a_calls_to_otlp_spans(report)
        self.assertGreater(len(spans), 0)
        self.assertIn("spanId", spans[0])
        self.assertIn("name", spans[0])

    def test_cli_has_a2a_tree_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["a2a-tree", "--format", "json"])
        self.assertEqual(args.format, "json")


