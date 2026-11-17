"""Tests for issue #45: A2A protocol support."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


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

    def test_otlp_call_span_status_is_unset_on_success_and_error_on_failure(self):
        # Recording Errors: instrumentation leaves status unset on success.
        from agent_trace.a2a import build_a2a_tree, a2a_calls_to_otlp_spans, make_a2a_event
        store = _make_store(self._tmp)
        sid = self._make_session(store, "orchestrator")
        for agent_id, success in (("ok-agent", True), ("bad-agent", False)):
            store.append_event(sid, make_a2a_event(
                session_id=sid, agent_id=agent_id, agent_url="http://agent",
                task="t", response={}, success=success,
            ))
        spans = a2a_calls_to_otlp_spans(build_a2a_tree(store, sid))
        statuses = {s["name"]: s.get("status") for s in spans}
        self.assertEqual(statuses["a2a_call:ok-agent"], {"code": 0})
        self.assertEqual(statuses["a2a_call:bad-agent"], {"code": 2})

    def test_cli_has_a2a_tree_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["a2a-tree", "--format", "json"])
        self.assertEqual(args.format, "json")


def _a2a_session(trace_dir: str, fields: dict) -> str:
    """Write a session with one A2A call and return its ID.

    *fields* sets agent_name, tool_calls, agent_id and task. The rest are
    plain text.
    """
    from agent_trace.a2a import make_a2a_event
    store = TraceStore(trace_dir, redact=False)
    meta = SessionMeta(
        agent_name=fields.get("agent_name", "plain"),
        tool_calls=fields.get("tool_calls", 3),
    )
    store.create_session(meta)
    store.append_event(meta.session_id, make_a2a_event(
        session_id=meta.session_id, agent_id=fields.get("agent_id", "plain"),
        agent_url="http://agent", task=fields.get("task", "plain"), response={},
    ))
    return meta.session_id


def _render_a2a_tree(fields: dict) -> str:
    from agent_trace.a2a import build_a2a_tree, format_a2a_tree
    with tempfile.TemporaryDirectory() as tmp:
        trace_dir = os.path.join(tmp, "traces")
        sid = _a2a_session(trace_dir, fields)
        out = io.StringIO()
        format_a2a_tree(build_a2a_tree(TraceStore(trace_dir), sid), out=out)
        return out.getvalue()


# (label, kind, build): build(value) returns the fields of a session with one
# A2A call, with value in the place the label names. The kinds are in
# tests/_escapes.py.
A2A_TREE_CASES = [
    ("agent name", "text", lambda v: {"agent_name": v}),
    ("tool calls", "text", lambda v: {"tool_calls": v}),
    ("call agent ID", "text", lambda v: {"agent_id": v}),
    ("task", "free", lambda v: {"task": v}),
]

# (label, cut, build): a2a-tree keeps cut characters of the value the label names.
A2A_TREE_CUTS = [
    ("task", 60, lambda v: {"task": v}),
]


class TestA2ATreeShowsTraceTextAsText(unittest.TestCase):
    """An agent name comes from the session, and a called agent's ID and task
    come from the A2A traffic. Both can hold escape codes and line breaks.
    a2a-tree shows them as text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, A2A_TREE_CASES, _render_a2a_tree)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, A2A_TREE_CASES, _render_a2a_tree)

    def test_whitespace_in_free_text_shows_as_one_space(self):
        _escapes.check_whitespace_in_free_text(self, A2A_TREE_CASES, _render_a2a_tree)

    def test_value_as_long_as_the_cut_shows_in_full(self):
        _escapes.check_each_cut(self, A2A_TREE_CUTS, _render_a2a_tree)

    def test_line_break_in_agent_id_shows_as_escape(self):
        output = _render_a2a_tree({"agent_id": "planner\nreviewer"})
        self.assertIn("a2a → planner\\nreviewer  ", output)

    def test_sgr_code_shows_as_text(self):
        output = _render_a2a_tree({"agent_id": "a\x1b[8mhidden"})
        self.assertIn(r"a2a → a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)

    def test_json_keeps_the_raw_text(self):
        from agent_trace.a2a import cmd_a2a_tree
        payload = _escapes.PAYLOAD
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = os.path.join(tmp, "traces")
            sid = _a2a_session(
                trace_dir, {"agent_name": payload, "agent_id": payload, "task": payload},
            )
            args = argparse.Namespace(trace_dir=trace_dir, session_id=sid, format="json")
            with patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertEqual(cmd_a2a_tree(args), 0)
        values = {
            attribute["key"]: attribute["value"]
            for span in json.loads(out.getvalue())["spans"]
            for attribute in span["attributes"]
        }
        self.assertEqual(values["agent.id"]["stringValue"], payload)
        self.assertEqual(values["a2a.agent_id"]["stringValue"], payload)
        self.assertEqual(values["a2a.task"]["stringValue"], payload)
