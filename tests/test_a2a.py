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


class TestA2ATreeSharedAndCyclicSessions(unittest.TestCase):
    """A session reached twice is shown in full once. Only a session already
    on the path from the root is a cycle."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = _make_store(tmp.name)

    def _session(self, name):
        meta = SessionMeta(agent_name=name)
        self.store.create_session(meta)
        return meta.session_id

    def _call(self, sid, sub_sid, name, task="t"):
        from agent_trace.a2a import make_a2a_event
        self.store.append_event(sid, make_a2a_event(
            session_id=sid, agent_id=name, agent_url="http://agent",
            task=task, response={}, sub_session_id=sub_sid,
        ))

    def _render(self, report):
        from agent_trace.a2a import format_a2a_tree
        out = io.StringIO()
        format_a2a_tree(report, out=out)
        return out.getvalue()

    def _spans(self, report):
        from agent_trace.a2a import a2a_calls_to_otlp_spans
        spans = a2a_calls_to_otlp_spans(report)
        ids = [span["spanId"] for span in spans]
        self.assertEqual(len(ids), len(set(ids)), ids)
        return {span["name"]: span for span in spans}

    def _link(self, span):
        (link,) = span["links"]
        return {a["key"]: next(iter(a["value"].values())) for a in link["attributes"]}, link

    def test_two_calls_to_one_sub_session_show_it_once_with_a_count(self):
        from agent_trace.a2a import build_a2a_tree
        planner = self._session("planner")
        researcher = self._session("researcher")
        self._call(planner, researcher, "researcher", "first")
        self._call(planner, researcher, "researcher", "second")

        report = build_a2a_tree(self.store, planner)
        self.assertEqual(report.total_agents, 2)
        self.assertEqual(report.max_depth, 1)
        (child,) = report.root.children
        self.assertEqual((child.session_id, child.ref, child.delegations), (researcher, "", 2))

        text = self._render(report)
        self.assertNotIn("[cycle]", text)
        self.assertIn("Total agents: 2 ", text)
        self.assertIn("└── researcher  [0 calls · $0.0000 · ? · 2 delegations]", text)
        self.assertEqual(text.count("a2a → researcher"), 2)

        spans = self._spans(report)
        self.assertEqual(
            spans["agent:researcher"]["parentSpanId"], spans["agent:planner"]["spanId"],
        )
        delegations = {
            a["key"]: a["value"] for a in spans["agent:researcher"]["attributes"]
        }["agent.delegations"]
        self.assertEqual(delegations, {"intValue": 2})
        self.assertNotIn("links", spans["agent:researcher"])

    def test_diamond_shows_the_shared_session_in_full_once(self):
        from agent_trace.a2a import build_a2a_tree
        root, left, right, shared = (self._session(n) for n in ("root", "left", "right", "shared"))
        self._call(root, left, "left")
        self._call(root, right, "right")
        self._call(left, shared, "shared")
        self._call(right, shared, "shared")

        report = build_a2a_tree(self.store, root)
        self.assertEqual(report.total_agents, 4)
        self.assertEqual(report.max_depth, 2)
        left_node, right_node = report.root.children
        self.assertEqual(left_node.children[0].ref, "")
        self.assertEqual(right_node.children[0].ref, "shared")

        text = self._render(report)
        self.assertNotIn("[cycle]", text)
        self.assertIn("└── shared  [0 calls", text)
        self.assertIn("└── shared  [shown above]", text)

        spans = self._spans(report)
        self.assertEqual(
            spans["agent:shared"]["parentSpanId"], spans["agent:left"]["spanId"],
        )
        attributes, link = self._link(spans["agent:shared"])
        self.assertEqual(link["spanId"], spans["agent:right"]["spanId"])
        self.assertEqual(attributes, {"agent.link": "shared", "agent.delegations": 1})

    def test_real_cycle_is_still_reported(self):
        from agent_trace.a2a import build_a2a_tree
        a = self._session("alpha")
        b = self._session("beta")
        self._call(a, b, "beta")
        self._call(b, a, "alpha")

        report = build_a2a_tree(self.store, a)
        self.assertEqual(report.total_agents, 2)
        self.assertEqual(report.max_depth, 1)
        (cycle,) = report.root.children[0].children
        self.assertEqual((cycle.session_id, cycle.ref), (a, "cycle"))

        text = self._render(report)
        self.assertIn("└── [cycle] → alpha\n", text)

        spans = self._spans(report)
        self.assertEqual(len([n for n in spans if n.startswith("agent:")]), 2)
        attributes, link = self._link(spans["agent:alpha"])
        self.assertEqual(link["spanId"], spans["agent:beta"]["spanId"])
        self.assertEqual(attributes, {"agent.link": "cycle", "agent.delegations": 1})


class TestLinkSubSession(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = os.path.join(tmp.name, "traces")

    def test_depth_is_one_more_than_the_parents(self):
        from agent_trace.a2a import link_sub_session
        store = TraceStore(self.trace_dir)
        parent = SessionMeta(agent_name="parent", depth=1)
        child = SessionMeta(agent_name="child")
        store.create_session(parent)
        store.create_session(child)
        link_sub_session(store, parent.session_id, "evt", child.session_id)
        linked = store.load_meta(child.session_id)
        self.assertEqual(linked.parent_session_id, parent.session_id)
        self.assertEqual(linked.parent_event_id, "evt")
        self.assertEqual(linked.depth, 2)

    def test_parent_depth_that_is_not_a_count_still_links_at_depth_one(self):
        from agent_trace.a2a import link_sub_session
        store = TraceStore(self.trace_dir)
        for bad in ("2", -3, True, None):
            with self.subTest(depth=bad):
                parent = SessionMeta(agent_name="parent", depth=bad)
                child = SessionMeta(agent_name="child")
                store.create_session(parent)
                store.create_session(child)
                link_sub_session(store, parent.session_id, "", child.session_id)
                linked = store.load_meta(child.session_id)
                self.assertEqual(linked.parent_session_id, parent.session_id)
                self.assertEqual(linked.depth, 1)

    def test_keeps_the_childs_own_update_written_during_the_link(self):
        import threading
        from agent_trace.a2a import link_sub_session
        parent = SessionMeta(agent_name="parent")
        child = SessionMeta(agent_name="child")
        TraceStore(self.trace_dir).create_session(parent)
        TraceStore(self.trace_dir).create_session(child)

        def child_update(meta):
            meta.tool_calls += 4
            meta.ended_at = 123.0
            meta.total_duration_ms = 4500.0

        # The child agent's own process, with its own store.
        child_writer = threading.Thread(
            target=TraceStore(self.trace_dir).mutate_meta,
            args=(child.session_id, child_update),
        )

        class RacingStore(TraceStore):
            def load_meta(self, session_id):
                meta = super().load_meta(session_id)
                if session_id == child.session_id and child_writer.ident is None:
                    # The child writes after this read. It waits for the
                    # store lock if the link holds it.
                    child_writer.start()
                    child_writer.join(timeout=0.5)
                return meta

        store = RacingStore(self.trace_dir)
        link_sub_session(store, parent.session_id, "", child.session_id)
        child_writer.join(timeout=10)
        self.assertFalse(child_writer.is_alive())

        linked = store.load_meta(child.session_id)
        self.assertEqual(linked.parent_session_id, parent.session_id)
        self.assertEqual(linked.tool_calls, 4)
        self.assertEqual(linked.ended_at, 123.0)
        self.assertEqual(linked.total_duration_ms, 4500.0)
        self.assertEqual(linked.depth, 1)


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
