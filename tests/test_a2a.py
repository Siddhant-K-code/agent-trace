"""Tests for issue #45: A2A protocol support."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.event_data import UnusableTimestampError
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import RefusedSessionError, TraceStore

try:
    import _escapes
    import _no_telemetry  # noqa: F401
    from test_compliance import REFUSALS, SYMLINKS, link_events, refused_message
    from test_hash_chain import _CliCase
    from test_otlp import (
        LAST_FLOAT,
        PAST_FLOAT,
        PAST_LIMIT,
        UINT64_MAX,
        YEAR_3000,
        unix_nanos_values,
    )
except ImportError:
    from tests import _escapes
    from tests import _no_telemetry  # noqa: F401
    from tests.test_compliance import REFUSALS, SYMLINKS, link_events, refused_message
    from tests.test_hash_chain import _CliCase
    from tests.test_otlp import (
        LAST_FLOAT,
        PAST_FLOAT,
        PAST_LIMIT,
        UINT64_MAX,
        YEAR_3000,
        unix_nanos_values,
    )


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

    def test_nested_reply_before_parent_reply_keeps_absolute_depth(self):
        from agent_trace.a2a import link_sub_session
        from agent_trace.http_proxy import _link_parent
        store = TraceStore(self.trace_dir)
        root = SessionMeta(agent_name="root")
        child = SessionMeta(agent_name="child")
        grand = SessionMeta(agent_name="grandchild")
        for meta in (root, child, grand):
            store.create_session(meta)
        # Each proxy links its inbound tracestate parent first. Replies then
        # come back innermost first.
        _link_parent(store, child, root.session_id)
        _link_parent(store, grand, child.session_id)
        link_sub_session(store, child.session_id, "", grand.session_id)
        link_sub_session(store, root.session_id, "", child.session_id)
        c = store.load_meta(child.session_id)
        g = store.load_meta(grand.session_id)
        self.assertEqual((c.parent_session_id, c.depth), (root.session_id, 1))
        self.assertEqual((g.parent_session_id, g.depth), (child.session_id, 2))

    def test_linking_a_parent_later_moves_its_sub_sessions_down(self):
        # No tracestate came in, so the grandchild is linked before its
        # parent is.
        from agent_trace.a2a import link_sub_session
        store = TraceStore(self.trace_dir)
        root = SessionMeta(agent_name="root", depth=2)
        child = SessionMeta(agent_name="child")
        grand = SessionMeta(agent_name="grandchild")
        for meta in (root, child, grand):
            store.create_session(meta)
        link_sub_session(store, child.session_id, "", grand.session_id)
        self.assertEqual(store.load_meta(grand.session_id).depth, 1)
        link_sub_session(store, root.session_id, "", child.session_id)
        self.assertEqual(store.load_meta(child.session_id).depth, 3)
        self.assertEqual(store.load_meta(grand.session_id).depth, 4)

    def test_parent_linked_while_its_depth_is_read_still_counts(self):
        import threading
        from agent_trace.a2a import link_sub_session
        root = SessionMeta(agent_name="root")
        parent = SessionMeta(agent_name="parent")
        child = SessionMeta(agent_name="child")
        for meta in (root, parent, child):
            TraceStore(self.trace_dir).create_session(meta)

        # The root's proxy, in another process, links the parent.
        parent_linker = threading.Thread(
            target=link_sub_session,
            args=(TraceStore(self.trace_dir), root.session_id, "", parent.session_id),
        )

        class RacingStore(TraceStore):
            def load_meta(self, session_id):
                meta = super().load_meta(session_id)
                if session_id == parent.session_id and parent_linker.ident is None:
                    # The parent's link lands after its depth is read. It
                    # waits for the store lock if this link holds it.
                    parent_linker.start()
                    parent_linker.join(timeout=0.5)
                return meta

        link_sub_session(RacingStore(self.trace_dir), parent.session_id, "", child.session_id)
        self.assertIsNotNone(parent_linker.ident)
        parent_linker.join(timeout=10)
        self.assertFalse(parent_linker.is_alive())

        store = TraceStore(self.trace_dir)
        self.assertEqual(store.load_meta(parent.session_id).depth, 1)
        self.assertEqual(store.load_meta(child.session_id).depth, 2)


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

    def test_tool_calls_that_are_not_a_count_show_as_zero(self):
        # meta.json can hold text where the count belongs.
        self.assertEqual(
            _render_a2a_tree({"tool_calls": _escapes.PAYLOAD}),
            _render_a2a_tree({"tool_calls": 0}),
        )

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


class TestA2ATreeRefusedSessions(_CliCase):
    """a2a-tree refuses a session the store refuses. Only a session that is
    not in the store shows as "[not found]"."""

    @staticmethod
    def _session(store, name):
        meta = SessionMeta(agent_name=name)
        store.create_session(meta)
        return meta.session_id

    @staticmethod
    def _call(store, sid, sub_sid):
        from agent_trace.a2a import make_a2a_event
        store.append_event(sid, make_a2a_event(
            session_id=sid, agent_id="child", agent_url="http://agent",
            task="t", response={}, sub_session_id=sub_sid,
        ))

    def tree_store(self, name):
        """Build a store where a root session delegates to a child session."""
        store = TraceStore(str(self.root / name), use_workspace_env=False)
        root = self._session(store, "root")
        child = self._session(store, "child")
        self._call(store, root, child)
        return store, root, child

    def run_each(self, runs):
        """Run each (args, trace_dir) pair in its own process."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(lambda run: self.cli(*run[0], trace_dir=run[1]), runs))

    def test_build_a2a_tree_raises_for_a_refused_session(self):
        from agent_trace.a2a import build_a2a_tree
        for index, (kind, refuse, error) in enumerate(REFUSALS):
            for place in ("root", "delegated"):
                with self.subTest(kind, place=place):
                    store, root, child = self.tree_store(f"tree-{index}-{place}")
                    sid = root if place == "root" else child
                    reason = refuse(store, sid)
                    with self.assertRaises(error) as caught:
                        build_a2a_tree(store, root)
                    self.assertEqual(str(caught.exception), refused_message(sid, reason))

    def test_a2a_tree_refuses_a_refused_session(self):
        runs, expected = [], []
        for index, (kind, refuse, _) in enumerate(REFUSALS):
            for place in ("root", "delegated"):
                name = f"cli-{index}-{place}"
                store, root, child = self.tree_store(name)
                sid = root if place == "root" else child
                line = f"Error: {refused_message(sid, refuse(store, sid))}\n"
                for args in (("a2a-tree", root), ("a2a-tree", root, "--format", "json")):
                    runs.append((args, self.root / name))
                    expected.append((f"{kind} {place} {args[2:]}", line))
        for (label, line), run in zip(expected, self.run_each(runs)):
            with self.subTest(label):
                self.assertEqual((run.returncode, run.stdout, run.stderr), (1, "", line))

    def test_build_a2a_tree_raises_for_a_symlinked_session_path(self):
        from agent_trace.a2a import build_a2a_tree
        for index, (kind, link) in enumerate(SYMLINKS):
            for place in ("root", "delegated"):
                with self.subTest(kind, place=place):
                    store, root, child = self.tree_store(f"link-{index}-{place}")
                    sid = root if place == "root" else child
                    reason = link(store, sid)
                    with self.assertRaises(RefusedSessionError) as caught:
                        build_a2a_tree(store, root)
                    self.assertEqual(str(caught.exception), refused_message(sid, reason))

    def test_a2a_tree_refuses_a_symlinked_session_path(self):
        cases = [(kind, link, "delegated") for kind, link in SYMLINKS]
        # find_session skips a root whose directory or meta.json is a symlink.
        cases.append(("symlinked events.ndjson", link_events, "root"))
        runs, expected = [], []
        for index, (kind, link, place) in enumerate(cases):
            name = f"cli-link-{index}"
            store, root, child = self.tree_store(name)
            sid = root if place == "root" else child
            line = f"Error: {refused_message(sid, link(store, sid))}\n"
            for args in (("a2a-tree", root), ("a2a-tree", root, "--format", "json")):
                runs.append((args, self.root / name))
                expected.append((f"{kind} {place} {args[2:]}", line))
        for (label, line), run in zip(expected, self.run_each(runs)):
            with self.subTest(label):
                self.assertEqual((run.returncode, run.stdout, run.stderr), (1, "", line))

    def test_a_session_not_in_the_store_shows_as_not_found(self):
        # This passes on main too. Only a refused session stops a2a-tree.
        from agent_trace.a2a import build_a2a_tree
        root = self._session(self.store, "root")
        no_meta = "no-meta-4c1d"
        self.store._session_dir(no_meta).mkdir(parents=True)
        no_events = self._session(self.store, "no-events")
        (self.store._session_dir(no_events) / "events.ndjson").unlink()
        missing = ["no-such-session-4c1d", "a/b", no_meta, no_events]
        for sub_sid in missing:
            self._call(self.store, root, sub_sid)

        report = build_a2a_tree(self.store, root)
        self.assertEqual(
            [(child.session_id, child.agent_id) for child in report.root.children],
            [(sid, "[not found]") for sid in missing],
        )
        text, spans = self.cli_many([("a2a-tree", root), ("a2a-tree", root, "--format", "json")])
        for run in (text, spans):
            self.assertEqual((run.returncode, run.stderr), (0, ""))
        self.assertEqual(text.stdout.count("[not found]"), len(missing))
        names = [span["name"] for span in json.loads(spans.stdout)["spans"]]
        self.assertEqual(names.count("agent:[not found]"), len(missing))


_A2A_ROOT, _A2A_CHILD = "a2a-root", "a2a-child"
_T0 = 1_700_000_000.0

# The data of an a2a_call with each field of the type make_a2a_event writes.
_A2A_CALL = {
    "event_subtype": "a2a_call", "agent_id": "helper", "agent_url": "http://helper",
    "task": "check the build", "response": {"ok": True}, "sub_session_id": _A2A_CHILD,
    "duration_ms": 10.0, "cost_usd": 0.5, "success": True, "error": "",
}

# Values with the wrong type, by label.
_WRONG_VALUES = {
    "null": None, "true": True, "false": False, "0": 0, "1": 1, "-5": -5, "1.5": 1.5,
    "NaN": float("nan"), "Infinity": float("inf"), "10**400": 10 ** 400,
    '"10"': "10", '"x"': "x", '"false"': "false", "a list": ["x"], "an object": {"a": 1},
}
_WRONG_FOR_FIELD = {
    **dict.fromkeys(
        ("agent_id", "agent_url", "task", "sub_session_id", "error"),
        ("null", "true", "0", "-5", "1.5", "NaN", "Infinity", "10**400", "a list", "an object"),
    ),
    **dict.fromkeys(
        ("duration_ms", "cost_usd"),
        ('"10"', "null", "true", "false", "a list", "an object", "NaN", "Infinity", "-5", "10**400"),
    ),
    "success": ('"x"', '"false"', "null", "0", "1", "a list", "an object", "NaN", "Infinity", "10**400"),
    "response": ('"x"', "null", "a list", "true", "NaN"),
}


def _refuse_constant(name: str):
    raise ValueError(f"not JSON: {name}")


def _call_line(text: str) -> str:
    """Return what the text tree shows after "a2a → " on the call line."""
    return next(line for line in text.splitlines() if "a2a → " in line).split("a2a → ", 1)[1]


class TestA2ACallFieldTypes(_CliCase):
    """events.ndjson can hold any JSON value in an a2a_call field. a2a-tree
    reads a field with the wrong type as missing."""

    def tree(self, name: str, call: dict) -> Path:
        """Write a store where the root session makes one A2A call with *call*
        as its data. IDs and times are fixed, so two stores differ only in
        *call*."""
        trace_dir = self.root / name
        store = TraceStore(str(trace_dir), redact=False, use_workspace_env=False)
        sessions = (
            (_A2A_CHILD, "helper", [("session_start", {})]),
            # The session records its usage, so the size of the call does not
            # change its cost.
            (_A2A_ROOT, "main", [
                ("session_start", {}),
                ("tool_call", call),
                ("llm_response", {"input_tokens": 1000, "output_tokens": 100}),
            ]),
        )
        for sid, agent, events in sessions:
            store.create_session(SessionMeta(
                session_id=sid, agent_name=agent, started_at=_T0, ended_at=_T0 + 5,
                total_duration_ms=5000.0,
            ))
            # json.dumps writes NaN and Infinity as bare tokens, as a hand
            # edit or an older release can store them.
            lines = "".join(
                json.dumps({
                    "event_type": kind, "timestamp": _T0 + index,
                    "event_id": f"{sid}-{index}", "session_id": sid, "data": data,
                }) + "\n"
                for index, (kind, data) in enumerate(events)
            )
            (store._session_dir(sid) / "events.ndjson").write_bytes(lines.encode())
        return trace_dir

    def run_trees(self, calls: list[dict]) -> list[tuple]:
        """Run a2a-tree as text and as JSON on a store for each call.

        Returns a (text run, JSON run) pair for each call, in order.
        """
        runs = []
        for index, call in enumerate(calls):
            trace_dir = self.tree(f"tree-{index}", call)
            runs.append((("a2a-tree", _A2A_ROOT), trace_dir))
            runs.append((("a2a-tree", _A2A_ROOT, "--format", "json"), trace_dir))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            done = list(pool.map(lambda run: self.cli(*run[0], trace_dir=run[1]), runs))
        return list(zip(done[::2], done[1::2]))

    def assert_runs_cleanly(self, text, spans) -> None:
        for run in (text, spans):
            self.assertEqual((run.returncode, run.stderr), (0, ""))
        json.loads(spans.stdout, parse_constant=_refuse_constant)

    def test_a_field_with_the_wrong_type_reads_as_missing(self):
        fields = list(_WRONG_FOR_FIELD)
        cases = [(field, label) for field in fields for label in _WRONG_FOR_FIELD[field]]
        runs = self.run_trees(
            [{key: value for key, value in _A2A_CALL.items() if key != field} for field in fields]
            + [{**_A2A_CALL, field: _WRONG_VALUES[label]} for field, label in cases]
        )
        missing = dict(zip(fields, runs))
        for field, (text, spans) in missing.items():
            with self.subTest(field, value="missing"):
                self.assert_runs_cleanly(text, spans)
        # A missing field reads as before.
        self.assertEqual(_call_line(missing["duration_ms"][0].stdout), "helper  ✅  $0.5000  0ms")
        self.assertEqual(_call_line(missing["cost_usd"][0].stdout), "helper  ✅  $0.0000  10ms")
        self.assertEqual(_call_line(missing["agent_id"][0].stdout), "  ✅  $0.5000  10ms")

        for (field, label), (text, spans) in zip(cases, runs[len(fields):]):
            with self.subTest(field, value=label):
                self.assert_runs_cleanly(text, spans)
                expected_text, expected_spans = missing[field]
                self.assertEqual(text.stdout, expected_text.stdout)
                self.assertEqual(spans.stdout, expected_spans.stdout)

    # The guards below pass on main too.

    def test_success_false_marks_the_call_failed(self):
        [(text, spans)] = self.run_trees([{**_A2A_CALL, "success": False}])
        self.assert_runs_cleanly(text, spans)
        self.assertEqual(_call_line(text.stdout), "helper  ❌  $0.5000  10ms")
        statuses = {span["name"]: span.get("status") for span in json.loads(spans.stdout)["spans"]}
        self.assertEqual(statuses["a2a_call:helper"], {"code": 2})

    def test_usable_duration_and_cost_read_as_stored(self):
        [(text, spans)] = self.run_trees([{**_A2A_CALL, "duration_ms": 10, "cost_usd": 0.5}])
        self.assert_runs_cleanly(text, spans)
        self.assertEqual(_call_line(text.stdout), "helper  ✅  $0.5000  10ms")

    def test_sub_session_id_that_names_no_session_shows_not_found(self):
        [(text, spans)] = self.run_trees([{**_A2A_CALL, "sub_session_id": "no-such-session"}])
        self.assert_runs_cleanly(text, spans)
        self.assertIn("└── [not found]", text.stdout)
        names = [span["name"] for span in json.loads(spans.stdout)["spans"]]
        self.assertIn("agent:[not found]", names)


def _limit_line(sid: str, more: str = "") -> str:
    """Return the line a2a-tree --format json prints when the a2a_call of
    *sid* is past the latest time OTLP can carry."""
    return f"Error: session {sid} event timestamp {PAST_LIMIT}: event 3 of 4 (tool_call){more}\n"


def _string_timestamp_line(sid: str) -> str:
    """Return the line a2a-tree --format json prints when the a2a_call of
    *sid* has a string timestamp."""
    return (
        f"Error: session {sid} event timestamp must be a Unix time in seconds: "
        "event 3 of 4 (tool_call) has a string timestamp\n"
    )


def _start_line(sid: str) -> str:
    """Return the line a2a-tree --format json prints when the started_at of
    *sid* is not a Unix time in seconds."""
    return f"Error: session {sid} started_at must be a Unix time in seconds\n"


def _past_start_line(sid: str) -> str:
    """Return the line a2a-tree --format json prints when the started_at of
    *sid* is past the latest time OTLP can carry."""
    return f"Error: session {sid} started_at {PAST_LIMIT}\n"


# The start and end of each span of the tree _A2ATreeCase writes, by span
# ID: the agent span of each session, then its call span. An agent span
# starts at its session's started_at and lasts its total_duration_ms.
_ROOT_START_NANOS = 1_700_000_000_000_000_000
_CHILD_START_NANOS = 1_700_000_010_000_000_000
_ROOT_CALL_NANOS = 1_700_000_002_000_000_000
_CHILD_CALL_NANOS = 1_700_000_012_000_000_000
_SPANS = {
    _A2A_ROOT: (_ROOT_START_NANOS, _ROOT_START_NANOS + 12_000_000_000),
    f"{_A2A_ROOT}-2": (_ROOT_CALL_NANOS, 1_700_000_002_500_000_000),
    _A2A_CHILD: (_CHILD_START_NANOS, _CHILD_START_NANOS + 12_000_000_000),
    f"{_A2A_CHILD}-2": (_CHILD_CALL_NANOS, 1_700_000_012_500_000_000),
}
# A call at _NEAR that lasts 1000 ms would end past the limit. One that
# lasts 500 ms ends in range.
_NEAR = 18446744073.0
_NEAR_NANOS = 18_446_744_072_999_999_488
_LAST_NANOS = 18_446_744_073_709_549_568


class _A2ATreeCase(_CliCase):
    """Runs a2a-tree on stores where the root session calls the child
    session, and the child makes a call of its own."""

    def setUp(self):
        super().setUp()
        self.trees = 0

    def tree(self, changes: dict | None = None) -> Path:
        """Write a store where the root session calls the child session, and
        the child makes a call of its own.

        Each session has four events: session_start, a tool_call, the
        a2a_call and session_end. *changes* maps a session ID to the values
        it changes: call_ts, duration_ms, other_ts, other_data,
        total_duration_ms, started_at, ended_at or calls. started_at and
        ended_at go into meta.json as given, so they can hold any JSON
        value. calls is a list of (sub_session_id, timestamp) pairs, with one
        a2a_call for each in place of the a2a_call.
        """
        self.trees += 1
        trace_dir = self.root / f"tree-{self.trees}"
        store = TraceStore(str(trace_dir), redact=False, use_workspace_env=False)
        for sid, agent, start, sub_sid in (
            (_A2A_CHILD, "helper", _T0 + 10, ""),
            (_A2A_ROOT, "main", _T0, _A2A_CHILD),
        ):
            change = (changes or {}).get(sid, {})
            store.create_session(SessionMeta(
                session_id=sid, agent_name=agent, started_at=start, ended_at=start + 12,
                total_duration_ms=change.get("total_duration_ms", 12000.0),
            ))
            times = {key: change[key] for key in ("started_at", "ended_at") if key in change}
            if times:
                # json.dumps writes NaN and Infinity as bare tokens.
                meta_path = store._session_dir(sid) / "meta.json"
                meta = {**json.loads(meta_path.read_bytes()), **times}
                meta_path.write_bytes(json.dumps(meta, allow_nan=True).encode())
            calls = change.get("calls", [(sub_sid, change.get("call_ts", start + 2))])
            duration = change.get("duration_ms", 500.0)
            other = {"tool_name": "Bash", "arguments": {"command": "ls"}}
            events = (
                ("session_start", start, {}),
                ("tool_call", change.get("other_ts", start + 1), change.get("other_data", other)),
                *(
                    ("tool_call", timestamp,
                     {**_A2A_CALL, "sub_session_id": sub, "duration_ms": duration})
                    for sub, timestamp in calls
                ),
                ("session_end", start + 3, {}),
            )
            lines = "".join(
                json.dumps({
                    "event_type": kind, "timestamp": timestamp,
                    "event_id": f"{sid}-{index}", "session_id": sid, "data": data,
                }) + "\n"
                for index, (kind, timestamp, data) in enumerate(events)
            )
            (store._session_dir(sid) / "events.ndjson").write_bytes(lines.encode())
        return trace_dir

    def run_trees(self, trees: list, fmt: str = "json") -> list:
        """Run a2a-tree on the root session of each store."""
        args = ("a2a-tree", _A2A_ROOT, "--format", fmt)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(lambda trace_dir: self.cli(*args, trace_dir=trace_dir), trees))

    def span_times(self, run) -> dict:
        """Check that *run* exited 0 and wrote every time as an int that OTLP
        can carry. Return the start and end of each span by span ID."""
        self.assertEqual((run.returncode, run.stderr), (0, ""))
        payload = json.loads(run.stdout)
        values = unix_nanos_values(payload)
        self.assertTrue(values)
        self.assertEqual([
            value for value in values
            if isinstance(value, bool) or not isinstance(value, int)
            or not 0 <= value <= UINT64_MAX
        ], [])
        return {
            span["spanId"]: (span["startTimeUnixNano"], span["endTimeUnixNano"])
            for span in payload["spans"]
        }

    def assert_refused(self, run, line: str) -> None:
        self.assertEqual((run.returncode, run.stdout, run.stderr), (1, "", line))


class TestA2ATreeOtlpTimeLimit(_A2ATreeCase):
    """OTLP carries each time as an unsigned 64-bit count of nanoseconds.
    a2a-tree --format json refuses an a2a_call timestamp past the latest time
    it can carry. A duration whose span would end past that time reads as
    missing."""

    def test_a_call_past_the_limit_is_refused(self):
        values = (YEAR_3000, 32503680000, PAST_FLOAT)
        runs = self.run_trees([self.tree({_A2A_ROOT: {"call_ts": value}}) for value in values])
        for value, run in zip(values, runs):
            with self.subTest(value=value):
                self.assert_refused(run, _limit_line(_A2A_ROOT))

    def test_a_call_at_the_last_time_in_range_is_written(self):
        control, last = self.run_trees(
            [self.tree(), self.tree({_A2A_ROOT: {"call_ts": LAST_FLOAT}})],
        )
        self.assertEqual(self.span_times(control), _SPANS)
        # 500 ms later is past the limit, so the span ends at its start.
        self.assertEqual(
            self.span_times(last), {**_SPANS, f"{_A2A_ROOT}-2": (_LAST_NANOS, _LAST_NANOS)},
        )

    def test_the_line_names_the_first_session_past_the_limit(self):
        past = {"call_ts": YEAR_3000}
        cases = [
            ("child", {_A2A_CHILD: past}, _limit_line(_A2A_CHILD)),
            ("root and child", {_A2A_ROOT: past, _A2A_CHILD: past}, _limit_line(_A2A_ROOT)),
            (
                "call and other event",
                {_A2A_ROOT: {**past, "other_ts": YEAR_3000}},
                _limit_line(_A2A_ROOT, "; 1 more event has a timestamp past that time"),
            ),
        ]
        runs = self.run_trees([self.tree(changes) for _, changes, _ in cases])
        for (label, _, line), run in zip(cases, runs):
            with self.subTest(label):
                self.assert_refused(run, line)

    def test_other_events_past_the_limit_are_not_read(self):
        past = {"other_ts": YEAR_3000}
        control, run = self.run_trees(
            [self.tree(), self.tree({_A2A_ROOT: past, _A2A_CHILD: past})],
        )
        self.assertEqual(self.span_times(run), _SPANS)
        self.assertEqual(run.stdout, control.stdout)

    def test_other_refusals_anywhere_in_the_tree_come_first(self):
        past = {"call_ts": YEAR_3000}
        cases = [
            (
                "child string timestamp",
                self.tree({_A2A_ROOT: past, _A2A_CHILD: {"call_ts": "x"}}),
                _string_timestamp_line(_A2A_CHILD),
            ),
            (
                "child array data",
                self.tree({_A2A_ROOT: past, _A2A_CHILD: {"other_data": [1]}}),
                f"Error: session {_A2A_CHILD} event data must be an object: "
                "event 2 of 4 (tool_call) has array data\n",
            ),
            (
                "root string timestamp",
                self.tree({_A2A_ROOT: {"call_ts": "x"}, _A2A_CHILD: past}),
                _string_timestamp_line(_A2A_ROOT),
            ),
        ]
        for kind, refuse, _ in REFUSALS:
            trace_dir = self.tree({_A2A_ROOT: past})
            reason = refuse(TraceStore(str(trace_dir), use_workspace_env=False), _A2A_CHILD)
            line = f"Error: {refused_message(_A2A_CHILD, reason)}\n"
            cases.append((f"child {kind}", trace_dir, line))
        runs = self.run_trees([trace_dir for _, trace_dir, _ in cases])
        for (label, _, line), run in zip(cases, runs):
            with self.subTest(label):
                self.assert_refused(run, line)

    def test_the_text_tree_does_not_read_call_times(self):
        control, run = self.run_trees(
            [self.tree(), self.tree({_A2A_ROOT: {"call_ts": YEAR_3000}})], "text",
        )
        for result in (control, run):
            self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertEqual(run.stdout, control.stdout)

    def test_a_call_duration_past_the_limit_reads_as_missing(self):
        cases = []
        for sid, start in ((_A2A_ROOT, _ROOT_CALL_NANOS), (_A2A_CHILD, _CHILD_CALL_NANOS)):
            for duration in (1e308, 1e14, 10**14):
                cases.append((sid, {"duration_ms": duration}, (start, start)))
            near = {"call_ts": _NEAR, "duration_ms": 1000.0}
            cases.append((sid, near, (_NEAR_NANOS, _NEAR_NANOS)))
            # These end in range, at the time main wrote.
            cases.append((sid, {"duration_ms": 1e12}, (start, start + 10**18)))
            near = {"call_ts": _NEAR, "duration_ms": 500.0}
            cases.append((sid, near, (_NEAR_NANOS, 18_446_744_073_500_000_256)))
        runs = self.run_trees([self.tree({sid: change}) for sid, change, _ in cases])
        for (sid, change, times), run in zip(cases, runs):
            with self.subTest(sid, **change):
                self.assertEqual(self.span_times(run), {**_SPANS, f"{sid}-2": times})

    def test_a_session_total_past_the_limit_reads_as_missing(self):
        # The span ends the total after its session's start, so the limit
        # moves with the start. A total past it ends the span at its start.
        cases = []
        for sid, start, past, last, last_end in (
            # The root's first total would end at exactly 2**64.
            (_A2A_ROOT, _ROOT_START_NANOS, 16746744073709.55, 16746744073709.549,
             18_446_744_073_709_549_568),
            (_A2A_CHILD, _CHILD_START_NANOS, 16746744063709.553, 16746744063709.55,
             18_446_744_073_709_550_592),
        ):
            for total in (1e308, 1e300, 1e14, 10**14, 2e13, past):
                cases.append((sid, total, (start, start)))
            # These end in range. The first is the last total that does.
            cases.append((sid, last, (start, last_end)))
            cases.append((sid, 1.6e13, (start, start + 16_000_000_000_000_000_000)))
            cases.append((sid, 0.009, (start, start + 9000)))
        runs = self.run_trees([
            self.tree({sid: {"total_duration_ms": total}}) for sid, total, _ in cases
        ])
        for (sid, total, times), run in zip(cases, runs):
            with self.subTest(sid, total=total):
                self.assertEqual(self.span_times(run), {**_SPANS, sid: times})


_A2A_GONE, _A2A_GONE_2, _A2A_ORPHAN = "a2a-gone", "a2a-gone-2", "a2a-orphan"


class TestA2ATreeAgentSpanStart(_A2ATreeCase):
    """a2a-tree --format json starts each agent span at its session's
    started_at. It refuses a started_at that is not a Unix time in seconds,
    or that OTLP cannot carry. A session that is not in the store starts
    where the first call that names it starts."""

    def test_each_agent_span_starts_where_export_starts_its_session(self):
        from agent_trace.otlp import _to_span_id
        int_starts = {
            _A2A_ROOT: {"started_at": 5_000_000_001, "ended_at": 5_000_000_013},
            _A2A_CHILD: {"started_at": 5_000_000_011, "ended_at": 5_000_000_023},
        }
        cases = [
            # The default tree has float starts. An int start converts exactly.
            (self.tree(), {_A2A_ROOT: _ROOT_START_NANOS, _A2A_CHILD: _CHILD_START_NANOS}),
            (self.tree(int_starts),
             {_A2A_ROOT: 5_000_000_001_000_000_000, _A2A_CHILD: 5_000_000_011_000_000_000}),
        ]
        exports = [
            (trace_dir, sid, self.root / f"{trace_dir.name}-{sid}.json")
            for trace_dir, _ in cases for sid in (_A2A_ROOT, _A2A_CHILD)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            done = list(pool.map(
                lambda export: self.cli(
                    "export", export[1], "--format", "otlp", "-o", str(export[2]),
                    trace_dir=export[0],
                ),
                exports,
            ))
        export_starts = {}
        for (trace_dir, sid, path), run in zip(exports, done):
            self.assertEqual(run.returncode, 0, run.stderr)
            payload = json.loads(path.read_bytes())
            spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
            root_id = _to_span_id(f"root-{sid}")
            root_span = next(span for span in spans if span["spanId"] == root_id)
            export_starts[trace_dir, sid] = int(root_span["startTimeUnixNano"])
        runs = self.run_trees([trace_dir for trace_dir, _ in cases])
        for (trace_dir, starts), run in zip(cases, runs):
            spans = self.span_times(run)
            for sid, start in starts.items():
                with self.subTest(trace_dir.name, sid=sid):
                    self.assertEqual(spans[sid][0], export_starts[trace_dir, sid])
                    self.assertEqual(spans[sid][0], start)

    def test_a_start_otlp_can_carry_is_written(self):
        cases = [
            (0, (0, 12_000_000_000)),
            (1_700_000_100, (1_700_000_100_000_000_000, 1_700_000_112_000_000_000)),
            (1_700_000_000.123456789, (1_700_000_000_123_456_768, 1_700_000_012_123_456_768)),
            # An int start converts exactly. As a float it would be 512 ns less.
            (5_000_000_001, (5_000_000_001_000_000_000, 5_000_000_013_000_000_000)),
            # 12000 ms later is past the limit, so the span ends at its start.
            (LAST_FLOAT, (_LAST_NANOS, _LAST_NANOS)),
        ]
        runs = self.run_trees([
            self.tree({_A2A_ROOT: {"started_at": start}}) for start, _ in cases
        ])
        for (start, times), run in zip(cases, runs):
            with self.subTest(started_at=start):
                self.assertEqual(self.span_times(run), {**_SPANS, _A2A_ROOT: times})

    def test_an_unusable_start_is_refused(self):
        values = ("x", None, True, False, [], {}, -1, float("nan"), float("inf"), 1e12)
        cases = [(sid, value) for sid in (_A2A_ROOT, _A2A_CHILD) for value in values]
        runs = self.run_trees([self.tree({sid: {"started_at": value}}) for sid, value in cases])
        for (sid, value), run in zip(cases, runs):
            with self.subTest(sid, started_at=value):
                self.assert_refused(run, _start_line(sid))

    def test_a_start_past_the_limit_is_refused(self):
        values = (PAST_FLOAT, YEAR_3000)
        cases = [(sid, value) for sid in (_A2A_ROOT, _A2A_CHILD) for value in values]
        runs = self.run_trees([self.tree({sid: {"started_at": value}}) for sid, value in cases])
        for (sid, value), run in zip(cases, runs):
            with self.subTest(sid, started_at=value):
                self.assert_refused(run, _past_start_line(sid))

    def test_the_line_names_the_first_refused_time(self):
        # An unusable time anywhere in the tree comes before a past one. In
        # one session, an unusable call comes before an unusable start, and a
        # past start before a past call.
        bad, past = "x", YEAR_3000
        cases = [
            ("root call and start", {_A2A_ROOT: {"call_ts": bad, "started_at": bad}},
             _string_timestamp_line(_A2A_ROOT)),
            ("root start and call past", {_A2A_ROOT: {"started_at": past, "call_ts": past}},
             _past_start_line(_A2A_ROOT)),
            ("root start past, child bad",
             {_A2A_ROOT: {"started_at": past}, _A2A_CHILD: {"started_at": bad}},
             _start_line(_A2A_CHILD)),
            ("root call past, child bad",
             {_A2A_ROOT: {"call_ts": past}, _A2A_CHILD: {"started_at": bad}},
             _start_line(_A2A_CHILD)),
            ("both bad", {_A2A_ROOT: {"started_at": bad}, _A2A_CHILD: {"started_at": bad}},
             _start_line(_A2A_ROOT)),
            ("root bad, child call bad",
             {_A2A_ROOT: {"started_at": bad}, _A2A_CHILD: {"call_ts": bad}},
             _start_line(_A2A_ROOT)),
            ("both past", {_A2A_ROOT: {"started_at": past}, _A2A_CHILD: {"started_at": past}},
             _past_start_line(_A2A_ROOT)),
            ("root call past, child past",
             {_A2A_ROOT: {"call_ts": past}, _A2A_CHILD: {"started_at": past}},
             _limit_line(_A2A_ROOT)),
        ]
        runs = self.run_trees([self.tree(changes) for _, changes, _ in cases])
        for (label, _, line), run in zip(cases, runs):
            with self.subTest(label):
                self.assert_refused(run, line)

    def test_the_text_tree_does_not_read_starts(self):
        values = ("x", None, float("nan"), 1e12, YEAR_3000)
        cases = [(sid, value) for sid in (_A2A_ROOT, _A2A_CHILD) for value in values]
        control, *runs = self.run_trees(
            [self.tree()] + [self.tree({sid: {"started_at": value}}) for sid, value in cases],
            "text",
        )
        self.assertEqual((control.returncode, control.stderr), (0, ""))
        for (sid, value), run in zip(cases, runs):
            with self.subTest(sid, started_at=value):
                self.assertEqual((run.returncode, run.stderr, run.stdout), (0, "", control.stdout))

    def test_ended_at_is_not_read(self):
        values = ("x", None, -1, float("nan"), 1e12, YEAR_3000, 0)
        cases = [(sid, value) for sid in (_A2A_ROOT, _A2A_CHILD) for value in values]
        control, *runs = self.run_trees(
            [self.tree()] + [self.tree({sid: {"ended_at": value}}) for sid, value in cases],
        )
        self.assertEqual(self.span_times(control), _SPANS)
        for (sid, value), run in zip(cases, runs):
            with self.subTest(sid, ended_at=value):
                self.assertEqual(self.span_times(run), _SPANS)
                self.assertEqual(run.stdout, control.stdout)

    def test_a_session_not_in_the_store_starts_at_the_first_call_that_names_it(self):
        [run] = self.run_trees([self.tree({
            # The first call that names _A2A_GONE is neither the earliest nor
            # the last.
            _A2A_ROOT: {"calls": [
                (_A2A_CHILD, _T0 + 2), (_A2A_GONE, _T0 + 5), (_A2A_GONE, _T0 + 1),
            ]},
            _A2A_CHILD: {"calls": [(_A2A_GONE_2, _T0 + 12)]},
        })])
        gone = 1_700_000_005_000_000_000
        self.assertEqual(self.span_times(run), {
            **_SPANS,
            f"{_A2A_ROOT}-3": (gone, 1_700_000_005_500_000_000),
            f"{_A2A_ROOT}-4": (1_700_000_001_000_000_000, 1_700_000_001_500_000_000),
            # A session that is not in the store lasts 0 ms.
            _A2A_GONE: (gone, gone),
            _A2A_GONE_2: (_CHILD_CALL_NANOS, _CHILD_CALL_NANOS),
        })
        names = {span["spanId"]: span["name"] for span in json.loads(run.stdout)["spans"]}
        self.assertEqual((names[_A2A_GONE], names[_A2A_GONE_2]), ("agent:[not found]",) * 2)

    def test_a_session_no_call_names_starts_where_its_parent_starts(self):
        trace_dir = self.tree()
        store = TraceStore(str(trace_dir), redact=False, use_workspace_env=False)
        # It names the root as its parent. With no events.ndjson it shows as
        # [not found], so its own started_at is not read.
        store.create_session(SessionMeta(
            session_id=_A2A_ORPHAN, agent_name="orphan", started_at=_T0 + 3,
            parent_session_id=_A2A_ROOT,
        ))
        (store._session_dir(_A2A_ORPHAN) / "events.ndjson").unlink(missing_ok=True)
        [run] = self.run_trees([trace_dir])
        self.assertEqual(
            self.span_times(run),
            {**_SPANS, _A2A_ORPHAN: (_ROOT_START_NANOS, _ROOT_START_NANOS)},
        )
        names = {span["spanId"]: span["name"] for span in json.loads(run.stdout)["spans"]}
        self.assertEqual(names[_A2A_ORPHAN], "agent:[not found]")


class TestA2ASpansOtlpTimeLimit(unittest.TestCase):
    """a2a_calls_to_otlp_spans writes each span time as an int from 0 to
    2**64 - 1, or refuses the call timestamp."""

    @staticmethod
    def times(timestamp=2, call_ms=500.0, node_ms=0.0, started_at=None) -> list:
        """Return the start and end of each span of a tree with one session
        that makes one call. The session has no start unless *started_at*
        gives one, so its span starts at 0."""
        from agent_trace.a2a import (
            A2ACallEvent,
            A2ATreeReport,
            AgentNode,
            a2a_calls_to_otlp_spans,
        )
        call = A2ACallEvent(
            event_id="call-0001-abcdef", session_id="d4e5f6a7b8c90004", timestamp=timestamp,
            agent_id="helper", agent_url="", task="t", response={}, sub_session_id="",
            duration_ms=call_ms, cost_usd=0.0, success=True,
        )
        node = AgentNode(
            session_id="d4e5f6a7b8c90004", agent_id="root", depth=0, cost_usd=0.0,
            duration_ms=node_ms, tool_calls=1, a2a_calls=[call], started_at=started_at,
        )
        report = A2ATreeReport(root=node, total_cost=0.0, total_agents=1, max_depth=0)
        return [
            (span["startTimeUnixNano"], span["endTimeUnixNano"])
            for span in a2a_calls_to_otlp_spans(report)
        ]

    def test_a_call_time_in_range_is_written(self):
        cases = [
            (2, (2_000_000_000, 2_500_000_000)),
            (1.5, (1_500_000_000, 2_000_000_000)),
            # 500 ms later is past the limit, so the span ends at its start.
            (LAST_FLOAT, (_LAST_NANOS, _LAST_NANOS)),
        ]
        for timestamp, times in cases:
            with self.subTest(timestamp=timestamp):
                self.assertEqual(self.times(timestamp), [(0, 0), times])

    def test_a_call_time_outside_the_range_is_refused(self):
        must_be = "timestamp must be a Unix time in seconds"
        cases = [
            (YEAR_3000, f"timestamp {PAST_LIMIT}"),
            (PAST_FLOAT, f"timestamp {PAST_LIMIT}"),
            (-5.0, must_be),
            (float("nan"), must_be),
            (float("inf"), must_be),
            (1e300, must_be),
        ]
        for timestamp, message in cases:
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(UnusableTimestampError) as caught:
                    self.times(timestamp)
                self.assertEqual(str(caught.exception), message)

    def test_an_unusable_call_duration_reads_as_0(self):
        for duration in (1e308, float("inf"), float("nan"), -5.0, True, "10"):
            with self.subTest(duration=duration):
                self.assertEqual(
                    self.times(call_ms=duration), [(0, 0), (2_000_000_000, 2_000_000_000)],
                )

    def test_an_unusable_session_duration_reads_as_0(self):
        # The span ends at its start. It starts at 5 s, so a span that ends
        # at 0 fails. 18446744073709.55 ms is 2**64 ns.
        for duration in (1e308, float("inf"), float("nan"), -5.0, True, 18446744073709.55, "10"):
            with self.subTest(duration=duration):
                self.assertEqual(
                    self.times(node_ms=duration, started_at=5),
                    [(5_000_000_000, 5_000_000_000), (2_000_000_000, 2_500_000_000)],
                )

    def test_a_usable_session_duration_is_written(self):
        cases = [
            (1234.567, 1_234_567_000),
            (0.0015, 1500),
            (0.009, 9000),
            (18446744073709.547, 18_446_744_073_709_547_520),
        ]
        for duration, end in cases:
            with self.subTest(duration=duration):
                self.assertEqual(
                    self.times(node_ms=duration), [(0, end), (2_000_000_000, 2_500_000_000)],
                )
