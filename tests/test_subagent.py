"""Tests for subagent tracing."""

import io
import json
import argparse
import tempfile
import unittest
from unittest.mock import patch

from agent_trace.cli import (
    build_parser,
    cmd_stats,
    _parent_depth,
    _parent_session_id,
    _resolve_parent_session_id,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.subagent import (
    AggregatedStats,
    SessionNode,
    aggregate_stats,
    build_tree,
    cmd_tree,
    format_tree,
    format_session_tree,
    format_tree_summary,
    in_session_subagents,
    tree_to_dict,
)


def _make_event(event_type: EventType, ts: float, session_id: str, **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)


def _make_store_with_sessions(sessions: list[tuple[SessionMeta, list[TraceEvent]]]) -> tuple[TraceStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = TraceStore(tmp.name)
    for meta, events in sessions:
        store.create_session(meta)
        for e in events:
            store.append_event(meta.session_id, e)
        store.update_meta(meta)
    return store, tmp


class TestSessionMetaSubagentFields(unittest.TestCase):
    def test_default_values(self):
        meta = SessionMeta(session_id="root1")
        self.assertEqual(meta.parent_session_id, "")
        self.assertEqual(meta.parent_event_id, "")
        self.assertEqual(meta.depth, 0)

    def test_subagent_fields_serialized(self):
        meta = SessionMeta(
            session_id="child1",
            parent_session_id="root1",
            parent_event_id="evt123",
            depth=1,
        )
        json_str = meta.to_json()
        restored = SessionMeta.from_json(json_str)
        self.assertEqual(restored.parent_session_id, "root1")
        self.assertEqual(restored.parent_event_id, "evt123")
        self.assertEqual(restored.depth, 1)

    def test_zero_depth_omitted_from_json(self):
        meta = SessionMeta(session_id="root1")
        d = json.loads(meta.to_json())
        # depth=0 should be omitted (zero value)
        self.assertNotIn("depth", d)

    def test_nonzero_depth_included_in_json(self):
        meta = SessionMeta(session_id="child1", depth=2)
        d = json.loads(meta.to_json())
        self.assertEqual(d["depth"], 2)


class TestBuildTree(unittest.TestCase):
    def _make_tree(self):
        """Create a root session with one child subagent."""
        root_meta = SessionMeta(
            session_id="root0001",
            started_at=0.0,
            tool_calls=5,
            llm_requests=3,
            total_tokens=1000,
            total_duration_ms=5000,
        )
        child_meta = SessionMeta(
            session_id="child001",
            started_at=1.0,
            parent_session_id="root0001",
            parent_event_id="evt_agent",
            depth=1,
            tool_calls=2,
            llm_requests=1,
            total_tokens=400,
            total_duration_ms=2000,
        )

        root_events = [
            _make_event(EventType.USER_PROMPT, 0.0, "root0001", prompt="do something"),
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=1.0,
                       event_id="evt_agent", session_id="root0001",
                       data={"tool_name": "Agent", "arguments": {"prompt": "subtask"}}),
            _make_event(EventType.SESSION_END, 5.0, "root0001"),
        ]
        child_events = [
            _make_event(EventType.TOOL_CALL, 1.5, "child001",
                        tool_name="Bash", arguments={"command": "ls"}),
            _make_event(EventType.SESSION_END, 3.0, "child001"),
        ]
        store, tmp = _make_store_with_sessions([
            (root_meta, root_events),
            (child_meta, child_events),
        ])
        return store, tmp

    def test_root_has_one_child(self):
        store, tmp = self._make_tree()
        tree = build_tree(store, "root0001")
        self.assertEqual(len(tree.children), 1)
        tmp.cleanup()

    def test_child_session_id(self):
        store, tmp = self._make_tree()
        tree = build_tree(store, "root0001")
        self.assertEqual(tree.children[0].meta.session_id, "child001")
        tmp.cleanup()

    def test_root_has_no_parent(self):
        store, tmp = self._make_tree()
        tree = build_tree(store, "root0001")
        self.assertEqual(tree.meta.parent_session_id, "")
        tmp.cleanup()

    def test_child_depth(self):
        store, tmp = self._make_tree()
        tree = build_tree(store, "root0001")
        self.assertEqual(tree.children[0].depth, 1)
        tmp.cleanup()

    def test_single_session_no_children(self):
        meta = SessionMeta(session_id="solo001", started_at=0.0)
        events = [_make_event(EventType.SESSION_END, 1.0, "solo001")]
        store, tmp = _make_store_with_sessions([(meta, events)])
        tree = build_tree(store, "solo001")
        self.assertEqual(len(tree.children), 0)
        tmp.cleanup()


class TestAggregateStats(unittest.TestCase):
    def test_single_node(self):
        meta = SessionMeta(
            session_id="s1",
            tool_calls=3,
            llm_requests=2,
            errors=1,
            total_tokens=500,
            total_duration_ms=3000,
        )
        node = SessionNode(meta=meta, events=[])
        stats = aggregate_stats(node)
        self.assertEqual(stats.session_count, 1)
        self.assertEqual(stats.tool_calls, 3)
        self.assertEqual(stats.llm_requests, 2)
        self.assertEqual(stats.errors, 1)
        self.assertEqual(stats.total_tokens, 500)

    def test_rolls_up_children(self):
        root_meta = SessionMeta(session_id="r1", tool_calls=5, llm_requests=3,
                                total_tokens=1000, total_duration_ms=5000)
        child_meta = SessionMeta(session_id="c1", tool_calls=2, llm_requests=1,
                                 total_tokens=400, total_duration_ms=2000, depth=1)
        child_node = SessionNode(meta=child_meta, events=[])
        root_node = SessionNode(meta=root_meta, events=[], children=[child_node])

        stats = aggregate_stats(root_node)
        self.assertEqual(stats.session_count, 2)
        self.assertEqual(stats.tool_calls, 7)
        self.assertEqual(stats.llm_requests, 4)
        self.assertEqual(stats.total_tokens, 1400)

    def test_duration_uses_max_not_sum(self):
        root_meta = SessionMeta(session_id="r1", total_duration_ms=5000)
        child_meta = SessionMeta(session_id="c1", total_duration_ms=3000, depth=1)
        child_node = SessionNode(meta=child_meta, events=[])
        root_node = SessionNode(meta=root_meta, events=[], children=[child_node])

        stats = aggregate_stats(root_node)
        # Should be max(5000, 3000) = 5000, not 8000
        self.assertEqual(stats.total_duration_ms, 5000)


class TestFormatTree(unittest.TestCase):
    def _simple_node(self) -> SessionNode:
        meta = SessionMeta(session_id="abc123def456", agent_name="claude-code",
                           started_at=0.0)
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "abc123def456", prompt="hello"),
            _make_event(EventType.TOOL_CALL, 1.0, "abc123def456",
                        tool_name="Bash", arguments={"command": "ls"}),
            _make_event(EventType.SESSION_END, 2.0, "abc123def456"),
        ]
        return SessionNode(meta=meta, events=events)

    def test_output_contains_session_id(self):
        node = self._simple_node()
        buf = io.StringIO()
        format_tree(node, out=buf)
        self.assertIn("abc123def4", buf.getvalue())

    def test_output_contains_tool_call(self):
        node = self._simple_node()
        buf = io.StringIO()
        format_tree(node, out=buf)
        self.assertIn("tool_call", buf.getvalue())
        self.assertIn("Bash", buf.getvalue())

    def test_output_contains_user_prompt(self):
        node = self._simple_node()
        buf = io.StringIO()
        format_tree(node, out=buf)
        self.assertIn("hello", buf.getvalue())

    def test_summary_contains_session_id(self):
        node = self._simple_node()
        buf = io.StringIO()
        format_tree_summary(node, out=buf)
        self.assertIn("abc123def4", buf.getvalue())

    def test_last_child_uses_corner_connector(self):
        """Last child in a multi-child tree should use └─, not ├─."""
        root_meta = SessionMeta(session_id="root0002", started_at=0.0, depth=0)
        child_a = SessionMeta(session_id="childa002", started_at=1.0, depth=1)
        child_b = SessionMeta(session_id="childb002", started_at=2.0, depth=1)
        root_node = SessionNode(
            meta=root_meta,
            events=[],
            children=[
                SessionNode(meta=child_a, events=[]),
                SessionNode(meta=child_b, events=[]),
            ],
        )
        buf = io.StringIO()
        format_tree_summary(root_node, out=buf)
        output = buf.getvalue()
        self.assertIn("└─", output)
        self.assertIn("├─", output)

    def test_expand_false_does_not_inline_subagents(self):
        """format_tree with expand=False should not recurse into children."""
        root_meta = SessionMeta(session_id="root0003", started_at=0.0, depth=0)
        child_meta = SessionMeta(session_id="child003", started_at=1.0, depth=1)
        root_events = [
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=1.0,
                       event_id="evt_spawn", session_id="root0003",
                       data={"tool_name": "Agent", "arguments": {"prompt": "subtask"}}),
        ]
        child_node = SessionNode(meta=child_meta, events=[
            _make_event(EventType.SESSION_END, 2.0, "child003"),
        ])
        root_node = SessionNode(meta=root_meta, events=root_events,
                                children=[child_node])
        buf = io.StringIO()
        format_tree(root_node, out=buf, expand=False)
        self.assertNotIn("child003", buf.getvalue())


class TestSessionTreeCommand(unittest.TestCase):
    def _make_tree(self):
        root_meta = SessionMeta(
            session_id="rootcmd1",
            agent_name="orchestrator",
            started_at=0.0,
            tool_calls=3,
            llm_requests=1,
            total_duration_ms=4000,
        )
        child_meta = SessionMeta(
            session_id="childcmd1",
            agent_name="worker",
            started_at=1.0,
            parent_session_id="rootcmd1",
            depth=1,
            tool_calls=2,
            total_duration_ms=1500,
        )
        return _make_store_with_sessions([
            (root_meta, [_make_event(EventType.SESSION_END, 4.0, "rootcmd1")]),
            (child_meta, [_make_event(EventType.SESSION_END, 2.0, "childcmd1")]),
        ])

    def test_tree_to_dict_contains_children(self):
        store, tmp = self._make_tree()
        tree = build_tree(store, "rootcmd1")
        data = tree_to_dict(tree, {"rootcmd1": 0.01, "childcmd1": 0.02})
        self.assertEqual(data["session_id"], "rootcmd1")
        self.assertEqual(data["children"][0]["session_id"], "childcmd1")
        self.assertEqual(data["children"][0]["cost_usd"], 0.02)
        tmp.cleanup()

    def test_format_session_tree_contains_cost_and_child(self):
        store, tmp = self._make_tree()
        tree = build_tree(store, "rootcmd1")
        buf = io.StringIO()
        format_session_tree(tree, costs={"rootcmd1": 0.01, "childcmd1": 0.02}, out=buf)
        output = buf.getvalue()
        self.assertIn("rootcmd1", output)
        self.assertIn("childcmd1", output)
        self.assertIn("$0.0100", output)
        tmp.cleanup()

    def test_cmd_tree_json(self):
        store, tmp = self._make_tree()
        captured = io.StringIO()
        args = argparse.Namespace(
            trace_dir=store.base_dir,
            session_id="root",
            format="json",
        )
        with patch("sys.stdout", captured):
            result = cmd_tree(args)
        self.assertEqual(result, 0)
        data = json.loads(captured.getvalue())
        self.assertEqual(data["total_sessions"], 2)
        self.assertEqual(data["tree"]["children"][0]["session_id"], "childcmd1")
        tmp.cleanup()

    def test_parser_registers_tree_and_parent_flags(self):
        parser = build_parser()
        tree_args = parser.parse_args(["tree", "root", "--format", "json"])
        self.assertEqual(tree_args.command, "tree")
        self.assertEqual(tree_args.session_id, "root")

        record_args = parser.parse_args(["record", "--parent", "root123", "--", "server"])
        self.assertEqual(record_args.parent, "root123")

    def test_parent_session_id_prefers_flag_over_env(self):
        args = argparse.Namespace(parent="from-flag")
        with patch.dict("os.environ", {"AGENT_STRACE_PARENT_SESSION": "from-env"}):
            self.assertEqual(_parent_session_id(args), "from-flag")

    def test_parent_session_id_reads_env(self):
        args = argparse.Namespace(parent=None)
        with patch.dict("os.environ", {"AGENT_STRACE_PARENT_SESSION": "from-env"}):
            self.assertEqual(_parent_session_id(args), "from-env")

    def test_parent_depth_uses_parent_meta(self):
        tmp = tempfile.TemporaryDirectory()
        store = TraceStore(tmp.name)
        store.create_session(SessionMeta(session_id="parent1", depth=2))
        self.assertEqual(_parent_depth(store, "parent1"), 3)
        self.assertEqual(_parent_depth(store, "missing"), 1)
        tmp.cleanup()

    def test_resolve_parent_session_id_expands_prefix(self):
        tmp = tempfile.TemporaryDirectory()
        store = TraceStore(tmp.name)
        store.create_session(SessionMeta(session_id="parent123456"))
        args = argparse.Namespace(parent="parent")
        self.assertEqual(_resolve_parent_session_id(store, args), "parent123456")
        tmp.cleanup()


class TestAggregateStatsDuration(unittest.TestCase):
    def test_child_exceeds_root_duration(self):
        """Child with longer duration than root should set the aggregate."""
        root_meta = SessionMeta(session_id="r1", total_duration_ms=1000)
        child_meta = SessionMeta(session_id="c1", total_duration_ms=5000, depth=1)
        child_node = SessionNode(meta=child_meta, events=[])
        root_node = SessionNode(meta=root_meta, events=[], children=[child_node])
        stats = aggregate_stats(root_node)
        self.assertEqual(stats.total_duration_ms, 5000)

    def test_two_children_takes_max_regardless_of_order(self):
        """Duration should be the max across all children, not dependent on order."""
        root_meta = SessionMeta(session_id="r2", total_duration_ms=2000)
        child_a = SessionNode(meta=SessionMeta(session_id="ca", total_duration_ms=4000, depth=1), events=[])
        child_b = SessionNode(meta=SessionMeta(session_id="cb", total_duration_ms=3000, depth=1), events=[])
        # child_a first (larger), child_b second (smaller)
        root_node = SessionNode(meta=root_meta, events=[], children=[child_a, child_b])
        stats = aggregate_stats(root_node)
        self.assertEqual(stats.total_duration_ms, 4000)

        # Reverse order: child_b first (smaller), child_a second (larger)
        root_node2 = SessionNode(meta=root_meta, events=[], children=[child_b, child_a])
        stats2 = aggregate_stats(root_node2)
        self.assertEqual(stats2.total_duration_ms, 4000)


class TestBuildTreeEdgeCases(unittest.TestCase):
    def test_unknown_session_raises_key_error(self):
        tmp = tempfile.TemporaryDirectory()
        store = TraceStore(tmp.name)
        with self.assertRaises(KeyError):
            build_tree(store, "nonexistent")
        tmp.cleanup()

    def test_max_depth_truncation(self):
        """A chain deeper than MAX_DEPTH should be truncated at MAX_DEPTH."""
        from agent_trace.subagent import MAX_DEPTH
        tmp = tempfile.TemporaryDirectory()
        store = TraceStore(tmp.name)

        # Build a linear chain: root → d1 → d2 → ... → d(MAX_DEPTH+1)
        session_ids = [f"sess{i:04d}" for i in range(MAX_DEPTH + 2)]
        for i, sid in enumerate(session_ids):
            parent_sid = session_ids[i - 1] if i > 0 else ""
            meta = SessionMeta(
                session_id=sid,
                started_at=float(i),
                parent_session_id=parent_sid,
                depth=i,
            )
            store.create_session(meta)
            store.update_meta(meta)

        tree = build_tree(store, session_ids[0])

        # Walk to depth MAX_DEPTH — should exist
        node = tree
        for _ in range(MAX_DEPTH):
            self.assertEqual(len(node.children), 1)
            node = node.children[0]

        # Node at MAX_DEPTH should have no children (truncated)
        self.assertEqual(len(node.children), 0)
        tmp.cleanup()


class TestInSessionSubagentAttribution(unittest.TestCase):
    AGENT = "d0000011-0000-4000-8000-000000000011"
    OTHER = "d0000012-0000-4000-8000-000000000012"
    SID = "s1attr"

    def events(self, attributed=True):
        sub = {"agent_id": self.AGENT, "agent_type": "default"} if attributed else {}
        stop = {"text": "Synthetic   subagent\nanswer."} if attributed else {}

        def ev(event_type, ts, event_id, parent_id="", **data):
            return TraceEvent(event_type=event_type, timestamp=ts, session_id=self.SID,
                              event_id=event_id, parent_id=parent_id, data=data)

        return [
            ev(EventType.SESSION_START, 0.0, "e0", agent_name="codex"),
            ev(EventType.USER_PROMPT, 0.1, "e1", prompt="Spawn one subagent."),
            ev(EventType.TOOL_CALL, 1.0, "c1", tool_name="spawn_agent",
               arguments={"message": "run it"}),
            ev(EventType.TOOL_RESULT, 1.2, "r1", "c1", tool_name="spawn_agent", result="ok"),
            ev(EventType.SUBAGENT_START, 1.3, "e2", provider="codex", **sub),
            ev(EventType.TOOL_CALL, 2.0, "c2", tool_name="Bash",
               arguments={"command": "echo synthetic"}, **sub),
            ev(EventType.TOOL_RESULT, 2.5, "r2", "c2", tool_name="Bash", result="synthetic", **sub),
            ev(EventType.TOOL_CALL, 2.6, "c3", tool_name="Bash", arguments={"command": "false"},
               **sub),
            ev(EventType.ERROR, 2.7, "r3", "c3", tool_name="Bash", error="exit code 1", **sub),
            ev(EventType.SUBAGENT_STOP, 3.0, "e3", provider="codex", **sub, **stop),
            ev(EventType.ASSISTANT_RESPONSE, 4.0, "e4", text="Synthetic final answer."),
            ev(EventType.SESSION_END, 5.0, "e5"),
        ]

    def node(self, attributed=True, children=None):
        meta = SessionMeta(session_id=self.SID, agent_name="codex", started_at=0.0, tool_calls=3)
        return SessionNode(meta=meta, events=self.events(attributed), children=children or [])

    def render(self, formatter, node):
        buf = io.StringIO()
        formatter(node, out=buf)
        return buf.getvalue()

    def test_in_session_subagents_groups_by_agent_id(self):
        events = self.events() + [
            _make_event(EventType.TOOL_CALL, 6.0, self.SID, tool_name="Read",
                        agent_id=self.OTHER),
            _make_event(EventType.TOOL_RESULT, 6.1, self.SID, tool_name="Read",
                        agent_id=self.OTHER),
            _make_event(EventType.TOOL_CALL, 6.2, self.SID, tool_name="Read", agent_id=""),
        ]
        first, second = in_session_subagents(events)
        self.assertEqual(
            (first.agent_id, first.agent_type, first.tool_calls, first.errors),
            (self.AGENT, "default", 2, 1),
        )
        self.assertEqual(
            (second.agent_id, second.agent_type, second.tool_calls, second.errors),
            (self.OTHER, "", 1, 0),
        )
        self.assertEqual(in_session_subagents(self.events(attributed=False)), [])

    def test_format_tree_marks_subagent_events(self):
        lines = self.render(format_tree, self.node()).splitlines()
        tag = "[subagent default 00000011]"
        self.assertIn(f"Bash  {tag}  $ echo synthetic", lines[5])
        self.assertIn(f"tool_result  {tag}  synthetic", lines[6])
        self.assertIn(f"error  {tag}  exit code 1", lines[8])
        self.assertIn("subagent_start  default  agent=00000011", lines[4])
        self.assertIn("subagent_stop  default  agent=00000011", lines[9])
        self.assertEqual(lines[10].strip(), 'returned: "Synthetic subagent answer."')
        self.assertNotIn("[subagent", lines[2] + lines[3])

    def test_format_tree_without_attribution_has_no_marks(self):
        lines = self.render(format_tree, self.node(attributed=False)).splitlines()
        self.assertEqual(lines[4].strip(), "+ 1.30s  ⤷ subagent_start")
        self.assertEqual(lines[9].strip(), "+ 3.00s  ⤶ subagent_stop")
        body = "\n".join(lines[1:])
        for marker in ("[subagent", "agent=", "returned:"):
            self.assertNotIn(marker, body)

    def test_summaries_list_in_session_subagents(self):
        for formatter in (format_tree_summary, format_session_tree):
            with self.subTest(formatter=formatter.__name__):
                lines = self.render(formatter, self.node()).splitlines()
                self.assertEqual(lines[1], "└─ subagent default 00000011  2 tools  ✗ 1 errors")
                plain = self.render(formatter, self.node(attributed=False))
                self.assertNotIn("subagent", plain)
                self.assertEqual(len(plain.splitlines()), 1)

    def test_summary_subagent_line_precedes_child_sessions(self):
        child = SessionNode(
            meta=SessionMeta(session_id="child0001", started_at=1.0, depth=1), events=[]
        )
        lines = self.render(format_tree_summary, self.node(children=[child])).splitlines()
        self.assertEqual(lines[1], "├─ subagent default 00000011  2 tools  ✗ 1 errors")
        self.assertIn("child0001", lines[2])

    def test_tree_to_dict_lists_subagents_only_when_attributed(self):
        data = tree_to_dict(self.node())
        self.assertEqual(data["subagents"], [{
            "agent_id": self.AGENT, "agent_type": "default", "tool_calls": 2, "errors": 1,
        }])
        self.assertNotIn("subagents", tree_to_dict(self.node(attributed=False)))

    def run_stats(self, attributed):
        meta = SessionMeta(session_id=self.SID, agent_name="codex", started_at=0.0)
        store, tmp = _make_store_with_sessions([(meta, self.events(attributed))])
        self.addCleanup(tmp.cleanup)
        out = io.StringIO()
        args = argparse.Namespace(
            trace_dir=store.base_dir, session_id=self.SID, include_subagents=False
        )
        with patch("sys.stdout", out):
            self.assertEqual(cmd_stats(args), 0)
        return out.getvalue()

    def test_stats_lists_subagent_tool_calls(self):
        output = self.run_stats(attributed=True)
        self.assertIn("Subagent Tool Calls:", output)
        self.assertIn(f"    {'default 00000011':<30} {2:>4}x  1 errors", output)

    def test_stats_without_attribution_has_no_subagent_section(self):
        attributed = self.run_stats(attributed=True)
        plain = self.run_stats(attributed=False)
        self.assertNotIn("Subagent", plain)
        start = attributed.index("\n  Subagent Tool Calls:")
        end = attributed.index("\n  Errors (")
        self.assertEqual(attributed[:start] + attributed[end:], plain)


if __name__ == "__main__":
    unittest.main()
