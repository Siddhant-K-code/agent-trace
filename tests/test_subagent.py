"""Tests for subagent tracing."""

import io
import json
import tempfile
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.subagent import (
    AggregatedStats,
    SessionNode,
    aggregate_stats,
    build_tree,
    format_tree,
    format_tree_summary,
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


if __name__ == "__main__":
    unittest.main()
