"""Tests for session diff and causal chain (why)."""

import io
import tempfile
import unittest

from agent_trace.diff import (
    PhaseDiff,
    SessionDiff,
    _lcs_indices,
    diff_sessions,
    format_diff,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.why import (
    CausalChain,
    build_causal_chain,
    format_why,
)


def _make_event(event_type: EventType, ts: float, session_id: str,
                event_id: str = "", parent_id: str = "", **data) -> TraceEvent:
    e = TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)
    if event_id:
        e.event_id = event_id
    if parent_id:
        e.parent_id = parent_id
    return e


def _make_store(sessions: list[tuple[SessionMeta, list[TraceEvent]]]) -> tuple[TraceStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = TraceStore(tmp.name)
    for meta, events in sessions:
        store.create_session(meta)
        for e in events:
            store.append_event(meta.session_id, e)
        store.update_meta(meta)
    return store, tmp


# ---------------------------------------------------------------------------
# LCS tests
# ---------------------------------------------------------------------------

class TestLCS(unittest.TestCase):
    def test_identical_lists(self):
        pairs = _lcs_indices(["a", "b", "c"], ["a", "b", "c"])
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2)])

    def test_empty_lists(self):
        self.assertEqual(_lcs_indices([], []), [])

    def test_no_common(self):
        self.assertEqual(_lcs_indices(["a", "b"], ["c", "d"]), [])

    def test_partial_match(self):
        pairs = _lcs_indices(["a", "b", "c"], ["a", "x", "c"])
        # Should match a and c
        self.assertIn((0, 0), pairs)
        self.assertIn((2, 2), pairs)

    def test_insertion(self):
        pairs = _lcs_indices(["a", "c"], ["a", "b", "c"])
        self.assertEqual(pairs, [(0, 0), (1, 2)])


# ---------------------------------------------------------------------------
# Diff tests
# ---------------------------------------------------------------------------

class TestDiffSessions(unittest.TestCase):
    def _two_sessions(self, cmds_a, cmds_b, failed_a=False, failed_b=False):
        def _events(sid, cmds, failed):
            evts = [_make_event(EventType.USER_PROMPT, 0.0, sid, prompt="run tests")]
            for i, cmd in enumerate(cmds):
                evts.append(_make_event(EventType.TOOL_CALL, float(i + 1), sid,
                                        tool_name="Bash", arguments={"command": cmd}))
            if failed:
                evts.append(_make_event(EventType.ERROR, float(len(cmds) + 1), sid,
                                        message="exit 1"))
            evts.append(_make_event(EventType.SESSION_END, float(len(cmds) + 2), sid))
            return evts

        meta_a = SessionMeta(session_id="sessa001", started_at=0.0,
                             total_duration_ms=5000, tool_calls=len(cmds_a))
        meta_b = SessionMeta(session_id="sessb001", started_at=0.0,
                             total_duration_ms=4000, tool_calls=len(cmds_b))
        store, tmp = _make_store([
            (meta_a, _events("sessa001", cmds_a, failed_a)),
            (meta_b, _events("sessb001", cmds_b, failed_b)),
        ])
        return store, tmp

    def test_identical_sessions(self):
        store, tmp = self._two_sessions(["pytest"], ["pytest"])
        result = diff_sessions(store, "sessa001", "sessb001")
        self.assertEqual(result.divergence_index, -1)
        tmp.cleanup()

    def test_different_commands_diverge(self):
        store, tmp = self._two_sessions(["pytest"], ["python -m pytest"])
        result = diff_sessions(store, "sessa001", "sessb001")
        self.assertGreater(len(result.phase_diffs), 0)
        tmp.cleanup()

    def test_failed_vs_passed(self):
        store, tmp = self._two_sessions(["pytest"], ["pytest"],
                                        failed_a=True, failed_b=False)
        result = diff_sessions(store, "sessa001", "sessb001")
        failed_diffs = [pd for pd in result.phase_diffs if pd.a_failed != pd.b_failed]
        self.assertTrue(len(failed_diffs) > 0)
        tmp.cleanup()

    def test_duration_captured(self):
        store, tmp = self._two_sessions(["pytest"], ["pytest"])
        result = diff_sessions(store, "sessa001", "sessb001")
        self.assertAlmostEqual(result.duration_a, 5.0, places=0)
        self.assertAlmostEqual(result.duration_b, 4.0, places=0)
        tmp.cleanup()


class TestFormatDiff(unittest.TestCase):
    def test_output_contains_session_ids(self):
        result = SessionDiff(
            session_a="aaa111bbb222",
            session_b="ccc333ddd444",
            divergence_index=0,
            phase_diffs=[
                PhaseDiff(index=0, label_a="run tests", label_b="run tests",
                          same_label=True, files_only_a=[], files_only_b=[],
                          cmds_only_a=["pytest"], cmds_only_b=["python -m pytest"],
                          a_failed=False, b_failed=False)
            ],
            duration_a=10.0, duration_b=8.0,
            events_a=5, events_b=4,
            tool_calls_a=2, tool_calls_b=2,
            retries_a=0, retries_b=0,
        )
        buf = io.StringIO()
        format_diff(result, out=buf)
        output = buf.getvalue()
        self.assertIn("aaa111bbb2", output)
        self.assertIn("ccc333ddd4", output)
        self.assertIn("pytest", output)

    def test_identical_sessions_message(self):
        result = SessionDiff(
            session_a="aaa", session_b="bbb",
            divergence_index=-1, phase_diffs=[],
            duration_a=5.0, duration_b=5.0,
            events_a=3, events_b=3,
            tool_calls_a=1, tool_calls_b=1,
            retries_a=0, retries_b=0,
        )
        buf = io.StringIO()
        format_diff(result, out=buf)
        self.assertIn("identical", buf.getvalue())


# ---------------------------------------------------------------------------
# Why / causal chain tests
# ---------------------------------------------------------------------------

class TestBuildCausalChain(unittest.TestCase):
    def test_empty_events(self):
        chain = build_causal_chain([], 0)
        self.assertEqual(chain.links, [])

    def test_out_of_range(self):
        events = [_make_event(EventType.TOOL_CALL, 0.0, "s1", tool_name="Bash",
                              arguments={"command": "ls"})]
        chain = build_causal_chain(events, 5)
        self.assertEqual(chain.links, [])

    def test_user_prompt_is_root(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="do it"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Bash",
                        arguments={"command": "ls"}),
        ]
        chain = build_causal_chain(events, 1)
        types = [l.event.event_type for l in chain.links]
        self.assertIn(EventType.USER_PROMPT, types)

    def test_parent_id_link(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0, "s1", event_id="call1",
                        tool_name="Bash", arguments={"command": "ls"}),
            _make_event(EventType.TOOL_RESULT, 1.0, "s1", parent_id="call1",
                        result="file.py"),
        ]
        chain = build_causal_chain(events, 1)
        types = [l.event.event_type for l in chain.links]
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertIn(EventType.TOOL_RESULT, types)

    def test_error_causes_retry(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="run"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Bash",
                        arguments={"command": "pytest"}),
            _make_event(EventType.ERROR, 2.0, "s1", message="exit 1"),
            _make_event(EventType.TOOL_CALL, 3.0, "s1", tool_name="Bash",
                        arguments={"command": "pytest"}),
        ]
        chain = build_causal_chain(events, 3)
        types = [l.event.event_type for l in chain.links]
        self.assertIn(EventType.ERROR, types)

    def test_chain_ordered_root_to_target(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="go"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Bash",
                        arguments={"command": "ls"}),
        ]
        chain = build_causal_chain(events, 1)
        # Last link should be the target
        self.assertEqual(chain.links[-1].event_index, 1)

    def test_single_event_chain(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="hello"),
        ]
        chain = build_causal_chain(events, 0)
        self.assertEqual(len(chain.links), 1)
        self.assertEqual(chain.links[0].event.event_type, EventType.USER_PROMPT)


class TestFormatWhy(unittest.TestCase):
    def test_output_contains_event_number(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="do it"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Bash",
                        arguments={"command": "ls"}),
        ]
        chain = build_causal_chain(events, 1)
        buf = io.StringIO()
        format_why(chain, events, out=buf)
        self.assertIn("#2", buf.getvalue())

    def test_no_chain_message(self):
        chain = CausalChain(target_index=0, links=[])
        buf = io.StringIO()
        format_why(chain, [], out=buf)
        self.assertIn("No causal chain", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
