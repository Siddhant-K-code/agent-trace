"""Tests for diff phase numbers when a phase exists in only one session."""

import io
import tempfile
import unittest

from agent_trace.diff import diff_sessions, format_diff
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

SID_A = "aaaa00000001"
SID_B = "bbbb00000002"


class TestDiffPhaseNumbers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TraceStore(self._tmp.name)

    def _session(self, sid, phases):
        """Store a session with one prompt and one Bash command per phase."""
        self.store.create_session(SessionMeta(session_id=sid, started_at=0.0))
        ts = 0.0
        for prompt, command in phases:
            for event_type, data in (
                (EventType.USER_PROMPT, {"prompt": prompt}),
                (EventType.TOOL_CALL, {"tool_name": "Bash", "arguments": {"command": command}}),
            ):
                self.store.append_event(sid, TraceEvent(
                    event_type=event_type, timestamp=ts, session_id=sid, data=data))
                ts += 1.0

    def _diff(self, phases_a, phases_b):
        self._session(SID_A, phases_a)
        self._session(SID_B, phases_b)
        result = diff_sessions(self.store, SID_A, SID_B)
        buf = io.StringIO()
        format_diff(result, out=buf)
        return result, buf.getvalue()

    def test_extra_phase_before_a_later_difference_is_the_divergence(self):
        result, text = self._diff(
            [("setup env", "make setup"), ("extra step", "rm -rf cache"),
             ("lint", "make lint"), ("run build", "make build")],
            [("setup env", "make setup"), ("lint", "make lint"),
             ("run build", "make build -j8")],
        )
        self.assertEqual(result.divergence_index, 1)
        self.assertIn("Diverged at phase 2:", text)
        self.assertIn(f"  Phase 2 ({SID_A} only): extra step\n", text)

    def test_aligned_phases_keep_their_own_numbers(self):
        result, text = self._diff(
            [("setup env", "make setup"), ("extra step", "rm -rf cache"),
             ("run build", "make build")],
            [("setup env", "make setup"), ("extra b1", "ls"), ("extra b2", "pwd"),
             ("run build", "make build -j8")],
        )
        build = result.phase_diffs[-1]
        self.assertEqual((build.index, build.index_a, build.index_b), (1, 2, 3))
        self.assertEqual(result.phases_only_a, [(1, "extra step")])
        self.assertEqual(result.phases_only_b, [(1, "extra b1"), (2, "extra b2")])
        self.assertEqual(result.divergence_index, 1)
        self.assertIn(
            "Diverged at phase 2:\n\n"
            f"  Phase 2 ({SID_A} only): extra step\n"
            f"  Phase 2 ({SID_B} only): extra b1\n"
            f"  Phase 3 ({SID_B} only): extra b2\n\n"
            f"  Phase 3 in {SID_A}, 4 in {SID_B}: run build\n"
            f"    {SID_A} only:  $ make build\n"
            f"    {SID_B} only:  $ make build -j8\n\n",
            text,
        )

    def test_trailing_phase_in_one_session_is_listed(self):
        result, text = self._diff(
            [("setup env", "make setup"), ("run build", "make build")],
            [("setup env", "make setup"), ("run build", "make build"), ("deploy", "make deploy")],
        )
        self.assertEqual(result.divergence_index, 2)
        self.assertIn(f"Diverged at phase 3:\n\n  Phase 3 ({SID_B} only): deploy\n\n", text)

    def test_same_phases_print_what_they_did_before(self):
        phases = [("setup env", "make setup"), ("run build", "make build")]
        result, text = self._diff(phases, phases[:1] + [("run build", "make build -j8")])
        self.assertEqual(result.divergence_index, 1)
        self.assertEqual(
            text,
            f"\nComparing: {SID_A} vs {SID_B}\n\n"
            "Diverged at phase 2:\n\n"
            "  Phase 2: run build\n"
            f"    {SID_A} only:  $ make build\n"
            f"    {SID_B} only:  $ make build -j8\n\n"
            f"  {SID_A}: 0s, 4 events, 0 tools, 0 retries\n"
            f"  {SID_B}: 0s, 4 events, 0 tools, 0 retries\n\n",
        )

    def test_identical_sessions_print_what_they_did_before(self):
        phases = [("setup env", "make setup"), ("run build", "make build")]
        result, text = self._diff(phases, phases)
        self.assertEqual(result.divergence_index, -1)
        self.assertEqual(
            text,
            f"\nComparing: {SID_A} vs {SID_B}\n\n"
            "Sessions are structurally identical.\n\n"
            f"  {SID_A}: 0s, 4 events, 0 tools, 0 retries\n"
            f"  {SID_B}: 0s, 4 events, 0 tools, 0 retries\n\n",
        )


if __name__ == "__main__":
    unittest.main()
