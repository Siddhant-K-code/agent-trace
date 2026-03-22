"""Tests for session explain."""

import io
import tempfile
import unittest

from agent_trace.explain import (
    ExplainResult,
    Phase,
    build_phases,
    explain_session,
    format_explain,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_event(event_type: EventType, ts: float, session_id: str = "s1", **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)



class TestBuildPhases(unittest.TestCase):
    def test_empty_events(self):
        phases = build_phases([], base_ts=0.0)
        self.assertEqual(phases, [])

    def test_single_phase_no_prompt(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0, tool_name="Bash", arguments={"command": "ls"}),
            _make_event(EventType.TOOL_RESULT, 1.0),
        ]
        phases = build_phases(events, base_ts=0.0)
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0].index, 1)
        self.assertEqual(phases[0].event_count, 2)

    def test_splits_on_user_prompt(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, prompt="first task"),
            _make_event(EventType.TOOL_CALL, 1.0, tool_name="Bash", arguments={"command": "ls"}),
            _make_event(EventType.USER_PROMPT, 2.0, prompt="second task"),
            _make_event(EventType.TOOL_CALL, 3.0, tool_name="Bash", arguments={"command": "pwd"}),
        ]
        phases = build_phases(events, base_ts=0.0)
        self.assertEqual(len(phases), 2)
        self.assertEqual(phases[0].name, "first task")
        self.assertEqual(phases[1].name, "second task")

    def test_phase_label_from_prompt(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, prompt="fix the auth module"),
        ]
        phases = build_phases(events, base_ts=0.0)
        self.assertEqual(phases[0].name, "fix the auth module")

    def test_phase_label_from_tool_when_no_prompt(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0, tool_name="Bash", arguments={"command": "ls"}),
        ]
        phases = build_phases(events, base_ts=0.0)
        self.assertEqual(phases[0].name, "Bash")

    def test_files_read_collected(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0, tool_name="Read",
                        arguments={"file_path": "src/auth.py"}),
            _make_event(EventType.TOOL_CALL, 1.0, tool_name="Read",
                        arguments={"file_path": "src/utils.py"}),
        ]
        phases = build_phases(events, base_ts=0.0)
        self.assertIn("src/auth.py", phases[0].files_read)
        self.assertIn("src/utils.py", phases[0].files_read)

    def test_files_written_collected(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0, tool_name="Write",
                        arguments={"file_path": "src/auth.py"}),
        ]
        phases = build_phases(events, base_ts=0.0)
        self.assertIn("src/auth.py", phases[0].files_written)

    def test_failed_phase_detected(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0, tool_name="Bash", arguments={"command": "pytest"}),
            _make_event(EventType.ERROR, 1.0, message="exit 1"),
        ]
        phases = build_phases(events, base_ts=0.0)
        self.assertTrue(phases[0].failed)

    def test_retry_detection(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0, tool_name="Bash", arguments={"command": "pytest"}),
            _make_event(EventType.ERROR, 1.0, message="fail"),
            _make_event(EventType.TOOL_CALL, 2.0, tool_name="Bash", arguments={"command": "pytest"}),
        ]
        phases = build_phases(events, base_ts=0.0)
        self.assertEqual(phases[0].retry_count, 1)

    def test_phase_offsets(self):
        base = 1000.0
        events = [
            _make_event(EventType.USER_PROMPT, base, prompt="go"),
            _make_event(EventType.TOOL_CALL, base + 5.0, tool_name="Bash", arguments={"command": "ls"}),
        ]
        phases = build_phases(events, base_ts=base)
        self.assertAlmostEqual(phases[0].start_offset, 0.0, places=1)
        self.assertAlmostEqual(phases[0].end_offset, 5.0, places=1)


class TestExplainSession(unittest.TestCase):
    def _build(self, events, session_id="sess1"):
        d = tempfile.mkdtemp()
        store = TraceStore(d)
        meta = SessionMeta(
            session_id=session_id,
            started_at=events[0].timestamp,
            total_duration_ms=(events[-1].timestamp - events[0].timestamp) * 1000,
        )
        store.create_session(meta)
        for e in events:
            store.append_event(session_id, e)
        store.update_meta(meta)
        self._tmpdir = d
        return store

    def test_explain_result_structure(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, session_id="sess1", prompt="do something"),
            _make_event(EventType.TOOL_CALL, 1.0, session_id="sess1",
                        tool_name="Bash", arguments={"command": "ls"}),
            _make_event(EventType.SESSION_END, 5.0, session_id="sess1"),
        ]
        store = self._build(events)
        result = explain_session(store, "sess1")
        self.assertIsInstance(result, ExplainResult)
        self.assertEqual(result.session_id, "sess1")
        self.assertGreater(len(result.phases), 0)

    def test_wasted_seconds_from_failed_phases(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, session_id="sess1", prompt="run tests"),
            _make_event(EventType.TOOL_CALL, 1.0, session_id="sess1",
                        tool_name="Bash", arguments={"command": "pytest"}),
            _make_event(EventType.ERROR, 3.0, session_id="sess1", message="fail"),
            _make_event(EventType.SESSION_END, 4.0, session_id="sess1"),
        ]
        store = self._build(events)
        result = explain_session(store, "sess1")
        failed = [p for p in result.phases if p.failed]
        self.assertTrue(len(failed) > 0)
        self.assertGreater(result.wasted_seconds, 0)


class TestFormatExplain(unittest.TestCase):
    def test_output_contains_session_id(self):
        result = ExplainResult(
            session_id="abc123",
            total_duration=60.0,
            total_events=10,
            phases=[
                Phase(name="Setup", index=1, start_offset=0, end_offset=10,
                      events=[], files_read=["README.md"], commands=["ls"])
            ],
            total_retries=0,
            wasted_seconds=0,
        )
        buf = io.StringIO()
        format_explain(result, out=buf)
        output = buf.getvalue()
        self.assertIn("abc123", output)
        self.assertIn("Phase 1", output)
        self.assertIn("README.md", output)

    def test_output_shows_retry_summary(self):
        result = ExplainResult(
            session_id="xyz",
            total_duration=120.0,
            total_events=20,
            phases=[
                Phase(name="Test", index=1, start_offset=0, end_offset=60,
                      events=[], failed=True, retry_count=3,
                      commands=["pytest", "pytest", "pytest"])
            ],
            total_retries=3,
            wasted_seconds=60.0,
        )
        buf = io.StringIO()
        format_explain(result, out=buf)
        output = buf.getvalue()
        self.assertIn("Retries", output)
        self.assertIn("3", output)


if __name__ == "__main__":
    unittest.main()
