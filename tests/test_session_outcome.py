from __future__ import annotations

import unittest

from agent_trace.models import EventType, TraceEvent
from agent_trace.session_outcome import behavior_analysis, recorded_session_status


def _events(*items: tuple[EventType, dict]) -> list[TraceEvent]:
    return [
        TraceEvent(event_type=event_type, timestamp=1_000.0 + index, data=data)
        for index, (event_type, data) in enumerate(items)
    ]


def _session(outcomes, command="make test"):
    """One tool call per outcome, all the same operation unless a command is given."""
    events = []
    for number, outcome in enumerate(outcomes):
        call_id = f"c{number}"
        ts = 1001.0 + number * 2
        events.append(TraceEvent(
            event_type=EventType.TOOL_CALL, event_id=call_id, timestamp=ts,
            data={"tool_name": "Bash", "command": command},
        ))
        if outcome == "ok":
            events.append(TraceEvent(
                event_type=EventType.TOOL_RESULT, parent_id=call_id, timestamp=ts + 1,
                data={"tool_name": "Bash", "result": "ok"},
            ))
        else:
            events.append(TraceEvent(
                event_type=EventType.ERROR, parent_id=call_id, timestamp=ts + 1,
                data={"tool_name": "Bash", "error": "boom"},
            ))
    return events


class RecordedSessionStatusTests(unittest.TestCase):
    def test_resume_after_end_is_not_recorded(self):
        for end in ({"status": "completed"}, {"status": "interrupted"}, {"exit_code": 1}):
            with self.subTest(end=end):
                events = _events(
                    (EventType.SESSION_START, {}),
                    (EventType.SESSION_END, end),
                    (EventType.SESSION_START, {"source": "resume"}),
                    (EventType.TOOL_CALL, {"tool_name": "Bash"}),
                )
                self.assertEqual(recorded_session_status(events), "not_recorded")

    def test_session_ended_after_resume_reads_final_end(self):
        events = _events(
            (EventType.SESSION_START, {}),
            (EventType.SESSION_END, {"status": "completed"}),
            (EventType.SESSION_START, {"source": "resume"}),
            (EventType.SESSION_END, {"status": "interrupted"}),
            (EventType.TOOL_RESULT, {"tool_name": "Bash"}),
        )
        self.assertEqual(recorded_session_status(events), "terminated")


class TestRecoveryFollowsLastOutcome(unittest.TestCase):
    def _summary(self, outcomes):
        analysis = behavior_analysis(_session(outcomes))
        return analysis, analysis.summary

    def test_failure_then_success_recovers(self):
        analysis, summary = self._summary(["fail", "ok"])
        self.assertEqual(summary.recovered_operation_count, 1)
        self.assertEqual(summary.unresolved_failed_operation_count, 0)
        self.assertEqual(analysis.unresolved_failures, ())

    def test_failure_after_recovery_is_unresolved(self):
        analysis, summary = self._summary(["fail", "ok", "fail"])
        self.assertEqual(summary.recovered_operation_count, 0)
        self.assertEqual(summary.unresolved_failed_operation_count, 1)
        # Only the last failure, the error at position 5, is unresolved.
        self.assertEqual(analysis.unresolved_failures, (5,))
        self.assertEqual(summary.failed_operation_count, 2)
        self.assertEqual(summary.retry_count, 2)
        self.assertEqual(summary.recovery_attempt_count, 1)

    def test_unresolved_failures_follow_the_last_success(self):
        analysis, summary = self._summary(["fail", "fail", "ok", "fail", "fail"])
        self.assertEqual(summary.unresolved_failed_operation_count, 1)
        self.assertEqual(analysis.unresolved_failures, (7, 9))

    def test_success_after_second_failure_recovers_again(self):
        analysis, summary = self._summary(["fail", "ok", "fail", "ok"])
        self.assertEqual(summary.recovered_operation_count, 1)
        self.assertEqual(summary.unresolved_failed_operation_count, 0)
        self.assertEqual(analysis.unresolved_failures, ())


if __name__ == "__main__":
    unittest.main()
