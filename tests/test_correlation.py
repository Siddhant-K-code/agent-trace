"""Tests for fail-closed tool outcome correlation."""

import unittest

from agent_trace.correlation import correlate_tool_events
from agent_trace.models import EventType, TraceEvent


def _event(
    event_type: EventType,
    timestamp: float,
    *,
    event_id: str,
    parent_id: str = "",
    **data,
) -> TraceEvent:
    return TraceEvent(
        event_type=event_type,
        timestamp=timestamp,
        event_id=event_id,
        parent_id=parent_id,
        data=data,
    )


class TestToolCorrelation(unittest.TestCase):
    def test_orphan_result_stays_unresolved(self):
        events = [
            _event(EventType.TOOL_CALL, 1.0, event_id="call-1", tool_name="Read"),
            _event(EventType.TOOL_RESULT, 2.0, event_id="result-1", result="ok"),
        ]
        result = correlate_tool_events(events)
        self.assertEqual(result.call_outcomes, {0: "result_unobserved"})
        self.assertEqual(result.outcome_calls, {})
        self.assertEqual(result.quality["orphan_results"], 1)
        self.assertEqual(result.quality["missing_results"], 1)

    def test_duplicate_call_id_makes_result_ambiguous(self):
        events = [
            _event(EventType.TOOL_CALL, 1.0, event_id="duplicate", tool_name="Read"),
            _event(EventType.TOOL_CALL, 2.0, event_id="duplicate", tool_name="Write"),
            _event(
                EventType.TOOL_RESULT,
                3.0,
                event_id="result",
                parent_id="duplicate",
                result="ok",
            ),
        ]
        result = correlate_tool_events(events)
        self.assertEqual(
            result.call_outcomes,
            {0: "result_unobserved", 1: "result_unobserved"},
        )
        self.assertEqual(result.outcome_calls, {})
        self.assertEqual(result.quality["ambiguous_results"], 1)
        self.assertEqual(result.quality["duplicate_call_ids"], 2)
        self.assertEqual(result.quality["missing_results"], 2)

    def test_multiple_outcomes_remove_the_first_tentative_match(self):
        events = [
            _event(EventType.TOOL_CALL, 1.0, event_id="call", tool_name="Bash"),
            _event(
                EventType.ERROR,
                2.0,
                event_id="error",
                parent_id="call",
                message="failed",
            ),
            _event(
                EventType.TOOL_RESULT,
                3.0,
                event_id="result",
                parent_id="call",
                result="ok",
            ),
        ]
        result = correlate_tool_events(events)
        self.assertEqual(result.call_outcomes, {0: "ambiguous_result"})
        self.assertEqual(result.outcome_calls, {})
        self.assertEqual(result.quality["multiple_results"], 1)

    def test_result_before_call_stays_unresolved(self):
        events = [
            _event(
                EventType.TOOL_RESULT,
                1.0,
                event_id="result",
                parent_id="call",
                result="ok",
            ),
            _event(EventType.TOOL_CALL, 2.0, event_id="call", tool_name="Read"),
        ]
        result = correlate_tool_events(events)
        self.assertEqual(result.call_outcomes, {1: "result_unobserved"})
        self.assertEqual(result.outcome_calls, {})
        self.assertEqual(result.quality["before_call_results"], 1)
        self.assertEqual(result.quality["missing_results"], 1)

    def test_unique_tool_use_id_is_an_explicit_correlation_key(self):
        events = [
            _event(
                EventType.TOOL_CALL,
                1.0,
                event_id="call",
                tool_name="Read",
                tool_use_id="use-1",
            ),
            _event(
                EventType.TOOL_RESULT,
                2.0,
                event_id="result",
                tool_use_id="use-1",
                result="ok",
            ),
        ]
        result = correlate_tool_events(events)
        self.assertEqual(result.call_outcomes, {0: "recorded_result"})
        self.assertEqual(result.outcome_calls, {1: 0})
        self.assertTrue(all(count == 0 for count in result.quality.values()))

    def test_conflicting_parent_and_tool_use_ids_are_ambiguous(self):
        events = [
            _event(
                EventType.TOOL_CALL,
                1.0,
                event_id="call-1",
                tool_name="Read",
                tool_use_id="use-1",
            ),
            _event(
                EventType.TOOL_CALL,
                2.0,
                event_id="call-2",
                tool_name="Write",
                tool_use_id="use-2",
            ),
            _event(
                EventType.TOOL_RESULT,
                3.0,
                event_id="result",
                parent_id="call-1",
                tool_use_id="use-2",
                result="ok",
            ),
        ]
        result = correlate_tool_events(events)
        self.assertEqual(result.outcome_calls, {})
        self.assertEqual(result.quality["ambiguous_results"], 1)


if __name__ == "__main__":
    unittest.main()
