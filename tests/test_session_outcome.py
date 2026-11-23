from __future__ import annotations

import unittest

from agent_trace.models import EventType, TraceEvent
from agent_trace.session_outcome import recorded_session_status


def _events(*items: tuple[EventType, dict]) -> list[TraceEvent]:
    return [
        TraceEvent(event_type=event_type, timestamp=1_000.0 + index, data=data)
        for index, (event_type, data) in enumerate(items)
    ]


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


if __name__ == "__main__":
    unittest.main()
