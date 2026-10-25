"""Tests for the shared check that refuses non-object event data."""

import types
import unittest

from agent_trace import disclosure
from agent_trace.event_data import (
    NonObjectEventDataError,
    require_object_event_data,
    skipped_event_warning,
)
from agent_trace.models import EventType, TraceEvent

_PREFIX = "session event data must be an object: "

_NON_OBJECT_DATA = (
    (["SECRET-ARRAY", {"command": "SECRET-COMMAND"}], "array"),
    ("SECRET-STRING", "string"),
    (42, "number"),
    (4.5, "number"),
    (False, "boolean"),
    (None, "null"),
)


def _events() -> list[TraceEvent]:
    return [
        TraceEvent(event_type=EventType.SESSION_START, data={}),
        TraceEvent(event_type=EventType.USER_PROMPT, data="SECRET-PROMPT"),
        TraceEvent(event_type=EventType.TOOL_CALL, data={"tool_name": "Bash"}),
        TraceEvent(event_type=EventType.TOOL_RESULT, data=["SECRET-RESULT"]),
        TraceEvent(event_type=EventType.ERROR, data=7),
        TraceEvent(event_type=EventType.SESSION_END, data={"exit_code": 0}),
    ]


class TestRequireObjectEventData(unittest.TestCase):
    def _message(self, events, **kwargs) -> str:
        with self.assertRaises(NonObjectEventDataError) as caught:
            require_object_event_data(events, **kwargs)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertNotIn("SECRET", str(caught.exception))
        return str(caught.exception)

    def test_old_disclosure_import_path_still_works(self):
        self.assertIs(disclosure.NonObjectEventDataError, NonObjectEventDataError)
        self.assertIs(disclosure.require_object_event_data, require_object_event_data)

    def test_every_event_is_checked_by_default(self):
        self.assertEqual(
            self._message(_events()),
            _PREFIX + "event 2 of 6 (user_prompt) has string data; "
            "2 more events have non-object data",
        )

    def test_every_json_type_is_named_without_its_value(self):
        for data, json_type in _NON_OBJECT_DATA:
            with self.subTest(json_type=json_type, data=data):
                events = [
                    TraceEvent(event_type=EventType.SESSION_START, data={}),
                    TraceEvent(event_type=EventType.TOOL_CALL, data=data),
                ]
                self.assertEqual(
                    self._message(events),
                    _PREFIX + f"event 2 of 2 (tool_call) has {json_type} data",
                )

    def test_event_types_limit_which_events_are_checked(self):
        events = _events()
        self.assertEqual(
            self._message(events, event_types={EventType.ERROR, EventType.TOOL_RESULT}),
            _PREFIX + "event 4 of 6 (tool_result) has array data; "
            "2 more events have non-object data",
        )
        require_object_event_data(events, event_types={EventType.TOOL_CALL, EventType.SESSION_END})
        require_object_event_data(events, event_types=())

    def test_indexes_limit_which_events_are_checked(self):
        events = _events()
        self.assertEqual(
            self._message(events, indexes=[5, 4]),
            _PREFIX + "event 5 of 6 (error) has number data; 2 more events have non-object data",
        )
        require_object_event_data(events, indexes=[0, 2, 5])
        require_object_event_data(events, indexes=[])

    def test_event_types_and_indexes_must_both_match(self):
        events = _events()
        require_object_event_data(events, event_types={EventType.USER_PROMPT}, indexes=[3, 4])
        self.assertEqual(
            self._message(events, event_types={EventType.TOOL_RESULT}, indexes=[1, 3]),
            _PREFIX + "event 4 of 6 (tool_result) has array data; "
            "2 more events have non-object data",
        )

    def test_count_covers_one_and_several_other_events(self):
        events = _events()
        self.assertEqual(
            self._message(events[:4]),
            _PREFIX + "event 2 of 4 (user_prompt) has string data; 1 more event has non-object data",
        )
        self.assertEqual(
            self._message(events[2:4]),
            _PREFIX + "event 2 of 2 (tool_result) has array data",
        )

    def test_session_id_names_the_session(self):
        self.assertEqual(
            self._message(_events(), session_id="0123456789abcdef"),
            "session 0123456789abcdef event data must be an object: event 2 of 6 "
            "(user_prompt) has string data; 2 more events have non-object data",
        )

    def test_event_with_an_unknown_type_is_still_named(self):
        events = [TraceEvent(event_type="not-a-type", data="SECRET-STRING")]
        self.assertEqual(
            self._message(events),
            _PREFIX + "event 1 of 1 (unknown type) has string data",
        )

    def test_objects_pass_the_check(self):
        require_object_event_data([])
        require_object_event_data([
            TraceEvent(event_type=EventType.SESSION_START),
            TraceEvent(event_type=EventType.TOOL_CALL, data={"command": "SECRET-COMMAND"}),
            TraceEvent(
                event_type=EventType.TOOL_RESULT,
                data=types.MappingProxyType({"result": "ok"}),
            ),
        ])


class TestSkippedEventWarning(unittest.TestCase):
    def test_warning_names_the_event_type_and_json_type_only(self):
        for data, phrase in (
            (["SECRET-ARRAY"], "an array"),
            ("SECRET-STRING", "a string"),
            (42, "a number"),
            (False, "a boolean"),
            (None, "null"),
        ):
            with self.subTest(data=data):
                event = TraceEvent(event_type=EventType.TOOL_RESULT, data=data)
                self.assertEqual(
                    skipped_event_warning("watch", event),
                    "[watch] Skipped checks for one tool_result event because its data "
                    f"is {phrase}, not an object.\n",
                )


if __name__ == "__main__":
    unittest.main()
