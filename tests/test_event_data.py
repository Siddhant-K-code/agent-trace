"""Tests for the shared event data checks and typed field readers."""

import json
import types
import unittest

from agent_trace import disclosure
from agent_trace.event_data import (
    NonObjectEventDataError,
    left_out_event_warning,
    list_field,
    number_field,
    object_field,
    refused_event_warning,
    require_object_event_data,
    skipped_event_warning,
    str_field,
    str_list_field,
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


class TestRefusedEventWarning(unittest.TestCase):
    def test_warning_names_the_event_type_and_json_type_only(self):
        for data, phrase in (
            (["SECRET-ARRAY"], "an array"),
            ("SECRET-STRING", "a string"),
            ("", "a string"),
            (42, "a number"),
            (0, "a number"),
            (True, "a boolean"),
            (None, "null"),
        ):
            with self.subTest(data=data):
                event = TraceEvent(
                    event_type=EventType.TOOL_CALL,
                    event_id="SECRET-EVENT-ID",
                    data=data,
                )
                self.assertEqual(
                    refused_event_warning("server", event),
                    "[server] Refused one tool_call event because its data "
                    f"is {phrase}, not an object.\n",
                )


class TestLeftOutEventWarning(unittest.TestCase):
    def test_warning_names_the_event_type_and_json_type_only(self):
        for data, phrase in (
            (["SECRET-ARRAY"], "an array"),
            ("SECRET-STRING", "a string"),
            (42, "a number"),
            (True, "a boolean"),
            (None, "null"),
        ):
            with self.subTest(data=data):
                event = TraceEvent(
                    event_type=EventType.TOOL_RESULT,
                    event_id="SECRET-EVENT-ID",
                    data=data,
                )
                self.assertEqual(
                    left_out_event_warning("stream", event),
                    "[stream] Left one tool_result event out of the OTLP stream because "
                    f"its data is {phrase}, not an object.\n",
                )


_MISSING = object()
_SAMPLES = {
    "string": "Bash",
    "empty string": "",
    "numeric string": "12",
    "int": 12,
    "float": 1.5,
    "true": True,
    "false": False,
    "null": None,
    "array": ["x", 1],
    "string array": ["x", "y"],
    "tuple": ("x", "y"),
    "object": {"k": "v"},
}


class TestTypedFieldReaders(unittest.TestCase):
    """A value of another JSON type reads the same as an absent key."""

    def _check(self, reader, accepted: set[str]) -> None:
        for name, value in _SAMPLES.items():
            with self.subTest(reader=reader.__name__, value=name):
                expected = value if name in accepted else _MISSING
                self.assertIs(reader({"key": value}, "key", _MISSING), expected)
        with self.subTest(reader=reader.__name__, value="absent"):
            self.assertIs(reader({}, "key", _MISSING), _MISSING)

    def test_str_field_accepts_strings_only(self):
        self._check(str_field, {"string", "empty string", "numeric string"})

    def test_number_field_accepts_int_and_float_but_not_bool_or_numeric_strings(self):
        self._check(number_field, {"int", "float"})

    def test_object_field_accepts_objects_only(self):
        self._check(object_field, {"object"})

    def test_list_field_accepts_arrays_only(self):
        self._check(list_field, {"array", "string array", "tuple"})

    def test_str_list_field_accepts_arrays_of_strings_only(self):
        self._check(str_list_field, {"string array", "tuple"})
        self.assertEqual(str_list_field({"key": []}, "key", _MISSING), [])

    def test_defaults_match_the_absent_key_defaults_callers_used(self):
        self.assertEqual(str_field({}, "key"), "")
        self.assertEqual(number_field({}, "key"), 0)
        self.assertIsNone(object_field({}, "key"))
        self.assertIsNone(list_field({}, "key"))
        self.assertIsNone(str_list_field({}, "key"))

    def test_non_object_data_still_raises(self):
        # require_object_event_data alone decides which sessions are refused,
        # so the readers must fail on non-object data where dict.get did.
        for data in (None, ["key"], "key", 42, True):
            for reader in (str_field, number_field, object_field, list_field, str_list_field):
                with self.subTest(data=data, reader=reader.__name__):
                    with self.assertRaises(AttributeError):
                        reader(data, "key", _MISSING)


# The arguments GitHub Copilot CLI 1.0.112 sent for an apply_patch call.
_PATCH = (
    "*** Begin Patch\n*** Add File: notes/hello.txt\n+hi\n*** Delete File: README.md\n"
    "*** Add File: README.md\n+demo\n*** End Patch\n"
)


def _tool_call(data) -> TraceEvent:
    return TraceEvent(event_type=EventType.TOOL_CALL, timestamp=1.0, data=data)


class TestToolArguments(unittest.TestCase):
    """tool_arguments decodes the stored arguments and keeps the text stored."""

    def test_objects_serialized_objects_and_text(self):
        from agent_trace.event_data import tool_arguments

        args = {"path": "/app/a.py"}
        cases = (
            ("object", {"arguments": args}, (args, None)),
            ("JSON text", {"arguments": json.dumps(args)}, (args, json.dumps(args))),
            ("Python literal", {"arguments": repr(args)}, (args, repr(args))),
            ("patch text", {"arguments": _PATCH}, ({}, _PATCH)),
            ("tool_input object", {"tool_input": args}, (args, None)),
            ("tool_input JSON text", {"tool_input": json.dumps(args)}, (args, json.dumps(args))),
            ("patch text, then a tool_input object", {"arguments": _PATCH, "tool_input": args}, (args, _PATCH)),
            ("array text", {"arguments": "[1]"}, ({}, "[1]")),
            ("number", {"arguments": 42}, ({}, None)),
            ("absent", {}, ({}, None)),
        )
        for label, data, expected in cases:
            with self.subTest(label):
                self.assertEqual(tool_arguments(_tool_call({"tool_name": "t", **data})), expected)

    def test_data_that_is_not_an_object(self):
        from agent_trace.event_data import tool_arguments

        for data in ("text", 42, None, ["arguments"]):
            with self.subTest(data=data):
                self.assertEqual(tool_arguments(_tool_call(data)), ({}, None))

    def test_over_the_limit_only_json_text_is_decoded(self):
        from agent_trace.event_data import MAX_ARGUMENT_CHARS, tool_arguments

        args = {"path": "/app/a.py", "content": "x" * MAX_ARGUMENT_CHARS}
        text = json.dumps(args)
        self.assertEqual(tool_arguments(_tool_call({"arguments": text})), (args, text))
        literal = repr(args)
        self.assertTrue(literal.startswith("{'path': "), literal[:20])
        self.assertEqual(tool_arguments(_tool_call({"arguments": literal})), ({}, literal))


class TestPatchPaths(unittest.TestCase):
    """patch_paths reads the path on each patch marker line, once each."""

    def test_marker_lines(self):
        from agent_trace.event_data import patch_paths

        cases = (
            ("captured Copilot patch", _PATCH, ["notes/hello.txt", "README.md"]),
            (
                "move",
                "*** Begin Patch\n*** Update File: src/old.env\n*** Move to: config/new.env\n"
                "@@\n-a\n+b\n*** End Patch\n",
                ["src/old.env", "config/new.env"],
            ),
            ("CRLF", "*** Begin Patch\r\n*** Add File: a.py\r\n+x\r\n*** End Patch\r\n", ["a.py"]),
            ("no Begin Patch", "*** Update File: a.py\n@@\n-a\n+b\n", ["a.py"]),
            ("trailing spaces", "*** Add File: a.py  \n", ["a.py"]),
            ("space in the path", "*** Add File: docs/my notes.md\n", ["docs/my notes.md"]),
        )
        for label, text, expected in cases:
            with self.subTest(label):
                self.assertEqual(patch_paths(text), expected)

    def test_content_and_malformed_lines_name_no_path(self):
        from agent_trace.event_data import patch_paths

        for text in (
            "-*** Add File: a.py",
            "+*** Add File: a.py",
            " *** Delete File: a.py",
            "@@ *** Move to: a.py",
            "  *** Update File: a.py",
            "*** Add File: ",
            "*** Add File:a.py",
            "*** End Patch",
            "",
        ):
            with self.subTest(text=text):
                self.assertEqual(patch_paths(text), [])


if __name__ == "__main__":
    unittest.main()
