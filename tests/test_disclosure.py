import json
import types
import unittest

from agent_trace.disclosure import (
    NonObjectEventDataError,
    build_disclosure_preview,
    require_object_event_data,
)
from agent_trace.models import EventType, TraceEvent

_NON_OBJECT_DATA = (
    (["SECRET-ARRAY", {"command": "SECRET-COMMAND"}], "array"),
    ("SECRET-STRING", "string"),
    (None, "null"),
    (42, "number"),
    (4.5, "number"),
    (True, "boolean"),
)


class TestDisclosurePreview(unittest.TestCase):
    def test_counts_content_categories_without_values(self):
        events = [
            TraceEvent(
                event_type=EventType.USER_PROMPT,
                data={"content": "SECRET-PROMPT"},
                redacted=True,
            ),
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={"tool": "shell", "command": "SECRET-COMMAND", "cwd": "/private/path"},
            ),
            TraceEvent(
                event_type=EventType.TOOL_RESULT,
                data={"output": "SECRET-RESULT"},
            ),
            TraceEvent(
                event_type=EventType.FILE_WRITE,
                data={"path": "/private/file.txt"},
            ),
            TraceEvent(
                event_type=EventType.ASSISTANT_RESPONSE,
                data={"content": "SECRET-RESPONSE"},
            ),
        ]

        preview = build_disclosure_preview(events, annotation_count=2)
        payload = json.dumps(preview.to_dict(), sort_keys=True)

        self.assertEqual(preview.mode, "minimized")
        self.assertEqual(preview.event_count, 5)
        self.assertEqual(preview.prompt_count, 1)
        self.assertEqual(preview.response_count, 1)
        self.assertEqual(preview.tool_input_count, 1)
        self.assertEqual(preview.tool_result_count, 1)
        self.assertEqual(preview.command_count, 1)
        self.assertEqual(preview.path_count, 2)
        self.assertEqual(preview.annotation_count, 2)
        self.assertEqual(preview.content_rich_event_count, 5)
        self.assertEqual(preview.redacted_event_count, 1)
        self.assertEqual(preview.unredacted_content_rich_event_count, 4)

        for secret in (
            "SECRET-PROMPT",
            "SECRET-COMMAND",
            "SECRET-RESULT",
            "SECRET-RESPONSE",
            "/private/path",
            "/private/file.txt",
        ):
            self.assertNotIn(secret, payload)

    def test_hook_style_nested_arguments_count_without_exposing_values(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={
                "tool_name": "Bash",
                "arguments": {
                    "command": "SECRET-EXECUTION",
                    "file_path": "/private/hook-file",
                },
            },
        )

        preview = build_disclosure_preview([event])
        self.assertEqual(preview.command_count, 1)
        self.assertEqual(preview.path_count, 1)
        self.assertEqual(preview.tool_input_count, 1)
        serialized = json.dumps(preview.to_dict())
        self.assertNotIn("SECRET-EXECUTION", serialized)
        self.assertNotIn("/private/hook-file", serialized)

    def test_mcp_proxy_launch_command_counts_without_exposing_values(self):
        # proxy.py records the MCP server launch command on session_start, and share
        # HTML prints it in the event summary and the JSON detail.
        events = [
            TraceEvent(
                event_type=EventType.SESSION_START,
                data={"command": [
                    "npx", "-y", "server-fs", "/private/mcp-repo", "--token=SECRET-TOKEN",
                ]},
            ),
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={"tool_name": "list_files", "arguments": {"limit": 5}},
            ),
        ]

        preview = build_disclosure_preview(events, include_content=True)
        self.assertEqual(preview.command_count, 1)
        self.assertEqual(preview.tool_input_count, 1)
        # Values are never inspected, so a path inside a command line is covered by
        # the command-bearing count rather than the path count.
        self.assertEqual(preview.path_count, 0)
        serialized = json.dumps(preview.to_dict())
        self.assertNotIn("SECRET-TOKEN", serialized)
        self.assertNotIn("/private/mcp-repo", serialized)

    def test_command_keys_count_on_every_event_type(self):
        for event_type in EventType:
            for key in ("command", "cmd", "argv"):
                with self.subTest(event_type=event_type.value, key=key):
                    event = TraceEvent(event_type=event_type, data={key: "SECRET-COMMAND"})
                    preview = build_disclosure_preview([event])
                    self.assertEqual(preview.command_count, 1)
                    self.assertNotIn("SECRET-COMMAND", json.dumps(preview.to_dict()))

    def test_include_content_changes_mode_not_preview_values(self):
        event = TraceEvent(
            event_type=EventType.USER_PROMPT,
            data={"content": "do not echo this"},
        )
        preview = build_disclosure_preview([event], include_content=True)
        self.assertEqual(preview.mode, "include_content")
        self.assertNotIn("do not echo this", json.dumps(preview.to_dict()))

    def test_negative_annotation_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_disclosure_preview([], annotation_count=-1)

    def test_non_object_data_is_rejected_by_position_and_type(self):
        # Key names cannot be read from these payloads, so counting the event as one
        # without commands or paths would under-count.
        for data, json_type in _NON_OBJECT_DATA:
            for include_content in (False, True):
                with self.subTest(json_type=json_type, data=data, include_content=include_content):
                    events = [
                        TraceEvent(event_type=EventType.SESSION_START, data={}),
                        TraceEvent(event_type=EventType.TOOL_CALL, data=data),
                        TraceEvent(event_type=EventType.SESSION_END, data={"exit_code": 0}),
                    ]
                    with self.assertRaises(NonObjectEventDataError) as caught:
                        build_disclosure_preview(events, include_content=include_content)
                    self.assertIsInstance(caught.exception, ValueError)
                    self.assertEqual(
                        str(caught.exception),
                        "session event data must be an object: "
                        f"event 2 of 3 (tool_call) has {json_type} data",
                    )

    def test_rejection_names_first_event_and_counts_the_rest(self):
        events = [
            TraceEvent(event_type=EventType.USER_PROMPT, data={"prompt": "SECRET-PROMPT"}),
            TraceEvent(event_type=EventType.TOOL_RESULT, data="SECRET-RESULT"),
            TraceEvent(event_type=EventType.ERROR, data=["SECRET-ERROR"]),
            TraceEvent(event_type=EventType.FILE_READ, data=7),
        ]
        prefix = "session event data must be an object: "
        for count, suffix in (
            (2, "event 2 of 2 (tool_result) has string data"),
            (3, "event 2 of 3 (tool_result) has string data; 1 more event has non-object data"),
            (4, "event 2 of 4 (tool_result) has string data; 2 more events have non-object data"),
        ):
            with self.subTest(count=count):
                with self.assertRaises(NonObjectEventDataError) as caught:
                    require_object_event_data(events[:count])
                self.assertEqual(str(caught.exception), prefix + suffix)
                self.assertNotIn("SECRET", str(caught.exception))

    def test_object_data_passes_the_check(self):
        require_object_event_data([])
        require_object_event_data([
            TraceEvent(event_type=EventType.SESSION_START),
            TraceEvent(event_type=EventType.TOOL_CALL, data={"command": "SECRET-COMMAND"}),
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data=types.MappingProxyType({"cwd": "/private/path"}),
            ),
        ])
        preview = build_disclosure_preview([
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data=types.MappingProxyType({"cwd": "/private/path"}),
            ),
        ])
        self.assertEqual((preview.event_count, preview.path_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
