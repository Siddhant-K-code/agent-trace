import json
import unittest

from agent_trace.disclosure import build_disclosure_preview
from agent_trace.models import EventType, TraceEvent


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


if __name__ == "__main__":
    unittest.main()
