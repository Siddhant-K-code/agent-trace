"""Tests for replay formatting."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, TraceEvent
from agent_trace.replay import C, _strip_markdown, _tool_call_detail, format_event


class TestStripMarkdown(unittest.TestCase):
    def test_bold(self):
        self.assertEqual(_strip_markdown("**hello**"), "hello")

    def test_italic(self):
        self.assertEqual(_strip_markdown("*hello*"), "hello")

    def test_inline_code(self):
        self.assertEqual(_strip_markdown("`code`"), "code")

    def test_link(self):
        self.assertEqual(_strip_markdown("[text](http://example.com)"), "text")

    def test_header(self):
        self.assertEqual(_strip_markdown("## Header"), "Header")

    def test_mixed(self):
        result = _strip_markdown("**75 tests**, all passing in `3.60s`.")
        self.assertEqual(result, "75 tests, all passing in 3.60s.")

    def test_multiline_collapsed(self):
        result = _strip_markdown("line one\n\nline two\nline three")
        self.assertEqual(result, "line one line two line three")

    def test_plain_text_unchanged(self):
        self.assertEqual(_strip_markdown("hello world"), "hello world")

    def test_table_stripped(self):
        table = "| File | Tests |\n|---|---|\n| test_a.py | 7 |\n| test_b.py | 25 |"
        result = _strip_markdown(table)
        self.assertNotIn("|---|", result)
        self.assertIn("test_a.py", result)
        self.assertIn("7", result)

    def test_table_separator_removed(self):
        result = _strip_markdown("|---|---|")
        self.assertEqual(result, "")

    def test_code_block_removed(self):
        result = _strip_markdown("before\n```python\nprint('hi')\n```\nafter")
        self.assertNotIn("print", result)
        self.assertIn("before", result)
        self.assertIn("after", result)

    def test_list_markers_stripped(self):
        result = _strip_markdown("- item one\n- item two\n* item three")
        self.assertIn("item one", result)
        self.assertNotIn("- ", result)
        self.assertNotIn("* ", result)

    def test_numbered_list_stripped(self):
        result = _strip_markdown("1. first\n2. second")
        self.assertIn("first", result)
        self.assertNotIn("1.", result)

    def test_multiple_spaces_collapsed(self):
        result = _strip_markdown("hello    world")
        self.assertEqual(result, "hello world")


class TestToolCallDetail(unittest.TestCase):
    def test_bash_command(self):
        result = _tool_call_detail("Bash", {"command": "npm test", "description": "run tests"})
        self.assertEqual(result, "$ npm test")

    def test_read_file(self):
        result = _tool_call_detail("Read", {"file_path": "/src/main.py"})
        self.assertEqual(result, "/src/main.py")

    def test_write_file(self):
        result = _tool_call_detail("Write", {"file_path": "/src/out.txt"})
        self.assertEqual(result, "/src/out.txt")

    def test_edit_file(self):
        result = _tool_call_detail("Edit", {"file_path": "/src/main.py", "old_string": "foo", "new_string": "bar"})
        self.assertIn("/src/main.py", result)
        self.assertIn("foo", result)

    def test_glob_pattern(self):
        result = _tool_call_detail("Glob", {"pattern": "tests/**/*.py"})
        self.assertEqual(result, "tests/**/*.py")

    def test_grep_pattern(self):
        result = _tool_call_detail("Grep", {"pattern": "TODO", "path": "/src"})
        self.assertEqual(result, "/TODO/ /src")

    def test_webfetch_url(self):
        result = _tool_call_detail("WebFetch", {"url": "https://example.com"})
        self.assertEqual(result, "https://example.com")

    def test_websearch_query(self):
        result = _tool_call_detail("WebSearch", {"query": "python async"})
        self.assertEqual(result, "python async")

    def test_agent_prompt(self):
        result = _tool_call_detail("Agent", {"prompt": "Find all API endpoints"})
        self.assertEqual(result, "Find all API endpoints")

    def test_empty_args(self):
        result = _tool_call_detail("Unknown", {})
        self.assertEqual(result, "")

    def test_long_bash_command_truncated(self):
        cmd = "x" * 200
        result = _tool_call_detail("Bash", {"command": cmd})
        self.assertIn("...", result)
        self.assertLess(len(result), 130)

    def test_mcp_tool_fallback(self):
        result = _tool_call_detail("mcp__custom_tool", {"query": "SELECT * FROM users"})
        self.assertEqual(result, "query: SELECT * FROM users")


class TestFormatEvent(unittest.TestCase):
    def test_tool_call_shows_detail(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id="test",
            data={"tool_name": "Bash", "arguments": {"command": "ls -la"}},
        )
        output = format_event(event, base_ts=event.timestamp)
        self.assertIn("Bash", output)
        self.assertIn("$ ls -la", output)

    def test_tool_result_shows_name_and_preview(self):
        event = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            session_id="test",
            data={"tool_name": "Bash", "result": "file1.py\nfile2.py\nfile3.py"},
            duration_ms=42,
        )
        output = format_event(event, base_ts=event.timestamp)
        self.assertIn("Bash", output)
        self.assertIn("42ms", output)
        self.assertIn("file1.py", output)

    def test_error_shows_message(self):
        event = TraceEvent(
            event_type=EventType.ERROR,
            session_id="test",
            data={"tool_name": "Bash", "error": "Command failed with exit code 1"},
        )
        output = format_event(event, base_ts=event.timestamp)
        self.assertIn("Bash", output)
        self.assertIn("Command failed", output)

    def test_error_shows_message_key(self):
        event = TraceEvent(
            event_type=EventType.ERROR,
            session_id="test",
            data={"message": "Connection refused"},
        )
        output = format_event(event, base_ts=event.timestamp)
        self.assertIn("Connection refused", output)

    def test_assistant_response_strips_markdown(self):
        event = TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE,
            session_id="test",
            data={"text": "**75 tests**, all passing in `3.60s`."},
        )
        output = format_event(event, base_ts=event.timestamp)
        self.assertIn("75 tests", output)
        self.assertNotIn("**", output)
        self.assertNotIn("`", output)

    def test_user_prompt_shows_text(self):
        event = TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id="test",
            data={"prompt": "Fix the login bug"},
        )
        output = format_event(event, base_ts=event.timestamp)
        self.assertIn("Fix the login bug", output)


class TestSubagentAttribution(unittest.TestCase):
    AGENT = "d0000011-0000-4000-8000-000000000011"
    TAG = "[subagent default 00000011]"

    def event(self, event_type, attributed=True, **data):
        if attributed:
            data.update(agent_id=self.AGENT, agent_type="default")
        return TraceEvent(event_type=event_type, session_id="test", timestamp=0.0, data=data)

    def render(self, event):
        return format_event(event, base_ts=0.0)

    def tool_events(self, attributed):
        return [
            self.event(EventType.TOOL_CALL, attributed, tool_name="Bash",
                       arguments={"command": "echo synthetic"}),
            self.event(EventType.TOOL_RESULT, attributed, tool_name="Bash", result="synthetic"),
            self.event(EventType.ERROR, attributed, tool_name="Bash", error="exit code 1"),
        ]

    def test_tool_events_carry_the_subagent_tag(self):
        for event in self.tool_events(attributed=True):
            with self.subTest(event=event.event_type.value):
                self.assertIn(f"Bash{C.RESET} {C.MAGENTA}{self.TAG}{C.RESET}", self.render(event))

    def test_tag_without_agent_type_uses_the_short_id(self):
        event = self.event(EventType.TOOL_CALL, attributed=False, tool_name="Bash",
                           arguments={"command": "true"}, agent_id=self.AGENT)
        self.assertIn("[subagent 00000011]", self.render(event))

    def test_unattributed_tool_events_render_as_before(self):
        for attributed, plain in zip(
            self.tool_events(attributed=True), self.tool_events(attributed=False)
        ):
            with self.subTest(event=plain.event_type.value):
                tagged = self.render(attributed)
                self.assertNotIn("[subagent", self.render(plain))
                self.assertEqual(
                    tagged.replace(f" {C.MAGENTA}{self.TAG}{C.RESET}", ""), self.render(plain)
                )

    def test_subagent_start_and_stop_show_the_agent_id(self):
        for event_type in (EventType.SUBAGENT_START, EventType.SUBAGENT_STOP):
            with self.subTest(event=event_type.value):
                output = self.render(self.event(event_type))
                self.assertIn(f"agent={self.AGENT}", output)
                self.assertNotIn("returned:", output)

    def test_subagent_stop_shows_what_the_subagent_returned(self):
        output = self.render(self.event(
            EventType.SUBAGENT_STOP, text="Synthetic   subagent\nanswer.", stop_reason="done"
        ))
        first, second = output.split("\n")
        self.assertIn(f"agent={self.AGENT}", first)
        self.assertIn("(done)", first)
        self.assertIn('returned: "Synthetic subagent answer."', second)

    def test_returned_preview_is_truncated(self):
        output = self.render(self.event(EventType.SUBAGENT_STOP, text="x" * 250))
        self.assertIn('returned: "' + "x" * 200 + '..."', output)

    def test_unattributed_subagent_events_render_as_before(self):
        for event_type in (EventType.SUBAGENT_START, EventType.SUBAGENT_STOP):
            with self.subTest(event=event_type.value):
                output = self.render(self.event(
                    event_type, attributed=False, agent_type="explore",
                    child_session_id="child", stop_reason="end_turn",
                ))
                self.assertTrue(output.endswith(
                    f"{C.WHITE}explore{C.RESET} {C.DIM}child=child{C.RESET} "
                    f"{C.DIM}(end_turn){C.RESET}"
                ))
                self.assertNotIn("agent=", output)
                self.assertNotIn("returned:", output)
                self.assertNotIn("\n", output)


if __name__ == "__main__":
    unittest.main()
