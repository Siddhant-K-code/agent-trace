"""Tests for policy suggestion (issue #19)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.policy import (
    _collapse_commands,
    _collapse_paths,
    observe_session,
    render_policy_json,
    suggest_policy,
)
from agent_trace.store import TraceStore


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


if __name__ == "__main__":
    unittest.main()
