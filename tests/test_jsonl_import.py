"""Tests for Claude Code JSONL import."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.jsonl_import import import_jsonl
from agent_trace.models import EventType
from agent_trace.store import TraceStore


class TestJSONLImport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def _write_jsonl(self, entries):
        """Write entries to a temp JSONL file."""
        path = os.path.join(self.tmpdir, "session.jsonl")
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return path

    def test_import_basic_session(self):
        path = self._write_jsonl([
            {
                "type": "user",
                "sessionId": "abc123",
                "gitBranch": "main",
                "version": "2.1.50",
                "timestamp": "2025-01-01T00:00:00Z",
                "message": {"role": "user", "content": "Hello"},
                "uuid": "u1",
            },
            {
                "type": "assistant",
                "sessionId": "abc123",
                "timestamp": "2025-01-01T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "Hi there!"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
                "uuid": "a1",
                "parentUuid": "u1",
            },
        ])

        session_id = import_jsonl(path, store=self.store)
        self.assertTrue(session_id)

        meta = self.store.load_meta(session_id)
        self.assertEqual(meta.agent_name, "claude-code")
        self.assertEqual(meta.total_tokens, 150)
        self.assertEqual(meta.llm_requests, 1)

        events = self.store.load_events(session_id)
        types = [e.event_type for e in events]
        self.assertIn(EventType.USER_PROMPT, types)
        self.assertIn(EventType.ASSISTANT_RESPONSE, types)
        self.assertIn(EventType.SESSION_END, types)

    def test_import_tool_calls(self):
        path = self._write_jsonl([
            {
                "type": "assistant",
                "sessionId": "def456",
                "timestamp": "2025-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool1",
                            "name": "Bash",
                            "input": {"command": "ls -la"},
                            "caller": {"type": "direct"},
                        }
                    ],
                    "usage": {"input_tokens": 50, "output_tokens": 20},
                },
                "uuid": "a1",
            },
        ])

        session_id = import_jsonl(path, store=self.store)
        meta = self.store.load_meta(session_id)
        self.assertEqual(meta.tool_calls, 1)

        events = self.store.load_events(session_id)
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].data["tool_name"], "Bash")
        self.assertEqual(tool_calls[0].data["arguments"]["command"], "ls -la")

    def test_import_subagent(self):
        path = self._write_jsonl([
            {
                "type": "assistant",
                "sessionId": "ghi789",
                "timestamp": "2025-01-01T00:00:00Z",
                "isSidechain": False,
                "message": {
                    "role": "assistant",
                    "model": "claude-opus-4-20250514",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "agent1",
                            "name": "Agent",
                            "input": {
                                "description": "Search code",
                                "subagent_type": "Explore",
                                "prompt": "Find all files",
                            },
                            "caller": {"type": "direct"},
                        }
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
                "uuid": "a1",
            },
        ])

        session_id = import_jsonl(path, store=self.store)
        events = self.store.load_events(session_id)
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].data["tool_name"], "Agent")
        self.assertEqual(tool_calls[0].data["subagent_type"], "Explore")

    def test_import_queue_operations_skipped(self):
        path = self._write_jsonl([
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "sessionId": "q1",
                "timestamp": "2025-01-01T00:00:00Z",
            },
            {
                "type": "user",
                "sessionId": "q1",
                "timestamp": "2025-01-01T00:00:01Z",
                "message": {"role": "user", "content": "test"},
                "uuid": "u1",
            },
        ])

        session_id = import_jsonl(path, store=self.store)
        events = self.store.load_events(session_id)
        # queue-operation should not become an event
        for e in events:
            self.assertNotEqual(e.data.get("type"), "queue-operation")

    def test_import_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            import_jsonl("/nonexistent/path.jsonl", store=self.store)


if __name__ == "__main__":
    unittest.main()
