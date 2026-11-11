"""Tests for Claude Code JSONL import."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cost import build_provider_cost_report
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
        response = next(
            event for event in events
            if event.event_type == EventType.ASSISTANT_RESPONSE
        )
        self.assertEqual(response.data["usage"]["input_tokens"], 100)
        self.assertEqual(response.data["context_stream"], "main")

    def test_import_counts_and_prices_each_response_usage_once(self):
        # Claude Code writes one line per content block of a response, and
        # every line repeats the response's usage.  Streaming can leave a
        # smaller output count on the earlier lines.
        usage = {
            "input_tokens": 10,
            "output_tokens": 200,
            "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 5000,
        }
        lines = []
        for index, (content, output_tokens) in enumerate((
            ({"type": "thinking", "thinking": "Look first."}, 1),
            ({"type": "text", "text": "Listing the files."}, 1),
            ({"type": "tool_use", "id": "tool1", "name": "Bash",
              "input": {"command": "ls"}}, 200),
        )):
            lines.append({
                "type": "assistant",
                "sessionId": "dup123",
                "timestamp": f"2025-01-01T00:00:0{index + 1}Z",
                "message": {
                    "id": "msg_1",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [content],
                    "usage": {**usage, "output_tokens": output_tokens},
                },
                "uuid": f"a{index}",
            })
        path = self._write_jsonl([
            {
                "type": "user",
                "sessionId": "dup123",
                "timestamp": "2025-01-01T00:00:00Z",
                "message": {"role": "user", "content": "List the files"},
                "uuid": "u1",
            },
            *lines,
        ])

        session_id = import_jsonl(path, store=self.store)
        meta = self.store.load_meta(session_id)
        self.assertEqual(meta.llm_requests, 1)
        self.assertEqual(meta.total_tokens, 10 + 200 + 1000 + 5000)

        events = self.store.load_events(session_id)
        with_usage = [e for e in events if "usage" in e.data]
        self.assertEqual(len(with_usage), 1)
        self.assertEqual(with_usage[0].data["usage"]["output_tokens"], 200)
        texts = [e.data.get("text") for e in events
                 if e.event_type == EventType.ASSISTANT_RESPONSE and e.data.get("text")]
        self.assertEqual(texts, ["Listing the files."])

        # claude-sonnet-4-6: input 3, cache read 0.30, 5m write 3.75, output 15.
        report = build_provider_cost_report(self.store, since="2020-01-01")
        [record] = report.records
        self.assertEqual(
            (record.model, record.input_tokens, record.output_tokens,
             record.cache_read_tokens, record.cache_write_tokens),
            ("claude-sonnet-4-6", 6010, 200, 5000, 1000),
        )
        self.assertAlmostEqual(
            record.cost_usd, (10 * 3 + 5000 * 0.30 + 1000 * 3.75 + 200 * 15) / 1e6, places=12,
        )

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

    def test_import_preserves_stable_sidechain_identity(self):
        path = self._write_jsonl([
            {
                "type": "assistant",
                "sessionId": "sidechains",
                "timestamp": "2025-01-01T00:00:00Z",
                "isSidechain": True,
                "agentId": "research",
                "uuid": "a1",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4",
                    "content": [],
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                },
            },
            {
                "type": "assistant",
                "sessionId": "sidechains",
                "timestamp": "2025-01-01T00:00:01Z",
                "isSidechain": True,
                "agentId": "research",
                "uuid": "a2",
                "parentUuid": "a1",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4",
                    "content": [],
                    "usage": {"input_tokens": 90, "output_tokens": 1},
                },
            },
            {
                "type": "assistant",
                "sessionId": "sidechains",
                "timestamp": "2025-01-01T00:00:02Z",
                "isSidechain": True,
                "agentId": "review",
                "uuid": "b1",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4",
                    "content": [],
                    "usage": {"input_tokens": 80, "output_tokens": 1},
                },
            },
        ])

        session_id = import_jsonl(path, store=self.store)
        responses = [
            event for event in self.store.load_events(session_id)
            if event.event_type == EventType.ASSISTANT_RESPONSE
        ]

        self.assertEqual(responses[0].data["context_stream"], "sidechain:research")
        self.assertEqual(responses[1].data["context_stream"], "sidechain:research")
        self.assertEqual(responses[2].data["context_stream"], "sidechain:review")

    def test_boundary_embedded_summary_is_not_overwritten_by_next_user(self):
        path = self._write_jsonl([
            {
                "type": "system",
                "subtype": "compact_boundary",
                "sessionId": "embedded-summary",
                "timestamp": "2025-01-01T00:00:00Z",
                "uuid": "boundary",
                "content": "Retain the Redis compatibility constraint.",
            },
            {
                "type": "user",
                "sessionId": "embedded-summary",
                "timestamp": "2025-01-01T00:00:01Z",
                "uuid": "user",
                "parentUuid": "boundary",
                "message": {"role": "user", "content": "Please continue."},
            },
            {
                "type": "assistant",
                "sessionId": "embedded-summary",
                "timestamp": "2025-01-01T00:00:02Z",
                "uuid": "assistant",
                "parentUuid": "user",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4",
                    "content": [],
                    "usage": {"input_tokens": 10_000},
                },
            },
        ])

        session_id = import_jsonl(path, store=self.store)
        events = self.store.load_events(session_id)
        prompt = next(e for e in events if e.event_type == EventType.USER_PROMPT)
        response = next(
            e for e in events if e.event_type == EventType.ASSISTANT_RESPONSE
        )

        self.assertNotIn("is_compaction_summary", prompt.data)
        self.assertEqual(
            response.data["context_summary"],
            "Retain the Redis compatibility constraint.",
        )

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

    def test_import_records_cwd_in_session_start(self):
        path = self._write_jsonl([
            {"type": "user", "sessionId": "cwd1", "timestamp": "2025-01-01T00:00:00Z",
             "message": {"role": "user", "content": "hi"}, "uuid": "u1"},
            {"type": "user", "sessionId": "cwd1", "cwd": "/work/repo", "timestamp": "2025-01-01T00:00:01Z",
             "message": {"role": "user", "content": "again"}, "uuid": "u2"},
        ])

        session_id = import_jsonl(path, store=self.store)
        events = self.store.load_events(session_id)

        self.assertEqual(events[0].event_type, EventType.SESSION_START)
        self.assertEqual(events[0].data["cwd"], "/work/repo")
        self.assertEqual(events[0].data["provider"], "claude")
        self.assertEqual(sum(e.event_type == EventType.SESSION_START for e in events), 1)

    def test_import_without_cwd_has_no_session_start(self):
        path = self._write_jsonl([
            {"type": "user", "sessionId": "nocwd", "timestamp": "2025-01-01T00:00:00Z",
             "message": {"role": "user", "content": "hi"}, "uuid": "u1"},
        ])

        session_id = import_jsonl(path, store=self.store)
        types = [e.event_type for e in self.store.load_events(session_id)]

        self.assertNotIn(EventType.SESSION_START, types)

    def test_import_stores_claude_session_id_as_conversation_id(self):
        native = "5e2d8c1a-93b4-4f7e-8a6d-0c1b2a3f4e5d"
        path = self._write_jsonl([
            {"type": "user", "sessionId": native, "timestamp": "2025-01-01T00:00:00Z",
             "message": {"role": "user", "content": "Hi"}, "uuid": "u1"},
        ])
        session_id = import_jsonl(path, store=self.store)
        meta = self.store.load_meta(session_id)
        self.assertEqual(meta.conversation_id, native)

    def test_import_without_session_id_has_no_conversation_id(self):
        path = self._write_jsonl([
            {"type": "user", "timestamp": "2025-01-01T00:00:00Z",
             "message": {"role": "user", "content": "Hi"}, "uuid": "u1"},
        ])
        session_id = import_jsonl(path, store=self.store)
        self.assertEqual(session_id, "session")
        self.assertEqual(self.store.load_meta(session_id).conversation_id, "")


if __name__ == "__main__":
    unittest.main()
