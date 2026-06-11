"""Tests for Claude Code hooks integration."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.hooks import (
    _read_active_session,
    _write_active_session,
    _clear_active_session,
    _read_pending_calls,
    _write_pending_calls,
    handle_session_start,
    handle_session_end,
    handle_pre_tool,
    handle_post_tool,
    handle_user_prompt,
    handle_stop,
)
from agent_trace.models import EventType
from agent_trace.store import TraceStore


class TestHooksSessionLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_REDACT", None)

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)

    def test_session_start_creates_session(self):
        handle_session_start({
            "session_id": "abc123def456ghij",
            "source": "startup",
            "model": "claude-sonnet-4-6",
        })

        session_id = _read_active_session()
        self.assertIsNotNone(session_id)

        store = TraceStore(self.tmpdir)
        meta = store.load_meta(session_id)
        self.assertEqual(meta.agent_name, "claude-code")
        self.assertIn("startup", meta.command)

        events = store.load_events(session_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.SESSION_START)
        self.assertEqual(events[0].data["mode"], "claude-code-hooks")

    def test_session_end_closes_session(self):
        handle_session_start({"session_id": "test1234test5678", "source": "startup"})
        session_id = _read_active_session()

        handle_session_end({})

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        end_events = [e for e in events if e.event_type == EventType.SESSION_END]
        self.assertEqual(len(end_events), 1)

        meta = store.load_meta(session_id)
        self.assertIsNotNone(meta.ended_at)
        self.assertGreater(meta.total_duration_ms, 0)

        # active session should be cleared
        self.assertIsNone(_read_active_session())

    def test_session_end_without_start_is_noop(self):
        _clear_active_session()
        handle_session_end({})  # should not raise


class TestHooksToolCapture(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_REDACT", None)
        handle_session_start({"session_id": "tooltest12345678", "source": "startup"})
        self.session_id = _read_active_session()

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)

    def test_pre_tool_logs_tool_call(self):
        handle_pre_tool({
            "tool_name": "Bash",
            "tool_input": {"command": "npm test"},
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].data["tool_name"], "Bash")
        self.assertEqual(tool_calls[0].data["arguments"]["command"], "npm test")

    def test_post_tool_logs_tool_result(self):
        handle_pre_tool({
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/test.py"},
        })
        handle_post_tool({
            "tool_name": "Read",
            "tool_output": "print('hello')",
        }, failed=False)

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].data["tool_name"], "Read")
        self.assertIn("hello", results[0].data["result"])

    def test_post_tool_links_to_pre_tool(self):
        handle_pre_tool({
            "tool_name": "Edit",
            "tool_input": {"file_path": "/tmp/test.py", "old_text": "a", "new_text": "b"},
        })
        handle_post_tool({
            "tool_name": "Edit",
            "tool_output": "File edited",
        }, failed=False)

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        results = [e for e in events if e.event_type == EventType.TOOL_RESULT]

        self.assertEqual(results[0].parent_id, calls[0].event_id)
        self.assertGreater(results[0].duration_ms, 0)

    def test_post_tool_failure_logs_error(self):
        handle_pre_tool({
            "tool_name": "Bash",
            "tool_input": {"command": "false"},
        })
        handle_post_tool({
            "tool_name": "Bash",
            "tool_output": "Command failed with exit code 1",
        }, failed=True)

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        self.assertEqual(len(errors), 1)
        self.assertIn("failed", errors[0].data["error"])

        meta = store.load_meta(self.session_id)
        self.assertEqual(meta.errors, 1)

    def test_meta_tool_count_increments(self):
        for i in range(3):
            handle_pre_tool({
                "tool_name": f"Tool{i}",
                "tool_input": {},
            })

        store = TraceStore(self.tmpdir)
        meta = store.load_meta(self.session_id)
        self.assertEqual(meta.tool_calls, 3)

    def test_captures_all_tool_types(self):
        """Verify we capture non-MCP tools like Bash, Edit, Write, Read, Agent."""
        tools = ["Bash", "Edit", "Write", "Read", "Agent", "Grep", "Glob", "WebFetch"]
        for tool in tools:
            handle_pre_tool({"tool_name": tool, "tool_input": {}})

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        captured_tools = [e.data["tool_name"] for e in tool_calls]

        for tool in tools:
            self.assertIn(tool, captured_tools)

    def test_pre_tool_without_session_is_noop(self):
        _clear_active_session()
        handle_pre_tool({"tool_name": "Bash", "tool_input": {}})  # should not raise

    def test_large_output_truncated(self):
        handle_pre_tool({"tool_name": "Bash", "tool_input": {"command": "cat big.txt"}})
        handle_post_tool({
            "tool_name": "Bash",
            "tool_output": "x" * 5000,
        }, failed=False)

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertIn("truncated", results[0].data["result"])
        self.assertLess(len(results[0].data["result"]), 2000)


class TestHooksUserPrompt(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_REDACT", None)
        handle_session_start({"session_id": "prompttest1234567", "source": "startup"})
        self.session_id = _read_active_session()

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)

    def test_user_prompt_logged(self):
        handle_user_prompt({
            "prompt": "Fix the authentication bug in src/auth.py",
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        prompts = [e for e in events if e.event_type == EventType.USER_PROMPT]
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].data["prompt"], "Fix the authentication bug in src/auth.py")

    def test_multiple_prompts_logged(self):
        handle_user_prompt({"prompt": "First prompt"})
        handle_user_prompt({"prompt": "Second prompt"})
        handle_user_prompt({"prompt": "Third prompt"})

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        prompts = [e for e in events if e.event_type == EventType.USER_PROMPT]
        self.assertEqual(len(prompts), 3)
        self.assertEqual(prompts[0].data["prompt"], "First prompt")
        self.assertEqual(prompts[2].data["prompt"], "Third prompt")

    def test_empty_prompt_logged(self):
        handle_user_prompt({"prompt": ""})

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        prompts = [e for e in events if e.event_type == EventType.USER_PROMPT]
        self.assertEqual(len(prompts), 1)

    def test_prompt_without_session_is_noop(self):
        _clear_active_session()
        handle_user_prompt({"prompt": "test"})  # should not raise


class TestHooksStop(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_REDACT", None)
        handle_session_start({"session_id": "stoptest12345678", "source": "startup"})
        self.session_id = _read_active_session()

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)

    def test_stop_logs_assistant_response(self):
        handle_stop({
            "last_assistant_message": "I've fixed the bug. The issue was in the token refresh logic.",
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        responses = [e for e in events if e.event_type == EventType.ASSISTANT_RESPONSE]
        self.assertEqual(len(responses), 1)
        self.assertIn("fixed the bug", responses[0].data["text"])

    def test_stop_skips_when_stop_hook_active(self):
        handle_stop({
            "stop_hook_active": True,
            "last_assistant_message": "Should not be logged",
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        responses = [e for e in events if e.event_type == EventType.ASSISTANT_RESPONSE]
        self.assertEqual(len(responses), 0)

    def test_stop_records_marker_without_message(self):
        handle_stop({
            "last_assistant_message": "",
            "stop_reason": "end_turn",
            "transcript_path": "/tmp/transcript.jsonl",
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        responses = [e for e in events if e.event_type == EventType.ASSISTANT_RESPONSE]
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].data["hook_event"], "stop")
        self.assertEqual(responses[0].data["stop_reason"], "end_turn")
        self.assertEqual(responses[0].data["transcript_path"], "/tmp/transcript.jsonl")

    def test_stop_without_session_is_noop(self):
        _clear_active_session()
        handle_stop({"last_assistant_message": "test"})  # should not raise

    def test_full_conversation_flow(self):
        """Simulate a real Claude Code turn: prompt -> tools -> response."""
        handle_user_prompt({"prompt": "Fix the login bug"})
        handle_pre_tool({"tool_name": "Read", "tool_input": {"file_path": "/src/auth.py"}})
        handle_post_tool({"tool_name": "Read", "tool_output": "def login(): ..."}, failed=False)
        handle_pre_tool({"tool_name": "Edit", "tool_input": {"file_path": "/src/auth.py", "old_string": "a", "new_string": "b"}})
        handle_post_tool({"tool_name": "Edit", "tool_output": "File edited"}, failed=False)
        handle_stop({"last_assistant_message": "I've fixed the login bug by updating the auth logic."})

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)

        # Filter out session_start
        events = [e for e in events if e.event_type != EventType.SESSION_START]

        # Verify the full flow
        self.assertEqual(events[0].event_type, EventType.USER_PROMPT)
        self.assertEqual(events[1].event_type, EventType.TOOL_CALL)
        self.assertEqual(events[1].data["tool_name"], "Read")
        self.assertEqual(events[2].event_type, EventType.TOOL_RESULT)
        self.assertEqual(events[3].event_type, EventType.TOOL_CALL)
        self.assertEqual(events[3].data["tool_name"], "Edit")
        self.assertEqual(events[4].event_type, EventType.TOOL_RESULT)
        self.assertEqual(events[5].event_type, EventType.ASSISTANT_RESPONSE)
        self.assertIn("fixed the login bug", events[5].data["text"])


class TestHooksRedaction(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ["AGENT_TRACE_REDACT"] = "1"
        handle_session_start({"session_id": "redacttest123456", "source": "startup"})
        self.session_id = _read_active_session()

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_REDACT", None)

    def test_secrets_redacted_in_user_prompt(self):
        handle_user_prompt({
            "prompt": "Use this API key: sk-abc123def456ghi789jkl012mno345pqr678",
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        prompts = [e for e in events if e.event_type == EventType.USER_PROMPT]
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("sk-abc123", str(prompts[0].data))

    def test_secrets_redacted_in_tool_input(self):
        handle_pre_tool({
            "tool_name": "Bash",
            "tool_input": {
                "command": "curl -H 'Authorization: Bearer sk-abc123def456ghi789jkl012mno345pqr678' https://api.example.com",
            },
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        self.assertEqual(len(tool_calls), 1)
        # The command string should have the key redacted
        cmd = str(tool_calls[0].data)
        self.assertNotIn("sk-abc123", cmd)


class TestPendingCalls(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_CLAUDE_SESSION_ID", None)

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_CLAUDE_SESSION_ID", None)

    def test_read_write_pending_calls(self):
        _write_pending_calls({"abc123": {"tool_name": "Bash", "timestamp": 1.0}})
        calls = _read_pending_calls()
        self.assertEqual(calls["abc123"]["tool_name"], "Bash")

    def test_read_empty_pending_calls(self):
        calls = _read_pending_calls()
        self.assertEqual(calls, {})

    def test_active_session_lifecycle(self):
        _write_active_session("test123")
        self.assertEqual(_read_active_session(), "test123")
        _clear_active_session()
        self.assertIsNone(_read_active_session())


class TestConcurrentToolCalls(unittest.TestCase):
    """Verify that two concurrent calls to the same tool don't collide."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_CLAUDE_SESSION_ID", None)
        handle_session_start({"session_id": "concurrent1234567", "source": "startup"})
        self.session_id = _read_active_session()

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_CLAUDE_SESSION_ID", None)

    def test_two_concurrent_same_tool_calls_linked_correctly(self):
        """Two Bash calls in flight simultaneously must each link to their own result."""
        handle_pre_tool({"tool_name": "Bash", "tool_input": {"command": "echo first"}})
        handle_pre_tool({"tool_name": "Bash", "tool_input": {"command": "echo second"}})

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        self.assertEqual(len(calls), 2)

        # Resolve first call
        handle_post_tool({"tool_name": "Bash", "tool_output": "first"}, failed=False)
        # Resolve second call
        handle_post_tool({"tool_name": "Bash", "tool_output": "second"}, failed=False)

        events = store.load_events(self.session_id)
        results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(len(results), 2)

        # Both results must have a parent_id pointing to one of the calls
        call_ids = {c.event_id for c in calls}
        for result in results:
            self.assertIn(result.parent_id, call_ids, "result must link to a call event_id")

        # The two results must link to different call events
        linked_ids = {r.parent_id for r in results}
        self.assertEqual(len(linked_ids), 2, "each result must link to a distinct call")

    def test_pending_calls_keyed_by_event_id(self):
        """Pending calls file must use event_id as key, not tool name."""
        handle_pre_tool({"tool_name": "Read", "tool_input": {"file_path": "/a"}})
        handle_pre_tool({"tool_name": "Read", "tool_input": {"file_path": "/b"}})

        pending = _read_pending_calls()
        self.assertEqual(len(pending), 2)
        for key, val in pending.items():
            # key must look like a UUID hex fragment, not a tool name
            self.assertNotEqual(key, "Read")
            self.assertEqual(val["tool_name"], "Read")


class TestConcurrentAgentIsolation(unittest.TestCase):
    """Verify that two agents sharing AGENT_TRACE_DIR don't corrupt each other."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_CLAUDE_SESSION_ID", None)

    def test_separate_claude_sessions_use_separate_state_files(self):
        """Two Claude Code sessions must write to different .active-session files."""
        # Agent 1
        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = "agent1session00001"
        handle_session_start({"session_id": "agent1session00001", "source": "startup"})
        sid1 = _read_active_session()

        # Agent 2
        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = "agent2session00002"
        handle_session_start({"session_id": "agent2session00002", "source": "startup"})
        sid2 = _read_active_session()

        self.assertNotEqual(sid1, sid2)

        # Switch back to agent 1 — its session must still be intact
        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = "agent1session00001"
        self.assertEqual(_read_active_session(), sid1)


if __name__ == "__main__":
    unittest.main()
