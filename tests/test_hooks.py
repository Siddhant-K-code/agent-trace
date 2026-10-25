"""Tests for Claude Code hooks integration."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from contextlib import redirect_stderr
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

import agent_trace.store as store_module
from agent_trace.evidence_health import assess_evidence_health
from agent_trace.hooks import (
    _PROVIDER_ENV,
    _active_session_path,
    _read_active_session,
    _write_active_session,
    _clear_active_session,
    _read_pending_calls,
    _write_pending_calls,
    _state_suffix,
    handle_session_start,
    handle_session_end,
    handle_pre_tool,
    handle_post_tool,
    handle_user_prompt,
    handle_stop,
    hook_main,
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


class TestHooksSessionContinuation(unittest.TestCase):
    """resume and compact SessionStart hooks continue the recorded session."""

    SID = "contsession00001"
    STARTED_AT = 1_700_000_000.0

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict(os.environ, {"AGENT_TRACE_DIR": tmp.name})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_TRACE_REDACT", None)
        self.store = TraceStore(tmp.name)

    def start(self, source, session_id=SID):
        handle_session_start({
            "session_id": session_id,
            "source": source,
            "model": "claude-sonnet-4-6",
        })

    def seed_meta(self, session_id=SID):
        meta = self.store.load_meta(session_id)
        meta.started_at = self.STARTED_AT
        meta.tool_calls = 3
        meta.errors = 1
        self.store.update_meta(meta)

    def start_sources(self, session_id=SID):
        return [
            event.data.get("source")
            for event in self.store.load_events(session_id)
            if event.event_type == EventType.SESSION_START
        ]

    def test_compact_keeps_existing_metadata(self):
        self.start("startup")
        self.seed_meta()
        before = asdict(self.store.load_meta(self.SID))

        self.start("compact")

        self.assertEqual(asdict(self.store.load_meta(self.SID)), before)
        self.assertEqual(self.start_sources(), ["startup", "compact"])
        self.assertEqual(_read_active_session(), self.SID)

    def test_resume_then_compact_keeps_metadata_and_reopens_session(self):
        self.start("startup")
        self.seed_meta()
        handle_session_end({"session_id": self.SID})
        ended = asdict(self.store.load_meta(self.SID))
        self.assertIsNotNone(ended["ended_at"])

        self.start("resume")
        self.start("compact")

        self.assertEqual(asdict(self.store.load_meta(self.SID)), dict(ended, ended_at=None))
        self.assertEqual(self.start_sources(), ["startup", "resume", "compact"])
        self.assertEqual(_read_active_session(), self.SID)

        handle_session_end({"session_id": self.SID})
        meta = self.store.load_meta(self.SID)
        self.assertEqual(meta.started_at, self.STARTED_AT)
        self.assertAlmostEqual(
            meta.total_duration_ms,
            (meta.ended_at - self.STARTED_AT) * 1000,
        )

    def test_compact_for_an_unrecorded_session_creates_it(self):
        self.start("compact", session_id="contsession00002")

        meta = self.store.load_meta("contsession00002")
        self.assertEqual(meta.command, "claude-code (compact)")
        self.assertIsNone(meta.ended_at)
        self.assertEqual(self.start_sources("contsession00002"), ["compact"])

    def test_sources_that_begin_a_session_replace_metadata(self):
        for source, session_id in (
            ("startup", "replacestartup01"),
            ("clear", "replaceclear0001"),
            ("fork", "replacefork00001"),
        ):
            with self.subTest(source=source):
                self.start("startup", session_id=session_id)
                self.seed_meta(session_id)

                self.start(source, session_id=session_id)

                meta = self.store.load_meta(session_id)
                self.assertNotEqual(meta.started_at, self.STARTED_AT)
                self.assertEqual(meta.tool_calls, 0)
                self.assertEqual(meta.command, f"claude-code ({source})")


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
        os.environ.pop("AGENT_TRACE_REDACT", None)
        os.environ.pop("AGENT_TRACE_NO_REDACT", None)
        handle_session_start({"session_id": "redacttest123456", "source": "startup"})
        self.session_id = _read_active_session()

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_DIR", None)
        os.environ.pop("AGENT_TRACE_REDACT", None)
        os.environ.pop("AGENT_TRACE_NO_REDACT", None)

    def test_secrets_redacted_in_user_prompt(self):
        handle_user_prompt({
            "prompt": "Use this API key: sk-abc123def456ghi789jkl012mno345pqr678",
        })

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        prompts = [e for e in events if e.event_type == EventType.USER_PROMPT]
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("sk-abc123", str(prompts[0].data))

    def test_no_redact_environment_opt_out_preserves_secret(self):
        os.environ["AGENT_TRACE_NO_REDACT"] = "1"
        handle_user_prompt({
            "prompt": "Use this API key: sk-abc123def456ghi789jkl012mno345pqr678",
        })

        store = TraceStore(self.tmpdir, redact=False)
        events = store.load_events(self.session_id)
        prompts = [e for e in events if e.event_type == EventType.USER_PROMPT]
        self.assertEqual(len(prompts), 1)
        self.assertIn("sk-abc123", str(prompts[0].data))

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


class TestHooksRedactionMode(unittest.TestCase):
    """session_start records the redaction mode the hook process wrote with."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        env = patch.dict(os.environ, {"AGENT_TRACE_DIR": self.tmpdir})
        env.start()
        self.addCleanup(env.stop)
        for name in (
            "AGENT_TRACE_REDACT",
            "AGENT_TRACE_NO_REDACT",
            "AGENT_STRACE_NO_REDACT",
            "AGENT_STRACE_ENDPOINT",
        ):
            os.environ.pop(name, None)

    def test_local_session_start_records_the_effective_mode(self):
        for index, (env, expected) in enumerate((
            ({}, "on"),
            ({"AGENT_TRACE_REDACT": "1"}, "on"),
            ({"AGENT_TRACE_NO_REDACT": "1"}, "off"),
            ({"AGENT_STRACE_NO_REDACT": "true"}, "off"),
            ({"AGENT_TRACE_REDACT": "1", "AGENT_TRACE_NO_REDACT": "1"}, "off"),
        )):
            session_id = f"redactmode{index:06d}"
            with self.subTest(env=env), patch.dict(os.environ, env):
                handle_session_start({"session_id": session_id, "source": "startup"})
                events = TraceStore(self.tmpdir, redact=False).load_events(session_id)
                self.assertEqual(events[0].event_type, EventType.SESSION_START)
                self.assertEqual(events[0].data["redaction"], expected)

    def test_remote_session_start_records_the_mode_it_sends_with(self):
        for index, (env, expected) in enumerate((({}, "on"), ({"AGENT_TRACE_NO_REDACT": "1"}, "off"))):
            with self.subTest(env=env), patch.dict(os.environ, {
                "AGENT_STRACE_ENDPOINT": "https://collector.invalid",
                **env,
            }), patch(
                "agent_trace.server.send_session_meta_to_endpoint", return_value=True
            ), patch(
                "agent_trace.server.send_event_to_endpoint", return_value=True
            ) as send_event:
                handle_session_start({"session_id": f"remotemode{index:06d}", "source": "startup"})

                sent = send_event.call_args.args[0]
                self.assertEqual(sent.event_type, EventType.SESSION_START)
                self.assertEqual(sent.data["redaction"], expected)


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

    def test_ending_older_session_preserves_newer_canonical_marker(self):
        """An older concurrent session must not clear the extension pointer."""
        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = "agent1session00001"
        _write_active_session("agent1session000")

        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = "agent2session00002"
        _write_active_session("agent2session000")

        canonical = os.path.join(self.tmpdir, ".active-session")
        with open(canonical) as fh:
            self.assertEqual(fh.read(), "agent2session000")

        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = "agent1session00001"
        _clear_active_session()
        with open(canonical) as fh:
            self.assertEqual(fh.read(), "agent2session000")

        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = "agent2session00002"
        _clear_active_session()
        self.assertFalse(os.path.exists(canonical))

    def test_provider_session_state_suffix_is_hashed_and_contained(self):
        raw = "abcdefghijklmnop/../../outside-marker"
        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = raw
        suffix = _state_suffix("claude")
        active = _active_session_path("claude")

        self.assertRegex(suffix, r"^\.v2\.claude\.[0-9a-f]+\.[0-9a-f]{64}$")
        self.assertEqual(active.parent, Path(self.tmpdir))
        self.assertNotIn(raw, active.name)

        handle_session_start({"session_id": raw, "source": "startup"})
        self.assertTrue(active.exists())
        self.assertFalse((Path(self.tmpdir).parent / "outside-marker").exists())

    def test_distinct_raw_provider_ids_cannot_collide(self):
        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = "same/id"
        first = _state_suffix("claude")
        os.environ["AGENT_TRACE_CLAUDE_SESSION_ID"] = "same_id"
        second = _state_suffix("claude")
        os.environ["AGENT_TRACE_CURSOR_SESSION_ID"] = "same/id"
        third = _state_suffix("cursor")

        self.assertEqual(len({first, second, third}), 3)


_HOOK_EVENTS = {
    "claude": {"start": "SessionStart", "pre": "PreToolUse", "post": "PostToolUse", "end": "SessionEnd"},
    "codex": {"start": "SessionStart", "pre": "PreToolUse", "post": "PostToolUse", "end": "SessionEnd"},
    "copilot": {"start": "sessionStart", "pre": "preToolUse", "post": "postToolUse", "end": "sessionEnd"},
}
_SESSION_KEY = {"claude": "session_id", "codex": "session_id", "copilot": "sessionId"}
_OTHER_PROVIDER = {"claude": "codex", "codex": "copilot", "copilot": "claude"}
_SRC = Path(__file__).resolve().parents[1] / "src"


def _raw_session_id(provider, n):
    return f"{provider[:4]}{n:04d}-0000-4000-8000-{n:012d}"


class _PendingCallsHookTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / ".agent-traces"
        env = patch.dict(os.environ, {"AGENT_TRACE_DIR": str(self.root), "DO_NOT_TRACK": "1"})
        env.start()
        self.addCleanup(env.stop)
        for name in (
            *_PROVIDER_ENV.values(),
            "AGENT_TRACE_REDACT",
            "AGENT_STRACE_TENANT_ID",
            "AGENT_STRACE_WORKSPACE",
            "AGENT_STRACE_ENDPOINT",
            "AGENT_STRACE_PARENT_SESSION",
        ):
            os.environ.pop(name, None)

    def hook(self, provider, phase, payload):
        err = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), redirect_stderr(err):
            hook_main(["--provider", provider, _HOOK_EVENTS[provider][phase]])
        self.assertEqual(err.getvalue(), "", f"{provider} {phase}")

    def run_hook(self, provider, phase, raw_sid, use_id="t1", **extra):
        payload = {_SESSION_KEY[provider]: raw_sid, **extra}
        if phase in ("pre", "post"):
            if provider == "copilot":
                payload.update({"toolName": "bash", "toolArgs": {"command": use_id}})
                if phase == "post":
                    payload["toolResult"] = {"resultType": "success", "textResultForLlm": "ok"}
            else:
                payload.update(
                    {"tool_name": "Bash", "tool_input": {"command": use_id}, "tool_use_id": use_id}
                )
                if phase == "post":
                    payload["tool_response"] = "ok"
        elif phase == "start":
            payload.setdefault("source", "startup")
        self.hook(provider, phase, payload)

    def pending_files(self):
        return sorted(path.name for path in self.root.glob(".pending-calls*.json"))

    def pending_file(self, provider, raw_sid):
        (name,) = [
            name
            for name in self.pending_files()
            if name.startswith(f".pending-calls.v2.{provider}.{raw_sid[:16].encode().hex()}.")
        ]
        return self.root / name

    def events(self, raw_sid, event_type):
        return [
            event
            for event in TraceStore(self.root).load_events(raw_sid[:16])
            if event.event_type == event_type
        ]


class TestPendingCallsCleanup(_PendingCallsHookTest):
    """Session end must not leave pending-calls state behind."""

    def test_lifecycle_leaves_no_pending_file(self):
        for provider in _HOOK_EVENTS:
            with self.subTest(provider=provider):
                sid = _raw_session_id(provider, 1)
                self.run_hook(provider, "start", sid)
                self.assertEqual(self.pending_files(), [])

                self.run_hook(provider, "pre", sid)
                pending = json.loads(self.pending_file(provider, sid).read_text())
                (call,) = self.events(sid, EventType.TOOL_CALL)
                self.assertEqual(list(pending), [call.event_id])
                self.assertEqual(pending[call.event_id]["session_id"], sid[:16])

                self.run_hook(provider, "post", sid)
                self.assertEqual(self.pending_files(), [])
                (result,) = self.events(sid, EventType.TOOL_RESULT)
                self.assertEqual(result.parent_id, call.event_id)

                self.run_hook(provider, "end", sid, reason="exit")
                self.assertEqual(self.pending_files(), [])
                self.assertTrue(self.events(sid, EventType.SESSION_END))

    def test_session_end_keeps_other_sessions_and_providers_files(self):
        for provider in _HOOK_EVENTS:
            with self.subTest(provider=provider):
                other = _OTHER_PROVIDER[provider]
                ending = _raw_session_id(provider, 2)
                sibling = _raw_session_id(provider, 3)
                foreign = _raw_session_id(other, 4)
                for p, sid in ((provider, sibling), (other, foreign), (provider, ending)):
                    self.run_hook(p, "start", sid)
                    self.run_hook(p, "pre", sid)
                kept = [self.pending_file(provider, sibling), self.pending_file(other, foreign)]
                before = {path: path.read_bytes() for path in kept}
                ending_file = self.pending_file(provider, ending)

                self.run_hook(provider, "post", ending)
                self.assertFalse(ending_file.exists())
                self.run_hook(provider, "end", ending, reason="exit")

                self.assertEqual({path: path.read_bytes() for path in kept}, before)
                self.assertEqual(sorted(path.name for path in kept), self.pending_files())
                for p, sid in ((provider, sibling), (other, foreign)):
                    self.run_hook(p, "post", sid)
                    self.run_hook(p, "end", sid, reason="exit")
                self.assertEqual(self.pending_files(), [])

    def test_unmatched_calls_are_dropped_but_stay_in_events(self):
        for provider in _HOOK_EVENTS:
            with self.subTest(provider=provider):
                sid = _raw_session_id(provider, 5)
                self.run_hook(provider, "start", sid)
                self.run_hook(provider, "pre", sid)
                self.assertTrue(self.pending_file(provider, sid).exists())

                self.run_hook(provider, "end", sid, reason="exit")
                self.assertEqual(self.pending_files(), [])
                (call,) = self.events(sid, EventType.TOOL_CALL)
                self.assertEqual(self.events(sid, EventType.TOOL_RESULT), [])
                events = TraceStore(self.root).load_events(sid[:16])
                health = assess_evidence_health(events, session_finalized=True)
                self.assertIn(
                    ("unpaired_tool_call", call.event_id),
                    {(reason.code, reason.event_id) for reason in health.reasons},
                )

                # A result arriving after the end is still recorded, unlinked,
                # and does not recreate the state file.
                self.run_hook(provider, "post", sid)
                self.assertEqual(self.pending_files(), [])
                (result,) = self.events(sid, EventType.TOOL_RESULT)
                self.assertEqual(result.parent_id, "")

    def test_session_end_without_provider_id_clears_only_this_session(self):
        for provider in _HOOK_EVENTS:
            with self.subTest(provider=provider):
                other = _OTHER_PROVIDER[provider]
                sibling = _raw_session_id(provider, 6)
                ending = _raw_session_id(provider, 7)
                local = ending[:16]
                self.run_hook(provider, "start", sibling)
                self.run_hook(provider, "pre", sibling)
                self.run_hook(provider, "start", ending)
                self.run_hook(provider, "pre", ending)
                ending_file = self.pending_file(provider, ending)

                same_local_other_provider = (
                    self.root / f".pending-calls.v2.{other}.{local.encode().hex()}.{'0' * 64}.json"
                )
                same_local_other_provider.write_text('{"x": {"tool_name": "Bash", "timestamp": 1}}')
                shared = self.root / ".pending-calls.json"
                shared.write_text(json.dumps({
                    "mine": {"session_id": local, "tool_name": "Bash", "timestamp": 1},
                    "theirs": {"session_id": "someone-else", "tool_name": "Bash", "timestamp": 1},
                }))
                kept = [self.pending_file(provider, sibling), same_local_other_provider]
                before = {path: path.read_bytes() for path in kept}

                # A new hook process: no provider session ID in env or payload.
                os.environ.pop(_PROVIDER_ENV[provider], None)
                self.hook(provider, "end", {})

                self.assertFalse(ending_file.exists())
                self.assertIsNotNone(TraceStore(self.root).load_meta(local).ended_at)
                self.assertEqual({path: path.read_bytes() for path in kept}, before)
                self.assertEqual(list(json.loads(shared.read_text())), ["theirs"])

                for path in (*kept, shared):
                    path.unlink()

    def test_hook_paths_never_nest_the_store_lock(self):
        """Pending-calls locking must not run inside another store-lock holder.

        flock is not re-entrant across file descriptors, so a nested
        acquisition, or a TraceStore built (journal recovery locks) while the
        lock is held, would deadlock a real hook process.
        """
        violations = []
        depth = [0]
        acquired = [0]
        real_lock = store_module._store_write_lock
        real_init = store_module.TraceStore.__init__

        @contextlib.contextmanager
        def guarded_lock(*args, **kwargs):
            if depth[0]:
                # Record instead of blocking forever on the real flock.
                violations.append("nested lock:\n" + "".join(traceback.format_stack(limit=8)))
                yield
                return
            depth[0] += 1
            acquired[0] += 1
            try:
                with real_lock(*args, **kwargs):
                    yield
            finally:
                depth[0] -= 1

        def guarded_init(self, *args, **kwargs):
            if depth[0]:
                violations.append("store built under lock:\n" + "".join(traceback.format_stack(limit=8)))
            real_init(self, *args, **kwargs)

        with patch.object(store_module, "_store_write_lock", guarded_lock), \
                patch.object(store_module.TraceStore, "__init__", guarded_init):
            self.test_lifecycle_leaves_no_pending_file()
            self.test_session_end_keeps_other_sessions_and_providers_files()
            self.test_unmatched_calls_are_dropped_but_stay_in_events()
            self.test_session_end_without_provider_id_clears_only_this_session()
        self.assertGreater(acquired[0], 0)
        self.assertEqual(violations, [])


@unittest.skipIf(fcntl is None, "store lock needs fcntl.flock")
class TestPendingCallsConcurrency(_PendingCallsHookTest):
    """Parallel hook processes of one session must not lose pending calls."""

    PROCESSES = 8
    TIMEOUT = 60

    def run_parallel(self, provider, phase, payloads):
        code = (
            "import io, sys\n"
            "from agent_trace.hooks import hook_main\n"
            "sys.stdin = io.StringIO(sys.argv[1])\n"
            "hook_main(sys.argv[2:])\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SRC), env.get("PYTHONPATH")]))
        event = _HOOK_EVENTS[provider][phase]
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", code, json.dumps(payload), "--provider", provider, event],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )
            for payload in payloads
        ]
        deadline = time.monotonic() + self.TIMEOUT
        try:
            outputs = [
                proc.communicate(timeout=max(0.1, deadline - time.monotonic())) for proc in procs
            ]
        except subprocess.TimeoutExpired:
            for proc in procs:
                proc.kill()
                proc.communicate()
            self.fail(f"{phase} hooks did not finish within {self.TIMEOUT}s (lock deadlock?)")
        for proc, (_, stderr) in zip(procs, outputs):
            self.assertEqual((proc.returncode, stderr), (0, ""), phase)

    def test_parallel_tool_hooks_keep_every_pending_call(self):
        provider = "claude"
        sid = _raw_session_id(provider, 8)
        use_ids = [f"call-{i}" for i in range(self.PROCESSES)]

        def payload(use_id, **extra):
            return {
                "session_id": sid,
                "tool_name": "Bash",
                "tool_input": {"command": use_id},
                "tool_use_id": use_id,
                **extra,
            }

        self.run_parallel(provider, "start", [{"session_id": sid, "source": "startup"}])
        self.run_parallel(provider, "pre", [payload(use_id) for use_id in use_ids])

        calls = {event.data["tool_use_id"]: event.event_id for event in self.events(sid, EventType.TOOL_CALL)}
        self.assertEqual(sorted(calls), sorted(use_ids))
        pending = json.loads(self.pending_file(provider, sid).read_text())
        self.assertEqual(sorted(pending), sorted(calls.values()))

        self.run_parallel(
            provider, "post", [payload(use_id, tool_response="ok") for use_id in use_ids]
        )
        self.assertEqual(self.pending_files(), [])
        results = {
            event.data["tool_use_id"]: event.parent_id
            for event in self.events(sid, EventType.TOOL_RESULT)
        }
        self.assertEqual(results, calls)

        self.run_parallel(provider, "end", [{"session_id": sid, "reason": "exit"}])
        self.assertEqual(self.pending_files(), [])


if __name__ == "__main__":
    unittest.main()
