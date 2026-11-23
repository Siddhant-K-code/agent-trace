"""Tests for Claude Code hooks integration."""

import base64
import contextlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

import agent_trace.store as store_module
from agent_trace.event_data import parse_mapping
from agent_trace.evidence_health import assess_evidence_health
from agent_trace.hooks import (
    _PROVIDER_ENV,
    _active_session_path,
    _cut_tool_output,
    _read_active_session,
    _write_active_session,
    _clear_active_session,
    _native_conversation_id,
    _read_pending_calls,
    _write_pending_calls,
    _state_suffix,
    handle_session_start,
    handle_session_end,
    handle_pre_tool,
    handle_post_tool,
    handle_user_prompt,
    handle_stop,
    handle_subagent,
    hook_main,
)
from agent_trace.models import EventType, TraceEvent
from agent_trace.otlp import session_to_otlp, session_to_otlp_genai
from agent_trace.redact import redact_data, redaction_marker
from agent_trace.reviewability import _result_status
from agent_trace.runner_summary import output_text, summary_outcome
from agent_trace.store import TraceStore
from agent_trace.telemetry import SESSION_COMPLETED

try:
    from _concurrency import run_concurrently
except ImportError:  # pragma: no cover - run as a package module
    from tests._concurrency import run_concurrently


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

    def _end_with_stored_start(self, n, started_at, **stored):
        """End a session whose meta.json holds ``started_at``, given as JSON
        text, as another writer can leave it. Return its trace dir and ID.

        The session has a pending tool call, so the end must clear it too.
        """
        root = Path(self.tmpdir) / f"stored-start-{n}"
        os.environ["AGENT_TRACE_DIR"] = str(root)
        sid = f"storedstart{n:05d}"
        handle_session_start({"session_id": sid, "source": "startup"})
        handle_pre_tool({
            "session_id": sid,
            "tool_name": "Bash",
            "tool_input": {"command": "true"},
            "tool_use_id": "t1",
        })
        meta_path = root / sid / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta.update(stored, started_at="@start@")
        meta_path.write_text(json.dumps(meta).replace('"@start@"', started_at))
        self.assertEqual(len(list(root.glob(".active-session*"))), 2)
        self.assertEqual(len(list(root.glob(".pending-calls*.json"))), 1)

        handle_session_end({"session_id": sid, "reason": "exit", "cwd": "/repo"})
        return root, sid

    def test_session_end_without_a_usable_start_still_ends_the_session(self):
        def reject(token):
            raise ValueError(f"{token} is not JSON")

        bad_starts = ('"x"', "null", "[]", "true", "NaN", "-1e308", "1e308", "4102444800.0")
        for n, started_at in enumerate(bad_starts):
            with self.subTest(started_at=started_at):
                root, sid = self._end_with_stored_start(n, started_at)

                store = TraceStore(root)
                meta = store.load_meta(sid)
                self.assertIsNotNone(meta.ended_at)
                self.assertEqual(meta.total_duration_ms, 0)
                events = store.load_events(sid)
                ends = [e for e in events if e.event_type == EventType.SESSION_END]
                self.assertEqual(len(ends), 1)
                self.assertNotIn("duration_ms", ends[0].data)
                self.assertEqual(ends[0].data, {"reason": "exit", "cwd": "/repo"})

                for line in (root / sid / "events.ndjson").read_text().splitlines():
                    json.loads(line, parse_constant=reject)
                meta_text = (root / sid / "meta.json").read_text()
                if started_at == "NaN":
                    # The end rewrites the whole record, so the stored start
                    # stays as it is. Nothing else may be a constant.
                    self.assertEqual(meta_text.count('"started_at": NaN'), 1)
                    meta_text = meta_text.replace('"started_at": NaN', '"started_at": null')
                json.loads(meta_text, parse_constant=reject)

                self.assertEqual(list(root.glob(".active-session*")), [])
                self.assertEqual(list(root.glob(".pending-calls*.json")), [])
                self.assertIsNone(_read_active_session())
                self.assertEqual(_read_pending_calls(), {})

    def test_session_end_without_a_usable_start_replaces_a_stored_total(self):
        root, sid = self._end_with_stored_start(
            0, '"x"', total_duration_ms=5000, ended_at=None
        )

        meta = TraceStore(root).load_meta(sid)
        self.assertIsNotNone(meta.ended_at)
        self.assertEqual(meta.total_duration_ms, 0)

    def test_session_end_telemetry_sends_only_a_usable_duration(self):
        started_at = time.time() - 2.0
        with patch("agent_trace.hooks._product_telemetry.capture") as capture:
            root, sid = self._end_with_stored_start(0, json.dumps(started_at))

        store = TraceStore(root)
        meta = store.load_meta(sid)
        (end,) = [e for e in store.load_events(sid) if e.event_type == EventType.SESSION_END]
        self.assertEqual(meta.total_duration_ms, (meta.ended_at - started_at) * 1000)
        self.assertEqual(end.data["duration_ms"], meta.total_duration_ms)
        capture.assert_called_once()
        event, properties = capture.call_args.args
        self.assertEqual(event, SESSION_COMPLETED)
        self.assertIsInstance(properties["duration_ms"], int)
        self.assertEqual(properties["duration_ms"], int(meta.total_duration_ms))

        for n, bad_start in enumerate(('"x"', "NaN", "4102444800.0"), start=1):
            with self.subTest(started_at=bad_start), patch(
                "agent_trace.hooks._product_telemetry.capture"
            ) as capture:
                self._end_with_stored_start(n, bad_start)

                capture.assert_called_once()
                event, properties = capture.call_args.args
                self.assertEqual(event, SESSION_COMPLETED)
                self.assertEqual(
                    properties,
                    {"provider": "claude", "capture_method": "hooks", "success": True},
                )

    def test_session_end_without_start_is_noop(self):
        _clear_active_session()
        handle_session_end({})  # should not raise

    def test_duplicate_session_end_keeps_the_first_end(self):
        sid = "dupend0000000001"
        handle_session_start({"session_id": sid, "source": "startup"})
        handle_session_end({"session_id": sid, "reason": "exit"})
        store = TraceStore(self.tmpdir)
        first = store.load_meta(sid)

        time.sleep(0.01)
        handle_session_end({"session_id": sid, "reason": "retry"})

        events = store.load_events(sid)
        ends = [e for e in events if e.event_type == EventType.SESSION_END]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0].data["reason"], "exit")
        meta = store.load_meta(sid)
        self.assertEqual(meta.ended_at, first.ended_at)
        self.assertEqual(meta.total_duration_ms, first.total_duration_ms)

    def test_resumed_session_records_a_later_end(self):
        sid = "resumeend0000001"
        handle_session_start({"session_id": sid, "source": "startup"})
        handle_session_end({"session_id": sid})
        store = TraceStore(self.tmpdir)
        first_end = store.load_meta(sid).ended_at

        time.sleep(0.01)
        handle_session_start({"session_id": sid, "source": "resume"})
        handle_session_end({"session_id": sid})
        handle_session_end({"session_id": sid})

        events = store.load_events(sid)
        types = [e.event_type for e in events]
        self.assertEqual(types, [
            EventType.SESSION_START,
            EventType.SESSION_END,
            EventType.SESSION_START,
            EventType.SESSION_END,
        ])
        self.assertGreater(store.load_meta(sid).ended_at, first_end)

    def test_concurrent_session_ends_write_one_end(self):
        sid = "raceend000000001"
        handle_session_start({"session_id": sid, "source": "startup"})
        script = (
            "import os\n"
            "from agent_trace.hooks import handle_session_end\n"
            "wait_for_start()\n"
            "handle_session_end({'session_id': sys.argv[2]})\n"
        )
        run_concurrently(script, [[sid]] * 4)

        events = TraceStore(self.tmpdir).load_events(sid)
        ends = [e for e in events if e.event_type == EventType.SESSION_END]
        self.assertEqual(len(ends), 1)


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


class TestHooksConversationId(unittest.TestCase):
    """Hooks store the agent's full session ID for gen_ai.conversation.id."""

    NATIVE_ID = "1f0c2b9e-7d3a-4e6b-9c1d-2a3b4c5d6e7f"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict(os.environ, {"AGENT_TRACE_DIR": tmp.name})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_TRACE_REDACT", None)
        self.store = TraceStore(tmp.name)
        self.local_id = self.NATIVE_ID[:16]

    def test_session_start_stores_full_native_id(self):
        handle_session_start({"session_id": self.NATIVE_ID, "source": "startup"})
        meta = self.store.load_meta(self.local_id)
        self.assertEqual(meta.session_id, self.local_id)
        self.assertEqual(meta.conversation_id, self.NATIVE_ID)

    def test_resume_backfills_session_recorded_without_it(self):
        handle_session_start({"session_id": self.NATIVE_ID, "source": "startup"})
        meta = self.store.load_meta(self.local_id)
        meta.conversation_id = ""
        meta.tool_calls = 4
        self.store.update_meta(meta)

        handle_session_start({"session_id": self.NATIVE_ID, "source": "resume"})

        meta = self.store.load_meta(self.local_id)
        self.assertEqual(meta.conversation_id, self.NATIVE_ID)
        self.assertEqual(meta.tool_calls, 4)

    def test_hook_without_session_start_stores_native_id(self):
        handle_user_prompt({"session_id": self.NATIVE_ID, "prompt": "hi"})
        meta = self.store.load_meta(self.local_id)
        self.assertEqual(meta.conversation_id, self.NATIVE_ID)

    def test_native_id_is_validated(self):
        local = self.local_id
        self.assertEqual(_native_conversation_id({"session_id": self.NATIVE_ID}, local), self.NATIVE_ID)
        rejected = {
            "missing": {},
            "not a string": {"session_id": 12345},
            "empty": {"session_id": ""},
            "prefix mismatch": {"session_id": "ffffffffffffffff-other"},
            "too long": {"session_id": local + "x" * 300},
            "control character": {"session_id": local + "\n"},
        }
        for label, payload in rejected.items():
            with self.subTest(label):
                self.assertEqual(_native_conversation_id(payload, local), "")
        self.assertEqual(_native_conversation_id(None, local), "")
        self.assertEqual(_native_conversation_id({"session_id": self.NATIVE_ID}, ""), "")


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

    def test_otlp_call_id_is_the_tool_use_id(self):
        # gen_ai.tool.call.id is the model's ID for the call (#578). A call
        # recorded without one keeps the event ID.
        read_output = {"type": "text", "file": {"filePath": "a.py", "content": "x"}}
        handle_pre_tool({"tool_name": "Read", "tool_input": {}, "tool_use_id": "toolu_01ok"})
        handle_post_tool({"tool_name": "Read", "tool_response": read_output,
                          "tool_use_id": "toolu_01ok"})
        handle_pre_tool({"tool_name": "Bash", "tool_input": {}, "tool_use_id": "toolu_01err"})
        handle_post_tool({"tool_name": "Bash", "tool_output": "out", "tool_use_id": "toolu_01err"},
                         failed=True)
        handle_pre_tool({"tool_name": "Glob", "tool_input": {}, "tool_use_id": "toolu_01open"})
        handle_pre_tool({"tool_name": "Grep", "tool_input": {}})

        store = TraceStore(self.tmpdir)
        meta = store.load_meta(self.session_id)
        events = store.load_events(self.session_id)
        calls = {e.data["tool_name"]: e.event_id for e in events
                 if e.event_type == EventType.TOOL_CALL}
        expected = {"Read": "toolu_01ok", "Bash": "toolu_01err",
                    "Glob": "toolu_01open", "Grep": calls["Grep"]}
        for convert, prefix in ((session_to_otlp_genai, "execute_tool "),
                                (session_to_otlp, "tool/")):
            with self.subTest(convert=convert.__name__):
                spans = convert(meta, events)["resourceSpans"][0]["scopeSpans"][0]["spans"]
                attrs = {
                    span["name"][len(prefix):]: {
                        a["key"]: next(iter(a["value"].values())) for a in span["attributes"]
                    }
                    for span in spans if span["name"].startswith(prefix)
                }
                self.assertEqual(
                    {tool: a["gen_ai.tool.call.id"] for tool, a in attrs.items()}, expected,
                )
                if convert is session_to_otlp_genai:
                    # The link to the stored event is unchanged.
                    self.assertEqual(
                        {tool: a["agent.strace.request.event_id"] for tool, a in attrs.items()},
                        calls,
                    )
                    # The object output is stored as JSON (#576) and exported
                    # on the span that carries the toolu_ ID.
                    self.assertEqual(
                        json.loads(attrs["Read"]["gen_ai.tool.call.result"]), read_output,
                    )

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


class TestHooksSubagentPhase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ["AGENT_TRACE_DIR"] = self.tmpdir
        os.environ.pop("AGENT_TRACE_REDACT", None)
        handle_session_start({"session_id": "subphase12345678", "source": "startup"})
        self.session_id = _read_active_session()
        self.payload = {
            "agent_type": "explore",
            "stop_reason": "end_turn",
            "last_assistant_message": "done",
        }

    def tearDown(self):
        _clear_active_session()
        os.environ.pop("AGENT_TRACE_DIR", None)

    def events(self):
        return TraceStore(self.tmpdir).load_events(self.session_id)

    def test_start_records_subagent_start(self):
        handle_subagent(self.payload, "start")
        (event,) = [e for e in self.events() if e.event_type == EventType.SUBAGENT_START]
        self.assertEqual(event.data["hook_event"], "subagent_start")
        self.assertNotIn("stop_reason", event.data)
        self.assertNotIn("text", event.data)
        self.assertEqual(
            [e for e in self.events() if e.event_type == EventType.SUBAGENT_STOP], []
        )

    def test_stop_records_subagent_stop(self):
        handle_subagent(self.payload, "stop")
        (event,) = [e for e in self.events() if e.event_type == EventType.SUBAGENT_STOP]
        self.assertEqual(event.data["hook_event"], "subagent_stop")
        self.assertEqual(event.data["stop_reason"], "end_turn")
        self.assertEqual(event.data["text"], "done")
        self.assertEqual(
            [e for e in self.events() if e.event_type == EventType.SUBAGENT_START], []
        )

    def test_unknown_phase_writes_no_event(self):
        before = self.events()
        for phase in ("end", "Stop", "", "subagent-stop"):
            with self.subTest(phase=phase):
                err = io.StringIO()
                with redirect_stderr(err):
                    handle_subagent(self.payload, phase)
                self.assertIn("unknown subagent phase", err.getvalue())
                self.assertEqual(len(self.events()), len(before))


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


class TestHooksMetadataOnly(unittest.TestCase):
    """`hook --metadata-only` keeps structure and drops content values."""

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
            "AGENT_TRACE_COPILOT_SESSION_ID",
            "AGENT_TRACE_SESSION_ID",
        ):
            os.environ.pop(name, None)

    def _hook(self, args, payload):
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))):
            hook_main(args)

    def _run_session(self, args_prefix, session="metaonlysession01"):
        self._hook([*args_prefix, "session-start"], {
            "session_id": session,
            "source": "startup",
            "model": "test-model",
            "cwd": "/home/dev/secret-project",
            "transcript_path": "/home/dev/.claude/t.jsonl",
        })
        self._hook([*args_prefix, "user-prompt"], {
            "session_id": session,
            "prompt": "please read the customer list",
        })
        self._hook([*args_prefix, "pre-tool"], {
            "session_id": session,
            "tool_name": "Bash",
            "tool_use_id": "tool_1",
            "tool_input": {"command": "cat customers.csv", "timeout": 30},
        })
        self._hook([*args_prefix, "post-tool"], {
            "session_id": session,
            "tool_name": "Bash",
            "tool_use_id": "tool_1",
            "tool_response": {"stdout": "alice@example.com"},
        })
        self._hook([*args_prefix, "session-end"], {"session_id": session, "reason": "exit"})
        return TraceStore(self.tmpdir, redact=False).load_events(session[:16])

    def test_content_values_are_not_written(self):
        events = self._run_session(["--metadata-only"])
        raw = "".join(json.dumps(event.data) for event in events)
        for content in (
            "please read the customer list",
            "cat customers.csv",
            "alice@example.com",
            "/home/dev/secret-project",
            "t.jsonl",
        ):
            self.assertNotIn(content, raw)

        by_type = {event.event_type: event for event in events}
        start = by_type[EventType.SESSION_START].data
        self.assertEqual(start["capture"], "metadata-only")
        self.assertEqual(start["redaction"], "on")
        self.assertEqual(start["model"], "test-model")
        self.assertEqual(start["cwd"], "[OMITTED]")
        self.assertEqual(by_type[EventType.USER_PROMPT].data["prompt"], "[OMITTED]")
        self.assertTrue(by_type[EventType.USER_PROMPT].redacted)
        call = by_type[EventType.TOOL_CALL].data
        self.assertEqual(call["tool_name"], "Bash")
        self.assertEqual(call["tool_use_id"], "tool_1")
        self.assertEqual(set(call["arguments"]), {"command", "timeout"})
        self.assertEqual(call["arguments"]["command"], "[OMITTED]")
        self.assertEqual(by_type[EventType.SESSION_END].data["reason"], "exit")
        self.assertIn("duration_ms", by_type[EventType.SESSION_END].data)
        for event in events:
            self.assertEqual(event.data["capture"], "metadata-only")

    def test_redaction_is_forced_on_despite_environment(self):
        with patch.dict(os.environ, {"AGENT_TRACE_NO_REDACT": "1"}):
            events = self._run_session(["--metadata-only"], session="metaonlynoredact")
        start = next(e for e in events if e.event_type == EventType.SESSION_START)
        self.assertEqual(start.data["redaction"], "on")

    def test_flag_order_and_provider_are_accepted(self):
        for index, prefix in enumerate((
            ["--provider", "copilot", "--metadata-only"],
            ["--metadata-only", "--provider", "copilot"],
            ["--metadata-only", "copilot"],
        )):
            session = f"metaonlyorder{index:03d}"
            with self.subTest(prefix=prefix):
                self._hook([*prefix, "sessionStart"], {"sessionId": session, "source": "startup"})
                self._hook([*prefix, "userPromptSubmitted"], {"sessionId": session, "prompt": "hello there"})
                events = TraceStore(self.tmpdir, redact=False).load_events(session[:16])
                start = next(e for e in events if e.event_type == EventType.SESSION_START)
                prompt = next(e for e in events if e.event_type == EventType.USER_PROMPT)
                self.assertEqual(start.data["provider"], "copilot")
                self.assertEqual(start.data["capture"], "metadata-only")
                self.assertEqual(prompt.data["prompt"], "[OMITTED]")

    def test_cli_parser_accepts_metadata_only(self):
        from agent_trace.cli import main

        for index, argv in enumerate((
            ["hook", "--provider", "copilot", "--metadata-only", "sessionStart"],
            ["hook", "--metadata-only", "--provider", "copilot", "sessionStart"],
            ["hook", "--metadata-only", "session-start"],
        )):
            session = f"metaonlycli{index:05d}"
            payload = json.dumps({"session_id": session, "sessionId": session, "cwd": "/secret"})
            with self.subTest(argv=argv), patch.object(sys, "argv", ["agent-strace", *argv]), \
                    patch.object(sys, "stdin", io.StringIO(payload)):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 0)
                start = TraceStore(self.tmpdir, redact=False).load_events(session[:16])[0]
                self.assertEqual(start.data["capture"], "metadata-only")
                self.assertEqual(start.data["cwd"], "[OMITTED]")

    def test_default_capture_is_unchanged(self):
        events = self._run_session([], session="fullcapturesess1")
        prompt = next(e for e in events if e.event_type == EventType.USER_PROMPT)
        self.assertEqual(prompt.data["prompt"], "please read the customer list")
        self.assertNotIn("capture", prompt.data)

    def test_conversation_id_is_kept_in_both_capture_modes(self):
        # The native session ID is an identifier, not content: metadata-only
        # keeps it so exports still join with native agent telemetry.
        from agent_trace.otlp import session_to_otlp_genai

        cases = (
            ("full", [], "session_id", "0a1b2c3d-full-4e5f-claude-000001"),
            ("metadata-only", ["--metadata-only"], "session_id", "0a1b2c3d-meta-4e5f-claude-000002"),
            ("metadata-only copilot", ["--metadata-only", "--provider", "copilot"],
             "sessionId", "9f8e7d6c-copi-4e5f-copilot-00003"),
        )
        for label, prefix, key, native in cases:
            with self.subTest(label):
                start = "sessionStart" if key == "sessionId" else "session-start"
                prompt = "userPromptSubmitted" if key == "sessionId" else "user-prompt"
                self._hook([*prefix, start], {key: native, "source": "startup"})
                self._hook([*prefix, prompt], {key: native, "prompt": "do not export me"})
                store = TraceStore(self.tmpdir, redact=False)
                meta = store.load_meta(native[:16])
                self.assertEqual(meta.conversation_id, native)
                payload = session_to_otlp_genai(meta, store.load_events(native[:16]))
                root = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
                attrs = {a["key"]: a["value"].get("stringValue") for a in root["attributes"]}
                self.assertEqual(attrs["gen_ai.conversation.id"], native)

    def test_remote_endpoint_receives_metadata_only_redacted_events(self):
        with patch.dict(os.environ, {
            "AGENT_STRACE_ENDPOINT": "https://collector.invalid",
            "AGENT_TRACE_NO_REDACT": "1",
        }), patch(
            "agent_trace.server.send_session_meta_to_endpoint", return_value=True
        ), patch(
            "agent_trace.server.send_event_to_endpoint", return_value=True
        ) as send_event:
            self._hook(["--metadata-only", "session-start"], {
                "session_id": "remotemetaonly01",
                "source": "startup",
                "cwd": "/home/dev/secret-project",
            })
        sent = send_event.call_args.args[0]
        self.assertEqual(sent.data["redaction"], "on")
        self.assertEqual(sent.data["capture"], "metadata-only")
        self.assertEqual(sent.data["cwd"], "[OMITTED]")


SUFFIX = "." * 3 + " (truncated)"
REDACTION_WARNING = "agent-strace: redacted secrets from trace event\n"
KEY_MARKER = redaction_marker("private-key")
SENSITIVE = redaction_marker("sensitive")
GITHUB_TOKEN = "ghp_" + "A1b2" * 9


def _as_json(value):
    """Return tool output an object sent as a hook stores it: compact JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _stored_result(text):
    return TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=1.0, data={"tool_name": "Bash", "result": text})


def _fake_key(label="RSA PRIVATE KEY", seed=1, size=1190):
    """Return a PEM block of random base64. The RSA default is 1675 characters."""
    text = base64.b64encode(random.Random(seed).randbytes(size)).decode()
    lines = [text[i : i + 64] for i in range(0, len(text), 64)]
    return "\n".join([f"-----BEGIN {label}-----", *lines, f"-----END {label}-----", ""])


def _too_deep_to_walk(data, *args, **kwargs):
    """Stand in for redact_data on output that nests too deep to walk."""
    if not isinstance(data, str):
        raise RecursionError("maximum recursion depth exceeded")
    return redact_data(data, *args, **kwargs)


class TestHooksRedactBeforeCut(unittest.TestCase):
    """PostToolUse redacts the raw tool output, then cuts it to 1000 or 500 characters."""

    KEY = _fake_key()
    COPILOT_SESSION = "copi0001-0000-4000-8000-000000000001"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmpdir = tmp.name
        env = patch.dict(os.environ, {"AGENT_TRACE_DIR": self.tmpdir})
        env.start()
        self.addCleanup(env.stop)
        for name in (
            *_PROVIDER_ENV.values(),
            "AGENT_TRACE_REDACT",
            "AGENT_TRACE_NO_REDACT",
            "AGENT_STRACE_NO_REDACT",
            "AGENT_STRACE_ENDPOINT",
        ):
            os.environ.pop(name, None)
        handle_session_start({"session_id": "cutredact0000001", "source": "startup"})
        self.session_id = _read_active_session()

    def last_result(self, session_id):
        events = TraceStore(self.tmpdir, redact=False).load_events(session_id)
        return [e for e in events if e.event_type in (EventType.TOOL_RESULT, EventType.ERROR)][-1]

    def post(self, output, failed=False, tool_name="Bash"):
        err = io.StringIO()
        with redirect_stderr(err):
            handle_post_tool({"tool_name": tool_name, "tool_response": output}, failed=failed)
        return self.last_result(self.session_id), err.getvalue()

    def hook(self, args, payload):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), \
                redirect_stdout(out), redirect_stderr(err):
            hook_main(args)
        return err.getvalue()

    def assert_no_key_text(self, text, key=KEY):
        body = [line for line in key.splitlines() if not line.startswith("-----")]
        leaked = [
            line[i : i + 16]
            for line in body
            for i in range(len(line) - 15)
            if line[i : i + 16] in text
        ]
        self.assertEqual(leaked, [])

    def assert_redacted_once(self, event, err):
        self.assertTrue(event.redacted)
        self.assertEqual(err, REDACTION_WARNING)

    def test_key_in_bash_stdout(self):
        output = {"stdout": self.KEY, "stderr": "", "interrupted": False, "isImage": False}
        event, err = self.post(output)
        self.assertEqual(event.data["result"], _as_json({**output, "stdout": KEY_MARKER + "\n"}))
        self.assert_redacted_once(event, err)

    def test_key_in_read_file_content(self):
        key = _fake_key("PRIVATE KEY", seed=3, size=1218)
        file = {
            "filePath": "/home/dev/.ssh/service.pem",
            "content": key,
            "numLines": 28,
            "startLine": 1,
            "totalLines": 28,
        }
        event, err = self.post({"type": "text", "file": file}, tool_name="Read")
        self.assertEqual(
            event.data["result"],
            _as_json({"type": "text", "file": {**file, "content": KEY_MARKER + "\n"}}),
        )
        self.assert_no_key_text(event.data["result"], key)
        self.assert_redacted_once(event, err)

    def test_key_in_failed_tool_output(self):
        event, err = self.post(self.KEY + "\nerror: permission denied", failed=True)
        self.assertEqual(event.event_type, EventType.ERROR)
        self.assertEqual(event.data["error"], KEY_MARKER + "\n\nerror: permission denied")
        self.assert_redacted_once(event, err)

    def test_key_in_copilot_error(self):
        sid = self.COPILOT_SESSION
        self.hook(["--provider", "copilot", "sessionStart"], {"sessionId": sid, "source": "startup"})
        tool = {"sessionId": sid, "toolName": "bash", "toolArgs": {"command": "cat id_rsa"}}
        cases = (
            ("message", "postToolUseFailure", {"error": {"message": self.KEY}},
             KEY_MARKER + "\n"),
            ("error dict", "postToolUseFailure", {"error": {"code": "EACCES", "detail": self.KEY}},
             _as_json({"code": "EACCES", "detail": KEY_MARKER + "\n"})),
            ("failed result", "postToolUse",
             {"toolResult": {"resultType": "failure", "error": self.KEY}},
             _as_json({"resultType": "failure", "error": KEY_MARKER + "\n"})),
        )
        for label, hook_event, payload, expected in cases:
            with self.subTest(label):
                err = self.hook(["--provider", "copilot", hook_event], {**tool, **payload})
                event = self.last_result(sid[:16])
                self.assertEqual(event.event_type, EventType.ERROR)
                self.assertEqual(event.data["error"], expected)
                self.assert_redacted_once(event, err)

    def test_claude_failure_records_its_error_field(self):
        # Claude Code's PostToolUseFailure sends its text in error, with no tool_response.
        failure = {
            "session_id": self.session_id,
            "hook_event_name": "PostToolUseFailure",
            "tool_name": "Bash",
            "tool_input": {"command": "ls missing.txt"},
            "is_interrupt": False,
        }
        text = "Exit code 1\nls: missing.txt: No such file or directory"
        long_text = "Exit code 1\n" + "ls: missing.txt: No such file or directory\n" * 15
        cases = (
            ("text", {"error": text}, text, False),
            ("message", {"error": {"message": text}}, text, False),
            ("secret", {"error": "Exit code 1\n" + self.KEY}, "Exit code 1\n" + KEY_MARKER + "\n", True),
            ("long", {"error": long_text}, long_text[:500], False),
            ("no error", {}, "tool call failed", False),
            ("tool_response", {"error": text, "tool_response": "kept"}, "kept", False),
        )
        for label, payload, expected, redacted in cases:
            with self.subTest(label):
                err = self.hook(["post-tool-failure"], {**failure, **payload})
                event = self.last_result(self.session_id)
                self.assertEqual(event.event_type, EventType.ERROR)
                self.assertEqual(event.data["error"], expected)
                if redacted:
                    self.assert_redacted_once(event, err)
                else:
                    self.assertFalse(event.redacted)
                    self.assertEqual(err, "")

        sid = "metaonlyclaude01"
        self.hook(["--metadata-only", "session-start"], {"session_id": sid, "source": "startup"})
        self.hook(["--metadata-only", "post-tool-failure"], {**failure, "session_id": sid, "error": text})
        self.assertEqual(self.last_result(sid).data["error"], "[OMITTED]")

    def test_sensitive_key_names_in_tool_output_dict(self):
        output = {"password": "hunter2-Rz2ArIYF", "api_key": "plainbLcOBXlE3A2P"}
        event, err = self.post(output, tool_name="mcp__vault__get_secret")
        self.assertEqual(event.data["result"], _as_json({"password": SENSITIVE, "api_key": SENSITIVE}))
        self.assert_redacted_once(event, err)

    def test_first_lines_of_a_key(self):
        head = "".join(self.KEY.splitlines(keepends=True)[:12])
        output = {"stdout": head, "stderr": "", "interrupted": False, "isImage": False}
        event, err = self.post(output)
        self.assertEqual(event.data["result"], _as_json({**output, "stdout": KEY_MARKER + "\n"}))
        self.assert_redacted_once(event, err)

    def test_key_in_json_text(self):
        key = _fake_key("PRIVATE KEY", seed=4, size=1218)
        account = {
            "type": "service_account",
            "project_id": "demo-project",
            "private_key_id": "5f1c" * 10,
            "private_key": key,
            "client_email": "deploy@demo-project.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        text = json.dumps(account, indent=2)
        event, err = self.post([{"type": "text", "text": text}], tool_name="mcp__gcp__get_key")
        expected = json.dumps({
            **account,
            "private_key_id": redaction_marker("hex-token"),
            "private_key": KEY_MARKER + "\n",
        }, indent=2)
        self.assertEqual(event.data["result"], _as_json([{"type": "text", "text": expected}]))
        self.assert_no_key_text(event.data["result"], key)
        self.assert_redacted_once(event, err)

        # A JSON line cut inside the key by the tool itself has no END line.
        line = json.dumps({"private_key": self.KEY})[:600]
        event, err = self.post({"stdout": line, "stderr": ""})
        self.assertEqual(
            event.data["result"],
            _as_json({"stdout": '{"private_key": "' + KEY_MARKER, "stderr": ""}),
        )
        self.assert_redacted_once(event, err)

    def test_secret_past_the_cut_stores_the_same_event(self):
        event, err = self.post("x" * 1200 + self.KEY)
        self.assertEqual(event.data, {"tool_name": "Bash", "result": "x" * 1000 + SUFFIX})
        self.assertFalse(event.redacted)
        self.assertEqual(err, "")

    def test_text_after_a_key_fills_the_cut(self):
        listing = "$ ls\n" + "notes.txt\n" * 100
        event, err = self.post(self.KEY + listing)
        self.assertEqual(event.data["result"], (KEY_MARKER + "\n" + listing)[:1000] + SUFFIX)
        self.assert_redacted_once(event, err)

    def test_cut_inside_a_marker_flags_the_event(self):
        for failed, kept, field, tail in (
            (False, 995, "result", "[REDA" + SUFFIX),
            (True, 495, "error", "[REDA"),
        ):
            with self.subTest(failed=failed):
                event, err = self.post("x" * kept + GITHUB_TOKEN, failed=failed)
                self.assertEqual(event.data[field], "x" * kept + tail)
                self.assert_redacted_once(event, err)

    def test_output_without_a_secret_stores_the_same_event(self):
        for output, failed in (
            ("ok", False),
            ("plain build output line\n" * 250, False),
            ({"stdout": "built 3 targets", "stderr": "", "exit_code": 0}, False),
            (["README.md", "setup.py"], False),
            ("compile error on line 4\n" * 40, True),
            ("", True),
        ):
            with self.subTest(output=str(output)[:24], failed=failed):
                event, err = self.post(output, failed=failed)
                text = output if isinstance(output, str) else _as_json(output)
                if failed:
                    expected = {"tool_name": "Bash", "error": text[:500] if output else "tool call failed"}
                else:
                    expected = {"tool_name": "Bash", "result": text[:1000] + SUFFIX if len(text) > 1000 else text}
                self.assertEqual(event.data, expected)
                self.assertFalse(event.redacted)
                self.assertEqual(err, "")

    def test_redaction_off_stores_the_raw_cut(self):
        with patch.dict(os.environ, {"AGENT_TRACE_NO_REDACT": "1"}):
            event, err = self.post(self.KEY)
        self.assertEqual(event.data, {"tool_name": "Bash", "result": self.KEY[:1000] + SUFFIX})
        self.assertFalse(event.redacted)
        self.assertEqual(err, "")

    def test_provider_keys_in_claude_hook_output(self):
        stripe = "sk_" + "live_" + "A1b2" * 6
        google = "AI" + "za" + "Sy" + "B-_c" * 8 + "Q"
        output = {
            "stdout": "STRIPE_SECRET_KEY=" + stripe + "\nGOOGLE_API_KEY=" + google + "\n",
            "stderr": "",
            "interrupted": False,
            "isImage": False,
        }
        err = self.hook(["post-tool"], {
            "session_id": self.session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "cat .env"},
            "tool_response": output,
        })
        event = self.last_result(self.session_id)
        stdout = (
            "STRIPE_SECRET_KEY=" + redaction_marker("stripe-key")
            + "\nGOOGLE_API_KEY=" + redaction_marker("google-api-key") + "\n"
        )
        self.assertEqual(event.data["result"], _as_json({**output, "stdout": stdout}))
        self.assertNotIn(stripe[8:], event.data["result"])
        self.assertNotIn(google[4:], event.data["result"])
        self.assert_redacted_once(event, err)

    def test_metadata_only_writes_no_warning(self):
        sid = "metaonlycutkey01"
        self.hook(["--metadata-only", "session-start"], {"session_id": sid, "source": "startup"})
        err = self.hook(["--metadata-only", "post-tool"], {
            "session_id": sid,
            "tool_name": "Bash",
            "tool_response": self.KEY,
        })
        event = self.last_result(sid)
        self.assertEqual(event.data["result"], "[OMITTED]")
        self.assertTrue(event.redacted)
        self.assertEqual(err, "")

    def test_remote_endpoint_receives_the_redacted_output(self):
        sid = "remotecutkey0001"
        with patch.dict(os.environ, {"AGENT_STRACE_ENDPOINT": "https://collector.invalid"}), patch(
            "agent_trace.server.send_session_meta_to_endpoint", return_value=True
        ), patch(
            "agent_trace.server.send_event_to_endpoint", return_value=True
        ) as send_event:
            handle_session_start({"session_id": sid, "source": "startup"})
            err = io.StringIO()
            with redirect_stderr(err):
                handle_post_tool({"session_id": sid, "tool_name": "Bash", "tool_response": self.KEY})
        sent = send_event.call_args.args[0]
        self.assertEqual(sent.event_type, EventType.TOOL_RESULT)
        self.assertEqual(sent.data["result"], KEY_MARKER + "\n")
        self.assert_redacted_once(sent, err.getvalue())

    def test_agent_trace_redact_flags_a_cut_key_once(self):
        with patch.dict(os.environ, {"AGENT_TRACE_REDACT": "1"}):
            event, err = self.post(self.KEY)
        self.assertEqual(event.data["result"], KEY_MARKER + "\n")
        self.assert_redacted_once(event, err)

    def test_agent_trace_redact_flags_a_token_once(self):
        with patch.dict(os.environ, {"AGENT_TRACE_REDACT": "1"}):
            event, err = self.post("token " + GITHUB_TOKEN)
        self.assertEqual(event.data["result"], "token " + redaction_marker("github-token"))
        self.assert_redacted_once(event, err)

    def test_output_too_deep_to_walk_is_redacted_as_text(self):
        output = {"items": [{"token_value": GITHUB_TOKEN}]}
        with patch("agent_trace.hooks.redact_data", side_effect=_too_deep_to_walk):
            event, err = self.post(output)
        self.assertEqual(
            event.data["result"],
            _as_json({"items": [{"token_value": redaction_marker("github-token")}]}),
        )
        self.assert_redacted_once(event, err)

    def test_output_too_deep_to_walk_is_redacted_before_the_cut(self):
        # The token crosses the cut. The 20 characters before the cut match
        # no pattern, so only the hook's pass before the cut can hide them.
        marker = redaction_marker("github-token")
        for failed, limit, field, suffix in (
            (False, 1000, "result", SUFFIX),
            (True, 500, "error", ""),
        ):
            with self.subTest(failed=failed):
                text = "x" * (limit - 20 - len('{"t":"'))
                with patch("agent_trace.hooks.redact_data", side_effect=_too_deep_to_walk):
                    event, err = self.post({"t": text + GITHUB_TOKEN}, failed=failed)
                self.assertEqual(event.data[field], _as_json({"t": text + marker})[:limit] + suffix)
                self.assert_redacted_once(event, err)


class TestHooksObjectOutput(unittest.TestCase):
    """A tool response sent as an object is stored as compact JSON, not a Python repr."""

    BASH = {"stdout": "hi", "stderr": "", "interrupted": False, "isImage": False}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmpdir = tmp.name
        env = patch.dict(os.environ, {"AGENT_TRACE_DIR": self.tmpdir})
        env.start()
        self.addCleanup(env.stop)
        for name in (
            *_PROVIDER_ENV.values(),
            "AGENT_TRACE_REDACT",
            "AGENT_TRACE_NO_REDACT",
            "AGENT_STRACE_NO_REDACT",
            "AGENT_STRACE_ENDPOINT",
        ):
            os.environ.pop(name, None)

    def hook(self, args, payload):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), \
                redirect_stdout(out), redirect_stderr(err):
            hook_main(args)
        return err.getvalue()

    def record(self, event_name, payload, provider=None, sid="objectoutput0001"):
        """Send one post-tool hook in a new session. Return its last result event and stderr."""
        args = ["--provider", provider] if provider else []
        self.hook([*args, "session-start"], {"session_id": sid, "source": "startup"})
        err = self.hook([*args, event_name], {"session_id": sid, **payload})
        events = TraceStore(self.tmpdir, redact=False).load_events(sid)
        results = [e for e in events if e.event_type in (EventType.TOOL_RESULT, EventType.ERROR)]
        return results[-1], err

    def test_claude_bash_object_is_stored_as_json(self):
        event, err = self.record("post-tool", {
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "echo hi"},
            "tool_response": self.BASH,
        })
        stored = event.data["result"]
        self.assertEqual(stored, '{"stdout":"hi","stderr":"","interrupted":false,"isImage":false}')
        self.assertEqual(json.loads(stored), self.BASH)
        self.assertEqual(output_text(event), "hi\n")
        self.assertEqual(parse_mapping(stored), self.BASH)
        self.assertEqual(err, "")

    def test_claude_read_object_is_stored_as_json(self):
        response = {"type": "text", "file": {
            "filePath": "/workspace/project/notes.txt",
            "content": "line-1\nline-2 café\n",
            "numLines": 3,
            "startLine": 1,
            "totalLines": 3,
        }}
        event, _ = self.record("post-tool", {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/workspace/project/notes.txt"},
            "tool_response": response,
        })
        stored = event.data["result"]
        self.assertEqual(stored, _as_json(response))
        self.assertTrue(stored.startswith('{"type":"text","file":{"filePath":'))
        self.assertIn("café", stored)
        self.assertEqual(json.loads(stored), response)

    def test_codex_object_keeps_its_exit_code(self):
        for exit_code, output, event_type, field in (
            (0, "3 passed", EventType.TOOL_RESULT, "result"),
            (1, "1 failed", EventType.ERROR, "error"),
        ):
            with self.subTest(exit_code=exit_code):
                response = {"exit_code": exit_code, "output": output}
                event, _ = self.record("post-tool", {
                    "tool_name": "Bash",
                    "tool_use_id": f"call_{exit_code}",
                    "tool_response": response,
                }, provider="codex", sid=f"codexobject0000{exit_code}")
                self.assertEqual(event.event_type, event_type)
                self.assertEqual(event.data[field], _as_json(response))
                self.assertEqual(_result_status(event)[:2], ("passed" if exit_code == 0 else "failed", exit_code))
        self.assertEqual(output_text(_stored_result(_as_json({"exit_code": 0, "output": "ok"}))), "ok")

    def test_long_object_is_cut_after_1000_characters(self):
        response = {**self.BASH, "stdout": "build line\n" * 150}
        event, _ = self.record("post-tool", {"tool_name": "Bash", "tool_response": response})
        stored = event.data["result"]
        self.assertEqual(stored, _as_json(response)[:1000] + SUFFIX)
        self.assertTrue(stored.startswith('{"stdout":"build line\\nbuild line\\n'))
        self.assertIsNone(output_text(event))

    def test_object_is_redacted_before_the_cut(self):
        # The token crosses the cut at 1000 characters, so a pass after the cut
        # would see only its first characters.
        marker = redaction_marker("github-token")
        stdout = "x" * (990 - len('{"stdout":"'))
        response = {"stdout": stdout + GITHUB_TOKEN + "\ndone\n", "stderr": ""}
        event, err = self.record("post-tool", {"tool_name": "Bash", "tool_response": response})
        expected = _as_json({"stdout": stdout + marker + "\ndone\n", "stderr": ""})
        self.assertEqual(event.data["result"], expected[:1000] + SUFFIX)
        self.assertNotIn(GITHUB_TOKEN[:12], event.data["result"])
        self.assertTrue(event.redacted)
        self.assertEqual(err, REDACTION_WARNING)

    def test_claude_failure_object_is_stored_as_json(self):
        failure = {
            "hook_event_name": "PostToolUseFailure",
            "tool_name": "Bash",
            "tool_input": {"command": "cat missing.txt"},
        }
        error = {"code": "ENOENT", "path": "missing.txt"}
        event, _ = self.record("post-tool-failure", {**failure, "error": error})
        self.assertEqual(event.event_type, EventType.ERROR)
        self.assertEqual(event.data["error"], '{"code":"ENOENT","path":"missing.txt"}')

        long_error = {"code": "ENOENT", "detail": "no such file\n" * 60}
        event, _ = self.record("post-tool-failure", {**failure, "error": long_error}, sid="objectoutput0002")
        self.assertEqual(event.data["error"], _as_json(long_error)[:500])

    def test_non_json_values_are_stored_as_text(self):
        when = datetime(2026, 1, 2, tzinfo=timezone.utc)
        self.assertEqual(_cut_tool_output({"at": when, "n": None}, False), '{"at":"2026-01-02 00:00:00+00:00","n":null}')
        self.assertEqual(_cut_tool_output(None, False), "null")
        self.assertEqual(_cut_tool_output(["a", 1], True), '["a",1]')
        self.assertEqual(_cut_tool_output({}, True), "tool call failed")
        self.assertEqual(_cut_tool_output("plain text", False), "plain text")

    def test_readers_still_parse_results_stored_before_the_fix(self):
        old = str({"stdout": "", "stderr": "Ran 2 tests in 0.0s\n\nOK", "interrupted": False, "isImage": False})
        new = _as_json({"stdout": "", "stderr": "Ran 2 tests in 0.0s\n\nOK", "interrupted": False, "isImage": False})
        for stored in (old, new):
            with self.subTest(stored=stored[:12]):
                self.assertEqual(output_text(_stored_result(stored)), "\nRan 2 tests in 0.0s\n\nOK")
                self.assertEqual(summary_outcome("unittest", _stored_result(stored)), "passed")
                self.assertEqual(parse_mapping(stored)["stderr"], "Ran 2 tests in 0.0s\n\nOK")
        self.assertEqual(_result_status(_stored_result(str({"exit_code": 1, "output": "x"})))[:2], ("failed", 1))


class TestMetadataOnlyData(unittest.TestCase):
    def test_transform_rules(self):
        from agent_trace.hooks import metadata_only_data

        data, omitted = metadata_only_data({
            "tool_name": "Edit",
            "text": "secret",
            "empty": "",
            "arguments": {"path": "/a", "content": "b"},
            "items": [1, 2],
            "count": 3,
            "flag": True,
            "nothing": None,
        })
        self.assertTrue(omitted)
        self.assertEqual(data, {
            "tool_name": "Edit",
            "text": "[OMITTED]",
            "empty": "",
            "arguments": {"path": "[OMITTED]", "content": "[OMITTED]"},
            "items": [],
            "count": 3,
            "flag": True,
            "nothing": None,
            "capture": "metadata-only",
        })

    def test_nothing_omitted_for_metadata_keys(self):
        from agent_trace.hooks import metadata_only_data

        data, omitted = metadata_only_data({"tool_name": "Bash", "duration_ms": 5})
        self.assertFalse(omitted)
        self.assertEqual(data["capture"], "metadata-only")


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

    def test_malformed_pending_entries_do_not_drop_the_result(self):
        """A corrupt pending entry is skipped instead of crashing the hook."""
        handle_pre_tool({"tool_name": "Bash", "tool_input": {"command": "ls"}})
        pending = _read_pending_calls()
        pending["corrupt"] = "Bash"
        pending["no_timestamp"] = {"tool_name": "Bash"}
        _write_pending_calls(pending)

        handle_post_tool({"tool_name": "Bash", "tool_output": "ok"}, failed=False)

        store = TraceStore(self.tmpdir)
        events = store.load_events(self.session_id)
        (call,) = [e for e in events if e.event_type == EventType.TOOL_CALL]
        (result,) = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(result.parent_id, call.event_id)
        self.assertGreaterEqual(result.duration_ms, 0)
        self.assertEqual(set(_read_pending_calls()), {"corrupt", "no_timestamp"})

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


@unittest.skipIf(fcntl is None, "store lock needs fcntl.flock")
class TestMetaCounterConcurrency(_PendingCallsHookTest):
    """Parallel hook processes of one session must not lose counter updates."""

    PROCESSES = 8
    HOOKS_PER_PROCESS = 6
    TOKENS = 7
    SCRIPT = (
        "import contextlib, io, json\n"
        "from agent_trace.hooks import hook_main\n"
        "provider, payload = sys.argv[2], sys.argv[3]\n"
        "events = sys.argv[4:]\n"
        "err = io.StringIO()\n"
        "wait_for_start()\n"
        "with contextlib.redirect_stderr(err):\n"
        "    for event in events:\n"
        "        sys.stdin = io.StringIO(payload)\n"
        "        hook_main(['--provider', provider, event])\n"
        "print(err.getvalue(), end='')\n"
    )

    def test_parallel_hooks_keep_every_counter_increment(self):
        provider = "codex"
        sid = _raw_session_id(provider, 9)
        local_id = sid[:16]
        self.run_hook(provider, "start", sid)
        store = TraceStore(str(self.root))
        payload = json.dumps({
            "session_id": sid,
            "tool_name": "Bash",
            "tool_input": {"command": "true"},
            "error": "boom",
            "token_usage": {"total_tokens": self.TOKENS},
        })
        events = ["PreToolUse", "PostToolUseFailure", "PostModelRequest"]
        args = [[provider, payload, *events * self.HOOKS_PER_PROCESS]] * self.PROCESSES
        outputs = run_concurrently(self.SCRIPT, args)
        self.assertEqual(outputs, [""] * self.PROCESSES)

        expected = self.PROCESSES * self.HOOKS_PER_PROCESS
        meta = store.load_meta(local_id)
        # The payload's "error" fails every tool call and every model
        # request, so each adds one error.
        self.assertEqual(
            (meta.tool_calls, meta.errors, meta.total_tokens),
            (expected, 2 * expected, expected * self.TOKENS),
        )
        recorded = store.load_events(local_id)
        self.assertEqual(
            sum(e.event_type == EventType.TOOL_CALL for e in recorded), expected
        )


if __name__ == "__main__":
    unittest.main()
