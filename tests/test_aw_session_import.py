"""Tests for importing gh-aw unified agent session files (aw_session.jsonl)."""

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

from agent_trace.cost import estimate_cost
from agent_trace.jsonl_import import cmd_import, import_jsonl
from agent_trace.models import EventType
from agent_trace.otlp import session_to_otlp_genai
from agent_trace.session_outcome import recorded_session_status
from agent_trace.store import TraceStore

FIXTURES = Path(__file__).parent / "fixtures" / "aw_session"
_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


def _header(version=1):
    return {
        "type": "session.format", "data": {"version": version},
        "provenance": {"component": "collector", "phase": "conclusion",
                       "path": "usage/aw_session.jsonl", "index": 0},
    }


def _record(record_type, data, index, *, timestamp=None, agent_id=None):
    record = {
        "type": record_type, "data": data,
        "provenance": {"component": "agent", "phase": "agent",
                       "path": "agent-session.jsonl", "index": index},
    }
    if timestamp is not None:
        record["timestamp"] = timestamp
    if agent_id is not None:
        record["agentId"] = agent_id
    return record


class _ImportCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.trace = self.tmp / "trace"
        self.store = TraceStore(str(self.trace), use_workspace_env=False)

    def _write(self, records, name="aw_session.jsonl"):
        path = self.tmp / name
        path.write_text(
            "".join(
                (r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records
            ),
            encoding="utf-8",
        )
        return path

    def _import(self, path, store=None):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            sid = import_jsonl(str(path), store=store or self.store)
        return sid, err.getvalue()

    def _cli(self, path):
        args = argparse.Namespace(trace_dir=str(self.trace), discover=False, path=str(path))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            code = cmd_import(args)
        return code, out.getvalue(), err.getvalue()

    def _sessions(self):
        if not self.trace.is_dir():
            return []
        return sorted(
            p.name for p in self.trace.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def _fixture(self, name):
        sid, _ = self._import(FIXTURES / name)
        return sid, self.store.load_events(sid)

    @staticmethod
    def _of(events, event_type, **match):
        return [
            e for e in events if e.event_type == event_type
            and all(e.data.get(k) == v for k, v in match.items())
        ]


class TestHeader(_ImportCase):
    def assert_refused(self, records, message):
        path = self._write(records)
        code, _, err = self._cli(path)
        self.assertEqual(code, 1)
        self.assertEqual(err, f"Cannot import {path}: {message}\n")
        self.assertEqual(self._sessions(), [])

    def test_version_2_is_refused(self):
        self.assert_refused(
            [_header(2), _record("user.message", {"content": "hi"}, 0, timestamp=1)],
            "unsupported aw_session format version 2; this agent-strace reads version 1",
        )

    def test_invalid_version_is_refused(self):
        for version in ("1", True, 0, 1.5, None):
            with self.subTest(version=version):
                self.assert_refused(
                    [_header(version)],
                    "invalid session.format header: data.version must be a whole "
                    "number of at least 1",
                )

    def test_missing_header_is_refused(self):
        self.assert_refused(
            [_record("user.message", {"content": "hi"}, 0, timestamp=1)],
            "no session.format header; an aw_session.jsonl file starts with one",
        )

    def test_header_is_never_read_as_a_claude_transcript(self):
        claude_line = {
            "type": "user", "sessionId": "claude-1", "uuid": "u1",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "from a Claude transcript"},
        }
        sid, _ = self._import(self._write([_header(), claude_line]))
        events = self.store.load_events(sid)
        self.assertEqual(events[0].data["mode"], "aw-session-import")
        self.assertEqual(self._of(events, EventType.USER_PROMPT), [])
        self.assertEqual(events[-1].data["skipped_records"], {"user": 1})

    def test_claude_transcript_still_imports(self):
        claude_line = {
            "type": "user", "sessionId": "claude-1", "uuid": "u1",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "hello"},
        }
        sid, _ = self._import(self._write([claude_line]))
        events = self.store.load_events(sid)
        self.assertEqual(events[0].data["mode"], "claude-jsonl-import")


class TestClaudeEngine(_ImportCase):
    def setUp(self):
        super().setUp()
        self.sid, self.events = self._fixture("claude_engine.jsonl")

    def _tool_events(self, call_id):
        return [e for e in self.events if e.data.get("tool_use_id") == call_id]

    def test_session_identity(self):
        self.assertEqual(self.sid, "claude-run-0001")
        meta = self.store.load_meta(self.sid)
        self.assertEqual(meta.agent_name, "claude")
        self.assertEqual(meta.tool_calls, 4)
        self.assertEqual(meta.llm_requests, 2)
        self.assertEqual(meta.errors, 1)
        start = self.events[0].data
        self.assertEqual(start["engine"], "claude")
        self.assertNotIn("provider", start)
        self.assertEqual(start["repository"], "example-org/sample-repo")
        self.assertEqual(start["run_id"], 1001)

    def test_completed_call_pairs_by_tool_call_id(self):
        call, result = self._tool_events("call-read")
        self.assertEqual(call.event_type, EventType.TOOL_CALL)
        self.assertEqual(call.data["arguments"], {"path": "README.md"})
        self.assertEqual(result.event_type, EventType.TOOL_RESULT)
        self.assertEqual(result.parent_id, call.event_id)
        self.assertEqual(result.duration_ms, 900)
        self.assertIs(result.data["success"], True)
        self.assertEqual(
            result.data["provenance"],
            {"component": "agent", "phase": "agent", "path": "agent-session.jsonl",
             "index": 3, "timestampMs": 1791568804000},
        )

    def test_success_false_is_a_failed_tool(self):
        call, error = self._tool_events("call-label")
        self.assertEqual(error.event_type, EventType.ERROR)
        self.assertEqual(error.parent_id, call.event_id)
        self.assertEqual(error.data["tool_name"], "bash")
        self.assertEqual(error.data["exit_code"], 1)
        self.assertIn("HTTP 403", error.data["error"])

    def test_success_absent_records_no_outcome(self):
        call, result = self._tool_events("call-grep")
        self.assertEqual(result.event_type, EventType.TOOL_RESULT)
        self.assertEqual(result.parent_id, call.event_id)
        self.assertNotIn("success", result.data)
        self.assertEqual(result.duration_ms, 1000)

    def test_dangling_call_has_no_result(self):
        (call,) = self._tool_events("call-dangling")
        self.assertEqual(call.event_type, EventType.TOOL_CALL)
        self.assertFalse(any(e.parent_id == call.event_id for e in self.events))

    def test_orphan_completion_stays_unpaired(self):
        (result,) = self._tool_events("call-orphan")
        self.assertEqual(result.event_type, EventType.TOOL_RESULT)
        self.assertFalse(result.parent_id)
        self.assertEqual(result.data["tool_name"], "view")

    def test_untimed_event_gets_an_inferred_anchor(self):
        (message,) = self._of(self.events, EventType.ASSISTANT_RESPONSE, source_type="assistant.message")
        self.assertEqual(message.data["text"], "Labeled the issue as a bug.")
        self.assertEqual(message.data["timing"], "inferred")
        # The latest time its own source had recorded: the orphan completion.
        self.assertEqual(message.timestamp, 1791568810.0)
        timed = [e for e in self.events[1:-1] if e is not message]
        self.assertFalse(any("timing" in e.data for e in timed))
        self.assertEqual(self.events[-1].data["inferred_times"], 1)

    def test_repeated_request_and_unmapped_records_are_counted_not_stored(self):
        usage = self._of(self.events, EventType.ASSISTANT_RESPONSE, source_type="firewall.token_usage")
        self.assertEqual([e.data["message_id"] for e in usage], ["req-1", "req-2"])
        self.assertFalse(any(e.data.get("source_type") == "otel.span" for e in self.events))
        self.assertEqual(
            self.events[-1].data["skipped_records"],
            {"engine.heartbeat": 1, "firewall.token_usage (repeated requestId)": 1, "otel.span": 1},
        )

    def test_firewall_usage_exports_chat_spans_with_recorded_tokens(self):
        meta = self.store.load_meta(self.sid)
        payload = session_to_otlp_genai(meta, self.events)
        spans = [
            span for rs in payload["resourceSpans"] for ss in rs["scopeSpans"]
            for span in ss["spans"]
        ]
        chats = [span for span in spans if span["name"].startswith("chat ")]
        self.assertEqual([span["name"] for span in chats], ["chat claude-sonnet-4.5"] * 2)
        attrs = {
            a["key"]: next(iter(a["value"].values())) for a in chats[0]["attributes"]
        }
        self.assertEqual(attrs["gen_ai.response.id"], "req-1")
        self.assertEqual(int(attrs["gen_ai.usage.input_tokens"]), 1400)
        self.assertEqual(int(attrs["gen_ai.usage.cache_read.input_tokens"]), 1000)
        self.assertEqual(int(attrs["gen_ai.usage.cache_write.input_tokens"]), 200)
        self.assertEqual(int(attrs["gen_ai.usage.output_tokens"]), 80)
        self.assertEqual(attrs["agent.strace.timing"], "inferred")
        tools = sorted(span["name"] for span in spans if span["name"].startswith("execute_tool "))
        self.assertEqual(tools, [
            "execute_tool bash", "execute_tool edit_file", "execute_tool grep",
            "execute_tool read_file", "execute_tool view",
        ])

    def test_cost_uses_recorded_usage(self):
        result = estimate_cost(self.store, self.sid)
        self.assertEqual(result.input_tokens, 1700)
        self.assertEqual(result.cache_read_tokens, 1000)
        self.assertEqual(result.cache_write_tokens, 200)
        self.assertEqual(result.output_tokens, 120)

    def test_one_session_result_snapshot(self):
        end = self.events[-1]
        self.assertEqual(end.event_type, EventType.SESSION_END)
        self.assertEqual(end.data["status"], "success")
        self.assertEqual(end.data["num_turns"], 4)
        self.assertEqual(end.data["reported_cost_usd"], 0.0123)
        self.assertEqual(end.data["exit_code"], 0)
        # The result's usage is a snapshot; it adds nothing to the recorded total.
        self.assertEqual(self.store.load_meta(self.sid).total_tokens, 1820)
        self.assertEqual(self.store.load_meta(self.sid).total_duration_ms, 12000)
        self.assertEqual(recorded_session_status(self.events), "completed")

    def test_reimport_reports_already_imported(self):
        code, _, err = self._cli(FIXTURES / "claude_engine.jsonl")
        self.assertEqual(code, 0)
        self.assertTrue(err.startswith(
            "Session claude-run-0001 is already imported; nothing written.\n"))
        self.assertEqual(len(self.store.load_events(self.sid)), len(self.events))


class TestImportOutput(_ImportCase):
    def test_summary_lines(self):
        code, _, err = self._cli(FIXTURES / "claude_engine.jsonl")
        self.assertEqual(code, 0)
        self.assertIn("  1 event has no recorded time (timing: inferred)\n", err)
        self.assertIn(
            "  Skipped 3 records not imported: engine.heartbeat 1, "
            "firewall.token_usage (repeated requestId) 1, otel.span 1\n",
            err,
        )


class TestCodexEngine(_ImportCase):
    def setUp(self):
        super().setUp()
        self.sid, self.events = self._fixture("codex_engine.jsonl")

    def _response(self, request_id):
        (event,) = self._of(self.events, EventType.ASSISTANT_RESPONSE, message_id=request_id)
        return event

    def test_input_that_includes_cache_is_not_double_counted(self):
        r1 = self._response("r-1")
        self.assertEqual(r1.data["usage"], {
            "input_tokens": 5000, "cached_input_tokens": 4000,
            "output_tokens": 300, "reasoning_output_tokens": 120,
        })
        result = estimate_cost(self.store, self.sid)
        self.assertEqual(result.input_tokens, 11700)
        self.assertEqual(result.cache_read_tokens, 9500)
        self.assertEqual(result.output_tokens, 550)
        self.assertEqual(self.store.load_meta(self.sid).total_tokens, 12250)

    def test_failed_request_and_detection_phase(self):
        r3 = self._response("r-3")
        self.assertEqual(r3.data["status"], "failed")
        self.assertEqual(r3.data["http_status"], 503)
        self.assertNotIn("usage", r3.data)
        d1 = self._response("d-1")
        self.assertEqual(d1.data["context_stream"], "detection")

    def test_results_are_snapshots_not_sums(self):
        end = self.events[-1].data
        self.assertEqual(end["status"], "failed")
        self.assertEqual(end["result_source_type"], "turn.failed")
        # Each result's error entries are kept, structured ones as JSON.
        self.assertEqual(
            end["errors"], ["upstream returned invalid_request_body", '{"message":"[redacted]"}'],
        )
        self.assertEqual(end["exit_code"], 78)
        self.assertEqual(end["error_codes"], ["invalid_request_body"])
        self.assertEqual(recorded_session_status(self.events), "terminated")


class TestSubagents(_ImportCase):
    def test_pi_subagents(self):
        sid, events = self._fixture("pi_interleaved.jsonl")
        starts = self._of(events, EventType.SUBAGENT_START)
        stops = self._of(events, EventType.SUBAGENT_STOP)
        self.assertEqual(
            [(e.data["agent_id"], e.data["agent_type"]) for e in starts],
            [("alpha-invocation", "a-reader"), ("zeta-invocation", "z-reader")],
        )
        self.assertEqual(
            sorted(e.data["agent_id"] for e in stops), ["alpha-invocation", "zeta-invocation"],
        )
        for event in starts + stops:
            self.assertTrue(event.data["is_sidechain"])
            self.assertEqual(event.data["context_stream"], f"sidechain:{event.data['agent_id']}")
        # subagent.request repeats the proxy's accounting, so only the proxy is kept.
        usage = self._of(events, EventType.ASSISTANT_RESPONSE)
        self.assertEqual(len(usage), 5)
        self.assertEqual(events[-1].data["skipped_records"], {"subagent.request": 4})
        self.assertEqual(self.store.load_meta(sid).total_tokens, 220)

    def test_subagent_tool_calls_pair_within_the_subagent(self):
        records = [
            _header(),
            _record("session.init", {"sessionId": "main-1", "sourceEngine": "claude"}, 0),
            _record("subagent.started", {"agentName": "explore", "agentType": "explore"}, 1,
                    timestamp=1000, agent_id="sub-1"),
            _record("tool.execution_start", {"toolCallId": "t1", "toolName": "view"}, 2,
                    timestamp=2000),
            _record("tool.execution_start", {"toolCallId": "t1", "toolName": "grep"}, 3,
                    timestamp=3000, agent_id="sub-1"),
            _record("tool.execution_complete", {"toolCallId": "t1", "success": True}, 4,
                    timestamp=4000, agent_id="sub-1"),
            _record("subagent.completed", {"agentName": "explore"}, 5,
                    timestamp=5000, agent_id="sub-1"),
        ]
        sid, _ = self._import(self._write(records))
        events = self.store.load_events(sid)
        main_call, sub_call = self._of(events, EventType.TOOL_CALL)
        (result,) = self._of(events, EventType.TOOL_RESULT)
        self.assertEqual(result.parent_id, sub_call.event_id)
        self.assertEqual(result.data["tool_name"], "grep")
        self.assertEqual(result.data["agent_type"], "explore")
        self.assertEqual(sub_call.data["context_stream"], "sidechain:sub-1")
        self.assertEqual(main_call.data["context_stream"], "main")


class TestTimesAndRedaction(_ImportCase):
    def test_fully_untimed_file_is_marked_untimed(self):
        records = [
            _header(),
            _record("session.init", {"sessionId": "untimed-1"}, 0),
            _record("user.message", {"content": "hi"}, 1),
        ]
        sid, _ = self._import(self._write(records))
        (prompt,) = self._of(self.store.load_events(sid), EventType.USER_PROMPT)
        self.assertEqual(prompt.data["timing"], "untimed")
        self.assertEqual(prompt.timestamp, 0.0)
        end = self.store.load_events(sid)[-1].data
        self.assertEqual(end["untimed_events"], 1)
        self.assertNotIn("inferred_times", end)

    def test_secrets_are_redacted_before_storing(self):
        records = [
            _header(),
            _record("session.init", {"sessionId": "secret-1"}, 0),
            _record("tool.execution_start",
                    {"toolCallId": "t1", "toolName": "bash", "input": f"echo {_TOKEN}"}, 1,
                    timestamp=1000),
            _record("tool.execution_complete",
                    {"toolCallId": "t1", "success": True, "output": f"token={_TOKEN}"}, 2,
                    timestamp=2000),
            _record("session.result", {"status": "success", "errors": [f"bad {_TOKEN}"]}, 3),
        ]
        store = TraceStore(str(self.trace), redact=True, use_workspace_env=False)
        sid, err = self._import(self._write(records), store=store)
        self.assertIn("redacted secrets", err)
        stored = "".join(
            p.read_text(encoding="utf-8") for p in (self.trace / sid).iterdir() if p.is_file()
        )
        self.assertNotIn("ghp_", stored)
        events = store.load_events(sid)
        (call,) = self._of(events, EventType.TOOL_CALL)
        (result,) = self._of(events, EventType.TOOL_RESULT)
        self.assertTrue(call.redacted)
        self.assertTrue(result.redacted)
        self.assertIn("[REDACTED:github-token]", result.data["result"])
        self.assertTrue(events[-1].data["redacted_errors"])

    def test_structured_values_are_redacted_by_key_before_they_become_text(self):
        records = [
            _header(),
            _record("session.init", {"sessionId": "secret-2"}, 0),
            _record("tool.execution_start",
                    {"toolCallId": "t1", "toolName": "login",
                     "input": [{"password": "hunter2-plain"}]}, 1, timestamp=1000),
            _record("tool.execution_complete",
                    {"toolCallId": "t1", "success": True,
                     "structuredContent": {"token": "tok-plain-123", "user": "octo"}}, 2,
                    timestamp=2000),
            _record("session.error", {"message": {"authorization": "s3cr3t-plain"}}, 3,
                    timestamp=3000),
            _record("session.result",
                    {"status": "failed", "errors": [{"api_key": "key-plain-456"}]}, 4),
        ]
        store = TraceStore(str(self.trace), redact=True, use_workspace_env=False)
        sid, _ = self._import(self._write(records), store=store)
        stored = "".join(
            p.read_text(encoding="utf-8") for p in (self.trace / sid).iterdir() if p.is_file()
        )
        for plain in ("hunter2-plain", "tok-plain-123", "s3cr3t-plain", "key-plain-456"):
            self.assertNotIn(plain, stored)
        events = store.load_events(sid)
        (call,) = self._of(events, EventType.TOOL_CALL)
        (result,) = self._of(events, EventType.TOOL_RESULT)
        (error,) = self._of(events, EventType.ERROR)
        self.assertTrue(call.redacted and result.redacted and error.redacted)
        self.assertEqual(
            json.loads(result.data["result"]),
            {"token": "[REDACTED:sensitive]", "user": "octo"},
        )
        self.assertTrue(events[-1].data["redacted_errors"])


class TestMalformedTimes(_ImportCase):
    def test_bad_nanosecond_timestamps_fall_back_to_the_anchor(self):
        otel = {"component": "otel", "phase": "agent", "path": "otel.jsonl"}
        records = [
            _header(),
            {"type": "session.collection",
             "data": {"sources": [dict(otel, timestampUnit="nanoseconds")]},
             "provenance": {"component": "collector", "phase": "conclusion",
                            "path": "usage/aw_session.jsonl", "index": 1}},
            _record("session.init", {"sessionId": "bad-times"}, 0,
                    timestamp="2026-01-01T00:00:00Z"),
        ]
        for index, stamp in enumerate(("1" * 400, "\u00b2", "1" * 5000)):
            records.append({
                "type": "user.message", "timestamp": stamp, "data": {"content": "hi"},
                "provenance": dict(otel, index=index),
            })
        code, _, err = self._cli(self._write(records))
        self.assertEqual(code, 0, err)
        prompts = self._of(self.store.load_events("bad-times"), EventType.USER_PROMPT)
        self.assertEqual([p.data["timing"] for p in prompts], ["inferred"] * 3)


if __name__ == "__main__":
    unittest.main()
