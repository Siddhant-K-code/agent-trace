"""Tests for Claude Code JSONL import."""

import argparse
import contextlib
import errno
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

from agent_trace import jsonl_import
from agent_trace.cost import build_provider_cost_report
from agent_trace.jsonl_import import cmd_import, discover_claude_sessions, import_jsonl
from agent_trace.models import EventType
from agent_trace.store import TraceStore
from agent_trace.tenancy import delete_tenant_data, enumerate_trace_stores
from agent_trace.validation import validate_trace_directory


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

    def test_failed_tool_result_keeps_is_error(self):
        path = self._write_jsonl([
            {
                "type": "user",
                "sessionId": "err1",
                "timestamp": "2025-01-01T00:00:00Z",
                "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "boom", "is_error": True},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "ok"},
                ]},
                "uuid": "u1",
            },
        ])

        session_id = import_jsonl(path, store=self.store)
        results = {
            event.data["tool_use_id"]: event.data
            for event in self.store.load_events(session_id)
            if event.event_type == EventType.TOOL_RESULT
        }

        self.assertIs(results["t1"]["is_error"], True)
        self.assertNotIn("is_error", results["t2"])

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

    def test_import_without_cwd_still_starts_the_session(self):
        path = self._write_jsonl([
            {"type": "user", "sessionId": "nocwd", "timestamp": "2025-01-01T00:00:00Z",
             "message": {"role": "user", "content": "hi"}, "uuid": "u1"},
        ])

        session_id = import_jsonl(path, store=self.store)
        events = self.store.load_events(session_id)
        types = [e.event_type for e in events]

        self.assertEqual(types[0], EventType.SESSION_START)
        self.assertEqual(types[-1], EventType.SESSION_END)
        self.assertEqual(types.count(EventType.SESSION_START), 1)
        self.assertEqual(events[0].data, {"mode": "claude-jsonl-import", "provider": "claude"})

        report = validate_trace_directory(self.store.base_dir / session_id)
        codes = {finding.code for finding in report.findings}
        self.assertNotIn("missing_session_start", codes)
        self.assertNotIn("events_before_session_start", codes)

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

    def _reimport_log(self):
        return self._write_jsonl([
            {"type": "user", "sessionId": "again1", "timestamp": "2025-01-01T00:00:00Z",
             "message": {"role": "user", "content": "list files"}, "uuid": "u1"},
            {"type": "assistant", "sessionId": "again1", "timestamp": "2025-01-01T00:00:05Z",
             "message": {"role": "assistant", "model": "claude-sonnet-4", "content": [
                 {"type": "text", "text": "ok"},
                 {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
             ], "usage": {"input_tokens": 10, "output_tokens": 5}}, "uuid": "u2"},
        ])

    def test_reimport_writes_nothing(self):
        path = self._reimport_log()
        session_id = import_jsonl(path, store=self.store)
        session_dir = self.store.base_dir / session_id
        events_before = (session_dir / "events.ndjson").read_bytes()
        meta_before = (session_dir / "meta.json").read_bytes()

        self.assertEqual(import_jsonl(path, store=self.store), session_id)

        self.assertEqual((session_dir / "events.ndjson").read_bytes(), events_before)
        self.assertEqual((session_dir / "meta.json").read_bytes(), meta_before)
        types = [e.event_type for e in self.store.load_events(session_id)]
        self.assertEqual(types.count(EventType.SESSION_START), 1)
        self.assertEqual(types.count(EventType.TOOL_CALL), 1)

    def test_cli_reimport_reports_existing_session_and_exits_zero(self):
        path = self._reimport_log()
        args = argparse.Namespace(trace_dir=self.tmpdir, discover=False, path=path)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cmd_import(args), 0)
        events_before = (self.store.base_dir / "again1" / "events.ndjson").read_bytes()

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cmd_import(args), 0)

        self.assertIn("Session again1 is already imported; nothing written.", err.getvalue())
        self.assertNotIn("Imported session", err.getvalue())
        self.assertEqual((self.store.base_dir / "again1" / "events.ndjson").read_bytes(), events_before)

    def test_log_that_fails_to_convert_leaves_no_partial_session(self):
        good = {"type": "user", "sessionId": "half1", "timestamp": "2025-01-01T00:00:00Z",
                "message": {"role": "user", "content": "hi"}, "uuid": "u1"}
        bad = {"type": "assistant", "sessionId": "half1", "timestamp": "2025-01-01T00:00:01Z",
               "message": {"role": "assistant", "content": "ok"}, "uuid": "u2"}
        path = self._write_jsonl([good, bad])
        real = jsonl_import._extract_text

        def fail_on_second(content):
            if content == "ok":
                raise RuntimeError("conversion failed")
            return real(content)

        with mock.patch.object(jsonl_import, "_extract_text", side_effect=fail_on_second):
            with self.assertRaises(RuntimeError):
                import_jsonl(path, store=self.store)
        self.assertFalse(self.store.session_exists("half1"))

        # The fixed log then imports in full instead of being reported as present.
        path = self._write_jsonl([good])
        self.assertEqual(import_jsonl(path, store=self.store), "half1")
        self.assertEqual(self.store.load_events("half1")[-1].event_type, EventType.SESSION_END)

    def _prompt_times(self, session_id):
        return {
            event.data["prompt"]: event.timestamp
            for event in self.store.load_events(session_id)
            if event.event_type == EventType.USER_PROMPT
        }

    def test_lines_that_are_not_objects_are_skipped(self):
        path = os.path.join(self.tmpdir, "session.jsonl")
        entry = {"type": "user", "sessionId": "notobj1", "timestamp": "2026-01-01T00:00:00.500Z",
                 "message": {"role": "user", "content": "kept"}, "uuid": "u1"}
        with open(path, "w") as f:
            f.write('[1, 2]\n"text"\n' + json.dumps(entry) + "\nnull\n42\n")
        args = argparse.Namespace(trace_dir=self.tmpdir, discover=False, path=path)

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cmd_import(args), 0)

        self.assertEqual(self._prompt_times("notobj1"), {"kept": 1767225600.5})

    def test_timestamp_that_is_not_a_unix_time_is_stored_as_zero(self):
        timestamps = {
            "good": "2026-01-01T00:00:00.500Z",
            "number": 1767225600,
            "bool": True,
            "object": {"seconds": 1767225600},
            "nan": float("nan"),
            "before-1970": "1900-01-01T00:00:00Z",
        }
        path = self._write_jsonl([
            {"type": "user", "sessionId": "badtime1", "timestamp": timestamp,
             "message": {"role": "user", "content": name}, "uuid": f"u{index}"}
            for index, (name, timestamp) in enumerate(timestamps.items())
        ])
        args = argparse.Namespace(trace_dir=self.tmpdir, discover=False, path=path)

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cmd_import(args), 0)

        expected = dict.fromkeys(timestamps, 0.0)
        expected["good"] = 1767225600.5
        self.assertEqual(self._prompt_times("badtime1"), expected)


NAN = float("nan")
INF = float("inf")
VALUES = ("x", "7", 5, -5, 1.5, NAN, INF, True, None, [1], {"a": 1}, 10**400)
NOT_STRINGS = tuple(value for value in VALUES if not isinstance(value, str))
NOT_TRUE = tuple(value for value in VALUES if value is not True)
TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
              "cache_read_input_tokens")
# json.loads raises RecursionError on this on every CI Python (see test_optimize.py).
TOO_DEEP = b"[" * 1_000_000 + b"]" * 1_000_000
_ABSENT = object()


def _entry(sid="good1", **over):
    """One user prompt line. ``sid=None`` leaves out the sessionId."""
    entry = {"type": "user", "sessionId": sid, "uuid": "u1", "timestamp": "2026-01-01T00:00:00Z",
             "cwd": "/w", "message": {"role": "user", "content": "hello"}}
    if sid is None:
        del entry["sessionId"]
    entry.update(over)
    return entry


def _full_log(sid="full1"):
    """A log that uses every field the importer reads."""
    return [
        {"type": "user", "sessionId": sid, "uuid": "u1", "timestamp": "2026-01-01T00:00:00Z",
         "cwd": "/w", "gitBranch": "main", "version": "2.0.1",
         "message": {"role": "user", "content": "hello"}},
        {"type": "assistant", "sessionId": sid, "uuid": "a1", "parentUuid": "u1",
         "timestamp": "2026-01-01T00:00:01Z",
         "message": {"id": "m1", "model": "claude-sonnet-4", "role": "assistant",
                     "content": [{"type": "text", "text": "ok"},
                                 {"type": "tool_use", "id": "t1", "name": "Agent",
                                  "input": {"subagent_type": "x"}, "caller": {"type": "direct"}}],
                     "usage": {"input_tokens": 1, "output_tokens": 2,
                               "cache_creation_input_tokens": 3, "cache_read_input_tokens": 4}}},
        {"type": "user", "sessionId": sid, "uuid": "u2", "parentUuid": "a1",
         "timestamp": "2026-01-01T00:00:02Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "t1",
              "content": [{"type": "text", "text": "res"}]}]}},
        {"type": "user", "sessionId": sid, "uuid": "u3", "timestamp": "2026-01-01T00:00:03Z",
         "message": {"role": "user", "content": [{"type": "text", "text": "x"}]},
         "toolUseResult": {"stdout": "out", "stderr": "err"}},
        {"type": "system", "subtype": "turn_duration", "durationMs": 5, "sessionId": sid,
         "timestamp": "2026-01-01T00:00:04Z"},
        {"type": "system", "subtype": "compact_boundary", "content": "sum", "sessionId": sid,
         "timestamp": "2026-01-01T00:00:05Z"},
        {"type": "assistant", "sessionId": sid, "uuid": "a2", "isSidechain": True, "agentId": "ag1",
         "timestamp": "2026-01-01T00:00:06Z",
         "message": {"id": "m2", "model": "claude-sonnet-4", "role": "assistant", "content": "side",
                     "usage": {"input_tokens": 10, "output_tokens": 20}}},
    ]


def _with_field(line, dotted, value):
    """``_full_log()`` with one field set, or removed when value is _ABSENT."""
    log = _full_log()
    keys = [int(key) if key.isdigit() else key for key in dotted.split(".")]
    target = log[line]
    for key in keys[:-1]:
        target = target[key]
    if value is _ABSENT:
        target.pop(keys[-1], None)
    else:
        target[keys[-1]] = value
    return log


def _nested(kind, levels):
    """A value nested ``levels`` deep: dicts, or lists inside one dict."""
    value = "x"
    for _ in range(levels - 1):
        value = {"k": value} if kind == "dict" else [value]
    return {"k": value}


def _deep_tool_line(sid, kind, levels):
    """An assistant line whose tool input is ``_nested(kind, levels)``, as bytes.

    It is built as text because json.dumps cannot encode 5000 levels on 3.10.
    """
    if kind == "dict":
        deep = '{"k":' * levels + '"x"' + "}" * levels
    else:
        deep = '{"k":' + "[" * (levels - 1) + '"x"' + "]" * (levels - 1) + "}"
    line = {"type": "assistant", "sessionId": sid, "uuid": "a1",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"id": "m1", "model": "claude-sonnet-4", "role": "assistant",
                        "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                     "input": "@"}]}}
    return json.dumps(line).replace('"@"', deep).encode()


class TestMalformedLog(unittest.TestCase):
    """A field of the wrong type reads as absent and a line that cannot be read is skipped."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_STRACE_WORKSPACE", None)
        os.environ.pop("AGENT_STRACE_TENANT_ID", None)
        self.trace = self.tmp / "trace"

    def _write(self, name, data):
        """Write bytes as they are, or a list of lines: a dict as JSON, bytes raw."""
        if not isinstance(data, bytes):
            data = b"".join(
                (line if isinstance(line, bytes) else json.dumps(line).encode()) + b"\n"
                for line in data
            )
        path = self.tmp / name
        path.write_bytes(data)
        return path

    def _import(self, path):
        with contextlib.redirect_stderr(io.StringIO()):
            return import_jsonl(str(path), store=TraceStore(str(self.trace), use_workspace_env=False))

    def _cli(self, path):
        args = argparse.Namespace(trace_dir=str(self.trace), discover=False, path=str(path))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = cmd_import(args)
        return code, err.getvalue()

    def _session_dirs(self):
        if not self.trace.is_dir():
            return []
        return sorted(p.name for p in self.trace.iterdir() if p.is_dir() and not p.name.startswith("."))

    def _meta(self, sid):
        return json.loads((self.trace / sid / "meta.json").read_text(encoding="utf-8"))

    def _events(self, sid):
        lines = (self.trace / sid / "events.ndjson").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _prompts(self, sid):
        return [event["data"]["prompt"] for event in self._events(sid)
                if event["event_type"] == EventType.USER_PROMPT.value]

    def _session(self, log):
        """The whole imported session as text, with the random event IDs renumbered."""
        shutil.rmtree(self.trace, ignore_errors=True)
        sid = self._import(self._write("full1.jsonl", log))
        meta, events = self._meta(sid), self._events(sid)
        ids = {}
        for event in events:
            for key in ("event_id", "parent_id"):
                if event.get(key):
                    event[key] = ids.setdefault(event[key], f"E{len(ids)}")
            if "prev_hash" in event:
                event["prev_hash"] = "H"
        shutil.rmtree(self.trace)
        return json.dumps([meta, events], sort_keys=True)

    def _counts(self, log):
        """(tokens, duration, LLM requests, tool calls, events) of the imported session."""
        shutil.rmtree(self.trace, ignore_errors=True)
        sid = self._import(self._write("full1.jsonl", log))
        meta, events = self._meta(sid), self._events(sid)
        shutil.rmtree(self.trace)
        return (meta.get("total_tokens", 0), meta.get("total_duration_ms", 0),
                meta.get("llm_requests", 0), meta.get("tool_calls", 0), len(events))

    def test_field_of_the_wrong_type_reads_as_absent(self):
        cases = [(1, "message.usage", [v for v in VALUES if not isinstance(v, dict)])]
        cases += [(line, field, VALUES) for line, field in (
            (1, "message.content.1.caller"), (1, "message.id"), (0, "message"), (1, "message"),
            (3, "toolUseResult"), (4, "subtype"), (5, "subtype"), (5, "content"), (0, "type"),
        )]
        cases += [(line, field, NOT_STRINGS) for line, field in (
            (1, "message.content.0.text"), (0, "message.content"), (1, "message.content"),
            (2, "message.content.0.content"), (2, "message.content.0.content.0.text"),
            (3, "toolUseResult.stdout"), (3, "toolUseResult.stderr"), (0, "cwd"), (0, "uuid"),
            (0, "parentUuid"), (6, "agentId"),
        )]
        cases.append((4, "durationMs", ("x", "7", -5, NAN, INF, True, None, [1], {"a": 1}, 10**400)))
        cases += [(line, field, NOT_TRUE) for line, field in (
            (2, "message.content.0.is_error"), (0, "isCompactSummary"),
            (0, "isCompactionSummary"), (6, "isSidechain"),
        )]
        for line, field, values in cases:
            absent = self._session(_with_field(line, field, _ABSENT))
            for value in values:
                with self.subTest(line=line, field=field, value=value):
                    self.assertEqual(self._session(_with_field(line, field, value)), absent)

    def test_only_a_whole_number_of_0_or_more_counts_as_tokens(self):
        for key in TOKEN_KEYS:
            field = f"message.usage.{key}"
            absent = self._counts(_with_field(1, field, _ABSENT))
            for value in ("x", "7", -5, 1.5, NAN, INF, True, None, [1], {"a": 1}):
                with self.subTest(key=key, value=value):
                    self.assertEqual(self._counts(_with_field(1, field, value)), absent)
            for value in (5, 10**400):
                with self.subTest(key=key, value=value):
                    self.assertEqual(self._counts(_with_field(1, field, value)),
                                     (absent[0] + value, *absent[1:]))

    def test_field_copied_raw_does_not_change_the_counts(self):
        for line, field in (
            (1, "message.content.1.input"), (1, "message.content.1.name"),
            (1, "message.content.1.id"), (1, "message.model"),
            (2, "message.content.0.tool_use_id"), (0, "gitBranch"), (0, "version"),
        ):
            absent = self._counts(_with_field(line, field, _ABSENT))
            for value in VALUES:
                with self.subTest(line=line, field=field, value=value):
                    self.assertEqual(self._counts(_with_field(line, field, value)), absent)

    def test_unusable_session_id_falls_back_to_the_file_name(self):
        for value in (123, True, 1.5, "../evil", "a/b", "has space", "a" * 300, [1],
                      "a\x1b[31mb", {}, ""):
            with self.subTest(value=value):
                shutil.rmtree(self.trace, ignore_errors=True)
                path = self._write("stem1.jsonl", [
                    _entry(value),
                    _entry(value, uuid="u2", message={"role": "user", "content": "two"}),
                ])

                self.assertEqual(self._import(path), "stem1")
                self.assertEqual(self._session_dirs(), ["stem1"])
                self.assertEqual(self._prompts("stem1"), ["hello", "two"])
                self.assertNotIn("conversation_id", self._meta("stem1"))

    def test_first_usable_session_id_wins(self):
        for name, first, later in (("stem2.jsonl", "../x", "later1"), ("stem3.jsonl", 7, "later2")):
            with self.subTest(first=first):
                path = self._write(name, [_entry(first), _entry(later, uuid="u2")])

                self.assertEqual(self._import(path), later)
                self.assertEqual(self._meta(later)["conversation_id"], later)

    def test_no_usable_session_id_fails_with_one_line(self):
        for name, line in (("has space.jsonl", _entry(None)), ("has space.jsonl", _entry("../evil")),
                           ("b" * 200 + ".jsonl", _entry(None))):
            with self.subTest(name=name, line=line):
                path = self._write(name, [line])

                code, err = self._cli(path)

                self.assertEqual(code, 1)
                self.assertEqual(
                    err, f"Cannot import {path}: no usable session ID in the log or the file name\n")
                self.assertEqual(self._session_dirs(), [])

    def test_usable_session_id_imports_with_an_unusable_file_name(self):
        code, _ = self._cli(self._write("has space.jsonl", [_entry("okid2")]))

        self.assertEqual(code, 0)
        self.assertEqual(self._session_dirs(), ["okid2"])

    def test_erased_session_fails_with_one_line(self):
        path = self._write("er1.jsonl", [_entry("erased1")])
        with mock.patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "t1"}):
            self.assertEqual(self._cli(path)[0], 0)
        delete_tenant_data(enumerate_trace_stores(str(self.trace)), "t1")

        code, err = self._cli(path)

        self.assertEqual(code, 1)
        self.assertEqual(
            err, f"Cannot import {path}: session erased1 was erased and cannot be recreated\n")
        self.assertEqual(self._session_dirs(), [])

    def test_line_cut_mid_character_is_skipped(self):
        path = self._write("cut1.jsonl", json.dumps(_entry("cut1")).encode() + b"\n"
                           + b'{"type":"user","sessionId":"cut1","message":{"content":"caf\xc3')

        self.assertEqual(self._import(path), "cut1")
        self.assertEqual(self._prompts("cut1"), ["hello"])

    def test_line_that_is_not_utf8_is_skipped(self):
        # The bad byte sits inside a JSON string, so only the UTF-8 check can skip the line.
        path = self._write("bad1.jsonl", [
            _entry("bad1"),
            b'{"type":"user","sessionId":"bad1","uuid":"u2","timestamp":"2026-01-01T00:00:01Z",'
            b'"message":{"role":"user","content":"\xff"}}',
            _entry("bad1", uuid="u3", timestamp="2026-01-01T00:00:02Z",
                   message={"role": "user", "content": "after"}),
        ])

        self.assertEqual(self._import(path), "bad1")
        self.assertEqual(self._prompts("bad1"), ["hello", "after"])

    def test_carriage_return_line_ends_still_split_lines(self):
        for sid, end in (("crlf1", b"\r\n"), ("cr1", b"\r")):
            with self.subTest(end=end):
                lines = [_entry(sid), _entry(sid, uuid="u2", timestamp="2026-01-01T00:00:01Z",
                                             message={"role": "user", "content": "two"})]
                path = self._write(f"{sid}.jsonl", b"".join(json.dumps(line).encode() + end
                                                            for line in lines))

                self.assertEqual(self._import(path), sid)
                self.assertEqual(self._prompts(sid), ["hello", "two"])

    def test_line_nested_too_deep_or_with_a_huge_number_is_skipped(self):
        for sid, bad in (("deep1", TOO_DEEP),
                         ("digits1", b'{"type":"user","sessionId":"digits1","n":' + b"7" * 5000 + b"}")):
            with self.subTest(sid=sid):
                path = self._write(f"{sid}.jsonl", [
                    _entry(sid),
                    bad,
                    _entry(sid, uuid="u3", timestamp="2026-01-01T00:00:02Z",
                           message={"role": "user", "content": "after"}),
                ])

                self.assertEqual(self._import(path), sid)
                self.assertEqual(self._prompts(sid), ["hello", "after"])

    def test_line_nested_more_than_the_limit_is_skipped(self):
        # On 3.12 and later json.loads parses 5000 levels, so only the depth check skips them.
        for kind in ("dict", "list"):
            for levels in (600, 5000):
                sid = f"deep{kind}{levels}"
                with self.subTest(kind=kind, levels=levels):
                    path = self._write(f"{sid}.jsonl", [
                        _entry(sid),
                        _deep_tool_line(sid, kind, levels),
                        _entry(sid, uuid="u3", timestamp="2026-01-01T00:00:02Z",
                               message={"role": "user", "content": "after"}),
                    ])

                    code, err = self._cli(path)

                    self.assertEqual(code, 0, err)
                    self.assertEqual(self._prompts(sid), ["hello", "after"])
                    self.assertNotIn(EventType.TOOL_CALL.value,
                                     [event["event_type"] for event in self._events(sid)])
                    events = (self.trace / sid / "events.ndjson").read_bytes()

                    code, err = self._cli(path)

                    self.assertEqual(code, 0, err)
                    self.assertEqual(err.splitlines()[0],
                                     f"Session {sid} is already imported; nothing written.")
                    self.assertEqual((self.trace / sid / "events.ndjson").read_bytes(), events)

    def test_line_at_the_depth_limit_imports_and_one_level_more_is_skipped(self):
        limit = jsonl_import.MAX_LINE_DEPTH
        without = _full_log()
        del without[1]
        for kind in ("dict", "list"):
            with self.subTest(kind=kind):
                # The tool input sits below 4 levels: the line, message, content and tool_use.
                at_limit = _nested(kind, limit - 4)
                _, events = json.loads(self._session(
                    _with_field(1, "message.content.1.input", at_limit)))
                self.assertEqual([event["data"]["arguments"] for event in events
                                  if event["event_type"] == EventType.TOOL_CALL.value], [at_limit])

                over = _with_field(1, "message.content.1.input", _nested(kind, limit - 3))
                self.assertEqual(self._session(over), self._session(without))

    def test_log_of_only_unusable_lines_imports_as_an_empty_session(self):
        path = self._write("empty2.jsonl", b'not json\n[1]\n'
                           b'{"type":"user","sessionId":"x1","message":{"content":"\xff"}}\n'
                           b'{"type":"user","sessionId":"x1","message":{"content":"caf\xc3')

        self.assertEqual(self._import(path), "empty2")
        self.assertEqual(len(self._events("empty2")), 2)
        self.assertNotIn("conversation_id", self._meta("empty2"))

    @unittest.skipIf(os.name == "nt", "POSIX file modes and symlinks")
    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_log_that_cannot_be_read_fails_with_one_line(self):
        dir_log = self.tmp / "dir1.jsonl"
        dir_log.mkdir()
        perm_log = self._write("perm1.jsonl", [_entry("perm1")])
        perm_log.chmod(0)
        self.addCleanup(perm_log.chmod, 0o600)
        closed = self.tmp / "closed"
        closed.mkdir()
        closed_log = self._write("closed/in1.jsonl", [_entry("in1")])
        closed.chmod(0o600)
        self.addCleanup(closed.chmod, 0o700)
        loop_log = self.tmp / "loop1.jsonl"
        os.symlink("loop1.jsonl", loop_log)
        through_file = self._write("plain.txt", b"x") / "rp1.jsonl"

        for path, code in ((dir_log, errno.EISDIR), (perm_log, errno.EACCES),
                           (closed_log, errno.EACCES), (loop_log, errno.ELOOP),
                           (through_file, errno.ENOTDIR)):
            with self.subTest(path=path.name, code=errno.errorcode[code]):
                rc, err = self._cli(path)

                self.assertEqual(rc, 1)
                self.assertEqual(err, f"Cannot read session log {path}: {os.strerror(code)}\n")
                self.assertEqual(self._session_dirs(), [])

    def test_missing_log_keeps_its_message(self):
        path = self.tmp / "gone1.jsonl"

        code, err = self._cli(path)

        self.assertEqual(code, 1)
        self.assertEqual(err, f"Session log not found: {os.path.realpath(path)}\n")
        self.assertEqual(self._session_dirs(), [])

    def test_os_error_while_a_line_is_processed_is_not_converted(self):
        boom = OSError(errno.EIO, "boom")
        path = self._write("io1.jsonl", [_entry("io1")])

        with mock.patch.object(jsonl_import, "_parse_iso_timestamp", side_effect=boom):
            with self.assertRaises(OSError) as ctx:
                self._import(path)

        self.assertIs(ctx.exception, boom)

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_store_that_cannot_be_written_still_raises(self):
        (self.trace / "sessions").mkdir(parents=True)
        self.trace.chmod(0o555)
        self.addCleanup(self.trace.chmod, 0o755)
        path = self._write("ro1.jsonl", [_entry("ro1")])

        with self.assertRaises(PermissionError):
            self._cli(path)

    def test_null_parent_uuid_is_not_stored(self):
        path = self._write("par1.jsonl", [
            _entry("par1", parentUuid=None),
            _entry("par1", uuid="u2", parentUuid="u1", message={"role": "user", "content": "two"}),
        ])

        self._import(path)

        self.assertEqual([event["data"].get("source_parent_uuid") for event in self._events("par1")
                          if event["event_type"] == EventType.USER_PROMPT.value], [None, "u1"])

    def test_usage_that_is_not_an_object_does_not_hide_an_earlier_one(self):
        def response(uuid, timestamp, text, usage):
            return {"type": "assistant", "sessionId": "us1", "uuid": uuid, "timestamp": timestamp,
                    "message": {"id": "m1", "role": "assistant",
                                "content": [{"type": "text", "text": text}], "usage": usage}}
        path = self._write("us1.jsonl", [
            _entry("us1"),
            response("a1", "2026-01-01T00:00:01Z", "t1", {"input_tokens": 1, "output_tokens": 2}),
            response("a2", "2026-01-01T00:00:02Z", "t2", "x"),
        ])

        self._import(path)

        meta = self._meta("us1")
        self.assertEqual((meta.get("total_tokens"), meta.get("llm_requests")), (3, 1))

    @unittest.skipIf(os.name == "nt", "POSIX symlinks and FIFOs")
    def test_discover_lists_only_regular_files(self):
        project = self.tmp / "claude" / "projects" / "-w"
        project.mkdir(parents=True)
        (project / "good1.jsonl").write_text(json.dumps(_entry("good1")) + "\n")
        os.symlink(self._write("target.jsonl", [_entry("linked")]), project / "linked.jsonl")
        os.symlink(self.tmp / "nowhere.jsonl", project / "dangle.jsonl")
        os.symlink("loop.jsonl", project / "loop.jsonl")
        (project / "dir.jsonl").mkdir()
        os.mkfifo(project / "fifo.jsonl")

        found = discover_claude_sessions(str(self.tmp / "claude"))

        self.assertEqual([session["session_id"] for session in found], ["good1", "linked"])

    @unittest.skipIf(os.name == "nt", "POSIX file modes and symlinks")
    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_discover_finds_nothing_in_a_directory_it_cannot_read(self):
        projects_file = self.tmp / "claude2"
        projects_file.mkdir()
        (projects_file / "projects").write_text("x")
        projects_closed = self.tmp / "claude3"
        (projects_closed / "projects" / "-w").mkdir(parents=True)
        (projects_closed / "projects").chmod(0)
        self.addCleanup((projects_closed / "projects").chmod, 0o700)
        claude_closed = self.tmp / "claude4"
        (claude_closed / "projects" / "-w").mkdir(parents=True)
        (claude_closed / "projects" / "-w" / "good4.jsonl").write_text(json.dumps(_entry("good4")))
        claude_closed.chmod(0)
        self.addCleanup(claude_closed.chmod, 0o700)
        claude_loop = self.tmp / "claude5"
        os.symlink("claude5", claude_loop)

        for claude_dir in (projects_file, projects_closed, claude_closed, claude_loop):
            with self.subTest(claude_dir=claude_dir.name):
                self.assertEqual(discover_claude_sessions(str(claude_dir)), [])


if __name__ == "__main__":
    unittest.main()
