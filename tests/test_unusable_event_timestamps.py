"""Commands refuse a session whose event timestamp is not a Unix time in seconds.

An event's ``timestamp`` should be a Unix time in seconds. A command that
reads one refuses the session with ``Error: <message>`` and exit code 1,
before any output, file write, send or browser open. A command that already
skips unreadable sessions or events skips these too.

OTLP, temporal, Langfuse, a2a-tree, watch and the dashboard OTLP route used to
multiply a timestamp by a billion. For a str or a list that repeats it a
billion times. So these tests store only numbers, null, true and {} on disk
for those paths. They reach them with text or arrays only through the str and
list subclasses below, which fail the test when multiplied.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import http.client
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from _no_telemetry import without_telemetry
except ImportError:
    from tests._no_telemetry import without_telemetry

import agent_trace
from agent_trace import a2a, compaction, event_data, otlp, watch
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.server import _make_handler
from agent_trace.store import TraceStore
from agent_trace.watch import EventStreamer, StreamConfig

try:
    from test_non_object_event_data import (
        _ENV_PREFIXES,
        BAD,
        BAD_LAYOUT,
        CHILD,
        CHILD_LAYOUT,
        GOOD,
        NOW,
        CliTestCase,
        Row,
        _live_event,
        _row,
        _tree_snapshot,
        _watcher_config,
        _write_session,
    )
except ImportError:
    from tests.test_non_object_event_data import (
        _ENV_PREFIXES,
        BAD,
        BAD_LAYOUT,
        CHILD,
        CHILD_LAYOUT,
        GOOD,
        NOW,
        CliTestCase,
        Row,
        _live_event,
        _row,
        _tree_snapshot,
        _watcher_config,
        _write_session,
    )


class _NoMulStr(str):
    """A str that fails the test when anything multiplies it."""

    def __mul__(self, other):
        raise AssertionError("a text timestamp was multiplied")

    __rmul__ = __mul__
    __imul__ = __mul__


class _NoMulList(list):
    """A list that fails the test when anything multiplies it."""

    def __mul__(self, other):
        raise AssertionError("an array timestamp was multiplied")

    __rmul__ = __mul__
    __imul__ = __mul__


# (name, value, kind). The names match the probe in the brief.
TIMESTAMP_VALUES = (
    ("iso", "2026-01-01T00:00:00Z", "string"),
    ("numstr", "1795100829", "string"),
    ("null", None, "null"),
    ("true", True, "boolean"),
    ("object", {}, "object"),
    ("array", [1], "array"),
    ("nan", float("nan"), "non-finite"),
    ("inf", float("inf"), "non-finite"),
    ("huge", 1e300, "out-of-range"),
    ("neghuge", -1e300, "negative"),
    ("ms", NOW * 1000, "out-of-range"),
    ("neg", -5.0, "negative"),
    ("bigint", 10**400, "out-of-range"),
)
# Text and arrays reach the OTLP-like paths only as _NoMulStr and _NoMulList.
ALLOCATING = frozenset({"iso", "numstr", "array"})
SAFE_VALUES = tuple(case for case in TIMESTAMP_VALUES if case[0] not in ALLOCATING)
NO_MUL_VALUES = (
    ("iso", _NoMulStr("2026-01-01T00:00:00Z"), "string"),
    ("numstr", _NoMulStr("1795100829"), "string"),
    ("array", _NoMulList([1]), "array"),
)

ENDPOINT = "http://127.0.0.1:9/v1/traces"
LANGFUSE = (
    "--backend", "langfuse", "--langfuse-public-key", "pk", "--langfuse-secret-key", "sk",
    "--langfuse-host", "http://127.0.0.1:9",
)
TRACE_ID = "a" * 32
PARENT_SPAN_ID = "b" * 16

OTLP_TYPES = (
    "user_prompt", "assistant_response", "decision", "tool_call", "tool_result", "error",
    "llm_request", "llm_response",
)
GENAI_TYPES = tuple(t for t in OTLP_TYPES if t != "decision")
NOT_OTLP_TYPES = (
    "session_start", "session_end", "file_read", "file_write", "subagent_start", "subagent_stop",
)
LANGFUSE_TYPES = ("tool_call", "tool_result", "llm_request", "llm_response", "error")
NOT_LANGFUSE_TYPES = ("user_prompt", "assistant_response", "decision", "session_start", "file_write")


def _message(session_id: str | None, position: int, total: int, event_type: str, kind: str,
             others: int) -> str:
    subject = f"session {session_id}" if session_id else "session"
    article = "an" if kind in ("object", "array", "out-of-range") else "a"
    if others == 0:
        suffix = ""
    elif others == 1:
        suffix = "; 1 more event has an unusable timestamp"
    else:
        suffix = f"; {others} more events have unusable timestamps"
    return (
        f"{subject} event timestamp must be a Unix time in seconds: event {position} of {total} "
        f"({event_type}) has {article} {kind} timestamp{suffix}"
    )


def ts_message(session_id: str, event_type: str, kind: str, *, named: bool) -> str:
    """The message for every event of *event_type* in BAD or CHILD having a *kind* timestamp."""
    layout, total = (BAD_LAYOUT, 25) if session_id == BAD else (CHILD_LAYOUT, 7)
    position, count = layout[event_type]
    return _message(session_id if named else None, position, total, event_type, kind, count - 1)


def expected_ts_error(session_id: str, event_type: str, kind: str, *, named: bool) -> str:
    return f"Error: {ts_message(session_id, event_type, kind, named=named)}\n"


def restamp(session_dir: Path, event_type: str, value, *, where=None) -> None:
    """Store *value* as the timestamp of each event of *event_type* ("ALL" for
    every event) for which *where* holds, and rebuild the hash chain."""
    path = session_dir / "events.ndjson"
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record.pop("prev_hash", None)
        if event_type in ("ALL", record["event_type"]) and (where is None or where(record)):
            record["timestamp"] = value
        if lines:
            record["prev_hash"] = hashlib.sha256(lines[-1].encode()).hexdigest()
        lines.append(json.dumps(record, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_meta(session_dir: Path, **fields) -> None:
    path = session_dir / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _type_value(event: TraceEvent) -> str:
    return getattr(event.event_type, "value", event.event_type)


def _comparable_otlp(data: bytes | str):
    """An OTLP GenAI payload without the values that cover every event.

    The evidence health of the session reports each unusable timestamp, as it
    does on main. restamp rebuilds the hash chain, so the previous_hash of
    each later event changes too.
    """
    def covers_every_event(name: str) -> bool:
        return "evidence" in name or name.endswith("previous_hash")

    def walk(node):
        if isinstance(node, list):
            return [
                walk(item) for item in node
                if not (isinstance(item, dict) and covers_every_event(str(item.get("key", ""))))
            ]
        if isinstance(node, dict):
            if node.get("key") == "metadata":
                metadata = json.loads(node["value"]["stringValue"])
                return {"key": "metadata", "value": {
                    group: {name: value for name, value in fields.items()
                            if not covers_every_event(name)}
                    for group, fields in metadata.items()
                }}
            return {key: walk(value) for key, value in node.items()}
        return node

    return walk(json.loads(data))


@contextlib.contextmanager
def injected_timestamps(session_id: str, event_type: str, value, *, where=None):
    """Give the loaded events of *event_type* in *session_id* the timestamp *value*."""
    load_events = TraceStore.load_events

    def patched(store, *args, **kwargs):
        events = load_events(store, *args, **kwargs)
        for event in events:
            if (
                event.session_id == session_id
                and event_type in ("ALL", _type_value(event))
                and (where is None or where(event))
            ):
                event.timestamp = value
        return events

    with mock.patch.object(TraceStore, "load_events", patched):
        yield


@contextlib.contextmanager
def injected_meta(session_id: str, **fields):
    """Set *fields* on the loaded meta of *session_id*."""
    load_meta = TraceStore.load_meta

    def patched(store, *args, **kwargs):
        meta = load_meta(store, *args, **kwargs)
        if meta.session_id == session_id:
            for name, value in fields.items():
                setattr(meta, name, value)
        return meta

    with mock.patch.object(TraceStore, "load_meta", patched):
        yield


class TimestampCliTestCase(CliTestCase):
    def make_ts_workspace(self, session_id: str | None = None, event_type: str = "ALL", value=None,
                          *, tenant: bool = False, temporal: bool = False, where=None) -> Path:
        root = self.make_workspace(tenant=tenant)
        traces = root / "work" / ".agent-traces"
        if temporal:
            set_meta(traces / BAD, trace_id=TRACE_ID, parent_span_id=PARENT_SPAN_ID)
        if session_id is not None:
            restamp(traces / session_id, event_type, value, where=where)
        return root

    def check_rows(self, rows: list[Row], session_id: str, values=TIMESTAMP_VALUES) -> None:
        for index, row in enumerate(rows):
            name, value, kind = values[index % len(values)]
            with self.subTest(argv=row.argv, event_type=row.event_type, value=name):
                root = self.make_ts_workspace(session_id, row.event_type, value, tenant=row.tenant)
                self.assert_refused(
                    root, list(row.argv),
                    expected_ts_error(session_id, row.event_type, kind, named=row.named),
                )

    def outcome(self, root: Path, argv, *, files: str | None = None, read=bytes) -> tuple:
        """Exit code, stdout, stderr, network attempts and the files *files* matches."""
        result = self.run_cli(root, *argv)
        written = {}
        if files:
            written = {
                path.name: read(path.read_bytes()) for path in sorted((root / "work").glob(files))
            }
        return (
            result.code,
            result.out.replace(str(root), "ROOT"),
            result.err.replace(str(root), "ROOT"),
            len(result.network),
            written,
        )


BAD_SESSION_REFUSALS = [
    _row("compare", GOOD, BAD, event_type="tool_call", named=True),
    _row("compare", GOOD, BAD, "--format", "json", event_type="llm_request", named=True),
    _row("compare", "--tag", "claude", "--last", "2", event_type="user_prompt", named=True),
    _row("compliance", "export", BAD, event_type="error"),
    _row("compliance", "export", BAD, "--framework", "all", "--output", "cx.json",
         event_type="tool_call"),
    _row("compliance", "export", "--since", "7d", event_type="tool_call", named=True),
    _row("cost", BAD, event_type="session_end"),
    _row("cost", BAD, "--csv", event_type="file_read"),
    _row("cost", "--tenant", "acme", event_type="llm_response", named=True, tenant=True),
    _row("cost", BAD, "--tenant", "acme", event_type="decision", tenant=True),
    _row("diff", GOOD, BAD, event_type="subagent_start", named=True),
    _row("diff", BAD, GOOD, event_type="subagent_stop", named=True),
    _row("diff", GOOD, BAD, "--semantic", event_type="assistant_response", named=True),
    _row("diff", GOOD, BAD, "--compare", event_type="tool_result", named=True),
    _row("explain", BAD, event_type="file_write"),
    _row("export", BAD, "--format", "eu-ai-act", event_type="session_start"),
    _row("export", BAD, "--format", "eu-ai-act", "--output", "eu.json", event_type="user_prompt"),
    _row("oncall", "--rotation-start", "2026-01-01", event_type="file_write", named=True),
    _row("postmortem", BAD, event_type="error"),
    _row("pr-comment", BAD, "--dry-run", event_type="tool_call", named=True),
    _row("pr-comment", BAD, "--dry-run", "--review", event_type="llm_request", named=True),
    _row("review", BAD, event_type="file_read", named=True),
    _row("review", BAD, "--format", "markdown", event_type="session_end", named=True),
    _row("replay", BAD, event_type="session_start"),
    _row("replay", BAD, "--format", "html", "--output", "out.html", event_type="tool_result"),
    _row("replay", BAD, "--format", "html", event_type="decision"),
    _row("replay", BAD, "--diff", GOOD, event_type="error", named=True),
    _row("replay", GOOD, "--diff", BAD, event_type="user_prompt", named=True),
    _row("replay", BAD, "--expand-subagents", event_type="subagent_start", named=True),
    # Both hide the bad events, and both still refuse.
    _row("replay", BAD, "--filter", "tool_call,error", event_type="file_read"),
    _row("replay", BAD, "--limit", "5", event_type="session_end"),
    _row("sample", "--strategy", "worst", "--n", "2", event_type="session_end", named=True),
    _row("sample", "--strategy", "diverse", "--n", "2", event_type="session_start", named=True),
    _row("sample", "--strategy", "random", "--n", "2", "--seed", "1", event_type="session_end",
         named=True),
    _row("sample", "--strategy", "recent", "--n", "2", event_type="session_start", named=True),
    _row("sample", "--strategy", "worst", "--n", "2", "--output", "sample.jsonl",
         event_type="session_end", named=True),
    _row("standup", BAD, "--no-llm", event_type="tool_call"),
    _row("tenant", "report", event_type="user_prompt", named=True),
    _row("tenant", "report", event_type="llm_request", named=True, tenant=True),
    _row("tenant", "report", "--format", "json", event_type="tool_call", named=True, tenant=True),
    _row("timeline", BAD, event_type="assistant_response"),
    _row("timeline", BAD, "--format", "json", event_type="subagent_stop"),
    _row("token-budget", BAD, event_type="llm_request"),
    _row("share", BAD, "--stdout", event_type="tool_call"),
    _row("share", BAD, "--dry-run", event_type="file_write"),
    _row("share", BAD, "--dry-run", "--format", "json", event_type="error"),
    _row("share", BAD, "--stdout", "--include-content", event_type="decision"),
    _row("share", BAD, "--output", "s.html", event_type="session_start"),
    # The timestamp message, not the --postmortem gate's.
    _row("share", BAD, "--stdout", "--postmortem", event_type="tool_call"),
    _row("annotate", BAD, "--at", "5s", "--note", "x", event_type="tool_call"),
    _row("annotate", BAD, "--at", "12s", "--note", "y", "--label", "bug", event_type="user_prompt"),
    _row("annotate", BAD, "--at", "1m", "--note", "late", event_type="session_end"),
]

# With no session ID these commands read the latest session, which is CHILD.
CHILD_SESSION_REFUSALS = [
    _row("cost", "--since", "7d", event_type="user_prompt"),
    _row("explain", event_type="tool_call"),
    _row("replay", CHILD, event_type="session_start"),
    _row("replay", event_type="error"),
    _row("standup", "--no-llm", event_type="tool_result"),
    _row("share", "--stdout", event_type="assistant_response"),
    _row("timeline", event_type="session_end"),
    _row("replay", BAD, "--expand-subagents", event_type="user_prompt", named=True),
    _row("compare", "--tag", "claude", "--last", "2", event_type="tool_call", named=True),
    _row("sample", "--strategy", "worst", "--n", "2", event_type="session_start", named=True),
    _row("tenant", "report", event_type="user_prompt", named=True),
    _row("oncall", "--rotation-start", "2026-01-01", event_type="tool_call", named=True),
    _row("cost", "--tenant", "acme", event_type="user_prompt", named=True, tenant=True),
    _row("compliance", "export", "--since", "7d", event_type="error", named=True),
]

# Stored values only: these paths multiplied the timestamp.
OTLP_REFUSALS = [
    _row("export", BAD, "--format", "otlp", event_type="decision"),
    _row("export", BAD, "--format", "otlp", "--endpoint", ENDPOINT, event_type="user_prompt"),
    _row("export", BAD, "--format", "otlp-genai", event_type="assistant_response"),
    _row("export", BAD, "--format", "otlp-genai", "--endpoint", ENDPOINT, event_type="tool_call"),
    _row("export", BAD, *LANGFUSE, event_type="llm_response"),
]


class TestRequireUsableTimestamps(unittest.TestCase):
    @staticmethod
    def _event(event_type: EventType = EventType.TOOL_CALL, ts=1.0) -> TraceEvent:
        return TraceEvent(event_type=event_type, timestamp=ts, session_id="unit-session", data={})

    def _message_for(self, events, **kwargs) -> str | None:
        try:
            event_data.require_usable_timestamps(events, **kwargs)
        except event_data.UnusableTimestampError as exc:
            return str(exc)
        return None

    def test_each_kind_gets_its_word_and_article(self):
        cases = [
            (True, "a boolean"), (False, "a boolean"),
            ("", "a string"), ("1795100829", "a string"), (_NoMulStr("2026-01-01"), "a string"),
            (None, "a null"),
            ({}, "an object"), ({"seconds": 1}, "an object"),
            ([1], "an array"), ((1,), "an array"), (_NoMulList([1]), "an array"),
            (float("nan"), "a non-finite"), (float("inf"), "a non-finite"),
            (float("-inf"), "a non-finite"),
            (-5.0, "a negative"), (-1e300, "a negative"), (-1, "a negative"),
            (-(10**400), "a negative"),
            (1e300, "an out-of-range"), (NOW * 1000, "an out-of-range"),
            (10**400, "an out-of-range"), (2**64, "an out-of-range"),
            (b"1795100829", "a non-JSON"),
        ]
        for value, words in cases:
            with self.subTest(value=type(value).__name__, words=words):
                self.assertEqual(
                    self._message_for([self._event(ts=value)]),
                    "session event timestamp must be a Unix time in seconds: "
                    f"event 1 of 1 (tool_call) has {words} timestamp",
                )

    def test_usable_timestamps_pass(self):
        for value in (0, 0.0, 1, 1.5, NOW, int(NOW), 1795111200):
            with self.subTest(value=value):
                self.assertIsNone(self._message_for([self._event(ts=value)]))
        self.assertIsNone(self._message_for([]))

    def test_message_counts_the_other_unusable_events(self):
        good, bad = 1.0, None
        cases = [
            ([good, bad, good], "event 2 of 3 (tool_call) has a null timestamp"),
            ([bad, good, bad], "event 1 of 3 (tool_call) has a null timestamp"
                               "; 1 more event has an unusable timestamp"),
            ([good, bad, bad, bad, bad], "event 2 of 5 (tool_call) has a null timestamp"
                                         "; 3 more events have unusable timestamps"),
        ]
        for stamps, tail in cases:
            with self.subTest(stamps=stamps):
                self.assertEqual(
                    self._message_for([self._event(ts=ts) for ts in stamps]),
                    f"session event timestamp must be a Unix time in seconds: {tail}",
                )

    def test_session_id_names_the_session(self):
        self.assertEqual(
            self._message_for([self._event(ts=None)], session_id="abc123"),
            "session abc123 event timestamp must be a Unix time in seconds: "
            "event 1 of 1 (tool_call) has a null timestamp",
        )

    def test_event_types_limit_the_check(self):
        events = [
            self._event(EventType.TOOL_CALL, None),
            self._event(EventType.ERROR, 1.0),
            self._event(EventType.ERROR, "late"),
        ]
        self.assertEqual(
            self._message_for(events, event_types={EventType.ERROR}),
            "session event timestamp must be a Unix time in seconds: "
            "event 3 of 3 (error) has a string timestamp; 1 more event has an unusable timestamp",
        )
        self.assertIsNone(self._message_for(events, event_types={EventType.LLM_REQUEST}))
        self.assertIsNone(self._message_for(events, event_types=set()))

    def test_indexes_limit_the_check(self):
        events = [self._event(ts=None), self._event(ts=True), self._event(ts=1.0)]
        self.assertEqual(
            self._message_for(events, indexes={1, 2}),
            "session event timestamp must be a Unix time in seconds: "
            "event 2 of 3 (tool_call) has a boolean timestamp; 1 more event has an unusable timestamp",
        )
        self.assertIsNone(self._message_for(events, indexes={2, 7}))
        self.assertIsNone(self._message_for(events, indexes=set()))

    def test_event_types_and_indexes_both_apply(self):
        events = [
            self._event(EventType.ERROR, None),
            self._event(EventType.TOOL_CALL, None),
            self._event(EventType.TOOL_CALL, {}),
        ]
        self.assertEqual(
            self._message_for(events, event_types={EventType.TOOL_CALL}, indexes={0, 2}),
            "session event timestamp must be a Unix time in seconds: "
            "event 3 of 3 (tool_call) has an object timestamp; 2 more events have unusable timestamps",
        )
        self.assertIsNone(
            self._message_for(events, event_types={EventType.ERROR}, indexes={1, 2}),
        )

    def test_message_never_shows_the_value(self):
        text = "QV" * 2121
        for value in (text, [text], {"QVQV": text}, 10**400, -(10**400), NOW * 1000, 1e300):
            with self.subTest(value=type(value).__name__):
                message = self._message_for([self._event(ts=value)], session_id="s")
                self.assertNotIn("QV", message)
                self.assertNotIn("4242", message)
                self.assertNotIn("1000000", message)
                self.assertNotIn("e+", message)
                self.assertNotIn("\n", message)
                self.assertNotIn(repr(value)[:12], message)

    def test_session_times(self):
        def meta(**fields) -> SessionMeta:
            return SessionMeta(session_id="s", **{"started_at": 1.0, **fields})

        event_data.require_usable_session_times(meta())
        event_data.require_usable_session_times(meta(ended_at=None))
        event_data.require_usable_session_times(meta(ended_at=0))
        for started in (None, True, "1795100829", float("nan"), -5.0, NOW * 1000, 10**400):
            with self.subTest(started_at=started):
                with self.assertRaises(event_data.UnusableTimestampError) as caught:
                    event_data.require_usable_session_times(
                        meta(started_at=started, ended_at="later"),
                    )
                self.assertEqual(
                    str(caught.exception), "session started_at must be a Unix time in seconds",
                )
        for ended in ("", False, float("nan"), [2.0], 10**400, NOW * 1000, 1e300):
            with self.subTest(ended_at=ended):
                with self.assertRaises(event_data.UnusableTimestampError) as caught:
                    event_data.require_usable_session_times(meta(ended_at=ended))
                self.assertEqual(
                    str(caught.exception), "session ended_at must be a Unix time in seconds",
                )

    def test_left_out_timestamp_warning(self):
        self.assertEqual(
            event_data.left_out_timestamp_warning("stream", self._event(ts=None)),
            "[stream] Left one tool_call event out of the OTLP stream because its timestamp "
            "is not a Unix time in seconds.\n",
        )


class TestCommandsRefuseUnusableTimestamps(TimestampCliTestCase):
    def test_commands_refuse_an_unusable_timestamp_in_the_session(self):
        self.check_rows(BAD_SESSION_REFUSALS, BAD)

    def test_commands_refuse_an_unusable_timestamp_in_the_subagent_session(self):
        self.check_rows(CHILD_SESSION_REFUSALS, CHILD)

    def test_otlp_and_langfuse_exports_refuse_stored_values(self):
        self.check_rows(OTLP_REFUSALS, BAD, SAFE_VALUES)

    def test_explain_refuses_every_kind(self):
        for name, value, kind in TIMESTAMP_VALUES:
            with self.subTest(value=name):
                root = self.make_ts_workspace(BAD, "tool_call", value)
                self.assert_refused(
                    root, ["explain", BAD], expected_ts_error(BAD, "tool_call", kind, named=False),
                )

    def test_message_counts_every_other_unusable_event(self):
        root = self.make_ts_workspace(BAD, "ALL", None)
        self.assert_refused(
            root, ["explain", BAD],
            "Error: session event timestamp must be a Unix time in seconds: event 1 of 25 "
            "(session_start) has a null timestamp; 24 more events have unusable timestamps\n",
        )

    def test_installed_entry_point_prints_no_traceback(self):
        root = self.make_ts_workspace(BAD, "tool_call", "2026-01-01T00:00:00Z")
        env = {key: value for key, value in os.environ.items() if not key.startswith(_ENV_PREFIXES)}
        env.update({
            "HOME": str(root / "home"),
            "PYTHONPATH": str(Path(agent_trace.__file__).resolve().parent.parent),
        })
        for argv, named in ((["explain", BAD], False), (["replay", BAD], False),
                            (["compare", GOOD, BAD], True)):
            with self.subTest(argv=argv):
                before = _tree_snapshot(root)
                completed = subprocess.run(
                    [sys.executable, "-m", "agent_trace.entry", *argv],
                    cwd=root / "work", env=without_telemetry(env),
                    capture_output=True, text=True, timeout=120,
                )
                self.assertNotIn("Traceback", completed.stderr)
                self.assertEqual(
                    (completed.returncode, completed.stdout, completed.stderr),
                    (1, "", expected_ts_error(BAD, "tool_call", "string", named=named)),
                )
                self.assertEqual(_tree_snapshot(root), before)


class TestTextAndArraysAreNeverMultiplied(TimestampCliTestCase):
    """Text and array timestamps are refused before the OTLP-like paths multiply them."""

    def test_exports_refuse_text_and_arrays(self):
        rows = [
            (("export", BAD, "--format", "otlp"), "tool_call", False),
            (("export", BAD, "--format", "otlp", "--endpoint", ENDPOINT), "llm_request", False),
            (("export", BAD, "--format", "otlp-genai"), "error", False),
            (("export", BAD, "--format", "otlp-genai", "--endpoint", ENDPOINT), "user_prompt", False),
            (("export", BAD, "--format", "temporal"), "tool_result", True),
            (("export", BAD, "--format", "temporal", "--endpoint", ENDPOINT), "llm_response", True),
            (("export", BAD, *LANGFUSE), "tool_call", False),
            (("export", BAD, *LANGFUSE), "error", False),
        ]
        for index, (argv, event_type, temporal) in enumerate(rows):
            name, value, kind = NO_MUL_VALUES[index % len(NO_MUL_VALUES)]
            with self.subTest(argv=argv, event_type=event_type, value=name):
                root = self.make_ts_workspace(temporal=temporal)
                with injected_timestamps(BAD, event_type, value):
                    self.assert_refused(
                        root, list(argv), expected_ts_error(BAD, event_type, kind, named=False),
                    )

    def test_ts_to_nanos_refuses_what_it_cannot_multiply(self):
        for value in (_NoMulStr("1795100829"), _NoMulList([1]), True, None, {}):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(event_data.UnusableTimestampError) as caught:
                    otlp._ts_to_nanos(value)
                self.assertEqual(str(caught.exception), "timestamp must be a Unix time in seconds")
        self.assertEqual(otlp._ts_to_nanos(1.5), "1500000000")
        self.assertEqual(otlp._ts_to_nanos(2), "2000000000")

    def test_a2a_spans_refuse_what_they_cannot_multiply(self):
        def report(timestamp) -> a2a.A2ATreeReport:
            call = a2a.A2ACallEvent(
                event_id="call-0001-abcdef", session_id="d4e5f6a7b8c90004", timestamp=timestamp,
                agent_id="helper", agent_url="", task="t", response={}, sub_session_id="",
                duration_ms=500.0, cost_usd=0.0, success=True,
            )
            node = a2a.AgentNode(
                session_id="d4e5f6a7b8c90004", agent_id="root", depth=0, cost_usd=0.0,
                duration_ms=0.0, tool_calls=1, a2a_calls=[call],
            )
            return a2a.A2ATreeReport(root=node, total_cost=0.0, total_agents=1, max_depth=0)

        for value in (_NoMulStr("1795100829"), _NoMulList([1]), True):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(event_data.UnusableTimestampError) as caught:
                    a2a.a2a_calls_to_otlp_spans(report(value))
                self.assertEqual(str(caught.exception), "timestamp must be a Unix time in seconds")
        spans = a2a.a2a_calls_to_otlp_spans(report(2))
        self.assertEqual(
            [(span["startTimeUnixNano"], span["endTimeUnixNano"]) for span in spans],
            [(0, 0), (2_000_000_000, 2_500_000_000)],
        )


class TestSessionTimes(TimestampCliTestCase):
    """OTLP export reads the meta started_at and ended_at for the root span."""

    STARTED = "Error: session started_at must be a Unix time in seconds\n"
    ENDED = "Error: session ended_at must be a Unix time in seconds\n"

    def test_null_started_at_loads_as_none(self):
        root = self.make_ts_workspace()
        traces = root / "work" / ".agent-traces"
        set_meta(traces / BAD, started_at=None)
        self.assertIsNone(TraceStore(str(traces)).load_meta(BAD).started_at)

    def test_exports_refuse_an_unusable_started_at(self):
        argvs = [
            ("export", BAD, "--format", "otlp"),
            ("export", BAD, "--format", "otlp", "--endpoint", ENDPOINT),
            ("export", BAD, "--format", "otlp-genai"),
            ("export", BAD, "--format", "temporal"),
            ("export", BAD, "--format", "temporal", "--endpoint", ENDPOINT),
        ]
        values = (None, float("nan"), True, -5.0, NOW * 1000, 10**400)
        for index, argv in enumerate(argvs):
            value = values[index % len(values)]
            with self.subTest(argv=argv, started_at=value):
                root = self.make_ts_workspace(temporal=True)
                set_meta(root / "work" / ".agent-traces" / BAD, started_at=value)
                self.assert_refused(root, list(argv), self.STARTED)

    def test_exports_refuse_text_started_at(self):
        for argv in (("export", BAD, "--format", "otlp"), ("export", BAD, "--format", "temporal")):
            with self.subTest(argv=argv):
                root = self.make_ts_workspace(temporal=True)
                with injected_meta(BAD, started_at=_NoMulStr("2026-01-01T00:00:00Z")):
                    self.assert_refused(root, list(argv), self.STARTED)

    def test_exports_refuse_an_unusable_ended_at(self):
        rows = [
            (("export", BAD, "--format", "otlp"), False),
            (("export", BAD, "--format", "otlp-genai"), float("nan")),
            (("export", BAD, "--format", "temporal"), {}),
            (("export", BAD, "--format", "otlp", "--endpoint", ENDPOINT), 10**400),
            (("export", BAD, "--format", "temporal", "--endpoint", ENDPOINT), NOW * 1000),
        ]
        for argv, value in rows:
            with self.subTest(argv=argv, ended_at=value):
                root = self.make_ts_workspace(temporal=True)
                set_meta(root / "work" / ".agent-traces" / BAD, ended_at=value)
                self.assert_refused(root, list(argv), self.ENDED)
        root = self.make_ts_workspace()
        with injected_meta(BAD, ended_at=_NoMulStr("")):
            self.assert_refused(root, ["export", BAD, "--format", "otlp-genai"], self.ENDED)

    def test_open_session_exports(self):
        for fmt in ("otlp", "otlp-genai", "temporal"):
            with self.subTest(format=fmt):
                root = self.make_ts_workspace(temporal=True)
                set_meta(root / "work" / ".agent-traces" / BAD, ended_at=None)
                result = self.run_cli(root, "export", BAD, "--format", fmt)
                self.assertEqual(result.code, 0, result.err)
                self.assertTrue((root / "work" / f"trace-{BAD[:12]}-{fmt}.json").is_file())


class TestCommandsCheckTheEventsTheyRead(TimestampCliTestCase):
    """A subset check refuses on each event it covers and ignores the others."""

    def assert_refuses_each(self, argv, event_types, *, named=False, temporal=False):
        for index, event_type in enumerate(event_types):
            name, value, kind = SAFE_VALUES[index % len(SAFE_VALUES)]
            with self.subTest(argv=argv, event_type=event_type, value=name):
                root = self.make_ts_workspace(BAD, event_type, value, temporal=temporal)
                self.assert_refused(
                    root, list(argv), expected_ts_error(BAD, event_type, kind, named=named),
                )

    def assert_unchanged(self, argv, event_types, *, files=None, read=bytes, temporal=False,
                         values=SAFE_VALUES):
        clean = self.outcome(self.make_ts_workspace(temporal=temporal), argv, files=files, read=read)
        for index, event_type in enumerate(event_types):
            name, value, _ = values[index % len(values)]
            with self.subTest(argv=argv, event_type=event_type, value=name):
                root = self.make_ts_workspace(BAD, event_type, value, temporal=temporal)
                self.assertEqual(self.outcome(root, argv, files=files, read=read), clean)
        return clean

    def test_otlp_reads_eight_types(self):
        argv = ("export", BAD, "--format", "otlp")
        self.assert_refuses_each(argv, OTLP_TYPES)
        clean = self.assert_unchanged(argv, NOT_OTLP_TYPES, files="trace-*.json")
        self.assertEqual(clean[0], 0)
        self.assertEqual(list(clean[4]), [f"trace-{BAD[:12]}-otlp.json"])

    def test_otlp_genai_reads_seven_types(self):
        argv = ("export", BAD, "--format", "otlp-genai")
        self.assert_refuses_each(argv, GENAI_TYPES)
        clean = self.assert_unchanged(
            argv, ("decision", *NOT_OTLP_TYPES), files="trace-*.json", read=_comparable_otlp,
        )
        self.assertEqual(clean[0], 0)
        self.assertEqual(list(clean[4]), [f"trace-{BAD[:12]}-otlp-genai.json"])

    def test_temporal_reads_seven_types(self):
        for argv in (("export", BAD, "--format", "temporal"),
                     ("export", BAD, "--format", "temporal", "--endpoint", ENDPOINT)):
            self.assert_refuses_each(argv, GENAI_TYPES, temporal=True)
        clean = self.assert_unchanged(
            ("export", BAD, "--format", "temporal"), ("decision", *NOT_OTLP_TYPES),
            files="trace-*.json", read=_comparable_otlp, temporal=True,
        )
        self.assertEqual(clean[0], 0)
        self.assertEqual(list(clean[4]), [f"trace-{BAD[:12]}-temporal.json"])

    def test_temporal_without_context_keeps_its_message(self):
        root = self.make_ts_workspace(BAD, "tool_call", None)
        result = self.run_cli(root, "export", BAD, "--format", "temporal")
        self.assertEqual(result.code, 1)
        self.assertIn("session has no Temporal trace context", result.err)

    def test_langfuse_reads_five_types(self):
        argv = ("export", BAD, *LANGFUSE)
        self.assert_refuses_each(argv, LANGFUSE_TYPES)
        clean = self.assert_unchanged(argv, NOT_LANGFUSE_TYPES)
        self.assertEqual(clean[:4], (
            1, "Langfuse: 0/1 sessions exported.\n",
            f"Langfuse: export failed for session {BAD[:12]}\n", 1,
        ))

    def test_compliance_reads_tool_calls_and_errors(self):
        argv = ("compliance", "export", BAD, "--framework", "all", "--output", "cx.json")
        self.assert_refuses_each(argv, ("tool_call", "error"))
        clean = self.assert_unchanged(argv, ("user_prompt", "llm_request", "file_write"))
        self.assertEqual(clean[0], 0)

    def test_oncall_reads_tool_calls_and_file_writes(self):
        argv = ("oncall", "--rotation-start", "2026-01-01")
        self.assert_refuses_each(argv, ("tool_call", "file_write"), named=True)
        clean = self.assert_unchanged(argv, ("user_prompt", "error"), values=TIMESTAMP_VALUES)
        self.assertEqual(clean[0], 0)

    def test_sample_reads_the_first_and_last_events(self):
        argv = ("sample", "--strategy", "worst", "--n", "2")
        self.assert_refuses_each(argv, ("session_start", "session_end"), named=True)
        clean = self.assert_unchanged(argv, ("tool_call", "user_prompt"), values=TIMESTAMP_VALUES)
        self.assertEqual(clean[0], 0)

    def test_token_budget_reads_the_first_event_and_llm_requests(self):
        argv = ("token-budget", BAD)
        self.assert_refuses_each(argv, ("session_start", "llm_request"))
        clean = self.assert_unchanged(
            argv, ("tool_call", "session_end", "llm_response"), values=TIMESTAMP_VALUES,
        )
        self.assertEqual(clean[0], 0)

    def test_commands_that_read_no_timestamp_are_unchanged(self):
        for argv in (("cost", BAD, "--breakdown", "provider"), ("postmortem", "--list"),
                     ("replay", BAD, "--tree")):
            clean = self.assert_unchanged(argv, ("ALL",) * len(TIMESTAMP_VALUES),
                                          values=TIMESTAMP_VALUES)
            self.assertEqual(clean[0], 0)

    def test_annotate_event_is_unchanged(self):
        for name, value, _ in TIMESTAMP_VALUES:
            with self.subTest(value=name):
                root = self.make_ts_workspace(BAD, "ALL", value)
                result = self.run_cli(root, "annotate", BAD, "--event", "3", "--note", "x")
                self.assertEqual((result.code, result.err), (0, ""))
                self.assertNotIn("timestamp", result.out)


A2A_ROOT = "d4e5f6a7b8c90004"
A2A_CHILD = "e5f6a7b8c9d00005"


def _a2a_events(session_id: str, start: float, *, sub_session_id: str = "") -> list[dict]:
    def event(event_type, offset, name, data):
        return {
            "event_type": event_type,
            "timestamp": start + offset,
            "event_id": f"{name}-{session_id[:8]}",
            "session_id": session_id,
            "data": data,
        }

    return [
        event("session_start", 0, "start", {"provider": "claude"}),
        event("tool_call", 1, "a2acall", {
            "event_subtype": a2a.A2A_CALL, "agent_id": "helper",
            "agent_url": "http://agents.example/a2a/helper", "task": "summarize the diff",
            "response": {}, "sub_session_id": sub_session_id, "duration_ms": 1500.0,
            "cost_usd": 0.0, "success": True, "error": "",
        }),
        event("tool_call", 2, "readcall", {"tool_name": "Read", "arguments": {"file_path": "a.py"}}),
        event("session_end", 3, "end", {}),
    ]


def _is_a2a_call(record) -> bool:
    data = record["data"] if isinstance(record, dict) else record.data
    return data.get("event_subtype") == a2a.A2A_CALL


class TestA2aTree(TimestampCliTestCase):
    """a2a-tree JSON reads each A2A call's timestamp. The text tree does not."""

    def make_a2a_workspace(self) -> Path:
        root = self.make_ts_workspace()
        traces = root / "work" / ".agent-traces"
        start = NOW - 1800
        _write_session(traces, A2A_ROOT, start,
                       _a2a_events(A2A_ROOT, start, sub_session_id=A2A_CHILD), total_tokens=0)
        _write_session(traces, A2A_CHILD, start + 1.5, _a2a_events(A2A_CHILD, start + 1.5),
                       total_tokens=0)
        return root

    def test_json_refuses_an_unusable_call_timestamp(self):
        for index, session_id in enumerate((A2A_ROOT, A2A_CHILD, A2A_ROOT, A2A_CHILD)):
            name, value, kind = SAFE_VALUES[index]
            with self.subTest(session_id=session_id, value=name):
                root = self.make_a2a_workspace()
                restamp(root / "work" / ".agent-traces" / session_id, "tool_call", value,
                        where=_is_a2a_call)
                self.assert_refused(
                    root, ["a2a-tree", A2A_ROOT, "--format", "json"],
                    "Error: " + _message(session_id, 2, 4, "tool_call", kind, 0) + "\n",
                )

    def test_json_refuses_text_and_arrays(self):
        for name, value, kind in NO_MUL_VALUES:
            with self.subTest(value=name):
                root = self.make_a2a_workspace()
                with injected_timestamps(A2A_CHILD, "tool_call", value, where=_is_a2a_call):
                    self.assert_refused(
                        root, ["a2a-tree", A2A_ROOT, "--format", "json"],
                        "Error: " + _message(A2A_CHILD, 2, 4, "tool_call", kind, 0) + "\n",
                    )

    def test_json_ignores_other_events(self):
        # The Read call sits inside the only phase, so the cost estimate does
        # not subtract its timestamp.
        clean = self.outcome(self.make_a2a_workspace(), ["a2a-tree", A2A_ROOT, "--format", "json"])
        self.assertEqual(clean[0], 0)
        for name, value, _ in SAFE_VALUES:
            with self.subTest(value=name):
                root = self.make_a2a_workspace()
                traces = root / "work" / ".agent-traces"
                for session_id in (A2A_ROOT, A2A_CHILD):
                    restamp(traces / session_id, "tool_call", value,
                            where=lambda r: not _is_a2a_call(r))
                self.assertEqual(
                    self.outcome(root, ["a2a-tree", A2A_ROOT, "--format", "json"]), clean,
                )

    def test_text_tree_reads_no_call_timestamp(self):
        clean = self.outcome(self.make_a2a_workspace(), ["a2a-tree", A2A_ROOT])
        self.assertEqual(clean[0], 0)
        for name, value, _ in SAFE_VALUES:
            with self.subTest(value=name):
                root = self.make_a2a_workspace()
                restamp(root / "work" / ".agent-traces" / A2A_ROOT, "tool_call", value,
                        where=_is_a2a_call)
                self.assertEqual(self.outcome(root, ["a2a-tree", A2A_ROOT]), clean)

    def test_text_duration_is_read_as_zero(self):
        root = self.make_a2a_workspace()
        set_meta(root / "work" / ".agent-traces" / A2A_ROOT, total_duration_ms="abc")
        result = self.run_cli(root, "a2a-tree", A2A_ROOT, "--format", "json")
        self.assertEqual((result.code, result.err), (0, ""))
        spans = json.loads(result.out)["spans"]
        root_span = next(span for span in spans if span["spanId"] == A2A_ROOT[:16])
        self.assertEqual(root_span["endTimeUnixNano"], 0)
        text = self.run_cli(root, "a2a-tree", A2A_ROOT)
        self.assertEqual((text.code, text.err), (0, ""))


class _Receiver(BaseHTTPRequestHandler):
    """Record the body of each POST."""

    received: ClassVar[list[bytes]] = []

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Receiver.received.append(body)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


def _spans(body: bytes) -> list[str]:
    """Each span of an OTLP body as sorted JSON, without the root span's times.

    The stream gives each session a meta whose started_at is the current time.
    The values that cover every event are left out, as in _comparable_otlp.
    """
    spans = []
    for resource in _comparable_otlp(body)["resourceSpans"]:
        for scope in resource["scopeSpans"]:
            for span in scope["spans"]:
                span = dict(span)
                if not span.get("parentSpanId"):
                    span.pop("startTimeUnixNano", None)
                    span.pop("endTimeUnixNano", None)
                spans.append(json.dumps(span, sort_keys=True))
    return spans


class TestWatchStreamLeavesOutUnusableTimestamps(unittest.TestCase):
    """watch --stream-otlp leaves out each event whose timestamp it would read and cannot."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _Receiver)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/collector"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def _stream(self, events: list[TraceEvent], batch_size: int) -> tuple[list[list[str]], str]:
        _Receiver.received.clear()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            streamer = EventStreamer(StreamConfig(
                url=self.url, otlp=True, batch_size=batch_size, flush_interval=60.0,
            ))
            for event in events:
                streamer.enqueue(event)
            streamer.stop()
        self.assertFalse(streamer._thread.is_alive())
        return [_spans(body) for body in _Receiver.received], stderr.getvalue()

    def _batch(self):
        prompt = _live_event(EventType.USER_PROMPT, {"prompt": "fix the test"}, ts=1.0)
        bad_stamp = _live_event(EventType.LLM_REQUEST, {"model": "m"}, ts=float("nan"))
        bad_data = _live_event(EventType.ERROR, ["not", "an", "object"], ts=3.0)
        both = _live_event(EventType.ASSISTANT_RESPONSE, "plain text", ts=None)
        call = _live_event(EventType.TOOL_CALL, {"tool_name": "Read", "arguments": {}}, ts=5.0)
        later = _live_event(EventType.ASSISTANT_RESPONSE, {"text": "done"}, ts=6.0)
        return prompt, bad_stamp, bad_data, both, call, later

    def test_lines_follow_the_batch_order(self):
        prompt, bad_stamp, bad_data, both, call, _ = self._batch()
        decision = _live_event(EventType.DECISION, {"decision": "x"}, ts=_NoMulStr("soon"))
        start = _live_event(EventType.SESSION_START, {}, ts=_NoMulList([1]))
        text = _live_event(EventType.TOOL_RESULT, {"result": "ok"}, ts=_NoMulStr("2026-01-01"))
        events = [prompt, bad_stamp, bad_data, both, decision, start, text, call]
        kept, lines = watch._otlp_readable_events(events)
        self.assertEqual([id(event) for event in kept],
                         [id(event) for event in (prompt, decision, start, call)])
        self.assertEqual(lines, "".join([
            "[stream] Left one llm_request event out of the OTLP stream because its timestamp "
            "is not a Unix time in seconds.\n",
            event_data.left_out_event_warning("stream", bad_data),
            "[stream] Left one assistant_response event out of the OTLP stream because its "
            "timestamp is not a Unix time in seconds.\n",
            event_data.left_out_timestamp_warning("stream", text),
        ]))
        self.assertIn("because its data is an array, not an object", lines)

    def test_stream_sends_the_rest_of_the_batch_and_later_batches(self):
        prompt, bad_stamp, bad_data, both, call, later = self._batch()
        posts, stderr = self._stream([prompt, bad_stamp, bad_data, both, call, later], batch_size=5)
        expected, expected_stderr = self._stream([prompt, call, later], batch_size=2)
        self.assertEqual(expected_stderr, "")
        self.assertEqual(len(expected), 2)
        self.assertEqual(posts, expected)
        self.assertEqual(stderr, "".join([
            event_data.left_out_timestamp_warning("stream", bad_stamp),
            event_data.left_out_event_warning("stream", bad_data),
            event_data.left_out_timestamp_warning("stream", both),
        ]))

    def test_stream_keeps_events_whose_timestamp_it_does_not_read(self):
        prompt, _, _, _, call, _ = self._batch()
        decision = _live_event(EventType.DECISION, {"decision": "x"}, ts=float("nan"))
        start = _live_event(EventType.SESSION_START, {"model": "m"}, ts=None)
        posts, stderr = self._stream([prompt, decision, start, call], batch_size=4)
        clean, _ = self._stream(
            [prompt, dataclasses.replace(decision, timestamp=2.0),
             dataclasses.replace(start, timestamp=0.5), call],
            batch_size=4,
        )
        self.assertEqual(stderr, "")
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts, clean)


@contextlib.contextmanager
def _dashboard(traces: Path):
    """Serve the dashboard for *traces* and yield its port."""
    store = TraceStore(str(traces), use_workspace_env=False)
    server = HTTPServer(("127.0.0.1", 0), _make_handler(store, threading.Lock(), dashboard=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _get(port: int, path: str) -> tuple[int | None, str]:
    """Return the status and body, or None when the server drops the connection."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        # The handler logs each request to stderr.
        with redirect_stderr(io.StringIO()):
            connection.request("GET", path)
            try:
                response = connection.getresponse()
            except ConnectionResetError:  # http.client.RemoteDisconnected
                return None, ""
            return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


class TestDashboardRoutesAnswer400(TimestampCliTestCase):
    ROUTES = [
        (f"/api/sessions/{BAD}/review", "tool_call"),
        (f"/api/sessions/{BAD}/compare?other={GOOD}", "llm_request"),
        (f"/api/sessions/{GOOD}/compare?other={BAD}", "file_read"),
        (f"/api/sessions/{BAD}/share?mode=minimized", "error"),
        (f"/api/sessions/{BAD}/otlp", "decision"),
    ]

    def get(self, root: Path, path: str) -> tuple[int | None, object]:
        with _dashboard(root / "work" / ".agent-traces") as port:
            status, body = _get(port, path)
        return status, (json.loads(body) if body else None)

    def test_routes_answer_400_with_the_message(self):
        for index, (path, event_type) in enumerate(self.ROUTES):
            name, value, kind = SAFE_VALUES[index % len(SAFE_VALUES)]
            with self.subTest(path=path, value=name):
                root = self.make_ts_workspace(BAD, event_type, value)
                self.assertEqual(
                    self.get(root, path),
                    (400, {"error": ts_message(BAD, event_type, kind, named=True)}),
                )

    def test_otlp_route_refuses_text_and_unusable_session_times(self):
        root = self.make_ts_workspace()
        path = f"/api/sessions/{BAD}/otlp"
        with injected_timestamps(BAD, "tool_result", _NoMulList([1])):
            self.assertEqual(
                self.get(root, path),
                (400, {"error": ts_message(BAD, "tool_result", "array", named=True)}),
            )
        set_meta(root / "work" / ".agent-traces" / BAD, started_at=float("nan"))
        self.assertEqual(
            self.get(root, path), (400, {"error": "session started_at must be a Unix time in seconds"}),
        )
        set_meta(root / "work" / ".agent-traces" / BAD, started_at=NOW - 3600, ended_at=True)
        self.assertEqual(
            self.get(root, path), (400, {"error": "session ended_at must be a Unix time in seconds"}),
        )

    def test_otlp_route_ignores_types_it_does_not_read(self):
        clean_root = self.make_ts_workspace()
        clean = self.get(clean_root, f"/api/sessions/{BAD}/otlp")
        self.assertEqual(clean[0], 200)
        root = self.make_ts_workspace(BAD, "file_write", float("nan"))
        self.assertEqual(self.get(root, f"/api/sessions/{BAD}/otlp"), clean)


class TestDriftAndFingerprintSkipTheSession(TimestampCliTestCase):
    """drift and fingerprint skip a session whose first or last timestamp is unusable."""

    ARGVS = (
        ("drift", "--since", "30d", "--format", "json"),
        ("fingerprint", "--sessions", "3", "--format", "json"),
        ("fingerprint", "--sessions", "3", "--output", "fp.json"),
    )

    @staticmethod
    def _strict_json(text: str):
        def refuse(name):
            raise ValueError(f"{name} is not valid JSON")
        return json.loads(text, parse_constant=refuse)

    def test_session_is_skipped(self):
        without_bad = self.make_ts_workspace()
        shutil.rmtree(without_bad / "work" / ".agent-traces" / BAD)
        clean_root = self.make_ts_workspace()
        for argv in self.ARGVS:
            skipped = self.outcome(without_bad, argv, files="fp.json")
            clean = self.outcome(clean_root, argv, files="fp.json")
            self.assertNotEqual(skipped, clean)
            for index, (event_type, (name, value, _)) in enumerate(
                    zip(("session_start", "session_end") * 7, TIMESTAMP_VALUES)):
                with self.subTest(argv=argv, event_type=event_type, value=name):
                    root = self.make_ts_workspace(BAD, event_type, value)
                    result = self.outcome(root, argv, files="fp.json")
                    self.assertEqual(result, skipped)
                    if "--output" in argv:
                        self._strict_json(result[4]["fp.json"].decode())
                    elif argv[0] == "fingerprint":
                        self._strict_json(result[1])

    def test_other_events_are_not_read(self):
        argv = ("fingerprint", "--sessions", "3", "--format", "json")
        clean = self.outcome(self.make_ts_workspace(), argv)
        for name, value, _ in TIMESTAMP_VALUES:
            with self.subTest(value=name):
                root = self.make_ts_workspace(BAD, "tool_call", value)
                self.assertEqual(self.outcome(root, argv), clean)


class TestBigintTimestamps(TimestampCliTestCase):
    """10**400 does not fit in a float. Each command treats it as its own rule says."""

    def compare(self, argv, value, like):
        big = self.outcome(self.make_ts_workspace(BAD, "ALL", value), argv)
        other = self.outcome(self.make_ts_workspace(BAD, "ALL", like), argv)
        self.assertEqual(big, other)
        return big

    def test_health_refuses_it_as_infinity(self):
        for argv in (("health", BAD), ("health", BAD, "--format", "json")):
            with self.subTest(argv=argv):
                self.assertEqual(self.compare(argv, 10**400, float("inf"))[0], 1)

    def test_skills_skips_it_as_infinity(self):
        for argv in (("skills", BAD), ("skills", BAD, "--format", "json")):
            with self.subTest(argv=argv):
                self.assertEqual(self.compare(argv, 10**400, float("inf"))[0], 0)

    def test_compliance_report_gives_the_message_for_its_sign(self):
        for framework in ("aicpa-tsc", "eu-ai-act"):
            argv = ("compliance-report", "--framework", framework, "--format", "json",
                    "--since", "7d")
            with self.subTest(argv=argv):
                big = self.compare(argv, 10**400, 1e300)
                self.assertEqual(
                    big[:3], (1, "", "Error: event timestamp is outside the supported UTC range\n"),
                )
                negative = self.compare(argv, -(10**400), -1e300)
                self.assertEqual(
                    negative[:3], (1, "", "Error: event timestamp must be finite and non-negative\n"),
                )


# compaction, the watch compaction checkpoint and eval. None of them multiplies
# a timestamp, so these tests store text and arrays on disk.
MODEL = "claude-sonnet-4"
COMPACTED = "c0a1b2c3d4e50001"
EVALUATED = "e0a1b2c3d4e50001"
EVALUATED_GOOD = "e0a1b2c3d4e50002"
# Its metadata holds no duration, so the duration_seconds metric reads event times.
EVALUATED_NO_DURATION = "e0a1b2c3d4e50003"
# One event and no duration in its metadata, so no duration reads its time.
EVALUATED_ONE = "e0a1b2c3d4e50004"
WATCHED = "a5a1b2c3d4e50001"

# (event_type, seconds after NOW, data)
COMPACTED_LAYOUT = (
    ("session_start", 0, {"mode": "claude-hooks", "model": MODEL}),
    ("user_prompt", 1, {"prompt": "Refactor the parser."}),
    ("file_read", 2, {"path": "src/parser.py"}),
    ("llm_request", 3, {"model": MODEL, "input_tokens": 180000}),
    ("llm_response", 4, {"model": MODEL, "input_tokens": 180000, "output_tokens": 50}),
    # The compaction: the input drops from 180000 to 12000 tokens.
    ("llm_request", 42, {"model": MODEL, "input_tokens": 12000}),
    ("llm_response", 43, {"model": MODEL, "input_tokens": 12000, "output_tokens": 40}),
    ("session_end", 50, {"reason": "completed"}),
)
EVALUATED_LAYOUT = (
    ("session_start", 0, {"mode": "claude-hooks", "model": MODEL}),
    ("user_prompt", 1, {"prompt": "Fix add()."}),
    ("llm_request", 2, {"model": MODEL, "input_tokens": 1000}),
    ("llm_response", 3, {"model": MODEL, "input_tokens": 1000, "output_tokens": 200}),
    ("tool_call", 10, {"tool_name": "Bash", "arguments": {"command": "pytest -q"}}),
    ("tool_result", 12, {"tool_name": "Bash", "result": "1 passed"}),
    ("assistant_response", 26, {"text": "Done."}),
    ("session_end", 30, {"reason": "completed"}),
)
# Saturated, so watch writes a checkpoint at startup.
WATCHED_LAYOUT = (
    ("session_start", 0, {"mode": "claude-hooks", "model": MODEL}),
    ("user_prompt", 1, {"prompt": "Refactor the parser."}),
    ("file_read", 2, {"path": "src/parser.py"}),
    ("llm_request", 3, {"model": MODEL, "input_tokens": 180000}),
    ("tool_call", 4, {"tool_name": "Read", "arguments": {"file_path": "src/lexer.py"}}),
)

EVAL_CONFIGS = {
    "duration.yaml": "scorers:\n  - type: duration_under\n    max_seconds: 120\n    threshold: 1.0\n",
    "cost.yaml": "scorers:\n  - type: cost_under\n    max_dollars: 1.0\n    threshold: 1.0\n",
    "plain.yaml": "scorers:\n  - type: no_errors\n",
    "gate_duration.yaml": (
        "evals:\n  - name: wall\n    scorer: duration_seconds\n    fail_on: above\n    threshold: 600\n"
    ),
    "gate_cost.yaml": (
        "evals:\n  - name: spend\n    scorer: cost_usd\n    fail_on: above\n    threshold: 5\n"
    ),
    "gate_plain.yaml": (
        "evals:\n  - name: calls\n    scorer: tool_call_count\n    fail_on: above\n    threshold: 50\n"
    ),
}


def _eval_argvs(config: str) -> tuple:
    return (
        ("eval", "run", EVALUATED, "--config", config),
        ("eval", "run", EVALUATED, "--config", config, "--format", "json"),
        ("eval", "compare", EVALUATED_GOOD, EVALUATED, "--config", config),
        ("eval", "ci", EVALUATED, "--config", config),
    )


def _layout_events(session_id: str, layout) -> list[dict]:
    return [
        {
            "event_type": event_type, "timestamp": NOW + offset, "event_id": f"{session_id}-{index}",
            "session_id": session_id, "data": data,
        }
        for index, (event_type, offset, data) in enumerate(layout)
    ]


def _at(*indexes: int):
    """A restamp *where* that picks the events at *indexes*."""
    return lambda record: int(record["event_id"].rsplit("-", 1)[1]) in indexes


def _layout_error(session_id: str | None, layout, index: int, kind: str) -> str:
    return f"Error: {_message(session_id, index + 1, len(layout), layout[index][0], kind, 0)}\n"


def _event(event_type: EventType, timestamp, data: dict, session_id: str = "direct") -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=timestamp, session_id=session_id, data=data)


def _request(timestamp, input_tokens: int) -> TraceEvent:
    return _event(EventType.LLM_REQUEST, timestamp, {"model": MODEL, "input_tokens": input_tokens})


class TestDetectCompactions(unittest.TestCase):
    """detect_compactions reads the first event's time and each compaction's."""

    def test_refuses_an_unusable_first_or_compaction_time(self):
        for name, value, kind in TIMESTAMP_VALUES:
            for position, events in (
                (1, [_request(value, 180000), _request(142.0, 12000)]),
                (2, [_request(100.0, 180000), _request(value, 12000)]),
            ):
                with self.subTest(value=name, position=position):
                    with self.assertRaises(event_data.UnusableTimestampError) as caught:
                        compaction.detect_compactions(events)
                    self.assertEqual(
                        str(caught.exception), _message(None, position, 2, "llm_request", kind, 0),
                    )

    def test_reads_no_time_without_a_compaction_or_before_one(self):
        start = _event(EventType.SESSION_START, 100.0, {})
        for name, value, _kind in TIMESTAMP_VALUES:
            with self.subTest(value=name):
                self.assertEqual(
                    compaction.detect_compactions([_request(value, 1000), _request(142.0, 2000)]),
                    [],
                )
                found = compaction.detect_compactions(
                    [start, _request(value, 180000), _request(142.0, 12000)],
                )
                self.assertEqual([item.offset_seconds for item in found], [42.0])


class TestCheckpointMarkdown(unittest.TestCase):
    """build_checkpoint_markdown reads the times a checkpoint shows."""

    TYPES = ("user_prompt", "file_read", "file_read", "llm_response")

    @staticmethod
    def _events(index: int | None = None, value=None) -> list[TraceEvent]:
        events = [
            _event(EventType.USER_PROMPT, 100.0, {"prompt": "Fix the parser."}),
            _event(EventType.FILE_READ, 101.0, {"path": "a.py"}),
            _event(EventType.FILE_READ, 102.0, {"path": "a.py"}),
            _event(EventType.LLM_RESPONSE, 103.0, {"text": "Done."}),
        ]
        if index is not None:
            events[index].timestamp = value
        return events

    def test_refuses_an_unusable_time_it_shows(self):
        # The first event, the first read of a.py and, with no timestamp
        # argument, the last event.
        for name, value, kind in TIMESTAMP_VALUES:
            for index in (0, 1, 3):
                with self.subTest(value=name, index=index):
                    with self.assertRaises(event_data.UnusableTimestampError) as caught:
                        compaction.build_checkpoint_markdown(self._events(index, value), "direct")
                    self.assertEqual(
                        str(caught.exception),
                        _message(None, index + 1, 4, self.TYPES[index], kind, 0),
                    )

    def test_reads_no_other_time(self):
        good = compaction.build_checkpoint_markdown(self._events(), "direct")
        for name, value, _kind in TIMESTAMP_VALUES:
            with self.subTest(value=name):
                self.assertEqual(
                    compaction.build_checkpoint_markdown(self._events(2, value), "direct"), good,
                )
                self.assertEqual(
                    compaction.build_checkpoint_markdown(self._events(3, value), "direct", 103.0),
                    good,
                )


class TestCompactionCommand(TimestampCliTestCase):
    """compaction refuses a session whose first event or compaction has an
    unusable time. lint finds the same compaction and shows no time."""

    ARGVS = (
        ("compaction", COMPACTED),
        ("compaction", COMPACTED, "--diff"),
        ("compaction", COMPACTED, "--behavior-diff"),
    )
    LINT_ARGVS = (
        ("lint", COMPACTED),
        ("lint", COMPACTED, "--format", "json"),
    )
    READ = (0, 5)

    def workspace(self, value=None, indexes=()) -> Path:
        root = self.make_ts_workspace()
        traces = root / "work" / ".agent-traces"
        _write_session(traces, COMPACTED, NOW, _layout_events(COMPACTED, COMPACTED_LAYOUT))
        if indexes:
            restamp(traces / COMPACTED, "ALL", value, where=_at(*indexes))
        return root

    def test_refuses_an_unusable_first_or_compaction_time(self):
        for name, value, kind in TIMESTAMP_VALUES:
            for index in self.READ:
                root = self.workspace(value, (index,))
                for argv in self.ARGVS:
                    with self.subTest(value=name, index=index, argv=argv):
                        self.assert_refused(
                            root, list(argv), _layout_error(None, COMPACTED_LAYOUT, index, kind),
                        )

    def test_other_times_and_lint_are_unchanged(self):
        clean = self.workspace()
        expected = {argv: self.outcome(clean, argv) for argv in self.ARGVS + self.LINT_ARGVS}
        unread = tuple(index for index in range(len(COMPACTED_LAYOUT)) if index not in self.READ)
        for name, value, _kind in TIMESTAMP_VALUES:
            others = self.workspace(value, unread)
            every = self.workspace(value, range(len(COMPACTED_LAYOUT)))
            for root, argvs in ((others, self.ARGVS), (every, self.LINT_ARGVS)):
                for argv in argvs:
                    with self.subTest(value=name, argv=argv):
                        self.assertEqual(self.outcome(root, argv), expected[argv])


class TestWatchCheckpointAtStartup(TimestampCliTestCase):
    """watch --compaction-checkpoint refuses a stored time a checkpoint shows."""

    ARGV = ("watch", WATCHED, "--dry-run", "--compaction-checkpoint")
    CHECKPOINTS = ".agent-traces/checkpoints/*.md"

    def workspace(self, value=None, index: int | None = None) -> Path:
        root = self.make_ts_workspace()
        traces = root / "work" / ".agent-traces"
        _write_session(
            traces, WATCHED, NOW, _layout_events(WATCHED, WATCHED_LAYOUT), ended_at=None,
        )
        if index is not None:
            restamp(traces / WATCHED, "ALL", value, where=_at(index))
        return root

    def test_refuses_a_stored_time_a_checkpoint_shows(self):
        # The first event, the first read of each file and the saturated request.
        for name, value, kind in TIMESTAMP_VALUES:
            for index in (0, 2, 3, 4):
                with self.subTest(value=name, index=index):
                    root = self.workspace(value, index)
                    # A regression would start tailing; an empty tail ends the watch.
                    with mock.patch("agent_trace.watch._tail_events", return_value=iter([])):
                        self.assert_refused(
                            root, list(self.ARGV), _layout_error(None, WATCHED_LAYOUT, index, kind),
                        )

    def test_checks_first_and_read_times_below_the_checkpoint_ratio(self):
        # The request fills 0.9 of its window, so nothing is saturated at
        # startup. Every checkpoint written later shows the first event and
        # the first reads, so watch still checks them. It reads no request time.
        argv = [*self.ARGV, "--checkpoint-at", "0.95"]
        with mock.patch("agent_trace.watch._tail_events", return_value=iter([])):
            expected = self.outcome(self.workspace(), argv, files=self.CHECKPOINTS)
        self.assertEqual(expected[0], 0)
        self.assertEqual(list(expected[4]), [])
        for name, value, kind in TIMESTAMP_VALUES:
            for index in (0, 2, 4):
                with self.subTest(value=name, index=index):
                    root = self.workspace(value, index)
                    with mock.patch("agent_trace.watch._tail_events", return_value=iter([])):
                        self.assert_refused(
                            root, argv, _layout_error(None, WATCHED_LAYOUT, index, kind),
                        )
            for index in (1, 3):
                with self.subTest(value=name, index=index):
                    root = self.workspace(value, index)
                    with mock.patch("agent_trace.watch._tail_events", return_value=iter([])):
                        self.assertEqual(self.outcome(root, argv, files=self.CHECKPOINTS), expected)

    def test_other_stored_times_are_unchanged(self):
        with mock.patch("agent_trace.watch._tail_events", return_value=iter([])):
            expected = self.outcome(self.workspace(), self.ARGV, files=self.CHECKPOINTS)
        self.assertEqual(list(expected[4]), [f"{WATCHED}.md"])
        for name, value, _kind in TIMESTAMP_VALUES:
            with self.subTest(value=name):
                root = self.workspace(value, 1)
                with mock.patch("agent_trace.watch._tail_events", return_value=iter([])):
                    self.assertEqual(self.outcome(root, self.ARGV, files=self.CHECKPOINTS), expected)


class TestWatchCheckpointLeavesOutLiveEvents(unittest.TestCase):
    """watch keeps running and leaves a live event out of the compaction
    checkpoint when the checkpoint would show its unusable time."""

    LIVE_VALUES = SAFE_VALUES + NO_MUL_VALUES

    def _watch(self, live: list[TraceEvent], stored: int = 4) -> tuple[str, str, str]:
        """Return the watch output, its alerts and the checkpoint it leaves.

        The session holds the first *stored* events of WATCHED_LAYOUT when
        watch starts, and *live* arrives after that.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = TraceStore(tmp.name, redact=False)
        store.create_session(SessionMeta(session_id=WATCHED, started_at=NOW))
        for event_type, offset, data in WATCHED_LAYOUT[:stored]:
            store.append_event(
                WATCHED, _event(EventType(event_type), NOW + offset, data, WATCHED),
            )
        watcher = compaction.CompactionCheckpointWatcher(store, WATCHED)
        out = io.StringIO()
        alerts = io.StringIO()
        alert = watch._alert_terminal
        with mock.patch("agent_trace.watch._tail_events", return_value=iter(live)), mock.patch(
            "agent_trace.watch._alert_terminal",
            side_effect=lambda message, *_args, **kwargs: alert(message, alerts, **kwargs),
        ):
            watch.watch_session(
                store, WATCHED, _watcher_config(), out=out, checkpoint_watcher=watcher,
            )
        checkpoint = store.checkpoint_path(WATCHED).read_text(encoding="utf-8")
        output = out.getvalue()
        for prefix in (str(Path(tmp.name).resolve()), tmp.name):
            output = output.replace(prefix, "ROOT")
        return output, alerts.getvalue(), checkpoint

    @staticmethod
    def _live(index: int | None = None, value=None) -> list[TraceEvent]:
        events = [
            _event(EventType.FILE_READ, NOW + 10, {"path": "src/lexer.py"}, WATCHED),
            # The input drops from 180000 to 12000 tokens, which arms the watcher.
            _event(EventType.LLM_REQUEST, NOW + 11, {"model": MODEL, "input_tokens": 12000}, WATCHED),
            # Saturated again, so it writes a checkpoint.
            _event(EventType.LLM_REQUEST, NOW + 12, {"model": MODEL, "input_tokens": 185000}, WATCHED),
            _event(EventType.SESSION_END, NOW + 13, {"reason": "completed"}, WATCHED),
        ]
        if index is not None:
            events[index].timestamp = value
        return events

    @staticmethod
    def _whole_session(index: int | None = None, value=None) -> list[TraceEvent]:
        """Every event of an empty session, tailed live."""
        events = [
            _event(EventType(event_type), NOW + offset, data, WATCHED)
            for event_type, offset, data in WATCHED_LAYOUT[:4]
        ]
        events.append(_event(EventType.SESSION_END, NOW + 13, {"reason": "completed"}, WATCHED))
        if index is not None:
            events[index].timestamp = value
        return events

    @staticmethod
    def _arm_and_write(index: int | None = None, value=None) -> list[TraceEvent]:
        """A request that arms the watcher and writes a checkpoint, since its
        input drops from 180000 tokens to 12000 of a 14000 token window. A
        full request follows, so TokenBudgetWatcher alerts."""
        events = [
            _event(
                EventType.LLM_REQUEST, NOW + 11,
                {"model": MODEL, "input_tokens": 12000, "context_window": 14000}, WATCHED,
            ),
            _event(EventType.LLM_REQUEST, NOW + 12, {"model": MODEL, "input_tokens": 185000}, WATCHED),
            _event(EventType.SESSION_END, NOW + 13, {"reason": "completed"}, WATCHED),
        ]
        if index is not None:
            events[index].timestamp = value
        return events

    @classmethod
    def _unread_live(cls, index: int | None = None, value=None) -> list[TraceEvent]:
        """_live with three events no checkpoint reads the time of, after the
        request that arms the watcher: a second read of a file, a tool result
        with a token count, and a response with none."""
        events = cls._live()
        events[2:2] = [
            _event(EventType.FILE_READ, NOW + 11, {"path": "src/parser.py"}, WATCHED),
            _event(
                EventType.TOOL_RESULT, NOW + 11,
                {"tool_name": "Task", "result": "Done.", "usage": {"input_tokens": 190000}},
                WATCHED,
            ),
            _event(EventType.LLM_RESPONSE, NOW + 11, {"model": MODEL, "text": "Done."}, WATCHED),
        ]
        if index is not None:
            events[index].timestamp = value
        return events

    def test_leaves_out_a_new_event_whose_time_a_checkpoint_shows(self):
        # The first event, a new read of a file, and a request that writes a
        # checkpoint.
        for event_type, index, stored, make in (
            ("session_start", 0, 0, self._whole_session),
            ("file_read", 0, 4, self._live),
            ("llm_request", 2, 4, self._live),
            ("llm_request", 0, 4, self._arm_and_write),
        ):
            alerts = self._watch(make(), stored)[1]
            self.assertIn("TokenBudgetWatcher", alerts)
            kept = [event for position, event in enumerate(make()) if position != index]
            expected = self._watch(kept, stored)[2]
            for name, value, _kind in self.LIVE_VALUES:
                with self.subTest(event_type=event_type, value=name):
                    live = make(index, value)
                    output, bad_alerts, checkpoint = self._watch(live, stored)
                    self.assertIn(
                        f"[watch] Left one {event_type} event out of the compaction checkpoint "
                        "because its timestamp is not a Unix time in seconds.\n",
                        output,
                    )
                    self.assertIn(f"[watch] Session ended ({len(live)} events,", output)
                    # The other watchers still see the event.
                    self.assertEqual(bad_alerts, alerts)
                    self.assertEqual(checkpoint, expected)

    def test_keeps_a_new_event_whose_time_no_checkpoint_shows(self):
        expected = self._watch(self._unread_live())
        self.assertIn("Compaction checkpoint written", expected[0])
        # The request that arms the watcher, then the three events _unread_live adds.
        for index in (1, 2, 3, 4):
            for name, value, _kind in self.LIVE_VALUES:
                with self.subTest(index=index, value=name):
                    self.assertEqual(self._watch(self._unread_live(index, value)), expected)


class TestEvalCommands(TimestampCliTestCase):
    """eval refuses a session when a scorer or metric it runs reads an unusable
    time, and names the session."""

    def workspace(self, value=None, indexes=()) -> Path:
        root = self.make_ts_workspace()
        work = root / "work"
        traces = work / ".agent-traces"
        no_duration = {"ended_at": None, "total_duration_ms": 0}
        for session_id, layout, extra in (
            (EVALUATED, EVALUATED_LAYOUT, {}),
            (EVALUATED_GOOD, EVALUATED_LAYOUT, {}),
            (EVALUATED_NO_DURATION, EVALUATED_LAYOUT, no_duration),
            (EVALUATED_ONE, EVALUATED_LAYOUT[:1], no_duration),
        ):
            _write_session(traces, session_id, NOW, _layout_events(session_id, layout), **extra)
        if indexes:
            for session_id in (EVALUATED, EVALUATED_NO_DURATION, EVALUATED_ONE):
                restamp(traces / session_id, "ALL", value, where=_at(*indexes))
        for name, text in EVAL_CONFIGS.items():
            (work / name).write_text(text, encoding="utf-8")
        return root

    def test_refuses_a_time_the_scorer_or_metric_reads(self):
        # cost_under and cost_usd read every event's time. duration_under reads
        # the first and the last. duration_seconds reads every one when the
        # metadata holds no duration.
        cost = _eval_argvs("cost.yaml") + (("eval", "gate", EVALUATED, "--config", "gate_cost.yaml"),)
        gate_duration = ("eval", "gate", EVALUATED_NO_DURATION, "--config", "gate_duration.yaml")
        for name, value, kind in TIMESTAMP_VALUES:
            for index in (0, 4, 7):
                root = self.workspace(value, (index,))
                rows = [(argv, EVALUATED) for argv in cost]
                rows.append((gate_duration, EVALUATED_NO_DURATION))
                if index != 4:
                    rows.extend((argv, EVALUATED) for argv in _eval_argvs("duration.yaml"))
                for argv, session_id in rows:
                    with self.subTest(value=name, index=index, argv=argv):
                        self.assert_refused(
                            root, list(argv), _layout_error(session_id, EVALUATED_LAYOUT, index, kind),
                        )

    def test_times_no_scorer_or_metric_reads_are_unchanged(self):
        duration = _eval_argvs("duration.yaml")
        plain = _eval_argvs("plain.yaml") + (
            ("eval", "gate", EVALUATED, "--config", "gate_plain.yaml"),
            # The metadata holds the duration.
            ("eval", "gate", EVALUATED, "--config", "gate_duration.yaml"),
            # The session has one event.
            ("eval", "run", EVALUATED_ONE, "--config", "duration.yaml"),
            ("eval", "gate", EVALUATED_ONE, "--config", "gate_duration.yaml"),
        )
        clean = self.workspace()
        expected = {argv: self.outcome(clean, argv) for argv in duration + plain}
        for name, value, _kind in TIMESTAMP_VALUES:
            middle = self.workspace(value, range(1, len(EVALUATED_LAYOUT) - 1))
            every = self.workspace(value, range(len(EVALUATED_LAYOUT)))
            for root, argvs in ((middle, duration), (every, plain)):
                for argv in argvs:
                    with self.subTest(value=name, argv=argv):
                        self.assertEqual(self.outcome(root, argv), expected[argv])


if __name__ == "__main__":
    unittest.main()
