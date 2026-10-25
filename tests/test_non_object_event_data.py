"""Commands on sessions whose stored event data is not a JSON object.

``agent-strace server`` stores array, string, number and boolean ``data`` as it
arrives, so these sessions exist without hand edits. A command that would read
such data as an object refuses up front, as ``share`` does: one ``Error:`` line
on stderr, exit code 1, and nothing printed, written, sent or opened. Commands
that never read the data keep their output and exit code.

The tests run ``main()`` in-process against a temporary store, so they also
check the telemetry record and that no exception escapes.
"""

from __future__ import annotations

import contextlib
import csv
import errno
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import warnings
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import agent_trace
from agent_trace.cli import main
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.watch import StreamConfig, WatcherConfig, watch_session

GOOD = "a1b2c3d4e5f60001"
BAD = "b2c3d4e5f6a70002"
CHILD = "c3d4e5f6a7b80003"

# Every JSON type that is not an object, with the value stored for it.
NON_OBJECT_VALUES = [
    ("array", ["unexpected", "list"]),
    ("string", "plain text data"),
    ("number", 42),
    ("boolean", True),
    ("null", None),
]
_WARNING_NOUN = {
    "array": "an array",
    "string": "a string",
    "number": "a number",
    "boolean": "a boolean",
    "null": "null",
}

NOW = float(int(time.time()) // 60 * 60)

# 1-based position of the first event of each type, and how many events of
# that type the session has. test_fixture_layout checks both tables.
BAD_LAYOUT = {
    "session_start": (1, 1),
    "user_prompt": (2, 1),
    "llm_request": (3, 2),
    "llm_response": (4, 2),
    "file_read": (5, 1),
    "tool_call": (6, 6),
    "tool_result": (7, 5),
    "error": (9, 1),
    "decision": (10, 1),
    "file_write": (12, 1),
    "subagent_start": (17, 1),
    "subagent_stop": (18, 1),
    "assistant_response": (24, 1),
    "session_end": (25, 1),
}
CHILD_LAYOUT = {
    "session_start": (1, 1),
    "user_prompt": (2, 1),
    "tool_call": (3, 1),
    "tool_result": (4, 1),
    "error": (5, 1),
    "assistant_response": (6, 1),
    "session_end": (7, 1),
}

POLICY = "../aux/policy.yaml"
AGENTS_MD = "../aux/AGENTS.md"
_CLOSED_PORT = "http://127.0.0.1:9"
_ENV_PREFIXES = ("AGENT_STRACE", "AGENT_TRACE", "OTEL_", "LANGFUSE", "GITHUB_", "CI")


# ---------------------------------------------------------------------------
# Fixture: a clean session (GOOD), a session with a subagent (BAD) and the
# subagent's own session (CHILD). Lines are written raw so any JSON value can
# be stored as data, and prev_hash keeps the chain valid for `verify`.
# ---------------------------------------------------------------------------

def _session_events(session_id: str, start: float, prefix: str, child_id: str = "") -> list[dict]:
    def event(event_type, data, offset, event_id, parent="", duration=None):
        record = {
            "event_type": event_type,
            "timestamp": start + offset,
            "event_id": f"{prefix}-{event_id}",
            "session_id": session_id,
            "data": data,
        }
        if parent:
            record["parent_id"] = f"{prefix}-{parent}"
        if duration is not None:
            record["duration_ms"] = duration
        return record

    events = [
        event("session_start", {"mode": "claude-hooks", "provider": "claude", "source": "startup",
                                "model": "claude-sonnet-4", "cwd": "/work/app"}, 0, "start"),
        event("user_prompt", {"prompt": "Fix the failing test in tests/test_app.py"}, 1, "prompt"),
        event("llm_request", {"method": "sampling/createMessage", "model": "claude-sonnet-4",
                              "message_count": 3, "input_tokens": 1200}, 2, "llmreq"),
        event("llm_response", {"model": "claude-sonnet-4", "content_preview": "I will read the file",
                               "input_tokens": 1200, "output_tokens": 300,
                               "usage": {"input_tokens": 1200, "output_tokens": 300}},
              4, "llmresp", parent="llmreq", duration=2000.0),
        event("file_read", {"path": "src/app.py"}, 5, "fread"),
        event("tool_call", {"tool_name": "Read", "arguments": {"file_path": "src/app.py"},
                            "tool_use_id": "tu1"}, 6, "call-read"),
        event("tool_result", {"tool_name": "Read", "result": "def add(a, b):\n    return a - b\n",
                              "tool_use_id": "tu1"}, 6.5, "res-read", parent="call-read", duration=12.0),
        event("tool_call", {"tool_name": "Bash", "arguments": {"command": "pytest -q tests/test_app.py"},
                            "tool_use_id": "tu2"}, 8, "call-test"),
        event("error", {"tool_name": "Bash", "error": "1 failed: test_add expected 5 got -1"},
              9, "err-test", parent="call-test", duration=900.0),
        event("decision", {"decision": "fix the subtraction bug", "reasoning": "test expects addition",
                           "source": "agent"}, 10, "decide"),
        event("tool_call", {"tool_name": "Edit", "arguments": {"file_path": "src/app.py",
                            "old_string": "a - b", "new_string": "a + b"}, "tool_use_id": "tu3"},
              11, "call-edit"),
        event("file_write", {"path": "src/app.py", "diff": "-    return a - b\n+    return a + b\n"},
              11.5, "fwrite"),
        event("tool_result", {"tool_name": "Edit", "result": "ok", "tool_use_id": "tu3"},
              12, "res-edit", parent="call-edit", duration=20.0),
        event("tool_call", {"tool_name": "mcp__github__get_issue",
                            "arguments": {"owner": "acme", "repo": "app", "issue_number": 7},
                            "tool_use_id": "tu4"}, 13, "call-mcp"),
        event("tool_result", {"tool_name": "mcp__github__get_issue", "result": "Issue 7: add is wrong",
                              "tool_use_id": "tu4"}, 13.4, "res-mcp", parent="call-mcp", duration=300.0),
    ]
    if child_id:
        events += [
            event("tool_call", {"tool_name": "task", "arguments": {"description": "verify",
                                "prompt": "run the tests"}, "tool_use_id": "tu5"}, 14, "call-task"),
            event("subagent_start", {"hook_event": "subagent_start", "provider": "claude",
                                     "child_session_id": child_id, "agent_type": "general-purpose",
                                     "child_link": "linked"}, 14.2, "sub-start", parent="call-task"),
            event("subagent_stop", {"hook_event": "subagent_stop", "provider": "claude",
                                    "child_session_id": child_id, "agent_type": "general-purpose",
                                    "stop_reason": "completed", "child_link": "linked"},
                  20, "sub-stop", parent="call-task"),
            event("tool_result", {"tool_name": "task", "result": "3 passed", "tool_use_id": "tu5"},
                  20.5, "res-task", parent="call-task", duration=6500.0),
        ]
    events += [
        event("tool_call", {"tool_name": "Bash", "arguments": {"command": "pytest -q"},
                            "tool_use_id": "tu6"}, 21, "call-test2"),
        event("tool_result", {"tool_name": "Bash", "result": "3 passed in 0.12s", "exit_code": 0,
                              "tool_use_id": "tu6"}, 22, "res-test2", parent="call-test2", duration=1000.0),
        event("llm_request", {"method": "sampling/createMessage", "model": "claude-sonnet-4",
                              "message_count": 9, "input_tokens": 2400}, 23, "llmreq2"),
        event("llm_response", {"model": "claude-sonnet-4", "content_preview": "Done",
                               "input_tokens": 2400, "output_tokens": 120,
                               "usage": {"input_tokens": 2400, "output_tokens": 120}},
              24, "llmresp2", parent="llmreq2", duration=1000.0),
        event("assistant_response", {"text": "Fixed add() and the tests pass.", "hook_event": "stop",
                                     "provider": "claude"}, 25, "answer"),
        event("session_end", {"duration_ms": 26000.0, "reason": "completed"}, 26, "end"),
    ]
    return events


def _child_events(start: float) -> list[dict]:
    def event(event_type, data, offset, event_id, **extra):
        return {"event_type": event_type, "timestamp": start + offset,
                "event_id": f"chld-{event_id}", "session_id": CHILD, "data": data, **extra}

    return [
        event("session_start", {"mode": "claude-hooks", "provider": "claude", "source": "startup",
                                "model": "claude-sonnet-4"}, 0, "start"),
        event("user_prompt", {"prompt": "run the tests"}, 0.5, "prompt"),
        event("tool_call", {"tool_name": "Bash", "arguments": {"command": "pytest -q"},
                            "tool_use_id": "c1"}, 1, "call"),
        event("tool_result", {"tool_name": "Bash", "result": "3 passed", "tool_use_id": "c1"}, 2, "res",
              parent_id="chld-call", duration_ms=1000.0),
        event("error", {"tool_name": "Bash", "error": "flaky warning"}, 2.5, "err"),
        event("assistant_response", {"text": "All tests pass.", "hook_event": "stop"}, 3, "answer"),
        event("session_end", {"duration_ms": 4000.0, "reason": "completed"}, 4, "end"),
    ]


def _poison(events: list[dict], event_type: str | None, value) -> list[dict]:
    if event_type is None:
        return events
    return [
        {**event, "data": value} if event_type in ("ALL", event["event_type"]) else event
        for event in events
    ]


def _write_session(traces: Path, session_id: str, started: float, events: list[dict], **extra) -> None:
    session_dir = traces / session_id
    session_dir.mkdir(parents=True)
    meta = {
        "session_id": session_id,
        "started_at": started,
        "ended_at": events[-1]["timestamp"],
        "agent_name": "claude",
        "command": "claude",
        "tool_calls": sum(1 for e in events if e["event_type"] == "tool_call"),
        "llm_requests": sum(1 for e in events if e["event_type"] == "llm_request"),
        "errors": sum(1 for e in events if e["event_type"] == "error"),
        "total_tokens": 4020,
        "total_duration_ms": (events[-1]["timestamp"] - started) * 1000,
        "attribution": {"author": "alice", "branch": "fix/add", "repo": "acme/app", "pr": "12"},
        **extra,
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines: list[str] = []
    for event in events:
        record = dict(event)
        if lines:
            record["prev_hash"] = hashlib.sha256(lines[-1].encode()).hexdigest()
        lines.append(json.dumps(record, separators=(",", ":")))
    (session_dir / "events.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_store(
    traces: Path,
    *,
    bad_type: str | None = None,
    child_type: str | None = None,
    value=None,
    tenant: bool = False,
) -> None:
    """Write GOOD, BAD and CHILD, storing *value* as the data of the chosen event types."""
    good_start = NOW - 2 * 86400
    bad_start = NOW - 3600
    child_start = bad_start + 14.5
    tenant_meta = {"tenant_id": "acme"} if tenant else {}
    _write_session(traces, GOOD, good_start, _session_events(GOOD, good_start, "good"), **tenant_meta)
    _write_session(
        traces, BAD, bad_start,
        _poison(_session_events(BAD, bad_start, "bad", child_id=CHILD), bad_type, value),
        **tenant_meta,
    )
    _write_session(
        traces, CHILD, child_start, _poison(_child_events(child_start), child_type, value),
        parent_session_id=BAD, parent_event_id="bad-call-task", depth=1, **tenant_meta,
    )
    (traces / ".gitignore").write_text("*\n", encoding="utf-8")


def expected_error(session_id: str, event_type: str, json_type: str, *, named: bool) -> str:
    layout, total = (BAD_LAYOUT, 25) if session_id == BAD else (CHILD_LAYOUT, 7)
    position, count = layout[event_type]
    subject = f"session {session_id}" if named else "session"
    more = count - 1
    if more == 0:
        suffix = ""
    elif more == 1:
        suffix = "; 1 more event has non-object data"
    else:
        suffix = f"; {more} more events have non-object data"
    return (
        f"Error: {subject} event data must be an object: "
        f"event {position} of {total} ({event_type}) has {json_type} data{suffix}\n"
    )


def _tree_snapshot(root: Path) -> dict[str, tuple[int, bytes] | None]:
    """Map every path under *root* to its file mtime and bytes (None for directories)."""
    return {
        str(path.relative_to(root)): (
            (path.stat().st_mtime_ns, path.read_bytes()) if path.is_file() else None
        )
        for path in root.rglob("*")
        if "__pycache__" not in path.parts
    }


@contextlib.contextmanager
def _chdir(path: Path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _Captured:
    out = ""
    err = ""


@contextlib.contextmanager
def _capture_output():
    """Capture stdout and stderr, including file descriptors 1 and 2.

    Many formatters take ``out: TextIO = sys.stdout``, which binds the real
    stream at import time, so redirecting ``sys.stdout`` alone misses them.
    Garbage collection can report objects that earlier tests leaked while a
    command runs. Those resource warnings and unraisable exceptions go to the
    real stderr instead of the captured output.
    """
    captured = _Captured()
    streams = [s for s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__) if s]

    def flush_all():
        for stream in streams:
            with contextlib.suppress(Exception):
                stream.flush()

    flush_all()
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    files = {fd: tempfile.TemporaryFile() for fd in (1, 2)}
    out_buffer, err_buffer = io.StringIO(), io.StringIO()
    show_other_warning = warnings.showwarning

    def to_real_stderr(text: str) -> None:
        os.write(saved[2], text.encode("utf-8", "replace"))

    def show_warning(message, category, filename, lineno, file=None, line=None):
        if issubclass(category, ResourceWarning):
            to_real_stderr(warnings.formatwarning(message, category, filename, lineno, line))
        else:
            show_other_warning(message, category, filename, lineno, file, line)

    def report_unraisable(info) -> None:
        header = info.err_msg or "Exception ignored in"
        if info.object is not None:
            header = f"{header}: {info.object!r}"
        to_real_stderr(
            header + "\n"
            + "".join(traceback.format_exception(info.exc_type, info.exc_value, info.exc_traceback))
        )

    try:
        for fd, handle in files.items():
            os.dup2(handle.fileno(), fd)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(warnings, "showwarning", show_warning))
            stack.enter_context(mock.patch.object(sys, "unraisablehook", report_unraisable))
            stack.enter_context(redirect_stdout(out_buffer))
            stack.enter_context(redirect_stderr(err_buffer))
            yield captured
    finally:
        flush_all()
        raw = {}
        for fd, handle in files.items():
            os.dup2(saved[fd], fd)
            os.close(saved[fd])
            handle.seek(0)
            raw[fd] = handle.read().decode("utf-8", "replace")
            handle.close()
        captured.out = out_buffer.getvalue() + raw[1]
        captured.err = err_buffer.getvalue() + raw[2]


@contextlib.contextmanager
def _isolated_env(home: Path):
    with mock.patch.dict(os.environ):
        for key in list(os.environ):
            if key.startswith(_ENV_PREFIXES):
                del os.environ[key]
        os.environ.update({"HOME": str(home), "DO_NOT_TRACK": "1", "AGENT_STRACE_TELEMETRY": "0"})
        yield


@dataclass
class CliResult:
    code: int
    out: str
    err: str
    telemetry: dict | None
    network: list
    browser: list


class CliTestCase(unittest.TestCase):
    """Build the fixture in a temporary cwd and run ``agent-strace`` in-process."""

    def make_workspace(self, **store_options) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        (root / "home").mkdir()
        (root / "work").mkdir()
        aux = root / "aux"
        aux.mkdir()
        (aux / "policy.yaml").write_text(json.dumps({
            "files": {"read": {"allow": ["src/app.py"]}, "write": {"allow": ["src/app.py"]}},
            "commands": {"allow": ["pytest *"]},
        }), encoding="utf-8")
        (aux / "AGENTS.md").write_text(
            "# Agents\n\nRun `pytest -q` before committing.\nEdit files under src/ only.\n",
            encoding="utf-8",
        )
        build_store(root / "work" / ".agent-traces", **store_options)
        return root

    def run_cli(self, root: Path, *argv: str) -> CliResult:
        network: list = []
        browser: list = []

        def refuse_connect(*args, **kwargs):
            network.append(args[1:] if args and isinstance(args[0], socket.socket) else args)
            raise ConnectionRefusedError("network is disabled in this test")

        def refuse_connect_ex(sock, address, *args, **kwargs):
            network.append(address)
            return errno.ECONNREFUSED

        def record_browser(url, *args, **kwargs):
            browser.append(url)
            return True

        with contextlib.ExitStack() as stack:
            stack.enter_context(_chdir(root / "work"))
            stack.enter_context(_isolated_env(root / "home"))
            stack.enter_context(mock.patch.object(socket.socket, "connect", refuse_connect))
            stack.enter_context(mock.patch.object(socket.socket, "connect_ex", refuse_connect_ex))
            stack.enter_context(mock.patch("socket.create_connection", refuse_connect))
            stack.enter_context(mock.patch("urllib.request.urlopen", refuse_connect))
            for name in ("open", "open_new", "open_new_tab"):
                stack.enter_context(mock.patch.object(webbrowser, name, record_browser))
            capture = stack.enter_context(mock.patch("agent_trace.cli._product_telemetry.capture"))
            stack.enter_context(
                mock.patch("agent_trace.cli._product_telemetry.maybe_show_telemetry_notice")
            )
            stack.enter_context(mock.patch.object(sys, "argv", ["agent-strace", *argv]))
            captured = stack.enter_context(_capture_output())
            try:
                main()
                code = 0
            except SystemExit as exc:
                code = 0 if exc.code is None else exc.code
        telemetry = capture.call_args.args[1] if capture.called else None
        return CliResult(code, captured.out, captured.err, telemetry, network, browser)

    def assert_refused(self, root: Path, argv: list[str], expected: str) -> None:
        before = _tree_snapshot(root)
        result = self.run_cli(root, *argv)
        self.assertEqual(result.err, expected)
        self.assertEqual(result.out, "")
        self.assertEqual(result.code, 1)
        self.assertIsNotNone(result.telemetry)
        self.assertEqual(
            (result.telemetry["success"], result.telemetry["exit_code"]), (False, 1),
        )
        # The wrapper records error_type only when an exception escapes, as
        # for a session refused by share.
        self.assertNotIn("error_type", result.telemetry)
        self.assertEqual(result.network, [])
        self.assertEqual(result.browser, [])
        self.assertEqual(_tree_snapshot(root), before)


@dataclass(frozen=True)
class Row:
    argv: tuple[str, ...]
    event_type: str
    named: bool = False
    tenant: bool = False


def _row(*argv: str, event_type: str, named: bool = False, tenant: bool = False) -> Row:
    return Row(tuple(argv), event_type, named, tenant)


# Each row stores non-object data on one event type of BAD. The command reads
# that type, so without the check it raised a Python exception. The subject
# names the session when the command reads more than one session.
BAD_SESSION_REFUSALS = [
    _row("a2a-tree", BAD, event_type="subagent_stop", named=True),
    _row("a2a-tree", BAD, "--format", "json", event_type="decision", named=True),
    _row("audit", BAD, event_type="tool_call"),
    _row("audit", BAD, "--verify-chain", event_type="tool_call"),
    _row("audit", BAD, "--policy", POLICY, event_type="tool_call"),
    _row("budget-report", event_type="tool_call", named=True),
    _row("budget-report", "--format", "json", event_type="tool_call", named=True),
    _row("budget-report", "--format", "markdown", event_type="tool_call", named=True),
    _row("cognitive-debt", event_type="file_write", named=True),
    _row("cognitive-debt", "--format", "json", event_type="tool_call", named=True),
    _row("cognitive-debt", "--session", BAD, event_type="file_write"),
    _row("compaction", BAD, event_type="error"),
    _row("compaction", BAD, "--diff", event_type="session_end"),
    _row("compaction", BAD, "--behavior-diff", event_type="file_read"),
    _row("compare", GOOD, BAD, event_type="decision", named=True),
    _row("compare", GOOD, BAD, "--format", "json", event_type="decision", named=True),
    _row("compare", "--tag", "claude", "--last", "2", event_type="decision", named=True),
    _row("compare", BAD, "--rerun", event_type="user_prompt"),
    _row("compliance", "export", BAD, event_type="error"),
    _row("compliance", "export", BAD, "--framework", "all", "--output", "cx.json",
         event_type="tool_call"),
    _row("compliance", "export", "--since", "7d", event_type="error", named=True),
    _row("context-score", "--file", AGENTS_MD, "--min-sessions", "1",
         event_type="tool_call", named=True),
    _row("cost", BAD, event_type="user_prompt"),
    _row("cost", BAD, "--csv", event_type="tool_call"),
    _row("cost", BAD, "--breakdown", "provider", event_type="llm_request", named=True),
    _row("cost", "--tenant", "acme", event_type="tool_call", named=True, tenant=True),
    _row("cost", BAD, "--tenant", "acme", event_type="user_prompt", tenant=True),
    _row("cost", "--tenant", "acme", "--breakdown", "provider",
         event_type="llm_response", named=True, tenant=True),
    _row("dashboard", "--trend", event_type="tool_call", named=True),
    _row("diff", GOOD, BAD, event_type="tool_call", named=True),
    _row("diff", BAD, GOOD, event_type="user_prompt", named=True),
    _row("diff", GOOD, BAD, "--semantic", event_type="user_prompt", named=True),
    _row("diff", GOOD, BAD, "--compare", event_type="tool_call", named=True),
    _row("explain", BAD, event_type="tool_call"),
    _row("export", BAD, "--format", "eu-ai-act", event_type="subagent_start"),
    _row("export", BAD, "--format", "eu-ai-act", "--output", "eu.json", event_type="session_start"),
    _row("export", BAD, "--backend", "langfuse", "--langfuse-public-key", "pk",
         "--langfuse-secret-key", "sk", "--langfuse-host", _CLOSED_PORT, event_type="tool_result"),
    _row("export", BAD, "--format", "otlp", event_type="assistant_response"),
    _row("export", BAD, "--format", "otlp", "--endpoint", f"{_CLOSED_PORT}/v1/traces",
         event_type="decision"),
    _row("export", BAD, "--format", "otlp-genai", event_type="llm_response"),
    _row("export", BAD, "--backend", "otlp", "--otlp-endpoint", f"{_CLOSED_PORT}/v1/metrics",
         event_type="file_write"),
    _row("export", BAD, "--metrics", "--scores", "--backend", "otlp",
         "--otlp-endpoint", f"{_CLOSED_PORT}/v1/metrics", event_type="tool_call"),
    _row("inflation", event_type="assistant_response", named=True),
    _row("inflation", "--sessions", "2", event_type="user_prompt", named=True),
    _row("mcp-scan", "--session", BAD, event_type="session_end", named=True),
    _row("mcp-scan", "--session", BAD, "--format", "json", event_type="file_read", named=True),
    _row("oncall", "--rotation-start", "2026-01-01", event_type="file_write", named=True),
    _row("optimize", BAD, "--dry-run", event_type="file_write"),
    _row("policy", "generate", "--output", "gen.yaml", event_type="tool_call", named=True),
    _row("policy", "generate", BAD, "--dry-run", event_type="tool_call", named=True),
    _row("postmortem", BAD, event_type="error"),
    _row("postmortem", "--list", event_type="session_end", named=True),
    _row("pr-comment", BAD, "--dry-run", event_type="subagent_stop", named=True),
    _row("replay", BAD, event_type="session_start"),
    _row("replay", BAD, "--format", "html", "--output", "out.html", event_type="tool_result"),
    _row("replay", BAD, "--format", "html", event_type="decision"),
    _row("replay", BAD, "--diff", GOOD, event_type="file_read", named=True),
    _row("replay", GOOD, "--diff", BAD, event_type="session_start", named=True),
    _row("replay", BAD, "--expand-subagents", event_type="subagent_start", named=True),
    _row("replay", BAD, "--filter", "tool_call,error", event_type="error"),
    _row("replay", BAD, "--limit", "5", event_type="llm_response"),
    _row("sample", "--strategy", "worst", "--n", "2", event_type="llm_response", named=True),
    _row("sample", "--strategy", "diverse", "--n", "2", event_type="file_write", named=True),
    _row("sample", "--strategy", "random", "--n", "2", "--seed", "1",
         event_type="tool_call", named=True),
    _row("sample", "--strategy", "recent", "--n", "2", event_type="tool_call", named=True),
    _row("sample", "--strategy", "worst", "--n", "2", "--output", "sample.jsonl",
         event_type="tool_call", named=True),
    _row("standup", BAD, "--no-llm", event_type="tool_result"),
    _row("stats", BAD, event_type="error"),
    _row("team-report", event_type="file_write", named=True),
    _row("team-report", "--export", "csv", event_type="tool_call", named=True),
    _row("team-report", "--export", "json", "--by", "branch", event_type="tool_call", named=True),
    _row("tenant", "report", event_type="user_prompt", named=True),
    _row("tenant", "report", event_type="tool_call", named=True, tenant=True),
    _row("tenant", "report", "--format", "json", event_type="user_prompt", named=True, tenant=True),
    _row("timeline", BAD, event_type="decision"),
    _row("timeline", BAD, "--format", "json", event_type="file_read"),
    _row("token-budget", BAD, event_type="llm_request"),
    _row("why", BAD, "6", event_type="tool_call"),
    _row("why", BAD, "10", event_type="decision"),
    _row("why", BAD, "25", event_type="user_prompt"),
]

# Each row stores non-object data on one event type of CHILD, the subagent
# session. These commands read CHILD because it is the newest session, it is
# in the aggregation window, or it is expanded under BAD.
CHILD_SESSION_REFUSALS = [
    _row("a2a-tree", BAD, event_type="session_start", named=True),
    _row("budget-report", event_type="tool_call", named=True),
    _row("cognitive-debt", event_type="tool_call", named=True),
    _row("compliance", "export", "--since", "7d", event_type="tool_call", named=True),
    _row("context-score", "--file", AGENTS_MD, "--min-sessions", "1",
         event_type="tool_call", named=True),
    _row("cost", BAD, "--breakdown", "provider", event_type="assistant_response", named=True),
    _row("cost", "--since", "7d", event_type="user_prompt"),
    _row("cost", "--tenant", "acme", event_type="user_prompt", named=True, tenant=True),
    _row("cost", "--tenant", "acme", "--breakdown", "provider",
         event_type="assistant_response", named=True, tenant=True),
    _row("dashboard", "--trend", event_type="tool_call", named=True),
    _row("explain", event_type="user_prompt"),
    _row("inflation", event_type="user_prompt", named=True),
    _row("oncall", "--rotation-start", "2026-01-01", event_type="tool_call", named=True),
    _row("policy", "generate", "--output", "gen.yaml", event_type="tool_call", named=True),
    _row("postmortem", "--list", event_type="session_end", named=True),
    _row("replay", CHILD, event_type="session_start"),
    _row("replay", event_type="session_start"),
    _row("replay", BAD, "--expand-subagents", event_type="user_prompt", named=True),
    _row("sample", "--strategy", "worst", "--n", "2", event_type="tool_call", named=True),
    _row("standup", "--no-llm", event_type="tool_call"),
    _row("team-report", event_type="tool_call", named=True),
    _row("tenant", "report", event_type="user_prompt", named=True),
]


class TestFixture(unittest.TestCase):
    def test_fixture_layout(self):
        for session_id, events, layout in (
            (BAD, _session_events(BAD, NOW, "bad", child_id=CHILD), BAD_LAYOUT),
            (CHILD, _child_events(NOW), CHILD_LAYOUT),
        ):
            with self.subTest(session_id=session_id):
                types = [event["event_type"] for event in events]
                self.assertEqual(
                    {t: (types.index(t) + 1, types.count(t)) for t in set(types)}, layout,
                )

    def test_store_loads_non_object_data_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            traces = Path(tmp)
            build_store(traces, bad_type="ALL", value=["unexpected", "list"])
            events = TraceStore(str(traces)).load_events(BAD)
        self.assertEqual(len(events), 25)
        self.assertTrue(all(event.data == ["unexpected", "list"] for event in events))


class TestCommandsRefuseNonObjectEventData(CliTestCase):
    def _check_rows(self, rows: list[Row], session_id: str) -> None:
        for index, row in enumerate(rows):
            json_type, value = NON_OBJECT_VALUES[index % len(NON_OBJECT_VALUES)]
            with self.subTest(argv=row.argv, event_type=row.event_type, json_type=json_type):
                poison = (
                    {"bad_type": row.event_type} if session_id == BAD
                    else {"child_type": row.event_type}
                )
                root = self.make_workspace(value=value, tenant=row.tenant, **poison)
                self.assert_refused(
                    root, list(row.argv),
                    expected_error(session_id, row.event_type, json_type, named=row.named),
                )

    def test_commands_refuse_non_object_data_in_the_session(self):
        self._check_rows(BAD_SESSION_REFUSALS, BAD)

    def test_commands_refuse_non_object_data_in_the_subagent_session(self):
        self._check_rows(CHILD_SESSION_REFUSALS, CHILD)

    def test_key_commands_refuse_every_non_object_json_type(self):
        rows = [
            _row("explain", BAD, event_type="tool_call"),
            _row("timeline", BAD, event_type="user_prompt"),
            _row("cost", BAD, event_type="tool_call"),
            _row("stats", BAD, event_type="tool_call"),
            _row("replay", BAD, event_type="session_start"),
        ]
        for row in rows:
            for json_type, value in NON_OBJECT_VALUES:
                with self.subTest(argv=row.argv, json_type=json_type):
                    root = self.make_workspace(bad_type=row.event_type, value=value)
                    self.assert_refused(
                        root, list(row.argv),
                        expected_error(BAD, row.event_type, json_type, named=False),
                    )

    def test_message_counts_every_other_non_object_event(self):
        root = self.make_workspace(bad_type="ALL", value="plain text data")
        self.assert_refused(
            root, ["explain", BAD],
            "Error: session event data must be an object: event 2 of 25 (user_prompt) "
            "has string data; 24 more events have non-object data\n",
        )

    def test_watch_compaction_checkpoint_refuses_at_startup(self):
        root = self.make_workspace(bad_type="user_prompt", value=["unexpected", "list"])
        # A regression would start tailing; an empty tail ends the watch at once.
        with mock.patch("agent_trace.watch._tail_events", return_value=iter([])):
            self.assert_refused(
                root, ["watch", BAD, "--dry-run", "--compaction-checkpoint"],
                expected_error(BAD, "user_prompt", "array", named=False),
            )

    def test_mcp_scan_watch_refuses_at_startup(self):
        root = self.make_workspace(bad_type="tool_result", value=42)
        with mock.patch(
            "agent_trace.mcp_scan.time.sleep",
            side_effect=AssertionError("mcp-scan --watch started tailing"),
        ):
            self.assert_refused(
                root, ["mcp-scan", "--watch", "--session", BAD],
                expected_error(BAD, "tool_result", "number", named=True),
            )

    def test_installed_entry_point_prints_no_traceback(self):
        root = self.make_workspace(bad_type="tool_call", value="plain text data")
        src = str(Path(agent_trace.__file__).resolve().parent.parent)
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(_ENV_PREFIXES)
        }
        env.update({
            "HOME": str(root / "home"),
            "PYTHONPATH": src,
            "DO_NOT_TRACK": "1",
            "AGENT_STRACE_TELEMETRY": "0",
        })
        for argv in (["explain", BAD], ["timeline", BAD], ["cost", BAD]):
            with self.subTest(argv=argv):
                before = _tree_snapshot(root)
                completed = subprocess.run(
                    [sys.executable, "-c", "from agent_trace.cli import main; main()", *argv],
                    cwd=root / "work", env=env, capture_output=True, text=True, timeout=120,
                )
                self.assertNotIn("Traceback", completed.stderr)
                self.assertEqual(
                    (completed.returncode, completed.stdout, completed.stderr),
                    (1, "", expected_error(BAD, "tool_call", "string", named=False)),
                )
                self.assertEqual(_tree_snapshot(root), before)


class TestCommandsThatDoNotReadTheDataAreUnchanged(CliTestCase):
    """Non-object data on an event type a command does not read changes nothing."""

    def test_output_matches_a_clean_store(self):
        rows = [
            (["explain", BAD], "file_read"),
            (["explain", BAD], "session_end"),
            (["cost", BAD], "session_end"),
            (["cost", BAD], "file_write"),
            (["stats", BAD], "tool_result"),
            (["timeline", BAD], "session_start"),
            (["timeline", BAD], "session_end"),
            (["replay", BAD, "--filter", "tool_call,error"], "file_read"),
            (["replay", BAD, "--limit", "5"], "tool_call"),
            (["token-budget", BAD], "tool_call"),
            (["audit", BAD], "file_read"),
            (["diff", GOOD, BAD], "file_read"),
            (["standup", BAD, "--no-llm"], "error"),
            (["why", BAD, "1"], "ALL"),
            (["list"], "ALL"),
            (["verify", BAD], "ALL"),
        ]
        clean_root = self.make_workspace()
        for index, (argv, event_type) in enumerate(rows):
            json_type, value = NON_OBJECT_VALUES[index % len(NON_OBJECT_VALUES)]
            with self.subTest(argv=argv, event_type=event_type, json_type=json_type):
                clean = self.run_cli(clean_root, *argv)
                root = self.make_workspace(bad_type=event_type, value=value)
                result = self.run_cli(root, *argv)
                self.assertEqual(clean.code, 0)
                self.assertEqual(
                    (result.code, result.out, result.err),
                    (clean.code, clean.out, clean.err),
                )

    def test_verify_still_checks_the_chain(self):
        for json_type, value in NON_OBJECT_VALUES:
            with self.subTest(json_type=json_type):
                root = self.make_workspace(bad_type="ALL", value=value)
                result = self.run_cli(root, "verify", BAD)
                self.assertEqual(result.code, 0)
                self.assertIn("Chain intact", result.out)
                self.assertIn("25 events verified", result.out)

    def test_export_json_and_csv_keep_the_raw_data(self):
        for json_type, value in NON_OBJECT_VALUES:
            with self.subTest(json_type=json_type):
                root = self.make_workspace(bad_type="ALL", value=value)
                exported = self.run_cli(root, "export", BAD, "--format", "json")
                self.assertEqual((exported.code, exported.err), (0, ""))
                events = json.loads(exported.out)
                self.assertEqual(len(events), 25)
                self.assertTrue(all(event.get("data") == value for event in events))

                table = self.run_cli(root, "export", BAD, "--format", "csv")
                self.assertEqual((table.code, table.err), (0, ""))
                rows = list(csv.reader(io.StringIO(table.out)))
                self.assertEqual(rows[0][-1], "data")
                self.assertEqual(len(rows), 26)

    def test_list_shows_every_session(self):
        root = self.make_workspace(bad_type="ALL", value=True, child_type="ALL")
        result = self.run_cli(root, "list")
        self.assertEqual((result.code, result.err), (0, ""))
        for session_id in (GOOD, BAD, CHILD):
            self.assertIn(session_id, result.out)
        self.assertIn("3 session(s)", result.out)

    def test_replay_html_keeps_data_that_dict_accepts(self):
        # The HTML renderer passes data to dict(), which accepts these values.
        for value in ([], "", [["key", "value"]]):
            with self.subTest(value=value):
                root = self.make_workspace(bad_type="tool_call", value=value)
                result = self.run_cli(
                    root, "replay", BAD, "--format", "html", "--output", "out.html",
                )
                self.assertEqual(result.code, 0)
                self.assertNotIn("Error:", result.err)
                self.assertTrue((root / "work" / "out.html").is_file())


def _live_event(event_type: EventType, data, ts: float = 2.0) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id="live-session", data=data)


def _watcher_config() -> WatcherConfig:
    return WatcherConfig(
        max_retries=3,
        max_cost_dollars=10.0,
        max_duration_seconds=3600.0,
        loop_threshold=99,
        loop_window=10,
        loop_sequence_length=3,
        loop_max_repeats=3,
        on_violation="terminal",
    )


class TestWatchSkipsLiveNonObjectEvents(unittest.TestCase):
    """watch keeps running and warns when a live event's data cannot be checked."""

    def _watch(self, events, **kwargs) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = TraceStore(tmp.name, redact=False)
        store.create_session(SessionMeta(session_id="live-session", started_at=1.0))
        out = io.StringIO()
        with mock.patch("agent_trace.watch._tail_events", return_value=iter(events)):
            watch_session(store, "live-session", _watcher_config(), out=out, **kwargs)
        return out.getvalue()

    @staticmethod
    def _skipped(event_type: str, json_type: str) -> str:
        return (
            f"[watch] Skipped checks for one {event_type} event because its data is "
            f"{_WARNING_NOUN[json_type]}, not an object.\n"
        )

    def test_events_checked_on_arrival_are_skipped_with_a_warning(self):
        for event_type in (EventType.TOOL_CALL, EventType.TOOL_RESULT, EventType.FILE_WRITE):
            for json_type, value in NON_OBJECT_VALUES:
                with self.subTest(event_type=event_type.value, json_type=json_type):
                    output = self._watch([
                        _live_event(event_type, value),
                        _live_event(EventType.SESSION_END, {"exit_code": 0}, ts=3.0),
                    ])
                    self.assertIn(self._skipped(event_type.value, json_type), output)
                    self.assertIn("[watch] Session ended (2 events, $0.0000)\n", output)

    def test_llm_request_is_skipped_only_when_the_token_budget_reads_it(self):
        known = _live_event(EventType.LLM_REQUEST, {"model": "claude-sonnet-4", "input_tokens": 10})
        no_model = _live_event(EventType.LLM_REQUEST, {"input_tokens": 10})
        unknown = _live_event(EventType.LLM_REQUEST, {"model": "unlisted-model-x", "input_tokens": 10})
        cases = [
            ("first request", [], True),
            ("after a known model", [known], True),
            ("before any model is known", [no_model], True),
            ("after an unknown model", [unknown], False),
        ]
        for label, before, skipped in cases:
            with self.subTest(case=label):
                output = self._watch([
                    *before,
                    _live_event(EventType.LLM_REQUEST, "plain text data", ts=3.0),
                    _live_event(EventType.SESSION_END, {"exit_code": 0}, ts=4.0),
                ])
                self.assertEqual(self._skipped("llm_request", "string") in output, skipped)
                self.assertIn(f"[watch] Session ended ({len(before) + 2} events,", output)

    def test_events_that_no_check_reads_are_not_skipped(self):
        for event_type in (EventType.USER_PROMPT, EventType.ERROR, EventType.SESSION_END):
            with self.subTest(event_type=event_type.value):
                output = self._watch([
                    _live_event(event_type, ["unexpected", "list"]),
                    _live_event(EventType.SESSION_END, {"exit_code": 0}, ts=3.0),
                ])
                self.assertNotIn("Skipped checks", output)
                self.assertIn("[watch] Session ended (", output)

    def test_skipped_event_is_still_streamed(self):
        skipped = _live_event(EventType.TOOL_CALL, 42)
        end = _live_event(EventType.SESSION_END, {"exit_code": 0}, ts=3.0)
        with mock.patch("agent_trace.watch.EventStreamer") as streamer_class:
            output = self._watch(
                [skipped, end], stream_config=StreamConfig(url=f"{_CLOSED_PORT}/events"),
            )
        streamer = streamer_class.return_value
        self.assertEqual(
            [entry.args[0] for entry in streamer.enqueue.call_args_list], [skipped, end],
        )
        streamer.stop.assert_called_once_with()
        self.assertIn(self._skipped("tool_call", "number"), output)

    def test_checkpoint_watcher_only_receives_object_data(self):
        class RecordingCheckpointWatcher:
            def __init__(self):
                self.updated = []

            def checkpoint_current(self):
                return None

            def update(self, event):
                self.updated.append(event)
                return None

        prompt = _live_event(EventType.USER_PROMPT, "plain text data")
        response = _live_event(EventType.LLM_RESPONSE, ["unexpected", "list"], ts=3.0)
        request = _live_event(EventType.LLM_REQUEST, {"model": "claude-sonnet-4"}, ts=4.0)
        end = _live_event(EventType.SESSION_END, {"exit_code": 0}, ts=5.0)
        watcher = RecordingCheckpointWatcher()
        output = self._watch([prompt, response, request, end], checkpoint_watcher=watcher)
        self.assertEqual(watcher.updated, [request, end])
        self.assertIn(self._skipped("llm_response", "array"), output)
        self.assertNotIn("Skipped checks for one user_prompt", output)
        self.assertIn("[watch] Session ended (4 events,", output)


class TestMcpScanWatchSkipsLiveNonObjectEvents(CliTestCase):
    def _watch_with_appended(self, records: list[dict]) -> CliResult:
        root = self.make_workspace()
        events_file = root / "work" / ".agent-traces" / BAD / "events.ndjson"
        sleeps = []

        def append_then_stop(seconds):
            sleeps.append(seconds)
            if len(sleeps) > 1:
                raise AssertionError("mcp-scan --watch did not stop at session_end")
            with events_file.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

        with mock.patch("agent_trace.mcp_scan.time.sleep", side_effect=append_then_stop):
            return self.run_cli(root, "mcp-scan", "--watch", "--session", BAD)

    def test_live_non_object_event_is_skipped_with_a_warning(self):
        result = self._watch_with_appended([
            {"event_type": "tool_call", "timestamp": NOW, "event_id": "live-call",
             "session_id": BAD, "data": "plain text data"},
            {"event_type": "session_end", "timestamp": NOW + 1, "event_id": "live-end",
             "session_id": BAD, "data": {"reason": "completed"}},
        ])
        self.assertEqual(result.code, 0)
        self.assertEqual(
            result.err,
            "[mcp-scan] Skipped checks for one tool_call event because its data is "
            "a string, not an object.\n",
        )
        self.assertTrue(result.out.endswith("Session ended after 2 watched events.\n"))

    def test_non_object_session_end_still_ends_the_watch(self):
        result = self._watch_with_appended([
            {"event_type": "session_end", "timestamp": NOW, "event_id": "live-end",
             "session_id": BAD, "data": ["unexpected", "list"]},
        ])
        self.assertEqual(result.code, 0)
        self.assertEqual(
            result.err,
            "[mcp-scan] Skipped checks for one session_end event because its data is "
            "an array, not an object.\n",
        )
        self.assertTrue(result.out.endswith("Session ended after 1 watched events.\n"))


if __name__ == "__main__":
    unittest.main()
