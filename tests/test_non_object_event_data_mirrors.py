"""Differential tests for event data checks that mirror what a command reads.

Most commands refuse a session when an event of a type they read has
non-object data. Some read only part of a session: an exporter pairs events
by ID, ``why`` follows a causal walk, ``sample --deduplicate`` skips repeated
sessions. Their checks use a helper that mirrors those reads. A helper that
misses a read lets a traceback through, and one that adds a read refuses a
session the command would have handled.

Each test stores a non-object value as the data of one event at a time and
runs every command twice: as shipped, and with ``require_object_event_data``
patched to a no-op in every agent_trace module. The shipped run must refuse
exactly when the unchecked run raises, with the standard ``Error:`` line and
nothing written, sent, opened or printed. The one exception is ``optimize``
with an LLM configured: when the LLM returns no patterns, it can send the
request and then refuse, as test_optimize shows. Without the check it sends
the same request and then raises.
"""

from __future__ import annotations

import contextlib
import errno
import importlib
import json
import os
import shutil
import socket
import sys
import tempfile
import traceback
import unittest
import urllib.error
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import agent_trace
from agent_trace import event_data
from agent_trace.cli import main
from agent_trace.temporal import TEMPORAL_TRACE_PARENT_ENV

try:
    from . import test_non_object_event_data as base
except ImportError:  # unittest discover -s tests imports test modules top-level
    import test_non_object_event_data as base

NOW = base.NOW
CLOSED = base._CLOSED_PORT
REFUSAL = "event data must be an object"

M1 = "e1e2e3e4e5e60011"
P_OK = "a1a2a3a4a5a60021"
P_FAIL = "a1a2a3a4a5a60022"
P_CRASH = "a1a2a3a4a5a60023"
S1 = "b1b2b3b4b5b60031"
S2 = "b1b2b3b4b5b60032"
S3 = "b1b2b3b4b5b60033"
MCP_EARLIER = "c1c2c3c4c5c60041"
MCP_SCANNED = "c1c2c3c4c5c60042"
MCP_LATER = "c1c2c3c4c5c60043"
TREE_PARENT = "d1d2d3d4d5d60051"
TREE_CHILD = "d1d2d3d4d5d60052"
TREE_STRAY = "d1d2d3d4d5d60053"
O1 = "f1f2f3f4f5f60061"
NO_CONTEXT = "f1f2f3f4f5f60071"

TEMPORAL_CONTEXT = {
    "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
    "parent_span_id": "00f067aa0ba902b7",
}

_START = {"command": ["claude"], "provider": "claude"}
_END = {"reason": "completed"}
_LLM_REQUEST = {"model": "claude-sonnet-4", "message_count": 2}
_LLM_RESPONSE = {
    "model": "claude-sonnet-4", "input_tokens": 900, "output_tokens": 100, "total_tokens": 1000,
}
_PYTEST = {"tool_name": "Bash", "arguments": {"command": "pytest -q"}}
_READ = {"tool_name": "Read", "arguments": {"file_path": "README.md"}}


def _result(tool_name: str, text: str) -> dict:
    return {"tool_name": tool_name, "result": text}


# Rows are (event_type, data[, event_id[, parent_id]]). The comments give the
# 1-based position that the refusal message reports.
EXPORT_ROWS = [
    ("session_start", _START),                                               # 1
    ("user_prompt", {"prompt": "Fix the failing test in src/app.py"}),       # 2
    ("llm_response", _LLM_RESPONSE),                                         # 3 before any request
    ("llm_request", _LLM_REQUEST, "llm-a"),                                  # 4
    ("llm_response", _LLM_RESPONSE, "", "llm-a"),                            # 5
    ("llm_request", _LLM_REQUEST, "llm-b"),                                  # 6
    ("llm_request", _LLM_REQUEST, "llm-c"),                                  # 7
    ("llm_response", _LLM_RESPONSE),                                         # 8 pairs with 6, the oldest open request
    ("llm_request", _LLM_REQUEST, "llm-b"),                                  # 9 reuses the ID of 6
    ("llm_request", _LLM_REQUEST, "llm-c"),                                  # 10 replaces 7 and keeps its place
    ("llm_response", _LLM_RESPONSE),                                         # 11 pairs with 10
    ("llm_request", _LLM_REQUEST, "llm-b"),                                  # 12 replaces 9, 38 replaces it
    ("tool_call", _PYTEST, "call-1"),                                        # 13
    ("tool_result", _result("Bash", "1 failed"), "", "call-1"),              # 14
    ("tool_call", _PYTEST, "call-2"),                                        # 15 an error pairs with it, 34 reuses its ID
    ("error", {"tool_name": "Bash", "message": "pytest exited with 1"}, "", "call-2"),  # 16
    ("tool_call", _PYTEST, "call-3"),                                        # 17 never paired
    ("tool_result", _result("Bash", "stray output"), "", "gone"),            # 18 parent not in the session
    ("error", {"message": "rate limited"}),                                  # 19
    ("llm_request", _LLM_REQUEST, "llm-l"),                                  # 20 replaced by 21
    ("llm_request", _LLM_REQUEST, "llm-l"),                                  # 21
    ("llm_response", _LLM_RESPONSE, "", "llm-l"),                            # 22
    ("decision", {"choice": "edit src/app.py", "reason": "the assertion fails there"}),  # 23
    ("file_read", {"path": "src/app.py"}),                                   # 24
    ("tool_call", {"tool_name": "Edit", "arguments": {"file_path": "src/app.py"}}, "call-edit"),  # 25
    ("tool_result", _result("Edit", "updated src/app.py"), "", "call-edit"),  # 26
    ("file_write", {"path": "src/app.py"}),                                  # 27
    ("assistant_response", {"text": "Fixed the failing test."}),             # 28
    ("tool_call", {"tool_name": "Bash", "arguments": {"command": "ls"}}, "call-dup"),  # 29 replaced by 30
    ("tool_call", {"tool_name": "Bash", "arguments": {"command": "ls -la"}}, "call-dup"),  # 30
    ("tool_result", _result("Bash", "src"), "", "call-dup"),                 # 31
    ("tool_call", _PYTEST, "call-4"),                                        # 32 an error pairs with it, nothing reuses its ID
    ("error", {"tool_name": "Bash", "message": "pytest timed out"}, "", "call-4"),  # 33
    ("tool_call", _PYTEST, "call-2"),                                        # 34 reuses the ID of 15 after the error at 16
    ("tool_call", _PYTEST, "call-1"),                                        # 35 reuses the ID of 13 after the tool_result at 14
    ("llm_request", _LLM_REQUEST, "llm-m"),                                  # 36
    ("llm_response", _LLM_RESPONSE, "", "llm-m"),                            # 37 pairs with 36 by parent while 12 is older
    ("llm_request", _LLM_REQUEST, "llm-b"),                                  # 38
    ("session_end", _END),                                                   # 39
]

POSTMORTEM_OK_ROWS = [
    ("session_start", _START),                                               # 1
    ("user_prompt", {"prompt": "Run the build"}),                            # 2
    ("tool_call", _PYTEST, "call-1"),                                        # 3
    ("tool_result", _result("Bash", "3 passed"), "", "call-1"),              # 4
    ("tool_call", {"tool_name": "Bash", "arguments": {"command": "make"}}, "call-2"),  # 5
    ("tool_result", _result("Bash", "make: exit code 2"), "", "call-2"),     # 6 first failure signal
    ("tool_call", {"tool_name": "Read", "arguments": {"file_path": "Makefile"}}, "call-3"),  # 7
    ("tool_result", _result("Read", "all: build"), "", "call-3"),            # 8 after the signal
    ("file_read", {"path": "Makefile"}),                                     # 9
    ("llm_request", _LLM_REQUEST),                                           # 10
    ("file_write", {"path": "Makefile"}),                                    # 11
    ("session_end", _END),                                                   # 12
]
POSTMORTEM_FAIL_ROWS = [
    ("session_start", _START),                                               # 1
    ("user_prompt", {"prompt": "Run the tests"}),                            # 2
    ("llm_request", _LLM_REQUEST),                                           # 3 not the last prompt
    ("tool_call", _PYTEST, "call-1"),                                        # 4
    ("tool_result", _result("Bash", "1 failed"), "", "call-1"),              # 5
    ("error", {"tool_name": "Bash", "message": "pytest exited with 1"}),     # 6
    ("file_write", {"path": "src/app.py"}),                                  # 7
    ("llm_request", _LLM_REQUEST),                                           # 8 last prompt
    ("llm_response", _LLM_RESPONSE),                                         # 9 last action
    ("session_end", _END),                                                   # 10
]
POSTMORTEM_CRASH_ROWS = [
    ("session_start", _START),                                               # 1
    ("user_prompt", {"prompt": "Deploy"}),                                   # 2
    ("tool_call", _PYTEST, "call-1"),                                        # 3
    ("tool_result", _result("Bash", "ok"), "", "call-1"),                    # 4
    ("file_write", {"path": "deploy.sh"}),                                   # 5
    ("llm_request", _LLM_REQUEST),                                           # 6 last prompt
    ("tool_call", {"tool_name": "Bash", "arguments": {"command": "./deploy.sh"}}, "call-2"),  # 7
    ("session_end", {"exit_code": 1, "reason": "crashed"}),                  # 8
]

TREE_PARENT_ROWS = [
    ("session_start", _START),
    ("user_prompt", {"prompt": "Test the app"}),
    ("tool_call", {"tool_name": "Task", "arguments": {"prompt": "Run the tests"}}, "task"),
    ("subagent_start", {"agent_type": "tester", "child_session_id": TREE_CHILD}),
    ("subagent_stop", {"agent_type": "tester", "stop_reason": "done"}),
    ("tool_result", _result("Task", "tests pass"), "", "task"),
    ("assistant_response", {"text": "The tests pass."}),
    ("session_end", _END),
]
TREE_CHILD_ROWS = [
    ("session_start", _START),
    ("user_prompt", {"prompt": "Run the tests"}),
    ("tool_call", _PYTEST, "call-1"),
    ("tool_result", _result("Bash", "1 failed"), "", "call-1"),
    ("error", {"message": "pytest exited with 1"}),
    ("assistant_response", {"text": "One test fails."}),
    ("session_end", _END),
]

OPTIMIZE_ROWS = [
    ("session_start", _START),                                               # 1
    ("user_prompt", {"prompt": "Fix the tests"}),                            # 2
    ("tool_call", _PYTEST, "call-1"),                                        # 3
    ("error", {"message": "pytest exited with 1"}),                          # 4
    ("tool_call", _PYTEST, "call-2"),                                        # 5
    ("error", {"message": "pytest exited with 1"}),                          # 6
    ("tool_call", _PYTEST, "call-3"),                                        # 7
    ("error", {"message": "pytest exited with 1"}),                          # 8 the prompt reads two errors
    ("file_write", {"path": "src/app.py"}),                                  # 9
    ("llm_request", _LLM_REQUEST),                                           # 10
    ("llm_response", _LLM_RESPONSE),                                         # 11
    ("session_end", _END),                                                   # 12
]
LLM_PATTERNS = json.dumps({"patterns": [{
    "name": "retry-loop",
    "description": "Reruns pytest after each failure.",
    "affected_sessions": [O1[:8]],
    "proposed_addition": "## Tests\nRead the failure before rerunning pytest.",
}]})


def _sample_rows(*calls: dict) -> list[tuple]:
    rows: list[tuple] = [("session_start", _START), ("user_prompt", {"prompt": "Run the tests"})]
    for number, call in enumerate(calls, 1):
        rows.append(("tool_call", call, f"call-{number}"))
        rows.append(("tool_result", _result(call["tool_name"], "done"), "", f"call-{number}"))
    return rows + [
        ("llm_request", _LLM_REQUEST),
        ("llm_response", _LLM_RESPONSE),
        ("file_write", {"path": "src/app.py"}),
        ("session_end", _END),
    ]


def _mcp_rows(description: str) -> list[tuple]:
    return [
        ("session_start", {**_START, "tools": [{"name": "fetch", "description": description}]}),
        ("user_prompt", {"prompt": "Summarise the README"}),
        ("tool_call", _READ, "call-1"),
        ("tool_result", _result("Read", "# App"), "", "call-1"),
        ("session_end", _END),
    ]


def _events(session_id: str, start: float, rows: list[tuple]) -> list[dict]:
    prefix = session_id[-4:]
    events = []
    for position, (event_type, data, *ids) in enumerate(rows, 1):
        record = {
            "event_type": event_type,
            "timestamp": start + position,
            "event_id": f"{prefix}-{ids[0] if ids and ids[0] else position}",
            "session_id": session_id,
            "data": data,
        }
        if len(ids) > 1:
            record["parent_id"] = f"{prefix}-{ids[1]}"
        events.append(record)
    return events


def _json_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    raise TypeError(f"not a JSON value: {value!r}")


def _same_value(a, b) -> bool:
    return type(a) is type(b) and a == b


# ---------------------------------------------------------------------------
# Unchecked mode
# ---------------------------------------------------------------------------

_ORIGINAL_CHECK = event_data.require_object_event_data


def _skip_check(*args, **kwargs) -> None:
    return None


def _checking_module_names() -> list[str]:
    package = Path(agent_trace.__file__).parent
    names = []
    for path in sorted(package.rglob("*.py")):
        if "require_object_event_data" not in path.read_text(encoding="utf-8"):
            continue
        parts = path.relative_to(package).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(("agent_trace", *parts)))
    return names


_CHECKING_MODULES = _checking_module_names()


def _bindings(target) -> list[str]:
    """Name every agent_trace module attribute that is *target*."""
    return sorted(
        f"{name}.{attr}"
        for name, module in list(sys.modules.items())
        if module is not None and (name == "agent_trace" or name.startswith("agent_trace."))
        for attr, value in list(vars(module).items())
        if value is target
    )


@contextlib.contextmanager
def _unchecked():
    """Patch require_object_event_data to a no-op wherever agent_trace binds it.

    Every module that uses the check is imported first. A module imported
    for the first time while the patch is active would keep the no-op.
    why._data raises for non-object data so the causal walk can refuse; here
    it returns the data so the walk reads it as the command would without
    the check.
    """
    for name in _CHECKING_MODULES:
        importlib.import_module(name)
    with contextlib.ExitStack() as stack:
        for binding in _bindings(_ORIGINAL_CHECK):
            module_name, attr = binding.rsplit(".", 1)
            stack.enter_context(mock.patch.object(sys.modules[module_name], attr, _skip_check))
        stack.enter_context(mock.patch("agent_trace.why._data", lambda event: event.data))
        yield


# ---------------------------------------------------------------------------
# Running a command
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]
    named: bool = False
    values: tuple = (42,)
    refuses: bool = True
    clean_code: int = 0
    clean_out: str = ""
    clean_err: str = ""
    llm: bool = False
    llm_reply: str | None = None


@dataclass
class Run:
    code: int | str | None
    out: str
    err: str
    raised: str | None
    network: list
    browser: list
    llm_calls: int
    changed: bool = False


def _run(root: Path, argv, llm_reply: str | None = None) -> Run:
    network: list = []
    browser: list = []
    prompts: list = []

    def refuse_connect(*args, **kwargs):
        network.append(args[1:] if args and isinstance(args[0], socket.socket) else args)
        raise ConnectionRefusedError("network is disabled in this test")

    def refuse_connect_ex(sock, address, *args, **kwargs):
        network.append(address)
        return errno.ECONNREFUSED

    def refuse_urlopen(request, *args, **kwargs):
        network.append(getattr(request, "full_url", request))
        raise urllib.error.URLError(ConnectionRefusedError("network is disabled in this test"))

    def record_browser(url, *args, **kwargs):
        browser.append(url)
        return True

    def answer_llm(prompt, *args, **kwargs):
        prompts.append(prompt)
        return llm_reply

    code, raised = None, None
    with contextlib.ExitStack() as stack:
        stack.enter_context(base._chdir(root / "work"))
        stack.enter_context(base._isolated_env(root / "home"))
        for key in [key for key in os.environ if key.startswith("OPENAI_")]:
            del os.environ[key]
        stack.enter_context(mock.patch.object(socket.socket, "connect", refuse_connect))
        stack.enter_context(mock.patch.object(socket.socket, "connect_ex", refuse_connect_ex))
        stack.enter_context(mock.patch("socket.create_connection", refuse_connect))
        stack.enter_context(mock.patch("urllib.request.urlopen", refuse_urlopen))
        for name in ("open", "open_new", "open_new_tab"):
            stack.enter_context(mock.patch.object(webbrowser, name, record_browser))
        stack.enter_context(mock.patch("agent_trace.cli._product_telemetry.capture"))
        stack.enter_context(
            mock.patch("agent_trace.cli._product_telemetry.maybe_show_telemetry_notice")
        )
        stack.enter_context(mock.patch("agent_trace.optimize._call_llm", answer_llm))
        stack.enter_context(mock.patch.object(sys, "argv", ["agent-strace", *argv]))
        captured = stack.enter_context(base._capture_output())
        try:
            main()
            code = 0
        except SystemExit as exc:
            code = 0 if exc.code is None else exc.code
        except Exception as exc:  # the unchecked run fails however the reader fails
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            raised = f"{type(exc).__name__}: {exc} ({Path(frame.filename).name}:{frame.lineno})"
    return Run(code, captured.out, captured.err, raised, network, browser, len(prompts))


def _restore_tree(root: Path, before: dict) -> bool:
    """Undo every change under *root* since *before*; return whether there was one."""
    after = base._tree_snapshot(root)
    if after == before:
        return False
    for name in set(after) - set(before):
        path = root / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for name, entry in before.items():
        path = root / name
        if entry is None:
            path.mkdir(parents=True, exist_ok=True)
        elif after.get(name) != entry:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(entry[1])
            os.utime(path, ns=(entry[0], entry[0]))
    return True


def _run_restoring(root: Path, command: Command) -> Run:
    before = base._tree_snapshot(root)
    run = _run(root, command.argv, command.llm_reply)
    run.changed = _restore_tree(root, before)
    return run


@contextlib.contextmanager
def _poisoned(path: Path, position: int, value):
    """Store *value* as the data of the event at 1-based *position*."""
    original = path.read_bytes()
    lines = original.decode("utf-8").splitlines(keepends=True)
    record = json.loads(lines[position - 1])
    record["data"] = value
    lines[position - 1] = json.dumps(record, separators=(",", ":")) + "\n"
    path.write_bytes("".join(lines).encode("utf-8"))
    try:
        yield
    finally:
        path.write_bytes(original)


@dataclass
class Sweep:
    refused: dict = field(default_factory=dict)
    llm_calls: dict = field(default_factory=dict)

    def positions(self, label: str, session_id: str, value=42) -> set[int]:
        """Return the positions in *session_id* where *label* refused *value*."""
        return {
            position for sid, position, shown in self.refused[label]
            if sid == session_id and shown == repr(value)
        }


def _mismatch(command: Command, normal: Run, unchecked: Run, expected: str) -> str | None:
    refused = normal.code == 1 and REFUSAL in normal.err
    if normal.raised:
        return f"raised {normal.raised}"
    if REFUSAL in normal.out + normal.err and not refused:
        return f"printed a refusal but exited {normal.code}"
    if refused and not unchecked.raised:
        return f"refused, but without the check it exits {unchecked.code}"
    if not refused and unchecked.raised:
        return f"exited {normal.code}, but without the check it raises {unchecked.raised}"
    if normal.llm_calls != unchecked.llm_calls:
        return f"called the LLM {normal.llm_calls} times, {unchecked.llm_calls} without the check"
    if not refused:
        if normal.code != unchecked.code:
            return f"exited {normal.code}, {unchecked.code} without the check"
        return None
    if (normal.out, normal.err) != ("", expected):
        return f"printed {normal.out!r} and {normal.err!r}, expected {expected!r}"
    if normal.changed or normal.network or normal.browser:
        return (
            f"refused after changing files ({normal.changed}), using the network "
            f"{normal.network} or opening {normal.browser}"
        )
    return None


class MirroredCheckTest(unittest.TestCase):
    """Poison one event at a time and compare each command with and without the check."""

    maxDiff = None

    def workspace(self, *sessions: tuple) -> tuple[Path, dict[str, list[dict]]]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        (root / "home").mkdir()
        traces = root / "work" / ".agent-traces"
        traces.mkdir(parents=True)
        (traces / ".gitignore").write_text("*\n", encoding="utf-8")
        written = {}
        for session_id, start, rows, extra in sessions:
            events = _events(session_id, start, rows)
            base._write_session(traces, session_id, start, events, **extra)
            written[session_id] = events
        return root, written

    def sweep(self, root: Path, sessions: dict[str, list[dict]], commands: list[Command]) -> Sweep:
        self.assertEqual(len({command.label for command in commands}), len(commands))

        # On the clean store both runs take the path the fixture is built for.
        normal = {command.label: _run_restoring(root, command) for command in commands}
        with _unchecked():
            unchecked = {command.label: _run_restoring(root, command) for command in commands}
        clean_problems = []
        for command in commands:
            n, u = normal[command.label], unchecked[command.label]
            got = (
                n.raised, u.raised, n.code, u.code, REFUSAL in n.err,
                command.clean_out in n.out, command.clean_err in n.err, n.llm_calls, u.llm_calls,
            )
            want = (
                None, None, command.clean_code, command.clean_code, False,
                True, True, int(command.llm), int(command.llm),
            )
            if got != want:
                clean_problems.append(
                    f"{command.label}: got {got}, want {want}; out {n.out[-400:]!r}; err {n.err[-400:]!r}"
                )
        self.assertEqual(clean_problems, [])

        values: list = []
        for command in commands:
            for value in command.values:
                if not any(_same_value(value, seen) for seen in values):
                    values.append(value)

        result = Sweep(
            refused={command.label: set() for command in commands},
            llm_calls={command.label: {} for command in commands},
        )
        problems = []
        traces = root / "work" / ".agent-traces"
        for session_id, events in sessions.items():
            path = traces / session_id / "events.ndjson"
            for position, event in enumerate(events, 1):
                for value in values:
                    selected = [
                        command for command in commands
                        if any(_same_value(value, own) for own in command.values)
                    ]
                    with _poisoned(path, position, value):
                        normal = {c.label: _run_restoring(root, c) for c in selected}
                        with _unchecked():
                            unchecked = {c.label: _run_restoring(root, c) for c in selected}
                    key = (session_id, position, repr(value))
                    for command in selected:
                        n, u = normal[command.label], unchecked[command.label]
                        result.llm_calls[command.label][key] = n.llm_calls
                        if n.code == 1 and REFUSAL in n.err:
                            result.refused[command.label].add(key)
                        subject = f"session {session_id}" if command.named else "session"
                        expected = (
                            f"Error: {subject} {REFUSAL}: event {position} of {len(events)} "
                            f"({event['event_type']}) has {_json_type(value)} data\n"
                        )
                        problem = _mismatch(command, n, u, expected)
                        if problem:
                            problems.append(
                                f"{command.label}: {event['event_type']} {session_id} "
                                f"#{position} = {value!r}: {problem}"
                            )
        self.assertEqual(problems, [])
        for command in commands:
            if command.refuses:
                self.assertTrue(result.refused[command.label], f"{command.label} never refused")
            else:
                self.assertEqual(result.refused[command.label], set(), command.label)
        return result

    def test_unchecked_mode_patches_every_binding(self):
        package = Path(agent_trace.__file__).parent
        raising = sorted(
            path.relative_to(package).as_posix()
            for path in package.rglob("*.py")
            if "NonObjectEventDataError(" in path.read_text(encoding="utf-8")
        )
        # Every refusal is raised by require_object_event_data.
        self.assertEqual(raising, ["event_data.py"])
        self.assertIn("agent_trace.event_data", _CHECKING_MODULES)
        with _unchecked():
            self.assertEqual(_bindings(_ORIGINAL_CHECK), [])
            patched = {binding.rsplit(".", 1)[0] for binding in _bindings(_skip_check)}
        self.assertEqual(sorted(set(_CHECKING_MODULES) - patched), [])
        self.assertEqual(_bindings(_skip_check), [])

    def test_exports_replay_why_token_budget_and_compare(self):
        root, sessions = self.workspace((M1, NOW - 600, EXPORT_ROWS, TEMPORAL_CONTEXT))

        def export(label, *flags, clean_code=0, clean_err=""):
            return Command(
                f"export {label}", ("export", M1, *flags), clean_code=clean_code, clean_err=clean_err,
            )

        commands = [
            export("otlp-genai -o", "--format", "otlp-genai", "--output", "genai.json",
                   clean_err="OTLP payload written to genai.json"),
            export("otlp-genai --endpoint", "--format", "otlp-genai", "--endpoint", CLOSED,
                   clean_code=1, clean_err="OTLP GenAI export failed"),
            export("temporal -o", "--format", "temporal", "--output", "temporal.json",
                   clean_err="OTLP payload written to temporal.json"),
            export("temporal --endpoint", "--format", "temporal", "--endpoint", CLOSED, clean_code=1),
            export("otlp -o", "--format", "otlp", "--output", "otlp.json",
                   clean_err="OTLP payload written to otlp.json"),
            export("otlp --endpoint", "--format", "otlp", "--endpoint", CLOSED, clean_code=1),
            export("langfuse", "--backend", "langfuse", "--langfuse-public-key", "pk",
                   "--langfuse-secret-key", "sk", "--langfuse-host", CLOSED, clean_code=1),
            Command("token-budget", ("token-budget", M1)),
            Command("replay", ("replay", M1)),
            Command("replay --filter", ("replay", M1, "--filter", "tool_call,error")),
            Command("replay --limit", ("replay", M1, "--limit", "5")),
            Command(
                "replay html", ("replay", M1, "--format", "html", "--output", "r.html"),
                values=(42, [], "", "text"), clean_out="HTML replay written to r.html",
            ),
            *(Command(f"why {target}", ("why", M1, str(target))) for target in (17, 19, 22, 25, 27)),
            Command(
                "compare --rerun", ("compare", M1, "--rerun"),
                clean_code=1, clean_err="--rerun is not yet automated",
            ),
        ]
        sweep = self.sweep(root, sessions, commands)

        # otlp-genai and temporal read a tool_call when a tool_result pairs
        # with it or nothing does, but not when an error pairs with it, whether
        # or not a later tool_call reuses its ID. An llm_response pairs with
        # the request it names, and a parentless one pairs with the oldest
        # open request. A request replaced by one with the same ID before it
        # pairs is not read.
        for label in ("export otlp-genai -o", "export temporal -o"):
            self.assertEqual(
                sweep.positions(label, M1),
                {2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 16, 17, 18, 19, 21, 22, 25, 26, 28, 30, 31, 33, 34,
                 35, 36, 37, 38},
                label,
            )
        # Plain otlp also reads the tool_call an error pairs with, even when a
        # later tool_call reuses its ID, but not a tool_call that a later one
        # with the same ID replaces before anything pairs with it.
        self.assertIn(15, sweep.positions("export otlp -o", M1))
        self.assertNotIn(29, sweep.positions("export otlp -o", M1))
        self.assertIn(30, sweep.positions("export otlp -o", M1))
        # replay --format html copies the data with dict(), which accepts an
        # empty array or string.
        self.assertEqual(sweep.positions("replay html", M1, []), set())
        self.assertEqual(sweep.positions("replay html", M1, ""), set())
        self.assertEqual(len(sweep.positions("replay html", M1, "text")), len(EXPORT_ROWS))

    def test_temporal_export_without_context_keeps_its_message(self):
        rows = [
            ("session_start", _START),
            ("tool_call", _PYTEST, "call-1"),
            ("tool_result", ["unexpected", "list"], "", "call-1"),
            ("session_end", _END),
        ]
        root, _ = self.workspace((NO_CONTEXT, NOW - 600, rows, {}))
        expected = (
            "Temporal export failed: session has no Temporal trace context; "
            f"run watch with {TEMPORAL_TRACE_PARENT_ENV} set\n"
        )
        for flags in (("--output", "t.json"), ("--endpoint", CLOSED)):
            with self.subTest(flags=flags):
                before = base._tree_snapshot(root)
                run = _run(root, ("export", NO_CONTEXT, "--format", "temporal", *flags))
                self.assertIsNone(run.raised)
                self.assertEqual((run.code, run.out, run.err), (1, "", expected))
                self.assertEqual((run.network, run.browser), ([], []))
                self.assertEqual(base._tree_snapshot(root), before)

    def test_postmortem(self):
        root, sessions = self.workspace(
            (P_OK, NOW - 900, POSTMORTEM_OK_ROWS, {}),
            (P_FAIL, NOW - 600, POSTMORTEM_FAIL_ROWS, {}),
            (P_CRASH, NOW - 300, POSTMORTEM_CRASH_ROWS, {}),
        )
        commands = [
            Command("postmortem ok", ("postmortem", P_OK)),
            Command("postmortem failed", ("postmortem", P_FAIL), clean_code=1),
            Command("postmortem crashed", ("postmortem", P_CRASH), clean_code=1),
            Command("postmortem --list", ("postmortem", "--list"), named=True, clean_code=1),
        ]
        sweep = self.sweep(root, sessions, commands)
        # Without errors, tool results are read up to the first failure
        # signal, and only a failed session builds the recovery context.
        self.assertEqual(sweep.positions("postmortem ok", P_OK), {2, 3, 4, 5, 6, 7, 9, 12})
        self.assertEqual(sweep.positions("postmortem failed", P_FAIL), {2, 4, 6, 7, 8, 10})
        self.assertEqual(sweep.positions("postmortem crashed", P_CRASH), {2, 3, 4, 5, 6, 7, 8})
        for label, session_id in (
            ("postmortem ok", P_OK), ("postmortem failed", P_FAIL), ("postmortem crashed", P_CRASH),
        ):
            self.assertEqual(
                {key[0] for key in sweep.refused[label]}, {session_id}, label,
            )
        # --list reads only the session_end of each session.
        self.assertEqual(
            sweep.refused["postmortem --list"],
            {(P_OK, 12, "42"), (P_FAIL, 10, "42"), (P_CRASH, 8, "42")},
        )

    def test_sample(self):
        root, sessions = self.workspace(
            (S1, NOW - 900, _sample_rows(_PYTEST, _PYTEST), {}),
            (S2, NOW - 600, _sample_rows(_PYTEST, _PYTEST), {}),
            (S3, NOW - 300, _sample_rows(_READ), {}),
        )
        argv = ("sample", "--strategy", "worst", "--n", "2", "--output", "s.jsonl")
        commands = [
            Command("sample", argv, named=True),
            Command("sample --deduplicate", (*argv, "--deduplicate"), named=True),
        ]
        sweep = self.sweep(root, sessions, commands)
        # Sessions are scored newest first, so --deduplicate drops S1 after
        # reading only its tool calls.
        self.assertEqual(sweep.positions("sample --deduplicate", S1), {3, 5})
        self.assertEqual(sweep.positions("sample", S1), {3, 5, 8, 9})
        self.assertEqual(sweep.positions("sample --deduplicate", S2), {3, 5, 8, 9})

    def test_mcp_scan(self):
        root, sessions = self.workspace(
            (MCP_EARLIER, NOW - 900, _mcp_rows("Fetch a URL."), {}),
            (MCP_SCANNED, NOW - 600, _mcp_rows("Fetch a URL and return the page text."), {}),
            (MCP_LATER, NOW - 300, _mcp_rows("Fetch a URL."), {}),
        )
        command = Command(
            "mcp-scan", ("mcp-scan", "--session", MCP_SCANNED),
            named=True, values=(42, [], "text", None),
        )
        sweep = self.sweep(root, sessions, [command])
        refused = sweep.refused["mcp-scan"]
        # Every event of the scanned session, and the session_start of an
        # earlier session unless its data is empty.
        self.assertEqual(
            {key for key in refused if key[0] == MCP_EARLIER},
            {(MCP_EARLIER, 1, "42"), (MCP_EARLIER, 1, "'text'")},
        )
        self.assertEqual(
            len({key for key in refused if key[0] == MCP_SCANNED}), 5 * len(command.values),
        )
        self.assertFalse({key for key in refused if key[0] == MCP_LATER})

    def test_subagent_tree_replay(self):
        task_id = f"{TREE_PARENT[-4:]}-task"
        root, sessions = self.workspace(
            (TREE_PARENT, NOW - 900, TREE_PARENT_ROWS, {}),
            (TREE_CHILD, NOW - 890, TREE_CHILD_ROWS,
             {"parent_session_id": TREE_PARENT, "parent_event_id": task_id, "depth": 1}),
            (TREE_STRAY, NOW - 880, TREE_CHILD_ROWS,
             {"parent_session_id": TREE_PARENT, "parent_event_id": "no-such-call", "depth": 1}),
        )
        commands = [
            Command("replay --expand-subagents", ("replay", TREE_PARENT, "--expand-subagents"),
                    named=True),
            Command("replay --tree", ("replay", TREE_PARENT, "--tree"), refuses=False),
        ]
        sweep = self.sweep(root, sessions, commands)
        # A child is rendered, and read, only under the tool_call that spawned it.
        self.assertEqual(
            {key[0] for key in sweep.refused["replay --expand-subagents"]},
            {TREE_PARENT, TREE_CHILD},
        )

    def test_optimize(self):
        root, sessions = self.workspace((O1, NOW - 600, OPTIMIZE_ROWS, {}))
        llm = ("optimize", O1, "--base-url", CLOSED, "--api-key", "k")
        commands = [
            Command("optimize llm patterns", llm, llm=True, llm_reply=LLM_PATTERNS),
            Command("optimize llm none", llm, llm=True, llm_reply=None),
            Command("optimize heuristics", ("optimize", O1, "--dry-run")),
        ]
        sweep = self.sweep(root, sessions, commands)

        def calls_at_refusal(label):
            return {
                position: sweep.llm_calls[label][(O1, position, "42")]
                for position in sweep.positions(label, O1)
            }

        # The prompt's events are checked before the request. With patterns
        # from the LLM the heuristics never run.
        self.assertEqual(calls_at_refusal("optimize llm patterns"), {3: 0, 4: 0, 5: 0, 6: 0, 7: 0})
        # Without patterns the heuristics run after the request and can refuse
        # then, where the command without the check sends it and then crashes.
        self.assertEqual(
            calls_at_refusal("optimize llm none"), {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 9: 1},
        )
        self.assertEqual(sweep.positions("optimize heuristics", O1), {3, 5, 7, 9})


if __name__ == "__main__":
    unittest.main()
