r"""MCP stdio proxy.

Sits between an agent and an MCP server process. Forwards the bytes each
side writes to the other, unchanged and in order, and traces the JSON-RPC
messages in them: tool calls, results and errors become trace events.

MCP over stdio sends one JSON-RPC message or batch per line. Some servers
frame each message with Content-Length headers instead, like LSP:

    Content-Length: 46\r\n
    \r\n
    {"jsonrpc":"2.0","id":1,"method":"tools/list"}

The proxy:
1. Spawns the MCP server as a subprocess
2. Reads from agent stdin, forwards to server stdin
3. Reads from server stdout, forwards to agent stdout
4. Traces a copy of each line or Content-Length body

Tracing never stops the relay or changes what it forwards. A message the
proxy cannot trace is still forwarded, and the next one is traced as usual.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from typing import IO, Any, Callable

from .event_data import list_field, number_field, object_field, str_field
from .models import EventType, SessionMeta, TraceEvent
from .masking import MaskingConfig, mask_event_data
from .pairing import TOOL_RESPONSE_TYPES, KeyedPairer, link_response
from .store import TraceStore


# Cap on tool calls waiting for a response. A call whose response never
# arrives (dropped stream, notification-style misuse) would otherwise stay
# forever.
MAX_PENDING_CALLS = 10_000


_CONTENT_LENGTH = "Content-Length:"

# Largest piece of a Content-Length body read at once. BufferedReader.read(n)
# allocates n bytes before it reads any, so a huge length is never read in one.
_BODY_PIECE = 64 * 1024


def _content_length(line: bytes) -> int | None:
    """The body length a Content-Length line gives. None for any other line.

    The value must be ASCII digits that int() converts. int() also takes a
    sign, underscores and other digits, such as "+5", "5_0" and a fullwidth
    5 (U+FF15), and raises ValueError on more digits than
    sys.get_int_max_str_digits().
    """
    if _CONTENT_LENGTH.encode() not in line:
        return None
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith(_CONTENT_LENGTH):
        return None
    value = text[len(_CONTENT_LENGTH):].strip()
    if not (value.isascii() and value.isdigit()):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_frame(source: IO[bytes]) -> tuple[bytes, bytes]:
    """Read one frame. Returns the frame and the copy of it to trace.

    A frame is a line, up to and including its LF. If the line is a
    Content-Length line, the frame also holds the header lines after it, up
    to and including the first blank one, and then that many body bytes.
    The copy to trace is the line, or the body.

    At the end of input the frame is what arrived of it, and the copy is
    what arrived of the body. The frame is empty once source has run out.
    """
    line = source.readline()
    length = _content_length(line)
    if length is None:
        return line, line
    head = [line]
    while True:
        header = source.readline()
        if not header:
            return b"".join(head), b""
        head.append(header)
        if not header.strip():
            break
    pieces = []
    while length:
        piece = source.read(min(length, _BODY_PIECE))
        if not piece:
            break
        pieces.append(piece)
        length -= len(piece)
    body = b"".join(pieces)
    return b"".join(head) + body, body


def _rpc_id(data: dict[str, Any]) -> str | int | float | None:
    """Return the ``request_id`` that links a tool call to its result.

    An id that is not a string or number reads as missing, so the call and
    its result are recorded without a link.
    """
    return str_field(data, "request_id", number_field(data, "request_id", None))


def _classify_message(msg: Any, direction: str) -> TraceEvent | None:
    """Turn a JSON-RPC message into a trace event.

    Never raises. A value that is not a JSON object gives no event, and a
    field with the wrong JSON type reads as missing. The id is recorded as
    it arrived; only a string or number id links a response to its call.
    """
    if not isinstance(msg, Mapping):
        return None
    method = msg.get("method", "")
    msg_id = msg.get("id")
    result = object_field(msg, "result", None)
    error = msg.get("error")

    # Request from agent to server
    if direction == "agent_to_server":
        params = object_field(msg, "params", {})
        if method == "tools/call":
            return TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={
                    "tool_name": str_field(params, "name", "unknown"),
                    "arguments": object_field(params, "arguments", {}),
                    "request_id": msg_id,
                },
            )
        elif method == "resources/read":
            return TraceEvent(
                event_type=EventType.FILE_READ,
                data={
                    "uri": str_field(params, "uri", ""),
                    "request_id": msg_id,
                },
            )
        elif method == "completion/create" or method == "sampling/createMessage":
            messages = list_field(params, "messages", [])
            last = messages[-1] if messages else None
            return TraceEvent(
                event_type=EventType.LLM_REQUEST,
                data={
                    "method": method,
                    "model": str_field(params, "model", ""),
                    "message_count": len(messages),
                    "request_id": msg_id,
                    # capture last message role + truncated content for context
                    "last_message": _truncate_message(last) if isinstance(last, Mapping) else None,
                },
            )

    # Response from server to agent
    if direction == "server_to_agent":
        # Any error but null means the request failed, even 0 or {}.
        # JSON-RPC 1.0 sends "error": null beside a result.
        if error is not None:
            if isinstance(error, Mapping):
                code = number_field(error, "code", None)
                message = str_field(error, "message", "")
            else:
                code = None
                message = error if isinstance(error, str) else ""
            return TraceEvent(
                event_type=EventType.ERROR,
                data={
                    "code": code,
                    "message": message,
                    "request_id": msg_id,
                },
            )
        if result and msg_id is not None:
            # tool results contain content array
            content = list_field(result, "content", [])
            # MCP marks a failed tool call with "isError": true. Only JSON
            # true counts, as in the jsonl import.
            failed = result.get("isError") is True
            if content or failed:
                texts = [str_field(c, "text", "") for c in content if isinstance(c, Mapping)]
                data = {
                    "request_id": msg_id,
                    "content_types": [
                        str_field(c, "type", "unknown") if isinstance(c, Mapping) else "unknown"
                        for c in content
                    ],
                    "content_preview": _truncate("\n".join(t for t in texts if t), 200),
                }
                if failed:
                    data["is_error"] = True
                return TraceEvent(event_type=EventType.TOOL_RESULT, data=data)

    # Notifications (no id)
    if direction == "server_to_agent" and method:
        if method == "notifications/resources/updated":
            params = object_field(msg, "params", {})
            return TraceEvent(
                event_type=EventType.FILE_WRITE,
                data={
                    "uri": str_field(params, "uri", ""),
                },
            )

    return None


def _truncate(s: str, max_len: int = 200) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _truncate_message(msg: Mapping[str, Any]) -> dict:
    content = msg.get("content", "")
    if isinstance(content, Mapping):
        text = str_field(content, "text", "")
    else:
        try:
            text = str(content)[:200]
        except (RecursionError, MemoryError):
            # A list nested deeper than the recursion limit allows.
            text = ""
    return {
        "role": str_field(msg, "role", ""),
        "content_preview": _truncate(text, 200),
    }


def _rpc_messages(value: Any) -> list[Any]:
    """Return the messages in a parsed JSON value.

    A JSON-RPC batch is an array of messages. Any other value is one message.
    """
    return value if isinstance(value, list) else [value]


def _write_stderr(text: str) -> None:
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass


def _note_untraced(reason: str, note: Callable[[str], None] = _write_stderr) -> None:
    """Say on stderr that a message was forwarded without being traced.

    ``reason`` is an exception type. The message itself is never printed.
    The line is written with ``note``.
    """
    try:
        note(f"agent-strace: could not trace a message ({reason}). It was still forwarded.\n")
    except Exception:
        pass


def _trace_message(
    msg: Any,
    direction: str,
    calls: KeyedPairer[TraceEvent],
    emit: Callable[[TraceEvent], None],
    note: Callable[[str], None] = _write_stderr,
) -> None:
    """Trace one JSON-RPC message and link a response to its call.

    Never raises, so tracing cannot stop the relay. If tracing fails, for
    example on arguments nested too deep to store, the event is dropped and
    a note goes to stderr. The caller forwards the message either way.
    """
    try:
        event = _classify_message(msg, direction)
        if event is None:
            return
        # Read before emit, which can redact the id.
        key = _rpc_id(event.data)
        if event.event_type in TOOL_RESPONSE_TYPES:
            call_event = calls.match(key)
            if call_event is not None:
                link_response(event, call_event.event_id, call_event.timestamp)
        emit(event)
        # Kept only once emitted, so no response links to a call the trace lacks.
        if event.event_type == EventType.TOOL_CALL:
            calls.add(key, event)
    except Exception as exc:
        _note_untraced(type(exc).__name__, note)


def _save_without_parent(
    store: TraceStore, meta: SessionMeta, refusal: ValueError,
    note: Callable[[str], None] = _write_stderr,
) -> bool:
    """Save meta without the parent link that the store refused.

    Writes one line with ``note`` when the link is dropped. Returns False, with
    meta unchanged, if there is no link or the save fails anyway.
    """
    from .inert import inert

    parent_session_id = meta.parent_session_id
    parent_event_id = meta.parent_event_id
    if not parent_session_id:
        return False
    meta.parent_session_id = meta.parent_event_id = ""
    try:
        store.update_meta(meta)
    except (ValueError, OSError):
        meta.parent_session_id = parent_session_id
        meta.parent_event_id = parent_event_id
        return False
    note(
        f"agent-strace: dropped the link to parent session "
        f"{inert(parent_session_id)}: {inert(refusal)}\n"
    )
    return True


class MCPProxy:
    """Proxies stdio between agent and MCP server, capturing traces."""

    def __init__(
        self,
        server_command: list[str],
        store: TraceStore,
        session_meta: SessionMeta,
        on_event: Callable[[TraceEvent], None] | None = None,
        redact: bool = False,
        masking_config: MaskingConfig | None = None,
        summary: Callable[[SessionMeta], str] | None = None,
    ):
        self.server_command = server_command
        self.store = store
        self.meta = session_meta
        self.on_event = on_event
        self.redact = redact
        self.masking_config = masking_config
        self._calls: KeyedPairer[TraceEvent] = KeyedPairer(MAX_PENDING_CALLS)
        # Relay threads emit concurrently. Once the session end is written
        # the trace is sealed: nothing is appended and no counter changes.
        self._emit_lock = threading.Lock()
        self._sealed = False
        # One is held for each write to the agent's stdout, the other for
        # each write to its stderr. The session end is written only while
        # holding both, so no write to the agent starts or finishes after it.
        self._out_gate = threading.Lock()
        self._err_gate = threading.Lock()
        self._agent_writers: tuple[Any, ...] = ()
        # Names of the agent streams whose writes were cancelled.
        self._cancelled: list[str] = []
        # The thread holding both gates to end the session.
        self._ending: int | None = None
        # Text record writes to the agent's stderr just before session_end.
        self.summary = summary

    def _emit(self, event: TraceEvent, final: bool = False) -> None:
        with self._emit_lock:
            if self._sealed:
                return
            if not final and self._ending not in (None, threading.get_ident()):
                # The session is ending: its counters are being saved.
                return
            event.session_id = self.meta.session_id
            if self.redact or self.masking_config:
                event.data = mask_event_data(
                    event.data,
                    config=self.masking_config,
                    redact_secrets=self.redact,
                )
            if final:
                self._sealed = True
            self.store.append_event(self.meta.session_id, event)

            # update counters
            if event.event_type == EventType.TOOL_CALL:
                self.meta.tool_calls += 1
            elif event.event_type == EventType.LLM_REQUEST:
                self.meta.llm_requests += 1
            elif event.event_type == EventType.ERROR:
                self.meta.errors += 1

        # Outside the lock: on_event may write to the agent through notice(),
        # which waits for a gate that _end_session holds while it emits.
        if self.on_event:
            self.on_event(event)

    def _forward_and_trace(
        self,
        source: IO[bytes],
        dest: IO[bytes],
        direction: str,
        stop_event: threading.Event,
        gate: Any = None,
    ) -> bool:
        """Forward each frame read from source to dest, and trace a copy of it.

        A frame is a line or a Content-Length message (see _read_frame). It
        is written to dest as it was read, in one write. Tracing never stops
        the relay or changes what it forwards: a frame that cannot be traced
        is still forwarded. Sets ``stop_event`` when the relay ends. Returns
        True when source ran out, False when dest could not take a frame.
        Nothing is forwarded once the session has ended. With a gate, each
        frame is checked against the session end, traced and written while
        holding it, so no trace event or write of the relay comes after the
        session end. A frame can be traced and then not written in full,
        when its write fails or is cancelled.
        """
        while not stop_event.is_set():
            frame, copy = _read_frame(source)
            if not frame:
                stop_event.set()
                return True

            with gate if gate is not None else _NO_GATE:
                if self._sealed:
                    stop_event.set()
                    return True

                self._trace_copy(copy, direction)

                try:
                    dest.write(frame)
                    dest.flush()
                except (OSError, ValueError):
                    stop_event.set()
                    return False
        return True

    def _trace_copy(self, copy: bytes, direction: str) -> None:
        """Trace the JSON-RPC message, or each message of a batch, in a copy.

        Never raises. A blank copy is not traced. A copy that is not JSON, or
        that json.loads gives up on, gets one note on the agent's stderr.
        """
        try:
            text = copy.decode("utf-8", errors="replace").strip()
            if not text:
                return
            value = json.loads(text)
        except Exception as exc:
            _note_untraced(type(exc).__name__, self.notice)
            return
        for msg in _rpc_messages(value):
            _trace_message(msg, direction, self._calls, self._emit, self.notice)

    def _end_session(self, returncode: int, output_truncated: bool = False) -> int:
        """Record the session end, seal the trace and save the final metadata."""
        # Wait for each write to the agent that has started to end.
        self._hold_agent_gates()
        with self._emit_lock:
            # From here only this thread records events, so the counters
            # saved below are final.
            self._ending = threading.get_ident()
        try:
            self.meta.ended_at = time.time()
            self.meta.total_duration_ms = (
                (self.meta.ended_at - self.meta.started_at) * 1000
            )
            data: dict[str, Any] = {
                "exit_code": returncode,
                "duration_ms": self.meta.total_duration_ms,
            }
            reported: list[str] = []
            try:
                self._report_cancelled(reported)
                # record's last lines are written before session_end, so
                # nothing reaches the agent after it. They can be cancelled
                # too, so cancellations are counted again after them.
                self._save_meta(self._held_notice)
                if self.summary is not None:
                    self._held_notice(self.summary(self.meta))
                if self._report_cancelled(reported):
                    self._save_meta(lambda text: None)
            finally:
                if reported or output_truncated:
                    data["output_truncated"] = True
                # The metadata is final: session_end changes no counter.
                self._emit(TraceEvent(event_type=EventType.SESSION_END, data=data), final=True)
        finally:
            with self._emit_lock:
                self._ending = None
            self._err_gate.release()
            self._out_gate.release()
        return returncode

    def _report_cancelled(self, reported: list[str]) -> bool:
        """Record an error for each agent stream cancelled since the last call.

        Returns True if there was one.
        """
        new = [name for name in self._cancelled if name not in reported]
        for name in new:
            reported.append(name)
            self._emit(TraceEvent(
                event_type=EventType.ERROR,
                data={"message": (
                    f"a write to the agent's {name} did not finish within "
                    f"{STOP_TIMEOUT:g}s and was cancelled"
                )},
            ))
        return bool(new)

    def _save_meta(self, note: Callable[[str], None]) -> None:
        # A parent the store refuses now, such as one deleted while the
        # proxy ran, must not cost the end time and counters.
        try:
            self.store.update_meta(self.meta)
        except ValueError as exc:
            if not _save_without_parent(self.store, self.meta, exc, note):
                raise

    def _hold_agent_gates(self) -> None:
        """Take both agent gates, once no write to the agent is running.

        A write the agent does not take within STOP_TIMEOUT seconds is
        cancelled, and a gate is taken only once the write holding it has
        returned. A writer that cannot be cancelled is waited for.
        """
        for gate in (self._out_gate, self._err_gate):
            timeout = STOP_TIMEOUT
            while not gate.acquire(timeout=timeout):
                self._cancel_busy_writes()
                timeout = 0.1

    def _cancel_writes(self, index: int) -> None:
        """Cancel every write to one agent stream (0 stdout, 1 stderr)."""
        if index >= len(self._agent_writers):
            return
        cancel = getattr(self._agent_writers[index], "cancel", None)
        if cancel is None:
            return
        name = ("stdout", "stderr")[index]
        if name not in self._cancelled:
            self._cancelled.append(name)
        cancel()

    def _cancel_busy_writes(self) -> None:
        """Cancel writes to each agent stream that has a write running."""
        for index, writer in enumerate(self._agent_writers):
            busy = getattr(writer, "busy", None)
            if busy is not None and busy():
                self._cancel_writes(index)

    def notice(self, text: str) -> None:
        """Write a line of record's own to the agent's stderr.

        Like the relay's writes, it is written before session_end or not at
        all, and one the agent does not take within STOP_TIMEOUT seconds is
        cancelled, with every later write to the agent.
        """
        if self._ending == threading.get_ident():
            # Ending the session: its gates are held here already.
            return
        if not self._agent_writers:
            _write_stderr(text)
            return
        with self._err_gate:
            if not self._sealed:
                self._write_held(text.encode("utf-8", "replace"))

    def _held_notice(self, text: str) -> None:
        """notice() for the thread that holds the err gate."""
        if self._agent_writers:
            self._write_held(text.encode("utf-8", "replace"))
        else:
            _write_stderr(text)

    def _write_held(self, data: bytes) -> None:
        """Write to the agent's stderr. The err gate must be held."""
        if "stderr" in self._cancelled:
            return
        timer = threading.Timer(STOP_TIMEOUT, self._cancel_writes, (1,))
        timer.daemon = True
        timer.start()
        try:
            self._agent_writers[1].write(data)
            self._agent_writers[1].flush()
        except (OSError, ValueError):
            pass
        finally:
            timer.cancel()

    def _start_failed(self, exc: Exception) -> int:
        """Report a server that could not start and end the session."""
        if isinstance(exc, FileNotFoundError):
            returncode = 127
        elif isinstance(exc, PermissionError):
            returncode = 126
        else:
            returncode = 1
        message = f"could not start the MCP server: {exc}"
        _write_stderr(f"agent-strace: {message}\n")
        self._emit(TraceEvent(event_type=EventType.ERROR, data={"message": message}))
        return self._end_session(returncode)

    def run(self, prepare: Callable[[], None] | None = None) -> int:
        """Start the proxy. Blocks until the server exits.

        prepare, if given, runs once the stop signal handlers are in, before
        the session starts: a stop signal from then on ends the session.

        The session ends when the server exits, or when the proxy stops the
        server because the agent stopped reading or record got SIGTERM or
        SIGINT. When the agent closes its side of stdin, the server's stdin
        is closed too, so the server sees the end of input. Server output is
        relayed until the server closes it, before the session end is
        recorded. Stopping the server stops the processes it started: every
        one on Windows, and on POSIX those _TreeTracker finds.
        """
        stop_server = threading.Event()
        signals: list[int] = []

        def on_signal(signum: int, frame: Any) -> None:
            signals.append(signum)
            stop_server.set()

        # The handlers go in before the server starts, so a stop signal at
        # any point after it starts reaches on_signal.
        previous_handlers = _handle_stop_signals(on_signal)
        try:
            if prepare is not None:
                prepare()
            returncode = self._run(stop_server, signals)
        finally:
            _restore_signal_handlers(previous_handlers)
        # Like a shell, report the signal that stopped record.
        if signals:
            return 128 + signals[0]
        return returncode

    def _run(self, stop_server: threading.Event, signals: list[int]) -> int:
        # emit session start
        self._emit(
            TraceEvent(
                event_type=EventType.SESSION_START,
                data={"command": self.server_command},
            )
        )
        if signals:
            return self._end_session(-signals[0])

        try:
            server = _Server(self.server_command)
        except (OSError, ValueError) as exc:
            return self._start_failed(exc)

        # Until the server has exited, any way out of here stops it and
        # every process it started, and ends the session.
        try:
            return self._relay(server, stop_server)
        finally:
            try:
                try:
                    server.close()
                finally:
                    if not self._sealed:
                        try:
                            self._end_session(server.returncode)
                        except Exception:
                            pass
            finally:
                for writer in reversed(self._agent_writers):
                    restore = getattr(writer, "restore", None)
                    if restore is not None:
                        restore()

    def _relay(self, server: _Server, stop_server: threading.Event) -> int:
        proc = server.proc
        agent_in, agent_out, agent_err = _agent_streams()
        self._agent_writers = (agent_out, agent_err)

        # agent stdin -> server stdin. When the agent's input ends, or the
        # server stops taking it, the server's stdin is closed so the server
        # sees the end of input.
        def forward_in() -> None:
            self._forward_and_trace(agent_in, proc.stdin, "agent_to_server", threading.Event())
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass

        # server stdout -> agent stdout, until the server closes it.
        def forward_out() -> None:
            if self._forward_and_trace(
                proc.stdout, agent_out, "server_to_agent", threading.Event(),
                gate=self._out_gate,
            ):
                _close(proc.stdout)
            else:
                stop_server.set()

        # server stderr -> agent stderr (passthrough, no tracing). Read to the
        # end even if the agent's stderr is gone, so the server never blocks.
        def forward_stderr() -> None:
            writable = True
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                if not writable:
                    continue
                with self._err_gate:
                    if self._sealed:
                        writable = False
                        continue
                    try:
                        agent_err.write(line)
                        agent_err.flush()
                    except (OSError, ValueError):
                        writable = False
            _close(proc.stderr)

        t_in = threading.Thread(target=forward_in, daemon=True)
        t_out = threading.Thread(target=forward_out, daemon=True)
        t_err = threading.Thread(target=forward_stderr, daemon=True)
        t_in.start()
        t_out.start()
        t_err.start()

        returncode = _wait_for_server(server, stop_server)

        # A server that exited has closed its output, unless a process it
        # started still holds it open. Those processes are stopped after
        # DRAIN_TIMEOUT, and the output they held is recorded as cut short.
        relays = (t_out, t_err)
        truncated = not _join_all(relays, DRAIN_TIMEOUT)
        if truncated:
            server.stop()
            if not _join_all(relays, DRAIN_TIMEOUT):
                # What is left is a write the agent is not taking.
                self._cancel_busy_writes()
            message = (
                f"server output was cut short: it was not all relayed "
                f"{DRAIN_TIMEOUT:g}s after the server exited"
            )
            self.notice(f"agent-strace: {message}\n")
            self._emit(TraceEvent(event_type=EventType.ERROR, data={"message": message}))

        if not t_out.is_alive():
            _close(proc.stdout)

        # Processes the server started and left running are stopped too.
        server.stop_leftovers()

        # The session end seals the trace: a relay still running after this
        # neither records nor forwards anything more.
        return self._end_session(returncode, output_truncated=truncated)


# Seconds to wait for server output after the server exits, and for a server
# to exit after it is asked to stop.
DRAIN_TIMEOUT = 5.0
STOP_TIMEOUT = 5.0


# Signals that stop record. The server is stopped and the session ended.
STOP_SIGNALS = ("SIGTERM", "SIGINT")


def _handle_stop_signals(handler: Callable[[int, Any], None]) -> dict[int, Any]:
    """Install handler for the stop signals. Returns the handlers it replaced.

    A signal record was started with ignored stays ignored. An ignored
    SIGCHLD is set back to the default, so the server is not reaped before
    record has stopped what it started (see _Server). Signal handlers can
    only be set in the main thread. Elsewhere nothing is installed.
    """
    previous: dict[int, Any] = {}
    if threading.current_thread() is not threading.main_thread():
        return previous
    sigchld = getattr(signal, "SIGCHLD", None)
    if sigchld is not None:
        try:
            if signal.getsignal(sigchld) == signal.SIG_IGN:
                previous[sigchld] = signal.signal(sigchld, signal.SIG_DFL)
        except (OSError, ValueError):
            pass
    for name in STOP_SIGNALS:
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            if signal.getsignal(signum) == signal.SIG_IGN:
                continue
            previous[signum] = signal.signal(signum, handler)
        except (OSError, ValueError):
            pass
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (OSError, TypeError, ValueError):
            pass


_NO_GATE = contextlib.nullcontext()


def _close(stream: IO[bytes]) -> None:
    try:
        stream.close()
    except (OSError, ValueError):
        pass


def _join_all(threads: tuple[threading.Thread, ...], timeout: float) -> bool:
    """Join threads within timeout seconds in all. True if all ended."""
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return not any(thread.is_alive() for thread in threads)


def _wait_for_server(server: _Server, stop_server: threading.Event) -> int:
    """Wait for the server to exit, or stop it once stop_server is set."""
    while True:
        try:
            return server.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            if stop_server.is_set():
                return server.stop()


class _Server:
    """The MCP server process and the processes it starts.

    On POSIX the server leads a new process group and session, and a
    _TreeTracker finds the processes it started, including those that left
    the group. The server is not reaped until close(), so while they are
    stopped its PID and process group ID name no other process. Its exit is
    seen with waitid(WNOWAIT), or a kqueue on macOS before Python 3.13. If
    the server is reaped sooner (it exits before the kqueue watches it, or
    SIGCHLD is ignored), its process group is signalled only while it runs. On Windows
    the server starts suspended and runs only once it is in a job object,
    which kills every process in it when record closes the job or exits. A
    server that cannot be put in a job is not run.
    """

    def __init__(self, command: list[str]):
        posix = os.name == "posix"
        self._job = None
        self.returncode: int | None = None
        self._tree: _TreeTracker | None = _TreeTracker() if posix else None
        self._kqueue: Any = None
        # True while the server's PID, and so its process group ID, can name
        # no other process: it runs, or is a zombie record has not reaped.
        self._auto_reaped = posix and _sigchld_ignored()
        self._held = posix and not self._auto_reaped
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=posix,
            creationflags=0 if posix else _CREATE_SUSPENDED,
            env=self._tree.environment() if self._tree is not None else None,
        )
        if self._tree is not None:
            self._tree.start(self.proc.pid)
            if _WAITID is None:
                self._watch_exit()
            return
        try:
            self._job = _windows_job_and_resume(self.proc.pid)
        except OSError:
            self.proc.kill()
            self.proc.wait()
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                _close(stream)
            raise

    def _watch_exit(self) -> None:
        """Watch for the server's exit without reaping it, where there is no waitid."""
        if sys.platform != "darwin" or not hasattr(select, "kqueue"):
            return
        kq = select.kqueue()
        try:
            kq.control([select.kevent(
                self.proc.pid, filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                fflags=select.KQ_NOTE_EXIT | _NOTE_EXITSTATUS,
            )], 0, 0)
        except OSError:
            # The server has exited already. A kqueue cannot watch a zombie.
            kq.close()
            return
        self._kqueue = kq

    def poll(self) -> int | None:
        """The server's exit code, or None while it runs.

        On POSIX an exited server is left a zombie, unreaped, until close(),
        where that can be done.
        """
        if self.returncode is not None:
            return self.returncode
        if self._tree is None:
            self.returncode = self.proc.poll()
            return self.returncode
        if self._held and _WAITID is not None:
            try:
                info = _WAITID(
                    os.P_PID, self.proc.pid,  # type: ignore[attr-defined]
                    os.WEXITED | os.WNOHANG | os.WNOWAIT,  # type: ignore[attr-defined]
                )
            except ChildProcessError:
                # Reaped elsewhere, so its exit code is lost.
                self._held = False
                self.returncode = 255
                return self.returncode
            if info is not None:
                exited = info.si_code == os.CLD_EXITED  # type: ignore[attr-defined]
                self.returncode = info.si_status if exited else -info.si_status
            return self.returncode
        if self._held and self._kqueue is not None:
            try:
                events = self._kqueue.control(None, 1, 0)
            except OSError:
                events = []
            if events:
                self.returncode = os.waitstatus_to_exitcode(events[0].data)
                self._kqueue.close()
                self._kqueue = None
            return self.returncode
        # Nothing watches the server without reaping it.
        returncode = self.proc.poll()
        if returncode is not None:
            self._held = False
            # With SIGCHLD ignored the system reaps the server, and Popen
            # reports 0 for an exit code it never saw.
            self.returncode = 255 if self._auto_reaped else returncode
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the server to exit. Returns its exit code.

        Raises subprocess.TimeoutExpired after timeout seconds.
        """
        if self._tree is None:
            self.returncode = self.proc.wait(timeout=timeout)
            return self.returncode
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            returncode = self.poll()
            if returncode is not None:
                return returncode
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.proc.args, timeout or 0)
            time.sleep(0.01)

    def _signal(self, kill: bool) -> None:
        if self._tree is not None:
            sig = signal.SIGKILL if kill else signal.SIGTERM  # type: ignore[attr-defined]
            running = self.poll() is None
            # The group is the server's while the server is held, or still
            # runs: a running process's PID names no other process.
            if self._held or (self._auto_reaped and running):
                try:
                    os.killpg(self.proc.pid, sig)  # type: ignore[attr-defined]
                except OSError:
                    pass
            self._tree.signal(sig)
            return
        # Windows has no polite stop: terminate the job.
        if self._job is not None:
            _windows_terminate_job(self._job)
        try:
            self.proc.kill()
        except OSError:
            pass

    def tree_alive(self) -> bool:
        """True while a process the server started is found running (POSIX)."""
        return self._tree is not None and bool(self._tree.alive())

    def stop(self) -> int:
        """Stop the server and the processes it started. Returns its exit code.

        They get STOP_TIMEOUT seconds after SIGTERM to exit before they are
        killed. The server may already have exited.
        """
        deadline = time.monotonic() + STOP_TIMEOUT
        self._signal(kill=False)
        try:
            returncode = self.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._signal(kill=True)
            returncode = self.wait()
        while self.tree_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        # A killed process's children can start more processes while they
        # are found, so go round until none is left.
        for _ in range(20):
            if not self.tree_alive():
                break
            self._signal(kill=True)
            time.sleep(0.1)
        return returncode

    def stop_leftovers(self) -> None:
        """Stop the server, if running, and the processes it left running."""
        if self.poll() is None or self.tree_alive():
            self.stop()
        elif self._job is not None:
            _windows_terminate_job(self._job)

    def close(self) -> None:
        """Stop what the server left running, then reap it and release the job."""
        try:
            self.stop_leftovers()
        finally:
            if self._tree is not None:
                self._tree.close()
                if self._kqueue is not None:
                    self._kqueue.close()
                    self._kqueue = None
                self._held = False
                try:
                    self.proc.wait(timeout=STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            if self._job is not None:
                _windows_close(self._job)
                self._job = None


_WAITID = getattr(os, "waitid", None)
# macOS: a NOTE_EXIT event carries the wait status.
_NOTE_EXITSTATUS = 0x04000000


def _sigchld_ignored() -> bool:
    sigchld = getattr(signal, "SIGCHLD", None)
    try:
        return sigchld is not None and signal.getsignal(sigchld) == signal.SIG_IGN
    except (OSError, ValueError):
        return False


# Set in the server's environment to a token of its run, and inherited by
# every process it starts that does not clear its environment.
TREE_ENV = "AGENT_STRACE_RECORD_TREE"

# Seconds between looks at the process table while the server runs.
TREE_POLL_INTERVAL = 1.0


class _TreeTracker:
    """Finds the processes the server started, on POSIX.

    A process is taken as the server's when any of these holds:

    - It descends from the server, or from a process found earlier, in the
      process table. The table is read every TREE_POLL_INTERVAL seconds, so
      a process seen once is still found after it leaves its parent.
    - It is in the server's process group, while the server is a root.
    - Its environment holds this run's TREE_ENV token, even after the
      server exits.

    Each process is kept with its start time (see process_identity), and
    is only signalled while its PID still has that start time. The server
    is a root only while its PID has the start time it had at start().
    """

    def __init__(self) -> None:
        import uuid

        self.token = uuid.uuid4().hex
        self._pid = 0
        self._identity: str | None = None
        self._found: dict[int, str] = {}
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def environment(self) -> dict[str, str]:
        return {**os.environ, TREE_ENV: self.token}

    def start(self, pid: int) -> None:
        from .process_identity import process_identity

        self._pid = pid
        self._identity = process_identity(pid)
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while not self._done.wait(TREE_POLL_INTERVAL):
            self._scan(full=False)

    def _scan(self, full: bool = True) -> dict[int, str]:
        """Add what the process table shows. Returns the found processes alive.

        A full scan also looks for the server's process group and the token.
        """
        from .process_identity import ProcessListError, descendants, list_processes

        try:
            table = list_processes()
        except ProcessListError:
            table = {}

        def same(pid: int, identity: str | None) -> bool:
            return pid in table and identity is not None and table[pid].identity == identity

        with self._lock:
            roots = [pid for pid, identity in self._found.items() if same(pid, identity)]
        if same(self._pid, self._identity):
            roots.append(self._pid)
        found: dict[int, str] = {}
        for root in roots:
            # -1 is no process, so none of record's own is left out of the walk.
            found.update(descendants(table, root, -1))
        if full:
            # The token is random, so it names this run's processes even
            # after the server exits. Its PID, and so its process group, is
            # only matched while the server still has its start time.
            pgid = self._pid if same(self._pid, self._identity) else None
            for pid in _tree_members(TREE_ENV, self.token, pgid):
                if pid in table:
                    found[pid] = table[pid].identity
        me = os.getpid()
        with self._lock:
            for pid, identity in found.items():
                if pid not in (me, self._pid):
                    self._found[pid] = identity
            return {pid: identity for pid, identity in self._found.items() if same(pid, identity)}

    def alive(self) -> dict[int, str]:
        return self._scan()

    def signal(self, sig: int) -> None:
        for pid in self._scan():
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    def close(self) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join(5)


def _tree_members(name: str, token: str, pgid: int | None) -> list[int]:
    """PIDs in process group pgid, if given, or whose environment holds name=token.

    Only the environments of processes of the same user can be read.
    """
    marker = f"{name}={token}".encode()
    found = []
    if os.path.exists("/proc/self/environ"):
        try:
            names = os.listdir("/proc")
        except OSError:
            return []
        for entry in names:
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "rb") as f:
                    raw = f.read()
                # The command name is in parentheses and may hold anything.
                if pgid is not None and int(raw[raw.rindex(b")") + 2:].split()[2]) == pgid:
                    found.append(int(entry))
                    continue
                with open(f"/proc/{entry}/environ", "rb") as f:
                    if marker in f.read().split(b"\0"):
                        found.append(int(entry))
            except (OSError, ValueError, IndexError):
                continue
        return found
    # ps -E appends the environment to the command. The token is random, so
    # it names this run wherever it appears as a whole word.
    try:
        result = subprocess.run(
            ["ps", "-A", "-E", "-ww", "-o", "pid=,pgid=,command="],
            capture_output=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    for line in result.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 2 or not (fields[0].isdigit() and fields[1].isdigit()):
            continue
        if (pgid is not None and int(fields[1]) == pgid) or (
            len(fields) == 3 and marker in fields[2].split()
        ):
            found.append(int(fields[0]))
    return found


_CREATE_SUSPENDED = 0x00000004
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


def _windows_job_and_resume(pid: int) -> Any:
    """Put the suspended process pid in a new job object, then resume it.

    The job kills every process in it when its last handle is closed, so
    they also stop if record dies. Returns the job. Raises OSError if any
    step fails; the process is then still suspended.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError as exc:
        raise OSError(f"could not put the server in a job object: {exc}") from exc

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    ntdll = ctypes.WinDLL("ntdll")  # type: ignore[attr-defined]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    ntdll.NtResumeProcess.restype = ctypes.c_long
    ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)

    def failed(step: str) -> OSError:
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        return OSError(f"could not {step} (Windows error {error})")

    # PROCESS_SUSPEND_RESUME | PROCESS_SET_QUOTA | PROCESS_TERMINATE
    handle = kernel32.OpenProcess(0x0800 | 0x0100 | 0x0001, False, pid)
    if not handle:
        raise failed("open the server process")
    job = None
    try:
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise failed("create a job object for the server")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits), ctypes.sizeof(limits),
        ):
            raise failed("set up the server's job object")
        if not kernel32.AssignProcessToJobObject(job, handle):
            raise failed("put the server in a job object")
        status = ntdll.NtResumeProcess(handle)
        if status < 0:
            raise OSError(
                f"could not resume the server process (NTSTATUS {status & 0xFFFFFFFF:#x})"
            )
    except BaseException:
        if job:
            kernel32.CloseHandle(job)
        raise
    finally:
        kernel32.CloseHandle(handle)
    return job


def _windows_terminate_job(job: Any) -> bool:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        return bool(kernel32.TerminateJobObject(job, 1))
    except (AttributeError, ImportError, OSError):
        return False


def _windows_close(job: Any) -> None:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle(job)
    except (AttributeError, ImportError, OSError):
        pass


class _WriteCancelled(OSError):
    """A write to the agent was cancelled because the session is ending."""


class _FdWriter:
    """Writes straight to a file descriptor, with no buffer.

    Nothing is left buffered for interpreter shutdown to flush after the
    agent closes the pipe, and no lock is held that shutdown would wait on.

    A write can be cancelled from another thread, even one waiting because
    the agent is not reading: cancel() returns once the write has returned,
    and every later write fails. On POSIX a descriptor that is not a
    regular file is made nonblocking until restore(), so no system call of a
    write waits: one that finds no room fails with EAGAIN, and the write
    waits in select() for a short while before it checks for cancellation
    and tries again. On Windows a waiting write is cancelled with
    CancelSynchronousIo.
    """

    def __init__(self, fd: int):
        self._fd = fd
        self._state = threading.Lock()
        self._cancelled = False
        self._writer: int | None = None
        self._was_blocking: bool | None = None
        if os.name == "posix":
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode) and os.get_blocking(fd):
                    os.set_blocking(fd, False)
                    self._was_blocking = True
            except OSError:
                pass

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        with self._state:
            if self._cancelled:
                raise _WriteCancelled("the write to the agent was cancelled")
            self._writer = threading.get_native_id()
        try:
            while view:
                with self._state:
                    if self._cancelled:
                        raise _WriteCancelled("the write to the agent was cancelled")
                try:
                    written = os.write(self._fd, view)
                except BlockingIOError:
                    select.select([], [self._fd], [], 0.05)
                    continue
                except InterruptedError:
                    continue
                except OSError:
                    with self._state:
                        if self._cancelled:
                            raise _WriteCancelled("the write to the agent was cancelled") from None
                    raise
                view = view[written:]
        finally:
            with self._state:
                self._writer = None

    def flush(self) -> None:
        pass

    def busy(self) -> bool:
        """True while a write is running."""
        with self._state:
            return self._writer is not None

    def cancel(self) -> None:
        """Fail every write from now on. Returns once none is running.

        Gives up after STOP_TIMEOUT seconds only if a Windows write cannot
        be cancelled.
        """
        deadline = time.monotonic() + STOP_TIMEOUT
        while True:
            with self._state:
                self._cancelled = True
                writer = self._writer
                # Under the lock, so the writer is still in this write and
                # no other call of its is cancelled.
                if writer is not None and os.name == "nt":
                    _windows_cancel_io(writer)
            if writer is None or (os.name == "nt" and time.monotonic() > deadline):
                return
            time.sleep(0.01)

    def restore(self) -> None:
        """Make the descriptor blocking again if it was."""
        if self._was_blocking:
            try:
                os.set_blocking(self._fd, True)
            except OSError:
                pass
            self._was_blocking = None


class _FdReader:
    """Reads lines and blocks from a file descriptor (POSIX).

    Works whether or not the descriptor is nonblocking: the agent's stdin
    can share its open file with the stdout that _FdWriter makes
    nonblocking, as a terminal or a socket does.
    """

    def __init__(self, fd: int):
        self._fd = fd
        self._buffer = bytearray()
        self._eof = False

    def _fill(self) -> bool:
        """Read more into the buffer. False at the end of input."""
        while not self._eof:
            try:
                chunk = os.read(self._fd, 65536)
            except BlockingIOError:
                select.select([self._fd], [], [])
                continue
            except InterruptedError:
                continue
            if not chunk:
                self._eof = True
                return False
            self._buffer += chunk
            return True
        return False

    def readline(self) -> bytes:
        start = 0
        while True:
            end = self._buffer.find(b"\n", start)
            if end >= 0:
                return self._take(end + 1)
            start = len(self._buffer)
            if not self._fill():
                return self._take(len(self._buffer))

    def read(self, size: int) -> bytes:
        while len(self._buffer) < size and self._fill():
            pass
        return self._take(min(size, len(self._buffer)))

    def _take(self, size: int) -> bytes:
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def close(self) -> None:
        pass


def _windows_cancel_io(thread_id: int) -> None:
    """Cancel the synchronous I/O the thread thread_id is blocked in."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenThread.restype = wintypes.HANDLE
        kernel32.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.CancelSynchronousIo.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        # THREAD_TERMINATE is the access CancelSynchronousIo needs.
        handle = kernel32.OpenThread(0x0001, False, thread_id)
        if handle:
            kernel32.CancelSynchronousIo(handle)
            kernel32.CloseHandle(handle)
    except (AttributeError, ImportError, OSError):
        pass


def _agent_streams() -> tuple[IO[bytes], Any, Any]:
    """Return the streams the proxy reads from and writes to the agent.

    These are private to the proxy, not sys.stdin and sys.stdout. A thread
    still blocked reading the agent's input when the session ends would
    otherwise hold sys.stdin's lock, and Python aborts at shutdown when it
    cannot take that lock. On POSIX, stdout and stderr are nonblocking until
    their writers' restore().
    """
    try:
        if os.name == "posix":
            agent_in: IO[bytes] = _FdReader(sys.stdin.fileno())  # type: ignore[assignment]
        else:
            agent_in = open(sys.stdin.fileno(), "rb", closefd=False)
    except (AttributeError, OSError, ValueError):
        agent_in = sys.stdin.buffer
    streams: list[Any] = []
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
            streams.append(_FdWriter(stream.fileno()))
        except (AttributeError, OSError, ValueError):
            streams.append(stream.buffer)
    return agent_in, streams[0], streams[1]
