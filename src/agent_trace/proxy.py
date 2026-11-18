"""MCP stdio proxy.

Sits between an agent and an MCP server process. Forwards all JSON-RPC
messages in both directions while capturing tool calls, results, and
errors as trace events.

MCP uses JSON-RPC 2.0 over stdio. Messages are framed with
Content-Length headers (like LSP).

    Content-Length: 42\r\n
    \r\n
    {"jsonrpc":"2.0","method":"tools/call",...}

The proxy:
1. Spawns the MCP server as a subprocess
2. Reads from agent stdin, forwards to server stdin
3. Reads from server stdout, forwards to agent stdout
4. Captures every message and emits trace events

Tracing never stops the relay. A message the proxy cannot trace is still
forwarded, and the next one is traced as usual.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from typing import IO, Any, Callable, NamedTuple

from .event_data import list_field, number_field, object_field, str_field
from .models import EventType, SessionMeta, TraceEvent
from .masking import MaskingConfig, mask_event_data
from .pairing import TOOL_RESPONSE_TYPES, KeyedPairer, link_response
from .store import TraceStore


# Cap on tool calls waiting for a response. A call whose response never
# arrives (dropped stream, notification-style misuse) would otherwise stay
# forever.
MAX_PENDING_CALLS = 10_000


class _Message(NamedTuple):
    """One message read from a stdio stream.

    ``text`` is the message as it arrived and ``value`` its parsed JSON.
    ``error`` names the exception that kept json.loads from reading the
    text, and is empty when it was read.
    """

    text: str
    value: Any = None
    error: str = ""


def _parse_message(text: str) -> _Message | None:
    """Parse the text of a message. None if the text is not JSON.

    JSON that json.loads gives up on, such as a value nested deeper than the
    recursion limit allows or an integer too long to convert, gives a
    message with ``error`` set, so it can still be forwarded.
    """
    try:
        return _Message(text, json.loads(text))
    except json.JSONDecodeError:
        return None
    except Exception as exc:
        return _Message(text, error=type(exc).__name__)


def _read_message(stream: IO[bytes]) -> _Message | None:
    """Read a single JSON-RPC message. None at the end of the stream.

    Supports two framing modes:
    1. Content-Length headers (LSP-style, used by some MCP servers)
    2. Newline-delimited JSON (MCP spec: messages delimited by newlines)

    Auto-detects the mode from the first bytes. A newline-delimited line
    that is not JSON is skipped. A Content-Length body that is not JSON
    ends the stream.
    """
    while True:
        line = stream.readline()
        if not line:
            return None

        line_str = line.decode("utf-8", errors="replace").strip()
        if not line_str:
            continue

        # Content-Length framing
        if line_str.startswith("Content-Length:"):
            content_length = int(line_str.split(":")[1].strip())
            # consume the blank line after headers
            while True:
                header_line = stream.readline()
                if not header_line:
                    return None
                if header_line.strip() == b"":
                    break
            body = stream.read(content_length)
            if not body:
                return None
            return _parse_message(body.decode("utf-8"))

        # Newline-delimited JSON
        if line_str.startswith("{"):
            message = _parse_message(line_str)
            if message is not None:
                return message


def _message_text(message: _Message) -> str:
    """Return the text to forward for a message.

    The message goes out as JSON, like it was read. One that json cannot
    read or write goes out as the text it arrived as, on one line. In JSON a
    line break can only be whitespace, so a space replaces it.
    """
    if not message.error:
        try:
            return json.dumps(message.value)
        except Exception:
            pass
    return message.text.replace("\r", " ").replace("\n", " ")


def _write_message(stream: IO[bytes], message: _Message, use_content_length: bool = False) -> None:
    """Write a JSON-RPC message.

    Uses newline-delimited JSON by default (MCP spec).
    Set use_content_length=True for LSP-style framing.
    """
    body = _message_text(message).encode("utf-8")
    if use_content_length:
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        stream.write(header + body)
    else:
        stream.write(body + b"\n")
    stream.flush()


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
            if content:
                first = content[0]
                return TraceEvent(
                    event_type=EventType.TOOL_RESULT,
                    data={
                        "request_id": msg_id,
                        "content_types": [
                            str_field(c, "type", "unknown") if isinstance(c, Mapping) else "unknown"
                            for c in content
                        ],
                        "content_preview": _truncate(
                            str_field(first, "text", "") if isinstance(first, Mapping) else "", 200
                        ),
                    },
                )

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


def _note_untraced(reason: str) -> None:
    """Say on stderr that a message was forwarded without being traced.

    ``reason`` is an exception type. The message itself is never printed.
    """
    try:
        sys.stderr.write(
            f"agent-strace: could not trace a message ({reason}). It was still forwarded.\n"
        )
        sys.stderr.flush()
    except Exception:
        pass


def _trace_message(
    msg: Any,
    direction: str,
    calls: KeyedPairer[TraceEvent],
    emit: Callable[[TraceEvent], None],
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
        _note_untraced(type(exc).__name__)


def _save_without_parent(
    store: TraceStore, meta: SessionMeta, refusal: ValueError
) -> bool:
    """Save meta without the parent link that the store refused.

    Writes one line to stderr when the link is dropped. Returns False, with
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
    sys.stderr.write(
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

    def _emit(self, event: TraceEvent, final: bool = False) -> None:
        with self._emit_lock:
            if self._sealed:
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

            if self.on_event:
                self.on_event(event)

    def _forward_and_trace(
        self,
        source: IO[bytes],
        dest: IO[bytes],
        direction: str,
        stop_event: threading.Event,
    ) -> bool:
        """Read messages from source, trace them, forward to dest.

        Tracing never stops the relay: a message that cannot be traced is
        still forwarded. Sets ``stop_event`` when the relay ends. Returns
        True when source ran out, False when dest could not take a message.
        Nothing is forwarded once the session has ended.
        """
        while not stop_event.is_set():
            message = _read_message(source)
            if message is None:
                stop_event.set()
                return True

            # trace the message, or each message of a batch
            if message.error:
                _note_untraced(message.error)
            else:
                for msg in _rpc_messages(message.value):
                    _trace_message(msg, direction, self._calls, self._emit)

            # forward the message
            if self._sealed:
                stop_event.set()
                return True
            try:
                _write_message(dest, message)
            except (OSError, ValueError):
                stop_event.set()
                return False
        return True

    def _end_session(self, returncode: int, output_truncated: bool = False) -> int:
        """Record the session end, seal the trace and save the final metadata."""
        self.meta.ended_at = time.time()
        self.meta.total_duration_ms = (self.meta.ended_at - self.meta.started_at) * 1000

        data: dict[str, Any] = {
            "exit_code": returncode,
            "duration_ms": self.meta.total_duration_ms,
        }
        if output_truncated:
            data["output_truncated"] = True
        self._emit(TraceEvent(event_type=EventType.SESSION_END, data=data), final=True)

        # A parent the store refuses now, such as one deleted while the
        # proxy ran, must not cost the end time and counters.
        try:
            self.store.update_meta(self.meta)
        except ValueError as exc:
            if not _save_without_parent(self.store, self.meta, exc):
                raise
        return returncode

    def _start_failed(self, exc: Exception) -> int:
        """Report a server that could not start and end the session."""
        if isinstance(exc, FileNotFoundError):
            returncode = 127
        elif isinstance(exc, PermissionError):
            returncode = 126
        else:
            returncode = 1
        message = f"could not start the MCP server: {exc}"
        try:
            sys.stderr.write(f"agent-strace: {message}\n")
            sys.stderr.flush()
        except Exception:
            pass
        self._emit(TraceEvent(event_type=EventType.ERROR, data={"message": message}))
        return self._end_session(returncode)

    def run(self) -> int:
        """Start the proxy. Blocks until the server exits.

        The session ends when the server exits, or when the proxy stops the
        server because the agent stopped reading or record got SIGTERM or
        SIGINT. When the agent closes its side of stdin, the server's stdin
        is closed too, so the server sees the end of input. Server output is
        relayed until the server closes it, before the session end is
        recorded. Stopping the server stops every process it started.
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
            if server.proc.poll() is None:
                server.stop()
            server.close()
            if not self._sealed:
                try:
                    self._end_session(server.proc.returncode)
                except Exception:
                    pass

    def _relay(self, server: _Server, stop_server: threading.Event) -> int:
        proc = server.proc
        agent_in, agent_out, agent_err = _agent_streams()

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
                proc.stdout, agent_out, "server_to_agent", threading.Event()
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
                if writable and not self._sealed:
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
            _join_all(relays, DRAIN_TIMEOUT)
            message = (
                f"server output was cut short: a process the server started "
                f"still held it open {DRAIN_TIMEOUT:g}s after the server exited"
            )
            try:
                agent_err.write(f"agent-strace: {message}\n".encode())
            except (OSError, ValueError):
                pass
            self._emit(TraceEvent(event_type=EventType.ERROR, data={"message": message}))

        if not t_out.is_alive():
            _close(proc.stdout)

        # The session end seals the trace: a relay still running after this
        # (one held by a process that left the server's process group)
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

    A signal record was started with ignored stays ignored. Signal handlers
    can only be set in the main thread. Elsewhere nothing is installed.
    """
    previous: dict[int, Any] = {}
    if threading.current_thread() is not threading.main_thread():
        return previous
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
            return server.proc.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            if stop_server.is_set():
                return server.stop()


class _Server:
    """The MCP server process and every process it starts.

    On POSIX the server leads a new process group. On Windows it is put in
    a job object. Stopping the server signals the whole group or job, so a
    process the server started is not left running.
    """

    def __init__(self, command: list[str]):
        posix = os.name == "posix"
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=posix,
            # On Windows the server starts suspended and runs only once it
            # is in the job, so every process it starts is in the job too.
            creationflags=0 if posix else _CREATE_SUSPENDED,
        )
        self._job = None
        if not posix:
            try:
                self._job = _windows_job_and_resume(self.proc.pid)
            except OSError:
                self.proc.kill()
                self.proc.wait()
                for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                    _close(stream)
                raise

    def _signal(self, kill: bool) -> None:
        if os.name == "posix":
            sig = signal.SIGKILL if kill else signal.SIGTERM  # type: ignore[attr-defined]
            try:
                os.killpg(self.proc.pid, sig)  # type: ignore[attr-defined]
            except OSError:
                pass
            return
        # Windows has no polite stop: terminate the job, or the server alone
        # if it could not be put in one.
        if self._job is not None and _windows_terminate_job(self._job):
            return
        try:
            self.proc.kill()
        except OSError:
            pass

    def _group_alive(self) -> bool:
        if os.name != "posix":
            return False
        try:
            os.killpg(self.proc.pid, 0)  # type: ignore[attr-defined]
        except OSError:
            return False
        return True

    def stop(self) -> int:
        """Stop the server and the processes it started. Returns its exit code.

        They get STOP_TIMEOUT seconds after SIGTERM to exit before they are
        killed.
        """
        deadline = time.monotonic() + STOP_TIMEOUT
        self._signal(kill=False)
        try:
            returncode = self.proc.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._signal(kill=True)
            return self.proc.wait()
        while self._group_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._group_alive():
            self._signal(kill=True)
        return returncode

    def close(self) -> None:
        if self._job is not None:
            _windows_close(self._job)
            self._job = None


_CREATE_SUSPENDED = 0x00000004


def _windows_job_and_resume(pid: int) -> Any:
    """Put the suspended process pid in a new job object, then resume it.

    Returns the job, or None if the process could not be put in one. Raises
    OSError if the process could not be resumed.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    ntdll = ctypes.WinDLL("ntdll")  # type: ignore[attr-defined]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    ntdll.NtResumeProcess.restype = ctypes.c_long
    ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)

    # PROCESS_SUSPEND_RESUME | PROCESS_SET_QUOTA | PROCESS_TERMINATE
    handle = kernel32.OpenProcess(0x0800 | 0x0100 | 0x0001, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
    try:
        job = kernel32.CreateJobObjectW(None, None)
        if job and not kernel32.AssignProcessToJobObject(job, handle):
            kernel32.CloseHandle(job)
            job = None
        status = ntdll.NtResumeProcess(handle)
        if status < 0:
            if job:
                kernel32.CloseHandle(job)
            raise OSError(f"could not resume the server process (NTSTATUS {status & 0xFFFFFFFF:#x})")
        return job or None
    finally:
        kernel32.CloseHandle(handle)


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


class _FdWriter:
    """Writes straight to a file descriptor, with no buffer.

    Nothing is left buffered for interpreter shutdown to flush after the
    agent closes the pipe, and no lock is held that shutdown would wait on.
    """

    def __init__(self, fd: int):
        self._raw = io.FileIO(fd, "wb", closefd=False)

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = self._raw.write(view)
            if written is None:
                raise BlockingIOError("write would block")
            view = view[written:]

    def flush(self) -> None:
        pass


def _agent_streams() -> tuple[IO[bytes], Any, Any]:
    """Return the streams the proxy reads from and writes to the agent.

    These are private to the proxy, not sys.stdin and sys.stdout. A thread
    still blocked reading the agent's input when the session ends would
    otherwise hold sys.stdin's lock, and Python aborts at shutdown when it
    cannot take that lock.
    """
    try:
        agent_in: IO[bytes] = open(sys.stdin.fileno(), "rb", closefd=False)
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
