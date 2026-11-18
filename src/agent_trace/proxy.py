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

import json
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

    def _emit(self, event: TraceEvent) -> None:
        event.session_id = self.meta.session_id
        if self.redact or self.masking_config:
            event.data = mask_event_data(
                event.data,
                config=self.masking_config,
                redact_secrets=self.redact,
            )
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
    ) -> None:
        """Read messages from source, trace them, forward to dest.

        Tracing never stops the relay: a message that cannot be traced is
        still forwarded.
        """
        while not stop_event.is_set():
            message = _read_message(source)
            if message is None:
                stop_event.set()
                break

            # trace the message, or each message of a batch
            if message.error:
                _note_untraced(message.error)
            else:
                for msg in _rpc_messages(message.value):
                    _trace_message(msg, direction, self._calls, self._emit)

            # forward the message
            try:
                _write_message(dest, message)
            except (BrokenPipeError, OSError):
                stop_event.set()
                break

    def run(self) -> int:
        """Start the proxy. Blocks until the server exits."""
        # emit session start
        self._emit(
            TraceEvent(
                event_type=EventType.SESSION_START,
                data={"command": self.server_command},
            )
        )

        proc = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stop = threading.Event()

        # agent stdin -> server stdin
        t_in = threading.Thread(
            target=self._forward_and_trace,
            args=(sys.stdin.buffer, proc.stdin, "agent_to_server", stop),
            daemon=True,
        )

        # server stdout -> agent stdout
        t_out = threading.Thread(
            target=self._forward_and_trace,
            args=(proc.stdout, sys.stdout.buffer, "server_to_agent", stop),
            daemon=True,
        )

        # server stderr -> agent stderr (passthrough, no tracing)
        def forward_stderr():
            while not stop.is_set():
                line = proc.stderr.readline()
                if not line:
                    break
                sys.stderr.buffer.write(line)
                sys.stderr.buffer.flush()

        t_err = threading.Thread(target=forward_stderr, daemon=True)

        t_in.start()
        t_out.start()
        t_err.start()

        # wait for server to exit
        returncode = proc.wait()
        stop.set()

        # finalize session
        self.meta.ended_at = time.time()
        self.meta.total_duration_ms = (self.meta.ended_at - self.meta.started_at) * 1000

        self._emit(
            TraceEvent(
                event_type=EventType.SESSION_END,
                data={
                    "exit_code": returncode,
                    "duration_ms": self.meta.total_duration_ms,
                },
            )
        )

        self.store.update_meta(self.meta)
        return returncode
