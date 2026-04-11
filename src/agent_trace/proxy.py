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
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from typing import IO, Any, Callable

from .models import EventType, SessionMeta, TraceEvent
from .masking import MaskingConfig, mask_event_data
from .store import TraceStore


def _read_message(stream: IO[bytes]) -> dict[str, Any] | None:
    """Read a single JSON-RPC message.

    Supports two framing modes:
    1. Content-Length headers (LSP-style, used by some MCP servers)
    2. Newline-delimited JSON (MCP spec: messages delimited by newlines)

    Auto-detects the mode from the first bytes.
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
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return None

        # Newline-delimited JSON
        if line_str.startswith("{"):
            try:
                return json.loads(line_str)
            except json.JSONDecodeError:
                continue


def _write_message(stream: IO[bytes], msg: dict[str, Any], use_content_length: bool = False) -> None:
    """Write a JSON-RPC message.

    Uses newline-delimited JSON by default (MCP spec).
    Set use_content_length=True for LSP-style framing.
    """
    body = json.dumps(msg).encode("utf-8")
    if use_content_length:
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        stream.write(header + body)
    else:
        stream.write(body + b"\n")
    stream.flush()


def _classify_message(msg: dict[str, Any], direction: str) -> TraceEvent | None:
    """Turn a JSON-RPC message into a trace event."""
    method = msg.get("method", "")
    msg_id = msg.get("id")
    result = msg.get("result")
    error = msg.get("error")

    # Request from agent to server
    if direction == "agent_to_server":
        if method == "tools/call":
            params = msg.get("params", {})
            return TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={
                    "tool_name": params.get("name", "unknown"),
                    "arguments": params.get("arguments", {}),
                    "request_id": msg_id,
                },
            )
        elif method == "resources/read":
            params = msg.get("params", {})
            return TraceEvent(
                event_type=EventType.FILE_READ,
                data={
                    "uri": params.get("uri", ""),
                    "request_id": msg_id,
                },
            )
        elif method == "completion/create" or method == "sampling/createMessage":
            params = msg.get("params", {})
            messages = params.get("messages", [])
            return TraceEvent(
                event_type=EventType.LLM_REQUEST,
                data={
                    "method": method,
                    "model": params.get("model", ""),
                    "message_count": len(messages),
                    "request_id": msg_id,
                    # capture last message role + truncated content for context
                    "last_message": _truncate_message(messages[-1]) if messages else None,
                },
            )

    # Response from server to agent
    if direction == "server_to_agent":
        if error:
            return TraceEvent(
                event_type=EventType.ERROR,
                data={
                    "code": error.get("code"),
                    "message": error.get("message", ""),
                    "request_id": msg_id,
                },
            )
        if result and msg_id is not None:
            # tool results contain content array
            content = result.get("content", [])
            if content:
                return TraceEvent(
                    event_type=EventType.TOOL_RESULT,
                    data={
                        "request_id": msg_id,
                        "content_types": [c.get("type", "unknown") for c in content],
                        "content_preview": _truncate(
                            content[0].get("text", "") if content else "", 200
                        ),
                    },
                )

    # Notifications (no id)
    if direction == "server_to_agent" and method:
        if method == "notifications/resources/updated":
            params = msg.get("params", {})
            return TraceEvent(
                event_type=EventType.FILE_WRITE,
                data={
                    "uri": params.get("uri", ""),
                },
            )

    return None


def _truncate(s: str, max_len: int = 200) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _truncate_message(msg: dict) -> dict:
    return {
        "role": msg.get("role", ""),
        "content_preview": _truncate(
            msg.get("content", {}).get("text", "")
            if isinstance(msg.get("content"), dict)
            else str(msg.get("content", ""))[:200],
            200,
        ),
    }


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
        self._pending_calls: dict[Any, TraceEvent] = {}

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
        """Read messages from source, trace them, forward to dest."""
        while not stop_event.is_set():
            msg = _read_message(source)
            if msg is None:
                stop_event.set()
                break

            # trace the message
            event = _classify_message(msg, direction)
            if event:
                # link tool_result to tool_call
                if event.event_type == EventType.TOOL_RESULT:
                    req_id = event.data.get("request_id")
                    if req_id in self._pending_calls:
                        call_event = self._pending_calls.pop(req_id)
                        event.parent_id = call_event.event_id
                        event.duration_ms = (event.timestamp - call_event.timestamp) * 1000
                elif event.event_type == EventType.TOOL_CALL:
                    req_id = event.data.get("request_id")
                    if req_id is not None:
                        self._pending_calls[req_id] = event

                self._emit(event)

            # forward the message
            try:
                _write_message(dest, msg)
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
