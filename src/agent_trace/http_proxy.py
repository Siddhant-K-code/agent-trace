"""HTTP/SSE MCP proxy.

Proxies HTTP-based MCP servers. The agent connects to the proxy on
a local port. The proxy forwards requests to the remote MCP server,
captures tool calls and results, and streams SSE responses back.

MCP over HTTP/SSE uses:
  - POST /message for JSON-RPC requests (agent -> server)
  - GET /sse for server-sent events (server -> agent)

The proxy:
1. Listens on a local port
2. Forwards POST /message to the remote server
3. Forwards GET /sse from the remote server
4. Captures every JSON-RPC message as a trace event
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable
from urllib.parse import urlparse

from .models import EventType, SessionMeta, TraceEvent
from .proxy import _classify_message
from .masking import MaskingConfig, mask_event_data
from .store import TraceStore


class _ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies to a remote MCP server."""

    # set by HTTPProxyServer before serving
    remote_url: str = ""
    store: TraceStore | None = None
    meta: SessionMeta | None = None
    on_event: Callable[[TraceEvent], None] | None = None
    redact: bool = False
    masking_config: MaskingConfig | None = None
    pending_calls: dict[Any, TraceEvent] = {}

    def _emit(self, event: TraceEvent) -> None:
        if self.meta:
            event.session_id = self.meta.session_id
        if self.redact or self.masking_config:
            event.data = mask_event_data(
                event.data,
                config=self.masking_config,
                redact_secrets=self.redact,
            )
        if self.store and self.meta:
            self.store.append_event(self.meta.session_id, event)

        if event.event_type == EventType.TOOL_CALL and self.meta:
            self.meta.tool_calls += 1
        elif event.event_type == EventType.LLM_REQUEST and self.meta:
            self.meta.llm_requests += 1
        elif event.event_type == EventType.ERROR and self.meta:
            self.meta.errors += 1

        if self.on_event:
            self.on_event(event)

    def _get_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        parsed = urlparse(self.remote_url)
        if parsed.scheme == "https":
            return http.client.HTTPSConnection(parsed.hostname, parsed.port or 443)
        return http.client.HTTPConnection(parsed.hostname, parsed.port or 80)

    def _remote_path(self, path: str) -> str:
        parsed = urlparse(self.remote_url)
        base = parsed.path.rstrip("/")
        return f"{base}{path}"

    def do_POST(self):
        """Forward POST requests (agent -> server) and trace them."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # trace the request
        try:
            msg = json.loads(body.decode("utf-8"))
            event = _classify_message(msg, "agent_to_server")
            if event:
                if event.event_type == EventType.TOOL_CALL:
                    req_id = event.data.get("request_id")
                    if req_id is not None:
                        self.pending_calls[req_id] = event
                self._emit(event)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # forward to remote
        conn = self._get_connection()
        headers = {
            "Content-Type": self.headers.get("Content-Type", "application/json"),
            "Content-Length": str(len(body)),
        }
        # forward auth headers
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth

        try:
            conn.request("POST", self._remote_path(self.path), body=body, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read()

            # trace the response
            try:
                resp_msg = json.loads(resp_body.decode("utf-8"))
                resp_event = _classify_message(resp_msg, "server_to_agent")
                if resp_event:
                    if resp_event.event_type == EventType.TOOL_RESULT:
                        req_id = resp_event.data.get("request_id")
                        if req_id in self.pending_calls:
                            call_event = self.pending_calls.pop(req_id)
                            resp_event.parent_id = call_event.event_id
                            resp_event.duration_ms = (resp_event.timestamp - call_event.timestamp) * 1000
                    self._emit(resp_event)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            # send response to agent
            self.send_response(resp.status)
            for key, value in resp.getheaders():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(resp_body)

        except Exception as e:
            self._emit(TraceEvent(
                event_type=EventType.ERROR,
                data={"message": f"Proxy error: {e}"},
            ))
            self.send_error(502, f"Proxy error: {e}")
        finally:
            conn.close()

    def do_GET(self):
        """Forward GET requests (SSE stream from server -> agent)."""
        conn = self._get_connection()
        headers = {}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        accept = self.headers.get("Accept")
        if accept:
            headers["Accept"] = accept

        try:
            conn.request("GET", self._remote_path(self.path), headers=headers)
            resp = conn.getresponse()

            self.send_response(resp.status)
            for key, value in resp.getheaders():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, value)
            self.end_headers()

            # stream SSE events
            while True:
                line = resp.readline()
                if not line:
                    break

                # trace SSE data lines
                line_str = line.decode("utf-8", errors="replace")
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str:
                        try:
                            msg = json.loads(data_str)
                            event = _classify_message(msg, "server_to_agent")
                            if event:
                                if event.event_type == EventType.TOOL_RESULT:
                                    req_id = event.data.get("request_id")
                                    if req_id in self.pending_calls:
                                        call_event = self.pending_calls.pop(req_id)
                                        event.parent_id = call_event.event_id
                                        event.duration_ms = (event.timestamp - call_event.timestamp) * 1000
                                self._emit(event)
                        except json.JSONDecodeError:
                            pass

                self.wfile.write(line)
                self.wfile.flush()

        except Exception as e:
            self._emit(TraceEvent(
                event_type=EventType.ERROR,
                data={"message": f"SSE proxy error: {e}"},
            ))
            self.send_error(502, f"SSE proxy error: {e}")
        finally:
            conn.close()

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


class HTTPProxyServer:
    """HTTP/SSE proxy for remote MCP servers."""

    def __init__(
        self,
        remote_url: str,
        local_port: int,
        store: TraceStore,
        session_meta: SessionMeta,
        on_event: Callable[[TraceEvent], None] | None = None,
        redact: bool = False,
    ):
        self.remote_url = remote_url.rstrip("/")
        self.local_port = local_port
        self.store = store
        self.meta = session_meta
        self.on_event = on_event
        self.redact = redact

    def run(self) -> None:
        """Start the proxy server. Blocks until interrupted."""
        # configure the handler class
        _ProxyHandler.remote_url = self.remote_url
        _ProxyHandler.store = self.store
        _ProxyHandler.meta = self.meta
        _ProxyHandler.on_event = self.on_event
        _ProxyHandler.redact = self.redact
        _ProxyHandler.pending_calls = {}

        self.store.append_event(
            self.meta.session_id,
            TraceEvent(
                event_type=EventType.SESSION_START,
                session_id=self.meta.session_id,
                data={
                    "mode": "http",
                    "remote_url": self.remote_url,
                    "local_port": self.local_port,
                },
            ),
        )

        server = HTTPServer(("127.0.0.1", self.local_port), _ProxyHandler)
        sys.stderr.write(
            f"agent-strace: HTTP proxy listening on http://127.0.0.1:{self.local_port}\n"
            f"agent-strace: forwarding to {self.remote_url}\n"
        )

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            self.meta.ended_at = time.time()
            self.meta.total_duration_ms = (self.meta.ended_at - self.meta.started_at) * 1000

            self.store.append_event(
                self.meta.session_id,
                TraceEvent(
                    event_type=EventType.SESSION_END,
                    session_id=self.meta.session_id,
                    data={
                        "duration_ms": self.meta.total_duration_ms,
                    },
                ),
            )
            self.store.update_meta(self.meta)
