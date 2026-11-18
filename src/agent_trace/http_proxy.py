"""HTTP MCP proxy for Streamable HTTP and legacy SSE servers.

Proxies HTTP-based MCP servers. The agent connects to the proxy on
a local port. The proxy forwards requests to the remote MCP server,
captures tool calls and results, and streams replies back.

The remote URL is the server's endpoint, such as ``http://host/mcp`` for
Streamable HTTP or ``http://host/sse`` for legacy SSE. The agent connects
to the proxy root:

- A request for ``/`` goes to the endpoint path.
- Any other path goes to the same host unchanged, such as the
  ``/messages/`` path a legacy SSE server gives the agent.
- The query is the remote URL's query, then the agent's.

The proxy forwards:

- POST, with a JSON-RPC message. A JSON reply is read in full, then sent.
  An event stream reply is sent line by line as it arrives, so a server
  can report progress or ask the agent something while a tool runs.
- GET, for the server's event stream.
- DELETE, which ends a Streamable HTTP session.

Every JSON-RPC message in a POST body, a JSON reply or an event of an event
stream is traced. An event's data is read the way the HTML standard's event
stream rules say: its ``data`` lines, with or without a space after the
colon, joined with a newline, and the event ends at a blank line.

Requests are served on separate threads, so a long-lived stream does not
block other requests. Closing the server shuts down open streams and waits
for in-flight requests before the session end is recorded.

Pages on other sites cannot use the proxy. The proxy sends the upstream's
own host as Host and drops Origin, so the upstream's checks never see the
browser's values. The proxy checks them itself, on every method, before it
reads the body:

- A Host that is not localhost, an IP address or an ``--allow-host`` name
  gets 421. This stops DNS rebinding pages.
- An Origin that is not a page on localhost, a loopback address or an
  ``--allow-host`` name gets 403. This stops other sites.

A refused request is not traced and never reaches the upstream.
"""

from __future__ import annotations

import codecs
import contextlib
import http.client
import ipaddress
import json
import re
import socket
import ssl
import sys
import threading
import time
from collections.abc import Iterable
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable
from urllib.parse import urlparse, urlsplit

from .models import EventType, SessionMeta, TraceEvent
from .proxy import MAX_PENDING_CALLS, _classify_message, _rpc_id
from .masking import MaskingConfig, mask_event_data
from .pairing import TOOL_RESPONSE_TYPES, KeyedPairer, link_response
from .propagation import extract_traceparent, inject_traceparent
from .server import _host_header_name, _is_ip_literal
from .store import TraceStore


# How long server_close waits for in-flight requests after shutting down
# their sockets.
SHUTDOWN_TIMEOUT = 5.0

# Request headers each method forwards when the client sends them. POST
# also sends Content-Type, Content-Length and traceparent. Nothing else is
# forwarded: not Origin, Cookie or Accept-Encoding.
_POST_HEADERS = ("Authorization", "Accept", "Mcp-Session-Id", "Mcp-Protocol-Version")
_GET_HEADERS = (
    "Authorization", "Accept", "Mcp-Session-Id", "Mcp-Protocol-Version", "Last-Event-ID",
)
_DELETE_HEADERS = ("Authorization", "Mcp-Session-Id", "Mcp-Protocol-Version")


def _is_event_stream(content_type: str | None) -> bool:
    """Return True for a text/event-stream Content-Type, in any case."""
    media_type = (content_type or "").split(";", 1)[0]
    return media_type.strip().lower() == "text/event-stream"


# Ends a line of an event stream: CRLF, LF or a lone CR.
_EVENT_STREAM_EOL = re.compile(r"\r\n|\r|\n")


class _EventStreamParser:
    """Read the data of each event in a text/event-stream.

    Follows the event stream interpretation rules of the HTML standard. The
    stream is UTF-8, less one leading BOM. A line ends in CRLF, LF or a lone
    CR. A line that starts with a colon is a comment. A field's value is the
    text after the first colon, less one leading space. The values of an
    event's ``data`` fields are joined with a newline. A blank line ends the
    event; an event with no ``data`` field is dropped. Other fields, such as
    ``event`` and ``id``, are ignored. An event still open when the stream
    ends is dropped.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
        self._line = ""
        self._data: list[str] = []
        # The last chunk ended in CR, so an LF that starts the next one
        # belongs to that CRLF and ends no line.
        self._after_cr = False

    def feed(self, chunk: bytes) -> list[str]:
        """Read a chunk of the stream. Return the data of each event it ends."""
        text = self._decoder.decode(chunk)
        if not text:
            return []
        if self._after_cr and text.startswith("\n"):
            text = text[1:]
        self._after_cr = text.endswith("\r")
        *lines, self._line = _EVENT_STREAM_EOL.split(self._line + text)
        events = []
        for line in lines:
            if not line:
                if self._data:
                    events.append("\n".join(self._data))
                    self._data = []
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if field == "data":
                self._data.append(value[1:] if value.startswith(" ") else value)
        return events


def _refusal_body(message: str) -> bytes:
    return json.dumps({
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32600, "message": message},
    }).encode("utf-8")


# Replies and log lines for refused requests. They are constant so that no
# header value is ever echoed.
_HOST_REFUSED = _refusal_body(
    "Host not allowed: connect to localhost or an IP address, "
    "or start record-http with --allow-host NAME"
)
_ORIGIN_REFUSED = _refusal_body(
    "Origin not allowed: only pages on localhost, a loopback address "
    "or an --allow-host name can use this proxy"
)
_HOST_REFUSED_LOG = (
    "agent-strace: refused a request (421 Host not allowed). Connect to "
    "localhost or an IP address, or restart record-http with --allow-host NAME.\n"
)
_ORIGIN_REFUSED_LOG = (
    "agent-strace: refused a request (403 Origin not allowed). Only pages on "
    "localhost, a loopback address or an --allow-host name can use this proxy. "
    "To allow pages on another name, restart record-http with --allow-host NAME.\n"
)

# An Origin value: http or https, a host as in a Host value, then an
# optional port of digits. Nothing may follow, not even "/".
_ORIGIN_VALUE = re.compile(
    r"https?://(\[[^\]]*\]|[^\[\]:]*)(?::[0-9]+)?", re.IGNORECASE
)
_IPV4_LOOPBACK = ipaddress.IPv4Network("127.0.0.0/8")
_IPV6_LOOPBACK = ipaddress.IPv6Address("::1")


def _is_loopback_ip(name: str) -> bool:
    """Return True for an address in 127.0.0.0/8 or a bracketed ::1.

    ``is_loopback`` is not used: it also accepts ::ffff:127.0.0.1.
    """
    try:
        if name.startswith("[") and name.endswith("]"):
            return ipaddress.IPv6Address(name[1:-1]) == _IPV6_LOOPBACK
        return ipaddress.IPv4Address(name) in _IPV4_LOOPBACK
    except ValueError:
        return False


def _shutdown_socket(sock: socket.socket) -> None:
    """Wake any thread blocked on sock. Errors mean it is already closed."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def upstream_tls_context(remote_url: str, ca_file: str = "") -> ssl.SSLContext | None:
    """Build the TLS context for an HTTPS upstream, or None for plain HTTP.

    Certificates and host names are always verified. Without ``ca_file`` the
    context trusts the system CA store, as http.client does by default. With
    ``ca_file`` it trusts only that PEM bundle, which replaces the system
    store, the same as ``org-report --ca-file``.

    Raises ValueError if ``ca_file`` is given for a non-HTTPS URL or cannot
    be loaded.
    """
    if urlparse(remote_url).scheme != "https":
        if ca_file:
            raise ValueError("--ca-file requires an https:// URL")
        return None
    try:
        context = ssl.create_default_context(cafile=ca_file or None)
    except (OSError, ssl.SSLError) as exc:
        raise ValueError(f"could not load CA file {ca_file}: {exc}") from exc
    context.set_alpn_protocols(["http/1.1"])
    return context


class _UpstreamHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS upstream connection whose TLS handshake can be interrupted.

    http.client wraps and handshakes in one step, so a server that accepts
    TCP but never answers the handshake blocks connect() on a socket nobody
    can reach. This connection hands the TLS socket to ``watch`` before the
    handshake, so closing the proxy shuts it down.

    ``context`` comes from :func:`upstream_tls_context`, so certificates and
    host names are always verified.
    """

    def __init__(
        self,
        host: str,
        port: int,
        watch: Callable[[socket.socket], None],
        context: ssl.SSLContext,
    ):
        super().__init__(host, port, context=context)
        self._tls_context = context
        self._watch = watch

    def connect(self) -> None:
        http.client.HTTPConnection.connect(self)
        self.sock = self._tls_context.wrap_socket(
            self.sock, server_hostname=self.host, do_handshake_on_connect=False
        )
        self._watch(self.sock)
        self.sock.do_handshake()


class _ProxyHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that owns the proxy config and shared state.

    Each server gets its own :class:`KeyedPairer` of pending calls, so two
    proxies in one process (or a restarted proxy) never match results
    against each other's calls.

    Requests run on daemon threads. ``state_lock`` serialises everything
    they share: trace store writes, session meta updates and ``on_event``.
    """

    daemon_threads = True
    # The stdlib default backlog of 5 resets bursts of concurrent clients.
    request_queue_size = 128

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        remote_url: str,
        store: TraceStore | None = None,
        meta: SessionMeta | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
        redact: bool = False,
        masking_config: MaskingConfig | None = None,
        max_pending_calls: int = MAX_PENDING_CALLS,
        upstream_tls: ssl.SSLContext | None = None,
        allowed_hosts: Iterable[str] = (),
    ):
        if upstream_tls is None:
            upstream_tls = upstream_tls_context(remote_url)
        super().__init__(server_address, _ProxyHandler)
        self.remote_url = remote_url
        # Shared by every upstream HTTPS connection; None for plain HTTP.
        self.upstream_tls = upstream_tls
        # --allow-host names, lower case.
        self.allowed_hosts = frozenset(name.lower() for name in allowed_hosts)
        self.store = store
        self.meta = meta
        self.on_event = on_event
        self.redact = redact
        self.masking_config = masking_config
        self.max_pending_calls = max_pending_calls
        self.calls: KeyedPairer[TraceEvent] = KeyedPairer(max_pending_calls)
        self.state_lock = threading.Lock()
        # Guards the fields below and signals when requests finish.
        self._requests_cond = threading.Condition()
        self._active_requests = 0
        self._open_sockets: set[socket.socket] = set()
        self._closing = False
        # Set under state_lock once close_streams is done; later events are
        # dropped so nothing lands after the session end.
        self._closed = False

    @property
    def closing(self) -> bool:
        return self._closing

    @contextlib.contextmanager
    def request_scope(self, client_sock: socket.socket):
        """Track one request. Yields False if the server is closing."""
        with self._requests_cond:
            if self._closing:
                accepted = False
            else:
                accepted = True
                self._active_requests += 1
                self._open_sockets.add(client_sock)
        if not accepted:
            yield False
            return
        try:
            yield True
        finally:
            with self._requests_cond:
                self._open_sockets.discard(client_sock)
                self._active_requests -= 1
                self._requests_cond.notify_all()

    def watch_socket(self, sock: socket.socket) -> None:
        """Register an upstream socket so close_streams can interrupt it."""
        with self._requests_cond:
            if not self._closing:
                self._open_sockets.add(sock)
                return
        _shutdown_socket(sock)

    def unwatch_socket(self, sock: socket.socket) -> None:
        with self._requests_cond:
            self._open_sockets.discard(sock)

    def close_streams(self, timeout: float = SHUTDOWN_TIMEOUT) -> bool:
        """Shut down open request sockets and wait for handlers to finish.

        Returns True if every request finished within timeout.
        """
        with self._requests_cond:
            self._closing = True
            sockets = list(self._open_sockets)
            self._open_sockets.clear()
        for sock in sockets:
            _shutdown_socket(sock)
        with self._requests_cond:
            finished = self._requests_cond.wait_for(
                lambda: self._active_requests == 0, timeout
            )
        with self.state_lock:
            self._closed = True
        return finished

    def server_close(self) -> None:
        self.close_streams()
        super().server_close()

    def emit(self, event: TraceEvent) -> None:
        """Mask, store and report one event. Safe to call from any thread."""
        if self.meta:
            event.session_id = self.meta.session_id
        if self.redact or self.masking_config:
            event.data = mask_event_data(
                event.data,
                config=self.masking_config,
                redact_secrets=self.redact,
            )
        with self.state_lock:
            if self._closed:
                return
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

    @property
    def pending_calls(self) -> dict[Any, TraceEvent]:
        """A copy of the calls waiting for a response, by JSON-RPC id."""
        return self.calls.pending()

    def track_call(self, event: TraceEvent) -> None:
        """Remember a tool_call so its response can be linked later."""
        self.calls.add(_rpc_id(event.data), event)

    def match_result(self, event: TraceEvent) -> None:
        """Link a tool_result or error to its pending tool_call, if one exists."""
        call_event = self.calls.match(_rpc_id(event.data))
        if call_event is not None:
            link_response(event, call_event.event_id, call_event.timestamp)

    def clear_pending(self) -> int:
        """Drop unmatched calls. Returns how many were dropped."""
        return self.calls.clear()


class _ProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies to a remote MCP server.

    All config and state live on the owning _ProxyHTTPServer (self.server).
    """

    server: _ProxyHTTPServer

    @property
    def remote_url(self) -> str:
        return self.server.remote_url

    @property
    def store(self) -> TraceStore | None:
        return self.server.store

    @property
    def meta(self) -> SessionMeta | None:
        return self.server.meta

    @property
    def on_event(self) -> Callable[[TraceEvent], None] | None:
        return self.server.on_event

    @property
    def redact(self) -> bool:
        return self.server.redact

    @property
    def masking_config(self) -> MaskingConfig | None:
        return self.server.masking_config

    def _emit(self, event: TraceEvent) -> None:
        self.server.emit(event)

    def parse_request(self) -> bool:
        """Parse the request line and headers, then refuse web page requests.

        The stdlib calls this before it looks up ``do_<METHOD>``, so the check
        covers every method, including ones the proxy does not implement. It
        runs before the body is read, so a refused request is neither traced
        nor forwarded.
        """
        if not super().parse_request():
            return False
        if not self._host_allowed():
            self._refuse(421, _HOST_REFUSED, _HOST_REFUSED_LOG)
            return False
        if not self._origin_allowed():
            self._refuse(403, _ORIGIN_REFUSED, _ORIGIN_REFUSED_LOG)
            return False
        return True

    def _host_allowed(self) -> bool:
        """Return False for a Host that a DNS rebinding page could send.

        No Host is allowed. Otherwise there must be one, and it must name
        localhost, an IP address or an ``--allow-host`` name. The port is not
        checked.
        """
        values = self.headers.get_all("Host") or []
        if not values:
            return True
        if len(values) > 1:
            return False
        name = _host_header_name(values[0])
        if not name:
            return False
        return (
            name == "localhost"
            or name in self.server.allowed_hosts
            or _is_ip_literal(name)
        )

    def _origin_allowed(self) -> bool:
        """Return False for an Origin other than a page this machine serves.

        No Origin is allowed. Otherwise there must be one, of the form
        ``http(s)://host[:port]``, and the host must be localhost, a loopback
        address or an ``--allow-host`` name.
        """
        values = self.headers.get_all("Origin") or []
        if not values:
            return True
        if len(values) > 1:
            return False
        match = _ORIGIN_VALUE.fullmatch(values[0].strip(" \t"))
        if not match:
            return False
        name = match.group(1).lower()
        return (
            name == "localhost"
            or name in self.server.allowed_hosts
            or _is_loopback_ip(name)
        )

    def _refuse(self, status: int, body: bytes, log_line: str) -> None:
        """Log the refusal, send a constant JSON-RPC error and close.

        The request body is left unread. A large one can reach the client as
        a connection reset instead of this reply.
        """
        sys.stderr.write(log_line)
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass

    def _get_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        parsed = urlparse(self.remote_url)
        if parsed.scheme == "https":
            return _UpstreamHTTPSConnection(
                parsed.hostname,
                parsed.port or 443,
                self.server.watch_socket,
                self.server.upstream_tls,
            )
        return http.client.HTTPConnection(parsed.hostname, parsed.port or 80)

    def _connect(self, conn: http.client.HTTPConnection) -> socket.socket:
        """Open the upstream socket and register it for shutdown."""
        try:
            conn.connect()
        except BaseException:
            # A failed TLS handshake has already registered its socket.
            if conn.sock is not None:
                self.server.unwatch_socket(conn.sock)
            raise
        sock = conn.sock
        self.server.watch_socket(sock)
        return sock

    def _report_failure(self, label: str, exc: Exception, headers_sent: bool) -> None:
        # A stream cut by server shutdown is expected, not a proxy error.
        if self.server.closing:
            return
        self._emit(TraceEvent(
            event_type=EventType.ERROR,
            data={"message": f"{label}: {exc}"},
        ))
        if headers_sent:
            return
        try:
            self.send_error(502, f"{label}: {exc}")
        except OSError:
            pass

    def _remote_path(self, path: str) -> str:
        """Return the upstream request target for the client's target.

        A request for ``/`` goes to the remote URL's path, or to ``/`` if it
        has none. Any other path goes unchanged. The query is the remote
        URL's query, then the client's.
        """
        remote = urlsplit(self.remote_url)
        request = urlsplit(path)
        if request.path in ("", "/"):
            target = remote.path or "/"
        else:
            target = request.path
        query = "&".join(part for part in (remote.query, request.query) if part)
        return f"{target}?{query}" if query else target

    def _client_headers(self, names: Iterable[str]) -> dict[str, str]:
        """Return the named request headers the client sent, and no others."""
        headers = {}
        for name in names:
            value = self.headers.get(name)
            if value:
                headers[name] = value
        return headers

    def _relay_head(self, resp: http.client.HTTPResponse) -> None:
        """Send the upstream's status and headers, less hop-by-hop ones."""
        self.send_response(resp.status)
        for key, value in resp.getheaders():
            if key.lower() not in ("transfer-encoding", "connection"):
                self.send_header(key, value)
        self.end_headers()

    def _relay_events(self, resp: http.client.HTTPResponse) -> None:
        """Send an event stream to the client line by line as it arrives.

        The data of each event is one JSON-RPC message from the server. It
        is traced when the blank line that ends the event arrives, before
        that line is sent. Data that is not a JSON object, such as a legacy
        SSE server's ``endpoint`` event, is not traced. The bytes sent are
        the upstream's, unchanged.
        """
        parser = _EventStreamParser()
        while True:
            line = resp.readline()
            if not line:
                break

            for data in parser.feed(line):
                self._trace_event_data(data)

            self.wfile.write(line)
            self.wfile.flush()

    def _trace_event_data(self, data: str) -> None:
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        event = _classify_message(msg, "server_to_agent")
        if event:
            if event.event_type in TOOL_RESPONSE_TYPES:
                self.server.match_result(event)
            self._emit(event)

    def _link_sub_session(self, resp: http.client.HTTPResponse) -> None:
        """Link the downstream agent's session to this one.

        A downstream agent that runs agent-trace names its session in the
        reply's tracestate. Only the headers are read, so this works for a
        streamed reply before its body arrives.
        """
        resp_ctx = extract_traceparent(dict(resp.getheaders()))
        if resp_ctx and resp_ctx.get("at_session_id") and self.meta:
            child_sid = resp_ctx["at_session_id"]
            if child_sid != self.meta.session_id and self.store:
                from .a2a import link_sub_session
                with self.server.state_lock:
                    link_sub_session(
                        self.store,
                        parent_session_id=self.meta.session_id,
                        parent_event_id="",
                        child_session_id=child_sid,
                    )

    def do_POST(self):
        """Forward POST requests (agent -> server) and trace them."""
        with self.server.request_scope(self.connection) as accepted:
            if accepted:
                self._forward_post()

    def _forward_post(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Extract upstream traceparent so we can continue the trace chain
        inbound_ctx = extract_traceparent(dict(self.headers))
        upstream_trace_id = inbound_ctx["trace_id"] if inbound_ctx else ""

        # If the upstream agent carries an agent-trace session ID in tracestate,
        # record it as the parent session for cross-agent correlation. The
        # link is kept only if this store accepts it. After close_streams the
        # final save owns the meta, so no link is tried then.
        if inbound_ctx and inbound_ctx.get("at_session_id") and self.meta and self.store:
            with self.server.state_lock:
                if not self.meta.parent_session_id and not self.server._closed:
                    _link_parent(self.store, self.meta, inbound_ctx["at_session_id"])

        # trace the request
        try:
            msg = json.loads(body.decode("utf-8"))
            event = _classify_message(msg, "agent_to_server")
            if event:
                if event.event_type == EventType.TOOL_CALL:
                    self.server.track_call(event)
                self._emit(event)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # forward to remote
        conn = self._get_connection()
        headers = {
            "Content-Type": self.headers.get("Content-Type", "application/json"),
            "Content-Length": str(len(body)),
        }
        headers.update(self._client_headers(_POST_HEADERS))

        # Inject W3C traceparent so downstream agents can correlate back
        if self.meta:
            headers = inject_traceparent(
                headers,
                session_id=self.meta.session_id,
                trace_id=upstream_trace_id,
            )

        sock = None
        headers_sent = False
        try:
            sock = self._connect(conn)
            conn.request("POST", self._remote_path(self.path), body=body, headers=headers)
            resp = conn.getresponse()
            self._link_sub_session(resp)

            if _is_event_stream(resp.getheader("Content-Type")):
                # A Streamable HTTP server can hold this stream open while a
                # tool runs, to report progress or ask the agent something.
                self._relay_head(resp)
                headers_sent = True
                self._relay_events(resp)
                return

            resp_body = resp.read()

            # trace the response
            try:
                resp_msg = json.loads(resp_body.decode("utf-8"))
                resp_event = _classify_message(resp_msg, "server_to_agent")
                if resp_event:
                    if resp_event.event_type in TOOL_RESPONSE_TYPES:
                        self.server.match_result(resp_event)
                    self._emit(resp_event)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            # send response to agent
            self._relay_head(resp)
            headers_sent = True
            self.wfile.write(resp_body)

        except Exception as e:
            self._report_failure("Proxy error", e, headers_sent)
        finally:
            if sock is not None:
                self.server.unwatch_socket(sock)
            conn.close()

    def do_GET(self):
        """Forward GET requests (SSE stream from server -> agent)."""
        with self.server.request_scope(self.connection) as accepted:
            if accepted:
                self._forward_get()

    def _forward_get(self):
        conn = self._get_connection()
        headers = self._client_headers(_GET_HEADERS)

        sock = None
        headers_sent = False
        try:
            sock = self._connect(conn)
            conn.request("GET", self._remote_path(self.path), headers=headers)
            resp = conn.getresponse()
            self._relay_head(resp)
            headers_sent = True
            self._relay_events(resp)

        except Exception as e:
            self._report_failure("SSE proxy error", e, headers_sent)
        finally:
            if sock is not None:
                self.server.unwatch_socket(sock)
            conn.close()

    def do_DELETE(self):
        """Forward DELETE requests, which end a Streamable HTTP session."""
        with self.server.request_scope(self.connection) as accepted:
            if accepted:
                self._forward_delete()

    def _forward_delete(self):
        conn = self._get_connection()
        headers = self._client_headers(_DELETE_HEADERS)

        sock = None
        headers_sent = False
        try:
            sock = self._connect(conn)
            conn.request("DELETE", self._remote_path(self.path), headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read()
            self._relay_head(resp)
            headers_sent = True
            self.wfile.write(resp_body)

        except Exception as e:
            self._report_failure("Proxy error", e, headers_sent)
        finally:
            if sock is not None:
                self.server.unwatch_socket(sock)
            conn.close()

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def _link_parent(store: TraceStore, meta: SessionMeta, parent_session_id: str) -> None:
    """Save meta with parent_session_id as its parent, if the store accepts it.

    The store refuses a parent that is not in it, is this session, or fails
    its tenant check. Then, or if the save fails, meta keeps no parent.
    """
    meta.parent_session_id = parent_session_id
    try:
        store.update_meta(meta)
    except (ValueError, OSError):
        meta.parent_session_id = ""


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


class HTTPProxyServer:
    """HTTP/SSE proxy for remote MCP servers.

    ``remote_url`` is the server's endpoint URL. It is used as given, with
    any trailing slash.

    ``ca_file`` trusts a PEM CA bundle for an HTTPS upstream instead of the
    system store. It is loaded here, so a bad file or a non-HTTPS URL raises
    ValueError before the proxy starts.

    ``allowed_hosts`` are extra names a request may give as its Host or as
    its Origin host, beside localhost and IP addresses. Each must match a
    whole name. Case is ignored.
    """

    def __init__(
        self,
        remote_url: str,
        local_port: int,
        store: TraceStore,
        session_meta: SessionMeta,
        on_event: Callable[[TraceEvent], None] | None = None,
        redact: bool = False,
        ca_file: str = "",
        allowed_hosts: Iterable[str] = (),
    ):
        self.remote_url = remote_url
        self.upstream_tls = upstream_tls_context(self.remote_url, ca_file)
        self.local_port = local_port
        self.store = store
        self.meta = session_meta
        self.on_event = on_event
        self.redact = redact
        self.allowed_hosts = frozenset(name.lower() for name in allowed_hosts)

    def make_server(self) -> _ProxyHTTPServer:
        """Bind a server with its own config and pending call state."""
        return _ProxyHTTPServer(
            ("127.0.0.1", self.local_port),
            remote_url=self.remote_url,
            store=self.store,
            meta=self.meta,
            on_event=self.on_event,
            redact=self.redact,
            upstream_tls=self.upstream_tls,
            allowed_hosts=self.allowed_hosts,
        )

    def run(self) -> None:
        """Start the proxy server. Blocks until interrupted."""
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

        server = self.make_server()
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
            server.clear_pending()
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
            # A parent the store refuses now, such as one deleted while the
            # proxy ran, must not cost the end time and counters.
            try:
                self.store.update_meta(self.meta)
            except ValueError as exc:
                if not _save_without_parent(self.store, self.meta, exc):
                    raise
