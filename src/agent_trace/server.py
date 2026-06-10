"""Server-side event collector.

A lightweight HTTP server that receives events from remote agents and stores
them in the same .agent-traces/ format as local mode. Zero new dependencies —
uses Python stdlib http.server only.

API:
    POST /events              Receive a batch of NDJSON events
    POST /sessions            Create or update session metadata
    GET  /sessions            List all sessions (JSON array)
    GET  /sessions/<id>/events  Stream events for a session (NDJSON)
    GET  /health              Liveness check

Usage:
    agent-strace server --port 4317 --storage ./traces

    # With API key authentication
    agent-strace server keygen                          # generate a key
    agent-strace server --auth-key ast_<key>            # enforce auth
    AGENT_STRACE_AUTH_KEY=ast_<key> agent-strace server # via env var

Agents point to it via environment variable:
    AGENT_STRACE_ENDPOINT=http://collector:4317 python my_agent.py

    # With auth key on the client side
    AGENT_STRACE_ENDPOINT=https://collector.example.com
    AGENT_STRACE_AUTH_KEY=ast_<key>

Without --auth-key / AGENT_STRACE_AUTH_KEY the server runs unauthenticated
(original behaviour, unchanged for local use).

See ADR-0012 for architecture decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import SessionMeta, TraceEvent
from .store import TraceStore, DEFAULT_TRACE_DIR
from . import web_dashboard as _wd


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

KEY_PREFIX = "ast_"


def generate_api_key() -> str:
    """Return a new ``ast_``-prefixed API key using secrets.token_hex."""
    return KEY_PREFIX + secrets.token_hex(16)


# ---------------------------------------------------------------------------
# Remote event sender (used by hooks when AGENT_STRACE_ENDPOINT is set)
# ---------------------------------------------------------------------------

def _auth_headers() -> dict[str, str]:
    """Return Authorization header dict if AGENT_STRACE_AUTH_KEY is set."""
    key = os.environ.get("AGENT_STRACE_AUTH_KEY", "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def send_event_to_endpoint(event: TraceEvent, endpoint: str) -> bool:
    """POST a single event to a remote collector.

    Returns True on success. Failures are logged to stderr but never raise —
    the hook must not block the agent.

    When AGENT_STRACE_AUTH_KEY is set, injects ``Authorization: Bearer``
    automatically.
    """
    import urllib.request
    import urllib.error

    url = endpoint.rstrip("/") + "/events"
    body = (event.to_json() + "\n").encode("utf-8")
    headers = {"Content-Type": "application/x-ndjson", **_auth_headers()}
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status in (200, 202)
    except Exception as exc:
        sys.stderr.write(f"[agent-strace] remote send failed: {exc}\n")
        return False


def send_session_meta_to_endpoint(meta: SessionMeta, endpoint: str) -> bool:
    """POST session metadata to a remote collector.

    When AGENT_STRACE_AUTH_KEY is set, injects ``Authorization: Bearer``
    automatically.
    """
    import urllib.request
    import urllib.error

    url = endpoint.rstrip("/") + "/sessions"
    body = meta.to_json().encode("utf-8")
    headers = {"Content-Type": "application/json", **_auth_headers()}
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status in (200, 202)
    except Exception as exc:
        sys.stderr.write(f"[agent-strace] remote session meta send failed: {exc}\n")
        return False


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class CollectorHandler(BaseHTTPRequestHandler):
    """HTTP handler for the event collector server."""

    # Injected by the server setup
    store: TraceStore
    _lock: threading.Lock
    _auth_key: str  # empty string = no auth required
    _dashboard: bool = False  # serve web dashboard UI

    def log_message(self, fmt: str, *args: Any) -> None:
        # Suppress default access log; write to stderr with our prefix
        sys.stderr.write(f"[server] {fmt % args}\n")

    def _check_auth(self) -> bool:
        """Return True if the request is authorised (or auth is disabled)."""
        if not self._auth_key:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self._auth_key}"

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_ndjson(self, status: int, lines: list[str]) -> None:
        body = "\n".join(lines).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return self.rfile.read(length)
        return b""

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._check_auth():
            self._send_json(401, {"error": "Unauthorized"})
            return
        path = urlparse(self.path).path.rstrip("/")

        # ------------------------------------------------------------------
        # Dashboard UI routes (only when --dashboard is enabled)
        # ------------------------------------------------------------------
        if self._dashboard:
            if path == "" or path == "/":
                self._send_html(200, _wd.render_sessions_page())
                return
            if path == "/cost":
                self._send_html(200, _wd.render_cost_page())
                return
            if path == "/violations":
                self._send_html(200, _wd.render_violations_page())
                return
            if path == "/health":
                self._send_html(200, _wd.render_health_page())
                return
            parts = path.split("/")
            if len(parts) == 3 and parts[1] == "session":
                self._send_html(200, _wd.render_detail_page(parts[2]))
                return
            # Dashboard API endpoints
            if path == "/api/sessions":
                body = _wd.api_sessions(self.store).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if len(parts) == 5 and parts[1] == "api" and parts[2] == "sessions" and parts[4] == "events":
                result = _wd.api_session_events(self.store, parts[3])
                if result is None:
                    self._send_json(404, {"error": f"session not found: {parts[3]}"})
                    return
                body = result.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

        if path == "/health":
            self._send_json(200, {"status": "ok", "sessions": len(self.store.list_sessions())})
            return

        if path == "/sessions":
            sessions = self.store.list_sessions()
            data = [json.loads(m.to_json()) for m in sessions]
            self._send_json(200, data)
            return

        # /sessions/<id>/events
        parts = path.split("/")
        if len(parts) == 4 and parts[1] == "sessions" and parts[3] == "events":
            session_id = parts[2]
            if not self.store.session_exists(session_id):
                found = self.store.find_session(session_id)
                if found:
                    session_id = found
                else:
                    self._send_json(404, {"error": f"session not found: {session_id}"})
                    return
            events = self.store.load_events(session_id)
            self._send_ndjson(200, [e.to_json() for e in events])
            return

        # /sessions/<id>
        if len(parts) == 3 and parts[1] == "sessions":
            session_id = parts[2]
            if not self.store.session_exists(session_id):
                found = self.store.find_session(session_id)
                if found:
                    session_id = found
                else:
                    self._send_json(404, {"error": f"session not found: {session_id}"})
                    return
            meta = self.store.load_meta(session_id)
            self._send_json(200, json.loads(meta.to_json()))
            return

        self._send_json(404, {"error": "not found"})

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        if not self._check_auth():
            self._send_json(401, {"error": "Unauthorized"})
            return
        path = urlparse(self.path).path.rstrip("/")
        body = self._read_body()

        if path == "/events":
            self._handle_post_events(body)
            return

        if path == "/sessions":
            self._handle_post_sessions(body)
            return

        self._send_json(404, {"error": "not found"})

    def _handle_post_events(self, body: bytes) -> None:
        """Accept a batch of NDJSON events."""
        accepted = 0
        errors = 0
        for line in body.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = TraceEvent.from_json(line)
                session_id = event.session_id
                if not session_id:
                    errors += 1
                    continue
                with self._lock:
                    # Auto-create session if it doesn't exist
                    if not self.store.session_exists(session_id):
                        meta = SessionMeta()
                        meta.session_id = session_id
                        self.store.create_session(meta)
                    self.store.append_event(session_id, event)
                accepted += 1
            except Exception as exc:
                sys.stderr.write(f"[server] event parse error: {exc}\n")
                errors += 1

        status = 200 if errors == 0 else 202
        self._send_json(status, {"accepted": accepted, "errors": errors})

    def _handle_post_sessions(self, body: bytes) -> None:
        """Create or update session metadata."""
        try:
            data = json.loads(body.decode("utf-8"))
            session_id = data.get("session_id", "")
            if not session_id:
                self._send_json(400, {"error": "session_id required"})
                return

            with self._lock:
                if self.store.session_exists(session_id):
                    # Update existing meta
                    meta = self.store.load_meta(session_id)
                    for k, v in data.items():
                        if hasattr(meta, k):
                            setattr(meta, k, v)
                    self.store.update_meta(meta)
                else:
                    meta = SessionMeta.from_json(body.decode("utf-8"))
                    self.store.create_session(meta)

            self._send_json(200, {"session_id": session_id, "status": "ok"})
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})


def _make_handler(
    store: TraceStore,
    lock: threading.Lock,
    auth_key: str = "",
    dashboard: bool = False,
) -> type:
    """Return a CollectorHandler subclass with store, lock, auth key, and dashboard flag injected."""
    class Handler(CollectorHandler):
        pass
    Handler.store = store
    Handler._lock = lock
    Handler._auth_key = auth_key
    Handler._dashboard = dashboard
    return Handler


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def run_server(
    port: int,
    storage_dir: str,
    host: str = "0.0.0.0",
    auth_key: str = "",
    dashboard: bool = False,
) -> None:
    """Start the collector server and block until interrupted.

    When *auth_key* is non-empty, all requests must include
    ``Authorization: Bearer <auth_key>`` or receive 401.
    When *dashboard* is True, the server also serves a browser-based UI at ``/``.
    """
    store = TraceStore(storage_dir)
    lock = threading.Lock()
    handler_class = _make_handler(store, lock, auth_key=auth_key, dashboard=dashboard)

    server = HTTPServer((host, port), handler_class)
    auth_note = " (auth enabled)" if auth_key else " (no auth)"
    dash_note = f"\n[agent-strace server] dashboard: http://{host}:{port}/" if dashboard else ""
    sys.stderr.write(
        f"[agent-strace server] listening on {host}:{port}{auth_note}\n"
        f"[agent-strace server] storage: {Path(storage_dir).resolve()}\n"
        f"[agent-strace server] health: http://{host}:{port}/health{dash_note}\n"
        f"\n"
        f"[agent-strace server] purpose : collect traces from remote agents\n"
        f"[agent-strace server] usage   : set AGENT_STRACE_ENDPOINT=http://<this-host>:{port} on each agent machine\n"
        f"[agent-strace server] api     : POST /events  POST /sessions  GET /sessions  GET /sessions/<id>/events\n"
        f"[agent-strace server] inspect : agent-strace list / replay / export  (reads the same storage dir)\n"
        f"[agent-strace server] html    : agent-strace replay --format html    (shareable single-file viewer)\n"
        f"[agent-strace server] otlp    : agent-strace export --format otlp --endpoint <collector-url>\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[agent-strace server] shutting down\n")
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_server(args: argparse.Namespace) -> int:
    subcommand = getattr(args, "server_subcommand", None)

    if subcommand == "keygen":
        sys.stdout.write(generate_api_key() + "\n")
        return 0

    port = getattr(args, "port", 4317)
    storage = getattr(args, "storage", None) or os.environ.get(
        "AGENT_STRACE_STORAGE", DEFAULT_TRACE_DIR
    )
    host = getattr(args, "host", "0.0.0.0")
    auth_key = (
        getattr(args, "auth_key", None)
        or os.environ.get("AGENT_STRACE_AUTH_KEY", "")
    ).strip()
    dashboard = getattr(args, "dashboard", False)
    run_server(port=port, storage_dir=storage, host=host, auth_key=auth_key, dashboard=dashboard)
    return 0
