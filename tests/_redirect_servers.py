"""Local servers and mocks for checking that a POST does not follow a redirect.

``RedirectServers`` starts two servers on 127.0.0.1. Server A answers the
chosen methods with a redirect to the same path on server B, and any other
method with 200. Server B answers every request with 200. Both record what
they receive, so a followed redirect shows up as a request on B.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import unittest
from collections.abc import Iterator
from dataclasses import dataclass
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest import mock

REDIRECT_NOTE = " (redirects are not followed; give the final URL)"
REDIRECTS = (301, 302, 303)
ALL_REDIRECTS = (301, 302, 303, 307, 308)
# An OpenAI-style chat reply. Every caller tested here reads it as success.
REPLY = {"choices": [{"message": {"content": "from B"}}], "ok": True}


@dataclass(frozen=True)
class Received:
    method: str
    path: str
    headers: Message
    body: bytes


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, reply: Any) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.server_address[1]}"
        self.reply = json.dumps(reply).encode()
        self.status = 200
        self.location = ""
        self.redirect_methods: frozenset[str] = frozenset()
        self.lock = threading.Lock()
        self._received: list[Received] = []

    def handle_error(self, request, client_address) -> None:
        pass  # the client gave up and closed the connection

    def record(self, request: Received) -> None:
        with self.lock:
            self._received.append(request)

    def received(self) -> list[Received]:
        with self.lock:
            return list(self._received)

    def methods(self) -> list[str]:
        return [request.method for request in self.received()]

    def clear(self) -> None:
        with self.lock:
            self._received.clear()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _answer(self) -> None:
        server = self.server
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        server.record(Received(self.command, self.path, self.headers, body))
        if self.command in server.redirect_methods:
            self.send_response(server.status)
            self.send_header("Location", server.location + self.path)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(server.reply)))
        self.end_headers()
        self.wfile.write(server.reply)

    do_GET = do_POST = do_PUT = do_PATCH = _answer


class RedirectServers:
    """Server A redirects *methods* to the same path on server B.

    B answers with *reply*. A answers other methods with *a_reply*, or
    *reply* when it is None. Call ``redirect`` to pick the status.
    """

    def __init__(
        self,
        test: unittest.TestCase,
        *,
        methods: tuple[str, ...] = ("POST",),
        reply: Any = REPLY,
        a_reply: Any = None,
    ) -> None:
        # A proxy from the environment or the system must not see these.
        environ = mock.patch.dict(os.environ, {"no_proxy": "*"})
        environ.start()
        test.addCleanup(environ.stop)
        self.b = self._start(test, reply)
        self.a = self._start(test, reply if a_reply is None else a_reply)
        self.a.redirect_methods = frozenset(methods)
        self.a.location = self.b.url
        self.url = self.a.url

    @staticmethod
    def _start(test: unittest.TestCase, reply: Any) -> _Server:
        server = _Server(reply)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        test.addCleanup(thread.join, 5)
        test.addCleanup(server.server_close)
        test.addCleanup(server.shutdown)
        return server

    def redirect(self, status: int) -> None:
        """Answer *methods* with *status* from now on and forget earlier requests."""
        self.a.status = status
        self.a.clear()
        self.b.clear()


@contextlib.contextmanager
def post_opener(target: str, **kwargs: Any) -> Iterator[mock.MagicMock]:
    """Patch the no_redirect_opener a module uses, and yield the mock of its open.

    *kwargs* configure that mock (return_value, side_effect) the way they
    configured the urlopen mock it replaces.
    """
    open_mock = mock.MagicMock(**kwargs)
    with mock.patch(target, return_value=mock.MagicMock(open=open_mock)):
        yield open_mock
