"""Tests for export --endpoint and --header.

Export checks both flags before it reads or writes anything, POSTs to the
OTLP traces URL that watch --stream-otlp uses, never follows a redirect,
counts any 2xx as success, and names only the collector's
scheme://host[:port] in its lines. The E numbers are the cases the fix was
reported with.
"""

from __future__ import annotations

import http
import http.client
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import agent_trace
from agent_trace import http_post, otlp, watch
from agent_trace.cli import build_parser, cmd_export
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

FORMATS = ("otlp", "otlp-genai", "temporal")
SECRET = "SECRET-SENTINEL"
TRACE_ID = "a" * 32
PARENT_SPAN_ID = "b" * 16
REDIRECT_NOTE = " (redirects are not followed; give the final URL)"
# Shown raw, this reason phrase sets the window title and clears the screen.
REASON = "Oops \x1b]0;PWNED-TITLE\x07\x1b[2J\x1b[31mRED\x1b[0m \x9b31m end"
REASON_SHOWN = "Oops \\x1b]0;PWNED-TITLE\\x07\\x1b[2J\\x1b[31mRED\\x1b[0m \\x9b31m end"
FORMATS_HINT = "use --format otlp, otlp-genai or temporal"
NAME_RULE = "its name may only contain ASCII letters, digits and !#$%&'*+-.^_`|~"
VALUE_RULE = "its value must contain only printable ASCII"
# Variables that would send export --scores, --metrics or --backend somewhere.
_ROUTE_ENV = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
)


class _Request:
    def __init__(self, method: str, path: str, headers: list[tuple[str, str]], body: bytes):
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body

    def header(self, name: str) -> list[str]:
        return [value for key, value in self.headers if key.lower() == name.lower()]


class _Collector(ThreadingHTTPServer):
    """Record every request and answer each POST with the next scripted reply.

    A reply is (status, body, hold). It waits up to *hold* seconds, or until
    ``release`` is set, before it is sent. A 3xx reply points Location at
    another host name for this server, so a followed redirect shows up as a
    GET. Each reply has the standard reason phrase, or ``reason`` when it is
    set.
    """

    daemon_threads = True
    block_on_close = False

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _CollectorHandler)
        self.lock = threading.Lock()
        self.requests: list[_Request] = []
        self.replies: list[tuple[int, bytes, float]] = []
        self.release = threading.Event()
        self.reason: str | None = None

    def handle_error(self, request, client_address) -> None:
        pass  # the client timed out and closed the connection

    def script(self, *replies: tuple[int, bytes, float]) -> None:
        with self.lock:
            self.requests.clear()
            self.replies[:] = replies
        self.release.clear()

    def received(self) -> list[_Request]:
        with self.lock:
            return list(self.requests)


class _CollectorHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        server = self.server
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with server.lock:
            server.requests.append(_Request("POST", self.path, list(self.headers.items()), body))
            status, reply, hold = server.replies.pop(0) if server.replies else (200, b"", 0.0)
        if hold:
            server.release.wait(hold)
        self.send_response(status, server.reason)
        if 300 <= status < 400:
            self.send_header("Location", f"http://localhost:{server.server_address[1]}/landing")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def do_GET(self) -> None:  # noqa: N802
        with self.server.lock:
            self.server.requests.append(_Request("GET", self.path, list(self.headers.items()), b""))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


class _RefusingProxy(ThreadingHTTPServer):
    """An HTTPS proxy that answers every CONNECT with 403 and *reason*."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, reason: str) -> None:
        super().__init__(("127.0.0.1", 0), _RefusingProxyHandler)
        self.reason = reason

    def handle_error(self, request, client_address) -> None:
        pass  # the client closed the connection first


class _RefusingProxyHandler(BaseHTTPRequestHandler):
    def do_CONNECT(self) -> None:  # noqa: N802
        self.send_response(403, self.server.reason)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


def _tree(root: str) -> list[tuple[str, int]]:
    """Return every path under *root* with its size, to show nothing was written."""
    return sorted(
        (path.relative_to(root).as_posix(), path.stat().st_size if path.is_file() else -1)
        for path in Path(root).rglob("*")
    )


def _response(status: int) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class _ExportCase(unittest.TestCase):
    """Run export in-process on a session with two events and a Temporal context."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = temp.name
        self.trace_dir = os.path.join(self.root, "traces")
        self.cwd = os.path.join(self.root, "cwd")
        os.makedirs(self.cwd)
        environ = patch.dict(os.environ, {"no_proxy": "*"})
        environ.start()
        self.addCleanup(environ.stop)
        for name in _ROUTE_ENV:
            os.environ.pop(name, None)
        store = TraceStore(self.trace_dir)
        meta = SessionMeta(
            agent_name="probe", trace_id=TRACE_ID, parent_span_id=PARENT_SPAN_ID, trace_flags="01",
        )
        store.create_session(meta)
        for timestamp, event_type, data in (
            (1.0, EventType.SESSION_START, {}),
            (2.0, EventType.TOOL_CALL, {"tool_name": "Bash", "arguments": {"command": "ls"}}),
        ):
            store.append_event(meta.session_id, TraceEvent(
                event_type=event_type, session_id=meta.session_id, timestamp=timestamp, data=data,
            ))
        self.session_id = meta.session_id

    def export(self, *flags: str, session_id: str | None = None) -> tuple[int, str, str]:
        """Run ``agent-strace export`` in self.cwd and return (code, stdout, stderr)."""
        args = build_parser().parse_args([
            "--trace-dir", self.trace_dir, "export",
            self.session_id if session_id is None else session_id, *flags,
        ])
        out, err = io.StringIO(), io.StringIO()
        previous = os.getcwd()
        os.chdir(self.cwd)
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = cmd_export(args)
        finally:
            os.chdir(previous)
        return code, out.getvalue(), err.getvalue()


class _CollectorCase(_ExportCase):
    """Export to a local collector."""

    def setUp(self) -> None:
        super().setUp()
        self.collector = _Collector()
        self.base = f"http://127.0.0.1:{self.collector.server_address[1]}"
        threading.Thread(
            target=self.collector.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        ).start()
        self.addCleanup(self.collector.server_close)
        self.addCleanup(self.collector.shutdown)
        self.addCleanup(self.collector.release.set)

    def send(
        self,
        fmt: str,
        *replies: tuple[int, bytes, float],
        endpoint: str | None = None,
        headers: tuple[str, ...] = ("X-Probe-Key: k",),
    ) -> tuple[int, str, str]:
        """Export *fmt* to the collector, which answers with *replies*."""
        self.collector.script(*replies)
        flags = ["--format", fmt, "--endpoint", self.base if endpoint is None else endpoint]
        for header in headers:
            flags += ["--header", header]
        return self.export(*flags)

    def failure(self, detail: str, endpoint: str | None = None) -> str:
        return f"OTLP export to {self.base if endpoint is None else endpoint} failed: {detail}\n"


class TestOtlpTracesUrl(unittest.TestCase):
    def test_rule(self):
        cases = {
            "http://h:4318": "http://h:4318/v1/traces",
            "http://h:4318/": "http://h:4318/v1/traces",
            "http://h:4318/v1/traces": "http://h:4318/v1/traces",
            "http://h:4318/v1/traces//": "http://h:4318/v1/traces",
            "https://h/api/public/otel": "https://h/api/public/otel/v1/traces",
            "https://h/ingest?team=a": "https://h/ingest/v1/traces?team=a",
            "https://h?team=a": "https://h/v1/traces?team=a",
            "https://h/base#frag": "https://h/base/v1/traces#frag",
            "https://h/?k=v#f": "https://h/v1/traces?k=v#f",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(http_post.otlp_traces_url(url), expected)

    def test_export_and_watch_use_the_same_rule(self):
        self.assertIs(otlp.otlp_traces_url, http_post.otlp_traces_url)
        self.assertIs(watch._otlp_traces_url, http_post.otlp_traces_url)


class TestParseHeader(unittest.TestCase):
    def test_name_and_value(self):
        cases = {
            "X-Probe-Key: k": ("X-Probe-Key", "k"),
            "  x-honeycomb-team :  two words  ": ("x-honeycomb-team", "two words"),
            "X-Url: http://a:b/c": ("X-Url", "http://a:b/c"),
            "X:v": ("X", "v"),
            "!#$%&'*+-.^_`|~09azAZ: ~": ("!#$%&'*+-.^_`|~09azAZ", "~"),
        }
        for header, expected in cases.items():
            with self.subTest(header=header):
                self.assertEqual(http_post.parse_header(header), expected)

    def test_reason_never_repeats_the_header(self):
        cases = (
            (f"Authorization {SECRET}", "it has no colon; give it as NAME: VALUE"),
            (f": {SECRET}", "its name is empty"),
            ("   : v", "its name is empty"),
            (f"X {SECRET}: v", NAME_RULE),
            (f"X\t{SECRET}: v", NAME_RULE),
            (f"X-Pr\u00f8be: {SECRET}", NAME_RULE),
            (f"X\x00{SECRET}: v", NAME_RULE),
            (f"{SECRET}:", "its value is empty"),
            (f"{SECRET}: \t ", "its value is empty"),
            (f"X-Probe-Key: a\n{SECRET}", VALUE_RULE),
            (f"X-Probe-Key: a\t{SECRET}", VALUE_RULE),
            (f"X-Probe-Key: {SECRET}\x7f", VALUE_RULE),
            (f"X-Probe-Key: {SECRET}\u2603", VALUE_RULE),
        )
        for header, reason in cases:
            with self.subTest(header=header):
                with self.assertRaises(ValueError) as caught:
                    http_post.parse_header(header)
                self.assertEqual(str(caught.exception), reason)


class TestRedirectHint(unittest.TestCase):
    def test_only_a_3xx_gets_the_note(self):
        cases = {
            300: REDIRECT_NOTE,
            301: REDIRECT_NOTE,
            308: REDIRECT_NOTE,
            399: REDIRECT_NOTE,
            400: "",
            404: "",
            500: "",
        }
        for code, note in cases.items():
            with self.subTest(code=code):
                error = urllib.error.HTTPError("http://h/x", code, "Reason", {}, None)
                self.addCleanup(error.close)
                self.assertEqual(http_post.redirect_hint(error), note)


class TestNoRedirectOpener(unittest.TestCase):
    def test_uses_a_proxy_from_the_environment(self):
        # urlopen sends through http_proxy and https_proxy, so the POSTs
        # moved off it must too. ProxyHandler reads the environment when
        # the opener is built.
        proxy = _Collector()
        threading.Thread(
            target=proxy.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        ).start()
        self.addCleanup(proxy.server_close)
        self.addCleanup(proxy.shutdown)
        proxy_url = f"http://127.0.0.1:{proxy.server_address[1]}"
        url = "http://collector.invalid/v1/x"
        environ = {
            "http_proxy": proxy_url, "HTTP_PROXY": proxy_url, "no_proxy": "", "NO_PROXY": "",
        }
        with patch.dict(os.environ, environ):
            request = urllib.request.Request(url, data=b"{}", method="POST")
            with http_post.no_redirect_opener().open(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
        self.assertEqual(
            [(r.method, r.path, r.body) for r in proxy.received()], [("POST", url, b"{}")],
        )


class TestPostErrorDetail(unittest.TestCase):
    """The reason comes from a server or a proxy, so it is shown inert."""

    def _http_error(self, code, reason):
        exc = urllib.error.HTTPError("http://h/", code, reason, {}, None)
        self.addCleanup(exc.close)
        return exc

    def test_http_error_reason(self):
        for code, note in ((500, ""), (302, REDIRECT_NOTE)):
            with self.subTest(code=code):
                exc = self._http_error(code, f"{REASON}\nforged")
                self.assertEqual(
                    http_post.post_error_detail(exc),
                    f"HTTP Error {code}: {REASON_SHOWN}\\nforged{note}",
                )

    def test_url_error_reason(self):
        cases = (
            ("bad \x1b[2K\nforged", "bad \\x1b[2K\\nforged"),
            (OSError(f"Tunnel connection failed: 403 {REASON}"),
             f"Tunnel connection failed: 403 {REASON_SHOWN}"),
        )
        for reason, shown in cases:
            with self.subTest(reason=reason):
                exc = urllib.error.URLError(reason)
                self.assertEqual(http_post.post_error_detail(exc), shown)

    def test_printable_reason_is_unchanged(self):
        cases = (
            (self._http_error(404, "Not Found"), "HTTP Error 404: Not Found"),
            (self._http_error(503, "Busy: café → 日本"), "HTTP Error 503: Busy: café → 日本"),
            (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
             "[Errno 61] Connection refused"),
            (urllib.error.URLError("café → 日本"), "café → 日本"),
        )
        for exc, detail in cases:
            with self.subTest(detail=detail):
                self.assertEqual(http_post.post_error_detail(exc), detail)


class TestExportSends(_CollectorCase):
    def test_any_2xx_is_success(self):
        # E01 to E04. main exited 1 on 201 and 204, and otlp-genai printed
        # "Exported" before it did.
        for fmt in FORMATS:
            for status in (200, 201, 202, 204):
                with self.subTest(format=fmt, status=status):
                    result = self.send(fmt, (status, b"", 0.0))
                    self.assertEqual(
                        result, (0, "", f"Exported 2 events to {self.base} (HTTP {status})\n"),
                    )
                    [post] = self.collector.received()
                    self.assertEqual((post.method, post.path), ("POST", "/v1/traces"))
                    self.assertEqual(post.header("X-Probe-Key"), ["k"])
                    self.assertEqual(post.header("Content-Type"), ["application/json"])
                    spans = json.loads(post.body)["resourceSpans"][0]["scopeSpans"][0]["spans"]
                    self.assertTrue(spans)
                    if fmt == "temporal":
                        self.assertEqual(
                            (spans[0]["traceId"], spans[0]["parentSpanId"]),
                            (TRACE_ID, PARENT_SPAN_ID),
                        )
                    self.assertEqual(os.listdir(self.cwd), [])

    def test_traces_url(self):
        # E15 to E17. main appended /v1/traces to the whole URL, so it
        # POSTed /?api_key=k/v1/traces, /v1/traces/v1/traces and /base.
        cases = (
            ("", "/v1/traces"),
            ("/", "/v1/traces"),
            ("/v1/traces", "/v1/traces"),
            ("/v1/traces/", "/v1/traces"),
            ("/otlp", "/otlp/v1/traces"),
            ("/api/public/otel//", "/api/public/otel/v1/traces"),
            ("/?api_key=k", "/v1/traces?api_key=k"),
            ("/ingest?team=a&x=1", "/ingest/v1/traces?team=a&x=1"),
            ("?team=a", "/v1/traces?team=a"),
            ("/base#frag", "/base/v1/traces"),
        )
        for fmt in FORMATS:
            for suffix, path in cases:
                with self.subTest(format=fmt, endpoint=suffix):
                    code, _, err = self.send(fmt, (200, b"", 0.0), endpoint=self.base + suffix)
                    self.assertEqual((code, err), (0, f"Exported 2 events to {self.base} (HTTP 200)\n"))
                    self.assertEqual(
                        [(r.method, r.path) for r in self.collector.received()], [("POST", path)],
                    )

    def test_line_names_only_the_endpoint(self):
        # E15. main printed the whole URL, with the token in its query.
        endpoint = f"{self.base}/{SECRET}/x?api_key={SECRET}#{SECRET}"
        for fmt in FORMATS:
            with self.subTest(format=fmt):
                result = self.send(
                    fmt, (200, b"", 0.0), endpoint=endpoint, headers=(f"X-Probe-Key: {SECRET}",),
                )
                self.assertEqual(result, (0, "", f"Exported 2 events to {self.base} (HTTP 200)\n"))
                [post] = self.collector.received()
                self.assertEqual(post.path, f"/{SECRET}/x/v1/traces?api_key={SECRET}")
                self.assertEqual(post.header("X-Probe-Key"), [SECRET])

    def test_headers(self):
        # Spaces around the name and the value are dropped, and the value
        # keeps every colon after the first.
        headers = ("X-Api-Key :  key one  ", "x-honeycomb-team: abc", "X-Url: http://a:b/c")
        for fmt in FORMATS:
            with self.subTest(format=fmt):
                code, _, err = self.send(fmt, (202, b"", 0.0), headers=headers)
                self.assertEqual((code, err), (0, f"Exported 2 events to {self.base} (HTTP 202)\n"))
                [post] = self.collector.received()
                self.assertEqual(post.header("X-Api-Key"), ["key one"])
                self.assertEqual(post.header("X-Honeycomb-Team"), ["abc"])
                self.assertEqual(post.header("X-Url"), ["http://a:b/c"])


class TestExportFailures(_CollectorCase):
    def test_redirects_are_not_followed(self):
        # E05, E06. main followed a 302 with a GET to another host that
        # carried the --header value, lost the spans and exited 0.
        for fmt in FORMATS:
            for status in (301, 302, 303, 307, 308):
                with self.subTest(format=fmt, status=status):
                    result = self.send(fmt, (status, b"", 0.0), headers=(f"X-Probe-Key: {SECRET}",))
                    detail = f"HTTP Error {status}: {http.HTTPStatus(status).phrase}{REDIRECT_NOTE}"
                    self.assertEqual(result, (1, "", self.failure(detail)))
                    self.assertEqual([r.method for r in self.collector.received()], ["POST"])

    def test_error_reply_body(self):
        # E07, E08. otlp-genai printed no body, and otlp let escape codes
        # and line breaks through.
        cases = (
            (b"collector says no", "collector says no"),
            (b"bad \x1b[31mred \x1b[0m\nline2", "bad  [31mred  [0m line2"),
            ("a\u2028b\x85c\u200bd\re".encode(), "a b c d e"),
            (b"\r\n\t padded \x1b\r\n", "padded"),
            (b"x" * 500, "x" * 200),
            ("\U0001F600".encode() * 300, "\U0001F600" * 200),
        )
        for fmt in FORMATS:
            for body, excerpt in cases:
                with self.subTest(format=fmt, body=body[:20]):
                    result = self.send(fmt, (500, body, 0.0))
                    line = self.failure("HTTP Error 500: Internal Server Error")
                    self.assertEqual(result, (1, "", f"{line}  {excerpt}\n"))

    def test_error_without_a_body_is_one_line(self):
        for fmt in FORMATS:
            for body in (b"", b" \x1b\r\n"):
                with self.subTest(format=fmt, body=body):
                    result = self.send(fmt, (404, body, 0.0))
                    self.assertEqual(result, (1, "", self.failure("HTTP Error 404: Not Found")))

    def test_reason_phrase_is_inert(self):
        # e500, e302. main printed the reason raw, so it set the window
        # title and cleared the screen.
        self.collector.reason = REASON
        for fmt in FORMATS:
            for status, note in ((500, ""), (302, REDIRECT_NOTE)):
                with self.subTest(format=fmt, status=status):
                    result = self.send(fmt, (status, b"", 0.0))
                    detail = f"HTTP Error {status}: {REASON_SHOWN}{note}"
                    self.assertEqual(result, (1, "", self.failure(detail)))

    def test_proxy_reason_phrase_is_inert(self):
        # http.client puts the reason of a refused CONNECT in an OSError and
        # urllib wraps that in a URLError. main printed it raw.
        proxy = _RefusingProxy(REASON)
        threading.Thread(
            target=proxy.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        ).start()
        self.addCleanup(proxy.server_close)
        self.addCleanup(proxy.shutdown)
        endpoint = "https://collector.test"
        env = {"https_proxy": f"http://127.0.0.1:{proxy.server_address[1]}", "no_proxy": ""}
        with patch.dict(os.environ, env):
            for fmt in FORMATS:
                with self.subTest(format=fmt):
                    result = self.send(fmt, endpoint=endpoint)
                    detail = f"Tunnel connection failed: 403 {REASON_SHOWN}"
                    self.assertEqual(result, (1, "", self.failure(detail, endpoint)))
        self.assertEqual(self.collector.received(), [])

    def test_timeout_is_one_line(self):
        # E09. main printed a TimeoutError traceback for otlp and temporal.
        self.assertEqual(otlp._EXPORT_TIMEOUT, 30)
        with patch.object(otlp, "_EXPORT_TIMEOUT", 0.3):
            for fmt in FORMATS:
                with self.subTest(format=fmt):
                    code, out, err = self.send(fmt, (200, b"", 10.0))
                    self.collector.release.set()
                    self.assertEqual((code, out), (1, ""))
                    self.assertIn(err, (self.failure("TimeoutError"), self.failure("timed out")))

    def test_connection_refused_is_one_line(self):
        # E10. The message names only the endpoint. Port 9 is the closed port
        # the rest of the suite uses; a bound socket that is not listening
        # times out on macOS instead of refusing.
        endpoint = "http://127.0.0.1:9"
        for fmt in FORMATS:
            with self.subTest(format=fmt):
                code, out, err = self.send(fmt, endpoint=f"{endpoint}/{SECRET}?k={SECRET}")
                self.assertEqual((code, out), (1, ""))
                prefix = self.failure("", endpoint)[:-1]
                self.assertTrue(err.startswith(prefix), err)
                self.assertIn("refused", err.lower())
                self.assertEqual(err.count("\n"), 1)
                self.assertTrue(err.endswith("\n"))
                self.assertNotIn(SECRET, err)

    def test_any_error_is_one_line_with_its_class_name(self):
        # Every exception from building and sending the request is caught,
        # and its text, which can repeat the URL, is not printed.
        errors = (
            ValueError(SECRET),
            http.client.InvalidURL(SECRET),
            UnicodeEncodeError("latin-1", SECRET, 0, 1, SECRET),
            OSError(SECRET),
            RuntimeError(SECRET),
        )
        for fmt in FORMATS:
            for error in errors:
                with self.subTest(format=fmt, error=type(error).__name__):
                    opener = MagicMock()
                    opener.open.side_effect = error
                    with patch.object(otlp, "no_redirect_opener", return_value=opener):
                        result = self.send(fmt)
                    self.assertEqual(result, (1, "", self.failure(type(error).__name__)))


class TestExportOtlpPayload(unittest.TestCase):
    ENDPOINT = "http://collector.test:4318/x?k=1"

    def post(self, payload: dict, opener: MagicMock) -> tuple[bool, str]:
        err = io.StringIO()
        with patch.object(otlp, "no_redirect_opener", return_value=opener), redirect_stderr(err):
            ok = otlp.export_otlp_payload(payload, self.ENDPOINT, headers={"X-A": "1"}, event_count=3)
        return ok, err.getvalue()

    def test_request(self):
        opener = MagicMock()
        opener.open.return_value = _response(202)
        result = self.post({"resourceSpans": []}, opener)
        self.assertEqual(result, (True, "Exported 3 events to http://collector.test:4318 (HTTP 202)\n"))
        [call] = opener.open.call_args_list
        request = call.args[0]
        self.assertEqual(call.kwargs, {"timeout": 30})
        self.assertEqual(request.full_url, "http://collector.test:4318/x/v1/traces?k=1")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(request.get_header("X-a"), "1")
        self.assertEqual(json.loads(request.data), {"resourceSpans": []})

    def test_payload_that_cannot_be_encoded(self):
        opener = MagicMock()
        result = self.post({"bad": object()}, opener)
        self.assertEqual(result, (False, "OTLP export to http://collector.test:4318 failed: TypeError\n"))
        opener.open.assert_not_called()

    def test_status_outside_2xx_is_never_exported(self):
        # The opener raises HTTPError for these; a response that carries one
        # anyway is still a failure.
        for status in (100, 199, 300, 304, 400, 500):
            with self.subTest(status=status):
                opener = MagicMock()
                opener.open.return_value = _response(status)
                result = self.post({}, opener)
                self.assertEqual(
                    result, (False, f"OTLP export to http://collector.test:4318 failed: HTTP {status}\n"),
                )


class TestExportRefusals(_CollectorCase):
    """Refusals: one line on stderr, exit 1, nothing read, written or sent."""

    def assertRefused(self, flags: list[str], line: str, *, session_id: str | None = None) -> None:
        before = _tree(self.root)
        result = self.export(*flags, session_id=session_id)
        self.assertEqual(result, (1, "", line + "\n"))
        self.assertEqual(_tree(self.root), before)
        self.assertEqual(self.collector.received(), [])

    def test_endpoint_value(self):
        # E11 to E14, E18, E19. main tried to send, with a traceback for
        # E19, or wrote a file and exited 0 for E18.
        port = self.collector.server_address[1]
        scheme = "it must start with http:// or https://"
        spaces = "it must not contain spaces or control characters"
        cases = (
            ("", "it is empty"),
            ("   ", "it is empty"),
            ("\t", "it is empty"),
            (f"127.0.0.1:{port}/{SECRET}", scheme),
            (f"ftp://127.0.0.1/{SECRET}", scheme),
            (f"http://127.0.0.1:99999/{SECRET}", "its port must be a number from 0 to 65535"),
            (f"http://user:{SECRET}@127.0.0.1:{port}", "it must not contain a user name or password"),
            (f"http:///{SECRET}", "it has no host"),
            (f"http://[{SECRET}", "it is not a valid URL"),
            (f"http://127.0.0.1:{port}/a {SECRET}", spaces),
            (f"http://127.0.0.1:{port}/a\n{SECRET}", spaces),
        )
        for fmt in FORMATS:
            for endpoint, reason in cases:
                with self.subTest(format=fmt, endpoint=endpoint):
                    self.assertRefused(
                        ["--format", fmt, "--endpoint", endpoint, "--header", "X-Probe-Key: k"],
                        f"invalid --endpoint value: {reason}",
                    )

    def test_endpoint_where_it_is_not_used(self):
        # E20 to E23. main printed or wrote the export, sent nothing and
        # exited 0, or sent to another URL.
        anonymize = "--endpoint is not used with --anonymize, which only writes a file; use --output"
        backend_otlp = "--endpoint is not used with --backend otlp; use --otlp-endpoint"
        langfuse = "use --langfuse-host, or --otlp-endpoint with --backend otlp"
        cases = (
            ((), f"--endpoint is not used with --format json; {FORMATS_HINT}"),
            (("--format", "csv"), f"--endpoint is not used with --format csv; {FORMATS_HINT}"),
            (("--format", "ndjson"), f"--endpoint is not used with --format ndjson; {FORMATS_HINT}"),
            (("--format", "eu-ai-act"), f"--endpoint is not used with --format eu-ai-act; {FORMATS_HINT}"),
            (("--anonymize",), anonymize),
            (("--anonymize", "--format", "otlp"), anonymize),
            (("--backend", "otlp"), backend_otlp),
            (("--backend", "otlp", "--metrics", "--format", "otlp"), backend_otlp),
            (("--backend", "langfuse"), "--endpoint is not used with --backend langfuse; use --langfuse-host"),
            (("--scores",), f"--endpoint is not used with --scores; {langfuse}"),
            (("--scores", "--metrics"), f"--endpoint is not used with --scores; {langfuse}"),
            (("--metrics", "--format", "otlp"), f"--endpoint is not used with --metrics; {langfuse}"),
        )
        endpoint = f"{self.base}/?api_key={SECRET}"
        for flags, line in cases:
            with self.subTest(flags=flags):
                self.assertRefused([*flags, "--endpoint", endpoint, "--header", "X-Probe-Key: k"], line)
        # E23 with OTEL_EXPORTER_OTLP_ENDPOINT set: main sent metrics there.
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": self.base}):
            self.assertRefused(["--backend", "otlp", "--endpoint", endpoint], backend_otlp)

    def test_header(self):
        # E24 to E30. main dropped E24, sent E25 and E27 as given, sent only
        # the last of E28, and printed a traceback for E26, E29 and E30, with
        # the secret for E29.
        no_colon = "it has no colon; give it as NAME: VALUE"
        same = "it has the same name as --header 1 (names ignore case)"
        cases = (
            ((f"Authorization {SECRET}",), f"invalid --header 1: {no_colon}"),
            ((f"X {SECRET}: v",), f"invalid --header 1: {NAME_RULE}"),
            ((f": {SECRET}",), "invalid --header 1: its name is empty"),
            ((f"{SECRET}:",), "invalid --header 1: its value is empty"),
            (("X-Probe-Key: a", f"x-probe-key: {SECRET}"), f"invalid --header 2: {same}"),
            ((f"X-Probe-Key: a\n{SECRET}",), f"invalid --header 1: {VALUE_RULE}"),
            ((f"X-Probe-Key: {SECRET}\u2603",), f"invalid --header 1: {VALUE_RULE}"),
            (("A: 1", "B: 2", f"a: {SECRET}"), f"invalid --header 3: {same}"),
            (("A: 1", "B: 2", f"C {SECRET}"), f"invalid --header 3: {no_colon}"),
        )
        for fmt in FORMATS:
            for headers, line in cases:
                with self.subTest(format=fmt, headers=headers):
                    flags = ["--format", fmt, "--endpoint", self.base]
                    for header in headers:
                        flags += ["--header", header]
                    self.assertRefused(flags, line)

    def test_first_problem_wins(self):
        self.assertRefused(
            ["--endpoint", "", "--header", "bad"],
            f"--endpoint is not used with --format json; {FORMATS_HINT}",
        )
        self.assertRefused(
            ["--format", "otlp", "--endpoint", "", "--header", "bad"],
            "invalid --endpoint value: it is empty",
        )
        self.assertRefused(
            ["--format", "otlp", "--endpoint", self.base, "--header", "A: 1",
             "--header", "bad", "--header", ": x"],
            "invalid --header 2: it has no colon; give it as NAME: VALUE",
        )

    def test_refusal_comes_before_the_session_lookup(self):
        # main reported the missing session instead.
        self.assertRefused(
            ["--format", "otlp", "--endpoint", ""], "invalid --endpoint value: it is empty",
            session_id="no-such-session",
        )


class TestExportRefusalThroughTheCli(_ExportCase):
    def test_one_line_and_no_traceback(self):
        # E19 and E29 printed a traceback on main, E29 with the secret.
        src = os.path.dirname(os.path.dirname(os.path.abspath(agent_trace.__file__)))
        home = os.path.join(self.root, "home")
        os.makedirs(home)
        env = {
            **os.environ,
            "PYTHONPATH": src,
            "HOME": home,
            "DO_NOT_TRACK": "1",
            "AGENT_STRACE_TELEMETRY": "0",
        }
        cases = (
            (["--endpoint", "  "], "invalid --endpoint value: it is empty\n"),
            (
                ["--endpoint", "http://127.0.0.1:9", "--header", f"X-Probe-Key: a\n{SECRET}"],
                f"invalid --header 1: {VALUE_RULE}\n",
            ),
        )
        for flags, line in cases:
            with self.subTest(flags=flags):
                run = subprocess.run(
                    [sys.executable, "-m", "agent_trace.entry", "--trace-dir", self.trace_dir,
                     "export", self.session_id, "--format", "otlp", *flags],
                    cwd=self.cwd, env=env, capture_output=True, text=True, timeout=60,
                )
                self.assertEqual((run.returncode, run.stdout, run.stderr), (1, "", line))
                self.assertEqual(os.listdir(self.cwd), [])


class TestExportWithoutEndpoint(_ExportCase):
    """Without --endpoint, export works as before."""

    def test_header_is_not_read(self):
        code, out, err = self.export("--format", "otlp", "--header", "not a header")
        self.assertEqual((code, out), (0, ""))
        self.assertEqual(os.listdir(self.cwd), [f"trace-{self.session_id[:12]}-otlp.json"])

    def test_ndjson(self):
        code, out, err = self.export("--format", "ndjson")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(len(out.splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
