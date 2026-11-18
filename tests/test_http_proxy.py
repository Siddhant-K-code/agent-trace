"""Tests for HTTP/SSE MCP proxy."""

import hashlib
import http.client
import io
import json
import os
import queue
import select
import shutil
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cli import build_parser, cmd_record_http
from agent_trace.http_proxy import HTTPProxyServer, _EventStreamParser, _ProxyHTTPServer
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


class FakeMCPHandler(BaseHTTPRequestHandler):
    """Simulates a remote MCP server."""

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            msg = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(400)
            return

        # respond to tool calls with a result
        if msg.get("method") == "tools/call":
            resp_id = msg.get("id")
            # simulate a server that answers with an id nobody asked for
            if msg.get("params", {}).get("name") == "mismatched_id":
                resp_id = f"other-{resp_id}"
            if msg.get("params", {}).get("name") == "fails":
                response = {
                    "jsonrpc": "2.0",
                    "id": resp_id,
                    "error": {"code": -32602, "message": "invalid arguments"},
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": resp_id,
                    "result": {
                        "content": [{"type": "text", "text": "tool output"}],
                    },
                }
        elif msg.get("method") == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "fake-mcp", "version": "0.1.0"},
                },
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": msg.get("id"),
                "result": {},
            }

        resp_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self):
        """Serve a short SSE stream with one JSON-RPC notification."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        notification = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {"message": "hello from server"},
        })
        self.wfile.write(f"data: {notification}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_DELETE(self):
        """End the session, as a Streamable HTTP server does."""
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        pass


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestHTTPProxy(unittest.TestCase):
    """Integration tests for the HTTP proxy."""

    @classmethod
    def setUpClass(cls):
        # start fake MCP server
        cls.fake_port = _find_free_port()
        cls.fake_server = HTTPServer(("127.0.0.1", cls.fake_port), FakeMCPHandler)
        cls.fake_thread = threading.Thread(target=cls.fake_server.serve_forever, daemon=True)
        cls.fake_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.fake_server.shutdown()

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)
        self.events = []

        self.meta = SessionMeta(
            command="test",
            agent_name="test-agent",
        )
        self.store.create_session(self.meta)

        self.proxy_port = _find_free_port()
        self.proxy = HTTPProxyServer(
            remote_url=f"http://127.0.0.1:{self.fake_port}",
            local_port=self.proxy_port,
            store=self.store,
            session_meta=self.meta,
            on_event=lambda e: self.events.append(e),
        )

        self.proxy_server = self.proxy.make_server()
        self.proxy_thread = threading.Thread(target=self.proxy_server.serve_forever, daemon=True)
        self.proxy_thread.start()
        time.sleep(0.05)  # let server start

    def tearDown(self):
        self.proxy_server.shutdown()
        self.proxy_server.server_close()

    def test_tool_call_traced(self):
        """POST a tools/call and verify both call and result are traced."""
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port)
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "read_file", "arguments": {"path": "/tmp/test.txt"}},
        })
        conn.request("POST", "/message", body=body, headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()

        self.assertEqual(resp.status, 200)

        # parse response
        resp_data = json.loads(resp_body.decode("utf-8"))
        self.assertIn("result", resp_data)

        # check traced events
        tool_calls = [e for e in self.events if e.event_type == EventType.TOOL_CALL]
        tool_results = [e for e in self.events if e.event_type == EventType.TOOL_RESULT]

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].data["tool_name"], "read_file")

        self.assertEqual(len(tool_results), 1)
        # result should link back to call
        self.assertEqual(tool_results[0].parent_id, tool_calls[0].event_id)
        self.assertGreater(tool_results[0].duration_ms, 0)

    def test_tool_call_with_an_array_or_object_id_is_traced_without_a_link(self):
        """An id that is not a string or number is forwarded but cannot link the result."""
        for request_id in ([1], {"k": 1}):
            with self.subTest(request_id=request_id):
                del self.events[:]
                conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port)
                body = json.dumps({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": "read_file", "arguments": {"path": "/tmp/test.txt"}},
                })
                conn.request("POST", "/message", body=body, headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                })
                resp = conn.getresponse()
                resp_body = resp.read()
                conn.close()

                self.assertEqual(resp.status, 200)
                self.assertEqual(json.loads(resp_body.decode("utf-8"))["id"], request_id)
                tool_calls = [e for e in self.events if e.event_type == EventType.TOOL_CALL]
                tool_results = [e for e in self.events if e.event_type == EventType.TOOL_RESULT]
                self.assertEqual(len(tool_calls), 1)
                self.assertEqual(len(tool_results), 1)
                self.assertEqual(tool_results[0].parent_id, "")

    def test_initialize_traced(self):
        """POST an initialize request and verify it's traced as LLM_REQUEST."""
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port)
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        conn.request("POST", "/message", body=body, headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        resp.read()
        conn.close()

        self.assertEqual(resp.status, 200)

    def test_proxy_forwards_auth_header(self):
        """Auth header should be forwarded to the remote server."""
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port)
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
            "params": {},
        })
        conn.request("POST", "/message", body=body, headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Authorization": "Bearer test-token",
        })
        resp = conn.getresponse()
        resp.read()
        conn.close()

        # if auth was rejected we'd get an error; 200 means it was forwarded
        self.assertEqual(resp.status, 200)

    def test_sse_stream_traced(self):
        """GET /sse should stream SSE events and trace them."""
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port)
        conn.request("GET", "/sse", headers={"Accept": "text/event-stream"})
        resp = conn.getresponse()

        # read the streamed data
        data = resp.read().decode("utf-8")
        conn.close()

        self.assertEqual(resp.status, 200)
        self.assertIn("data:", data)

    def test_meta_counts_updated(self):
        """Tool calls should increment meta.tool_calls."""
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port)
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "exec", "arguments": {"cmd": "ls"}},
        })
        conn.request("POST", "/message", body=body, headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        resp.read()
        conn.close()

        self.assertGreaterEqual(self.meta.tool_calls, 1)

    def test_events_persisted_to_store(self):
        """Events should be written to the trace store."""
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port)
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {"name": "write_file", "arguments": {"path": "/tmp/out.txt", "content": "hi"}},
        })
        conn.request("POST", "/message", body=body, headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        resp.read()
        conn.close()

        stored_events = self.store.load_events(self.meta.session_id)
        self.assertGreater(len(stored_events), 0)

        event_types = [e.event_type for e in stored_events]
        self.assertIn(EventType.TOOL_CALL, event_types)
        self.assertIn(EventType.TOOL_RESULT, event_types)


class TestHTTPProxyRedaction(unittest.TestCase):
    """Test that redaction works through the HTTP proxy."""

    @classmethod
    def setUpClass(cls):
        cls.fake_port = _find_free_port()
        cls.fake_server = HTTPServer(("127.0.0.1", cls.fake_port), FakeMCPHandler)
        cls.fake_thread = threading.Thread(target=cls.fake_server.serve_forever, daemon=True)
        cls.fake_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.fake_server.shutdown()

    def test_redaction_enabled(self):
        """With redact=True, secrets in tool args should be redacted in traces."""
        tmpdir = tempfile.mkdtemp()
        store = TraceStore(tmpdir)
        events = []

        meta = SessionMeta(command="test", agent_name="test-agent")
        store.create_session(meta)

        proxy_port = _find_free_port()

        proxy_server = _ProxyHTTPServer(
            ("127.0.0.1", proxy_port),
            remote_url=f"http://127.0.0.1:{self.fake_port}",
            store=store,
            meta=meta,
            on_event=events.append,
            redact=True,
        )
        proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
        proxy_thread.start()
        time.sleep(0.05)

        try:
            conn = http.client.HTTPConnection("127.0.0.1", proxy_port)
            body = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "http_request",
                    "arguments": {
                        "api_key": "sk-abc123def456ghi789jkl012mno345pqr678",
                        "url": "https://api.example.com",
                    },
                },
            })
            conn.request("POST", "/message", body=body, headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            })
            resp = conn.getresponse()
            resp.read()
            conn.close()

            tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
            self.assertEqual(len(tool_calls), 1)

            # the api_key should be redacted
            args = tool_calls[0].data.get("arguments", {})
            self.assertNotIn("sk-abc", str(args))
        finally:
            proxy_server.shutdown()
            proxy_server.server_close()


def _post_tool_call(port, req_id, name="read_file"):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": {}},
    })
    conn.request("POST", "/message", body=body, headers={
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp.status


class TestHTTPProxyPendingCalls(unittest.TestCase):
    """Tool call/result correlation state is per server, bounded, and cleared."""

    @classmethod
    def setUpClass(cls):
        cls.fake_port = _find_free_port()
        cls.fake_server = HTTPServer(("127.0.0.1", cls.fake_port), FakeMCPHandler)
        cls.fake_thread = threading.Thread(target=cls.fake_server.serve_forever, daemon=True)
        cls.fake_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.fake_server.shutdown()
        cls.fake_server.server_close()

    def _start_proxy(self):
        store = TraceStore(tempfile.mkdtemp())
        meta = SessionMeta(command="test", agent_name="test-agent")
        store.create_session(meta)
        events = []
        proxy = HTTPProxyServer(
            remote_url=f"http://127.0.0.1:{self.fake_port}",
            local_port=_find_free_port(),
            store=store,
            session_meta=meta,
            on_event=events.append,
        )
        server = proxy.make_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()

        self.addCleanup(stop)
        return server, events

    def _make_server(self, **kwargs):
        server = _ProxyHTTPServer(
            ("127.0.0.1", 0),
            remote_url="http://127.0.0.1:1",
            **kwargs,
        )
        self.addCleanup(server.server_close)
        return server

    def test_two_servers_keep_independent_pending_maps(self):
        a = self._make_server()
        b = self._make_server()
        self.assertIsNot(a.calls, b.calls)

        call_a = TraceEvent(event_type=EventType.TOOL_CALL, data={"request_id": 1})
        a.track_call(call_a)
        self.assertIn(1, a.pending_calls)
        self.assertNotIn(1, b.pending_calls)

        # a result with the same id on server b must not steal a's call
        result_b = TraceEvent(event_type=EventType.TOOL_RESULT, data={"request_id": 1})
        b.match_result(result_b)
        self.assertFalse(result_b.parent_id)
        self.assertIn(1, a.pending_calls)

        result_a = TraceEvent(event_type=EventType.TOOL_RESULT, data={"request_id": 1})
        a.match_result(result_a)
        self.assertEqual(result_a.parent_id, call_a.event_id)
        self.assertEqual(a.pending_calls, {})

    def test_two_live_proxies_link_results_to_their_own_calls(self):
        server_a, events_a = self._start_proxy()
        server_b, events_b = self._start_proxy()

        # same JSON-RPC id on both proxies
        self.assertEqual(_post_tool_call(server_a.server_address[1], 7, "tool_a"), 200)
        self.assertEqual(_post_tool_call(server_b.server_address[1], 7, "tool_b"), 200)

        for events in (events_a, events_b):
            calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
            results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].parent_id, calls[0].event_id)
            self.assertEqual(
                results[0].duration_ms,
                (results[0].timestamp - calls[0].timestamp) * 1000,
            )
            self.assertGreaterEqual(results[0].duration_ms, 0)

        self.assertEqual(server_a.pending_calls, {})
        self.assertEqual(server_b.pending_calls, {})

    def test_result_links_to_matching_call_among_several(self):
        server = self._make_server()
        call_1 = TraceEvent(event_type=EventType.TOOL_CALL, timestamp=100.0, data={"request_id": 1})
        call_2 = TraceEvent(event_type=EventType.TOOL_CALL, timestamp=101.0, data={"request_id": 2})
        server.track_call(call_1)
        server.track_call(call_2)

        result = TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=101.5, data={"request_id": 2})
        server.match_result(result)

        self.assertEqual(result.parent_id, call_2.event_id)
        self.assertAlmostEqual(result.duration_ms, 500.0)
        self.assertEqual(list(server.pending_calls), [1])

    def test_result_with_an_id_that_is_not_a_string_or_number_leaves_pending_calls_alone(self):
        for request_id in (True, [1], {"k": 1}):
            with self.subTest(request_id=request_id):
                server = self._make_server()
                server.track_call(TraceEvent(event_type=EventType.TOOL_CALL, data={"request_id": 1}))
                result = TraceEvent(event_type=EventType.TOOL_RESULT, data={"request_id": request_id})
                server.match_result(result)
                self.assertFalse(result.parent_id)
                self.assertEqual(list(server.pending_calls), [1])

    def test_unmatched_result_recorded_without_crash(self):
        server, events = self._start_proxy()
        status = _post_tool_call(server.server_address[1], 3, "mismatched_id")
        self.assertEqual(status, 200)

        calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].parent_id)
        self.assertEqual(results[0].data["request_id"], "other-3")
        # the call is still waiting for its real result
        self.assertIn(3, server.pending_calls)

        stored = server.store.load_events(server.meta.session_id)
        self.assertIn(EventType.TOOL_RESULT, [e.event_type for e in stored])

    def test_error_links_to_matching_call_and_closes_it(self):
        server = self._make_server()
        call = TraceEvent(event_type=EventType.TOOL_CALL, timestamp=100.0, data={"request_id": 4})
        server.track_call(call)

        error = TraceEvent(event_type=EventType.ERROR, timestamp=100.25, data={"request_id": 4})
        server.match_result(error)

        self.assertEqual(error.parent_id, call.event_id)
        self.assertAlmostEqual(error.duration_ms, 250.0)
        self.assertEqual(server.pending_calls, {})

    def test_live_error_response_links_to_its_call(self):
        server, events = self._start_proxy()
        self.assertEqual(_post_tool_call(server.server_address[1], 9, "fails"), 200)

        (call,) = [e for e in events if e.event_type == EventType.TOOL_CALL]
        (error,) = [e for e in events if e.event_type == EventType.ERROR]
        self.assertEqual(error.parent_id, call.event_id)
        self.assertGreaterEqual(error.duration_ms, 0)
        self.assertEqual(server.pending_calls, {})

    def test_pending_calls_are_bounded(self):
        server = self._make_server(max_pending_calls=3)
        for i in range(5):
            server.track_call(TraceEvent(event_type=EventType.TOOL_CALL, data={"request_id": i}))
        self.assertEqual(list(server.pending_calls), [2, 3, 4])

    def test_clear_pending_drops_unmatched_calls(self):
        server = self._make_server()
        server.track_call(TraceEvent(event_type=EventType.TOOL_CALL, data={"request_id": "x"}))
        self.assertEqual(server.clear_pending(), 1)
        self.assertEqual(server.pending_calls, {})

    def test_run_clears_pending_calls_at_session_end(self):
        store = TraceStore(tempfile.mkdtemp())
        meta = SessionMeta(command="test", agent_name="test-agent")
        store.create_session(meta)
        proxy = HTTPProxyServer(
            remote_url=f"http://127.0.0.1:{self.fake_port}",
            local_port=_find_free_port(),
            store=store,
            session_meta=meta,
        )
        created = []
        real_make_server = proxy.make_server

        def make_server():
            server = real_make_server()
            server.track_call(TraceEvent(event_type=EventType.TOOL_CALL, data={"request_id": 1}))
            created.append(server)
            return server

        proxy.make_server = make_server
        thread = threading.Thread(target=proxy.run, daemon=True)
        thread.start()
        for _ in range(100):
            if created:
                break
            time.sleep(0.01)
        created[0].shutdown()
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(created[0].pending_calls, {})
        event_types = [e.event_type for e in store.load_events(meta.session_id)]
        self.assertEqual(event_types[-1], EventType.SESSION_END)


class StreamingMCPHandler(FakeMCPHandler):
    """Fake MCP server whose SSE stream stays open until released."""

    release = threading.Event()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        notification = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {"message": "stream open"},
        })
        self.wfile.write(f"data: {notification}\n\n".encode("utf-8"))
        self.wfile.flush()
        self.release.wait(10)


class _ThreadingFakeServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def _open_sse(port):
    """Open GET /sse through the proxy and read up to the first data line."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/sse", headers={"Accept": "text/event-stream"})
    resp = conn.getresponse()
    while True:
        line = resp.readline()
        if not line or line.startswith(b"data: "):
            break
    return conn, resp, line


class TestHTTPProxyConcurrency(unittest.TestCase):
    """An open SSE stream must not block POSTs or shutdown."""

    @classmethod
    def setUpClass(cls):
        StreamingMCPHandler.release.clear()
        cls.fake_server = _ThreadingFakeServer(("127.0.0.1", 0), StreamingMCPHandler)
        cls.fake_port = cls.fake_server.server_address[1]
        threading.Thread(target=cls.fake_server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        StreamingMCPHandler.release.set()
        cls.fake_server.shutdown()
        cls.fake_server.server_close()

    def _make_proxy(self):
        store = TraceStore(tempfile.mkdtemp())
        meta = SessionMeta(command="test", agent_name="test-agent")
        store.create_session(meta)
        events = []
        proxy = HTTPProxyServer(
            remote_url=f"http://127.0.0.1:{self.fake_port}",
            local_port=_find_free_port(),
            store=store,
            session_meta=meta,
            on_event=events.append,
        )
        return proxy, events

    def _start_proxy(self):
        proxy, events = self._make_proxy()
        server = proxy.make_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()

        self.addCleanup(stop)
        return server, events

    def test_post_completes_while_sse_stream_open(self):
        server, events = self._start_proxy()
        port = server.server_address[1]
        conn, _resp, line = _open_sse(port)
        self.addCleanup(conn.close)
        self.assertTrue(line.startswith(b"data: "))

        result = {}

        def post():
            result["status"] = _post_tool_call(port, 42, "while_streaming")

        poster = threading.Thread(target=post, daemon=True)
        poster.start()
        poster.join(timeout=5)

        self.assertFalse(poster.is_alive(), "POST blocked by open SSE stream")
        self.assertEqual(result.get("status"), 200)
        calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].parent_id, calls[0].event_id)
        # the stream is still being served
        self.assertTrue(_wait_until(lambda: server._active_requests == 1))

    def test_server_close_ends_open_stream(self):
        server, events = self._start_proxy()
        conn, resp, line = _open_sse(server.server_address[1])
        self.addCleanup(conn.close)
        self.assertTrue(line.startswith(b"data: "))

        start = time.monotonic()
        server.shutdown()
        server.server_close()
        elapsed = time.monotonic() - start

        # the fake upstream holds the stream for 10s; close must not wait for it
        self.assertLess(elapsed, 3)
        self.assertTrue(server.close_streams(timeout=0))
        # the proxy closed the stream: read returns at EOF instead of timing out
        self.assertEqual(resp.read().strip(), b"")
        self.assertEqual(
            [e for e in events if e.event_type == EventType.ERROR], []
        )

    def test_run_records_session_end_last_with_open_stream(self):
        proxy, _events = self._make_proxy()
        created = []
        real_make_server = proxy.make_server

        def make_server():
            server = real_make_server()
            created.append(server)
            return server

        proxy.make_server = make_server
        thread = threading.Thread(target=proxy.run, daemon=True)
        thread.start()
        for _ in range(100):
            if created:
                break
            time.sleep(0.01)
        conn, _resp, line = _open_sse(proxy.local_port)
        self.addCleanup(conn.close)
        self.assertTrue(line.startswith(b"data: "))

        created[0].shutdown()
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        event_types = [e.event_type for e in proxy.store.load_events(proxy.meta.session_id)]
        self.assertEqual(event_types[0], EventType.SESSION_START)
        self.assertEqual(event_types[-1], EventType.SESSION_END)
        self.assertEqual(event_types.count(EventType.SESSION_END), 1)
        self.assertNotIn(EventType.ERROR, event_types)

    def test_concurrent_posts_keep_store_and_counts_consistent(self):
        server, events = self._start_proxy()
        port = server.server_address[1]
        statuses = []
        workers = [
            threading.Thread(
                target=lambda i=i: statuses.append(_post_tool_call(port, i, f"tool_{i}")),
                daemon=True,
            )
            for i in range(20)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=10)

        self.assertEqual(statuses, [200] * 20)
        self.assertEqual(server.meta.tool_calls, 20)
        self.assertEqual(len(events), 40)
        self.assertEqual(server.pending_calls, {})

        results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        calls = {e.event_id: e for e in events if e.event_type == EventType.TOOL_CALL}
        for result in results:
            self.assertEqual(
                calls[result.parent_id].data["request_id"], result.data["request_id"]
            )

        path = os.path.join(
            server.store.base_dir, server.meta.session_id, "events.ndjson"
        )
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        self.assertEqual(len(lines), 40)
        for previous, line in zip(lines, lines[1:]):
            self.assertEqual(
                json.loads(line)["prev_hash"],
                hashlib.sha256(previous.encode()).hexdigest(),
            )

    def test_events_after_close_are_dropped(self):
        server, events = self._start_proxy()
        server.shutdown()
        server.server_close()
        server.emit(TraceEvent(event_type=EventType.TOOL_CALL, data={"request_id": 1}))
        self.assertEqual(events, [])
        self.assertEqual(server.meta.tool_calls, 0)
        self.assertEqual(server.store.load_events(server.meta.session_id), [])


# Test-only certificates. See tests/fixtures/tls/generate.sh.
TLS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "tls")
TLS_CA = os.path.join(TLS_DIR, "ca.pem")
TLS_KEY = os.path.join(TLS_DIR, "server.key")
TLS_LOCALHOST_CERT = os.path.join(TLS_DIR, "localhost.pem")
TLS_WRONG_HOST_CERT = os.path.join(TLS_DIR, "wrong-host.pem")


class TLSMCPHandler(FakeMCPHandler):
    """Fake MCP server for HTTPS tests.

    POST /fail answers 503 with a plain body. A tools/call named "via_sse"
    gets 202 and its result is pushed onto the open GET /sse stream, as in
    the MCP SSE transport. GET /sse stays open until the server releases it.
    """

    server: "_TLSUpstream"

    def do_POST(self):
        self.server.hosts.append(self.headers.get("Host"))
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if self.path.endswith("/fail"):
            body = b"upstream unavailable"
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        msg = json.loads(raw.decode("utf-8"))
        if msg.get("params", {}).get("name") == "via_sse":
            self.server.sse_queue.put({
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": {"content": [{"type": "text", "text": "streamed"}]},
            })
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # let FakeMCPHandler answer the body we already read
        self.rfile = io.BytesIO(raw)
        super().do_POST()

    def do_GET(self):
        self.server.hosts.append(self.headers.get("Host"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b": stream open\n\n")
        self.wfile.flush()
        while not self.server.release.is_set():
            try:
                msg = self.server.sse_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode("utf-8"))
            self.wfile.flush()


class _TLSUpstream(ThreadingHTTPServer):
    """Threaded HTTPS server that does the TLS handshake on the handler thread."""

    daemon_threads = True
    request_queue_size = 128

    def __init__(self, certfile):
        super().__init__(("127.0.0.1", 0), TLSMCPHandler)
        self.tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.tls.load_cert_chain(certfile, TLS_KEY)
        self.tls.sni_callback = self._record_sni
        self.sni: list = []
        self.hosts: list = []
        self.handshake_errors: list = []
        self.sse_queue: queue.Queue = queue.Queue()
        self.release = threading.Event()

    def _record_sni(self, _sock, server_name, _context):
        self.sni.append(server_name)

    def finish_request(self, request, client_address):
        try:
            tls_sock = self.tls.wrap_socket(request, server_side=True)
        except OSError as exc:
            self.handshake_errors.append(exc)
            return
        try:
            super().finish_request(tls_sock, client_address)
        finally:
            tls_sock.close()


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _post_json(port, msg, path="/message"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = json.dumps(msg)
    conn.request("POST", path, body=body, headers={
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def _tool_call(req_id, name="read_file"):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": {"path": "/tmp/x"}},
    }


class TestHTTPProxyHTTPSUpstream(unittest.TestCase):
    """The proxy forwards to HTTPS upstreams and verifies their certificates.

    By default the proxy trusts the system CA store. A custom CA is trusted
    either with ca_file (``record-http --ca-file``) or through OpenSSL's
    SSL_CERT_FILE variable, which most tests set to the test CA.
    """

    def _upstream(self, certfile=TLS_LOCALHOST_CERT):
        upstream = _TLSUpstream(certfile)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()

        def stop():
            upstream.release.set()
            upstream.shutdown()
            upstream.server_close()

        self.addCleanup(stop)
        return upstream

    def _proxy(self, remote_url, ca_file=""):
        store = TraceStore(tempfile.mkdtemp())
        meta = SessionMeta(command="test", agent_name="test-agent")
        store.create_session(meta)
        events = []
        proxy = HTTPProxyServer(
            remote_url=remote_url,
            local_port=0,
            store=store,
            session_meta=meta,
            on_event=events.append,
            ca_file=ca_file,
        )
        server = proxy.make_server()
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def stop():
            server.shutdown()
            server.server_close()

        self.addCleanup(stop)
        return server, events

    def _trust(self, cafile):
        """Point OpenSSL's default CA file at cafile, or at nothing."""
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SSL_CERT_DIR", None)
        if cafile:
            os.environ["SSL_CERT_FILE"] = cafile
        else:
            os.environ.pop("SSL_CERT_FILE", None)

    def _assert_idle(self, server):
        """Every request finished and no upstream socket is still registered."""
        self.assertTrue(_wait_until(
            lambda: server._active_requests == 0 and not server._open_sockets
        ))

    def _errors(self, events):
        return [e for e in events if e.event_type == EventType.ERROR]

    def test_tool_call_and_result_are_paired(self):
        self._trust(TLS_CA)
        upstream = self._upstream()
        port = upstream.server_address[1]
        server, events = self._proxy(f"https://127.0.0.1:{port}")

        status, body = _post_json(server.server_address[1], _tool_call(1))

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["result"]["content"][0]["text"], "tool output")
        (call,) = [e for e in events if e.event_type == EventType.TOOL_CALL]
        (result,) = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(call.data["tool_name"], "read_file")
        self.assertEqual(result.parent_id, call.event_id)
        self.assertGreaterEqual(result.duration_ms, 0)
        self.assertEqual(server.pending_calls, {})
        stored = [e.event_type for e in server.store.load_events(server.meta.session_id)]
        self.assertEqual(stored, [EventType.TOOL_CALL, EventType.TOOL_RESULT])
        # Host names the upstream, not the proxy; no SNI is sent for an IP
        self.assertEqual(upstream.hosts, [f"127.0.0.1:{port}"])
        self.assertEqual(upstream.sni, [None])
        self._assert_idle(server)

    def test_host_header_and_sni_name_the_upstream_host(self):
        self._trust(TLS_CA)
        upstream = self._upstream()
        port = upstream.server_address[1]
        server, events = self._proxy(f"https://localhost:{port}")

        status, _body = _post_json(server.server_address[1], _tool_call(2))

        self.assertEqual(status, 200)
        self.assertEqual(upstream.hosts, [f"localhost:{port}"])
        self.assertEqual(upstream.sni, ["localhost"])
        self.assertEqual(self._errors(events), [])

    def test_json_rpc_error_links_to_its_call(self):
        self._trust(TLS_CA)
        upstream = self._upstream()
        server, events = self._proxy(f"https://127.0.0.1:{upstream.server_address[1]}")

        status, body = _post_json(server.server_address[1], _tool_call(3, "fails"))

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["error"]["code"], -32602)
        (call,) = [e for e in events if e.event_type == EventType.TOOL_CALL]
        (error,) = self._errors(events)
        self.assertEqual(error.parent_id, call.event_id)
        self.assertEqual(server.pending_calls, {})

    def test_upstream_http_error_is_passed_through(self):
        self._trust(TLS_CA)
        upstream = self._upstream()
        server, events = self._proxy(f"https://127.0.0.1:{upstream.server_address[1]}")

        status, body = _post_json(server.server_address[1], _tool_call(4), path="/fail")

        self.assertEqual(status, 503)
        self.assertEqual(body, b"upstream unavailable")
        # a non-JSON body is not a result, so the call stays open
        self.assertEqual(self._errors(events), [])
        self.assertIn(4, server.pending_calls)
        self._assert_idle(server)

    def test_sse_stream_carries_result_for_post(self):
        self._trust(TLS_CA)
        upstream = self._upstream()
        server, events = self._proxy(f"https://127.0.0.1:{upstream.server_address[1]}")
        port = server.server_address[1]

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)
        conn.request("GET", "/sse", headers={"Accept": "text/event-stream"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
        self.assertEqual(resp.readline(), b": stream open\n")

        status, body = _post_json(port, _tool_call(5, "via_sse"))
        self.assertEqual((status, body), (202, b""))

        line = resp.readline()
        while line and not line.startswith(b"data: "):
            line = resp.readline()
        self.assertEqual(json.loads(line[6:])["id"], 5)

        self.assertTrue(_wait_until(
            lambda: any(e.event_type == EventType.TOOL_RESULT for e in events)
        ))
        (call,) = [e for e in events if e.event_type == EventType.TOOL_CALL]
        (result,) = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(result.parent_id, call.event_id)
        self.assertEqual(server.pending_calls, {})

    def test_server_close_ends_open_https_stream(self):
        self._trust(TLS_CA)
        upstream = self._upstream()
        server, events = self._proxy(f"https://127.0.0.1:{upstream.server_address[1]}")

        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        self.addCleanup(conn.close)
        conn.request("GET", "/sse")
        resp = conn.getresponse()
        self.assertEqual(resp.readline(), b": stream open\n")

        start = time.monotonic()
        server.shutdown()
        server.server_close()

        self.assertLess(time.monotonic() - start, 3)
        self.assertTrue(server.close_streams(timeout=0))
        self.assertEqual(resp.read().strip(), b"")
        self.assertEqual(self._errors(events), [])

    def test_untrusted_certificate_is_rejected(self):
        self._trust(None)
        upstream = self._upstream()
        server, events = self._proxy(f"https://127.0.0.1:{upstream.server_address[1]}")
        port = server.server_address[1]

        status, _body = _post_json(port, _tool_call(6))
        self.assertEqual(status, 502)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/sse")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 502)

        messages = [e.data["message"] for e in self._errors(events)]
        self.assertEqual(len(messages), 2)
        self.assertTrue(messages[0].startswith("Proxy error: "))
        self.assertTrue(messages[1].startswith("SSE proxy error: "))
        for message in messages:
            self.assertIn("CERTIFICATE_VERIFY_FAILED", message)
        # the request was never sent upstream
        self.assertEqual(upstream.hosts, [])
        self._assert_idle(server)

    def test_certificate_for_another_host_is_rejected(self):
        self._trust(TLS_CA)
        upstream = self._upstream(TLS_WRONG_HOST_CERT)
        server, events = self._proxy(f"https://127.0.0.1:{upstream.server_address[1]}")

        status, _body = _post_json(server.server_address[1], _tool_call(7))

        self.assertEqual(status, 502)
        (error,) = self._errors(events)
        self.assertIn("CERTIFICATE_VERIFY_FAILED", error.data["message"])
        self.assertEqual(upstream.hosts, [])
        self._assert_idle(server)

    def test_ca_file_trusts_a_private_ca(self):
        self._trust(None)
        upstream = self._upstream()
        port = upstream.server_address[1]
        server, events = self._proxy(f"https://localhost:{port}", ca_file=TLS_CA)
        proxy_port = server.server_address[1]

        status, body = _post_json(proxy_port, _tool_call(10))

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["result"]["content"][0]["text"], "tool output")
        (call,) = [e for e in events if e.event_type == EventType.TOOL_CALL]
        (result,) = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(result.parent_id, call.event_id)
        self.assertEqual(upstream.sni, ["localhost"])

        conn = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=5)
        self.addCleanup(conn.close)
        conn.request("GET", "/sse")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.readline(), b": stream open\n")
        self.assertEqual(self._errors(events), [])

    def test_ca_file_still_rejects_a_certificate_for_another_host(self):
        self._trust(None)
        upstream = self._upstream(TLS_WRONG_HOST_CERT)
        server, events = self._proxy(
            f"https://127.0.0.1:{upstream.server_address[1]}", ca_file=TLS_CA
        )

        status, _body = _post_json(server.server_address[1], _tool_call(11))

        self.assertEqual(status, 502)
        (error,) = self._errors(events)
        self.assertIn("CERTIFICATE_VERIFY_FAILED", error.data["message"])
        self.assertEqual(upstream.hosts, [])
        self._assert_idle(server)

    def test_ca_file_replaces_the_default_ca_store(self):
        # SSL_CERT_FILE would trust the upstream, but ca_file names a bundle
        # that does not hold its issuer, so the default store is not used.
        self._trust(TLS_CA)
        upstream = self._upstream()
        server, events = self._proxy(
            f"https://127.0.0.1:{upstream.server_address[1]}",
            ca_file=TLS_WRONG_HOST_CERT,
        )

        status, _body = _post_json(server.server_address[1], _tool_call(12))

        self.assertEqual(status, 502)
        (error,) = self._errors(events)
        self.assertIn("CERTIFICATE_VERIFY_FAILED", error.data["message"])
        self.assertEqual(upstream.hosts, [])

    def test_ca_file_errors_are_raised_before_the_proxy_starts(self):
        missing = os.path.join(tempfile.mkdtemp(), "missing.pem")
        cases = [
            ("http://127.0.0.1:1", TLS_CA, "--ca-file requires an https:// URL"),
            ("https://127.0.0.1:1", missing, "could not load CA file"),
            ("https://127.0.0.1:1", TLS_DIR, "could not load CA file"),
            # a PEM file without a certificate
            ("https://127.0.0.1:1", TLS_KEY, "could not load CA file"),
        ]
        for url, ca_file, message in cases:
            with self.subTest(url=url, ca_file=ca_file):
                store = TraceStore(tempfile.mkdtemp())
                with self.assertRaisesRegex(ValueError, message):
                    HTTPProxyServer(
                        remote_url=url,
                        local_port=0,
                        store=store,
                        session_meta=SessionMeta(command="test"),
                        ca_file=ca_file,
                    )

    def test_https_url_for_a_plain_http_server_is_an_error(self):
        self._trust(TLS_CA)
        plain = HTTPServer(("127.0.0.1", 0), FakeMCPHandler)
        threading.Thread(target=plain.serve_forever, daemon=True).start()
        self.addCleanup(plain.server_close)
        self.addCleanup(plain.shutdown)
        server, events = self._proxy(f"https://127.0.0.1:{plain.server_address[1]}")

        status, _body = _post_json(server.server_address[1], _tool_call(8))

        self.assertEqual(status, 502)
        (error,) = self._errors(events)
        self.assertTrue(error.data["message"].startswith("Proxy error: "))
        self._assert_idle(server)

    def test_server_close_interrupts_a_stalled_tls_handshake(self):
        self._trust(TLS_CA)
        for method in ("POST", "GET"):
            with self.subTest(method=method):
                # accepts TCP through the backlog but never answers the handshake
                stalled = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                stalled.bind(("127.0.0.1", 0))
                stalled.listen(8)
                self.addCleanup(stalled.close)
                server, events = self._proxy(f"https://127.0.0.1:{stalled.getsockname()[1]}")

                conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
                self.addCleanup(conn.close)
                body = json.dumps(_tool_call(9)) if method == "POST" else None
                conn.request(method, "/message", body=body)
                # the client socket and the upstream TLS socket
                self.assertTrue(_wait_until(lambda: len(server._open_sockets) == 2))

                start = time.monotonic()
                server.shutdown()
                server.server_close()

                self.assertLess(time.monotonic() - start, 3)
                self.assertTrue(server.close_streams(timeout=0))
                self.assertEqual(server._open_sockets, set())
                self.assertEqual(self._errors(events), [])


class TestRecordHTTPCAFileFlag(unittest.TestCase):
    """record-http --ca-file is checked before a session is created."""

    def setUp(self):
        self.trace_dir = tempfile.mkdtemp()
        patcher = mock.patch(
            "agent_trace.cli.HTTPProxyServer.run",
            side_effect=AssertionError("proxy must not start"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, *argv):
        args = build_parser().parse_args(
            ["--trace-dir", self.trace_dir, "record-http", *argv]
        )
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            code = cmd_record_http(args)
        return code, stderr.getvalue()

    def test_flag_is_parsed(self):
        args = build_parser().parse_args(
            ["record-http", "--url", "https://example.com", "--ca-file", TLS_CA]
        )
        self.assertEqual(args.ca_file, TLS_CA)
        args = build_parser().parse_args(["record-http", "--url", "https://example.com"])
        self.assertIsNone(args.ca_file)

    def test_startup_errors_exit_2_without_a_session(self):
        missing = os.path.join(self.trace_dir, "missing.pem")
        cases = [
            ("http://127.0.0.1:1", TLS_CA, "record-http: --ca-file requires an https:// URL\n"),
            ("https://127.0.0.1:1", missing, f"record-http: could not load CA file {missing}: "),
            ("https://127.0.0.1:1", TLS_KEY, f"record-http: could not load CA file {TLS_KEY}: "),
        ]
        for url, ca_file, message in cases:
            with self.subTest(url=url, ca_file=ca_file):
                code, stderr = self._run("--url", url, "--ca-file", ca_file)
                self.assertEqual(code, 2)
                self.assertTrue(stderr.startswith(message), stderr)
                self.assertEqual(TraceStore(self.trace_dir).list_sessions(), [])

    @unittest.skipIf(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        "root can read a file without read permission",
    )
    def test_unreadable_ca_file_exits_2(self):
        unreadable = os.path.join(self.trace_dir, "ca.pem")
        with open(TLS_CA, "rb") as src, open(unreadable, "wb") as dst:
            dst.write(src.read())
        os.chmod(unreadable, 0)
        self.addCleanup(os.chmod, unreadable, 0o600)

        code, stderr = self._run("--url", "https://127.0.0.1:1", "--ca-file", unreadable)

        self.assertEqual(code, 2)
        self.assertIn("could not load CA file", stderr)
        self.assertIn("Permission denied", stderr)
        self.assertEqual(TraceStore(self.trace_dir).list_sessions(), [])


HOST_REFUSED = {
    "jsonrpc": "2.0",
    "id": None,
    "error": {
        "code": -32600,
        "message": (
            "Host not allowed: connect to localhost or an IP address, "
            "or start record-http with --allow-host NAME"
        ),
    },
}
ORIGIN_REFUSED = {
    "jsonrpc": "2.0",
    "id": None,
    "error": {
        "code": -32600,
        "message": (
            "Origin not allowed: only pages on localhost, a loopback address "
            "or an --allow-host name can use this proxy"
        ),
    },
}
HOST_REFUSED_LOG = (
    "agent-strace: refused a request (421 Host not allowed). Connect to "
    "localhost or an IP address, or restart record-http with --allow-host NAME.\n"
)
ORIGIN_REFUSED_LOG = (
    "agent-strace: refused a request (403 Origin not allowed). Only pages on "
    "localhost, a loopback address or an --allow-host name can use this proxy. "
    "To allow pages on another name, restart record-http with --allow-host NAME.\n"
)

# Origins of pages served by this machine.
LOOPBACK_ORIGINS = [
    "http://localhost",
    "http://localhost:6274",
    "https://localhost:8443",
    "http://127.0.0.1:3000",
    "http://127.0.0.2",
    "http://[::1]:5173",
    "HTTP://LOCALHOST:6274",
]
REFUSED_ORIGINS = [
    "null",
    "",
    "http://192.0.2.10",
    "http://attacker.test",
    "http://localhost.attacker.test",
    "http://foo.localhost",
    "http://localhost.",
    "chrome-extension://abc",
    "file://",
    "ws://localhost:6274",
    "http://localhost:6274/",
    "http://user@localhost",
    "http://[::ffff:127.0.0.1]",
]

PING = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()


class _CountingUpstream(ThreadingHTTPServer):
    """A fake MCP upstream that counts the connections it accepts."""

    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), FakeMCPHandler)
        self._lock = threading.Lock()
        self._connections = 0

    @property
    def connections(self):
        with self._lock:
            return self._connections

    def process_request(self, request, client_address):
        with self._lock:
            self._connections += 1
        super().process_request(request, client_address)


def _start_counting_upstream(test):
    upstream = _CountingUpstream()
    threading.Thread(
        target=upstream.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    test.addCleanup(upstream.server_close)
    test.addCleanup(upstream.shutdown)
    return upstream


def _send(port, method, headers, body=None):
    """Send one request with exactly ``headers``, in order.

    No Host is sent unless ``headers`` has one. A body gets a Content-Length
    and, unless ``headers`` sets one, a JSON Content-Type. Returns the
    status, Content-Type and body of the reply.
    """
    path = "/message" if method == "POST" else "/sse"
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for name, value in headers:
            conn.putheader(name, value)
        if body is not None:
            if not any(name.lower() == "content-type" for name, _ in headers):
                conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        resp = conn.getresponse()
        return resp.status, resp.getheader("Content-Type"), resp.read()
    finally:
        conn.close()


def _read_all(sock):
    chunks = []
    while True:
        try:
            chunk = sock.recv(65536)
        except ConnectionResetError:
            chunk = b""
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


class TestHTTPProxyRefusesWebPages(unittest.TestCase):
    """The proxy refuses a DNS rebinding Host or a cross-site Origin.

    The check covers every method and runs before the body is read, so a
    refused request never reaches the upstream and adds nothing to the trace.
    """

    def setUp(self):
        self.upstream = _start_counting_upstream(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name)
        self.meta = SessionMeta(command="test", agent_name="test-agent")
        self.store.create_session(self.meta)
        self.events = []
        proxy = HTTPProxyServer(
            remote_url=f"http://127.0.0.1:{self.upstream.server_address[1]}",
            local_port=0,
            store=self.store,
            session_meta=self.meta,
            on_event=self.events.append,
        )
        self.server = proxy.make_server()
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]
        self.local_host = ("Host", f"127.0.0.1:{self.port}")

    def _request(self, method, headers, body=None):
        """Send one request. Returns status, Content-Type, body and stderr."""
        if body is None and method == "POST":
            body = PING
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            status, content_type, data = _send(self.port, method, headers, body)
        return status, content_type, data, stderr.getvalue()

    def _assert_forwarded(self, method, headers):
        status, _, data, stderr = self._request(method, headers)
        self.assertEqual(status, 200)
        if method == "POST":
            self.assertEqual(json.loads(data), {"jsonrpc": "2.0", "id": 1, "result": {}})
        self.assertEqual(stderr, "")

    def _assert_refused(self, method, headers, status, reply, log, body=None):
        got_status, content_type, data, stderr = self._request(method, headers, body)
        self.assertEqual(got_status, status)
        self.assertEqual(content_type, "application/json")
        self.assertEqual(json.loads(data), reply)
        self.assertEqual(stderr, log)

    def _assert_nothing_traced_or_forwarded(self):
        """Check that no request so far was traced or reached the upstream.

        A request that is forwarded goes last. Once its reply is back, the
        upstream has accepted every earlier connection, so the count is final.
        """
        self.assertEqual(self.events, [])
        self.assertEqual(self.store.load_events(self.meta.session_id), [])
        status, _, _ = _send(self.port, "GET", [self.local_host])
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.connections, 1)

    def _allowed_host_headers(self):
        other_port = self.port + 1 if self.port < 65535 else self.port - 1
        return [
            [("Host", "localhost")],
            [("Host", f"LOCALHOST:{self.port}")],
            [("Host", f"127.0.0.1:{self.port}")],
            [("Host", "127.0.0.2")],
            [("Host", f"[::1]:{self.port}")],
            [("Host", f"192.0.2.10:{self.port}")],
            [("Host", f"localhost:{other_port}")],
            [],
        ]

    def _refused_host_headers(self):
        return [
            [("Host", "attacker.test")],
            [("Host", f"attacker.test:{self.port}")],
            [("Host", "localhost.")],
            [("Host", f"localhost.:{self.port}")],
            [("Host", f"foo.localhost:{self.port}")],
            [("Host", "")],
            [("Host", "localhost:abc")],
            [("Host", "[::1")],
            [self.local_host, ("Host", "attacker.test")],
            [("Host", "attacker.test"), self.local_host],
        ]

    def test_local_requests_without_origin_are_forwarded_and_traced(self):
        status, _, _, get_stderr = self._request("GET", [self.local_host])
        self.assertEqual(status, 200)
        status, _, data, post_stderr = self._request(
            "POST", [self.local_host], json.dumps(_tool_call(3)).encode()
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["result"]["content"][0]["text"], "tool output")
        self.assertEqual(get_stderr + post_stderr, "")
        types = [event.event_type for event in self.events]
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertIn(EventType.TOOL_RESULT, types)

    def test_allowed_hosts_are_forwarded(self):
        for method in ("GET", "POST", "DELETE"):
            for headers in self._allowed_host_headers():
                with self.subTest(method=method, headers=headers):
                    self._assert_forwarded(method, headers)

    def test_refused_hosts_get_421(self):
        for method in ("GET", "POST", "DELETE"):
            for headers in self._refused_host_headers():
                with self.subTest(method=method, headers=headers):
                    self._assert_refused(
                        method, headers, 421, HOST_REFUSED, HOST_REFUSED_LOG
                    )
        self._assert_nothing_traced_or_forwarded()

    def test_loopback_origins_are_forwarded(self):
        for method in ("GET", "POST", "DELETE"):
            for origin in LOOPBACK_ORIGINS:
                with self.subTest(method=method, origin=origin):
                    self._assert_forwarded(method, [self.local_host, ("Origin", origin)])

    def test_refused_origins_get_403(self):
        cases = [[self.local_host, ("Origin", origin)] for origin in REFUSED_ORIGINS]
        loopback, attacker = ("Origin", "http://localhost:6274"), ("Origin", "http://attacker.test")
        cases.append([self.local_host, loopback, attacker])
        cases.append([self.local_host, attacker, loopback])
        for method in ("GET", "POST", "DELETE"):
            for headers in cases:
                with self.subTest(method=method, headers=headers):
                    self._assert_refused(
                        method, headers, 403, ORIGIN_REFUSED, ORIGIN_REFUSED_LOG
                    )
        self._assert_nothing_traced_or_forwarded()

    def test_host_is_checked_before_origin(self):
        headers = [("Host", "attacker.test"), ("Origin", "http://attacker.test")]
        for method in ("GET", "POST", "DELETE"):
            with self.subTest(method=method):
                self._assert_refused(method, headers, 421, HOST_REFUSED, HOST_REFUSED_LOG)
        self._assert_nothing_traced_or_forwarded()

    def test_refused_post_records_no_tool_call(self):
        # Any page can send a text/plain POST without a CORS preflight.
        body = json.dumps(_tool_call(7, name="delete_repo")).encode()
        plain = ("Content-Type", "text/plain")
        cases = [
            ([("Host", f"attacker.test:{self.port}"), plain], 421, HOST_REFUSED, HOST_REFUSED_LOG),
            ([self.local_host, ("Origin", "http://attacker.test"), plain],
             403, ORIGIN_REFUSED, ORIGIN_REFUSED_LOG),
        ]
        for headers, status, reply, log in cases:
            with self.subTest(status=status):
                self._assert_refused("POST", headers, status, reply, log, body)
        self.assertEqual(self.meta.tool_calls, 0)
        self._assert_nothing_traced_or_forwarded()

    def test_refusals_never_echo_header_values(self):
        marker = "marker-7f3a"
        cases = [
            (421, f"Host: {marker}.attacker.test:{self.port}\r\n"),
            (403, f"Host: 127.0.0.1:{self.port}\r\nOrigin: http://{marker}.attacker.test\r\n"),
        ]
        for method in ("GET", "POST"):
            for status, headers in cases:
                with self.subTest(method=method, status=status):
                    body = PING if method == "POST" else b""
                    request = (
                        f"{method} /message HTTP/1.1\r\n{headers}"
                        f"Content-Length: {len(body)}\r\n\r\n"
                    ).encode() + body
                    stderr = io.StringIO()
                    with mock.patch("sys.stderr", stderr):
                        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
                            sock.sendall(request)
                            reply = _read_all(sock)
                    self.assertTrue(reply.startswith(f"HTTP/1.0 {status} ".encode()), reply)
                    self.assertNotIn(marker.encode(), reply)
                    self.assertNotIn(marker, stderr.getvalue())
                    self.assertEqual(stderr.getvalue().count("\n"), 1)


class TestRecordHTTPAllowHostFlag(unittest.TestCase):
    """record-http --allow-host accepts more names as a Host or an Origin host."""

    def test_flag_is_parsed(self):
        args = build_parser().parse_args([
            "record-http", "--url", "http://127.0.0.1:1",
            "--allow-host", "tunnel.test", "--allow-host", "Second.Test",
        ])
        self.assertEqual(args.allow_host, ["tunnel.test", "Second.Test"])
        args = build_parser().parse_args(["record-http", "--url", "http://127.0.0.1:1"])
        self.assertIsNone(args.allow_host)

    def test_a_value_that_is_not_a_bare_name_exits_2(self):
        for value in ("https://tunnel.test/", "tunnel.test:80", ""):
            with self.subTest(value=value):
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr), self.assertRaises(SystemExit) as raised:
                    build_parser().parse_args([
                        "record-http", "--url", "http://127.0.0.1:1", "--allow-host", value,
                    ])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("invalid host name", stderr.getvalue())

    def test_live_proxy_accepts_the_names_and_nothing_else(self):
        upstream = _start_counting_upstream(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        args = build_parser().parse_args([
            "--trace-dir", tmp.name, "record-http",
            "--url", f"http://127.0.0.1:{upstream.server_address[1]}",
            "--port", "0", "--quiet",
            "--allow-host", "tunnel.test", "--allow-host", "Second.Test",
        ])
        servers = queue.Queue()
        real_make_server = HTTPProxyServer.make_server

        def make_server(proxy):
            server = real_make_server(proxy)
            servers.put(server)
            return server

        stderr = io.StringIO()
        exit_codes = []
        with mock.patch.object(HTTPProxyServer, "make_server", make_server), \
                mock.patch("sys.stderr", stderr):
            thread = threading.Thread(target=lambda: exit_codes.append(cmd_record_http(args)))
            thread.start()
            server = servers.get(timeout=10)
            try:
                port = server.server_address[1]
                local = ("Host", f"127.0.0.1:{port}")
                forwarded = [
                    ("GET", [("Host", f"tunnel.test:{port}")], None),
                    ("GET", [("Host", "TUNNEL.test")], None),
                    ("GET", [("Host", "second.test")], None),
                    ("POST", [local, ("Origin", "https://tunnel.test")], PING),
                ]
                refused = [
                    ("GET", [("Host", "eviltunnel.test")], 421),
                    ("GET", [("Host", "tunnel.test.attacker.test")], 421),
                    ("POST", [local, ("Origin", "https://eviltunnel.test")], 403),
                ]
                for method, headers, body in forwarded:
                    with self.subTest(headers=headers):
                        self.assertEqual(_send(port, method, headers, body)[0], 200)
                for method, headers, status in refused:
                    with self.subTest(headers=headers):
                        body = PING if method == "POST" else None
                        self.assertEqual(_send(port, method, headers, body)[0], status)
            finally:
                server.shutdown()
                thread.join(timeout=10)

        self.assertEqual(exit_codes, [0])
        # --quiet keeps the refusal lines.
        refusals = [line + "\n" for line in stderr.getvalue().splitlines() if "refused" in line]
        self.assertEqual(refusals, [HOST_REFUSED_LOG, HOST_REFUSED_LOG, ORIGIN_REFUSED_LOG])


class TestRecordHTTPTracestateParent(unittest.TestCase):
    """record-http links a tracestate parent only if its store accepts it.

    Each test records one tools/call, stops the proxy as Ctrl-C does and
    reads the session's meta.json.
    """

    def setUp(self):
        self.upstream = _start_counting_upstream(self)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = tmp.name
        self.store = TraceStore(self.trace_dir)

    def _parent(self, store=None):
        meta = SessionMeta(command="parent agent")
        (store or self.store).create_session(meta)
        return meta.session_id

    def _record(self, *argv, tracestate=None, before_stop=None):
        """Return the exit codes, the POST status, the saved meta and stderr."""
        args = build_parser().parse_args([
            "--trace-dir", self.trace_dir, "record-http",
            "--url", f"http://127.0.0.1:{self.upstream.server_address[1]}",
            "--port", "0", "--quiet", *argv,
        ])
        servers = queue.Queue()
        real_make_server = HTTPProxyServer.make_server

        def make_server(proxy):
            server = real_make_server(proxy)
            servers.put(server)
            return server

        stderr = io.StringIO()
        exit_codes = []
        with mock.patch.object(HTTPProxyServer, "make_server", make_server), \
                mock.patch("sys.stderr", stderr):
            thread = threading.Thread(target=lambda: exit_codes.append(cmd_record_http(args)))
            thread.start()
            server = servers.get(timeout=10)
            try:
                port = server.server_address[1]
                headers = [
                    ("Host", f"127.0.0.1:{port}"),
                    ("traceparent", f"00-{'a' * 32}-{'b' * 16}-01"),
                ]
                if tracestate is not None:
                    headers.append(("tracestate", tracestate))
                status = _send(port, "POST", headers, json.dumps(_tool_call(1)).encode())[0]
                if before_stop:
                    before_stop()
            finally:
                server.shutdown()
                thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        meta = self.store.load_meta(server.meta.session_id)
        return exit_codes, status, meta, stderr.getvalue()

    def _assert_saved_without_parent(self, exit_codes, status, meta):
        self.assertEqual(status, 200)
        self.assertEqual(exit_codes, [0])
        self.assertIsNotNone(meta.ended_at)
        self.assertGreater(meta.total_duration_ms, 0)
        self.assertEqual(meta.tool_calls, 1)
        self.assertEqual(meta.parent_session_id, "")

    def test_parent_from_another_store_is_ignored(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        parent_id = self._parent(TraceStore(other.name))

        exit_codes, status, meta, stderr = self._record(tracestate=f"agent-trace={parent_id}")

        self._assert_saved_without_parent(exit_codes, status, meta)
        self.assertNotIn("dropped", stderr)

    def test_malformed_parent_id_is_ignored(self):
        for value in ("a/b", "..", "x\x1b[2Jy"):
            with self.subTest(value=value):
                exit_codes, status, meta, stderr = self._record(
                    tracestate=f"agent-trace={value}"
                )

                self._assert_saved_without_parent(exit_codes, status, meta)
                self.assertNotIn("dropped", stderr)

    def test_parent_deleted_while_recording_is_dropped_at_the_end(self):
        for link in ("--parent", "tracestate"):
            with self.subTest(link=link), \
                    mock.patch.dict(os.environ, {"AGENT_STRACE_PARENT_EVENT": "evt-1"}):
                parent_id = self._parent()
                if link == "--parent":
                    kwargs = {}
                    argv = ("--parent", parent_id)
                else:
                    kwargs = {"tracestate": f"agent-trace={parent_id}"}
                    argv = ()

                exit_codes, status, meta, stderr = self._record(
                    *argv,
                    before_stop=lambda: shutil.rmtree(self.store.base_dir / parent_id),
                    **kwargs,
                )

                self._assert_saved_without_parent(exit_codes, status, meta)
                self.assertEqual(meta.parent_event_id, "")
                dropped = [line for line in stderr.splitlines() if "dropped" in line]
                self.assertEqual(dropped, [
                    f"agent-strace: dropped the link to parent session {parent_id}: "
                    "parent session must exist in the same scoped store"
                ])

    def test_save_failure_not_about_the_parent_still_fails(self):
        parent_id = self._parent()
        real_update_meta = TraceStore.update_meta
        failing = threading.Event()
        refused = []

        def update_meta(store, meta):
            if failing.is_set():
                refused.append(meta.parent_session_id)
                raise ValueError("refusing symlinked session directory")
            return real_update_meta(store, meta)

        with mock.patch.object(TraceStore, "update_meta", update_meta):
            exit_codes, status, _, stderr = self._record(
                tracestate=f"agent-trace={parent_id}", before_stop=failing.set
            )

        self.assertEqual(status, 200)
        # The link was in place when the final save ran.
        self.assertEqual(refused[:1], [parent_id])
        self.assertNotEqual(exit_codes, [0])
        self.assertIn("refusing symlinked session directory", stderr)
        self.assertNotIn("dropped", stderr)

    def test_parent_in_this_store_is_linked(self):
        parent_id = self._parent()

        exit_codes, status, meta, _ = self._record(tracestate=f"agent-trace={parent_id}")

        self.assertEqual(status, 200)
        self.assertEqual(exit_codes, [0])
        self.assertIsNotNone(meta.ended_at)
        self.assertEqual(meta.tool_calls, 1)
        self.assertEqual(meta.parent_session_id, parent_id)


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Records each request, then lets the server's reply function answer."""

    protocol_version = "HTTP/1.1"

    def _record_and_reply(self):
        length = int(self.headers.get("Content-Length", 0))
        request = {
            "method": self.command,
            "target": self.path,
            "headers": self.headers,
            "body": self.rfile.read(length) if length > 0 else b"",
        }
        self.server.record(request)
        self.server.reply(self, request)

    do_GET = do_POST = do_DELETE = _record_and_reply

    def log_message(self, format, *args):
        pass


def _reply(handler, status=200, body=b"", headers=()):
    """Answer with a whole body, then close the connection."""
    handler.send_response(status)
    for name, value in headers:
        handler.send_header(name, value)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def _reply_json(handler, request):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode()
    _reply(handler, body=body, headers=[("Content-Type", "application/json")])


class _ScriptedUpstream(ThreadingHTTPServer):
    """A fake MCP upstream that records requests and answers as a test says.

    Each request is recorded with its method, target, headers and body.
    Then ``reply(handler, request)`` answers it. The default is a JSON result.
    """

    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _ScriptedHandler)
        self.reply = _reply_json
        self._lock = threading.Lock()
        self._requests = []

    @property
    def requests(self):
        with self._lock:
            return list(self._requests)

    def record(self, request):
        with self._lock:
            self._requests.append(request)


def _event(message):
    return f"event: message\ndata: {json.dumps(message)}\n\n".encode()


class _EventStream:
    """An event stream reply, sent in chunks the way HTTP/1.1 servers do."""

    def __init__(self, handler, content_type="text/event-stream", headers=()):
        self._wfile = handler.wfile
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        for name, value in headers:
            handler.send_header(name, value)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.send_header("Connection", "close")
        handler.end_headers()

    def send(self, message):
        data = _event(message)
        self._wfile.write(b"%x\r\n%s\r\n" % (len(data), data))

    def end(self):
        self._wfile.write(b"0\r\n\r\n")


def _next_message(resp):
    """Read up to the next data line of an event stream. None at its end."""
    while True:
        line = resp.readline()
        if not line:
            return None
        if line.startswith(b"data: "):
            return json.loads(line[6:])


def _progress(value):
    return {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progressToken": 1, "progress": value},
    }


def _text_result(req_id, text):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"content": [{"type": "text", "text": text}]},
    }


# POST bodies and replies that are not one well-formed message: a field with
# the wrong JSON type, a batch, or a value that is not an object. Each one
# failed the request or cut the stream before the proxy read fields by type.
ODD_POSTS = [
    [_tool_call(1)],
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": None},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["x"]},
    {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": ["x"]},
    {"jsonrpc": "2.0", "id": 1, "method": "sampling/createMessage", "params": {"messages": "hi"}},
    5,
]
ODD_REPLIES = [
    {"jsonrpc": "2.0", "id": 1, "result": "x"},
    {"jsonrpc": "2.0", "id": 1, "result": {"content": "text"}},
    {"jsonrpc": "2.0", "id": 1, "error": "boom"},
    {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": 5}]}},
    {"jsonrpc": "2.0", "id": 1, "result": {"content": ["a"]}},
    [{"jsonrpc": "2.0", "id": 1, "result": "x"}],
    "x",
]

# Error values that are present but falsy. Each still means the call failed.
FALSY_ERRORS = [0, False, "", [], {}]

# json.loads cannot read a list nested this deep on Python 3.10 and 3.11,
# and no Python can store an event that holds one.
DEEP = 2000

UNTRACED_LOG = "agent-strace: could not trace a message ({}). It was still forwarded.\n"


def _deep_text(depth):
    """JSON text of a list nested depth deep, as json.dumps writes it."""
    return "[" * depth + "]" * depth


def _too_long_int():
    """JSON text of an integer longer than int() converts. None if unlimited."""
    limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    return "1" * (limit + 1) if limit else None


# What a Streamable HTTP client sends with every POST.
STREAMABLE_POST_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


class TestEventStreamParser(unittest.TestCase):
    def _events(self, chunks):
        parser = _EventStreamParser()
        return [data for chunk in chunks for data in parser.feed(chunk)]

    def test_bom_and_a_crlf_split_across_chunks(self):
        # The LF after a chunk that ends in CR is part of that CRLF, not a
        # blank line, so the event ends only at the second CRLF.
        chunks = [b"\xef\xbb\xbfdata: a\r", b"\ndata:b\r", b"\n\r", b"\n"]
        self.assertEqual(self._events(chunks), ["a\nb"])

    def test_empty_data_field_is_an_event_and_no_data_field_is_not(self):
        self.assertEqual(self._events([b"data\n\nevent: x\nid: 1\n\n"]), [""])

    def test_a_line_split_across_chunks(self):
        self.assertEqual(self._events([b"da", b"ta: x", b"\n\n"]), ["x"])

    def test_a_character_split_across_chunks(self):
        self.assertEqual(self._events([b"data: \xc3", b"\xa9\n\n"]), ["\u00e9"])

    def test_invalid_utf8_shows_as_a_replacement_character(self):
        self.assertEqual(self._events([b"data: \xff\n\n"]), ["\ufffd"])


class TestHTTPProxyStreamableHTTP(unittest.TestCase):
    """record-http works with Streamable HTTP servers.

    --url names the server's endpoint and the client connects to the proxy
    root. POST, GET and DELETE carry the headers the transport needs, and an
    event stream reply to a POST reaches the client as it arrives.

    Every client read times out after 5 seconds, so a reply the proxy holds
    back fails the test instead of hanging it.
    """

    def setUp(self):
        self.upstream = _ScriptedUpstream()
        threading.Thread(
            target=self.upstream.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        self.addCleanup(self.upstream.server_close)
        self.addCleanup(self.upstream.shutdown)
        self.upstream_url = f"http://127.0.0.1:{self.upstream.server_address[1]}"
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name)
        self.meta = SessionMeta(command="test", agent_name="test-agent")
        self.store.create_session(self.meta)
        self.events = []

    def _proxy(self, url_rest="/mcp", upstream_url=None):
        """Start a proxy whose --url is the upstream's URL plus url_rest.

        upstream_url, if given, replaces the scripted upstream's URL.
        """
        proxy = HTTPProxyServer(
            remote_url=(upstream_url or self.upstream_url) + url_rest,
            local_port=0,
            store=self.store,
            session_meta=self.meta,
            on_event=self.events.append,
        )
        server = proxy.make_server()
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def _hold(self):
        """Return an Event that a held upstream reply waits on. Cleanup sets it."""
        event = threading.Event()
        self.addCleanup(event.set)
        return event

    def _open(self, port, method, target="/", headers=None, body=None):
        """Send one request through the proxy and return the reply unread."""
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)
        conn.request(method, target, body=body, headers=headers or {})
        return conn.getresponse()

    def _forward(self, port, method, target="/", headers=None, body=None):
        """Send one request through the proxy and read the whole reply.

        Returns the reply, its body and the one request the upstream got.
        """
        before = len(self.upstream.requests)
        resp = self._open(port, method, target, headers, body)
        data = resp.read()
        requests = self.upstream.requests[before:]
        self.assertEqual(
            len(requests), 1, f"{method} {target} reached the upstream {len(requests)} times"
        )
        return resp, data, requests[0]

    def _of_type(self, event_type):
        return [event for event in self.events if event.event_type == event_type]

    def _assert_sent_once(self, request, headers):
        for name, value in headers.items():
            with self.subTest(header=name):
                self.assertEqual(request["headers"].get_all(name), [value])

    def _child_session(self):
        child = SessionMeta(command="child", agent_name="child-agent")
        self.store.create_session(child)
        self.assertEqual(self.store.load_meta(child.session_id).parent_session_id, "")
        return child.session_id

    def _child_trace_headers(self, child_id):
        """Headers a downstream agent that runs agent-trace adds to its reply."""
        return [
            ("traceparent", f"00-{'a' * 32}-{'b' * 16}-01"),
            ("tracestate", f"agent-trace={child_id}"),
        ]

    def _assert_linked(self, child_id):
        child = self.store.load_meta(child_id)
        self.assertEqual(child.parent_session_id, self.meta.session_id)
        self.assertEqual(child.depth, 1)

    def test_request_target_follows_the_url(self):
        cases = {
            "/mcp": [
                ("/", "/mcp"),
                ("/?a=1", "/mcp?a=1"),
                ("/mcp", "/mcp"),
                ("/messages/?session_id=S", "/messages/?session_id=S"),
            ],
            "/mcp/": [("/", "/mcp/")],
            "": [("/", "/"), ("/mcp", "/mcp")],
            "/mcp?key=K": [
                ("/", "/mcp?key=K"),
                ("/?a=1", "/mcp?key=K&a=1"),
                ("/messages/?session_id=S", "/messages/?key=K&session_id=S"),
            ],
            "?key=K": [("/mcp", "/mcp?key=K")],
        }
        for url_rest, targets in cases.items():
            port = self._proxy(url_rest).server_address[1]
            for target, expected in targets:
                for method in ("GET", "POST", "DELETE"):
                    with self.subTest(url=f"http://U{url_rest}", target=target, method=method):
                        body = PING if method == "POST" else None
                        _, _, request = self._forward(port, method, target, body=body)
                        self.assertEqual(request["target"], expected)

    def test_post_forwards_the_streamable_http_headers(self):
        port = self._proxy().server_address[1]
        headers = {
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer token-1",
            "Content-Type": "application/json",
            "Mcp-Session-Id": "session-1",
            "Mcp-Protocol-Version": "2025-06-18",
        }
        _, _, request = self._forward(port, "POST", "/", headers, PING)
        self._assert_sent_once(request, headers)

    def test_get_forwards_the_streamable_http_headers(self):
        port = self._proxy().server_address[1]
        headers = {
            "Accept": "text/event-stream",
            "Authorization": "Bearer token-1",
            "Mcp-Session-Id": "session-1",
            "Mcp-Protocol-Version": "2025-06-18",
            "Last-Event-ID": "event-7",
        }
        _, _, request = self._forward(port, "GET", "/", headers)
        self._assert_sent_once(request, headers)

    def test_delete_is_forwarded_and_its_reply_relayed(self):
        def reply(handler, request):
            _reply(handler, 202, b'{"ended": true}', [
                ("Content-Type", "application/json"),
                ("Mcp-Session-Id", "session-1"),
            ])

        self.upstream.reply = reply
        port = self._proxy().server_address[1]
        headers = {
            "Authorization": "Bearer token-1",
            "Mcp-Session-Id": "session-1",
            "Mcp-Protocol-Version": "2025-06-18",
        }
        resp, data, request = self._forward(port, "DELETE", "/", headers)
        self.assertEqual(request["method"], "DELETE")
        self._assert_sent_once(request, headers)
        self.assertEqual(resp.status, 202)
        self.assertEqual(resp.getheader("Mcp-Session-Id"), "session-1")
        self.assertEqual(data, b'{"ended": true}')
        self.assertEqual(self.events, [])

    def test_headers_the_client_did_not_send_are_not_sent(self):
        port = self._proxy().server_address[1]
        for method in ("GET", "POST", "DELETE"):
            with self.subTest(method=method):
                body = PING if method == "POST" else None
                _, _, request = self._forward(port, method, "/", {}, body)
                for name in (
                    "Accept", "Authorization", "Mcp-Session-Id",
                    "Mcp-Protocol-Version", "Last-Event-ID",
                ):
                    self.assertIsNone(request["headers"].get(name), name)

    def test_origin_cookie_and_accept_encoding_are_not_forwarded(self):
        port = self._proxy().server_address[1]
        headers = {
            "Origin": "http://localhost:6274",
            "Cookie": "sid=secret",
            "Accept-Encoding": "gzip",
        }
        for method in ("GET", "POST", "DELETE"):
            with self.subTest(method=method):
                body = PING if method == "POST" else None
                _, _, request = self._forward(port, method, "/", headers, body)
                self.assertIsNone(request["headers"].get("Origin"))
                self.assertIsNone(request["headers"].get("Cookie"))
                # The proxy's own HTTP client asks for an unencoded reply.
                self.assertEqual(request["headers"].get_all("Accept-Encoding"), ["identity"])

    def test_streamed_reply_reaches_the_client_as_it_arrives(self):
        for content_type in ("text/event-stream", "Text/Event-Stream; charset=utf-8"):
            with self.subTest(content_type=content_type):
                port = self._proxy().server_address[1]
                first_read = self._hold()

                def reply(handler, request, content_type=content_type, first_read=first_read):
                    stream = _EventStream(handler, content_type)
                    stream.send(_progress(1))
                    first_read.wait(10)
                    stream.send(_progress(2))
                    stream.end()

                self.upstream.reply = reply
                resp = self._open(port, "POST", "/", STREAMABLE_POST_HEADERS, PING)
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.getheader("Content-Type"), content_type)
                self.assertEqual(_next_message(resp), _progress(1))
                first_read.set()
                self.assertEqual(_next_message(resp), _progress(2))
                self.assertIsNone(_next_message(resp))

    def test_client_answers_the_server_during_a_streamed_reply(self):
        # A tool that asks the user something, as MCP elicitation does.
        question = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "elicitation/create",
            "params": {"message": "Your name?", "requestedSchema": {"type": "object"}},
        }
        answer = {
            "jsonrpc": "2.0",
            "id": 0,
            "result": {"action": "accept", "content": {"name": "bob"}},
        }
        answered = self._hold()

        def reply(handler, request):
            msg = json.loads(request["body"])
            if msg.get("method") != "tools/call":
                answered.set()
                _reply(handler, 202)
                return
            stream = _EventStream(handler)
            stream.send(question)
            answered.wait(10)
            stream.send(_text_result(msg["id"], "hello bob"))
            stream.end()

        self.upstream.reply = reply
        port = self._proxy().server_address[1]
        call = json.dumps(_tool_call(12, name="ask"))
        resp = self._open(port, "POST", "/", STREAMABLE_POST_HEADERS, call)
        self.assertEqual(_next_message(resp), question)

        answer_resp, _, _ = self._forward(
            port, "POST", "/", STREAMABLE_POST_HEADERS, json.dumps(answer)
        )
        self.assertEqual(answer_resp.status, 202)
        self.assertEqual(_next_message(resp), _text_result(12, "hello bob"))
        self.assertIsNone(_next_message(resp))

    def test_streamed_tool_result_is_traced_and_linked_to_its_call(self):
        def reply(handler, request):
            stream = _EventStream(handler)
            stream.send(_progress(1))
            stream.send(_text_result(json.loads(request["body"])["id"], "streamed output"))
            stream.end()

        self.upstream.reply = reply
        server = self._proxy()
        resp, data, _ = self._forward(
            server.server_address[1], "POST", "/", STREAMABLE_POST_HEADERS,
            json.dumps(_tool_call(9)),
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(data, _event(_progress(1)) + _event(_text_result(9, "streamed output")))
        (call,) = self._of_type(EventType.TOOL_CALL)
        (result,) = self._of_type(EventType.TOOL_RESULT)
        self.assertEqual(result.parent_id, call.event_id)
        self.assertEqual(result.data["content_preview"], "streamed output")
        self.assertEqual(server.pending_calls, {})
        self.assertEqual(
            [event.event_type for event in self.store.load_events(self.meta.session_id)],
            [EventType.TOOL_CALL, EventType.TOOL_RESULT],
        )

    def test_close_streams_ends_an_open_streamed_reply(self):
        server = self._proxy()
        held = self._hold()

        def reply(handler, request):
            stream = _EventStream(handler)
            stream.send(_progress(1))
            held.wait(10)
            stream.end()

        self.upstream.reply = reply
        resp = self._open(
            server.server_address[1], "POST", "/", STREAMABLE_POST_HEADERS,
            json.dumps(_tool_call(11)),
        )
        self.assertEqual(_next_message(resp), _progress(1))

        start = time.monotonic()
        server.shutdown()
        self.assertTrue(server.close_streams(timeout=3))
        self.assertLess(time.monotonic() - start, 3)
        # The proxy closed the stream, so the read ends instead of timing out.
        self.assertEqual(resp.read().strip(), b"")
        self.assertEqual(self._of_type(EventType.ERROR), [])

    def test_streamed_reply_cut_by_the_upstream_ends_without_an_error_page(self):
        def reply(handler, request):
            stream = _EventStream(handler)
            stream.send(_progress(1))
            # The upstream goes away inside the next chunk.
            handler.wfile.write(b'100\r\ndata: {"jsonrpc"')

        self.upstream.reply = reply
        server = self._proxy()
        resp, data, _ = self._forward(
            server.server_address[1], "POST", "/", STREAMABLE_POST_HEADERS,
            json.dumps(_tool_call(13)),
        )
        self.assertEqual(resp.status, 200)
        # The status line is already sent, so the cut ends the reply. A 502
        # page after the events would corrupt the stream.
        self.assertEqual(data, _event(_progress(1)))
        (error,) = self._of_type(EventType.ERROR)
        self.assertTrue(error.data["message"].startswith("Proxy error: "))
        self.assertEqual(
            [event.event_type for event in self.store.load_events(self.meta.session_id)],
            [EventType.TOOL_CALL, EventType.ERROR],
        )

    def test_requests_after_close_streams_never_reach_the_upstream(self):
        # A listener that never accepts. A connection to it waits in its backlog.
        listener = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(listener.close)
        server = self._proxy(upstream_url=f"http://127.0.0.1:{listener.getsockname()[1]}")
        # serve_forever still runs, so each request below reaches its handler
        # after close_streams has started, like one accepted just before Ctrl-C.
        self.assertTrue(server.close_streams(timeout=0))
        for method, body in (("GET", None), ("POST", PING), ("DELETE", None)):
            with self.subTest(method=method):
                # The proxy closes the connection without a reply.
                with self.assertRaises(ConnectionResetError):
                    self._open(server.server_address[1], method, "/", body=body)
                # The proxy ends the reply after its handler returns, so any
                # connection the handler opened is in the backlog by now.
                ready, _, _ = select.select([listener], [], [], 0)
                self.assertEqual(ready, [], f"{method} connected to the upstream")

    def test_json_reply_is_relayed_whole_and_traced(self):
        body = json.dumps(_text_result(4, "json output"), indent=2).encode()

        def reply(handler, request):
            _reply(handler, 200, body, [("Content-Type", "application/json")])

        self.upstream.reply = reply
        server = self._proxy()
        resp, data, _ = self._forward(
            server.server_address[1], "POST", "/", STREAMABLE_POST_HEADERS,
            json.dumps(_tool_call(4)),
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        self.assertEqual(resp.getheader("Content-Length"), str(len(body)))
        self.assertEqual(data, body)
        (call,) = self._of_type(EventType.TOOL_CALL)
        (result,) = self._of_type(EventType.TOOL_RESULT)
        self.assertEqual(result.parent_id, call.event_id)
        self.assertEqual(server.pending_calls, {})

    def test_json_reply_links_the_sub_session(self):
        child_id = self._child_session()
        headers = [("Content-Type", "application/json")] + self._child_trace_headers(child_id)

        def reply(handler, request):
            _reply(handler, 200, json.dumps(_text_result(5, "ok")).encode(), headers)

        self.upstream.reply = reply
        port = self._proxy().server_address[1]
        self._forward(port, "POST", "/", STREAMABLE_POST_HEADERS, json.dumps(_tool_call(5)))
        self._assert_linked(child_id)

    def test_sub_session_depth_is_one_more_than_this_sessions(self):
        self.meta.depth = 1
        self.store.update_meta(self.meta)
        child_id = self._child_session()
        headers = [("Content-Type", "application/json")] + self._child_trace_headers(child_id)

        def reply(handler, request):
            _reply(handler, 200, json.dumps(_text_result(5, "ok")).encode(), headers)

        self.upstream.reply = reply
        port = self._proxy().server_address[1]
        self._forward(port, "POST", "/", STREAMABLE_POST_HEADERS, json.dumps(_tool_call(5)))
        child = self.store.load_meta(child_id)
        self.assertEqual(child.parent_session_id, self.meta.session_id)
        self.assertEqual(child.depth, 2)

    def test_streamed_reply_links_the_sub_session(self):
        child_id = self._child_session()
        headers = self._child_trace_headers(child_id)

        def reply(handler, request):
            stream = _EventStream(handler, headers=headers)
            stream.send(_text_result(6, "ok"))
            stream.end()

        self.upstream.reply = reply
        port = self._proxy().server_address[1]
        self._forward(port, "POST", "/", STREAMABLE_POST_HEADERS, json.dumps(_tool_call(6)))
        self._assert_linked(child_id)

    def _relay_raw_stream(self, req_id, stream):
        """POST a tool call whose reply is the given event stream bytes.

        Checks the client gets the bytes unchanged and returns the texts of
        the traced tool results.
        """
        def reply(handler, request):
            _reply(handler, 200, stream, [("Content-Type", "text/event-stream")])

        self.upstream.reply = reply
        self.events.clear()
        port = self._proxy().server_address[1]
        resp, data, _ = self._forward(
            port, "POST", "/", STREAMABLE_POST_HEADERS, json.dumps(_tool_call(req_id))
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(data, stream)
        self.assertEqual(self._of_type(EventType.ERROR), [])
        return [event.data["content_preview"] for event in self._of_type(EventType.TOOL_RESULT)]

    def test_event_data_without_a_space_after_the_colon_is_traced(self):
        message = json.dumps(_text_result(21, "no space"))
        texts = self._relay_raw_stream(21, f"data:{message}\n\n".encode())
        self.assertEqual(texts, ["no space"])

    def test_event_data_split_over_several_lines_is_traced_as_one_message(self):
        message = json.dumps(_text_result(22, "split"), indent=1)
        lines = "".join(f"data: {line}\n" for line in message.split("\n"))
        texts = self._relay_raw_stream(22, f"event: message\nid: 1\n{lines}\n".encode())
        self.assertEqual(texts, ["split"])

    def test_event_lines_may_end_in_crlf_or_a_lone_cr(self):
        message = json.dumps(_text_result(23, "crlf"), indent=1)
        lines = "".join(f"data: {line}\r\n" for line in message.split("\n"))
        texts = self._relay_raw_stream(23, f": comment\r\n{lines}\r\n".encode())
        self.assertEqual(texts, ["crlf"])

        message = json.dumps(_text_result(24, "lone cr"))
        texts = self._relay_raw_stream(24, f"data: {message}\r\r".encode())
        self.assertEqual(texts, ["lone cr"])

    def test_events_that_are_not_json_objects_or_never_end_are_not_traced(self):
        stream = (
            b"event: endpoint\ndata: /messages/?session_id=S\n\n"
            b"data: [1, 2]\n\n"
            b"data\n\n"
            b": " + json.dumps(_text_result(25, "comment")).encode() + b"\n\n"
            # The stream ends before the blank line that would end this event.
            b"data: " + json.dumps(_text_result(25, "unended")).encode() + b"\n"
        )
        self.assertEqual(self._relay_raw_stream(25, stream), [])

    def test_delete_with_a_refused_host_never_reaches_the_upstream(self):
        port = self._proxy().server_address[1]
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            status, _, data = _send(
                port, "DELETE", [("Host", "attacker.test"), ("Mcp-Session-Id", "session-1")]
            )
        self.assertEqual(status, 421)
        self.assertEqual(json.loads(data), HOST_REFUSED)
        self.assertEqual(stderr.getvalue(), HOST_REFUSED_LOG)
        # A forwarded request goes last. Once its reply is back, the upstream
        # has seen every earlier request that was forwarded.
        self._forward(port, "GET", "/")
        self.assertEqual([request["method"] for request in self.upstream.requests], ["GET"])
        self.assertEqual(self.events, [])

    def test_delete_to_an_upstream_that_refuses_the_connection_gets_502(self):
        server = self._proxy(upstream_url=f"http://127.0.0.1:{_find_free_port()}")
        # _open times out after 5 seconds, so a proxy that never answers fails fast.
        resp = self._open(
            server.server_address[1], "DELETE", "/", {"Mcp-Session-Id": "session-1"}
        )
        resp.read()
        self.assertEqual(resp.status, 502)
        (error,) = self._of_type(EventType.ERROR)
        self.assertTrue(error.data["message"].startswith("Proxy error: "))
        self.assertEqual(
            [event.event_type for event in self.store.load_events(self.meta.session_id)],
            [EventType.ERROR],
        )

    def _answer(self, body, content_type="application/json"):
        """Have the upstream answer each request with these bytes."""
        def reply(handler, request):
            _reply(handler, 200, body, [("Content-Type", content_type)])

        self.upstream.reply = reply

    def _links(self):
        """Map the id of each traced response to the id of the call it links to."""
        calls = {event.event_id: event.data["request_id"] for event in self._of_type(EventType.TOOL_CALL)}
        return {
            event.data.get("request_id"): calls.get(event.parent_id)
            for event in self.events
            if event.event_type in (EventType.TOOL_RESULT, EventType.ERROR)
        }

    def _call_ids(self):
        return [event.data["request_id"] for event in self._of_type(EventType.TOOL_CALL)]

    def test_post_bodies_with_wrong_typed_fields_are_forwarded(self):
        port = self._proxy().server_address[1]
        upstream_reply = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode()
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            for n, odd in enumerate(ODD_POSTS, start=100):
                with self.subTest(body=odd):
                    for body in (json.dumps(odd).encode(), json.dumps(_tool_call(n)).encode()):
                        resp, data, request = self._forward(
                            port, "POST", "/", STREAMABLE_POST_HEADERS, body
                        )
                        self.assertEqual(
                            (resp.status, request["body"], data), (200, body, upstream_reply)
                        )
                    self.assertIn(n, self._call_ids())
        self.assertEqual(stderr.getvalue(), "")

    def test_json_replies_with_wrong_typed_fields_reach_the_client(self):
        port = self._proxy().server_address[1]
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            for n, odd in enumerate(ODD_REPLIES, start=100):
                with self.subTest(reply=odd):
                    exchanges = [
                        (PING, json.dumps(odd).encode()),
                        (json.dumps(_tool_call(n)), json.dumps(_text_result(n, "after")).encode()),
                    ]
                    for body, reply in exchanges:
                        self._answer(reply)
                        resp, data, _ = self._forward(
                            port, "POST", "/", STREAMABLE_POST_HEADERS, body
                        )
                        self.assertEqual((resp.status, data), (200, reply))
                    self.assertEqual(self._links().get(n), n)
        self.assertEqual(stderr.getvalue(), "")

    def test_event_data_with_wrong_typed_fields_reaches_the_client(self):
        port = self._proxy().server_address[1]
        odd_events = [b"data: " + json.dumps(odd).encode() + b"\n\n" for odd in ODD_REPLIES]
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            for n, (odd, odd_event) in enumerate(zip(ODD_REPLIES, odd_events), start=100):
                with self.subTest(data=odd):
                    stream = odd_event + _event(_text_result(n, "after"))
                    self._answer(stream, "text/event-stream")
                    resp, data, _ = self._forward(
                        port, "POST", "/", STREAMABLE_POST_HEADERS, json.dumps(_tool_call(n))
                    )
                    self.assertEqual((resp.status, data), (200, stream))
                    self.assertEqual(self._links().get(n), n)

            # A GET stream is relayed the same way.
            stream = b"".join(odd_events) + _event(_text_result(200, "after"))
            self._answer(stream, "text/event-stream")
            resp, data, _ = self._forward(port, "GET", "/", {"Accept": "text/event-stream"})
            self.assertEqual((resp.status, data), (200, stream))
            self.assertIn(200, self._links())
        self.assertEqual(stderr.getvalue(), "")

    def test_a_batch_and_its_json_reply_are_traced_one_message_at_a_time(self):
        body = json.dumps([
            _tool_call(31, "a"),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            _tool_call(32, "b"),
        ]).encode()
        reply = json.dumps([
            _text_result(32, "b done"),
            {"jsonrpc": "2.0", "id": 31, "error": {"code": -1, "message": "no"}},
        ]).encode()
        self._answer(reply)
        server = self._proxy()
        resp, data, request = self._forward(
            server.server_address[1], "POST", "/", STREAMABLE_POST_HEADERS, body
        )
        self.assertEqual((resp.status, request["body"], data), (200, body, reply))
        self.assertEqual([call.data["tool_name"] for call in self._of_type(EventType.TOOL_CALL)], ["a", "b"])
        self.assertEqual(self._links(), {32: 32, 31: 31})
        self.assertEqual(server.pending_calls, {})

    def test_event_data_that_is_a_batch_is_traced_one_message_at_a_time(self):
        body = json.dumps([_tool_call(33, "a"), _tool_call(34, "b")]).encode()
        stream = _event(_progress(1)) + _event([_text_result(34, "b"), _text_result(33, "a")])
        self._answer(stream, "text/event-stream")
        server = self._proxy()
        resp, data, request = self._forward(
            server.server_address[1], "POST", "/", STREAMABLE_POST_HEADERS, body
        )
        self.assertEqual((resp.status, request["body"], data), (200, body, stream))
        self.assertEqual(self._links(), {34: 34, 33: 33})
        self.assertEqual(server.pending_calls, {})

    def test_a_post_body_the_tracer_cannot_read_is_forwarded(self):
        # Arguments this deep cannot be stored, and on Python 3.10 and 3.11
        # json.loads cannot read them.
        bodies = [(
            "RecursionError",
            '{"jsonrpc": "2.0", "id": 41, "method": "tools/call", "params": {"name": "deep", '
            '"arguments": {"rows": ' + _deep_text(DEEP) + "}}}",
        )]
        digits = _too_long_int()
        if digits:
            bodies.append((
                "ValueError",
                '{"jsonrpc": "2.0", "id": 42, "method": "tools/call", "params": {"name": "big", '
                '"arguments": {"n": ' + digits + "}}}",
            ))
        server = self._proxy()
        port = server.server_address[1]
        for n, (error, text) in enumerate(bodies, start=43):
            with self.subTest(error=error):
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    for body in (text.encode(), json.dumps(_tool_call(n)).encode()):
                        resp, _, request = self._forward(
                            port, "POST", "/", STREAMABLE_POST_HEADERS, body
                        )
                        self.assertEqual((resp.status, request["body"]), (200, body))
                self.assertEqual(stderr.getvalue(), UNTRACED_LOG.format(error))
        expected = list(range(43, 43 + len(bodies)))
        self.assertEqual(self._call_ids(), expected)
        self.assertEqual(list(server.pending_calls), expected)

    def _unreadable_results(self, deep_id, big_id):
        """Tool results the tracer cannot read, with the notes each may give.

        Python 3.12 and later read and trace the deep one.
        """
        results = [(
            ("", UNTRACED_LOG.format("RecursionError")),
            '{"jsonrpc": "2.0", "id": ' + str(deep_id) + ', "result": {"content": '
            '[{"type": "text", "text": "deep"}], "structuredContent": {"rows": '
            + _deep_text(DEEP) + "}}}",
        )]
        digits = _too_long_int()
        if digits:
            results.append((
                (UNTRACED_LOG.format("ValueError"),),
                '{"jsonrpc": "2.0", "id": ' + str(big_id) + ', "result": {"content": [], "n": '
                + digits + "}}",
            ))
        return results

    def test_a_json_reply_the_tracer_cannot_read_reaches_the_client(self):
        port = self._proxy().server_address[1]
        for n, (notes, text) in enumerate(self._unreadable_results(51, 52), start=53):
            with self.subTest(note=notes[-1]):
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    exchanges = [
                        (PING, text.encode()),
                        (json.dumps(_tool_call(n)), json.dumps(_text_result(n, "after")).encode()),
                    ]
                    for body, reply in exchanges:
                        self._answer(reply)
                        resp, data, _ = self._forward(
                            port, "POST", "/", STREAMABLE_POST_HEADERS, body
                        )
                        self.assertEqual((resp.status, data), (200, reply))
                self.assertIn(stderr.getvalue(), notes)
                self.assertEqual(self._links().get(n), n)

    def test_event_data_the_tracer_cannot_read_reaches_the_client(self):
        results = self._unreadable_results(61, 62)
        stream = b"".join(b"data: " + text.encode() + b"\n\n" for _, text in results)
        stream += _event(_text_result(63, "after"))
        self._answer(stream, "text/event-stream")
        port = self._proxy().server_address[1]
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            resp, data, _ = self._forward(
                port, "POST", "/", STREAMABLE_POST_HEADERS, json.dumps(_tool_call(63))
            )
        self.assertEqual((resp.status, data), (200, stream))
        self.assertEqual(self._links().get(63), 63)
        notes = {line + "\n" for line in stderr.getvalue().splitlines()}
        self.assertLessEqual(notes, {note for note_set, _ in results for note in note_set if note})
        if _too_long_int():
            self.assertIn(UNTRACED_LOG.format("ValueError"), notes)

    def test_a_message_the_store_cannot_take_is_still_forwarded(self):
        append_event = self.store.append_event
        failures = [OSError("No space left on device")]

        def fail_once(session_id, event):
            if failures:
                raise failures.pop()
            append_event(session_id, event)

        server = self._proxy()
        port = server.server_address[1]
        stderr = io.StringIO()
        with mock.patch.object(self.store, "append_event", side_effect=fail_once), \
                mock.patch("sys.stderr", stderr):
            for n in (71, 72):
                body = json.dumps(_tool_call(n)).encode()
                reply = json.dumps(_text_result(n, "done")).encode()
                self._answer(reply)
                resp, data, request = self._forward(
                    port, "POST", "/", STREAMABLE_POST_HEADERS, body
                )
                self.assertEqual((resp.status, request["body"], data), (200, body, reply))
        self.assertEqual(stderr.getvalue(), UNTRACED_LOG.format("OSError"))
        self.assertEqual(self._call_ids(), [72])
        # The result of a call the trace lacks is kept, with no link.
        self.assertEqual(self._links(), {71: None, 72: 72})
        self.assertEqual(server.pending_calls, {})

    def _assert_falsy_errors_close_their_calls(self, reply_body, content_type):
        """Answer one call per falsy error, with reply_body(message) as the reply."""
        server = self._proxy()
        port = server.server_address[1]
        ids = list(range(81, 81 + len(FALSY_ERRORS)))
        for n, error in zip(ids, FALSY_ERRORS):
            with self.subTest(error=error):
                body = json.dumps(_tool_call(n)).encode()
                reply = reply_body({"jsonrpc": "2.0", "id": n, "error": error})
                self._answer(reply, content_type)
                resp, data, request = self._forward(
                    port, "POST", "/", STREAMABLE_POST_HEADERS, body
                )
                self.assertEqual((resp.status, request["body"], data), (200, body, reply))
        self.assertEqual(
            [(e.data["request_id"], e.data["code"], e.data["message"]) for e in self._of_type(EventType.ERROR)],
            [(n, None, "") for n in ids],
        )
        self.assertEqual(self._links(), {n: n for n in ids})
        self.assertEqual(server.pending_calls, {})

    def test_a_json_reply_with_a_falsy_error_closes_its_call(self):
        self._assert_falsy_errors_close_their_calls(
            lambda message: json.dumps(message).encode(), "application/json"
        )

    def test_event_data_with_a_falsy_error_closes_its_call(self):
        self._assert_falsy_errors_close_their_calls(
            lambda message: _event(_progress(1)) + _event(message), "text/event-stream"
        )


if __name__ == "__main__":
    unittest.main()
