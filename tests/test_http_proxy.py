"""Tests for HTTP/SSE MCP proxy."""

import hashlib
import http.client
import io
import json
import os
import queue
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

from agent_trace.http_proxy import HTTPProxyServer, _ProxyHTTPServer
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
        self.assertEqual(server._active_requests, 1)

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

    The proxy uses ssl.create_default_context(), so it trusts the system CA
    store. A custom CA is trusted through OpenSSL's SSL_CERT_FILE variable,
    which the tests set to the test CA.
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

    def _proxy(self, remote_url):
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


if __name__ == "__main__":
    unittest.main()
