"""Tests for request/response pairing in the stdio MCP proxy."""

import io
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

from agent_trace import proxy as proxy_module
from agent_trace.models import EventType, SessionMeta
from agent_trace.proxy import MCPProxy
from agent_trace.store import TraceStore


def _call(request_id, name="read_file"):
    return {
        "jsonrpc": "2.0", "id": request_id, "method": "tools/call",
        "params": {"name": name, "arguments": {}},
    }


def _result(request_id):
    return {
        "jsonrpc": "2.0", "id": request_id,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }


def _error(request_id):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": "bad"}}


class StdioProxyPairingTest(unittest.TestCase):
    def setUp(self):
        self.store = TraceStore(tempfile.mkdtemp(), redact=False)
        self.meta = SessionMeta()
        self.store.create_session(self.meta)
        self.proxy = MCPProxy(["true"], self.store, self.meta)

    def _send(self, message, direction):
        source = io.BytesIO((json.dumps(message) + "\n").encode("utf-8"))
        self.proxy._forward_and_trace(source, io.BytesIO(), direction, threading.Event())

    def _events(self, event_type):
        return [
            e for e in self.store.load_events(self.meta.session_id)
            if e.event_type == event_type
        ]

    def test_result_links_to_the_call_with_its_id(self):
        self._send(_call(1, "a"), "agent_to_server")
        self._send(_call(2, "b"), "agent_to_server")
        self._send(_result(2), "server_to_agent")

        calls = {c.data["request_id"]: c for c in self._events(EventType.TOOL_CALL)}
        (result,) = self._events(EventType.TOOL_RESULT)
        self.assertEqual(result.parent_id, calls[2].event_id)
        self.assertEqual(result.duration_ms, (result.timestamp - calls[2].timestamp) * 1000)
        self.assertEqual(list(self.proxy._calls.pending()), [1])

    def test_error_links_to_its_call_and_closes_it(self):
        self._send(_call(5), "agent_to_server")
        self._send(_error(5), "server_to_agent")
        self._send(_result(5), "server_to_agent")

        (call,) = self._events(EventType.TOOL_CALL)
        (error,) = self._events(EventType.ERROR)
        (late,) = self._events(EventType.TOOL_RESULT)
        self.assertEqual(error.parent_id, call.event_id)
        self.assertIsNotNone(error.duration_ms)
        self.assertEqual(late.parent_id, "")
        self.assertEqual(self.proxy._calls.pending(), {})

    def test_result_with_unknown_id_stays_unlinked(self):
        self._send(_call(1), "agent_to_server")
        self._send(_result(2), "server_to_agent")

        (result,) = self._events(EventType.TOOL_RESULT)
        self.assertEqual(result.parent_id, "")
        self.assertEqual(list(self.proxy._calls.pending()), [1])

    def test_error_without_id_stays_unlinked(self):
        self._send(_call(1), "agent_to_server")
        self._send(_error(None), "server_to_agent")

        (error,) = self._events(EventType.ERROR)
        self.assertEqual(error.parent_id, "")
        self.assertEqual(list(self.proxy._calls.pending()), [1])

    def test_pending_calls_are_bounded(self):
        with patch.object(proxy_module, "MAX_PENDING_CALLS", 2):
            proxy = MCPProxy(["true"], self.store, self.meta)
        self.proxy = proxy
        for request_id in range(4):
            self._send(_call(request_id), "agent_to_server")
        self.assertEqual(list(proxy._calls.pending()), [2, 3])

    def test_each_proxy_has_its_own_pending_calls(self):
        other = MCPProxy(["true"], self.store, self.meta)
        self.assertIsNot(self.proxy._calls, other._calls)


if __name__ == "__main__":
    unittest.main()
