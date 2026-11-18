"""Tests for the stdio MCP proxy: pairing, and relaying what it cannot trace."""

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

try:
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry

import agent_trace
from agent_trace import proxy as proxy_module
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.proxy import MCPProxy, _classify_message
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


# Error values that are present but falsy. Each still means the call failed.
FALSY_ERRORS = [0, False, "", [], {}]


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

    def test_a_falsy_error_links_to_its_call_and_closes_it(self):
        for request_id, error in enumerate(FALSY_ERRORS, start=1):
            self._send(_call(request_id), "agent_to_server")
            self._send({"jsonrpc": "2.0", "id": request_id, "error": error}, "server_to_agent")

        calls = {c.data["request_id"]: c.event_id for c in self._events(EventType.TOOL_CALL)}
        errors = self._events(EventType.ERROR)
        self.assertEqual({e.data["request_id"]: e.parent_id for e in errors}, calls)
        self.assertEqual(len(errors), len(FALSY_ERRORS))
        self.assertEqual(self.proxy._calls.pending(), {})

    def test_a_result_links_to_its_call_when_redaction_masks_the_id(self):
        # _emit masks the id in the stored event, so pairing keys on the id as it
        # arrived. The key is split so the source does not hold one.
        request_id = "sk_live_" + "a1B2c3D4e5F6g7H8i9J0k1L2"
        self.proxy = MCPProxy(["true"], self.store, self.meta, redact=True)
        self._send(_call(request_id), "agent_to_server")
        self._send(_result(request_id), "server_to_agent")

        (call,) = self._events(EventType.TOOL_CALL)
        (result,) = self._events(EventType.TOOL_RESULT)
        self.assertEqual(call.data["request_id"], "[REDACTED:stripe-key]")
        self.assertEqual(result.parent_id, call.event_id)
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


def _line(message):
    """A message as one line of newline-delimited JSON."""
    return (json.dumps(message) + "\n").encode("utf-8")


def _framed(text):
    """A message body with a Content-Length header."""
    body = text.encode("utf-8")
    return b"Content-Length: %d\r\n\r\n%s" % (len(body), body)


def _nested_list(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


def _deep_text(depth):
    """JSON text of a list nested depth deep, as json.dumps writes it.

    Built as text because json.dumps fails at this depth on some Pythons.
    """
    return "[" * depth + "]" * depth


def _too_long_int():
    """JSON text of an integer longer than int() converts. None if unlimited."""
    limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    return "1" * (limit + 1) if limit else None


def _deep_call(request_id):
    """A tools/call whose arguments are nested too deep to store."""
    return (
        '{"jsonrpc": "2.0", "id": ' + str(request_id) + ', "method": "tools/call", '
        '"params": {"name": "deep", "arguments": {"rows": ' + _deep_text(DEEP) + "}}}"
    )


# json.loads cannot read a list nested this deep on Python 3.10 and 3.11,
# and no Python can store an event that holds one.
DEEP = 2000

UNTRACED_LOG = "agent-strace: could not trace a message ({}). It was still forwarded.\n"

# Messages with a field of the wrong JSON type. Each one stopped the relay
# before the proxy read fields by type.
ODD_REQUESTS = [
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": None},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["x"]},
    {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": ["x"]},
    {"jsonrpc": "2.0", "id": 1, "method": "sampling/createMessage", "params": {"messages": "hi"}},
    {
        "jsonrpc": "2.0", "id": 1, "method": "completion/create",
        "params": {"messages": [{"role": "user", "content": {"text": 5}}]},
    },
]
ODD_RESPONSES = [
    {"jsonrpc": "2.0", "id": 1, "result": "x"},
    {"jsonrpc": "2.0", "id": 1, "error": "boom"},
    {"jsonrpc": "2.0", "id": 1, "result": {"content": "text"}},
    {"jsonrpc": "2.0", "id": 1, "result": {"content": ["a"]}},
    {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": 5}]}},
    {"jsonrpc": "2.0", "method": "notifications/resources/updated", "params": ["x"]},
]


class ClassifyMessageTest(unittest.TestCase):
    def test_never_raises_for_any_json_value_in_either_direction(self):
        values = [
            None, True, 5, 1.5, "x", [], ["x"], [_call(1)], [_result(1)], {},
            {"method": 5}, {"method": None, "id": {"a": 1}},
            *ODD_REQUESTS,
            *ODD_RESPONSES,
            {"id": 1, "method": "tools/call", "params": {"name": 5, "arguments": ["x"]}},
            {"id": 1, "method": "resources/read", "params": {"uri": 5}},
            {"id": 1, "method": "sampling/createMessage", "params": {"messages": {"a": 1}}},
            {"id": 1, "method": "sampling/createMessage", "params": {"messages": ["hi"]}},
            {"id": 1, "method": "completion/create", "params": {"model": 5, "messages": [{"role": 5}]}},
            {"id": 1, "result": ["x"]},
            {"id": 1, "result": {"content": {"type": "text"}}},
            {"id": 1, "result": {"content": [None, {"type": 5}]}},
            {"id": 1, "result": {"content": [{"type": "text", "text": ["x"] * 300}]}},
            {"id": 1, "error": 5},
            {"id": 1, "error": ["x"]},
            {"id": 1, "error": {"code": "x", "message": 5}},
            {"method": "notifications/resources/updated", "params": None},
            {"method": "notifications/resources/updated", "params": {"uri": ["x"]}},
        ]
        for value in values:
            for direction in ("agent_to_server", "server_to_agent"):
                with self.subTest(value=value, direction=direction):
                    event = _classify_message(value, direction)
                    self.assertIsInstance(event, (TraceEvent, type(None)))

    def test_a_last_message_too_deep_to_print_has_no_preview(self):
        # Deeper than str() goes on any Python, so it raises RecursionError.
        message = {
            "id": 1, "method": "completion/create",
            "params": {"messages": [{"role": "user", "content": _nested_list(100_000)}]},
        }
        for direction in ("agent_to_server", "server_to_agent"):
            with self.subTest(direction=direction):
                event = _classify_message(message, direction)
                if direction == "agent_to_server":
                    self.assertEqual(
                        event.data["last_message"], {"role": "user", "content_preview": ""}
                    )
                else:
                    self.assertIsNone(event)

    def test_a_field_with_the_wrong_type_reads_as_missing(self):
        def llm(request_id, method, message_count, last_message):
            return {
                "method": method, "model": "", "message_count": message_count,
                "request_id": request_id, "last_message": last_message,
            }

        cases = [
            ("agent_to_server", {"id": 1, "method": "tools/call", "params": None},
             EventType.TOOL_CALL, {"tool_name": "unknown", "arguments": {}, "request_id": 1}),
            ("agent_to_server", {"id": 1, "method": "tools/call", "params": {"name": 5, "arguments": ["x"]}},
             EventType.TOOL_CALL, {"tool_name": "unknown", "arguments": {}, "request_id": 1}),
            ("agent_to_server", {"id": 2, "method": "resources/read", "params": ["x"]},
             EventType.FILE_READ, {"uri": "", "request_id": 2}),
            ("agent_to_server", {"id": 2, "method": "resources/read", "params": {"uri": 5}},
             EventType.FILE_READ, {"uri": "", "request_id": 2}),
            ("agent_to_server", {"id": 3, "method": "sampling/createMessage", "params": {"messages": "hi"}},
             EventType.LLM_REQUEST, llm(3, "sampling/createMessage", 0, None)),
            # The last message is not an object, so it has no role or preview.
            ("agent_to_server",
             {"id": 4, "method": "completion/create", "params": {"model": 5, "messages": [{"role": "user"}, "hi"]}},
             EventType.LLM_REQUEST, llm(4, "completion/create", 2, None)),
            ("agent_to_server",
             {"id": 5, "method": "completion/create", "params": {"messages": [{"role": 5, "content": {"text": 5}}]}},
             EventType.LLM_REQUEST, llm(5, "completion/create", 1, {"role": "", "content_preview": ""})),
            # An error that is not an object still means the request failed.
            ("server_to_agent", {"id": 6, "error": "boom"},
             EventType.ERROR, {"code": None, "message": "boom", "request_id": 6}),
            ("server_to_agent", {"id": 7, "error": ["x"]},
             EventType.ERROR, {"code": None, "message": "", "request_id": 7}),
            ("server_to_agent", {"id": 8, "error": {"code": "x", "message": 5}},
             EventType.ERROR, {"code": None, "message": "", "request_id": 8}),
            ("server_to_agent", {"id": 9, "result": {"content": ["a", {"type": 5, "text": 5}]}},
             EventType.TOOL_RESULT, {"request_id": 9, "content_types": ["unknown", "unknown"], "content_preview": ""}),
            ("server_to_agent", {"method": "notifications/resources/updated", "params": ["x"]},
             EventType.FILE_WRITE, {"uri": ""}),
            ("server_to_agent", {"method": "notifications/resources/updated", "params": {"uri": ["x"]}},
             EventType.FILE_WRITE, {"uri": ""}),
        ]
        for direction, message, event_type, data in cases:
            with self.subTest(message=message):
                event = _classify_message(message, direction)
                self.assertEqual((event.event_type, event.data), (event_type, data))
        for message in ({"id": 10, "result": "x"}, {"id": 11, "result": {"content": "text"}}):
            with self.subTest(message=message):
                self.assertIsNone(_classify_message(message, "server_to_agent"))

    def test_an_error_that_is_present_and_not_null_is_an_error(self):
        result = {"content": [{"type": "text", "text": "ok"}]}
        for error in FALSY_ERRORS:
            # An error wins over a result sent beside it, as a truthy one does.
            for message in ({"id": 1, "error": error}, {"id": 1, "result": result, "error": error}):
                with self.subTest(message=message):
                    event = _classify_message(message, "server_to_agent")
                    self.assertEqual(
                        (event.event_type, event.data),
                        (EventType.ERROR, {"code": None, "message": "", "request_id": 1}),
                    )
        # JSON-RPC 1.0 sends "error": null beside a result.
        self.assertIsNone(_classify_message({"id": 2, "error": None}, "server_to_agent"))
        event = _classify_message({"id": 3, "result": result, "error": None}, "server_to_agent")
        self.assertEqual(
            (event.event_type, event.data),
            (EventType.TOOL_RESULT, {"request_id": 3, "content_types": ["text"], "content_preview": "ok"}),
        )

    def test_well_formed_messages_give_the_same_events_as_before(self):
        image = {"type": "image", "data": "AAAA", "mimeType": "image/png"}
        cases = [
            ("agent_to_server", _call(1),
             (EventType.TOOL_CALL, {"tool_name": "read_file", "arguments": {}, "request_id": 1})),
            ("agent_to_server",
             {"id": "r2", "method": "tools/call", "params": {"name": "list", "arguments": {"path": "/tmp/x"}}},
             (EventType.TOOL_CALL, {"tool_name": "list", "arguments": {"path": "/tmp/x"}, "request_id": "r2"})),
            ("agent_to_server", {"id": 2, "method": "resources/read", "params": {"uri": "file:///tmp/x"}},
             (EventType.FILE_READ, {"uri": "file:///tmp/x", "request_id": 2})),
            ("agent_to_server",
             {"id": 3, "method": "sampling/createMessage",
              "params": {"messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}]}},
             (EventType.LLM_REQUEST, {
                 "method": "sampling/createMessage", "model": "", "message_count": 1, "request_id": 3,
                 "last_message": {"role": "user", "content_preview": "hi"},
             })),
            ("agent_to_server",
             {"id": 4, "method": "completion/create",
              "params": {"model": "m1", "messages": [{"role": "system", "content": "be brief"},
                                                     {"role": "user", "content": "x" * 300}]}},
             (EventType.LLM_REQUEST, {
                 "method": "completion/create", "model": "m1", "message_count": 2, "request_id": 4,
                 "last_message": {"role": "user", "content_preview": "x" * 200},
             })),
            ("agent_to_server",
             {"id": 5, "method": "completion/create",
              "params": {"messages": [{"role": "user", "content": {"type": "text", "text": "z" * 300}}]}},
             (EventType.LLM_REQUEST, {
                 "method": "completion/create", "model": "", "message_count": 1, "request_id": 5,
                 "last_message": {"role": "user", "content_preview": "z" * 200 + "..."},
             })),
            ("agent_to_server", {"id": 6, "method": "completion/create", "params": {}},
             (EventType.LLM_REQUEST, {
                 "method": "completion/create", "model": "", "message_count": 0, "request_id": 6,
                 "last_message": None,
             })),
            ("server_to_agent", _error(9),
             (EventType.ERROR, {"code": -32602, "message": "bad", "request_id": 9})),
            ("server_to_agent", {"id": 10, "error": {"code": -1}},
             (EventType.ERROR, {"code": -1, "message": "", "request_id": 10})),
            ("server_to_agent",
             {"id": 11, "result": {"content": [{"type": "text", "text": "y" * 250}, image]}},
             (EventType.TOOL_RESULT, {
                 "request_id": 11, "content_types": ["text", "image"], "content_preview": "y" * 200 + "...",
             })),
            ("server_to_agent", {"id": 12, "result": {"content": [image]}},
             (EventType.TOOL_RESULT, {"request_id": 12, "content_types": ["image"], "content_preview": ""})),
            ("server_to_agent", {"method": "notifications/resources/updated", "params": {"uri": "file:///x"}},
             (EventType.FILE_WRITE, {"uri": "file:///x"})),
            ("agent_to_server", {"id": 7, "method": "ping"}, None),
            ("agent_to_server", _result(8), None),
            ("server_to_agent", {"id": 13, "result": {"content": []}}, None),
            ("server_to_agent", {"id": 14, "result": {}}, None),
            ("server_to_agent", {"result": {"content": [{"type": "text", "text": "no id"}]}}, None),
            ("server_to_agent", {"method": "notifications/progress", "params": {"progress": 1}}, None),
            ("server_to_agent", _call(15), None),
        ]
        for direction, message, expected in cases:
            with self.subTest(direction=direction, message=message):
                event = _classify_message(message, direction)
                got = None if event is None else (event.event_type, event.data)
                self.assertEqual(got, expected)
                if event is not None:
                    # Key order shows in the stored JSON.
                    self.assertEqual(list(event.data), list(expected[1]))


class StdioProxyRelayTest(unittest.TestCase):
    """The stdio proxy forwards every message, including one it cannot trace."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name, redact=False)
        self.meta = SessionMeta()
        self.store.create_session(self.meta)
        self.proxy = MCPProxy(["true"], self.store, self.meta)

    def _relay(self, data, direction):
        """Relay data in one direction until it runs out.

        Returns the lines forwarded and what went to stderr.
        """
        dest = io.BytesIO()
        stderr = io.StringIO()
        stop = threading.Event()
        with patch("sys.stderr", stderr):
            self.proxy._forward_and_trace(io.BytesIO(data), dest, direction, stop)
        self.assertTrue(stop.is_set())
        return dest.getvalue().splitlines(), stderr.getvalue()

    def _request_ids(self, event_type):
        return [
            e.data["request_id"] for e in self.store.load_events(self.meta.session_id)
            if e.event_type == event_type
        ]

    def test_messages_with_wrong_typed_fields_are_forwarded(self):
        cases = [("agent_to_server", odd, _call, EventType.TOOL_CALL) for odd in ODD_REQUESTS]
        cases += [("server_to_agent", odd, _result, EventType.TOOL_RESULT) for odd in ODD_RESPONSES]
        for request_id, (direction, odd, good, event_type) in enumerate(cases, start=100):
            with self.subTest(direction=direction, message=odd):
                lines, stderr = self._relay(_line(odd) + _line(good(request_id)), direction)
                self.assertEqual([json.loads(line) for line in lines], [odd, good(request_id)])
                self.assertEqual(stderr, "")
                self.assertIn(request_id, self._request_ids(event_type))

    def test_json_too_deep_for_json_loads_is_forwarded(self):
        # Python 3.10 and 3.11 cannot read this. Later versions trace it.
        deep = (
            '{"jsonrpc": "2.0", "id": 7, "result": {"content": [{"type": "text", "text": "deep"}], '
            '"structuredContent": {"rows": ' + _deep_text(DEEP) + "}}}"
        )
        lines, _ = self._relay(deep.encode() + b"\n" + _line(_result(8)), "server_to_agent")
        self.assertEqual(lines, [deep.encode(), json.dumps(_result(8)).encode()])
        self.assertIn(8, self._request_ids(EventType.TOOL_RESULT))

    def test_a_call_too_deep_to_store_is_forwarded_but_not_traced(self):
        deep = _deep_call(7)
        lines, stderr = self._relay(deep.encode() + b"\n" + _line(_call(8)), "agent_to_server")
        self.assertEqual(lines, [deep.encode(), json.dumps(_call(8)).encode()])
        self.assertEqual(stderr, UNTRACED_LOG.format("RecursionError"))
        self.assertEqual(self._request_ids(EventType.TOOL_CALL), [8])
        # Its result will not link to a call the trace does not have.
        self.assertEqual(list(self.proxy._calls.pending()), [8])

    def test_a_note_stderr_cannot_take_does_not_stop_the_relay(self):
        class ClosedPipe(io.StringIO):
            def write(self, text):
                raise BrokenPipeError(32, "Broken pipe")

        deep = _deep_call(7)
        dest = io.BytesIO()
        source = io.BytesIO(deep.encode() + b"\n" + _line(_call(8)))
        with patch("sys.stderr", ClosedPipe()):
            self.proxy._forward_and_trace(source, dest, "agent_to_server", threading.Event())
        self.assertEqual(
            dest.getvalue().splitlines(), [deep.encode(), json.dumps(_call(8)).encode()]
        )
        self.assertEqual(self._request_ids(EventType.TOOL_CALL), [8])

    def test_an_integer_too_long_for_json_loads_is_forwarded(self):
        digits = _too_long_int()
        if digits is None:
            self.skipTest("int() has no digit limit")
        big = '{"jsonrpc": "2.0", "id": 7, "result": {"content": [], "n": ' + digits + "}}"
        lines, stderr = self._relay(big.encode() + b"\n" + _line(_result(8)), "server_to_agent")
        self.assertEqual(lines, [big.encode(), json.dumps(_result(8)).encode()])
        self.assertEqual(stderr, UNTRACED_LOG.format("ValueError"))
        self.assertEqual(self._request_ids(EventType.TOOL_RESULT), [8])

    def test_a_framed_message_json_loads_cannot_read_goes_out_on_one_line(self):
        digits = _too_long_int()
        if digits is None:
            self.skipTest("int() has no digit limit")
        head = '{"jsonrpc": "2.0",'
        tail = '"id": 7, "result": {"content": [], "n": ' + digits + "}}"
        data = _framed(head + "\r\n" + tail) + _framed(json.dumps(_result(8)))
        lines, stderr = self._relay(data, "server_to_agent")
        self.assertEqual(lines, [(head + "  " + tail).encode(), json.dumps(_result(8)).encode()])
        self.assertEqual(stderr, UNTRACED_LOG.format("ValueError"))
        self.assertEqual(self._request_ids(EventType.TOOL_RESULT), [8])

    def test_a_message_json_dumps_cannot_write_is_forwarded_as_it_arrived(self):
        # In a thread on Python 3.14, json.loads reads a list nested a little
        # deeper than json.dumps can write.
        text = '{"jsonrpc": "2.0", "id": 7, "result": {"content": [{"type": "text", "text": "x"}]}}'
        dumps = json.dumps

        def fail_on_message(value, *args, **kwargs):
            if value == json.loads(text):
                raise RecursionError("maximum recursion depth exceeded")
            return dumps(value, *args, **kwargs)

        with patch("json.dumps", side_effect=fail_on_message):
            lines, stderr = self._relay((text + "\n").encode() + _line(_result(8)), "server_to_agent")
        self.assertEqual(lines, [text.encode(), json.dumps(_result(8)).encode()])
        self.assertEqual(stderr, "")
        self.assertEqual(self._request_ids(EventType.TOOL_RESULT), [7, 8])

    def test_a_message_with_a_duplicate_key_goes_out_as_it_was_traced(self):
        # json.loads keeps the last "params". A server that keeps the first would
        # run another call than the trace shows, so the parsed value goes out.
        text = (
            '{"jsonrpc": "2.0", "id": 7, "method": "tools/call", '
            '"params": {"name": "first", "arguments": {}}, '
            '"params": {"name": "last", "arguments": {}}}'
        )
        lines, stderr = self._relay(text.encode() + b"\n", "agent_to_server")
        self.assertEqual(lines, [json.dumps(_call(7, "last")).encode()])
        self.assertEqual(stderr, "")
        (call,) = [
            e for e in self.store.load_events(self.meta.session_id)
            if e.event_type == EventType.TOOL_CALL
        ]
        self.assertEqual(call.data["tool_name"], "last")

    def test_a_message_the_store_cannot_take_is_still_forwarded(self):
        append_event = self.store.append_event
        failures = [OSError("No space left on device")]

        def fail_once(session_id, event):
            if failures:
                raise failures.pop()
            append_event(session_id, event)

        with patch.object(self.store, "append_event", side_effect=fail_once):
            lines, stderr = self._relay(_line(_call(7)) + _line(_call(8)), "agent_to_server")
        self.assertEqual([json.loads(line) for line in lines], [_call(7), _call(8)])
        self.assertEqual(stderr, UNTRACED_LOG.format("OSError"))
        self.assertEqual(self._request_ids(EventType.TOOL_CALL), [8])
        self.assertEqual(list(self.proxy._calls.pending()), [8])

    def test_content_length_values_that_are_not_objects_are_forwarded(self):
        bodies = ["null", "5", '"x"', "[]", '[1, "x"]']
        data = b"".join(_framed(body) for body in bodies) + _framed(json.dumps(_call(7)))
        lines, stderr = self._relay(data, "agent_to_server")
        self.assertEqual(
            [json.loads(line) for line in lines], [None, 5, "x", [], [1, "x"], _call(7)]
        )
        self.assertEqual(stderr, "")
        self.assertEqual(self._request_ids(EventType.TOOL_CALL), [7])

    def test_content_length_batch_is_traced_one_message_at_a_time(self):
        calls = [_call(7, "a"), {"jsonrpc": "2.0", "method": "notifications/initialized"}, _call(8, "b")]
        responses = [_result(8), _error(7)]
        for direction, batch in (("agent_to_server", calls), ("server_to_agent", responses)):
            lines, stderr = self._relay(_framed(json.dumps(batch)), direction)
            self.assertEqual([json.loads(line) for line in lines], [batch])
            self.assertEqual(stderr, "")
        events = self.store.load_events(self.meta.session_id)
        call_ids = {e.data["request_id"]: e.event_id for e in events if e.event_type == EventType.TOOL_CALL}
        parents = {
            e.data["request_id"]: e.parent_id for e in events
            if e.event_type in (EventType.TOOL_RESULT, EventType.ERROR)
        }
        self.assertEqual(len(call_ids), 2)
        self.assertEqual(parents, call_ids)
        self.assertEqual(self.proxy._calls.pending(), {})


SRC = os.path.dirname(os.path.dirname(os.path.abspath(agent_trace.__file__)))

# Fake MCP servers. Each reply is _result(id).
_REPLY = (
    "import json, os, sys, time\n"
    "def reply(i):\n"
    "    return json.dumps({'jsonrpc': '2.0', 'id': i,\n"
    "        'result': {'content': [{'type': 'text', 'text': 'ok'}]}}) + '\\n'\n"
)

# Answers each request, then says on stderr that its stdin closed.
ECHO_SERVER = _REPLY + (
    "for line in sys.stdin:\n"
    "    sys.stdout.write(reply(json.loads(line)['id']))\n"
    "    sys.stdout.flush()\n"
    "sys.stderr.write('server saw EOF\\n')\n"
)

# Writes 200 replies without reading stdin, then exits with code 3.
BURST_SERVER = _REPLY + (
    "for i in range(200):\n"
    "    sys.stdout.write(reply(i))\n"
    "sys.stdout.flush()\n"
    "sys.stderr.write('burst done\\n')\n"
    "sys.exit(3)\n"
)

# Writes replies until it is stopped.
ENDLESS_SERVER = _REPLY + (
    "i = 0\n"
    "while True:\n"
    "    sys.stdout.write(reply(i))\n"
    "    sys.stdout.flush()\n"
    "    i += 1\n"
    "    time.sleep(0.01)\n"
)

# Writes its pid to the file named by its argument, then idles.
IDLE_SERVER = (
    "import os, sys, time\n"
    "with open(sys.argv[1] + '.tmp', 'w') as f:\n"
    "    f.write(str(os.getpid()))\n"
    "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
    "time.sleep(60)\n"
)

# Started by HOLDING_SERVER with its stdout. Writes its pid, then a reply and
# a heartbeat every 0.05s: 40 times if told to leave the server's process
# group (POSIX), else for a minute. Touches <pid file>.done when it ends.
HOLDING_CHILD = _REPLY + (
    "pid_file, beat_file, escape = sys.argv[1:4]\n"
    "if escape == '1':\n"
    "    os.setsid()\n"
    "with open(pid_file + '.tmp', 'w') as f:\n"
    "    f.write(str(os.getpid()))\n"
    "os.replace(pid_file + '.tmp', pid_file)\n"
    "for i in range(40 if escape == '1' else 1200):\n"
    "    with open(beat_file, 'a') as f:\n"
    "        f.write('x')\n"
    "    try:\n"
    "        sys.stdout.write(reply(1000 + i))\n"
    "        sys.stdout.flush()\n"
    "    except OSError:\n"
    "        pass\n"
    "    time.sleep(0.05)\n"
    "open(pid_file + '.done', 'w').close()\n"
)

# Starts HOLDING_CHILD with its own stdout, writes one reply once the child
# is running, then exits 0 while the child still holds stdout open.
HOLDING_SERVER = _REPLY + (
    "import subprocess\n"
    "child, pid_file = sys.argv[1:3]\n"
    "subprocess.Popen([sys.executable, *sys.argv[1:]], stdin=subprocess.DEVNULL,\n"
    "                 stdout=sys.stdout, stderr=subprocess.DEVNULL)\n"
    "while not os.path.exists(pid_file):\n"
    "    time.sleep(0.01)\n"
    "sys.stdout.write(reply(0))\n"
    "sys.stdout.flush()\n"
)

# Starts a child that writes its pid to the file named by its argument and
# sleeps, then idles itself.
TREE_SERVER = (
    "import subprocess, sys, time\n"
    "child = ('import os, sys, time\\n'\n"
    "         'with open(sys.argv[1] + \".tmp\", \"w\") as f: f.write(str(os.getpid()))\\n'\n"
    "         'os.replace(sys.argv[1] + \".tmp\", sys.argv[1])\\n'\n"
    "         'time.sleep(60)\\n')\n"
    "subprocess.Popen([sys.executable, '-c', child, sys.argv[1]],\n"
    "                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,\n"
    "                 stderr=subprocess.DEVNULL)\n"
    "time.sleep(60)\n"
)

# Runs agent-strace with a stop signal sent to itself as soon as the server
# process exists, before record has done anything else with it.
# argv: signal name, marker argument of the server command, agent-strace argv.
_SIGNAL_AT_SPAWN = (
    "import os, signal, subprocess, sys\n"
    "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
    "sig, marker = getattr(signal, sys.argv[1]), sys.argv[2]\n"
    "class Popen(subprocess.Popen):\n"
    "    def __init__(self, args, *a, **k):\n"
    "        super().__init__(args, *a, **k)\n"
    "        if isinstance(args, list) and marker in args:\n"
    "            with open(marker + '.popen', 'w') as f:\n"
    "                f.write(str(self.pid))\n"
    "            os.kill(os.getpid(), sig)\n"
    "subprocess.Popen = Popen\n"
    "from agent_trace.entry import main\n"
    "sys.argv = ['agent-strace'] + sys.argv[3:]\n"
    "sys.exit(main())\n"
)


def _pid_gone(pid, timeout=5.0):
    """True once no process has this pid (POSIX), polling up to timeout."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def _kill_pid(pid):
    import signal

    try:
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except OSError:
        pass


class _Sink:
    """A stand-in for the agent's stdout or stderr."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = b""

    def write(self, data):
        with self.lock:
            self.data += bytes(data)

    def flush(self):
        pass


# Deletes the session directory named by its argument, then exits.
PARENT_DELETING_SERVER = "import shutil, sys\nshutil.rmtree(sys.argv[1])\n"

# Runs agent-strace with SIGINT handled by Python, even when the test
# process was started with SIGINT ignored, which a child would inherit.
_SIGINT_DEFAULT = [
    sys.executable, "-c",
    "import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); "
    "os.execv(sys.executable, [sys.executable] + sys.argv[1:])",
]


class StdioProxyLifecycleTest(unittest.TestCase):
    """`record` ends when either side goes away and loses no server output."""

    TIMEOUT = 30

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.trace_dir = os.path.join(self.root, "traces")
        home = os.path.join(self.root, "home")
        os.makedirs(home)
        self.env = _no_telemetry.without_telemetry(
            {**os.environ, "PYTHONPATH": SRC, "HOME": home, "USERPROFILE": home}
        )

    def _record(self, server, stdin, *server_args, server_cmd=None, record_args=(),
                launcher=(sys.executable,)):
        """Start `agent-strace record` on a fake server script."""
        if server_cmd is None:
            script = os.path.join(self.root, "server.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(server)
            server_cmd = [sys.executable, script, *server_args]
        return subprocess.Popen(
            [*launcher, "-m", "agent_trace.entry", "--trace-dir", self.trace_dir,
             "record", "--quiet", *record_args, "--", *server_cmd],
            cwd=self.root, env=self.env,
            stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def _finish(self, proc, data=b"", keep_stdin_open=False, before_wait=None):
        """Send data, wait for record to exit and return (code, stdout, stderr).

        With keep_stdin_open, the agent's side of stdin stays open until
        record exits, like an agent that is still running.
        """
        out, err = [], []
        readers = [
            threading.Thread(target=lambda s=s, b=b: b.append(s.read()), daemon=True)
            for s, b in ((proc.stdout, out), (proc.stderr, err))
            if s is not None and not s.closed
        ]
        for reader in readers:
            reader.start()
        try:
            if proc.stdin is not None:
                proc.stdin.write(data)
                proc.stdin.flush()
                if not keep_stdin_open:
                    proc.stdin.close()
            if before_wait is not None:
                before_wait()
            try:
                code = proc.wait(timeout=self.TIMEOUT)
            except subprocess.TimeoutExpired:
                self.fail(f"record did not exit within {self.TIMEOUT}s")
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            for reader in readers:
                reader.join(5)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
        return code, b"".join(out), b"".join(err)

    def _session(self, session_id=None):
        store = TraceStore(self.trace_dir)
        if session_id is None:
            (meta,) = store.list_sessions()
        else:
            meta = store.load_meta(session_id)
        return meta, store.load_events(meta.session_id)

    def _replies(self, out):
        return [json.loads(line) for line in out.splitlines()]

    def test_agent_eof_closes_the_server_stdin(self):
        proc = self._record(ECHO_SERVER, subprocess.PIPE)
        code, out, err = self._finish(proc, _line(_call(1)))

        self.assertEqual(code, 0, err)
        self.assertEqual(self._replies(out), [_result(1)])
        self.assertIn(b"server saw EOF", err)
        meta, events = self._session()
        self.assertIsNotNone(meta.ended_at)
        self.assertEqual(
            [e.event_type for e in events],
            [EventType.SESSION_START, EventType.TOOL_CALL, EventType.TOOL_RESULT,
             EventType.SESSION_END],
        )
        self.assertEqual(events[-1].data["exit_code"], 0)

    def test_all_server_output_reaches_the_agent_before_session_end(self):
        # Held open, the agent's stdin is still being read when the server
        # exits. record must still exit with the server's code.
        for name, stdin, keep_open in (
            ("agent stdin closed", subprocess.DEVNULL, False),
            ("agent stdin held open", subprocess.PIPE, True),
        ):
            with self.subTest(name):
                self.trace_dir = os.path.join(self.root, name.replace(" ", "-"))
                proc = self._record(BURST_SERVER, stdin)
                code, out, err = self._finish(proc, keep_stdin_open=keep_open)

                self.assertEqual(code, 3, err)
                self.assertEqual(self._replies(out), [_result(i) for i in range(200)])
                self.assertEqual(err.splitlines(), [b"burst done"])
                meta, events = self._session()
                self.assertIsNotNone(meta.ended_at)
                self.assertEqual(
                    [e.data["request_id"] for e in events
                     if e.event_type == EventType.TOOL_RESULT],
                    list(range(200)),
                )
                self.assertEqual(events[-1].event_type, EventType.SESSION_END)
                self.assertEqual(events[-1].data["exit_code"], 3)

    def test_an_agent_that_stops_reading_ends_the_session(self):
        proc = self._record(ENDLESS_SERVER, subprocess.PIPE)
        proc.stdout.close()
        code, _, err = self._finish(proc, keep_stdin_open=True)

        self.assertNotIn(b"Traceback", err)
        self.assertNotIn(b"Exception ignored", err)
        meta, events = self._session()
        self.assertIsNotNone(meta.ended_at)
        self.assertEqual(events[-1].event_type, EventType.SESSION_END)
        # The exit status holds the server's code modulo 256: -15 exits as 241.
        self.assertEqual(events[-1].data["exit_code"] % 256, code % 256)

    def test_a_server_that_cannot_start_ends_the_session(self):
        missing = os.path.join(self.root, "no-such-server")
        proc = self._record("", subprocess.DEVNULL, server_cmd=[missing])
        code, out, err = self._finish(proc)

        self.assertEqual(code, 127)
        self.assertEqual(out, b"")
        self.assertNotIn(b"Traceback", err)
        self.assertTrue(
            err.decode().startswith("agent-strace: could not start the MCP server: "), err
        )
        meta, events = self._session()
        self.assertIsNotNone(meta.ended_at)
        self.assertEqual(
            [e.event_type for e in events],
            [EventType.SESSION_START, EventType.ERROR, EventType.SESSION_END],
        )
        self.assertEqual(events[-1].data["exit_code"], 127)

    def test_a_parent_deleted_while_recording_is_dropped_at_the_end(self):
        store = TraceStore(self.trace_dir)
        parent = SessionMeta()
        store.create_session(parent)
        parent_dir = str(store.base_dir / parent.session_id)

        proc = self._record(
            PARENT_DELETING_SERVER, subprocess.DEVNULL, parent_dir,
            record_args=("--parent", parent.session_id),
        )
        code, _, err = self._finish(proc)

        self.assertEqual(code, 0, err)
        self.assertIn(
            f"agent-strace: dropped the link to parent session {parent.session_id}: ".encode(),
            err,
        )
        self.assertNotIn(b"Traceback", err)
        meta, events = self._session()
        self.assertIsNotNone(meta.ended_at)
        self.assertEqual(meta.parent_session_id, "")
        self.assertEqual(events[-1].event_type, EventType.SESSION_END)

    @unittest.skipIf(sys.platform == "win32", "POSIX signals")
    def test_a_stop_signal_stops_the_server_and_ends_the_session(self):
        import signal
        import time

        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(sig.name):
                self.trace_dir = os.path.join(self.root, sig.name)
                pid_file = os.path.join(self.root, f"{sig.name}.pid")
                proc = self._record(
                    IDLE_SERVER, subprocess.PIPE, pid_file, launcher=_SIGINT_DEFAULT
                )
                server_pid = []
                orphaned = False

                def send_signal():
                    deadline = time.monotonic() + self.TIMEOUT
                    while not os.path.exists(pid_file):
                        if time.monotonic() > deadline or proc.poll() is not None:
                            self.fail("the server did not start")
                        time.sleep(0.05)
                    with open(pid_file, encoding="utf-8") as f:
                        server_pid.append(int(f.read()))
                    proc.send_signal(sig)

                try:
                    code, _, err = self._finish(
                        proc, keep_stdin_open=True, before_wait=send_signal
                    )
                finally:
                    for pid in server_pid:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            orphaned = True
                        except ProcessLookupError:
                            orphaned = False
                self.assertFalse(orphaned, "the server outlived record")
                self.assertEqual(code, 128 + sig, err)
                self.assertNotIn(b"Traceback", err)
                meta, events = self._session()
                self.assertIsNotNone(meta.ended_at)
                self.assertEqual(
                    [e.event_type for e in events],
                    [EventType.SESSION_START, EventType.SESSION_END],
                )
                self.assertEqual(events[-1].data["exit_code"], -signal.SIGTERM)


    def _wait_for_file(self, path):
        import time

        deadline = time.monotonic() + self.TIMEOUT
        while not os.path.exists(path):
            if time.monotonic() > deadline:
                self.fail(f"{os.path.basename(path)} never appeared")
            time.sleep(0.02)
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _run_holding_server(self, escape):
        """Run the proxy in this process on HOLDING_SERVER, with short timeouts."""
        child = os.path.join(self.root, "child.py")
        server = os.path.join(self.root, "server.py")
        for path, text in ((child, HOLDING_CHILD), (server, HOLDING_SERVER)):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        self.pid_file = os.path.join(self.root, "child.pid")
        self.beat_file = os.path.join(self.root, "beat")

        store = TraceStore(self.trace_dir)
        meta = SessionMeta()
        store.create_session(meta)
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        agent_in = open(read_fd, "rb")
        self.addCleanup(agent_in.close)
        self.agent_out, agent_err = _Sink(), _Sink()

        proxy = MCPProxy(
            [sys.executable, server, child, self.pid_file, self.beat_file, escape],
            store, meta,
        )
        try:
            with patch.object(proxy_module, "DRAIN_TIMEOUT", 0.5), \
                    patch.object(proxy_module, "STOP_TIMEOUT", 1.0), \
                    patch.object(proxy_module, "_agent_streams",
                                 return_value=(agent_in, self.agent_out, agent_err)):
                code = proxy.run()
        finally:
            if os.path.exists(self.pid_file):
                pid = int(self._wait_for_file(self.pid_file))
                self.addCleanup(_kill_pid, pid)
        return code, agent_err.data, store, meta

    def _assert_final(self, store, meta):
        """session_end is the last event and the saved counters match the trace."""
        events = store.load_events(meta.session_id)
        saved = store.load_meta(meta.session_id)
        self.assertEqual(events[-1].event_type, EventType.SESSION_END)
        self.assertEqual(
            [e for e in events if e.event_type == EventType.SESSION_END], [events[-1]]
        )
        for counter, event_type in (("tool_calls", EventType.TOOL_CALL),
                                    ("errors", EventType.ERROR)):
            count = sum(e.event_type == event_type for e in events)
            self.assertEqual(getattr(saved, counter), count, counter)
            self.assertEqual(getattr(meta, counter), count, counter)
        return events

    def test_output_held_past_the_drain_timeout_is_stopped_and_recorded(self):
        import time

        code, err, store, meta = self._run_holding_server("0")

        self.assertEqual(code, 0)
        self.assertIn(b"agent-strace: server output was cut short", err)
        events = self._assert_final(store, meta)
        self.assertTrue(events[-1].data.get("output_truncated"))
        self.assertEqual(events[-2].event_type, EventType.ERROR)
        self.assertIn("cut short", events[-2].data["message"])
        self.assertIn(_line(_result(0)), self.agent_out.data)
        # The child holding the output was stopped with the server.
        beats = os.path.getsize(self.beat_file)
        time.sleep(0.5)
        self.assertEqual(os.path.getsize(self.beat_file), beats, "the child kept running")
        self.assertEqual(len(store.load_events(meta.session_id)), len(events))

    @unittest.skipIf(os.name != "posix", "a child leaving the process group needs setsid")
    def test_nothing_is_relayed_or_recorded_after_session_end(self):
        import time

        code, _, store, meta = self._run_holding_server("1")
        relayed = self.agent_out.data
        events = self._assert_final(store, meta)

        # The child left the server's process group, so it outlives the
        # stop and keeps writing after the session has ended.
        self._wait_for_file(self.pid_file + ".done")
        time.sleep(0.3)
        self.assertEqual(code, 0)
        self.assertTrue(events[-1].data.get("output_truncated"))
        self.assertEqual(self.agent_out.data, relayed)
        self.assertEqual(self._assert_final(store, meta), events)

    @unittest.skipIf(sys.platform == "win32", "POSIX signals")
    def test_a_stop_signal_stops_every_process_the_server_started(self):
        import signal

        pid_file = os.path.join(self.root, "child.pid")
        proc = self._record(TREE_SERVER, subprocess.PIPE, pid_file)
        child_pid = []

        def send_signal():
            child_pid.append(int(self._wait_for_file(pid_file)))
            proc.send_signal(signal.SIGTERM)

        try:
            code, _, err = self._finish(proc, keep_stdin_open=True, before_wait=send_signal)
            self.assertEqual(code, 128 + signal.SIGTERM, err)
            self.assertTrue(_pid_gone(child_pid[0]), "the server's child outlived record")
        finally:
            for pid in child_pid:
                _kill_pid(pid)
        meta, events = self._session()
        self.assertIsNotNone(meta.ended_at)
        self.assertEqual(events[-1].event_type, EventType.SESSION_END)

    @unittest.skipIf(sys.platform == "win32", "POSIX signals")
    def test_a_stop_signal_right_after_the_server_starts_stops_it(self):
        import signal

        script = os.path.join(self.root, "server.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(IDLE_SERVER)
        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(sig.name):
                self.trace_dir = os.path.join(self.root, sig.name)
                marker = os.path.join(self.root, f"{sig.name}.pid")
                proc = subprocess.Popen(
                    [sys.executable, "-c", _SIGNAL_AT_SPAWN, sig.name, marker,
                     "--trace-dir", self.trace_dir, "record", "--quiet", "--",
                     sys.executable, script, marker],
                    cwd=self.root, env=self.env, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                server_pid = []
                try:
                    code, _, err = self._finish(proc, keep_stdin_open=True)
                    server_pid.append(int(self._wait_for_file(marker + ".popen")))
                    self.assertTrue(_pid_gone(server_pid[0]), "the server outlived record")
                finally:
                    for pid in server_pid:
                        _kill_pid(pid)
                self.assertEqual(code, 128 + sig, err)
                self.assertNotIn(b"Traceback", err)
                meta, events = self._session()
                self.assertIsNotNone(meta.ended_at)
                self.assertEqual(events[-1].event_type, EventType.SESSION_END)


if __name__ == "__main__":
    unittest.main()
