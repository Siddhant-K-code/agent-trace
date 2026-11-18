"""Tests for the stdio MCP proxy: pairing, and relaying what it cannot trace."""

import io
import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
