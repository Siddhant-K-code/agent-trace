"""Tests for push-based event streaming (Issue #103)."""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import session_to_otlp_genai
from agent_trace.store import TraceStore
from agent_trace.watch import EventStreamer, StreamConfig, WatcherConfig, watch_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: EventType = EventType.TOOL_CALL, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id="s1", data={})


class _CollectorHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that collects NDJSON POST bodies."""

    received: ClassVar[list[bytes]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _CollectorHandler.received.append(body)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:  # suppress access logs
        pass


# ---------------------------------------------------------------------------
# StreamConfig tests
# ---------------------------------------------------------------------------

class TestStreamConfig(unittest.TestCase):
    def test_from_url(self) -> None:
        cfg = StreamConfig.from_url("http://example.com/events")
        self.assertEqual(cfg.url, "http://example.com/events")
        self.assertEqual(cfg.headers, {})

    def test_from_url_with_headers(self) -> None:
        cfg = StreamConfig.from_url("http://x.com", headers={"Authorization": "Bearer tok"})
        self.assertEqual(cfg.headers["Authorization"], "Bearer tok")

    def test_defaults(self) -> None:
        cfg = StreamConfig(url="http://x.com")
        self.assertEqual(cfg.batch_size, 10)
        self.assertEqual(cfg.flush_interval, 1.0)
        self.assertEqual(cfg.event_types, [])


# ---------------------------------------------------------------------------
# EventStreamer unit tests (mock HTTP)
# ---------------------------------------------------------------------------

class TestEventStreamerUnit(unittest.TestCase):
    def _make_streamer(self, **kwargs) -> EventStreamer:
        cfg = StreamConfig(url="http://localhost:9999/events", **kwargs)
        return EventStreamer(cfg)

    def test_enqueue_and_flush_on_stop(self) -> None:
        """Events enqueued before stop() are flushed."""
        flushed: list[list[TraceEvent]] = []

        streamer = self._make_streamer(batch_size=100, flush_interval=60.0)
        streamer._flush = lambda batch: flushed.append(list(batch))  # type: ignore[method-assign]

        for _ in range(3):
            streamer.enqueue(_make_event())
        streamer.stop()

        self.assertEqual(len(flushed), 1)
        self.assertEqual(len(flushed[0]), 3)

    def test_batch_size_triggers_flush(self) -> None:
        """Reaching batch_size triggers an intermediate flush."""
        flushed: list[list[TraceEvent]] = []

        streamer = self._make_streamer(batch_size=3, flush_interval=60.0)
        streamer._flush = lambda batch: flushed.append(list(batch))  # type: ignore[method-assign]

        for _ in range(6):
            streamer.enqueue(_make_event())
        # Give worker time to process
        time.sleep(0.1)
        streamer.stop()

        total = sum(len(b) for b in flushed)
        self.assertEqual(total, 6)
        # Each batch should be <= batch_size
        for batch in flushed:
            self.assertLessEqual(len(batch), 3)

    def test_flush_interval_triggers_flush(self) -> None:
        """flush_interval causes a flush even when batch_size not reached."""
        flushed: list[list[TraceEvent]] = []

        streamer = self._make_streamer(batch_size=100, flush_interval=0.05)
        streamer._flush = lambda batch: flushed.append(list(batch))  # type: ignore[method-assign]

        streamer.enqueue(_make_event())
        time.sleep(0.2)  # wait for interval flush
        streamer.stop()

        self.assertGreater(len(flushed), 0)
        self.assertEqual(flushed[0][0].session_id, "s1")

    def test_event_type_filter(self) -> None:
        """Only events matching event_types are enqueued."""
        flushed: list[list[TraceEvent]] = []

        streamer = self._make_streamer(
            batch_size=100,
            flush_interval=60.0,
            event_types=["tool_call"],
        )
        streamer._flush = lambda batch: flushed.append(list(batch))  # type: ignore[method-assign]

        streamer.enqueue(_make_event(EventType.TOOL_CALL))
        streamer.enqueue(_make_event(EventType.LLM_RESPONSE))
        streamer.enqueue(_make_event(EventType.TOOL_CALL))
        streamer.stop()

        total = sum(len(b) for b in flushed)
        self.assertEqual(total, 2)

    def test_empty_event_types_passes_all(self) -> None:
        """Empty event_types list means no filtering."""
        flushed: list[list[TraceEvent]] = []

        streamer = self._make_streamer(batch_size=100, flush_interval=60.0)
        streamer._flush = lambda batch: flushed.append(list(batch))  # type: ignore[method-assign]

        streamer.enqueue(_make_event(EventType.TOOL_CALL))
        streamer.enqueue(_make_event(EventType.LLM_RESPONSE))
        streamer.enqueue(_make_event(EventType.SESSION_END))
        streamer.stop()

        total = sum(len(b) for b in flushed)
        self.assertEqual(total, 3)

    def test_stop_is_idempotent(self) -> None:
        """Calling stop() twice does not raise."""
        streamer = self._make_streamer()
        streamer._flush = lambda batch: None  # type: ignore[method-assign]
        streamer.stop()
        # Second stop should not hang or raise
        streamer._queue.put(None)  # re-add sentinel so join returns quickly
        streamer.stop()


# ---------------------------------------------------------------------------
# EventStreamer HTTP integration test
# ---------------------------------------------------------------------------

class TestEventStreamerHTTP(unittest.TestCase):
    """Spin up a real HTTP server and verify NDJSON payloads arrive."""

    @classmethod
    def setUpClass(cls) -> None:
        _CollectorHandler.received = []
        cls.server = HTTPServer(("127.0.0.1", 0), _CollectorHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        _CollectorHandler.received.clear()

    def test_events_arrive_as_ndjson(self) -> None:
        url = f"http://127.0.0.1:{self.port}/events"
        cfg = StreamConfig(url=url, batch_size=2, flush_interval=0.1)
        streamer = EventStreamer(cfg)

        streamer.enqueue(_make_event(EventType.TOOL_CALL, ts=1.0))
        streamer.enqueue(_make_event(EventType.LLM_RESPONSE, ts=2.0))
        streamer.stop()

        self.assertGreater(len(_CollectorHandler.received), 0)
        # Parse all received lines
        all_lines: list[dict] = []
        for body in _CollectorHandler.received:
            for line in body.decode().strip().splitlines():
                all_lines.append(json.loads(line))

        self.assertEqual(len(all_lines), 2)
        types = {e["event_type"] for e in all_lines}
        self.assertIn("tool_call", types)
        self.assertIn("llm_response", types)

    def test_content_type_header(self) -> None:
        """Requests must carry Content-Type: application/x-ndjson."""
        received_headers: list[dict] = []

        class HeaderCapture(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                received_headers.append(dict(self.headers))
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: object) -> None:
                pass

        srv = HTTPServer(("127.0.0.1", 0), HeaderCapture)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()

        try:
            url = f"http://127.0.0.1:{port}/events"
            cfg = StreamConfig(url=url, batch_size=1, flush_interval=60.0)
            streamer = EventStreamer(cfg)
            streamer.enqueue(_make_event())
            streamer.stop()

            self.assertGreater(len(received_headers), 0)
            ct = received_headers[0].get("Content-Type", "")
            self.assertEqual(ct, "application/x-ndjson")
        finally:
            srv.shutdown()

    def test_http_error_does_not_crash_watch(self) -> None:
        """A 500 from the server is logged but does not raise."""
        class ErrorHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.send_response(500)
                self.end_headers()

            def log_message(self, *args: object) -> None:
                pass

        srv = HTTPServer(("127.0.0.1", 0), ErrorHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()

        try:
            url = f"http://127.0.0.1:{port}/events"
            cfg = StreamConfig(url=url, batch_size=1, flush_interval=60.0)
            streamer = EventStreamer(cfg)
            streamer.enqueue(_make_event())
            # Should not raise
            streamer.stop()
        finally:
            srv.shutdown()


# ---------------------------------------------------------------------------
# watch_session integration: streamer is wired in
# ---------------------------------------------------------------------------

class TestWatchSessionStreaming(unittest.TestCase):
    """Verify watch_session passes events to EventStreamer.

    _tail_events seeks to EOF and waits for new writes, so we patch it to
    replay a fixed sequence of events without touching the filesystem.
    """

    def _fake_tail(self, events: list[TraceEvent]):
        """Return a generator that yields the given events then stops."""
        def _gen(_path, poll_interval=0.5):
            yield from events
        return _gen

    def test_streamer_receives_events(self) -> None:
        import tempfile
        from agent_trace.store import TraceStore
        from agent_trace.watch import WatcherConfig, watch_session, _IDLE_SENTINEL

        fake_events = [
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=1.0,
                       session_id="s1", data={}),
            TraceEvent(event_type=EventType.SESSION_END, timestamp=2.0,
                       session_id="s1", data={}),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            from agent_trace.models import SessionMeta
            meta = SessionMeta(agent_name="test", command="test")
            session_path = store.create_session(meta)
            session_id = session_path.name
            # Create the events file so watch_session doesn't bail early
            (store._session_dir(session_id) / "events.ndjson").touch()

            enqueued: list[TraceEvent] = []
            mock_streamer = MagicMock()
            mock_streamer.enqueue.side_effect = enqueued.append

            cfg = WatcherConfig()
            stream_cfg = StreamConfig(url="http://localhost:9999/events")

            with patch("agent_trace.watch.EventStreamer", return_value=mock_streamer), \
                 patch("agent_trace.watch._tail_events", self._fake_tail(fake_events)):
                out = StringIO()
                watch_session(store, session_id, cfg, out=out, stream_config=stream_cfg)

            self.assertGreaterEqual(len(enqueued), 1)
            mock_streamer.stop.assert_called_once()

    def test_no_streamer_when_no_stream_config(self) -> None:
        """watch_session without stream_config does not create an EventStreamer."""
        import tempfile
        from agent_trace.store import TraceStore
        from agent_trace.watch import WatcherConfig, watch_session

        fake_events = [
            TraceEvent(event_type=EventType.SESSION_END, timestamp=1.0,
                       session_id="s1", data={}),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            from agent_trace.models import SessionMeta
            meta = SessionMeta(agent_name="test", command="test")
            session_path = store.create_session(meta)
            session_id = session_path.name
            (store._session_dir(session_id) / "events.ndjson").touch()

            with patch("agent_trace.watch.EventStreamer") as mock_cls, \
                 patch("agent_trace.watch._tail_events", self._fake_tail(fake_events)):
                cfg = WatcherConfig()
                watch_session(store, session_id, cfg, stream_config=None)
                mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# OTLP streaming of events whose data is not a JSON object
# ---------------------------------------------------------------------------

# Every JSON type that is not an object, with the phrase the stderr line uses.
# The values carry markers so a test can show that stderr never quotes them.
_NON_OBJECT_DATA = (
    (["SECRET-ARRAY", {"command": "SECRET-COMMAND"}], "an array"),
    ("SECRET-STRING", "a string"),
    (4242, "a number"),
    (True, "a boolean"),
    (None, "null"),
)
_QUERY = "token=SECRET-QUERY"
_SESSION = "otlp-session"
_USER = "gen_ai.user.message"
_ASSISTANT = "gen_ai.assistant.message"
_LLM_SPAN = "gen_ai.client.operation"
_BASH_SPAN = "gen_ai.tool.call/Bash"
# Event types whose data session_to_otlp_genai never reads.
_UNREAD_TYPES = (
    "session_start", "session_end", "file_read", "file_write", "decision",
    "subagent_start", "subagent_stop",
)


def _event(event_type: str, data: object, event_id: str, *, parent_id: str = "",
           session_id: str = _SESSION, ts: float = 1.0) -> TraceEvent:
    return TraceEvent(
        event_type=EventType(event_type),
        timestamp=ts,
        event_id=f"SECRET-ID-{event_id}",
        session_id=session_id,
        parent_id=f"SECRET-ID-{parent_id}" if parent_id else "",
        data=data,
        tenant_id="",
    )


def _wrapped(events: list[TraceEvent]) -> list[TraceEvent]:
    """Put a good user prompt before *events* and a good reply after them."""
    return [
        _event("user_prompt", {"prompt": "hello"}, "p1"),
        *events,
        _event("assistant_response", {"text": "done"}, "a1", ts=9.0),
    ]


def _left_out(event_type: str, phrase: str) -> str:
    return (
        f"[stream] Left one {event_type} event out of the OTLP stream because its "
        f"data is {phrase}, not an object.\n"
    )


def _read_cases(bad: object) -> dict[str, tuple[list[TraceEvent], list[str], list[str]]]:
    """Return the cases where the converter reads the non-object data *bad*.

    Each case gives its events, the types of the events left out in batch
    order, and the child spans sent for the events that are left.
    """
    return {
        "user_prompt": (
            [_event("user_prompt", bad, "x1", ts=2.0)], ["user_prompt"], [],
        ),
        "assistant_response": (
            [_event("assistant_response", bad, "x1", ts=2.0)], ["assistant_response"], [],
        ),
        "llm_request with a response": (
            [
                _event("llm_request", bad, "r1", ts=2.0),
                _event("llm_response", {"model": "m"}, "r2", parent_id="r1", ts=3.0),
            ],
            ["llm_request"],
            [_LLM_SPAN],
        ),
        "llm_request without a response": (
            [_event("llm_request", bad, "r1", ts=2.0)], ["llm_request"], [],
        ),
        "llm_response": (
            [
                _event("llm_request", {"model": "m"}, "r1", ts=2.0),
                _event("llm_response", bad, "r2", parent_id="r1", ts=3.0),
            ],
            ["llm_response"],
            [_LLM_SPAN],
        ),
        "tool_call without a result": (
            [_event("tool_call", bad, "c1", ts=2.0)], ["tool_call"], [],
        ),
        "tool_call with a tool_result": (
            [
                _event("tool_call", bad, "c1", ts=2.0),
                _event("tool_result", {"tool_name": "Bash", "result": "ok"}, "c2",
                       parent_id="c1", ts=3.0),
            ],
            ["tool_call"],
            [_BASH_SPAN],
        ),
        "tool_result": (
            [
                _event("tool_call", {"tool_name": "Bash"}, "c1", ts=2.0),
                _event("tool_result", bad, "c2", parent_id="c1", ts=3.0),
            ],
            ["tool_result"],
            [_BASH_SPAN],
        ),
        "error": ([_event("error", bad, "e1", ts=2.0)], ["error"], []),
        # The error is read and left out; the call it paired with is then
        # unpaired, so the converter reads it too.
        "error and the tool_call it pairs with": (
            [
                _event("tool_call", bad, "c1", ts=2.0),
                _event("error", bad, "e1", parent_id="c1", ts=3.0),
            ],
            ["tool_call", "error"],
            [],
        ),
    }


def _summary(body: bytes) -> tuple[str, list[str], list[str]]:
    """Return the session ID, root span event names and child span names of a POST."""
    resource_spans = json.loads(body)["resourceSpans"]
    attributes = {
        attribute["key"]: attribute["value"]["stringValue"]
        for attribute in resource_spans[0]["resource"]["attributes"]
    }
    root, *children = resource_spans[0]["scopeSpans"][0]["spans"]
    return (
        attributes["agent.session_id"],
        [event["name"] for event in root.get("events", [])],
        [span["name"] for span in children],
    )


class _RecordingReceiver(BaseHTTPRequestHandler):
    """Record the path and body of each POST."""

    received: ClassVar[list[tuple[str, bytes]]] = []

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _RecordingReceiver.received.append((self.path, body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


class _ReceiverTestCase(unittest.TestCase):
    """Run a real local HTTP receiver for the streamer to POST to."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _RecordingReceiver)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/collector?{_QUERY}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _RecordingReceiver.received.clear()

    def _stream_config(self, *, otlp: bool = True, batch_size: int = 100) -> StreamConfig:
        return StreamConfig(
            url=self.url, otlp=otlp, batch_size=batch_size, flush_interval=60.0,
        )

    def _stream(self, events: list[TraceEvent], *, otlp: bool = True,
                batch_size: int = 100) -> tuple[list[tuple[str, bytes]], str]:
        """Stream *events* and return the POSTs received and what stderr got."""
        _RecordingReceiver.received.clear()
        stderr = StringIO()
        with redirect_stderr(stderr):
            streamer = EventStreamer(self._stream_config(otlp=otlp, batch_size=batch_size))
            for event in events:
                streamer.enqueue(event)
            streamer.stop()
        self.assertFalse(streamer._thread.is_alive())
        return list(_RecordingReceiver.received), stderr.getvalue()

    def _sent(self, posts: list[tuple[str, bytes]]) -> list[tuple[str, list[str], list[str]]]:
        for path, _ in posts:
            self.assertTrue(path.endswith("/v1/traces"), path)
        return [_summary(body) for _, body in posts]

    def _assert_quotes_nothing(self, stderr: str) -> None:
        """Stderr never shows data values, event IDs or the URL query string."""
        for marker in ("SECRET", "4242", "True", "true", "token"):
            self.assertNotIn(marker, stderr)


class TestOtlpStreamNonObjectData(_ReceiverTestCase):
    """--stream-otlp leaves out each event whose non-object data the converter
    reads, prints one line for it, and still sends the other events."""

    def test_each_read_event_is_left_out_and_the_rest_of_the_batch_is_sent(self) -> None:
        for data, phrase in _NON_OBJECT_DATA:
            for label, (events, left_out, children) in _read_cases(data).items():
                with self.subTest(json_type=phrase, case=label):
                    posts, stderr = self._stream(_wrapped(events))
                    self.assertEqual(
                        stderr, "".join(_left_out(event_type, phrase) for event_type in left_out),
                    )
                    self.assertEqual(
                        self._sent(posts), [(_SESSION, [_USER, _ASSISTANT], children)],
                    )
                    self._assert_quotes_nothing(stderr)

    def test_events_the_converter_does_not_read_are_sent_without_a_line(self) -> None:
        for data, phrase in _NON_OBJECT_DATA:
            cases = {
                "tool_call that an error pairs with": (
                    [
                        _event("tool_call", data, "c1", ts=2.0),
                        _event("error", {"error": "boom"}, "e1", parent_id="c1", ts=3.0),
                    ],
                    ["gen_ai.tool.call/error"],
                ),
                **{
                    event_type: ([_event(event_type, data, "u1", ts=2.0)], [])
                    for event_type in _UNREAD_TYPES
                },
            }
            for label, (events, children) in cases.items():
                with self.subTest(json_type=phrase, case=label):
                    posts, stderr = self._stream(_wrapped(events))
                    self.assertEqual(stderr, "")
                    self.assertEqual(
                        self._sent(posts), [(_SESSION, [_USER, _ASSISTANT], children)],
                    )

    def test_each_session_of_a_batch_is_sent_with_the_events_left(self) -> None:
        batch = [
            _event("user_prompt", {"prompt": "hi"}, "a1", session_id="session-a"),
            _event("user_prompt", {"prompt": "hi"}, "b1", session_id="session-b"),
            _event("user_prompt", "SECRET-STRING", "c1", session_id="session-c"),
            _event("tool_call", {"tool_name": "Bash"}, "b2", session_id="session-b", ts=2.0),
            _event("tool_result", ["SECRET-ARRAY"], "b3", parent_id="b2",
                   session_id="session-b", ts=3.0),
            _event("assistant_response", {"text": "done"}, "a2", session_id="session-a", ts=4.0),
            _event("error", None, "c2", session_id="session-c", ts=5.0),
        ]
        posts, stderr = self._stream(batch)
        self.assertEqual(self._sent(posts), [
            ("session-a", [_USER, _ASSISTANT], []),
            ("session-b", [_USER], [_BASH_SPAN]),
        ])
        self.assertEqual(
            stderr,
            _left_out("tool_result", "an array")
            + _left_out("user_prompt", "a string")
            + _left_out("error", "null"),
        )
        self._assert_quotes_nothing(stderr)

    def test_session_with_no_event_left_sends_nothing(self) -> None:
        posts, stderr = self._stream([
            _event("tool_call", 4242, "c1"),
            _event("tool_result", "SECRET-STRING", "c2", parent_id="c1", ts=2.0),
        ])
        self.assertEqual(posts, [])
        self.assertEqual(
            stderr, _left_out("tool_call", "a number") + _left_out("tool_result", "a string"),
        )

    def test_wrong_typed_fields_inside_object_data_are_sent_without_a_line(self) -> None:
        posts, stderr = self._stream(_wrapped([
            _event("tool_call", {"tool_name": 5, "arguments": "x"}, "c1", ts=2.0),
            _event("tool_result", {"tool_name": 5, "result": {}}, "c2", parent_id="c1", ts=3.0),
            _event("llm_request", {"model": [1]}, "r1", ts=4.0),
            _event("llm_response", {"model": [1]}, "r2", parent_id="r1", ts=5.0),
        ]))
        self.assertEqual(stderr, "")
        self.assertEqual(
            self._sent(posts),
            [(_SESSION, [_USER, _ASSISTANT], ["gen_ai.tool.call/5", _LLM_SPAN])],
        )

    def test_ndjson_stream_sends_non_object_data_as_stored(self) -> None:
        events = [
            _event(event_type, data, f"n{index}")
            for index, (data, _) in enumerate(_NON_OBJECT_DATA)
            for event_type in ("user_prompt", "tool_result", "error")
        ]
        posts, stderr = self._stream(events, otlp=False)
        self.assertEqual(stderr, "")
        self.assertEqual([path for path, _ in posts], [f"/collector?{_QUERY}"])
        self.assertEqual(
            posts[0][1].decode().splitlines(), [event.to_json() for event in events],
        )


class TestOtlpStreamConversionFailure(_ReceiverTestCase):
    """A conversion error is reported with a count, and later batches are sent."""

    def test_failed_batch_is_counted_and_the_next_batch_is_sent(self) -> None:
        real = session_to_otlp_genai
        converted: list[int] = []

        def fail_once(meta, events, **kwargs):
            converted.append(len(events))
            if len(converted) == 1:
                raise KeyError("SECRET-MESSAGE")
            return real(meta, events, **kwargs)

        first = [
            _event("user_prompt", {"prompt": "SECRET-PROMPT"}, f"p{index}", ts=float(index))
            for index in range(4)
        ]
        second = [_event("assistant_response", {"text": "done"}, "a1", ts=9.0)]
        with patch("agent_trace.otlp.session_to_otlp_genai", new=fail_once):
            posts, stderr = self._stream(first + second, batch_size=4)
        self.assertEqual(converted, [4, 1])
        self.assertEqual(
            stderr,
            "[stream] Could not convert 4 events to OTLP spans (KeyError); "
            "they were not sent.\n",
        )
        self.assertEqual(self._sent(posts), [(_SESSION, [_ASSISTANT], [])])
        self._assert_quotes_nothing(stderr)

    def test_count_leaves_out_the_events_already_reported(self) -> None:
        with patch(
            "agent_trace.otlp.session_to_otlp_genai",
            side_effect=ValueError("SECRET-MESSAGE"),
        ):
            posts, stderr = self._stream([
                _event("user_prompt", {"prompt": "hi"}, "p1"),
                _event("tool_result", ["SECRET-ARRAY"], "r1", ts=2.0),
            ])
        self.assertEqual(posts, [])
        self.assertEqual(
            stderr,
            _left_out("tool_result", "an array")
            + "[stream] Could not convert 1 event to OTLP spans (ValueError); "
            "it was not sent.\n",
        )
        self._assert_quotes_nothing(stderr)

    def test_events_that_cannot_be_paired_are_counted_and_streaming_goes_on(self) -> None:
        unhashable_id = TraceEvent(
            event_type=EventType.TOOL_CALL,
            event_id=["SECRET-ID"],  # type: ignore[arg-type]
            session_id=_SESSION,
            data={"tool_name": "Bash"},
            tenant_id="",
        )
        posts, stderr = self._stream(
            [
                _event("user_prompt", {"prompt": "hi"}, "p1"),
                unhashable_id,
                _event("assistant_response", {"text": "done"}, "a1", ts=9.0),
            ],
            batch_size=2,
        )
        self.assertEqual(
            stderr,
            "[stream] Could not convert 2 events to OTLP spans (TypeError); "
            "they were not sent.\n",
        )
        self.assertEqual(self._sent(posts), [(_SESSION, [_ASSISTANT], [])])
        self._assert_quotes_nothing(stderr)


class TestWatchSessionOtlpStreamNonObjectData(_ReceiverTestCase):
    """watch passes a skipped event to the streamer, which leaves it out."""

    def test_watch_streams_the_events_around_a_skipped_event(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = TraceStore(tmp.name, redact=False)
        store.create_session(SessionMeta(session_id="live-session", started_at=1.0))
        events = [
            _event("user_prompt", {"prompt": "hi"}, "p1", session_id="live-session"),
            _event("tool_call", {"tool_name": "Bash", "arguments": {"command": "ls"}}, "c1",
                   session_id="live-session", ts=2.0),
            _event("tool_result", "SECRET-STRING", "c2", parent_id="c1",
                   session_id="live-session", ts=3.0),
            _event("session_end", {"exit_code": 0}, "e1", session_id="live-session", ts=4.0),
        ]
        out, stderr = StringIO(), StringIO()
        with patch("agent_trace.watch._tail_events", return_value=iter(events)), \
                redirect_stderr(stderr):
            watch_session(
                store, "live-session", WatcherConfig(loop_threshold=99), out=out,
                stream_config=self._stream_config(),
            )
        self.assertIn(
            "[watch] Skipped checks for one tool_result event because its data is a "
            "string, not an object.\n",
            out.getvalue(),
        )
        self.assertEqual(stderr.getvalue(), _left_out("tool_result", "a string"))
        self.assertEqual(
            self._sent(list(_RecordingReceiver.received)),
            [("live-session", [_USER], [_BASH_SPAN])],
        )


if __name__ == "__main__":
    unittest.main()
