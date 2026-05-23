"""Tests for push-based event streaming (Issue #103)."""

from __future__ import annotations

import json
import queue
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

from agent_trace.models import EventType, TraceEvent
from agent_trace.watch import EventStreamer, StreamConfig


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


if __name__ == "__main__":
    unittest.main()
