"""Tests for Temporal W3C parenting and OTLP export (issue #209)."""

import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agent_trace.cli import build_parser, cmd_export
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.temporal import (
    TEMPORAL_TRACE_PARENT_ENV,
    TEMPORAL_TRACE_PARENT_FALLBACK_ENV,
    attach_temporal_trace_context,
    export_temporal_otlp,
    get_temporal_trace_context,
    parse_temporal_traceparent,
    session_to_temporal_otlp,
)
from agent_trace.watch import WatcherConfig, watch_session


TRACE_ID = "a" * 32
PARENT_SPAN_ID = "b" * 16
TRACEPARENT = f"00-{TRACE_ID}-{PARENT_SPAN_ID}-01"


def _spans(payload):
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


class TestTemporalTraceContext(unittest.TestCase):
    def test_reads_agent_strace_environment_name(self):
        context = get_temporal_trace_context({
            TEMPORAL_TRACE_PARENT_ENV: TRACEPARENT,
        })

        self.assertIsNotNone(context)
        self.assertEqual(context.trace_id, TRACE_ID)
        self.assertEqual(context.parent_span_id, PARENT_SPAN_ID)
        self.assertEqual(context.trace_flags, "01")

    def test_supports_temporal_fallback_environment_name(self):
        context = get_temporal_trace_context({
            TEMPORAL_TRACE_PARENT_FALLBACK_ENV: TRACEPARENT,
        })

        self.assertIsNotNone(context)
        self.assertEqual(context.source_env, TEMPORAL_TRACE_PARENT_FALLBACK_ENV)

    def test_agent_strace_name_takes_precedence(self):
        other = f"00-{'c' * 32}-{'d' * 16}-00"
        context = get_temporal_trace_context({
            TEMPORAL_TRACE_PARENT_ENV: TRACEPARENT,
            TEMPORAL_TRACE_PARENT_FALLBACK_ENV: other,
        })

        self.assertEqual(context.trace_id, TRACE_ID)

    def test_returns_none_without_temporal_environment(self):
        self.assertIsNone(get_temporal_trace_context({}))

    def test_rejects_malformed_traceparent(self):
        with self.assertRaisesRegex(ValueError, "valid W3C traceparent"):
            parse_temporal_traceparent("not-a-traceparent")

    def test_rejects_zero_trace_or_span_ids(self):
        with self.assertRaisesRegex(ValueError, "invalid trace ID"):
            parse_temporal_traceparent(f"00-{'0' * 32}-{PARENT_SPAN_ID}-01")
        with self.assertRaisesRegex(ValueError, "invalid parent span ID"):
            parse_temporal_traceparent(f"00-{TRACE_ID}-{'0' * 16}-01")


class TestTemporalSessionParenting(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.temp_dir.name)
        self.meta = SessionMeta(agent_name="temporal-agent")
        self.store.create_session(self.meta)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_attach_persists_context_in_session_metadata(self):
        context = attach_temporal_trace_context(
            self.store,
            self.meta.session_id,
            {TEMPORAL_TRACE_PARENT_ENV: TRACEPARENT},
        )

        restored = self.store.load_meta(self.meta.session_id)
        self.assertEqual(context.parent_span_id, PARENT_SPAN_ID)
        self.assertEqual(restored.trace_id, TRACE_ID)
        self.assertEqual(restored.parent_span_id, PARENT_SPAN_ID)
        self.assertEqual(restored.trace_flags, "01")

        roundtrip = SessionMeta.from_json(restored.to_json())
        self.assertEqual(roundtrip.parent_span_id, PARENT_SPAN_ID)

    def test_watch_attaches_context_before_monitoring(self):
        end = TraceEvent(
            event_type=EventType.SESSION_END,
            session_id=self.meta.session_id,
        )
        output = io.StringIO()
        with patch.dict(os.environ, {TEMPORAL_TRACE_PARENT_ENV: TRACEPARENT}), \
             patch("agent_trace.watch._tail_events", return_value=iter([end])):
            watch_session(
                self.store,
                self.meta.session_id,
                WatcherConfig(),
                out=output,
            )

        restored = self.store.load_meta(self.meta.session_id)
        self.assertEqual(restored.trace_id, TRACE_ID)
        self.assertEqual(restored.parent_span_id, PARENT_SPAN_ID)
        self.assertIn("Attached Temporal parent span", output.getvalue())


class TestTemporalOtlp(unittest.TestCase):
    def _meta_and_events(self):
        meta = SessionMeta(
            agent_name="temporal-agent",
            trace_id=TRACE_ID,
            parent_span_id=PARENT_SPAN_ID,
            trace_flags="01",
        )
        request = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            timestamp=10.0,
            data={"model": "gpt-4o", "input_tokens": 12},
        )
        response = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            timestamp=11.0,
            parent_id=request.event_id,
            data={"model": "gpt-4o", "output_tokens": 7},
        )
        call = TraceEvent(
            event_type=EventType.TOOL_CALL,
            timestamp=12.0,
            data={"tool_name": "read", "arguments": {"path": "README.md"}},
        )
        result = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            timestamp=13.0,
            parent_id=call.event_id,
            duration_ms=1000,
            data={"tool_name": "read", "result": "ok"},
        )
        return meta, [request, response, call, result]

    def test_export_uses_temporal_trace_and_parent_span(self):
        meta, events = self._meta_and_events()
        payload = session_to_temporal_otlp(meta, events)
        spans = _spans(payload)
        root = spans[0]

        self.assertEqual(root["traceId"], TRACE_ID)
        self.assertEqual(root["parentSpanId"], PARENT_SPAN_ID)
        self.assertEqual(root["flags"], 1)
        self.assertTrue(any(s["name"] == "gen_ai.client.operation" for s in spans))
        self.assertTrue(any(s["name"] == "gen_ai.tool.call/read" for s in spans))
        self.assertTrue(all(s["traceId"] == TRACE_ID for s in spans))
        self.assertTrue(all(s["flags"] == 1 for s in spans))
        self.assertTrue(all(
            s["parentSpanId"] == root["spanId"] for s in spans[1:]
        ))

    def test_export_requires_stored_temporal_context(self):
        with self.assertRaisesRegex(ValueError, "no Temporal trace context"):
            session_to_temporal_otlp(SessionMeta(), [])

    def test_http_export_without_context_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TraceStore(temp_dir)
            meta = SessionMeta()
            store.create_session(meta)
            store.append_event(meta.session_id, TraceEvent(
                event_type=EventType.SESSION_START,
                session_id=meta.session_id,
            ))
            with patch("agent_trace.temporal.sys.stderr", new=io.StringIO()) as err:
                ok = export_temporal_otlp(
                    store,
                    meta.session_id,
                    "http://tempo:4318",
                )

        self.assertFalse(ok)
        self.assertIn("Temporal export failed", err.getvalue())

    def test_http_export_posts_temporal_payload(self):
        meta, events = self._meta_and_events()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TraceStore(temp_dir)
            store.create_session(meta)
            for event in events:
                event.session_id = meta.session_id
                store.append_event(meta.session_id, event)

            response = MagicMock()
            response.status = 202
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            with patch("agent_trace.otlp.urllib.request.urlopen", return_value=response) as urlopen:
                ok = export_temporal_otlp(
                    store,
                    meta.session_id,
                    "http://tempo:4318/",
                    headers={"Authorization": "Bearer test"},
                )

        self.assertTrue(ok)
        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        root = _spans(body)[0]
        self.assertEqual(request.full_url, "http://tempo:4318/v1/traces")
        self.assertEqual(root["traceId"], TRACE_ID)
        self.assertEqual(root["parentSpanId"], PARENT_SPAN_ID)

    def test_cli_writes_temporal_otlp_json(self):
        meta, events = self._meta_and_events()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TraceStore(temp_dir)
            store.create_session(meta)
            for event in events:
                event.session_id = meta.session_id
                store.append_event(meta.session_id, event)
            output_path = os.path.join(temp_dir, "temporal.json")
            args = build_parser().parse_args([
                "--trace-dir", temp_dir,
                "export", meta.session_id,
                "--format", "temporal",
                "--output", output_path,
            ])

            self.assertEqual(cmd_export(args), 0)
            with open(output_path, encoding="utf-8") as exported:
                root = _spans(json.load(exported))[0]

        self.assertEqual(root["traceId"], TRACE_ID)
        self.assertEqual(root["parentSpanId"], PARENT_SPAN_ID)


if __name__ == "__main__":
    unittest.main()
