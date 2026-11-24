"""Tests for Temporal W3C parenting and OTLP export (issue #209)."""

import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry

try:
    from test_otlp import (
        LAST_FLOAT,
        LAST_NANOS,
        PAST_FLOAT,
        PAST_LIMIT,
        UINT64_MAX,
        YEAR_3000,
        all_spans,
        assert_unix_nanos_in_range,
        library_meta,
        near_limit_session,
        span_times,
        tool_events,
    )
    from test_unusable_event_timestamps import BAD, ENDPOINT, TimestampCliTestCase, set_meta
except ImportError:  # pragma: no cover - run as a package module
    from tests.test_otlp import (
        LAST_FLOAT,
        LAST_NANOS,
        PAST_FLOAT,
        PAST_LIMIT,
        UINT64_MAX,
        YEAR_3000,
        all_spans,
        assert_unix_nanos_in_range,
        library_meta,
        near_limit_session,
        span_times,
        tool_events,
    )
    from tests.test_unusable_event_timestamps import (
        BAD,
        ENDPOINT,
        TimestampCliTestCase,
        set_meta,
    )

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

    def test_watch_shows_an_invalid_context_as_one_inert_line(self):
        end = TraceEvent(
            event_type=EventType.SESSION_END,
            session_id=self.meta.session_id,
        )
        output = io.StringIO()
        with patch(
            "agent_trace.temporal.attach_temporal_trace_context",
            side_effect=ValueError("bad \x1b[2K\nforged"),
        ), patch("agent_trace.watch._tail_events", return_value=iter([end])):
            watch_session(self.store, self.meta.session_id, WatcherConfig(), out=output)

        self.assertIn(
            "[watch] Invalid Temporal trace context: bad \\x1b[2K\\nforged\n",
            output.getvalue(),
        )


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
        self.assertTrue(any(s["name"].startswith("chat ") for s in spans))
        self.assertTrue(any(s["name"] == "execute_tool read" for s in spans))
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
            opener = MagicMock()
            opener.open.return_value = response
            with patch("agent_trace.otlp.no_redirect_opener", return_value=opener):
                ok = export_temporal_otlp(
                    store,
                    meta.session_id,
                    "http://tempo:4318/",
                    headers={"Authorization": "Bearer test"},
                )

        self.assertTrue(ok)
        request = opener.open.call_args.args[0]
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


TEMPORAL_CONTEXT = {"trace_id": TRACE_ID, "parent_span_id": PARENT_SPAN_ID, "trace_flags": "01"}


class TestTemporalOtlpTimeLimit(unittest.TestCase):
    """Every time the Temporal export writes is from 0 to 2**64 - 1, the
    range OTLP carries."""

    def test_duration_past_the_limit_reads_as_missing(self):
        meta = library_meta(**TEMPORAL_CONTEXT)
        payload = session_to_temporal_otlp(meta, tool_events(1e14))
        self.assertEqual(payload, session_to_temporal_otlp(meta, tool_events()))
        _, tool, error = all_spans(payload)
        self.assertEqual(span_times(tool), ("1001000000000", "1001001000000"))
        self.assertEqual(span_times(error), ("1003000000000", "1003001000000"))
        assert_unix_nanos_in_range(self, payload)

    def test_root_total_past_the_limit_ends_at_the_start(self):
        without_total = session_to_temporal_otlp(
            library_meta(ended_at=None, **TEMPORAL_CONTEXT), tool_events(),
        )
        for total in (1e14, 1e308):
            with self.subTest(total=total):
                meta = library_meta(ended_at=None, total_duration_ms=total, **TEMPORAL_CONTEXT)
                payload = session_to_temporal_otlp(meta, tool_events())
                self.assertEqual(payload, without_total)
                self.assertEqual(
                    span_times(all_spans(payload)[0]), ("1000000000000", "1000000000000"),
                )
                assert_unix_nanos_in_range(self, payload)

    def test_spans_near_the_limit_end_at_the_limit(self):
        meta, events = near_limit_session()
        for name, value in TEMPORAL_CONTEXT.items():
            setattr(meta, name, value)
        payload = session_to_temporal_otlp(meta, events)
        root, *children = all_spans(payload)
        self.assertEqual(span_times(root), (LAST_NANOS, LAST_NANOS))
        limit = (LAST_NANOS, str(UINT64_MAX))
        # The transcript response keeps its zero length. Each other span ends
        # at the limit.
        self.assertEqual(
            sorted(span_times(span) for span in children), [(LAST_NANOS, LAST_NANOS)] + [limit] * 5,
        )
        assert_unix_nanos_in_range(self, payload)


class TestTemporalExportTimeLimit(TimestampCliTestCase):
    """export --format temporal refuses a time OTLP cannot carry before it
    writes or sends anything."""

    ARGVS = (
        ["export", BAD, "--format", "temporal", "--output", "out.json"],
        ["export", BAD, "--format", "temporal", "--endpoint", ENDPOINT],
    )

    def test_refuses_session_times_past_the_limit(self):
        for field in ("started_at", "ended_at"):
            for value in (YEAR_3000, PAST_FLOAT):
                for argv in self.ARGVS:
                    with self.subTest(field=field, value=value, argv=argv):
                        root = self.make_ts_workspace(temporal=True)
                        set_meta(root / "work" / ".agent-traces" / BAD, **{field: value})
                        self.assert_refused(root, argv, f"Error: session {field} {PAST_LIMIT}\n")

    def test_refuses_event_timestamps_past_the_limit(self):
        expected = (
            f"Error: session event timestamp {PAST_LIMIT}: event 6 of 25 (tool_call); "
            "5 more events have timestamps past that time\n"
        )
        for value in (YEAR_3000, PAST_FLOAT):
            for argv in self.ARGVS:
                with self.subTest(value=value, argv=argv):
                    root = self.make_ts_workspace(BAD, "tool_call", value, temporal=True)
                    self.assert_refused(root, argv, expected)

    def test_accepts_the_last_time_that_fits(self):
        root = self.make_ts_workspace(
            BAD, "tool_call", LAST_FLOAT, temporal=True,
            where=lambda record: record["event_id"] == "bad-call-test2",
        )
        set_meta(root / "work" / ".agent-traces" / BAD, started_at=LAST_FLOAT, ended_at=LAST_FLOAT)
        code, out, err, network, written = self.outcome(
            root, self.ARGVS[0], files="out.json", read=json.loads,
        )
        self.assertEqual((code, out, network), (0, "", 0), err)
        payload = written["out.json"]
        root_span, *children = all_spans(payload)
        self.assertEqual(span_times(root_span), (LAST_NANOS, LAST_NANOS))
        # The call's result lasts 1 s, which would end past the limit.
        self.assertEqual(
            [span_times(span) for span in children if span["startTimeUnixNano"] == LAST_NANOS],
            [(LAST_NANOS, str(UINT64_MAX))],
        )
        assert_unix_nanos_in_range(self, payload)


if __name__ == "__main__":
    unittest.main()
