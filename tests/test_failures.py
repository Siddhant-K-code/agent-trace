"""Tests for the shared failed-model-request predicate and session error count."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from agent_trace import failures, otlp
from agent_trace.backfill import derive_meta
from agent_trace.baseline import _session_metrics as baseline_metrics
from agent_trace.cli import cmd_stats
from agent_trace.dashboard import _extract_trend_point, build_dashboard
from agent_trace.drift import compute_fingerprint
from agent_trace.eval.dataset import _session_passes_filter
from agent_trace.eval.scorers import measure_error_count
from agent_trace.failures import (
    count_session_errors,
    is_failed_response,
    llm_failure,
    response_failure,
    session_errors,
)
from agent_trace.langfuse_export import _session_metrics as langfuse_metrics
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import session_to_otlp_genai
from agent_trace.pr_comment import summarize_session
from agent_trace.sample import _score_session
from agent_trace.store import TraceStore
from agent_trace.timeline import build_timeline, format_timeline, format_timeline_json


def _event(event_type: EventType, event_id: str, parent_id: str = "", data=None) -> TraceEvent:
    return TraceEvent(
        event_type=event_type,
        event_id=event_id,
        parent_id=parent_id,
        timestamp=1000.0,
        data={} if data is None else data,
    )


def _request(event_id: str = "req") -> TraceEvent:
    return _event(EventType.LLM_REQUEST, event_id, data={"model": "gpt-4o"})


def _response(event_id: str, parent_id: str = "", **data) -> TraceEvent:
    return _event(EventType.LLM_RESPONSE, event_id, parent_id, data)


def _error(event_id: str, parent_id: str = "", message: str = "boom") -> TraceEvent:
    return _event(EventType.ERROR, event_id, parent_id, {"error": message})


def _codex(event_id: str, status: str, error=None) -> TraceEvent:
    """An llm_response as the Codex PostModelRequest hook records it."""
    data = {"hook_event": "post_model_request", "provider": "codex", "status": status}
    if error is not None:
        data["error"] = error
    if status == "completed":
        data["token_usage"] = {"input_tokens": 10, "output_tokens": 2}
    else:
        data["token_usage"] = None
    return _response(event_id, **data)


class TestPredicate(unittest.TestCase):
    def test_exports_use_the_shared_predicate(self):
        self.assertIs(otlp.llm_failure, failures.llm_failure)
        self.assertIs(otlp.LLMFailure, failures.LLMFailure)
        self.assertEqual(otlp.ERROR_TYPE_CANCELLED, "cancelled")
        self.assertEqual(otlp.ERROR_TYPE_OTHER, "_OTHER")

    def test_failed_and_cancelled_responses(self):
        failed = _codex("a", "failed", "stream disconnected")
        cancelled = _codex("b", "cancelled")
        completed = _codex("c", "completed")
        self.assertTrue(is_failed_response(failed))
        self.assertFalse(is_failed_response(cancelled))
        self.assertTrue(response_failure(cancelled).cancelled)
        self.assertFalse(is_failed_response(completed))
        self.assertIsNone(response_failure(completed))

    def test_status_spellings(self):
        self.assertTrue(is_failed_response(_response("a", status=" FAILED ")))
        self.assertTrue(response_failure(_response("b", status="Canceled")).cancelled)
        self.assertTrue(is_failed_response(_response("c", error="upstream 500")))
        self.assertFalse(is_failed_response(_response("d", status="completed", error="")))

    def test_only_responses_are_failed_responses(self):
        error = _error("e")
        self.assertIsNotNone(llm_failure(error))
        self.assertIsNone(response_failure(error))
        self.assertFalse(is_failed_response(error))

    def test_non_object_data_is_not_a_failure(self):
        event = _event(EventType.LLM_RESPONSE, "a", data=["failed"])
        self.assertIsNone(response_failure(event))
        self.assertEqual(count_session_errors([event]), 0)


class TestSessionErrors(unittest.TestCase):
    def test_no_events(self):
        self.assertEqual(session_errors([]), failures.SessionErrors())
        self.assertEqual(count_session_errors([]), 0)

    def test_error_events_count_as_before(self):
        events = [_error("e1"), _request(), _response("r", "req"), _error("e2")]
        result = session_errors(events)
        self.assertEqual((result.error_events, result.failed_requests), (2, 0))
        self.assertEqual(result.errors, 2)

    def test_codex_requests(self):
        events = [
            _codex("a", "completed"),
            _codex("b", "failed", "stream disconnected"),
            _codex("c", "cancelled"),
            _codex("d", "failed"),
        ]
        result = session_errors(events)
        self.assertEqual(result.errors, 2)
        self.assertEqual(result.failed_requests, 2)
        self.assertEqual(result.cancelled_requests, 1)
        self.assertEqual(result.failed_responses, frozenset({1, 3}))
        self.assertEqual(result.cancelled_responses, frozenset({2}))

    def test_linked_error_before_failed_response_counts_once(self):
        events = [_request(), _error("e", "req"), _response("r", "req", status="failed")]
        result = session_errors(events)
        self.assertEqual((result.error_events, result.failed_requests, result.errors), (1, 0, 1))

    def test_linked_error_after_failed_response_counts_once(self):
        events = [_request(), _response("r", "req", status="failed"), _error("e", "req")]
        result = session_errors(events)
        self.assertEqual((result.error_events, result.failed_requests, result.errors), (1, 0, 1))

    def test_duplicate_failed_responses_count_once(self):
        events = [
            _request(),
            _response("r1", "req", status="failed"),
            _response("r2", "req", error="again"),
        ]
        result = session_errors(events)
        self.assertEqual(result.errors, 1)
        self.assertEqual(result.failed_responses, frozenset({1}))

    def test_responses_pair_without_parent_ids(self):
        events = [_request(), _response("r", status="failed"), _error("e", "req")]
        self.assertEqual(count_session_errors(events), 1)

    def test_unlinked_error_is_a_separate_error(self):
        events = [_request(), _error("e"), _response("r", "req", status="failed")]
        self.assertEqual(count_session_errors(events), 2)

    def test_cancelled_requests_are_not_errors(self):
        events = [_request(), _response("r", "req", status="cancelled")]
        result = session_errors(events)
        self.assertEqual((result.errors, result.cancelled_requests), (0, 1))

    def test_failed_wins_over_cancelled_for_one_request(self):
        events = [
            _request(),
            _response("r1", "req", status="cancelled"),
            _response("r2", "req", status="failed"),
        ]
        result = session_errors(events)
        self.assertEqual((result.errors, result.cancelled_requests), (1, 0))

    def test_cancelled_request_with_linked_error_is_one_error(self):
        events = [_request(), _response("r", "req", status="cancelled"), _error("e", "req")]
        result = session_errors(events)
        self.assertEqual((result.errors, result.cancelled_requests), (1, 0))

    def test_accepts_any_iterable(self):
        events = [_codex("a", "failed"), _error("e")]
        self.assertEqual(count_session_errors(iter(events)), 2)


class TestExportsAgree(unittest.TestCase):
    """The GenAI export marks failed exactly the requests the count includes."""

    def _llm_spans(self, events):
        meta = SessionMeta(session_id="s1", agent_name="codex", started_at=1000.0, ended_at=1100.0)
        payload = session_to_otlp_genai(meta, events)
        spans = [
            span
            for resource in payload["resourceSpans"]
            for scope in resource["scopeSpans"]
            for span in scope["spans"]
        ]
        return [span for span in spans if span["name"] == "gen_ai.client.operation"]

    @staticmethod
    def _error_type(span):
        for attribute in span.get("attributes", []):
            if attribute["key"] == "error.type":
                return attribute["value"]["stringValue"]
        return None

    def test_codex_session(self):
        events = [
            _codex("a", "completed"),
            _codex("b", "failed", "stream disconnected"),
            _codex("c", "cancelled"),
            _codex("d", "failed"),
        ]
        result = session_errors(events)
        error_types = [
            self._error_type(span)
            for span in self._llm_spans(events)
            if span["status"].get("code") == 2
        ]
        self.assertEqual(
            sum(error_type != "cancelled" for error_type in error_types),
            result.failed_requests,
        )
        self.assertEqual(error_types.count("cancelled"), result.cancelled_requests)

    def test_sdk_error_linked_to_request(self):
        events = [_request(), _error("e", "req")]
        failed = [span for span in self._llm_spans(events) if span["status"].get("code") == 2]
        self.assertEqual(len(failed), 1)
        self.assertEqual(count_session_errors(events), 1)


class TestSessionErrorReaders(unittest.TestCase):
    """Every session error count and error rate uses the shared count.

    The session has two tool calls, a failed tool call (an error event), a
    failed Codex request, a cancelled Codex request, and an SDK request with
    both a linked error and a failed response: 3 errors. Its stored count,
    2, was written before failed model requests were counted.
    """

    ERRORS = 3
    ERROR_RATE = 3 / 2

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name)
        self.meta = SessionMeta(
            session_id="s1", agent_name="codex", started_at=1000.0, ended_at=1100.0,
            tool_calls=2, llm_requests=1, errors=2, total_duration_ms=100_000,
        )
        self.store.create_session(self.meta)
        ok_call = _event(EventType.TOOL_CALL, "c1", data={"tool_name": "Read"})
        failed_call = _event(EventType.TOOL_CALL, "c2", data={"tool_name": "Bash"})
        self.events = [
            _event(EventType.SESSION_START, "start", data={"provider": "codex"}),
            ok_call,
            _event(EventType.TOOL_RESULT, "t1", "c1", {"tool_name": "Read", "result": "ok"}),
            failed_call,
            _error("e1", "c2", "exit 1"),
            _codex("m1", "failed", "stream disconnected"),
            _codex("m2", "cancelled"),
            _request("req"),
            _response("m3", "req", status="failed", error="rate limited"),
            _error("e2", "req", "rate limited"),
            _event(EventType.SESSION_END, "end"),
        ]
        for index, event in enumerate(self.events):
            event.session_id = "s1"
            event.timestamp = 1000.0 + index
            self.store.append_event("s1", event)

    def test_shared_count(self):
        result = session_errors(self.events)
        self.assertEqual(
            (result.error_events, result.failed_requests, result.cancelled_requests),
            (2, 1, 1),
        )
        self.assertEqual(measure_error_count(self.events), self.ERRORS)
        self.assertTrue(_session_passes_filter(self.store, "s1", "has-errors"))
        self.assertEqual(derive_meta("s1", self.events)[0].errors, self.ERRORS)

    def test_error_rates(self):
        self.assertEqual(baseline_metrics(self.store, "s1")["error_rate"], self.ERROR_RATE)
        self.assertEqual(_extract_trend_point(self.store, self.meta).error_rate, self.ERROR_RATE)
        self.assertEqual(compute_fingerprint(self.store, ["s1"]).error_rate.mean, self.ERROR_RATE)
        self.assertEqual(_score_session("s1", self.meta, self.events).error_rate, self.ERROR_RATE)
        self.assertEqual(
            langfuse_metrics(self.store, self.meta)["agent_strace.session.error_rate"],
            self.ERROR_RATE,
        )

    def test_session_summaries_use_the_larger_count(self):
        [summary] = build_dashboard(self.store).summaries
        self.assertEqual(summary.errors, self.ERRORS)
        self.assertFalse(summary.succeeded)
        self.assertEqual(summarize_session(self.store, "s1").errors, self.ERRORS)

    def test_timeline(self):
        timeline = build_timeline(self.store, "s1")
        self.assertEqual(timeline.error_count, self.ERRORS)
        statuses = [
            (entry.status, entry.label)
            for phase in timeline.phases
            for entry in phase.entries
            if entry.label.startswith("LLM response")
        ]
        self.assertEqual(statuses, [
            ("fail", "LLM response failed"),
            ("cancelled", "LLM response cancelled"),
            ("fail", "LLM response failed"),
        ])
        out = io.StringIO()
        format_timeline(timeline, out)
        self.assertIn("\u2298 LLM response cancelled", out.getvalue())
        self.assertIn(f"| {self.ERRORS} errors", out.getvalue())
        out = io.StringIO()
        format_timeline_json(timeline, out)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["error_count"], self.ERRORS)

    def test_stats_lists_failed_requests(self):
        out = io.StringIO()
        args = argparse.Namespace(
            trace_dir=self.store.base_dir, session_id="s1", include_subagents=False
        )
        with patch("sys.stdout", out):
            self.assertEqual(cmd_stats(args), 0)
        output = out.getvalue()
        self.assertIn(f"Errors ({self.ERRORS}):", output)
        self.assertIn("model request failed: stream disconnected", output)
        self.assertNotIn("model request failed: rate limited", output)


if __name__ == "__main__":
    unittest.main()
