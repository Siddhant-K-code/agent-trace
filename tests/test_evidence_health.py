import unittest

from agent_trace.evidence_health import (
    EVIDENCE_HEALTH_SCHEMA_VERSION,
    EvidenceHealthReasonKind,
    EvidenceHealthStatus,
    assess_evidence_health,
)
from agent_trace.models import EventType, TraceEvent


SESSION_ID = "review-fixture"


def event(
    event_type: EventType,
    timestamp: float,
    event_id: str,
    *,
    parent_id: str = "",
    session_id: str = SESSION_ID,
) -> TraceEvent:
    return TraceEvent(
        event_type=event_type,
        timestamp=timestamp,
        event_id=event_id,
        session_id=session_id,
        parent_id=parent_id,
    )


class TestEvidenceHealth(unittest.TestCase):
    def test_complete_paired_session_is_healthy(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.LLM_REQUEST, 2.0, "llm-request"),
            event(EventType.LLM_RESPONSE, 3.0, "llm-response", parent_id="llm-request"),
            event(EventType.TOOL_CALL, 4.0, "tool-call"),
            event(EventType.TOOL_RESULT, 5.0, "tool-result", parent_id="tool-call"),
            event(EventType.SESSION_END, 6.0, "end"),
        ]

        result = assess_evidence_health(
            events,
            provider="codex",
            capture_method="hooks",
            session_finalized=True,
        )

        self.assertEqual(result.status, EvidenceHealthStatus.HEALTHY)
        self.assertEqual(result.reasons, ())
        self.assertTrue(result.observed_start)
        self.assertTrue(result.observed_end)
        self.assertEqual(result.event_count, 6)
        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.capture_method, "hooks")

    def test_failed_tool_call_error_is_a_terminal_outcome(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.TOOL_CALL, 2.0, "tool-call"),
            event(EventType.ERROR, 3.0, "tool-error", parent_id="tool-call"),
            event(EventType.SESSION_END, 4.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)

        self.assertEqual(result.status, EvidenceHealthStatus.HEALTHY)
        self.assertNotIn("unpaired_tool_call", [reason.code for reason in result.reasons])

    def test_out_of_order_tool_result_is_partial(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.TOOL_RESULT, 2.0, "tool-result", parent_id="tool-call"),
            event(EventType.TOOL_CALL, 3.0, "tool-call"),
            event(EventType.SESSION_END, 4.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)
        codes = [reason.code for reason in result.reasons]

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertIn("out_of_order_tool_outcome", codes)
        self.assertNotIn("unpaired_tool_call", codes)

    def test_duplicate_tool_terminal_outcomes_are_partial(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.TOOL_CALL, 2.0, "tool-call"),
            event(EventType.TOOL_RESULT, 3.0, "tool-result", parent_id="tool-call"),
            event(EventType.ERROR, 4.0, "tool-error", parent_id="tool-call"),
            event(EventType.SESSION_END, 5.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)
        codes = [reason.code for reason in result.reasons]

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertIn("duplicate_tool_outcome", codes)
        self.assertNotIn("unpaired_tool_call", codes)

    def test_duplicate_llm_responses_are_partial(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.LLM_REQUEST, 2.0, "llm-request"),
            event(EventType.LLM_RESPONSE, 3.0, "llm-response-1", parent_id="llm-request"),
            event(EventType.LLM_RESPONSE, 4.0, "llm-response-2", parent_id="llm-request"),
            event(EventType.SESSION_END, 5.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertIn("duplicate_llm_outcome", [reason.code for reason in result.reasons])

    def test_empty_evidence_is_unknown(self):
        result = assess_evidence_health([])

        self.assertEqual(result.status, EvidenceHealthStatus.UNKNOWN)
        self.assertEqual([reason.code for reason in result.reasons], ["no_events"])

    def test_empty_evidence_keeps_observable_failures_and_declared_limits(self):
        result = assess_evidence_health(
            [],
            provider_blind_spots=["tool execution is not exposed by this capture adapter"],
            export_failures=["collector rejected batch 1"],
        )

        self.assertEqual(result.status, EvidenceHealthStatus.UNKNOWN)
        self.assertEqual(
            [reason.code for reason in result.reasons],
            ["no_events", "export_failure", "provider_blind_spot"],
        )
        self.assertEqual(
            result.reasons[-1].kind,
            EvidenceHealthReasonKind.PROVIDER_LIMITATION,
        )

    def test_active_session_without_end_is_unknown_not_partial(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.USER_PROMPT, 2.0, "prompt"),
        ]

        result = assess_evidence_health(events, session_finalized=False)

        self.assertEqual(result.status, EvidenceHealthStatus.UNKNOWN)
        self.assertEqual([reason.code for reason in result.reasons], ["session_active"])

    def test_finalized_session_without_end_is_partial(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.USER_PROMPT, 2.0, "prompt"),
        ]

        result = assess_evidence_health(events, session_finalized=True)

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertIn("missing_session_end", [reason.code for reason in result.reasons])

    def test_unpaired_tool_relationships_are_partial(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.TOOL_CALL, 2.0, "call-without-result"),
            event(EventType.TOOL_RESULT, 3.0, "orphan-result", parent_id="unknown-call"),
            event(EventType.SESSION_END, 4.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)
        codes = {reason.code for reason in result.reasons}

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertEqual(codes, {"unpaired_tool_call", "orphan_tool_result"})

    def test_timestamp_regression_is_partial(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.USER_PROMPT, 3.0, "prompt"),
            event(EventType.ASSISTANT_RESPONSE, 2.0, "response"),
            event(EventType.SESSION_END, 4.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertIn("timestamp_regression", [reason.code for reason in result.reasons])

    def test_provider_blind_spots_are_declared_limitations(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.SESSION_END, 2.0, "end"),
        ]

        result = assess_evidence_health(
            events,
            provider="example-provider",
            provider_blind_spots=["file reads are not exposed by this capture adapter"],
            session_finalized=True,
        )

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertEqual(len(result.reasons), 1)
        self.assertEqual(result.reasons[0].code, "provider_blind_spot")
        self.assertEqual(
            result.reasons[0].kind,
            EvidenceHealthReasonKind.PROVIDER_LIMITATION,
        )

    def test_mixed_session_ids_are_invalid(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start", session_id="one"),
            event(EventType.USER_PROMPT, 2.0, "prompt", session_id="two"),
            event(EventType.SESSION_END, 3.0, "end", session_id="one"),
        ]

        result = assess_evidence_health(events, session_finalized=True)

        self.assertEqual(result.status, EvidenceHealthStatus.INVALID)
        self.assertIn("mixed_session_ids", [reason.code for reason in result.reasons])

    def test_duplicate_boundaries_and_event_ids_are_invalid(self):
        events = [
            event(EventType.SESSION_START, 1.0, "same"),
            event(EventType.SESSION_START, 2.0, "same"),
            event(EventType.SESSION_END, 3.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)
        codes = {reason.code for reason in result.reasons}

        self.assertEqual(result.status, EvidenceHealthStatus.INVALID)
        self.assertIn("duplicate_session_start", codes)
        self.assertIn("duplicate_event_id", codes)

    def test_export_failures_are_observed_partial_evidence(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.SESSION_END, 2.0, "end"),
        ]

        result = assess_evidence_health(
            events,
            export_failures=["collector rejected batch 4"],
            session_finalized=True,
        )

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertEqual(result.reasons[0].code, "export_failure")
        self.assertEqual(result.reasons[0].kind, EvidenceHealthReasonKind.OBSERVED)

    def test_result_serializes_with_versioned_machine_readable_fields(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.SESSION_END, 2.0, "end"),
        ]

        payload = assess_evidence_health(events, session_finalized=True).to_dict()

        self.assertEqual(payload["schema_version"], EVIDENCE_HEALTH_SCHEMA_VERSION)
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["reasons"], ())


if __name__ == "__main__":
    unittest.main()
