import unittest

from agent_trace.evidence_health import (
    CONTINUATION_START_SOURCES,
    EVIDENCE_HEALTH_SCHEMA_VERSION,
    EvidenceHealthReasonKind,
    EvidenceHealthStatus,
    assess_evidence_health,
    is_continuation_source,
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
    data: dict | None = None,
) -> TraceEvent:
    return TraceEvent(
        event_type=event_type,
        timestamp=timestamp,
        event_id=event_id,
        session_id=session_id,
        parent_id=parent_id,
        data=data if data is not None else {},
    )


def start(timestamp: float, event_id: str, source=None) -> TraceEvent:
    data = {} if source is None else {"source": source}
    return event(EventType.SESSION_START, timestamp, event_id, data=data)


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

    def test_failed_llm_call_error_is_a_terminal_outcome(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.LLM_REQUEST, 2.0, "llm-request"),
            event(EventType.ERROR, 3.0, "llm-error", parent_id="llm-request"),
            event(EventType.SESSION_END, 4.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)

        self.assertEqual(result.status, EvidenceHealthStatus.HEALTHY)
        self.assertNotIn("unpaired_llm_request", [reason.code for reason in result.reasons])

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

    def test_active_tool_call_without_result_is_treated_as_in_flight(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.TOOL_CALL, 2.0, "tool-call"),
        ]

        result = assess_evidence_health(events, session_finalized=False)

        self.assertEqual(result.status, EvidenceHealthStatus.UNKNOWN)
        self.assertEqual([reason.code for reason in result.reasons], ["session_active"])

    def test_active_llm_request_without_response_is_treated_as_in_flight(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.LLM_REQUEST, 2.0, "llm-request"),
        ]

        result = assess_evidence_health(events, session_finalized=False)

        self.assertEqual(result.status, EvidenceHealthStatus.UNKNOWN)
        self.assertEqual([reason.code for reason in result.reasons], ["session_active"])

    def test_active_session_still_reports_orphan_outcomes(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.TOOL_CALL, 2.0, "in-flight-tool-call"),
            event(EventType.TOOL_RESULT, 3.0, "orphan-result", parent_id="unknown-call"),
        ]

        result = assess_evidence_health(events, session_finalized=False)
        codes = {reason.code for reason in result.reasons}

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertEqual(codes, {"session_active", "orphan_tool_result"})

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

    def test_orphan_error_with_unknown_parent_is_partial(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.TOOL_CALL, 2.0, "call-without-result"),
            event(EventType.ERROR, 3.0, "orphan-error", parent_id="typo-call"),
            event(EventType.SESSION_END, 4.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)
        reasons = {reason.code: reason for reason in result.reasons}

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertIn("unpaired_tool_call", reasons)
        self.assertEqual(reasons["orphan_error"].event_id, "orphan-error")
        self.assertEqual(reasons["orphan_error"].kind, EvidenceHealthReasonKind.OBSERVED)
        self.assertIn("typo-call", reasons["orphan_error"].message)

    def test_error_without_parent_cannot_be_silently_healthy(self):
        events = [
            event(EventType.SESSION_START, 1.0, "start"),
            event(EventType.ERROR, 2.0, "unlinked-error"),
            event(EventType.SESSION_END, 3.0, "end"),
        ]

        result = assess_evidence_health(events, session_finalized=True)

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertEqual([reason.code for reason in result.reasons], ["unlinked_error"])
        self.assertEqual(result.reasons[0].event_id, "unlinked-error")

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

        payload = assess_evidence_health(
            events,
            provider="codex",
            capture_method="hooks",
            capture_matrix_version="codex-hooks-matrix-v1",
            session_finalized=True,
        ).to_dict()

        self.assertEqual(payload["capture_matrix_version"], "codex-hooks-matrix-v1")
        self.assertEqual(payload["schema_version"], EVIDENCE_HEALTH_SCHEMA_VERSION)
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["reasons"], ())


class TestSessionContinuation(unittest.TestCase):
    """resume and compact starts continue a session instead of duplicating it."""

    @staticmethod
    def assess(events, *, session_finalized=True):
        return assess_evidence_health(events, session_finalized=session_finalized)

    @staticmethod
    def located(result):
        return [(reason.code, reason.event_id) for reason in result.reasons]

    def test_continuation_sources_are_exactly_resume_and_compact(self):
        self.assertEqual(CONTINUATION_START_SOURCES, frozenset({"resume", "compact"}))
        self.assertTrue(is_continuation_source("resume"))
        self.assertTrue(is_continuation_source("compact"))
        for source in (
            None, "", "startup", "clear", "fork", "new", "SessionStart", "sessionStart",
            "Resume", "COMPACT", " resume", 1, ["resume"], {"compact": True},
        ):
            with self.subTest(source=source):
                self.assertFalse(is_continuation_source(source))

    def test_compact_start_continues_an_open_session(self):
        result = self.assess([
            start(1.0, "start", "startup"),
            event(EventType.USER_PROMPT, 2.0, "prompt"),
            start(3.0, "compact", "compact"),
            event(EventType.USER_PROMPT, 4.0, "prompt-2"),
            event(EventType.SESSION_END, 5.0, "end"),
        ])

        self.assertEqual(result.status, EvidenceHealthStatus.HEALTHY)
        self.assertEqual(result.reasons, ())

    def test_resume_start_reopens_an_ended_session(self):
        result = self.assess([
            start(1.0, "start", "startup"),
            event(EventType.USER_PROMPT, 2.0, "prompt"),
            event(EventType.SESSION_END, 3.0, "end"),
            start(4.0, "resume", "resume"),
            event(EventType.USER_PROMPT, 5.0, "prompt-2"),
            event(EventType.SESSION_END, 6.0, "end-2"),
        ])

        self.assertEqual(result.status, EvidenceHealthStatus.HEALTHY)
        self.assertEqual(result.reasons, ())
        self.assertTrue(result.observed_start)
        self.assertTrue(result.observed_end)

    def test_resume_then_compact_continue_the_same_session(self):
        result = self.assess([
            start(1.0, "start", "startup"),
            event(EventType.SESSION_END, 2.0, "end"),
            start(3.0, "resume", "resume"),
            start(4.0, "compact", "compact"),
            event(EventType.USER_PROMPT, 5.0, "prompt"),
            event(EventType.SESSION_END, 6.0, "end-2"),
        ])

        self.assertEqual(result.status, EvidenceHealthStatus.HEALTHY)
        self.assertEqual(result.reasons, ())

    def test_continuation_start_without_an_earlier_end_is_valid(self):
        # Codex has no end hook, and a crashed Claude Code session resumes
        # without one.
        result = self.assess([
            start(1.0, "start", "startup"),
            event(EventType.USER_PROMPT, 2.0, "prompt"),
            start(3.0, "resume", "resume"),
            start(4.0, "compact", "compact"),
            event(EventType.SESSION_END, 5.0, "end"),
        ])

        self.assertEqual(result.status, EvidenceHealthStatus.HEALTHY)
        self.assertEqual(result.reasons, ())

    def test_first_start_may_be_a_continuation_start(self):
        result = self.assess([
            start(1.0, "resume", "resume"),
            event(EventType.USER_PROMPT, 2.0, "prompt"),
            event(EventType.SESSION_END, 3.0, "end"),
        ])

        self.assertEqual(result.status, EvidenceHealthStatus.HEALTHY)
        self.assertEqual(result.reasons, ())

    def test_repeated_start_without_a_continuation_source_is_a_duplicate(self):
        for source in (
            None, "", "startup", "clear", "fork", "new", "SessionStart", "sessionStart",
            "Resume", "COMPACT", 1, ["resume"],
        ):
            with self.subTest(source=source):
                result = self.assess([
                    start(1.0, "start", "startup"),
                    start(2.0, "again", source),
                    event(EventType.SESSION_END, 3.0, "end"),
                ])

                self.assertEqual(result.status, EvidenceHealthStatus.INVALID)
                self.assertEqual(self.located(result), [("duplicate_session_start", "again")])

    def test_start_with_non_mapping_data_is_not_a_continuation(self):
        result = self.assess([
            start(1.0, "start", "startup"),
            TraceEvent(
                event_type=EventType.SESSION_START,
                timestamp=2.0,
                event_id="again",
                session_id=SESSION_ID,
                data=["resume"],
            ),
            event(EventType.SESSION_END, 3.0, "end"),
        ])

        self.assertEqual(self.located(result), [("duplicate_session_start", "again")])

    def test_startup_after_end_stays_invalid(self):
        result = self.assess([
            start(1.0, "start", "startup"),
            event(EventType.SESSION_END, 2.0, "end"),
            start(3.0, "restart", "startup"),
            event(EventType.USER_PROMPT, 4.0, "prompt"),
            event(EventType.SESSION_END, 5.0, "end-2"),
        ])

        self.assertEqual(result.status, EvidenceHealthStatus.INVALID)
        self.assertEqual(
            self.located(result),
            [
                ("duplicate_session_start", "restart"),
                ("duplicate_session_end", "end-2"),
                ("events_after_session_end", "prompt"),
            ],
        )

    def test_two_ends_without_a_continuation_start_are_duplicates(self):
        result = self.assess([
            start(1.0, "start", "startup"),
            event(EventType.USER_PROMPT, 2.0, "prompt"),
            event(EventType.SESSION_END, 3.0, "end"),
            event(EventType.SESSION_END, 4.0, "end-2"),
        ])

        self.assertEqual(result.status, EvidenceHealthStatus.INVALID)
        self.assertEqual(self.located(result), [("duplicate_session_end", "end-2")])
        self.assertIn("'end'", result.reasons[0].message)

    def test_events_after_end_before_a_continuation_start_are_invalid(self):
        result = self.assess([
            start(1.0, "start", "startup"),
            event(EventType.SESSION_END, 2.0, "end"),
            event(EventType.USER_PROMPT, 3.0, "late"),
            event(EventType.USER_PROMPT, 4.0, "later"),
            start(5.0, "resume", "resume"),
            event(EventType.USER_PROMPT, 6.0, "prompt"),
            event(EventType.SESSION_END, 7.0, "end-2"),
        ])

        self.assertEqual(result.status, EvidenceHealthStatus.INVALID)
        self.assertEqual(self.located(result), [("events_after_session_end", "late")])

    def test_events_after_end_are_reported_once_per_closed_span(self):
        result = self.assess([
            start(1.0, "start", "startup"),
            event(EventType.SESSION_END, 2.0, "end"),
            event(EventType.USER_PROMPT, 3.0, "late"),
            start(4.0, "resume", "resume"),
            event(EventType.SESSION_END, 5.0, "end-2"),
            event(EventType.USER_PROMPT, 6.0, "late-2"),
            event(EventType.USER_PROMPT, 7.0, "late-3"),
        ])

        self.assertEqual(
            self.located(result),
            [("events_after_session_end", "late"), ("events_after_session_end", "late-2")],
        )

    def test_active_continued_session_reports_session_active(self):
        result = self.assess(
            [
                start(1.0, "start", "startup"),
                event(EventType.SESSION_END, 2.0, "end"),
                start(3.0, "resume", "resume"),
                event(EventType.USER_PROMPT, 4.0, "prompt"),
            ],
            session_finalized=False,
        )

        self.assertEqual(result.status, EvidenceHealthStatus.UNKNOWN)
        self.assertEqual(self.located(result), [("session_active", "")])
        self.assertTrue(result.observed_end)

    def test_continued_session_without_a_final_end_is_missing_its_end(self):
        result = self.assess(
            [
                start(1.0, "start", "startup"),
                event(EventType.SESSION_END, 2.0, "end"),
                start(3.0, "resume", "resume"),
                event(EventType.USER_PROMPT, 4.0, "prompt"),
            ],
            session_finalized=None,
        )

        self.assertEqual(result.status, EvidenceHealthStatus.PARTIAL)
        self.assertEqual(self.located(result), [("missing_session_end", "")])
        self.assertIn("continued", result.reasons[0].message)


if __name__ == "__main__":
    unittest.main()
