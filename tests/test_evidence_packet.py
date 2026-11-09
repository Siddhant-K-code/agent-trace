"""Tests for portable, privacy-minimized evidence packets."""

import json
import tempfile
import unittest
from pathlib import Path

from agent_trace.annotate import Annotation, add_annotation
from agent_trace.evidence_packet import (
    EVIDENCE_PACKET_SCHEMA,
    EVIDENCE_PACKET_SCHEMA_VERSION,
    build_evidence_packet,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _event(
    event_type: EventType,
    timestamp: float,
    session_id: str,
    *,
    event_id: str,
    parent_id: str = "",
    **data,
) -> TraceEvent:
    return TraceEvent(
        event_type=event_type,
        timestamp=timestamp,
        event_id=event_id,
        session_id=session_id,
        parent_id=parent_id,
        data=data,
    )


class TestEvidencePacket(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.store = TraceStore(self.root)
        self.session_id = "packet-session-01"
        self.meta = SessionMeta(
            session_id=self.session_id,
            started_at=100.0,
            ended_at=109.0,
            agent_name="PRIVATE-AGENT-NAME",
            command="/private/runner --token SECRET-TOKEN",
            total_duration_ms=9000,
        )
        self.store.create_session(self.meta)
        events = [
            _event(
                EventType.SESSION_START,
                100.0,
                self.session_id,
                event_id="start",
                agent_name="github-copilot",
                provider="copilot",
                source="startup",
                mode="copilot-hooks",
                redaction="on",
                command=["runner", "--token", "SECRET-TOKEN"],
            ),
            _event(
                EventType.USER_PROMPT,
                101.0,
                self.session_id,
                event_id="prompt",
                prompt="Email alice@example.com and use SECRET-PROMPT",
            ),
            _event(
                EventType.TOOL_CALL,
                102.0,
                self.session_id,
                event_id="SECRET-CALL-ID",
                tool_name="Bash",
                arguments={
                    "command": "curl -H 'Authorization: SECRET-TOKEN' https://example.test",
                    "cwd": "/Users/alice/private/repo",
                },
            ),
            _event(
                EventType.ERROR,
                103.0,
                self.session_id,
                event_id="error-1",
                parent_id="SECRET-CALL-ID",
                message="SECRET-FAILURE at /Users/alice/private/repo",
            ),
            _event(
                EventType.TOOL_CALL,
                104.0,
                self.session_id,
                event_id="call-2",
                tool_name="Bash",
                arguments={
                    "command": "curl -H 'Authorization: SECRET-TOKEN' https://example.test",
                    "cwd": "/Users/alice/private/repo",
                },
            ),
            _event(
                EventType.TOOL_RESULT,
                105.0,
                self.session_id,
                event_id="result-2",
                parent_id="call-2",
                result="SECRET-RESULT",
                success=True,
            ),
            _event(
                EventType.FILE_WRITE,
                106.0,
                self.session_id,
                event_id="write",
                path="/Users/alice/private/repo/secrets.txt",
                content="SECRET-FILE-CONTENT",
            ),
            _event(
                EventType.ASSISTANT_RESPONSE,
                108.0,
                self.session_id,
                event_id="response",
                text="SECRET-RESPONSE",
            ),
            _event(
                EventType.SESSION_END,
                109.0,
                self.session_id,
                event_id="end",
                exit_code=0,
            ),
        ]
        for event in events:
            self.store.append_event(self.session_id, event)
        self.store.update_meta(self.meta)
        add_annotation(
            self.store,
            self.session_id,
            Annotation(
                event_id="error-1",
                label="root-cause",
                note="SHAREABLE-NOTE",
                author="alice@example.com",
                shareable=True,
            ),
        )
        add_annotation(
            self.store,
            self.session_id,
            Annotation(
                event_id="prompt",
                label="private",
                note="PRIVATE-ANNOTATION",
            ),
        )

    def _source_bytes(self) -> tuple[bytes, bytes]:
        session_dir = self.root / self.session_id
        return (
            (session_dir / "events.ndjson").read_bytes(),
            (session_dir / "annotations.jsonl").read_bytes(),
        )

    def test_minimized_packet_omits_content_and_preserves_review_signals(self):
        before = self._source_bytes()
        packet = build_evidence_packet(self.store, self.session_id)
        after = self._source_bytes()
        self.assertEqual(after, before)

        payload = json.dumps(
            [json.loads(event.to_json()) for event in packet.events],
            sort_keys=True,
        )
        for secret in (
            "SECRET-TOKEN",
            "SECRET-PROMPT",
            "SECRET-FAILURE",
            "SECRET-RESULT",
            "SECRET-FILE-CONTENT",
            "SECRET-RESPONSE",
            "alice@example.com",
            "/Users/alice/private/repo",
            "PRIVATE-AGENT-NAME",
            "PRIVATE-ANNOTATION",
            "SECRET-CALL-ID",
        ):
            self.assertNotIn(secret, payload)

        source_events = self.store.load_events(self.session_id)
        self.assertEqual(
            [event.event_type for event in packet.events],
            [event.event_type for event in source_events],
        )
        self.assertEqual(
            [event.timestamp for event in packet.events],
            [event.timestamp for event in source_events],
        )
        calls = [
            event for event in packet.events if event.event_type == EventType.TOOL_CALL
        ]
        self.assertEqual(calls[0].data["operation_ref"], calls[1].data["operation_ref"])
        self.assertFalse(calls[0].data["retry"])
        self.assertTrue(calls[1].data["retry"])
        self.assertTrue(calls[1].data["recovery_attempt"])
        result = next(
            event for event in packet.events if event.event_type == EventType.TOOL_RESULT
        )
        self.assertTrue(result.data["recovered"])
        self.assertEqual(packet.behavior.failure_event_count, 1)
        self.assertEqual(packet.behavior.retry_count, 1)
        self.assertEqual(packet.behavior.recovered_operation_count, 1)
        self.assertEqual([annotation.note for annotation in packet.annotations], ["SHAREABLE-NOTE"])
        self.assertEqual(packet.annotations[0].annotation_id, "annotation-000001")
        self.assertEqual(packet.annotations[0].event_id, "event-000004")
        self.assertEqual(packet.annotations[0].author, "")

    def test_manifest_is_versioned_deterministic_and_links_source_and_packet(self):
        first = build_evidence_packet(self.store, self.session_id)
        second = build_evidence_packet(self.store, self.session_id)
        self.assertEqual(first.manifest, second.manifest)

        manifest = first.manifest
        self.assertEqual(manifest["schema"], EVIDENCE_PACKET_SCHEMA)
        self.assertEqual(manifest["schema_version"], EVIDENCE_PACKET_SCHEMA_VERSION)
        self.assertEqual(manifest["mode"], "minimized")
        self.assertTrue(manifest["privacy"]["source_trace_unchanged"])
        self.assertEqual(
            manifest["privacy"]["annotations"],
            "explicitly_shareable_only",
        )
        self.assertEqual(manifest["provenance"]["provider"], "copilot")
        self.assertNotEqual(manifest["evidence_health"]["status"], "")
        for key in (
            "source_trace_sha256",
            "packet_events_sha256",
            "included_annotations_sha256",
            "manifest_sha256",
        ):
            self.assertRegex(manifest["integrity"][key], r"^sha256:[0-9a-f]{64}$")
        self.assertIn(
            "decoded ordered source events",
            manifest["integrity"]["source_trace_digest_scope"],
        )

    def test_include_content_is_explicit_and_includes_all_annotations(self):
        minimized = build_evidence_packet(self.store, self.session_id)
        full = build_evidence_packet(
            self.store,
            self.session_id,
            include_content=True,
        )
        serialized = json.dumps(
            [json.loads(event.to_json()) for event in full.events],
            sort_keys=True,
        )
        self.assertIn("SECRET-PROMPT", serialized)
        self.assertIn("SECRET-RESULT", serialized)
        self.assertEqual(
            {annotation.note for annotation in full.annotations},
            {"SHAREABLE-NOTE", "PRIVATE-ANNOTATION"},
        )
        self.assertEqual(full.preview.mode, "include_content")
        self.assertGreater(full.preview.unresolved_sensitive_field_warning_count, 0)
        self.assertEqual(
            full.manifest["integrity"]["source_trace_sha256"],
            minimized.manifest["integrity"]["source_trace_sha256"],
        )
        self.assertNotEqual(
            full.manifest["integrity"]["packet_events_sha256"],
            minimized.manifest["integrity"]["packet_events_sha256"],
        )

    def test_incomplete_session_keeps_health_status_and_gaps(self):
        session_id = "incomplete-session"
        meta = SessionMeta(session_id=session_id, started_at=1.0)
        self.store.create_session(meta)
        self.store.append_event(
            session_id,
            _event(
                EventType.SESSION_START,
                1.0,
                session_id,
                event_id="start-incomplete",
                agent_name="github-copilot",
                provider="copilot",
                source="startup",
                mode="copilot-hooks",
                redaction="off",
            ),
        )
        self.store.append_event(
            session_id,
            _event(
                EventType.TOOL_CALL,
                2.0,
                session_id,
                event_id="unpaired",
                tool_name="Read",
                arguments={"file_path": "/private/incomplete.txt"},
            ),
        )
        packet = build_evidence_packet(self.store, session_id)
        codes = {reason.code for reason in packet.health.result.reasons}
        self.assertIn("missing_session_end", codes)
        self.assertIn("unpaired_tool_call", codes)
        self.assertEqual(
            packet.preview.evidence_health_status,
            packet.health.result.status.value,
        )
        self.assertGreaterEqual(packet.preview.known_gap_count, 1)


if __name__ == "__main__":
    unittest.main()
