import json
import unittest
from pathlib import Path

from agent_trace.annotate import Annotation
from agent_trace.models import EventType, TraceEvent


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "representative_session"


class TestRepresentativeEvidenceFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = [
            TraceEvent.from_json(line)
            for line in (FIXTURE_DIR / "events.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.annotations = [
            Annotation.from_json(line)
            for line in (FIXTURE_DIR / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.expected = json.loads((FIXTURE_DIR / "expected.json").read_text(encoding="utf-8"))

    def test_fixture_uses_one_session_and_expected_count(self):
        self.assertEqual(len(self.events), self.expected["expected_event_count"])
        self.assertEqual({event.session_id for event in self.events}, {self.expected["session_id"]})

    def test_fixture_metadata_matches_session_start(self):
        starts = [event for event in self.events if event.event_type == EventType.SESSION_START]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].data["provider"], self.expected["provider"])
        self.assertEqual(starts[0].data["fixture_version"], self.expected["fixture_version"])
        self.assertEqual(starts[0].session_id, self.expected["session_id"])

    def test_failed_call_has_one_terminal_outcome(self):
        by_id = {event.event_id: event for event in self.events}
        failure = by_id[self.expected["expected_failure_event_id"]]
        children = [
            child for child in self.events
            if child.parent_id == failure.parent_id
            and child.event_type in (EventType.TOOL_RESULT, EventType.ERROR)
        ]
        self.assertEqual([child.event_id for child in children], [failure.event_id])

    def test_fixture_covers_review_workflow(self):
        kinds = {event.event_type for event in self.events}
        for required in (
            EventType.SESSION_START,
            EventType.SESSION_END,
            EventType.USER_PROMPT,
            EventType.ASSISTANT_RESPONSE,
            EventType.FILE_READ,
            EventType.FILE_WRITE,
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.ERROR,
            EventType.DECISION,
        ):
            self.assertIn(required, kinds)

    def test_fixture_contains_failure_retry_and_recovery(self):
        by_id = {event.event_id: event for event in self.events}
        failure = by_id[self.expected["expected_failure_event_id"]]
        retry = by_id[self.expected["expected_retry_call_id"]]
        recovery = by_id[self.expected["expected_recovery_result_id"]]
        self.assertEqual(failure.event_type, EventType.ERROR)
        self.assertEqual(retry.data["retry_of"], failure.parent_id)
        self.assertEqual(recovery.parent_id, retry.event_id)
        self.assertEqual(recovery.data["exit_code"], 0)

    def test_fixture_has_explicit_redaction_and_human_correction(self):
        self.assertTrue(any(event.redacted for event in self.events))
        self.assertEqual(len(self.annotations), 1)
        annotation = self.annotations[0]
        self.assertEqual(annotation.annotation_id, self.expected["expected_annotation_id"])
        self.assertIn(annotation.event_id, {event.event_id for event in self.events})
        self.assertTrue(annotation.note.startswith("Human correction:"))

    def test_fixture_has_one_deliberate_orphan_gap(self):
        by_id = {event.event_id: event for event in self.events}
        gap = self.expected["intentional_gap"]
        orphan = by_id[gap["event_id"]]
        self.assertEqual(orphan.parent_id, gap["parent_id"])
        self.assertNotIn(orphan.parent_id, by_id)
        self.assertTrue(orphan.data["fixture_gap"])


if __name__ == "__main__":
    unittest.main()
