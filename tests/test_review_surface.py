from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_trace.annotate import Annotation, add_annotation
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.review_surface import (
    build_minimized_review_payload,
    build_session_review,
    render_minimized_review_html,
)
from agent_trace.store import TraceStore


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "representative_session"


class ReviewSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.tmp.name)
        expected = json.loads((FIXTURE_DIR / "expected.json").read_text(encoding="utf-8"))
        self.session_id = expected["session_id"]
        events = [
            TraceEvent.from_json(line)
            for line in (FIXTURE_DIR / "events.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.store.create_session(SessionMeta(
            session_id=self.session_id,
            started_at=events[0].timestamp,
            ended_at=events[-1].timestamp,
            agent_name="synthetic-review-agent",
            command="fixture capture",
        ))
        for event in events:
            self.store.append_event(self.session_id, event)
        for line in (FIXTURE_DIR / "annotations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                add_annotation(self.store, self.session_id, Annotation.from_json(line))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_builds_review_header_filters_and_relationships(self) -> None:
        review = build_session_review(self.store, self.session_id)

        self.assertEqual(review["schema_version"], 1)
        session = review["session"]
        self.assertEqual(session["session_id"], self.session_id)
        self.assertEqual(session["provider"], "synthetic")
        self.assertEqual(session["capture_method"], "fixture")
        self.assertEqual(session["duration_ms"], 11_000)
        self.assertEqual(review["privacy"]["content_mode"], "mixed")
        self.assertEqual(review["live_capture"]["status"], "not_supported")

        by_id = {item["event_id"]: item for item in review["items"]}
        self.assertEqual(by_id["result-test-2"]["parent_id"], "tool-test-2")
        self.assertIn("result-test-2", by_id["tool-test-2"]["child_ids"])
        self.assertEqual(by_id["ev-orphan"]["relationship"], "orphaned")
        self.assertEqual(by_id["ev-orphan"]["relationship_issue"], "missing_parent")
        self.assertIn("gaps", by_id["ev-orphan"]["categories"])

        failure = by_id["ev-error"]
        self.assertIn("annotations", failure["categories"])
        self.assertIn("corrections", failure["categories"])
        self.assertTrue(failure["annotations"][0]["correction"])
        self.assertIn("Human correction:", failure["annotations"][0]["note"])

        counts = {entry["id"]: entry["count"] for entry in review["filters"]}
        self.assertEqual(counts["tests"], 2)
        self.assertEqual(counts["retry"], 1)
        self.assertEqual(counts["recovery"], 1)
        self.assertEqual(counts["corrections"], 1)
        self.assertGreaterEqual(counts["gaps"], 1)

    def test_minimized_share_omits_recorded_values(self) -> None:
        review = build_session_review(self.store, self.session_id)
        minimized = build_minimized_review_payload(review)
        encoded = json.dumps(minimized)

        self.assertEqual(minimized["privacy"]["mode"], "minimized")
        self.assertNotIn(self.session_id, encoded)
        self.assertNotIn("src/example.py", encoded)
        self.assertNotIn("python -m pytest", encoded)
        self.assertNotIn("Human correction:", encoded)
        self.assertNotIn("1 passed", encoded)

        rendered = render_minimized_review_html(minimized)
        self.assertIn("Privacy-minimized evidence review", rendered)
        self.assertNotIn("src/example.py", rendered)
        self.assertNotIn("python -m pytest", rendered)

    def test_surfaces_live_capture_verification_for_supported_provider(self) -> None:
        session_id = "claude-review"
        self.store.create_session(SessionMeta(
            session_id=session_id,
            started_at=100,
            ended_at=103,
            agent_name="Claude Code",
            command="Claude Code (startup)",
        ))
        for event in (
            TraceEvent(
                EventType.SESSION_START,
                timestamp=100,
                session_id=session_id,
                event_id="claude-start",
                data={
                    "mode": "claude-code",
                    "provider": "claude",
                    "source": "startup",
                    "redaction": "on",
                },
            ),
            TraceEvent(
                EventType.USER_PROMPT,
                timestamp=101,
                session_id=session_id,
                event_id="claude-prompt",
                data={"prompt": "synthetic prompt", "provider": "claude"},
            ),
            TraceEvent(
                EventType.ASSISTANT_RESPONSE,
                timestamp=102,
                session_id=session_id,
                event_id="claude-response",
                data={"text": "synthetic response", "provider": "claude"},
            ),
            TraceEvent(
                EventType.SESSION_END,
                timestamp=103,
                session_id=session_id,
                event_id="claude-end",
                data={"reason": "other"},
            ),
        ):
            self.store.append_event(session_id, event)

        review = build_session_review(self.store, session_id)

        self.assertTrue(review["live_capture"]["supported"])
        self.assertEqual(review["live_capture"]["status"], "observed")
        self.assertEqual(review["live_capture"]["provider"], "claude")


if __name__ == "__main__":
    unittest.main()
