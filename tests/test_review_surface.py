from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent_trace
from agent_trace import replay
from agent_trace.annotate import Annotation, add_annotation
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.review_surface import (
    build_minimized_review_payload,
    build_session_review,
    render_minimized_review_html,
)
from agent_trace.store import TraceStore


FIXTURE_DIR = (
    Path(agent_trace.__file__).parent
    / "corpus"
    / "v1"
    / "fixture-review-001"
)


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
        self.fixture_ids = [event.event_id for event in events]
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
        self.assertIn("failures", failure["categories"])
        annotated = next(item for item in review["items"] if item["annotations"])
        self.assertIn("annotations", annotated["categories"])
        self.assertIn("corrections", annotated["categories"])
        self.assertTrue(annotated["annotations"][0]["correction"])
        self.assertIn("Human correction:", annotated["annotations"][0]["note"])

        counts = {entry["id"]: entry["count"] for entry in review["filters"]}
        self.assertEqual(counts["tests"], 2)
        self.assertEqual(counts["retry"], 1)
        self.assertEqual(counts["recovery"], 1)
        self.assertEqual(counts["corrections"], 1)
        self.assertGreaterEqual(counts["gaps"], 1)

    def test_replay_shows_the_review_items_in_the_same_order(self) -> None:
        shown: list[TraceEvent] = []
        format_event = replay.format_event

        def record(event: TraceEvent, *args, **kwargs) -> str:
            shown.append(event)
            return format_event(event, *args, **kwargs)

        with patch.object(replay, "format_event", side_effect=record):
            replay.replay_session(self.store, self.session_id, out=io.StringIO())
        review = build_session_review(self.store, self.session_id)

        replayed = [
            (event.event_id, event.event_type.value, event.parent_id or None, event.redacted)
            for event in shown
        ]
        reviewed = [
            (item["event_id"], item["event_type"], item["parent_id"], item["redacted"])
            for item in review["items"]
        ]
        self.assertEqual(replayed, reviewed)
        self.assertEqual([event_id for event_id, *_ in replayed], self.fixture_ids)

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
        minimized = build_minimized_review_payload(review)
        self.assertEqual(minimized["session"]["provider"], "claude")
        self.assertEqual(minimized["session"]["capture_method"], "cli_hooks")

    def _resumed_session(
        self,
        session_id: str,
        *,
        meta_ended_at: float | None = None,
        ended_again_at: float | None = None,
    ) -> dict:
        self.store.create_session(SessionMeta(
            session_id=session_id,
            started_at=1000,
            ended_at=meta_ended_at,
            agent_name="Claude Code",
            command="Claude Code (resume)",
        ))
        events = [
            TraceEvent(EventType.SESSION_START, timestamp=1000, event_id="start",
                       data={"source": "startup"}),
            TraceEvent(EventType.USER_PROMPT, timestamp=1001, event_id="prompt-1",
                       data={"prompt": "first"}),
            TraceEvent(EventType.SESSION_END, timestamp=1010, event_id="end-1", data={}),
            TraceEvent(EventType.SESSION_START, timestamp=1100, event_id="resume",
                       data={"source": "resume"}),
            TraceEvent(EventType.USER_PROMPT, timestamp=1101, event_id="prompt-2",
                       data={"prompt": "second"}),
        ]
        if ended_again_at is not None:
            events.append(TraceEvent(
                EventType.SESSION_END, timestamp=ended_again_at, event_id="end-2", data={},
            ))
        for event in events:
            event.session_id = session_id
            self.store.append_event(session_id, event)
        return build_session_review(self.store, session_id)["session"]

    def test_resumed_session_is_live_until_it_ends_again(self) -> None:
        session = self._resumed_session("resumed-live")

        self.assertIsNone(session["ended_at"])
        self.assertIsNone(session["duration_ms"])
        self.assertTrue(session["active"])

    def test_resumed_session_that_ended_again_uses_the_last_end(self) -> None:
        session = self._resumed_session("resumed-ended", ended_again_at=1120)

        self.assertEqual(session["ended_at"], 1120)
        self.assertEqual(session["duration_ms"], 120_000)
        self.assertFalse(session["active"])

    def test_recorded_meta_end_wins_over_events(self) -> None:
        session = self._resumed_session("resumed-meta-end", meta_ended_at=1200)

        self.assertEqual(session["ended_at"], 1200)
        self.assertEqual(session["duration_ms"], 200_000)
        self.assertFalse(session["active"])

    def _parentless_response_session(self, session_id: str, provider: str) -> dict:
        self.store.create_session(SessionMeta(
            session_id=session_id,
            started_at=100,
            ended_at=103,
            agent_name="synthetic",
            command="synthetic",
        ))
        for event in (
            TraceEvent(
                EventType.SESSION_START,
                timestamp=100,
                session_id=session_id,
                event_id=f"{provider}-start",
                data={"provider": provider, "source": "startup"},
            ),
            TraceEvent(
                EventType.LLM_RESPONSE,
                timestamp=101,
                session_id=session_id,
                event_id=f"{provider}-model-request",
                data={
                    "provider": provider,
                    "hook_event": "post_model_request",
                    "status": "completed",
                    "token_usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                },
            ),
            TraceEvent(
                EventType.SESSION_END,
                timestamp=103,
                session_id=session_id,
                event_id=f"{provider}-end",
                data={"reason": "other"},
            ),
        ):
            self.store.append_event(session_id, event)
        review = build_session_review(self.store, session_id)
        return {item["event_id"]: item for item in review["items"]}[
            f"{provider}-model-request"
        ]

    def test_declared_parentless_response_is_not_orphaned(self) -> None:
        item = self._parentless_response_session("codex-review", "codex")

        self.assertNotEqual(item["relationship"], "orphaned")
        self.assertNotIn("gaps", item["categories"])

    def test_undeclared_parentless_response_is_orphaned(self) -> None:
        item = self._parentless_response_session("claude-review", "claude")

        self.assertEqual(item["relationship"], "orphaned")
        self.assertEqual(item["relationship_issue"], "missing_parent")
        self.assertIn("gaps", item["categories"])

    def test_minimized_share_allowlists_capture_and_annotation_metadata(self) -> None:
        review = build_session_review(self.store, self.session_id)
        annotated = next(item for item in review["items"] if item["annotations"])
        annotated["annotations"][0]["label"] = "customer-secret-label"

        minimized = build_minimized_review_payload(review)
        encoded = json.dumps(minimized)

        self.assertEqual(minimized["session"]["provider"], "unknown")
        self.assertEqual(minimized["session"]["capture_method"], "unknown")
        self.assertNotIn(self.session_id, encoded)
        self.assertNotIn("customer-secret-label", encoded)
        self.assertNotIn("src/example.py", encoded)
        self.assertNotIn("python -m pytest", encoded)
        self.assertNotIn("Human correction:", encoded)
        self.assertNotIn("1 passed", encoded)
        labels = [
            annotation["label"]
            for item in minimized["items"]
            for annotation in item["annotations"]
        ]
        self.assertEqual(labels, ["correction"])

        rendered = render_minimized_review_html(minimized)
        self.assertIn("Privacy-minimized evidence review", rendered)
        self.assertNotIn("customer-secret-label", rendered)

    def test_preserves_missing_and_duplicate_target_annotations_as_gaps(self) -> None:
        add_annotation(self.store, self.session_id, Annotation(
            annotation_id="ann-missing",
            event_id="not-recorded",
            note="Missing target note",
        ))
        for timestamp in (20, 21):
            self.store.append_event(self.session_id, TraceEvent(
                EventType.FILE_READ,
                timestamp=1789300800 + timestamp,
                session_id=self.session_id,
                event_id="duplicate-target",
                data={"path": "src/duplicate.py"},
            ))
        add_annotation(self.store, self.session_id, Annotation(
            annotation_id="ann-duplicate",
            event_id="duplicate-target",
            note="Duplicate target note",
        ))

        review = build_session_review(self.store, self.session_id)

        self.assertEqual(
            [annotation["annotation_id"] for annotation in review["unresolved_annotations"]],
            ["ann-missing", "ann-duplicate"],
        )
        self.assertEqual(
            [
                reason["code"]
                for reason in review["capture_gaps"]
                if reason["code"] == "unresolved_annotation"
            ],
            ["unresolved_annotation", "unresolved_annotation"],
        )


if __name__ == "__main__":
    unittest.main()
