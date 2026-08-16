"""Tests for deterministic context compaction analysis (issue #217)."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.compaction import (
    CompactionCheckpointWatcher,
    analyze_compactions,
    behavior_diff_for_compaction,
    build_checkpoint_markdown,
    cmd_compaction,
    context_diff_for_compaction,
    context_fill_ratio,
    detect_compactions,
    format_compaction_report,
    write_checkpoint,
    _rule_post_compaction_regression,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.jsonl_import import import_jsonl
from agent_trace.store import TraceStore


def _event(event_type: EventType, timestamp: float, **data) -> TraceEvent:
    return TraceEvent(
        event_type=event_type,
        timestamp=timestamp,
        session_id="session-1",
        data=data,
    )


def _request(timestamp: float, tokens: int, **data) -> TraceEvent:
    return _event(
        EventType.LLM_REQUEST,
        timestamp,
        model="claude-sonnet-4",
        input_tokens=tokens,
        **data,
    )


def _read(timestamp: float, path: str) -> TraceEvent:
    return _event(
        EventType.TOOL_CALL,
        timestamp,
        tool_name="Read",
        arguments={"file_path": path},
    )


def _store(events: list[TraceEvent]) -> tuple[tempfile.TemporaryDirectory, TraceStore, str]:
    temporary = tempfile.TemporaryDirectory()
    store = TraceStore(temporary.name, redact=False)
    meta = SessionMeta(session_id="session-1", started_at=events[0].timestamp if events else 0)
    store.create_session(meta)
    for event in events:
        store.append_event(meta.session_id, event)
    return temporary, store, meta.session_id


class CompactionDetectionTests(unittest.TestCase):
    def test_detects_drop_greater_than_threshold_and_estimates_cost(self):
        events = [_request(100.0, 180_000), _request(142.0, 12_000)]

        compactions = detect_compactions(events)

        self.assertEqual(len(compactions), 1)
        event = compactions[0]
        self.assertEqual(event.before_request_index, 1)
        self.assertEqual(event.after_request_index, 2)
        self.assertEqual(event.tokens_dropped, 168_000)
        self.assertAlmostEqual(event.drop_ratio, 168_000 / 180_000)
        self.assertAlmostEqual(event.estimated_cost_usd, 0.504)
        self.assertEqual(event.offset_seconds, 42.0)

    def test_exactly_fifty_percent_is_not_more_than_threshold(self):
        self.assertEqual(
            detect_compactions([_request(1, 1000), _request(2, 500)]),
            [],
        )
        self.assertEqual(
            len(detect_compactions(
                [_request(1, 1000), _request(2, 500)], threshold=0.49,
            )),
            1,
        )

    def test_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            detect_compactions([], threshold=1.0)

    def test_uses_response_usage_for_request_context(self):
        events = [
            _event(EventType.LLM_REQUEST, 1, model="gpt-4o", messages=[]),
            _event(
                EventType.LLM_RESPONSE,
                2,
                model="gpt-4o",
                usage={"prompt_tokens": 120_000, "completion_tokens": 10},
            ),
            _event(EventType.LLM_REQUEST, 3, model="gpt-4o", messages=[]),
            _event(
                EventType.LLM_RESPONSE,
                4,
                model="gpt-4o",
                usage={"prompt_tokens": 10_000, "completion_tokens": 10},
            ),
        ]

        compactions = detect_compactions(events)

        self.assertEqual(len(compactions), 1)
        self.assertEqual(compactions[0].tokens_before, 120_000)
        self.assertEqual(compactions[0].tokens_after, 10_000)

    def test_includes_cache_tokens_and_response_only_imported_turns(self):
        # Claude Code imports can retain one usage-bearing assistant event per turn.
        events = [
            _event(
                EventType.ASSISTANT_RESPONSE,
                1,
                model="claude-sonnet-4",
                usage={
                    "input_tokens": 1_000,
                    "cache_creation_input_tokens": 20_000,
                    "cache_read_input_tokens": 160_000,
                },
            ),
            _event(
                EventType.ASSISTANT_RESPONSE,
                2,
                model="claude-sonnet-4",
                usage={"input_tokens": 12_000},
            ),
        ]

        compactions = detect_compactions(events)

        self.assertEqual(len(compactions), 1)
        self.assertEqual(compactions[0].tokens_before, 181_000)
        self.assertEqual(compactions[0].tokens_after, 12_000)

    def test_does_not_report_or_price_estimated_only_context_drops(self):
        events = [
            _event(EventType.LLM_REQUEST, 1, prompt="x" * 40_000),
            _event(EventType.LLM_REQUEST, 2, prompt="short"),
        ]

        self.assertEqual(detect_compactions(events), [])

    def test_compares_only_requests_from_the_same_context_stream(self):
        events = [
            _event(
                EventType.ASSISTANT_RESPONSE,
                1,
                context_stream="main",
                usage={"input_tokens": 180_000},
            ),
            _event(
                EventType.ASSISTANT_RESPONSE,
                2,
                context_stream="sidechain:research",
                usage={"input_tokens": 10_000},
            ),
        ]

        self.assertEqual(detect_compactions(events), [])

        events.extend([
            _event(
                EventType.ASSISTANT_RESPONSE,
                3,
                context_stream="sidechain:review",
                usage={"input_tokens": 180_000},
            ),
            _event(
                EventType.ASSISTANT_RESPONSE,
                4,
                context_stream="main",
                usage={"input_tokens": 185_000},
            ),
            _event(
                EventType.ASSISTANT_RESPONSE,
                5,
                context_stream="sidechain:review",
                context_summary="Review the compaction implementation.",
                usage={"input_tokens": 12_000},
            ),
        ])

        compactions = detect_compactions(events)
        self.assertEqual(len(compactions), 1)
        self.assertEqual(compactions[0].stream_id, "sidechain:review")


class ImportedCompactionTests(unittest.TestCase):
    def test_imported_usage_and_compaction_summary_drive_the_report(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "claude.jsonl"
        entries = [
            {
                "type": "user",
                "sessionId": "imported-session",
                "timestamp": "2025-01-01T00:00:00Z",
                "uuid": "u1",
                "message": {
                    "role": "user",
                    "content": (
                        "Implement rate limiting middleware.\n"
                        "Do not modify src/auth.py."
                    ),
                },
            },
            {
                "type": "assistant",
                "sessionId": "imported-session",
                "timestamp": "2025-01-01T00:00:01Z",
                "uuid": "a1",
                "parentUuid": "u1",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4",
                    "content": [{"type": "text", "text": "Working on it."}],
                    "usage": {"input_tokens": 180_000, "output_tokens": 10},
                },
            },
            {
                "type": "system",
                "subtype": "compact_boundary",
                "sessionId": "imported-session",
                "timestamp": "2025-01-01T00:00:02Z",
                "uuid": "c1",
            },
            {
                "type": "user",
                "sessionId": "imported-session",
                "timestamp": "2025-01-01T00:00:03Z",
                "uuid": "u2",
                "parentUuid": "c1",
                "message": {
                    "role": "user",
                    "content": (
                        "This session is being continued from a previous conversation "
                        "that ran out of context. Implement rate limiting middleware."
                    ),
                },
            },
            {
                "type": "assistant",
                "sessionId": "imported-session",
                "timestamp": "2025-01-01T00:00:04Z",
                "uuid": "a2",
                "parentUuid": "u2",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4",
                    "content": [{"type": "text", "text": "Continuing."}],
                    "usage": {"input_tokens": 12_000, "output_tokens": 10},
                },
            },
        ]
        source.write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries),
            encoding="utf-8",
        )
        store = TraceStore(Path(temporary.name) / "traces", redact=False)

        session_id = import_jsonl(source, store=store)
        events = store.load_events(session_id)
        report = analyze_compactions(events, session_id)

        self.assertEqual(len(report.events), 1)
        compaction = report.events[0]
        self.assertEqual(compaction.tokens_before, 180_000)
        self.assertEqual(compaction.tokens_after, 12_000)
        self.assertTrue(compaction.context_diff.context_available)
        survived = {item.text for item in compaction.context_diff.survived}
        dropped = {item.text for item in compaction.context_diff.likely_dropped}
        self.assertIn("Implement rate limiting middleware.", survived)
        self.assertIn("Do not modify src/auth.py.", dropped)

    def test_response_text_is_not_treated_as_the_compaction_summary(self):
        events = [
            _event(
                EventType.ASSISTANT_RESPONSE,
                1,
                text="Original response",
                usage={"input_tokens": 180_000},
            ),
            _event(
                EventType.ASSISTANT_RESPONSE,
                2,
                text="A short answer, not the retained context",
                usage={"input_tokens": 12_000},
            ),
        ]

        compaction = detect_compactions(events)[0]
        diff = context_diff_for_compaction(events, compaction)

        self.assertFalse(diff.context_available)
        self.assertEqual(diff.survived, [])
        self.assertEqual(diff.likely_dropped, [])


class ContextAndBehaviorDiffTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            _event(
                EventType.USER_PROMPT,
                1,
                prompt=(
                    "Implement rate limiting middleware.\n"
                    "It must be Redis-compatible.\n"
                    "Do not modify src/auth.py."
                ),
            ),
            _read(2, "src/middleware.py"),
            _read(3, "src/config.py"),
            _event(
                EventType.DECISION,
                4,
                decision="Token bucket chosen over fixed window",
                reason="memory cost",
            ),
            _request(5, 190_000, messages=[{"role": "user", "content": "work"}]),
            _request(
                6,
                12_000,
                messages=[{
                    "role": "system",
                    "content": (
                        "Continue implementing rate limiting middleware with a token bucket. "
                        "Previously read src/middleware.py."
                    ),
                }],
            ),
            _read(7, "src/middleware.py"),
            _read(8, "src/middleware.py"),
            _read(9, "src/middleware.py"),
            _event(
                EventType.TOOL_CALL,
                10,
                tool_name="Bash",
                arguments={"command": "ls src"},
            ),
        ]
        self.compaction = detect_compactions(self.events)[0]

    def test_context_diff_identifies_survived_and_high_risk_dropped_items(self):
        diff = context_diff_for_compaction(self.events, self.compaction)

        survived = {(item.kind, item.text) for item in diff.survived}
        dropped = {(item.kind, item.text) for item in diff.likely_dropped}
        self.assertIn(("Task goal", "Implement rate limiting middleware."), survived)
        self.assertIn(("File read", "src/middleware.py"), survived)
        self.assertIn(("Constraint", "It must be Redis-compatible."), dropped)
        self.assertIn(("Constraint", "Do not modify src/auth.py."), dropped)
        self.assertEqual(len(diff.high_risk_drops), 2)

    def test_context_diff_inventories_modified_files(self):
        events = [
            _event(EventType.USER_PROMPT, 1, prompt="Implement middleware."),
            _event(EventType.FILE_WRITE, 2, path="src/middleware.py"),
            _request(3, 180_000),
            _request(4, 10_000, context_summary="Implement middleware."),
        ]

        diff = context_diff_for_compaction(events, detect_compactions(events)[0])

        self.assertIn(
            ("File modified", "src/middleware.py"),
            {(item.kind, item.text) for item in diff.likely_dropped},
        )

    def test_behavior_diff_flags_rereads_redundancy_and_reexploration(self):
        behavior = behavior_diff_for_compaction(
            self.events,
            self.compaction,
            redundant_read_threshold=3,
        )

        self.assertTrue(behavior.regressed)
        self.assertEqual(behavior.reread_files, ("src/middleware.py",))
        self.assertEqual(behavior.before.redundant_read_violations, 0)
        self.assertEqual(behavior.after.redundant_read_violations, 1)
        self.assertEqual(behavior.after.reexploration_calls, 1)
        self.assertTrue(any("re-read" in reason for reason in behavior.regressions))

    def test_behavior_diff_includes_saturation_and_registered_lint_rules(self):
        events = [
            _request(1, 100_000),
            _request(2, 10_000),
            _request(3, 180_000),
            _event(EventType.TOOL_CALL, 4, tool_name="Bash", arguments={}),
            _event(EventType.ERROR, 5, message="failed"),
            _event(EventType.TOOL_CALL, 6, tool_name="Bash", arguments={}),
            _event(EventType.ERROR, 7, message="failed"),
            _event(EventType.TOOL_CALL, 8, tool_name="Bash", arguments={}),
            _event(EventType.ERROR, 9, message="failed"),
        ]
        compaction = detect_compactions(events)[0]

        behavior = behavior_diff_for_compaction(events, compaction)

        self.assertEqual(behavior.before.context_saturation_violations, 0)
        self.assertEqual(behavior.after.context_saturation_violations, 1)
        self.assertIn(("context-saturation", 0, 1), behavior.lint_deltas)
        self.assertIn(("error-retry-loop", 0, 1), behavior.lint_deltas)
        self.assertTrue(any(
            "context-saturation" in reason for reason in behavior.regressions
        ))

    def test_behavior_window_counts_events_from_only_the_compacted_stream(self):
        events = [
            _read(1, "src/a.py"),
            _request(2, 180_000),
            _request(3, 10_000),
            _event(
                EventType.TOOL_CALL,
                4,
                context_stream="sidechain:review",
                tool_name="Read",
                arguments={"file_path": "review-1.py"},
            ),
            _read(5, "src/a.py"),
            _event(
                EventType.TOOL_CALL,
                6,
                context_stream="sidechain:review",
                tool_name="Read",
                arguments={"file_path": "review-2.py"},
            ),
            _read(7, "src/a.py"),
            _event(
                EventType.TOOL_CALL,
                8,
                context_stream="sidechain:review",
                tool_name="Read",
                arguments={"file_path": "review-3.py"},
            ),
            _read(9, "src/a.py"),
        ]

        behavior = behavior_diff_for_compaction(
            events, detect_compactions(events)[0], window=4,
        )

        self.assertEqual(behavior.after.redundant_read_violations, 1)

    def test_report_and_formatter_include_requested_sections(self):
        report = analyze_compactions(self.events, "session-1")
        output = io.StringIO()

        format_compaction_report(
            report, output, show_diff=True, show_behavior_diff=True,
        )

        text = output.getvalue()
        self.assertIn("1 compaction event(s)", text)
        self.assertIn("Estimated cost of dropped context", text)
        self.assertIn("Likely dropped", text)
        self.assertIn("High-risk drop", text)
        self.assertIn("Behavior change after compaction #1", text)
        self.assertIn("Verdict: behavior regressed", text)

    def test_no_event_report_is_explicit(self):
        output = io.StringIO()
        format_compaction_report(analyze_compactions([], "empty"), output)
        self.assertIn("No compaction events detected", output.getvalue())


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            _event(
                EventType.USER_PROMPT,
                100,
                prompt="Implement rate limiting. Must use Redis. Do not modify auth.py.",
            ),
            _read(112, "src/middleware.py"),
            _event(
                EventType.DECISION,
                118,
                decision="Chose token bucket instead of fixed window",
            ),
            _event(
                EventType.FILE_WRITE,
                124,
                path="src/middleware.py",
            ),
            _event(
                EventType.ASSISTANT_RESPONSE,
                130,
                text="Middleware implemented; tests are not yet passing.",
            ),
            _request(138, 170_000, context_window=200_000),
        ]

    def test_build_checkpoint_contains_recoverable_state(self):
        markdown = build_checkpoint_markdown(self.events, "session-1", timestamp=138)

        self.assertIn("# Session checkpoint - session-1 - 0m 38s", markdown)
        self.assertIn("## Task\nImplement rate limiting.", markdown)
        self.assertIn("- Must use Redis.", markdown)
        self.assertIn("- Do not modify auth.py.", markdown)
        self.assertIn("src/middleware.py (read at 0m 12s)", markdown)
        self.assertIn("Chose token bucket instead of fixed window", markdown)
        self.assertIn("Modified files: src/middleware.py", markdown)

    def test_write_checkpoint_is_additive_sidecar(self):
        temporary, store, session_id = _store(self.events)
        self.addCleanup(temporary.cleanup)
        events_path = store._session_dir(session_id) / "events.ndjson"
        original_events = events_path.read_text()

        path = write_checkpoint(store, session_id)

        self.assertEqual(path, Path(temporary.name) / "checkpoints" / "session-1.md")
        self.assertTrue(path.exists())
        self.assertEqual(events_path.read_text(), original_events)

    def test_context_fill_ratio_reads_nested_usage(self):
        event = _event(
            EventType.LLM_RESPONSE,
            1,
            model="claude-sonnet-4",
            usage={"input_tokens": 160_000},
        )
        self.assertEqual(context_fill_ratio(event), 0.8)

    def test_watcher_writes_once_then_rearms_after_compaction(self):
        temporary, store, session_id = _store(self.events[:-1])
        self.addCleanup(temporary.cleanup)
        watcher = CompactionCheckpointWatcher(store, session_id, checkpoint_at=0.8)

        first = watcher.update(_request(138, 170_000, context_window=200_000))
        duplicate = watcher.update(_request(140, 175_000, context_window=200_000))
        compacted = watcher.update(_request(142, 10_000, context_window=200_000))
        second = watcher.update(_request(150, 165_000, context_window=200_000))

        self.assertIsNotNone(first)
        self.assertIsNone(duplicate)
        self.assertIsNone(compacted)
        self.assertEqual(second, first)

    def test_watcher_can_checkpoint_existing_saturated_context(self):
        temporary, store, session_id = _store(self.events)
        self.addCleanup(temporary.cleanup)
        watcher = CompactionCheckpointWatcher(store, session_id, checkpoint_at=0.8)

        self.assertIsNotNone(watcher.checkpoint_current())
        self.assertIsNone(watcher.checkpoint_current())

    def test_existing_checkpoint_uses_latest_exact_sample_from_each_stream(self):
        events = [
            _event(
                EventType.ASSISTANT_RESPONSE,
                1,
                context_stream="main",
                model="claude-sonnet-4",
                usage={"input_tokens": 180_000},
            ),
            _event(
                EventType.ASSISTANT_RESPONSE,
                2,
                context_stream="sidechain:review",
                model="claude-sonnet-4",
                usage={"input_tokens": 10_000},
            ),
        ]
        temporary, store, session_id = _store(events)
        self.addCleanup(temporary.cleanup)

        watcher = CompactionCheckpointWatcher(store, session_id, checkpoint_at=0.8)

        self.assertIsNotNone(watcher.checkpoint_current())

    def test_existing_checkpoint_ignores_estimated_request_payloads(self):
        events = [_event(
            EventType.LLM_REQUEST,
            1,
            model="claude-sonnet-4",
            prompt="x" * 800_000,
        )]
        temporary, store, session_id = _store(events)
        self.addCleanup(temporary.cleanup)

        watcher = CompactionCheckpointWatcher(store, session_id, checkpoint_at=0.8)

        self.assertIsNone(watcher.checkpoint_current())

    def test_watcher_processes_updates_without_rescanning_history(self):
        temporary, store, session_id = _store(self.events[:-1])
        self.addCleanup(temporary.cleanup)
        watcher = CompactionCheckpointWatcher(store, session_id, checkpoint_at=0.8)

        with patch(
            "agent_trace.compaction._request_samples",
            side_effect=AssertionError("watch update rescanned the session"),
        ):
            path = watcher.update(
                _request(138, 170_000, context_window=200_000)
            )

        self.assertIsNotNone(path)

    def test_watcher_uses_paired_response_usage_without_false_rearming(self):
        temporary, store, session_id = _store(self.events[:-1])
        self.addCleanup(temporary.cleanup)
        watcher = CompactionCheckpointWatcher(store, session_id, checkpoint_at=0.8)

        request = _event(
            EventType.LLM_REQUEST, 138, model="claude-sonnet-4", messages=[]
        )
        response = _event(
            EventType.LLM_RESPONSE,
            139,
            model="claude-sonnet-4",
            usage={"input_tokens": 170_000},
        )

        self.assertIsNone(watcher.update(request))
        self.assertIsNotNone(watcher.update(response))

    def test_watcher_pairs_interleaved_responses_with_their_stream(self):
        temporary, store, session_id = _store(self.events[:-1])
        self.addCleanup(temporary.cleanup)
        watcher = CompactionCheckpointWatcher(store, session_id, checkpoint_at=0.8)

        self.assertIsNone(watcher.update(_event(
            EventType.LLM_REQUEST, 138, context_stream="main", messages=[],
        )))
        self.assertIsNone(watcher.update(_event(
            EventType.LLM_REQUEST,
            139,
            context_stream="sidechain:review",
            messages=[],
        )))
        checkpoint = watcher.update(_event(
            EventType.LLM_RESPONSE,
            140,
            context_stream="main",
            model="claude-sonnet-4",
            usage={"input_tokens": 180_000},
        ))

        self.assertIsNotNone(checkpoint)
        self.assertEqual(watcher._last_tokens_by_stream["main"], 180_000)


class CompactionLintAndCommandTests(unittest.TestCase):
    def test_post_compaction_regression_lint_rule(self):
        events = [
            _read(1, "src/a.py"),
            _request(2, 180_000),
            _request(3, 10_000),
            _read(4, "src/a.py"),
        ]

        findings = _rule_post_compaction_regression(events, {})

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].rule, "post-compaction-regression")
        self.assertEqual(findings[0].line_start, 3)
        self.assertIn("re-read 1", findings[0].message)

    def test_lint_rule_does_not_flag_compaction_without_regression(self):
        events = [_request(1, 180_000), _request(2, 10_000)]
        self.assertEqual(_rule_post_compaction_regression(events, {}), [])

    def test_cmd_compaction_resolves_prefix_and_prints_report(self):
        temporary, _store_instance, _session_id = _store([
            _request(1, 180_000), _request(2, 10_000),
        ])
        self.addCleanup(temporary.cleanup)
        args = argparse.Namespace(
            trace_dir=temporary.name,
            session_id="session",
            compaction_threshold=0.5,
            diff=False,
            behavior_diff=False,
        )
        output = io.StringIO()

        with patch("sys.stdout", output):
            result = cmd_compaction(args)

        self.assertEqual(result, 0)
        self.assertIn("Compaction events - session session-1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
