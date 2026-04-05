"""Tests for the eval framework (scorers, dataset, runner, config, CI)."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.eval.scorers import (
    ScoreResult,
    score_no_errors,
    score_regex,
    score_cost_under,
    score_files_scoped,
    score_duration_under,
    score_custom,
    run_scorer,
)
from agent_trace.eval.config import EvalConfig, ScorerConfig, load_config
from agent_trace.eval.dataset import DatasetEntry, add_entry, list_entries, export_entries
from agent_trace.eval.runner import EvalReport, run_eval, format_report_table, format_report_json, format_compare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: EventType, ts: float, session_id: str = "s1", **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)


def _make_store(events: list[TraceEvent], session_id: str = "s1") -> tuple[TraceStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = TraceStore(tmp.name)
    meta = SessionMeta(
        session_id=session_id,
        started_at=events[0].timestamp if events else 0.0,
        total_duration_ms=(events[-1].timestamp - events[0].timestamp) * 1000 if len(events) > 1 else 0,
    )
    store.create_session(meta)
    for e in events:
        store.append_event(session_id, e)
    store.update_meta(meta)
    return store, tmp


def _ok_events(session_id: str = "s1") -> list[TraceEvent]:
    return [
        _make_event(EventType.USER_PROMPT, 0.0, session_id, prompt="run tests"),
        _make_event(EventType.TOOL_CALL, 1.0, session_id,
                    tool_name="Bash", arguments={"command": "pytest"}),
        _make_event(EventType.TOOL_RESULT, 2.0, session_id, result="5 passed"),
        _make_event(EventType.ASSISTANT_RESPONSE, 3.0, session_id, text="All tests passed."),
        _make_event(EventType.SESSION_END, 4.0, session_id, exit_code=0),
    ]


def _error_events(session_id: str = "s1") -> list[TraceEvent]:
    return [
        _make_event(EventType.USER_PROMPT, 0.0, session_id, prompt="run tests"),
        _make_event(EventType.TOOL_CALL, 1.0, session_id,
                    tool_name="Bash", arguments={"command": "pytest"}),
        _make_event(EventType.ERROR, 2.0, session_id, message="exit 1"),
        _make_event(EventType.SESSION_END, 3.0, session_id, exit_code=1),
    ]


# ---------------------------------------------------------------------------
# Scorer: no_errors
# ---------------------------------------------------------------------------

class TestScoreNoErrors(unittest.TestCase):
    def test_passes_with_no_errors(self):
        events = _ok_events()
        result = score_no_errors(events)
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.passed)

    def test_fails_with_error_event(self):
        events = _error_events()
        result = score_no_errors(events)
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.passed)

    def test_scorer_name(self):
        result = score_no_errors([])
        self.assertEqual(result.scorer, "no_errors")

    def test_custom_threshold(self):
        events = _error_events()
        result = score_no_errors(events, threshold=0.0)
        self.assertTrue(result.passed)  # threshold=0 always passes


# ---------------------------------------------------------------------------
# Scorer: regex
# ---------------------------------------------------------------------------

class TestScoreRegex(unittest.TestCase):
    def test_matches_pattern(self):
        events = _ok_events()
        result = score_regex(events, pattern=r"tests passed", event_type="assistant_response")
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.passed)

    def test_no_match(self):
        events = _ok_events()
        result = score_regex(events, pattern=r"NEVER_MATCHES_XYZ")
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.passed)

    def test_invalid_event_type(self):
        result = score_regex([], pattern=r"x", event_type="not_a_real_type")
        self.assertFalse(result.passed)
        self.assertIn("unknown event_type", result.reason)

    def test_matches_tool_result(self):
        events = _ok_events()
        result = score_regex(events, pattern=r"5 passed", event_type="tool_result")
        self.assertTrue(result.passed)


# ---------------------------------------------------------------------------
# Scorer: cost_under
# ---------------------------------------------------------------------------

class TestScoreCostUnder(unittest.TestCase):
    def setUp(self):
        events = _ok_events("cost1")
        self.store, self.tmp = _make_store(events, "cost1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_passes_with_high_limit(self):
        result = score_cost_under(self.store, "cost1", max_dollars=100.0)
        self.assertTrue(result.passed)

    def test_fails_with_zero_limit(self):
        result = score_cost_under(self.store, "cost1", max_dollars=0.0)
        self.assertFalse(result.passed)

    def test_score_proportional_when_over(self):
        result = score_cost_under(self.store, "cost1", max_dollars=0.0)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)


# ---------------------------------------------------------------------------
# Scorer: files_scoped
# ---------------------------------------------------------------------------

class TestScoreFilesScoped(unittest.TestCase):
    def _write_events(self, paths: list[str]) -> list[TraceEvent]:
        return [
            _make_event(EventType.TOOL_CALL, float(i), "s1",
                        tool_name="Write", arguments={"file_path": p})
            for i, p in enumerate(paths)
        ]

    def test_all_in_scope(self):
        events = self._write_events(["src/main.py", "src/utils.py"])
        result = score_files_scoped(events, allowed_paths=["src/"])
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.passed)

    def test_out_of_scope(self):
        events = self._write_events(["src/main.py", "/etc/passwd"])
        result = score_files_scoped(events, allowed_paths=["src/"])
        self.assertFalse(result.passed)

    def test_no_restrictions(self):
        events = self._write_events(["anywhere/file.py"])
        result = score_files_scoped(events, allowed_paths=[])
        self.assertTrue(result.passed)

    def test_no_file_events(self):
        events = _ok_events()
        result = score_files_scoped(events, allowed_paths=["src/"])
        self.assertTrue(result.passed)


# ---------------------------------------------------------------------------
# Scorer: duration_under
# ---------------------------------------------------------------------------

class TestScoreDurationUnder(unittest.TestCase):
    def test_passes_within_limit(self):
        events = _ok_events()  # 4 seconds
        result = score_duration_under(events, max_seconds=10.0)
        self.assertTrue(result.passed)

    def test_fails_over_limit(self):
        events = _ok_events()  # 4 seconds
        result = score_duration_under(events, max_seconds=1.0)
        self.assertFalse(result.passed)

    def test_single_event(self):
        events = [_make_event(EventType.USER_PROMPT, 0.0)]
        result = score_duration_under(events, max_seconds=10.0)
        self.assertTrue(result.passed)


# ---------------------------------------------------------------------------
# Scorer: custom
# ---------------------------------------------------------------------------

class TestScoreCustom(unittest.TestCase):
    def test_custom_scorer_called(self):
        events = _ok_events()
        result = score_custom(events, fn=lambda evts: 0.75, name="my_scorer")
        self.assertAlmostEqual(result.score, 0.75)
        self.assertEqual(result.scorer, "my_scorer")

    def test_custom_scorer_exception_returns_zero(self):
        def bad_scorer(evts):
            raise ValueError("oops")
        result = score_custom([], fn=bad_scorer)
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.passed)

    def test_score_clamped_to_one(self):
        result = score_custom([], fn=lambda evts: 999.0)
        self.assertEqual(result.score, 1.0)

    def test_score_clamped_to_zero(self):
        result = score_custom([], fn=lambda evts: -5.0)
        self.assertEqual(result.score, 0.0)


# ---------------------------------------------------------------------------
# run_scorer dispatcher
# ---------------------------------------------------------------------------

class TestRunScorer(unittest.TestCase):
    def test_dispatches_no_errors(self):
        events = _ok_events()
        result = run_scorer("no_errors", {}, events)
        self.assertEqual(result.scorer, "no_errors")

    def test_dispatches_regex(self):
        events = _ok_events()
        result = run_scorer("regex", {"pattern": "passed", "event_type": "assistant_response"}, events)
        self.assertTrue(result.passed)

    def test_unknown_scorer(self):
        result = run_scorer("nonexistent_scorer", {}, [])
        self.assertFalse(result.passed)
        self.assertIn("unknown scorer", result.reason)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestEvalConfig(unittest.TestCase):
    def test_default_config(self):
        config = EvalConfig.default()
        self.assertGreater(len(config.scorers), 0)
        self.assertEqual(config.scorers[0].type, "no_errors")

    def test_load_missing_file_returns_default(self):
        config = load_config("/nonexistent/path/.agent-evals.yaml")
        self.assertIsInstance(config, EvalConfig)
        self.assertGreater(len(config.scorers), 0)

    def test_load_valid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "scorers:\n"
                "  - type: no_errors\n"
                "    threshold: 1.0\n"
                "  - type: duration_under\n"
                "    max_seconds: 60\n"
                "    threshold: 0.8\n"
                "thresholds:\n"
                "  pass: 0.9\n"
                "  warn: 0.6\n"
            )
            path = f.name
        try:
            config = load_config(path)
            self.assertEqual(len(config.scorers), 2)
            self.assertEqual(config.scorers[0].type, "no_errors")
            self.assertEqual(config.scorers[1].type, "duration_under")
            self.assertAlmostEqual(config.pass_threshold, 0.9)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset_path = os.path.join(self.tmp.name, "test.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_and_list(self):
        entry = DatasetEntry(session_id="abc123", label="test run")
        add_entry(self.dataset_path, entry)
        entries = list_entries(self.dataset_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].session_id, "abc123")
        self.assertEqual(entries[0].label, "test run")

    def test_multiple_entries(self):
        for i in range(3):
            add_entry(self.dataset_path, DatasetEntry(session_id=f"sess{i}"))
        entries = list_entries(self.dataset_path)
        self.assertEqual(len(entries), 3)

    def test_empty_dataset(self):
        entries = list_entries(self.dataset_path)
        self.assertEqual(entries, [])

    def test_export_entries(self):
        add_entry(self.dataset_path, DatasetEntry(session_id="xyz", label="export test"))
        buf = io.StringIO()
        export_entries(self.dataset_path, out=buf)
        output = buf.getvalue()
        self.assertIn("xyz", output)
        # Should be valid JSONL
        for line in output.strip().splitlines():
            data = json.loads(line)
            self.assertIn("session_id", data)

    def test_entry_roundtrip(self):
        entry = DatasetEntry(session_id="round1", label="roundtrip")
        add_entry(self.dataset_path, entry)
        loaded = list_entries(self.dataset_path)[0]
        self.assertEqual(loaded.session_id, entry.session_id)
        self.assertEqual(loaded.label, entry.label)
        self.assertEqual(loaded.entry_id, entry.entry_id)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TestRunEval(unittest.TestCase):
    def setUp(self):
        events = _ok_events("eval1")
        self.store, self.tmp = _make_store(events, "eval1")
        self.config = EvalConfig(
            scorers=[
                ScorerConfig(type="no_errors", threshold=1.0),
                ScorerConfig(type="duration_under", threshold=1.0, params={"max_seconds": 10.0}),
            ]
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_eval_report(self):
        report = run_eval(self.store, "eval1", self.config)
        self.assertIsInstance(report, EvalReport)

    def test_all_pass_for_ok_session(self):
        report = run_eval(self.store, "eval1", self.config)
        self.assertTrue(report.overall_passed)
        self.assertEqual(report.failed, 0)

    def test_fails_for_error_session(self):
        events = _error_events("eval2")
        store2, tmp2 = _make_store(events, "eval2")
        try:
            config = EvalConfig(scorers=[ScorerConfig(type="no_errors", threshold=1.0)])
            report = run_eval(store2, "eval2", config)
            self.assertFalse(report.overall_passed)
            self.assertGreater(report.failed, 0)
        finally:
            tmp2.cleanup()

    def test_weighted_score_between_zero_and_one(self):
        report = run_eval(self.store, "eval1", self.config)
        self.assertGreaterEqual(report.weighted_score, 0.0)
        self.assertLessEqual(report.weighted_score, 1.0)


class TestFormatReportTable(unittest.TestCase):
    def setUp(self):
        events = _ok_events("fmt1")
        self.store, self.tmp = _make_store(events, "fmt1")
        self.config = EvalConfig(scorers=[ScorerConfig(type="no_errors", threshold=1.0)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_table_contains_session_id(self):
        report = run_eval(self.store, "fmt1", self.config)
        buf = io.StringIO()
        format_report_table(report, out=buf)
        self.assertIn("fmt1", buf.getvalue())

    def test_table_contains_scorer_name(self):
        report = run_eval(self.store, "fmt1", self.config)
        buf = io.StringIO()
        format_report_table(report, out=buf)
        self.assertIn("no_errors", buf.getvalue())

    def test_json_format_valid(self):
        report = run_eval(self.store, "fmt1", self.config)
        buf = io.StringIO()
        format_report_json(report, out=buf)
        data = json.loads(buf.getvalue())
        self.assertIn("session_id", data)
        self.assertIn("results", data)
        self.assertIn("passed", data)


class TestFormatCompare(unittest.TestCase):
    def setUp(self):
        events_a = _ok_events("cmp1")
        events_b = _error_events("cmp2")
        self.store_a, self.tmp_a = _make_store(events_a, "cmp1")
        self.store_b, self.tmp_b = _make_store(events_b, "cmp2")

    def tearDown(self):
        self.tmp_a.cleanup()
        self.tmp_b.cleanup()

    def test_compare_output_contains_both_sessions(self):
        config = EvalConfig(scorers=[ScorerConfig(type="no_errors", threshold=1.0)])
        report_a = run_eval(self.store_a, "cmp1", config)
        report_b = run_eval(self.store_b, "cmp2", config)
        buf = io.StringIO()
        format_compare(report_a, report_b, out=buf)
        output = buf.getvalue()
        self.assertIn("cmp1", output)
        self.assertIn("cmp2", output)
        self.assertIn("no_errors", output)


# ---------------------------------------------------------------------------
# CI exit code logic
# ---------------------------------------------------------------------------

class TestCIExitCode(unittest.TestCase):
    def test_all_pass_returns_zero(self):
        events = _ok_events("ci1")
        store, tmp = _make_store(events, "ci1")
        try:
            config = EvalConfig(scorers=[ScorerConfig(type="no_errors", threshold=1.0)])
            report = run_eval(store, "ci1", config)
            self.assertEqual(0 if report.overall_passed else 1, 0)
        finally:
            tmp.cleanup()

    def test_failure_returns_one(self):
        events = _error_events("ci2")
        store, tmp = _make_store(events, "ci2")
        try:
            config = EvalConfig(scorers=[ScorerConfig(type="no_errors", threshold=1.0)])
            report = run_eval(store, "ci2", config)
            self.assertEqual(0 if report.overall_passed else 1, 1)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
