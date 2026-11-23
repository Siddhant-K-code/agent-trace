"""Tests for named raw-score eval gates (GitHub issue #210)."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.eval.config import EvalConfig, EvalCriterionConfig, load_gate_config
from agent_trace.eval.runner import (
    cmd_eval_gate,
    format_gate_json,
    format_gate_table,
    load_gate_baseline,
    run_eval_gate,
    save_gate_baseline,
    write_gate_github_summary,
)
from agent_trace.eval.scorers import (
    MetricError,
    measure_context_fill_ratio,
    measure_cost_usd,
    measure_duration_seconds,
    measure_error_count,
    measure_lint_violations,
    measure_metric,
    measure_redundant_read_ratio,
    measure_session_status,
    measure_tool_call_count,
)
from agent_trace.cli import _normalise_eval_argv, build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace import session_outcome
from agent_trace.store import TraceStore


def custom_event_count(events):
    return len(events)


def _event(kind, timestamp, **data):
    return TraceEvent(event_type=kind, timestamp=timestamp, session_id="gate-session", data=data)


class GateTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.tmp.name, redact=False)
        self.meta = SessionMeta(
            session_id="gate-session",
            started_at=1_000.0,
            ended_at=1_042.0,
            total_duration_ms=42_000,
        )
        self.store.create_session(self.meta)
        self.events = [
            _event(EventType.SESSION_START, 1_000.0),
            _event(EventType.LLM_REQUEST, 1_001.0, messages=[{"role": "user", "content": "go"}]),
            _event(EventType.TOOL_CALL, 1_002.0, tool_name="Read", arguments={"file_path": "src/a.py"}),
            _event(EventType.TOOL_CALL, 1_003.0, tool_name="read_file", arguments={"path": "src/a.py"}),
            _event(EventType.FILE_READ, 1_004.0, path="src/b.py"),
            _event(EventType.TOOL_CALL, 1_005.0, tool_name="Write", arguments={"file_path": "out.txt"}),
            _event(EventType.ERROR, 1_006.0, message="one error"),
            _event(
                EventType.LLM_RESPONSE,
                1_041.0,
                usage={"input_tokens": 180_000, "output_tokens": 500},
                context_window=200_000,
            ),
            _event(EventType.SESSION_END, 1_042.0, exit_code=0),
        ]
        for event in self.events:
            self.store.append_event(self.meta.session_id, event)

    def tearDown(self):
        self.tmp.cleanup()


class TestGateConfig(unittest.TestCase):
    def test_loads_issue_example_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".agent-evals.yaml"
            path.write_text(
                "evals:\n"
                "  - name: cost-ceiling\n"
                "    scorer: cost_usd\n"
                "    threshold: 0.50\n"
                "    fail_on: above\n"
                "  - name: task-completed\n"
                "    scorer: session_status\n"
                "    expected: completed\n"
                "    fail_on: not_equal\n"
            )
            config = load_gate_config(path)
        self.assertEqual([item.name for item in config.evals], ["cost-ceiling", "task-completed"])
        self.assertEqual(config.evals[0].threshold, 0.5)
        self.assertEqual(config.evals[1].expected, "completed")

    def test_missing_eval_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("scorers:\n  - type: no_errors\n")
            with self.assertRaisesRegex(ValueError, "evals"):
                load_gate_config(path)

    def test_duplicate_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.yaml"
            path.write_text(
                "evals:\n"
                "  - name: same\n    scorer: error_count\n    threshold: 0\n    fail_on: above\n"
                "  - name: same\n    scorer: tool_call_count\n    threshold: 2\n    fail_on: above\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_gate_config(path)


class TestBuiltInMetrics(GateTestCase):
    def test_cost_usd(self):
        self.assertGreater(measure_cost_usd(self.store, self.meta.session_id), 0)

    def test_error_count(self):
        self.assertEqual(measure_error_count(self.events), 1)

    def test_session_status(self):
        self.assertEqual(measure_session_status(self.events, self.meta), "completed")
        timeout = [_event(EventType.SESSION_END, 1, exit_code=124)]
        self.assertEqual(measure_session_status(timeout), "timeout")
        self.assertEqual(measure_session_status([]), "killed")

    def test_redundant_read_ratio(self):
        self.assertAlmostEqual(measure_redundant_read_ratio(self.events), 1 / 3)

    def test_tool_call_count(self):
        self.assertEqual(measure_tool_call_count(self.events), 3)

    def test_context_fill_ratio(self):
        self.assertAlmostEqual(measure_context_fill_ratio(self.events), 0.9)

    def test_duration_seconds(self):
        self.assertEqual(measure_duration_seconds(self.events, self.meta), 42.0)

    def test_lint_violations(self):
        self.assertIsInstance(measure_lint_violations(self.store, self.meta.session_id), int)

    def test_lint_violations_is_not_a_count_when_a_rule_could_not_run(self):
        def broken(events, cfg):
            raise RuntimeError("boom")

        with patch.dict("agent_trace.lint._RULES", {"no-output": broken}):
            with self.assertRaisesRegex(MetricError, "^lint could not run rule\\(s\\): no-output$"):
                measure_lint_violations(self.store, self.meta.session_id)

    def test_custom_python_callable(self):
        value = measure_metric(
            "tests.test_eval_gate:custom_event_count",
            self.events,
            self.store,
            self.meta.session_id,
        )
        self.assertEqual(value, len(self.events))


def _ended(**data):
    return [_event(EventType.SESSION_END, 1.0, **data)]


class TestSessionStatus(unittest.TestCase):
    def test_copilot_error_and_abort_reasons_are_killed(self):
        for data in (
            {"reason": "error"},
            {"reason": "abort"},
            {"reason": "aborted"},
            {"reason": "Error"},
            {"stop_reason": "error_max_turns"},
            # The reason decides before the exit code.
            {"reason": "error", "exit_code": 124},
        ):
            with self.subTest(**data):
                self.assertEqual(measure_session_status(_ended(**data)), "killed")

    def test_complete_and_user_exit_reasons_are_completed(self):
        for reason in ("complete", "user_exit"):
            with self.subTest(reason=reason):
                self.assertEqual(measure_session_status(_ended(reason=reason)), "completed")

    def test_exit_code_too_large_for_a_float_is_killed(self):
        self.assertEqual(measure_session_status(_ended(exit_code=10**400)), "killed")

    def test_reads_more_than_the_recorded_status(self):
        meta = SessionMeta(session_id="gate-session", started_at=1.0, ended_at=2.0)
        self.assertEqual(measure_session_status([], meta), "completed")
        self.assertEqual(measure_session_status(_ended(exit_code="1")), "killed")
        self.assertEqual(measure_session_status(_ended(reason=["kill"])), "killed")
        self.assertEqual(measure_session_status(_ended(exit_code=True)), "completed")

    def test_a_resumed_session_has_not_ended(self):
        def resumed(**end):
            return [
                _event(EventType.SESSION_START, 1.0),
                _event(EventType.SESSION_END, 2.0, **end),
                _event(EventType.SESSION_START, 3.0, source="resume"),
            ]

        meta = SessionMeta(session_id="gate-session", started_at=1.0, ended_at=4.0)
        # The session_end before the resume does not decide.
        self.assertEqual(measure_session_status(resumed(reason="complete")), "killed")
        self.assertEqual(measure_session_status(resumed(reason="error"), meta), "completed")
        # A session_end after the resume decides.
        self.assertEqual(
            measure_session_status(
                resumed(reason="error") + [_event(EventType.SESSION_END, 4.0, reason="complete")]
            ),
            "completed",
        )
        self.assertEqual(
            measure_session_status(
                resumed(reason="complete") + [_event(EventType.SESSION_END, 4.0, reason="error")]
            ),
            "killed",
        )

    def test_matches_recorded_session_status(self):
        statuses = sorted(
            session_outcome._COMPLETED_STATUSES
            | session_outcome._TIMEOUT_STATUSES
            | session_outcome._TERMINATED_STATUSES
        )
        words = (
            *session_outcome._TIMEOUT_REASON_WORDS,
            *session_outcome._TERMINATED_REASON_WORDS,
            "complete",
            "user_exit",
        )
        rows = [{}]
        rows += [{"status": status} for status in statuses]
        rows += [{"session_status": f" {status.upper().replace('_', '-')} "} for status in statuses]
        rows += [
            {key: f"user {word}"}
            for key in session_outcome.SESSION_END_REASON_KEYS
            for word in words
        ]
        rows += [{"exit_code": code} for code in (0, 1, 124, 10**400, True, float("nan"))]
        rows += [{"status": 1}]
        rows += [
            {"status": "completed", "reason": "error", "exit_code": 1},
            {"status": "unknown", "reason": "abort", "exit_code": 0},
            {"reason": "timed out after an error"},
            {"reason": "complete", "exit_code": 124},
        ]
        for data in rows:
            events = _ended(**data)
            recorded = session_outcome.recorded_session_status(events)
            with self.subTest(data=data, recorded=recorded):
                self.assertEqual(
                    measure_session_status(events),
                    "killed" if recorded == "terminated" else recorded,
                )


class TestRunEvalGate(GateTestCase):
    def _config(self):
        return EvalConfig(evals=[
            EvalCriterionConfig("no-errors", "error_count", "above", threshold=0),
            EvalCriterionConfig("completed", "session_status", "not_equal", expected="completed"),
            EvalCriterionConfig("context", "context_fill_ratio", "above", threshold=0.95),
        ])

    def test_named_criteria_fail_and_pass(self):
        report = run_eval_gate(self.store, self.meta.session_id, self._config())
        self.assertFalse(report.overall_passed)
        self.assertEqual(report.failed, 1)
        self.assertFalse(report.results[0].passed)
        self.assertTrue(report.results[1].passed)

    def test_direction_aware_baseline_regression(self):
        config = EvalConfig(evals=[
            EvalCriterionConfig("calls", "tool_call_count", "above", threshold=100),
        ])
        report = run_eval_gate(
            self.store,
            self.meta.session_id,
            config,
            baseline={"calls": 2},
            tolerance=0.10,
        )
        self.assertTrue(report.results[0].criterion_passed)
        self.assertTrue(report.results[0].regressed)
        self.assertFalse(report.overall_passed)

    def test_tolerance_allows_small_regression(self):
        config = EvalConfig(evals=[
            EvalCriterionConfig("context", "context_fill_ratio", "above", threshold=1.0),
        ])
        report = run_eval_gate(
            self.store,
            self.meta.session_id,
            config,
            baseline={"context": 0.85},
            tolerance=0.10,
        )
        self.assertFalse(report.results[0].regressed)
        self.assertTrue(report.overall_passed)

    def test_unknown_scorer_is_a_failed_eval(self):
        config = EvalConfig(evals=[
            EvalCriterionConfig("unknown", "does_not_exist", "above", threshold=1),
        ])
        report = run_eval_gate(self.store, self.meta.session_id, config)
        self.assertFalse(report.overall_passed)
        self.assertIn("unknown scorer", report.results[0].reason)

    def test_table_and_json_output(self):
        report = run_eval_gate(self.store, self.meta.session_id, self._config())
        table = io.StringIO()
        format_gate_table(report, table)
        self.assertIn("Eval results — session gate-session", table.getvalue())
        self.assertIn("Overall: FAIL", table.getvalue())
        machine = io.StringIO()
        format_gate_json(report, machine)
        data = json.loads(machine.getvalue())
        self.assertEqual(data["session_id"], "gate-session")
        self.assertEqual(data["fail_count"], 1)

    def test_baseline_round_trip(self):
        report = run_eval_gate(self.store, self.meta.session_id, self._config())
        path = Path(self.tmp.name) / "baseline.json"
        save_gate_baseline(path, report)
        loaded = load_gate_baseline(path)
        self.assertEqual(loaded["no-errors"], 1)
        self.assertEqual(loaded["completed"], "completed")

    def test_github_summary_markdown(self):
        report = run_eval_gate(self.store, self.meta.session_id, self._config())
        path = Path(self.tmp.name) / "summary.md"
        write_gate_github_summary(report, path)
        summary = path.read_text()
        self.assertIn("| Eval | Scorer | Score | Threshold | Baseline | Result |", summary)
        self.assertIn("**Overall: FAIL**", summary)


class TestGateCommand(GateTestCase):
    def test_direct_cli_form_routes_to_gate(self):
        argv = _normalise_eval_argv(["eval", "gate-session", "--format", "json"])
        args = build_parser().parse_args(argv)
        self.assertEqual(args.eval_command, "gate")
        self.assertEqual(args.session_id, "gate-session")
        self.assertEqual(args.format, "json")

    def test_legacy_eval_subcommand_is_preserved(self):
        self.assertEqual(
            _normalise_eval_argv(["eval", "ci"]),
            ["eval", "ci"],
        )

    def test_trace_directory_named_eval_is_not_a_command(self):
        self.assertEqual(
            _normalise_eval_argv(["--trace-dir", "eval", "cost"]),
            ["--trace-dir", "eval", "cost"],
        )

    def test_exit_one_on_failed_eval_and_json_output(self):
        config_path = Path(self.tmp.name) / ".agent-evals.yaml"
        config_path.write_text(
            "evals:\n"
            "  - name: no-errors\n"
            "    scorer: error_count\n"
            "    threshold: 0\n"
            "    fail_on: above\n"
        )
        args = argparse.Namespace(
            trace_dir=self.tmp.name,
            session_id=self.meta.session_id,
            config=str(config_path),
            format="json",
            baseline=None,
            save_baseline=None,
            tolerance=0.0,
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cmd_eval_gate(args)
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(stdout.getvalue())["passed"])

    def test_exit_code_too_large_for_a_float_passes_expected_killed(self):
        meta = SessionMeta(session_id="huge-exit", started_at=1_000.0, ended_at=1_001.0)
        self.store.create_session(meta)
        self.store.append_event(
            meta.session_id,
            TraceEvent(
                event_type=EventType.SESSION_END,
                timestamp=1_001.0,
                session_id=meta.session_id,
                data={"exit_code": 10**400},
            ),
        )
        config_path = Path(self.tmp.name) / ".agent-evals.yaml"
        config_path.write_text(
            "evals:\n"
            "  - name: was-killed\n"
            "    scorer: session_status\n"
            "    expected: killed\n"
            "    fail_on: not_equal\n"
        )
        args = argparse.Namespace(
            trace_dir=self.tmp.name,
            session_id=meta.session_id,
            config=str(config_path),
            format="json",
            baseline=None,
            save_baseline=None,
            tolerance=0.0,
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cmd_eval_gate(args)
        results = json.loads(stdout.getvalue())["results"]
        self.assertEqual(code, 0)
        self.assertEqual([(item["score"], item["passed"]) for item in results], [("killed", True)])
        self.assertNotIn("int too large", stdout.getvalue() + stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
