"""Tests for agent-strace lint (Issue #116)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_trace.lint import (
    DEFAULT_CONFIG,
    LintLevel,
    LintReport,
    LintResult,
    _rule_budget_proximity,
    _rule_context_saturation,
    _rule_error_retry_loop,
    _rule_no_output,
    _rule_reasoning_spiral,
    _rule_redundant_read,
    _rule_tool_loop,
    _load_config,
    lint_session,
    format_report,
    format_report_json,
    cmd_lint,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(event_type: EventType, **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=1.0, session_id="s1", data=data)


def _tool_call(tool: str, **kwargs) -> TraceEvent:
    return _ev(EventType.TOOL_CALL, tool_name=tool, arguments=kwargs)


def _llm_response() -> TraceEvent:
    return _ev(EventType.LLM_RESPONSE, content="thinking...")


def _llm_request() -> TraceEvent:
    return _ev(EventType.LLM_REQUEST, prompt="do something")


def _error(msg: str = "fail") -> TraceEvent:
    return _ev(EventType.ERROR, message=msg)


def _tool_result(is_error: bool = False) -> TraceEvent:
    return _ev(EventType.TOOL_RESULT, is_error=is_error)


def _session_end() -> TraceEvent:
    return _ev(EventType.SESSION_END)


def _make_store_with_events(events: list[TraceEvent]) -> tuple[TraceStore, str]:
    tmp = tempfile.mkdtemp()
    store = TraceStore(Path(tmp))
    meta = SessionMeta(agent_name="test", command="test")
    session_path = store.create_session(meta)
    session_id = session_path.name
    for e in events:
        e.session_id = session_id
        store.append_event(session_id, e)
    return store, session_id


# ---------------------------------------------------------------------------
# Rule: tool-loop
# ---------------------------------------------------------------------------

class TestRuleToolLoop(unittest.TestCase):
    def _cfg(self, threshold=5):
        return {"enabled": True, "level": LintLevel.WARN, "threshold": threshold}

    def test_triggers_on_consecutive_calls(self):
        events = [_tool_call("Bash")] * 6
        results = _rule_tool_loop(events, self._cfg(threshold=5))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].rule, "tool-loop")
        self.assertEqual(results[0].level, LintLevel.WARN)
        self.assertIn("Bash", results[0].message)
        self.assertIn("6", results[0].message)

    def test_no_trigger_below_threshold(self):
        events = [_tool_call("Bash")] * 4
        results = _rule_tool_loop(events, self._cfg(threshold=5))
        self.assertEqual(results, [])

    def test_resets_on_different_tool(self):
        events = [_tool_call("Bash")] * 4 + [_tool_call("Read")] + [_tool_call("Bash")] * 4
        results = _rule_tool_loop(events, self._cfg(threshold=5))
        self.assertEqual(results, [])

    def test_triggers_exactly_at_threshold(self):
        events = [_tool_call("Write")] * 5
        results = _rule_tool_loop(events, self._cfg(threshold=5))
        self.assertEqual(len(results), 1)

    def test_non_tool_call_breaks_run(self):
        events = [_tool_call("Bash")] * 3 + [_llm_response()] + [_tool_call("Bash")] * 3
        results = _rule_tool_loop(events, self._cfg(threshold=5))
        self.assertEqual(results, [])

    def test_line_numbers_set(self):
        events = [_tool_call("Bash")] * 6
        results = _rule_tool_loop(events, self._cfg(threshold=5))
        self.assertEqual(results[0].line_start, 1)
        self.assertEqual(results[0].line_end, 6)

    def test_multiple_runs_reported(self):
        events = [_tool_call("Bash")] * 6 + [_llm_response()] + [_tool_call("Read")] * 6
        results = _rule_tool_loop(events, self._cfg(threshold=5))
        self.assertEqual(len(results), 2)


# ---------------------------------------------------------------------------
# Rule: reasoning-spiral
# ---------------------------------------------------------------------------

class TestRuleReasoningSpiral(unittest.TestCase):
    def _cfg(self, threshold=3):
        return {"enabled": True, "level": LintLevel.WARN, "threshold": threshold}

    def test_triggers_on_consecutive_llm_calls(self):
        events = [_llm_request(), _llm_response(), _llm_request(), _llm_response()]
        results = _rule_reasoning_spiral(events, self._cfg(threshold=3))
        self.assertEqual(len(results), 1)
        self.assertIn("4", results[0].message)

    def test_no_trigger_below_threshold(self):
        events = [_llm_request(), _llm_response()]
        results = _rule_reasoning_spiral(events, self._cfg(threshold=3))
        self.assertEqual(results, [])

    def test_tool_call_resets_run(self):
        # 2 LLM calls, then a tool call, then 2 more — neither run reaches threshold=3
        events = [_llm_request(), _llm_response(),
                  _tool_call("Bash"),
                  _llm_request(), _llm_response()]
        results = _rule_reasoning_spiral(events, self._cfg(threshold=3))
        self.assertEqual(results, [])

    def test_triggers_at_end_of_stream(self):
        events = [_llm_request()] * 4
        results = _rule_reasoning_spiral(events, self._cfg(threshold=3))
        self.assertEqual(len(results), 1)


# ---------------------------------------------------------------------------
# Rule: budget-proximity
# ---------------------------------------------------------------------------

class TestRuleBudgetProximity(unittest.TestCase):
    def _cfg(self, threshold=0.90):
        return {"enabled": True, "level": LintLevel.ERROR, "threshold": threshold}

    def test_no_trigger_without_budget(self):
        events = [_llm_request()] * 10
        results = _rule_budget_proximity(events, self._cfg())
        self.assertEqual(results, [])

    def test_triggers_when_near_budget(self):
        # SESSION_START with a tiny budget so estimated cost exceeds 90%
        start = TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=1.0,
            session_id="s1",
            data={"budget_dollars": 0.000001},  # tiny budget
        )
        # Add many LLM events to push estimated cost over threshold
        events = [start] + [_llm_request()] * 100
        results = _rule_budget_proximity(events, self._cfg())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].level, LintLevel.ERROR)

    def test_no_trigger_well_below_budget(self):
        start = TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=1.0,
            session_id="s1",
            data={"budget_dollars": 1000.0},  # huge budget
        )
        events = [start, _llm_request()]
        results = _rule_budget_proximity(events, self._cfg())
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# Rule: context-saturation
# ---------------------------------------------------------------------------

class TestRuleContextSaturation(unittest.TestCase):
    def _cfg(self, threshold=0.80):
        return {"enabled": True, "level": LintLevel.INFO, "threshold": threshold}

    def test_no_trigger_on_small_session(self):
        events = [_llm_request()] * 5
        results = _rule_context_saturation(events, self._cfg())
        self.assertEqual(results, [])

    def test_triggers_on_large_input(self):
        # Create a very large LLM request to exceed 80% of 200k context
        big_data = "x" * (200_000 * 4 * 4)  # way over threshold
        ev = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            timestamp=1.0,
            session_id="s1",
            data={"prompt": big_data},
        )
        results = _rule_context_saturation([ev], self._cfg())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].level, LintLevel.INFO)

    def test_reports_only_once(self):
        big_data = "x" * (200_000 * 4 * 4)
        events = [
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=1.0,
                       session_id="s1", data={"prompt": big_data}),
            TraceEvent(event_type=EventType.LLM_REQUEST, timestamp=2.0,
                       session_id="s1", data={"prompt": big_data}),
        ]
        results = _rule_context_saturation(events, self._cfg())
        self.assertEqual(len(results), 1)


# ---------------------------------------------------------------------------
# Rule: redundant-read
# ---------------------------------------------------------------------------

class TestRuleRedundantRead(unittest.TestCase):
    def _cfg(self, threshold=3):
        return {"enabled": True, "level": LintLevel.INFO, "threshold": threshold}

    def test_triggers_on_repeated_reads(self):
        events = [
            _tool_call("Read", file_path="README.md"),
            _tool_call("Read", file_path="README.md"),
            _tool_call("Read", file_path="README.md"),
        ]
        results = _rule_redundant_read(events, self._cfg())
        self.assertEqual(len(results), 1)
        self.assertIn("README.md", results[0].message)
        self.assertIn("3", results[0].message)

    def test_no_trigger_below_threshold(self):
        events = [
            _tool_call("Read", file_path="README.md"),
            _tool_call("Read", file_path="README.md"),
        ]
        results = _rule_redundant_read(events, self._cfg())
        self.assertEqual(results, [])

    def test_different_files_not_counted_together(self):
        events = [
            _tool_call("Read", file_path="a.py"),
            _tool_call("Read", file_path="b.py"),
            _tool_call("Read", file_path="c.py"),
        ]
        results = _rule_redundant_read(events, self._cfg())
        self.assertEqual(results, [])

    def test_multiple_files_each_reported(self):
        events = (
            [_tool_call("Read", file_path="a.py")] * 3
            + [_tool_call("Read", file_path="b.py")] * 3
        )
        results = _rule_redundant_read(events, self._cfg())
        self.assertEqual(len(results), 2)


# ---------------------------------------------------------------------------
# Rule: error-retry-loop
# ---------------------------------------------------------------------------

class TestRuleErrorRetryLoop(unittest.TestCase):
    def _cfg(self, threshold=3):
        return {"enabled": True, "level": LintLevel.WARN, "threshold": threshold}

    def test_triggers_on_repeated_errors(self):
        events = [
            _tool_call("Bash"),
            _error("exit 1"),
            _tool_call("Bash"),
            _error("exit 1"),
            _tool_call("Bash"),
            _error("exit 1"),
        ]
        results = _rule_error_retry_loop(events, self._cfg())
        self.assertEqual(len(results), 1)
        self.assertIn("Bash", results[0].message)

    def test_no_trigger_below_threshold(self):
        events = [
            _tool_call("Bash"),
            _error("exit 1"),
            _tool_call("Bash"),
            _error("exit 1"),
        ]
        results = _rule_error_retry_loop(events, self._cfg())
        self.assertEqual(results, [])

    def test_tool_result_is_error_counts(self):
        events = [
            _tool_call("Write"),
            _tool_result(is_error=True),
            _tool_call("Write"),
            _tool_result(is_error=True),
            _tool_call("Write"),
            _tool_result(is_error=True),
        ]
        results = _rule_error_retry_loop(events, self._cfg())
        self.assertEqual(len(results), 1)

    def test_different_tools_counted_separately(self):
        events = [
            _tool_call("Bash"), _error(),
            _tool_call("Read"), _error(),
            _tool_call("Write"), _error(),
        ]
        results = _rule_error_retry_loop(events, self._cfg())
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# Rule: no-output
# ---------------------------------------------------------------------------

class TestRuleNoOutput(unittest.TestCase):
    def _cfg(self):
        return {"enabled": True, "level": LintLevel.WARN}

    def test_triggers_when_no_writes(self):
        events = [_tool_call("Read", file_path="a.py"), _session_end()]
        results = _rule_no_output(events, self._cfg())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].level, LintLevel.WARN)

    def test_no_trigger_with_write_tool(self):
        events = [_tool_call("Write", file_path="out.py"), _session_end()]
        results = _rule_no_output(events, self._cfg())
        self.assertEqual(results, [])

    def test_no_trigger_with_edit_tool(self):
        events = [_tool_call("edit", file_path="a.py"), _session_end()]
        results = _rule_no_output(events, self._cfg())
        self.assertEqual(results, [])

    def test_no_trigger_without_session_end(self):
        # Session still in progress — don't flag
        events = [_tool_call("Read", file_path="a.py")]
        results = _rule_no_output(events, self._cfg())
        self.assertEqual(results, [])

    def test_file_write_event_counts(self):
        events = [
            _ev(EventType.FILE_WRITE, path="out.txt"),
            _session_end(),
        ]
        results = _rule_no_output(events, self._cfg())
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig(unittest.TestCase):
    def test_defaults_returned_when_no_file(self):
        config = _load_config(None)
        self.assertIn("tool-loop", config)
        self.assertTrue(config["tool-loop"]["enabled"])

    def test_override_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"tool-loop": {"threshold": 99}}, f)
            path = f.name
        config = _load_config(path)
        self.assertEqual(config["tool-loop"]["threshold"], 99)
        # Other defaults preserved
        self.assertTrue(config["tool-loop"]["enabled"])

    def test_disable_rule_via_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"tool-loop": {"enabled": False}}, f)
            path = f.name
        config = _load_config(path)
        self.assertFalse(config["tool-loop"]["enabled"])

    def test_bad_config_file_falls_back_to_defaults(self):
        config = _load_config("/nonexistent/path.json")
        self.assertIn("tool-loop", config)


# ---------------------------------------------------------------------------
# lint_session integration
# ---------------------------------------------------------------------------

class TestLintSession(unittest.TestCase):
    def test_clean_session_no_findings(self):
        events = [
            _tool_call("Read", file_path="a.py"),
            _tool_call("Write", file_path="b.py"),
            _session_end(),
        ]
        store, sid = _make_store_with_events(events)
        report = lint_session(store, sid)
        self.assertIsInstance(report, LintReport)
        self.assertEqual(report.session_id, sid)
        # no-output should not fire (Write present), no loops
        self.assertEqual(report.errors, 0)
        self.assertEqual(report.warnings, 0)

    def test_tool_loop_detected(self):
        events = [_tool_call("Bash")] * 7 + [_session_end()]
        store, sid = _make_store_with_events(events)
        report = lint_session(store, sid)
        rules_fired = {f.rule for f in report.findings}
        self.assertIn("tool-loop", rules_fired)

    def test_no_output_detected(self):
        events = [_tool_call("Read", file_path="a.py"), _session_end()]
        store, sid = _make_store_with_events(events)
        report = lint_session(store, sid)
        rules_fired = {f.rule for f in report.findings}
        self.assertIn("no-output", rules_fired)

    def test_disabled_rule_not_run(self):
        events = [_tool_call("Bash")] * 7 + [_session_end()]
        store, sid = _make_store_with_events(events)
        config = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
        config["tool-loop"]["enabled"] = False
        report = lint_session(store, sid, config=config)
        rules_fired = {f.rule for f in report.findings}
        self.assertNotIn("tool-loop", rules_fired)

    def test_rule_failure_does_not_prevent_others(self):
        """A rule that raises must not stop other rules from running."""
        from agent_trace import lint as lint_mod
        original = lint_mod._RULES.get("tool-loop")
        try:
            lint_mod._RULES["tool-loop"] = lambda events, cfg: (_ for _ in ()).throw(RuntimeError("boom"))
            events = [_tool_call("Read", file_path="a.py"), _session_end()]
            store, sid = _make_store_with_events(events)
            report = lint_session(store, sid)
            # no-output should still fire
            rules_fired = {f.rule for f in report.findings}
            self.assertIn("no-output", rules_fired)
        finally:
            if original:
                lint_mod._RULES["tool-loop"] = original


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatReport(unittest.TestCase):
    def test_clean_report_message(self):
        import io
        report = LintReport(session_id="abc123", findings=[])
        out = io.StringIO()
        format_report(report, out)
        self.assertIn("No issues found", out.getvalue())

    def test_findings_printed(self):
        import io
        report = LintReport(session_id="abc123", findings=[
            LintResult(rule="tool-loop", level=LintLevel.WARN, message="Bash looped"),
            LintResult(rule="no-output", level=LintLevel.WARN, message="No writes"),
        ])
        out = io.StringIO()
        format_report(report, out)
        text = out.getvalue()
        self.assertIn("tool-loop", text)
        self.assertIn("no-output", text)
        self.assertIn("WARN", text)

    def test_json_format(self):
        report = LintReport(session_id="abc123", findings=[
            LintResult(rule="tool-loop", level=LintLevel.ERROR, message="loop"),
        ])
        data = json.loads(format_report_json(report))
        self.assertEqual(data["session_id"], "abc123")
        self.assertEqual(data["errors"], 1)
        self.assertEqual(len(data["findings"]), 1)
        self.assertEqual(data["findings"][0]["rule"], "tool-loop")


# ---------------------------------------------------------------------------
# CLI: cmd_lint
# ---------------------------------------------------------------------------

class TestCmdLint(unittest.TestCase):
    def _make_args(self, store_dir, session_id=None, strict=False,
                   fmt="text", lint_all=False, since=None, config=None):
        import argparse
        args = argparse.Namespace()
        args.trace_dir = store_dir
        args.session_id = session_id
        args.strict = strict
        args.format = fmt
        args.all = lint_all
        args.since = since
        args.config = config
        return args

    def test_returns_0_on_clean_session(self):
        events = [_tool_call("Write", file_path="a.py"), _session_end()]
        store, sid = _make_store_with_events(events)
        args = self._make_args(store.base_dir, session_id=sid)
        result = cmd_lint(args)
        self.assertEqual(result, 0)

    def test_returns_1_on_error_finding(self):
        # Budget proximity triggers ERROR when budget is tiny
        start = TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=1.0,
            session_id="s1",
            data={"budget_dollars": 0.000001},
        )
        events = [start] + [_llm_request()] * 100 + [_session_end()]
        store, sid = _make_store_with_events(events)
        args = self._make_args(store.base_dir, session_id=sid)
        result = cmd_lint(args)
        self.assertEqual(result, 1)

    def test_strict_returns_1_on_warning(self):
        events = [_tool_call("Bash")] * 7 + [_session_end()]
        store, sid = _make_store_with_events(events)
        args = self._make_args(store.base_dir, session_id=sid, strict=True)
        result = cmd_lint(args)
        self.assertEqual(result, 1)

    def test_non_strict_returns_0_on_warning_only(self):
        # Only warnings (tool-loop), no errors
        events = [_tool_call("Bash")] * 7 + [_tool_call("Write", file_path="a.py"), _session_end()]
        store, sid = _make_store_with_events(events)
        args = self._make_args(store.base_dir, session_id=sid, strict=False)
        result = cmd_lint(args)
        self.assertEqual(result, 0)

    def test_json_format_output(self):
        import io
        from unittest.mock import patch
        events = [_tool_call("Bash")] * 7 + [_session_end()]
        store, sid = _make_store_with_events(events)
        args = self._make_args(store.base_dir, session_id=sid, fmt="json")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cmd_lint(args)
        data = json.loads(captured.getvalue())
        self.assertIn("findings", data)
        self.assertIn("session_id", data)

    def test_all_flag_lints_multiple_sessions(self):
        tmp = tempfile.mkdtemp()
        store = TraceStore(Path(tmp))
        for _ in range(3):
            meta = SessionMeta(agent_name="test", command="test")
            sp = store.create_session(meta)
            sid = sp.name
            e = TraceEvent(event_type=EventType.SESSION_END, timestamp=1.0,
                           session_id=sid, data={})
            store.append_event(sid, e)
        args = self._make_args(store.base_dir, lint_all=True)
        result = cmd_lint(args)
        self.assertIn(result, (0, 1))  # just verify it runs without crash

    def test_missing_session_returns_1(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        store = TraceStore(Path(tmp))
        args = self._make_args(store.base_dir, session_id="nonexistent")
        result = cmd_lint(args)
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
