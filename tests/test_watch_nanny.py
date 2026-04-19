"""Tests for issue #48: agent nanny rule-based kill switch."""

from __future__ import annotations

import json
import time
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _tool_call(tool_name: str, args: dict) -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        data={"tool_name": tool_name, "arguments": args},
    )


class TestNannyRuleParsing(unittest.TestCase):
    def test_parse_numeric_condition(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="cost", condition="cost_usd > 5.00", action="kill")
        self.assertEqual(rule._metric, "cost_usd")
        self.assertEqual(rule._op, ">")
        self.assertEqual(rule._threshold, 5.0)

    def test_parse_file_path_matches(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="sensitive", condition='file_path matches "/etc/**"', action="kill")
        self.assertEqual(rule._metric, "file_path")
        self.assertEqual(rule._op, "matches")
        self.assertEqual(rule._pattern, "/etc/**")

    def test_evaluate_numeric_true(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="cost", condition="cost_usd > 5.00", action="kill")
        self.assertTrue(rule.evaluate({"cost_usd": 6.0}))

    def test_evaluate_numeric_false(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="cost", condition="cost_usd > 5.00", action="kill")
        self.assertFalse(rule.evaluate({"cost_usd": 3.0}))

    def test_evaluate_files_modified_over(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="files", condition="files_modified > 20", action="pause")
        self.assertTrue(rule.evaluate({"files_modified": 21}))

    def test_evaluate_files_modified_at_limit(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="files", condition="files_modified > 20", action="pause")
        self.assertFalse(rule.evaluate({"files_modified": 20}))

    def test_evaluate_file_path_matches(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="sensitive", condition='file_path matches "/etc/**"', action="kill")
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "write", "arguments": {"file_path": "/etc/passwd"}},
        )
        self.assertTrue(rule.evaluate({}, event))

    def test_evaluate_file_path_no_match(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="sensitive", condition='file_path matches "/etc/**"', action="kill")
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "write", "arguments": {"file_path": "/home/user/file.py"}},
        )
        self.assertFalse(rule.evaluate({}, event))


class TestNannyRuleLoading(unittest.TestCase):
    def test_load_json_rules(self):
        import tempfile, os
        from agent_trace.watch import _load_nanny_rules
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"rules": [
                {"name": "cost-limit", "condition": "cost_usd > 5.00", "action": "kill"},
                {"name": "too-many-files", "condition": "files_modified > 20", "action": "pause"},
            ]}, f)
            path = f.name
        try:
            rules = _load_nanny_rules(path)
            self.assertEqual(len(rules), 2)
            self.assertEqual(rules[0].name, "cost-limit")
            self.assertEqual(rules[1].action, "pause")
        finally:
            os.unlink(path)

    def test_load_yaml_rules(self):
        import tempfile, os
        from agent_trace.watch import _load_nanny_rules
        yaml_text = (
            "rules:\n"
            "  - name: cost-limit\n"
            "    condition: cost_usd > 5.00\n"
            "    action: kill\n"
            "  - name: too-many-files\n"
            "    condition: files_modified > 20\n"
            "    action: pause\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            rules = _load_nanny_rules(path)
            self.assertEqual(len(rules), 2)
            self.assertEqual(rules[0].name, "cost-limit")
            self.assertEqual(rules[1].name, "too-many-files")
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        from agent_trace.watch import _load_nanny_rules
        rules = _load_nanny_rules("/nonexistent/path/rules.yaml")
        self.assertEqual(rules, [])


class TestWatchStateNanny(unittest.TestCase):
    def test_nanny_metrics(self):
        from agent_trace.watch import WatchState
        state = WatchState(start_time=time.time() - 120)
        state.files_modified = 5
        state.estimated_cost = 2.5
        state.consecutive_test_failures = 2
        metrics = state.nanny_metrics()
        self.assertEqual(metrics["files_modified"], 5)
        self.assertEqual(metrics["cost_usd"], 2.5)
        self.assertEqual(metrics["consecutive_test_failures"], 2)
        self.assertGreaterEqual(metrics["duration_minutes"], 2.0)

    def test_check_event_tracks_files_modified(self):
        from agent_trace.watch import WatcherConfig, WatchState, check_event
        config = WatcherConfig()
        state = WatchState()
        check_event(_tool_call("write", {"file_path": "src/foo.py"}), config, state)
        check_event(_tool_call("write", {"file_path": "src/bar.py"}), config, state)
        check_event(_tool_call("write", {"file_path": "src/foo.py"}), config, state)  # duplicate
        self.assertEqual(state.files_modified, 2)

    def test_check_event_tracks_test_failures(self):
        from agent_trace.watch import WatcherConfig, WatchState, check_event
        config = WatcherConfig()
        state = WatchState()
        fail_event = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            data={"content": "FAILED: test_foo AssertionError", "is_error": True},
        )
        check_event(fail_event, config, state)
        check_event(fail_event, config, state)
        self.assertEqual(state.consecutive_test_failures, 2)

    def test_cli_has_rules_and_dry_run_flags(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["watch", "--rules", "rules.yaml", "--dry-run"])
        self.assertEqual(args.rules, "rules.yaml")
        self.assertTrue(args.dry_run)
