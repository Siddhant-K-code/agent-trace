"""Tests for per-operation enforcement and token budget in watch mode (issues #21, #27)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.watch import (
    OperationRule,
    WatchState,
    WatcherConfig,
    check_event,
)


def _make_tool_call(tool_name, **kwargs):
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        data={"tool_name": tool_name, "arguments": kwargs},
    )


def _make_llm_request(model="claude-sonnet-4", input_tokens=1000):
    return TraceEvent(
        event_type=EventType.LLM_REQUEST,
        data={"model": model, "input_tokens": input_tokens},
    )


class TestOperationRule(unittest.TestCase):
    def test_wildcard_tool_matches_any(self):
        rule = OperationRule(tool_name="*", pattern="*", action="alert")
        self.assertTrue(rule.matches("bash", "anything"))

    def test_specific_tool_matches(self):
        rule = OperationRule(tool_name="bash", pattern="rm *", action="block")
        self.assertTrue(rule.matches("bash", "rm -rf /tmp"))

    def test_specific_tool_no_match_other_tool(self):
        rule = OperationRule(tool_name="bash", pattern="*", action="alert")
        self.assertFalse(rule.matches("write", "some/path"))

    def test_glob_pattern(self):
        rule = OperationRule(tool_name="write", pattern="*.env", action="block")
        self.assertTrue(rule.matches("write", ".env"))
        self.assertTrue(rule.matches("write", "prod.env"))
        self.assertFalse(rule.matches("write", "main.py"))

    def test_case_insensitive(self):
        rule = OperationRule(tool_name="BASH", pattern="*", action="alert")
        self.assertTrue(rule.matches("bash", "cmd"))


class TestCheckEventOperationRules(unittest.TestCase):
    def _make_config_with_rule(self, tool, pattern, action="alert"):
        return WatcherConfig(
            operation_rules=[OperationRule(tool_name=tool, pattern=pattern, action=action)]
        )

    def test_rule_fires_on_match(self):
        config = self._make_config_with_rule("bash", "rm *")
        state = WatchState()
        event = _make_tool_call("bash", command="rm -rf /tmp/test")
        violations = check_event(event, config, state)
        self.assertTrue(any("OperationWatcher" in v for v in violations))

    def test_rule_does_not_fire_on_no_match(self):
        config = self._make_config_with_rule("bash", "rm *")
        state = WatchState()
        event = _make_tool_call("bash", command="pytest tests/")
        violations = check_event(event, config, state)
        self.assertFalse(any("OperationWatcher" in v for v in violations))

    def test_rule_fires_only_once_per_target(self):
        config = self._make_config_with_rule("bash", "rm *")
        state = WatchState()
        event = _make_tool_call("bash", command="rm -rf /tmp/test")
        first = check_event(event, config, state)
        second = check_event(event, config, state)
        op_first = [v for v in first if "OperationWatcher" in v]
        op_second = [v for v in second if "OperationWatcher" in v]
        self.assertTrue(len(op_first) > 0)
        self.assertEqual(len(op_second), 0)

    def test_write_rule_on_file_path(self):
        config = self._make_config_with_rule("write", "*.env")
        state = WatchState()
        event = _make_tool_call("write", file_path=".env")
        violations = check_event(event, config, state)
        self.assertTrue(any("OperationWatcher" in v for v in violations))

    def test_rule_includes_reason(self):
        config = WatcherConfig(operation_rules=[
            OperationRule(tool_name="bash", pattern="rm *", action="block", reason="no deletions")
        ])
        state = WatchState()
        event = _make_tool_call("bash", command="rm file.txt")
        violations = check_event(event, config, state)
        self.assertTrue(any("no deletions" in v for v in violations))


class TestTokenBudgetInWatch(unittest.TestCase):
    def test_token_budget_no_fire_below_threshold(self):
        # token_budget module may not be present on this branch — watcher is disabled gracefully
        config = WatcherConfig(max_context_pct=90)
        state = WatchState()
        event = _make_llm_request(input_tokens=10_000)
        violations = check_event(event, config, state)
        # Should not raise; token budget watcher is optional
        self.assertFalse(any("TokenBudgetWatcher" in v for v in violations))


if __name__ == "__main__":
    unittest.main()
