"""Tests for issue #48: agent nanny rule-based kill switch."""

from __future__ import annotations

import json
import time

import pytest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _tool_call(tool_name: str, args: dict) -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        data={"tool_name": tool_name, "arguments": args},
    )


class TestNannyRules:
    def test_parse_numeric_condition(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="cost", condition="cost_usd > 5.00", action="kill")
        assert rule._metric == "cost_usd"
        assert rule._op == ">"
        assert rule._threshold == 5.0

    def test_parse_file_path_matches(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="sensitive", condition='file_path matches "/etc/**"', action="kill")
        assert rule._metric == "file_path"
        assert rule._op == "matches"
        assert rule._pattern == "/etc/**"

    def test_evaluate_numeric_true(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="cost", condition="cost_usd > 5.00", action="kill")
        assert rule.evaluate({"cost_usd": 6.0}) is True

    def test_evaluate_numeric_false(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="cost", condition="cost_usd > 5.00", action="kill")
        assert rule.evaluate({"cost_usd": 3.0}) is False

    def test_evaluate_files_modified(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="files", condition="files_modified > 20", action="pause")
        assert rule.evaluate({"files_modified": 21}) is True
        assert rule.evaluate({"files_modified": 20}) is False

    def test_evaluate_file_path_matches(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="sensitive", condition='file_path matches "/etc/**"', action="kill")
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "write", "arguments": {"file_path": "/etc/passwd"}},
        )
        assert rule.evaluate({}, event) is True

    def test_evaluate_file_path_no_match(self):
        from agent_trace.watch import NannyRule
        rule = NannyRule(name="sensitive", condition='file_path matches "/etc/**"', action="kill")
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "write", "arguments": {"file_path": "/home/user/file.py"}},
        )
        assert rule.evaluate({}, event) is False

    def test_load_json_rules(self, tmp_path):
        from agent_trace.watch import _load_nanny_rules
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps({
            "rules": [
                {"name": "cost-limit", "condition": "cost_usd > 5.00", "action": "kill"},
                {"name": "too-many-files", "condition": "files_modified > 20", "action": "pause"},
            ]
        }))
        rules = _load_nanny_rules(str(rules_file))
        assert len(rules) == 2
        assert rules[0].name == "cost-limit"
        assert rules[1].action == "pause"

    def test_load_yaml_rules(self, tmp_path):
        from agent_trace.watch import _load_nanny_rules
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(
            "rules:\n"
            "  - name: cost-limit\n"
            "    condition: cost_usd > 5.00\n"
            "    action: kill\n"
            "  - name: too-many-files\n"
            "    condition: files_modified > 20\n"
            "    action: pause\n"
        )
        rules = _load_nanny_rules(str(rules_file))
        assert len(rules) == 2
        assert rules[0].name == "cost-limit"
        assert rules[1].name == "too-many-files"

    def test_load_missing_file(self, tmp_path):
        from agent_trace.watch import _load_nanny_rules
        rules = _load_nanny_rules(str(tmp_path / "nonexistent.yaml"))
        assert rules == []

    def test_watch_state_nanny_metrics(self):
        from agent_trace.watch import WatchState
        state = WatchState(start_time=time.time() - 120)
        state.files_modified = 5
        state.estimated_cost = 2.5
        state.consecutive_test_failures = 2
        metrics = state.nanny_metrics()
        assert metrics["files_modified"] == 5
        assert metrics["cost_usd"] == 2.5
        assert metrics["consecutive_test_failures"] == 2
        assert metrics["duration_minutes"] >= 2.0

    def test_check_event_tracks_files_modified(self):
        from agent_trace.watch import WatcherConfig, WatchState, check_event
        config = WatcherConfig()
        state = WatchState()
        e1 = _tool_call("write", {"file_path": "src/foo.py"})
        e2 = _tool_call("write", {"file_path": "src/bar.py"})
        e3 = _tool_call("write", {"file_path": "src/foo.py"})  # duplicate
        check_event(e1, config, state)
        check_event(e2, config, state)
        check_event(e3, config, state)
        assert state.files_modified == 2  # only 2 distinct files

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
        assert state.consecutive_test_failures == 2

    def test_cli_has_rules_and_dry_run_flags(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["watch", "--rules", "rules.yaml", "--dry-run"])
        assert args.rules == "rules.yaml"
        assert args.dry_run is True
