"""Tests for historical policy backtesting (issue #211)."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from agent_trace.cli import _normalise_policy_argv, build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.policy_backtest import (
    PolicyBacktestError,
    backtest_policy,
    diff_policies,
    format_backtest,
    format_coverage,
    format_policy_diff,
    policy_coverage,
)
from agent_trace.store import TraceStore


NOW = 2_000_000_000.0
DAY = 86400.0


def _tool(tool_name: str, **arguments) -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        timestamp=NOW,
        data={"tool_name": tool_name, "arguments": arguments},
    )


class PolicyBacktestTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TraceStore(Path(self.tmp.name) / "traces", redact=False)

    def tearDown(self):
        self.tmp.cleanup()

    def add_session(
        self,
        session_id: str,
        started_at: float,
        events: list[TraceEvent],
    ) -> None:
        meta = SessionMeta(session_id=session_id, started_at=started_at)
        self.store.create_session(meta)
        for event in events:
            event.session_id = session_id
            self.store.append_event(session_id, event)

    def write_policy(self, name: str, policy: dict) -> Path:
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps(policy))
        return path


class TestPolicyBacktest(PolicyBacktestTestCase):
    def setUp(self):
        super().setUp()
        self.policy = self.write_policy("policy.json", {
            "files": {
                "read": {"allow": ["src/**"]},
                "write": {"deny": ["/etc/**"]},
            },
            "commands": {
                "allow": ["git *"],
                "deny": ["rm -rf *"],
            },
        })

    def test_backtest_filters_window_and_counts_each_rule(self):
        self.add_session("recent", NOW - 3600, [
            _tool("Read", file_path="src/main.py"),
            _tool("Write", file_path="/etc/hosts"),
            _tool("Bash", command="git status"),
            _tool("Bash", command="rm -rf build"),
            _tool("Read", file_path="README.md"),
            _tool("TodoWrite", todos=[]),
        ])
        self.add_session("too-old", NOW - 31 * DAY, [
            _tool("Write", file_path="/etc/passwd"),
        ])

        report = backtest_policy(self.store, self.policy, days=30, now=NOW)

        self.assertEqual(report.session_count, 1)
        self.assertEqual(report.total_tool_calls, 6)
        self.assertEqual(report.covered_tool_calls, 4)
        self.assertEqual(report.uncovered_tool_calls, 2)
        self.assertAlmostEqual(report.coverage_percent, 100 * 4 / 6)
        self.assertEqual(len(report.blocked_calls), 3)
        self.assertEqual(report.affected_session_ids, ["recent"])

        by_label = {stats.rule.label: stats for stats in report.rules}
        self.assertEqual(by_label["allow: read src/**"].matched, 1)
        self.assertEqual(by_label["allow: read src/**"].would_allow, 1)
        self.assertEqual(by_label["deny: write /etc/**"].would_block, 1)
        self.assertEqual(by_label["allow: bash git *"].would_allow, 1)
        self.assertEqual(by_label["deny: bash rm -rf *"].would_block, 1)

        # README.md is denied by the implicit allow-list default while the
        # arbitrary tool is allowed by default. Neither matched a pattern.
        default = by_label["default (no rule matched)"]
        self.assertEqual(default.matched, 2)
        self.assertEqual(default.would_block, 1)
        self.assertEqual(default.would_allow, 1)

    def test_all_history_includes_old_sessions(self):
        self.add_session("old", NOW - 90 * DAY, [
            _tool("Read", file_path="src/old.py"),
        ])
        report = backtest_policy(self.store, self.policy, days=None, now=NOW)
        self.assertEqual(report.session_count, 1)
        self.assertEqual(report.total_tool_calls, 1)
        self.assertIsNone(report.window_start)

    def test_zero_call_history_has_zero_coverage(self):
        self.add_session("empty", NOW - 60, [
            TraceEvent(event_type=EventType.USER_PROMPT, data={"prompt": "hello"}),
        ])
        report = backtest_policy(self.store, self.policy, now=NOW)
        self.assertEqual(report.total_tool_calls, 0)
        self.assertEqual(report.coverage_percent, 0.0)
        self.assertTrue(all(stats.matched == 0 for stats in report.rules))

    def test_missing_policy_and_invalid_window_are_errors(self):
        with self.assertRaises(PolicyBacktestError):
            backtest_policy(self.store, Path(self.tmp.name) / "missing.json", now=NOW)
        with self.assertRaisesRegex(PolicyBacktestError, "greater than zero"):
            backtest_policy(self.store, self.policy, days=0, now=NOW)

    def test_structurally_invalid_policy_fields_are_rejected(self):
        invalid_policies = [
            (["not", "an", "object"], "<root>"),
            ({"files": []}, "files"),
            ({"files": {"read": {"allow": "src/**"}}}, "files.read.allow"),
            ({"commands": {"deny": ["rm *", 7]}}, "commands.deny"),
            ({"network": {"deny_all": "false"}}, "network.deny_all"),
            ({"network": {"allow": "localhost"}}, "network.allow"),
        ]
        for index, (policy_data, field_name) in enumerate(invalid_policies):
            with self.subTest(field=field_name):
                path = self.write_policy(f"invalid-{index}.json", policy_data)
                with self.assertRaisesRegex(PolicyBacktestError, field_name.replace(".", r"\.")):
                    backtest_policy(self.store, path, now=NOW)

    def test_malformed_tool_arguments_are_uncovered_instead_of_aborting(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            timestamp=NOW,
            data={"tool_name": "Bash", "arguments": "rm -rf /tmp/build"},
        )
        self.add_session("malformed", NOW - 1, [event])

        report = backtest_policy(self.store, self.policy, now=NOW)

        self.assertEqual(report.total_tool_calls, 1)
        self.assertEqual(report.covered_tool_calls, 0)
        self.assertEqual(len(report.blocked_calls), 0)
        self.assertEqual(
            report.calls[0].reason,
            "malformed tool call arguments; no policy rule evaluated",
        )
        self.assertEqual(
            report.calls[0].action,
            "Tool: Bash (malformed arguments)",
        )
        default = next(stats for stats in report.rules if stats.rule.effect == "default")
        self.assertEqual(default.would_allow, 1)

    def test_backtest_json_is_machine_readable(self):
        self.add_session("recent", NOW - 1, [_tool("Read", file_path="src/a.py")])
        report = backtest_policy(self.store, self.policy, now=NOW)
        data = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(data["type"], "policy_backtest")
        self.assertEqual(data["tool_calls"], 1)
        self.assertEqual(data["coverage"]["percent"], 100.0)
        self.assertIn("session_ids", data["would_block"])

    def test_show_sessions_lists_affected_ids(self):
        self.add_session("blocked-session", NOW - 1, [
            _tool("Write", file_path="/etc/hosts"),
        ])
        report = backtest_policy(self.store, self.policy, now=NOW)
        out = io.StringIO()
        format_backtest(report, out=out, show_sessions=True)
        text = out.getvalue()
        self.assertIn("Policy backtest", text)
        self.assertIn("Affected sessions", text)
        self.assertIn("blocked-session", text)


class TestNetworkBacktest(PolicyBacktestTestCase):
    def test_network_rules_are_attributed_once_per_call(self):
        policy = self.write_policy("network.json", {
            "network": {"deny_all": True, "allow": ["localhost"]},
        })
        self.add_session("network", NOW - 1, [
            _tool("Bash", command="curl http://localhost:8080/health"),
            _tool("Bash", command="curl https://example.com/data"),
            _tool("Bash", command="printf done"),
        ])

        report = backtest_policy(self.store, policy, now=NOW)

        self.assertEqual(report.total_tool_calls, 3)
        self.assertEqual(report.covered_tool_calls, 2)
        self.assertEqual(len(report.blocked_calls), 1)
        by_label = {stats.rule.label: stats for stats in report.rules}
        self.assertEqual(by_label["allow: network localhost"].would_allow, 1)
        self.assertEqual(by_label["deny: network *"].would_block, 1)
        self.assertEqual(by_label["default (no rule matched)"].would_allow, 1)


class TestPolicyDiff(PolicyBacktestTestCase):
    def test_diff_uses_one_snapshot_when_active_session_grows(self):
        self.add_session("active", NOW - 1, [
            _tool("Read", file_path="src/main.py"),
        ])
        old = self.write_policy("old.json", {})
        new = self.write_policy("new.json", {
            "files": {"write": {"deny": ["tmp/**"]}},
        })
        appended = _tool("Write", file_path="tmp/cache.txt")
        appended.session_id = "active"
        original_load_events = self.store.load_events
        load_count = 0

        def load_then_grow(session_id: str):
            nonlocal load_count
            events = original_load_events(session_id)
            load_count += 1
            if load_count == 1:
                self.store.append_event(session_id, appended)
            return events

        self.store.load_events = load_then_grow
        try:
            report = diff_policies(self.store, old, new, days=30, now=NOW)
        finally:
            self.store.load_events = original_load_events

        self.assertEqual(load_count, 1)
        self.assertEqual(report.old.total_tool_calls, 1)
        self.assertEqual(report.new.total_tool_calls, 1)
        self.assertEqual(report.introduced, [])

    def test_diff_finds_introduced_and_removed_blocks(self):
        self.add_session("writes", NOW - 1, [
            _tool("Write", file_path="tmp/cache.txt"),
        ])
        self.add_session("network", NOW - 2, [
            _tool("Bash", command="curl https://example.com/data"),
        ])
        old = self.write_policy("old.json", {
            "files": {"write": {"allow": ["**"]}},
            "commands": {"deny": ["curl *"]},
        })
        new = self.write_policy("new.json", {
            "files": {"write": {"deny": ["tmp/**"]}},
            "commands": {"allow": ["curl *"]},
        })

        report = diff_policies(self.store, old, new, days=30, now=NOW)

        self.assertEqual(len(report.old.blocked_calls), 1)
        self.assertEqual(len(report.new.blocked_calls), 1)
        self.assertEqual(report.blocked_delta, 0)
        self.assertEqual(len(report.introduced), 1)
        self.assertEqual(len(report.removed), 1)
        self.assertEqual(report.introduced[0].session_id, "writes")
        self.assertEqual(report.introduced[0].rule_label, "deny: write tmp/**")
        self.assertEqual(report.removed[0].session_id, "network")
        self.assertEqual(report.removed[0].rule_label, "deny: bash curl *")

        data = report.to_dict()
        self.assertEqual(data["new_blocks"]["tool_calls"], 1)
        self.assertEqual(data["blocks_removed"]["tool_calls"], 1)
        self.assertEqual(data["new_blocks"]["by_rule"][0]["tool_calls"], 1)

    def test_diff_format_recommends_reviewing_new_blocks(self):
        self.add_session("writes", NOW - 1, [
            _tool("Write", file_path="tmp/cache.txt"),
        ])
        old = self.write_policy("old.json", {})
        new = self.write_policy("new.json", {
            "files": {"write": {"deny": ["tmp/**"]}},
        })
        report = diff_policies(self.store, old, new, now=NOW)
        out = io.StringIO()
        format_policy_diff(report, out=out)
        text = out.getvalue()
        self.assertIn("New blocks introduced", text)
        self.assertIn("review 1 new blocks", text)


class TestPolicyCoverage(PolicyBacktestTestCase):
    def test_coverage_reports_and_lists_uncovered_calls(self):
        policy = self.write_policy("coverage.json", {
            "files": {"read": {"allow": ["src/**"]}},
        })
        self.add_session("coverage", NOW - 1, [
            _tool("Read", file_path="src/main.py"),
            _tool("Read", file_path="README.md"),
            _tool("Agent", prompt="delegate"),
        ])

        report = policy_coverage(self.store, policy, now=NOW)

        self.assertEqual(report.total_tool_calls, 3)
        self.assertEqual(report.covered_tool_calls, 1)
        self.assertEqual(report.uncovered_tool_calls, 2)
        self.assertAlmostEqual(report.coverage_percent, 100 / 3)
        data = report.to_dict(show_uncovered=True)
        self.assertEqual(len(data["uncovered_calls"]), 2)

        out = io.StringIO()
        format_coverage(report, out=out, show_uncovered=True)
        text = out.getvalue()
        self.assertIn("Coverage: 33.3%", text)
        self.assertIn("README.md", text)
        self.assertIn("Tool: Agent", text)


class TestPolicyBacktestCLI(unittest.TestCase):
    def test_parses_policy_subcommands(self):
        parser = build_parser()
        backtest = parser.parse_args([
            "policy", "backtest", "--policy", "scope.json", "--days", "14",
            "--show-sessions", "--format", "json",
        ])
        self.assertEqual(backtest.policy_command, "backtest")
        self.assertEqual(backtest.policy, "scope.json")
        self.assertEqual(backtest.days, 14)
        self.assertTrue(backtest.show_sessions)
        self.assertEqual(backtest.format, "json")

        diff = parser.parse_args(["policy", "diff", "old.json", "new.json"])
        self.assertEqual(diff.policy_command, "diff")
        self.assertEqual(diff.old_policy, "old.json")
        self.assertEqual(diff.new_policy, "new.json")

        coverage = parser.parse_args([
            "policy", "coverage", "--show-uncovered",
        ])
        self.assertEqual(coverage.policy_command, "coverage")
        self.assertTrue(coverage.show_uncovered)

    def test_legacy_policy_generation_is_normalised(self):
        self.assertEqual(
            _normalise_policy_argv(["policy", "abc123", "--dry-run"]),
            ["policy", "generate", "abc123", "--dry-run"],
        )
        self.assertEqual(
            _normalise_policy_argv(["--trace-dir", "traces", "policy"]),
            ["--trace-dir", "traces", "policy", "generate"],
        )
        self.assertEqual(
            _normalise_policy_argv(["policy", "backtest"]),
            ["policy", "backtest"],
        )


if __name__ == "__main__":
    unittest.main()
