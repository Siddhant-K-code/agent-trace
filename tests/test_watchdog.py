"""Tests for watchdog mode: --timeout, --budget, --on-death, post-mortem JSON."""

import io
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import agent_trace
import agent_trace.watch as watch_mod
from agent_trace.cli import build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.project_budget import ProjectBudgetConfig
from agent_trace.store import TraceStore
from agent_trace.watch import (
    WatcherConfig,
    WatchState,
    _IDLE_SENTINEL,
    _parse_duration,
    _write_watchdog_postmortem,
    _invoke_on_death,
    cmd_watch,
)

SESSION = "watchdog-session"
PID = 424242
HAS_SIGNALS = hasattr(signal, "SIGSTOP")


# ---------------------------------------------------------------------------
# _parse_duration
# ---------------------------------------------------------------------------

class TestParseDuration(unittest.TestCase):
    def test_bare_number_is_seconds(self):
        self.assertAlmostEqual(_parse_duration("90"), 90.0)

    def test_seconds_suffix(self):
        self.assertAlmostEqual(_parse_duration("30s"), 30.0)

    def test_minutes_suffix(self):
        self.assertAlmostEqual(_parse_duration("5m"), 300.0)

    def test_hours_suffix(self):
        self.assertAlmostEqual(_parse_duration("2h"), 7200.0)

    def test_days_suffix(self):
        self.assertAlmostEqual(_parse_duration("1d"), 86400.0)

    def test_compound_hours_minutes(self):
        self.assertAlmostEqual(_parse_duration("1h30m"), 5400.0)

    def test_float_minutes(self):
        self.assertAlmostEqual(_parse_duration("1.5m"), 90.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _parse_duration("")

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _parse_duration("xyz")

    def test_unit_words_case_and_spaces_are_accepted(self):
        """Guard: passes on main too."""
        for value, seconds in (
            ("30mins", 1800.0), ("2hr", 7200.0), ("2hrs", 7200.0), ("5 minutes", 300.0),
            ("1h 30m", 5400.0), ("1H30M", 5400.0), ("1 hour 30 minutes", 5400.0),
            ("10sec", 10.0), ("2 days", 172800.0), (" 90 ", 90.0),
        ):
            with self.subTest(value=value):
                self.assertAlmostEqual(_parse_duration(value), seconds)

    def test_a_bare_number_is_read_by_float(self):
        """Guard: passes on main too. NaN, infinity and signs reach the
        callers' checks."""
        for value, seconds in (("-5", -5.0), ("1e3", 1000.0), ("inf", float("inf"))):
            with self.subTest(value=value):
                self.assertEqual(_parse_duration(value), seconds)
        self.assertTrue(math.isnan(_parse_duration("nan")))

    def test_a_number_may_start_with_a_point(self):
        self.assertAlmostEqual(_parse_duration(".5h"), 1800.0)

    def test_text_it_does_not_use_is_refused(self):
        # Every group needs a known unit; nothing else may be left over.
        for value in (
            "1w", "3y", "1h30", "1e3m", "0.5.5h", "5 apples", "5ms", "-5m", "+5m", "5.h",
            "1h,30m", "5 m s", "\uff15m",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as ctx:
                    _parse_duration(value)
                self.assertEqual(str(ctx.exception), f"cannot parse duration: {value!r}")

    def test_groups_that_add_up_to_zero_are_refused(self):
        """Guard: passes on main too. Groups that add up to 0 are refused
        here, not returned as a 0 second limit."""
        for value in ("0m", "0s", "0h 0m", ".0h"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as ctx:
                    _parse_duration(value)
                self.assertEqual(str(ctx.exception), f"cannot parse duration: {value!r}")


# ---------------------------------------------------------------------------
# _write_watchdog_postmortem
# ---------------------------------------------------------------------------

class TestWriteWatchdogPostmortem(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)
        self.meta = SessionMeta()
        self.store.create_session(self.meta)

    def _add_event(self, event_type: EventType, **data):
        ev = TraceEvent(
            event_type=event_type,
            session_id=self.meta.session_id,
            data=data,
        )
        self.store.append_event(self.meta.session_id, ev)
        return ev

    def test_writes_json_file(self):
        self._add_event(EventType.TOOL_CALL, tool_name="Bash", arguments={"command": "ls"})
        state = WatchState(start_time=time.time() - 60)
        state.estimated_cost = 1.23

        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="DurationWatcher: 60s elapsed"
        )

        self.assertIsNotNone(pm_path)
        self.assertTrue(pm_path.exists())
        data = json.loads(pm_path.read_text())
        self.assertEqual(data["session_id"], self.meta.session_id)
        self.assertIn("reason", data)
        self.assertIn("cost_at_death", data)
        self.assertIn("recovery_context", data)
        self.assertAlmostEqual(data["cost_at_death"], 1.23, places=4)

    def test_last_tool_call_captured(self):
        self._add_event(EventType.TOOL_CALL, tool_name="Bash", arguments={"command": "pytest"})
        state = WatchState(start_time=time.time())
        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="test"
        )
        data = json.loads(pm_path.read_text())
        self.assertIsNotNone(data["last_tool_call"])
        self.assertEqual(data["last_tool_call"]["tool_name"], "Bash")

    def test_last_llm_response_captured(self):
        self._add_event(EventType.LLM_RESPONSE, model="claude-3-5-sonnet", content="done")
        state = WatchState(start_time=time.time())
        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="test"
        )
        data = json.loads(pm_path.read_text())
        self.assertIsNotNone(data["last_llm_response"])

    def test_empty_session_still_writes(self):
        state = WatchState(start_time=time.time())
        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="budget exceeded"
        )
        self.assertIsNotNone(pm_path)
        data = json.loads(pm_path.read_text())
        self.assertIsNone(data["last_tool_call"])

    def test_recovery_context_mentions_reason(self):
        state = WatchState(start_time=time.time() - 120)
        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="CostWatcher: $5.00"
        )
        data = json.loads(pm_path.read_text())
        self.assertIn("CostWatcher", data["recovery_context"])

    def test_invalid_session_returns_none(self):
        state = WatchState(start_time=time.time())
        result = _write_watchdog_postmortem(
            self.store, "nonexistent-session-id", state, reason="test"
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# WatcherConfig: on_death_cmd field
# ---------------------------------------------------------------------------

class TestWatcherConfigOnDeath(unittest.TestCase):
    def test_default_on_death_is_empty(self):
        config = WatcherConfig()
        self.assertEqual(config.on_death_cmd, "")

    def test_on_death_cmd_set(self):
        config = WatcherConfig(on_death_cmd="python recover.py --pm {post_mortem_path}")
        self.assertIn("{post_mortem_path}", config.on_death_cmd)


# ---------------------------------------------------------------------------
# _invoke_on_death: substitution
# ---------------------------------------------------------------------------

class TestInvokeOnDeath(unittest.TestCase):
    def test_no_crash_on_none_path(self):
        # Should not raise even with None path
        _invoke_on_death("echo done", None)

    def test_no_crash_on_empty_cmd(self):
        _invoke_on_death("", None)


# ---------------------------------------------------------------------------
# CLI: --timeout and --budget flags parsed correctly
# ---------------------------------------------------------------------------

class TestCLIWatchdogFlags(unittest.TestCase):
    def test_timeout_flag_registered(self):
        import argparse
        from agent_trace.cli import main
        import sys
        from io import StringIO

        # Just verify the flags exist by checking help output
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "watch", "--help"]
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            try:
                main()
            except SystemExit:
                pass
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self.assertIn("--timeout", output)
        self.assertIn("--budget", output)
        self.assertIn("--on-death", output)


# ---------------------------------------------------------------------------
# --timeout and --budget kill the agent
# ---------------------------------------------------------------------------

def _tool_call() -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL, timestamp=time.time(), session_id=SESSION,
        data={"tool_name": "Bash", "arguments": {"command": "make test"}},
    )


def _costly_response() -> TraceEvent:
    return TraceEvent(
        event_type=EventType.ASSISTANT_RESPONSE, timestamp=time.time(), session_id=SESSION,
        data={"text": "x" * 10000},
    )


def _three_dollars() -> TraceEvent:
    """An LLM response of 200,000 output tokens: $3.00 at the sonnet price."""
    return TraceEvent(
        event_type=EventType.LLM_RESPONSE, timestamp=time.time(), session_id=SESSION,
        data={"input_tokens": 0, "output_tokens": 200_000},
    )


@unittest.skipUnless(HAS_SIGNALS, "--agent-pid needs POSIX signals")
class WatchdogLimitTest(unittest.TestCase):
    """Run cmd_watch from parsed flags with the event stream and signals faked.

    watch() takes steps: events and _IDLE_SENTINEL are what the tail yields,
    one per poll cycle, and callables run between them. The fake agent PID
    answers signal 0 until agent_exits() runs; signal_errors are raised
    first, one per signal.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        # The default alert log and the project budget file are cwd-relative.
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.root)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in (
            "AGENT_STRACE_TENANT_ID", "AGENT_STRACE_SLACK_WEBHOOK", "AGENT_STRACE_TEAMS_WEBHOOK",
        ):
            os.environ.pop(name, None)
        self.trace_dir = self.root / "traces"
        TraceStore(self.trace_dir, redact=False).create_session(
            SessionMeta(session_id=SESSION, started_at=1.0),
        )
        self.agent_alive = True
        self.exit_error = ProcessLookupError
        self.signal_errors = []
        self.loop_state = None
        self.yielded = 0
        self.mocks = {}
        for name, target, kwargs in (
            ("pause", "agent_trace.watch._pause_process", {"return_value": None}),
            ("resume", "agent_trace.watch._resume_process", {"return_value": None}),
            # True: SIGTERM was sent, so the post-mortem and --on-death follow.
            ("kill", "agent_trace.watch._kill_process", {"return_value": True}),
            ("os_kill", "os.kill", {"side_effect": self._signal}),
            ("identity", "agent_trace.watch.process_identity", {"return_value": None}),
            ("terminal", "agent_trace.watch._alert_terminal", {}),
        ):
            p = patch(target, **kwargs)
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)

    def _signal(self, pid, sig):
        if self.signal_errors:
            raise self.signal_errors.pop(0)(pid)
        if not self.agent_alive:
            raise self.exit_error(pid)

    def pass_deadline(self):
        """Move the watch start back so a 60 s --timeout has passed."""
        self.loop_state.start_time -= 61

    def pass_grace(self):
        """Move the watch start back past the 10 s kill grace too."""
        self.loop_state.start_time -= 11

    def agent_exits(self):
        self.agent_alive = False

    def another_user_gets_the_agent_pid(self):
        """The agent exits and another user's process gets its PID."""
        self.agent_alive = False
        self.exit_error = PermissionError

    def agent_pid_is_reused(self):
        """The next signal 0 finds no process; later ones find the process
        that was given the agent's PID."""
        self.signal_errors.append(ProcessLookupError)

    @staticmethod
    def idle():
        time.sleep(0.1)

    def _script(self, steps):
        for step in steps:
            if callable(step):
                step()
            else:
                self.yielded += 1
                yield step

    def watch(self, *flags, steps=(), agent_pid=True, max_idle_seconds=300.0):
        argv = ["--trace-dir", str(self.trace_dir), "watch", SESSION, *flags]
        if agent_pid:
            argv += ["--agent-pid", str(PID)]
        args = build_parser().parse_args(argv)
        out, err = io.StringIO(), io.StringIO()
        real_watch_session = watch_mod.watch_session
        real_state = watch_mod.WatchState

        def run_watch(*a, **kw):
            return real_watch_session(*a, out=out, max_idle_seconds=max_idle_seconds, **kw)

        def make_state(**kw):
            self.loop_state = real_state(**kw)
            return self.loop_state

        with patch("agent_trace.watch.watch_session", side_effect=run_watch), \
             patch("agent_trace.watch._tail_events", return_value=self._script(steps)), \
             patch("agent_trace.watch.WatchState", side_effect=make_state), \
             redirect_stderr(err):
            status = cmd_watch(args)
        self.assertEqual(status, 0, err.getvalue())
        return out.getvalue()

    def alerts(self):
        return [c.args[0] for c in self.mocks["terminal"].call_args_list]

    def count(self, prefix):
        return sum(1 for alert in self.alerts() if alert.startswith(prefix))

    def cost_alerts(self):
        return [alert for alert in self.alerts() if alert.startswith("CostWatcher:")]


class TestTimeoutKills(WatchdogLimitTest):
    def test_timeout_kills_an_agent_that_writes_no_events(self):
        self.watch("--timeout", "1m", steps=(
            self.pass_deadline, _IDLE_SENTINEL, _IDLE_SENTINEL,
        ))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertIn("DurationWatcher: 61s elapsed (threshold: 60s)", self.alerts())
        self.assertIn(f"Killing agent process {PID}", self.alerts())

    def test_timeout_kill_of_a_silent_agent_writes_the_post_mortem_and_runs_on_death_once(self):
        # The kill on a poll cycle, with no events, gets what a kill on an
        # event gets: the post-mortem with its reason and --on-death once.
        post_mortem = (
            TraceStore(self.trace_dir, redact=False)._session_dir(SESSION)
            / "watchdog-postmortem.json"
        )
        with patch("agent_trace.watch.subprocess.Popen") as popen:
            self.watch(
                "--timeout", "1m", "--on-death", "restart-agent --from {post_mortem_path}",
                steps=(self.pass_deadline, _IDLE_SENTINEL, _IDLE_SENTINEL, _IDLE_SENTINEL),
            )
        self.mocks["kill"].assert_called_once_with(PID, None)
        popen.assert_called_once_with(f"restart-agent --from {post_mortem}", shell=True)
        self.assertEqual(
            json.loads(post_mortem.read_text())["reason"],
            "DurationWatcher: 61s elapsed (threshold: 60s)",
        )

    def test_timeout_kills_once_whatever_on_violation_says(self):
        self.watch("--timeout", "1m", "--on-violation", "terminal", steps=(
            self.pass_deadline, _tool_call(), _IDLE_SENTINEL, _tool_call(),
        ))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertEqual(self.count("DurationWatcher:"), 1)

    def test_timeout_kills_while_watch_holds_a_pause(self):
        (self.trace_dir / ".pause-request").write_text(SESSION)
        self.watch("--timeout", "1m", steps=(
            _IDLE_SENTINEL, self.pass_deadline, _IDLE_SENTINEL,
        ))
        self.mocks["pause"].assert_called_once_with(PID)
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_dry_run_timeout_reports_the_kill_without_killing(self):
        self.watch("--timeout", "1m", "--dry-run", steps=(
            self.pass_deadline, _IDLE_SENTINEL,
        ))
        self.assertIn("[dry-run] would kill agent process", self.alerts())
        self.mocks["kill"].assert_not_called()

    def test_timeout_without_agent_pid_alerts_on_time_and_warns_once(self):
        self.watch("--timeout", "1m", agent_pid=False, steps=(
            self.pass_deadline, _IDLE_SENTINEL, _IDLE_SENTINEL, _IDLE_SENTINEL,
        ))
        self.assertEqual(self.count("DurationWatcher:"), 1)
        self.assertEqual(self.count("Cannot kill the agent: no agent PID is known"), 1)
        self.mocks["kill"].assert_not_called()

    def test_timeout_kills_nothing_after_the_agent_exits(self):
        """Guard: passes on main too. A later event acts with --on-violation."""
        self.watch("--timeout", "1m", steps=(
            self.agent_exits, self.pass_deadline, _IDLE_SENTINEL, _IDLE_SENTINEL, _tool_call(),
        ))
        self.mocks["kill"].assert_not_called()
        self.assertEqual(self.count("DurationWatcher:"), 1)

    def test_a_pid_another_user_now_has_counts_as_exited(self):
        # Signal 0 raises PermissionError: the PID is another user's process.
        out = self.watch("--timeout", "1m", max_idle_seconds=0.05, steps=(
            self.another_user_gets_the_agent_pid, self.pass_deadline,
            _IDLE_SENTINEL, self.idle, _IDLE_SENTINEL, _IDLE_SENTINEL,
        ))
        self.assertTrue(self.loop_state.agent_exited)
        self.mocks["kill"].assert_not_called()
        self.assertIn("stopping", out)

    def test_an_exited_agent_stays_exited_when_its_pid_is_reused(self):
        self.watch("--timeout", "1m", steps=(
            self.agent_pid_is_reused, self.pass_deadline,
            _IDLE_SENTINEL, _IDLE_SENTINEL, _IDLE_SENTINEL,
        ))
        self.assertEqual(self.signal_errors, [])
        self.assertTrue(self.loop_state.agent_exited)
        self.mocks["kill"].assert_not_called()


class TestTimeoutIdleStop(WatchdogLimitTest):
    def test_idle_stop_waits_for_the_deadline_while_the_agent_runs(self):
        out = self.watch("--timeout", "1m", max_idle_seconds=0.05, steps=(
            _IDLE_SENTINEL, self.idle, _IDLE_SENTINEL, self.idle, _IDLE_SENTINEL,
        ))
        self.assertNotIn("stopping", out)
        self.assertEqual(self.yielded, 3)
        self.mocks["kill"].assert_not_called()

    def test_idle_stop_waits_out_the_kill_grace_then_stops(self):
        out = self.watch("--timeout", "1m", max_idle_seconds=0.05, steps=(
            self.pass_deadline, _IDLE_SENTINEL,
            self.idle, _IDLE_SENTINEL,
            self.pass_grace, _IDLE_SENTINEL,
            _IDLE_SENTINEL,
        ))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertIn("No events for 0s - stopping", out)
        self.assertEqual(self.yielded, 3)

    def test_idle_stop_applies_once_the_agent_exits(self):
        """Guard: passes on main too."""
        out = self.watch("--timeout", "1m", max_idle_seconds=0.05, steps=(
            self.agent_exits, _IDLE_SENTINEL, self.idle, _IDLE_SENTINEL, _IDLE_SENTINEL,
        ))
        self.assertIn("stopping", out)
        self.assertEqual(self.yielded, 2)
        self.mocks["kill"].assert_not_called()

    def test_idle_stop_is_unchanged_without_agent_pid(self):
        """Guard: passes on main too."""
        out = self.watch("--timeout", "1m", agent_pid=False, max_idle_seconds=0.05, steps=(
            _IDLE_SENTINEL, self.idle, _IDLE_SENTINEL, _IDLE_SENTINEL,
        ))
        self.assertIn("stopping", out)
        self.assertEqual(self.yielded, 2)


class TestBudgetKills(WatchdogLimitTest):
    def test_budget_kills_whatever_on_violation_says(self):
        self.watch("--budget", "0", "--on-violation", "terminal", steps=(_costly_response(),))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertEqual(self.count("CostWatcher:"), 1)

    def test_dry_run_budget_reports_the_kill_without_killing(self):
        self.watch("--budget", "0", "--dry-run", steps=(_costly_response(),))
        self.assertIn("[dry-run] would kill agent process", self.alerts())
        self.mocks["kill"].assert_not_called()


class TestOtherLimitsKeepOnViolation(WatchdogLimitTest):
    def test_max_duration_and_max_cost_act_with_on_violation(self):
        """Guard: passes on main too."""
        self.watch("--max-duration", "60", "--max-cost", "0", steps=(
            self.pass_deadline, _costly_response(), _IDLE_SENTINEL,
        ))
        self.assertEqual(self.count("DurationWatcher:"), 1)
        self.assertEqual(self.count("CostWatcher:"), 1)
        self.mocks["kill"].assert_not_called()


class TestLimitsWithConfigFile(WatchdogLimitTest):
    def setUp(self):
        super().setUp()
        self.config = self.root / "watch.json"

    def test_timeout_kills_with_a_config_file(self):
        self.config.write_text('{"watchers": {"retry": {"alert": "terminal"}}}')
        self.watch("--config", str(self.config), "--timeout", "1m", steps=(
            self.pass_deadline, _IDLE_SENTINEL,
        ))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertIn("DurationWatcher: 61s elapsed (threshold: 60s)", self.alerts())

    def test_budget_kills_with_a_config_file(self):
        self.config.write_text('{"watchers": {"retry": {"alert": "terminal"}}}')
        self.watch("--config", str(self.config), "--budget", "0", steps=(_costly_response(),))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertEqual(self.count("CostWatcher:"), 1)

    def test_config_file_limits_act_with_on_violation(self):
        """Guard: passes on main too. Only the flags kill."""
        self.config.write_text(
            '{"watchers": {"duration": {"max_minutes": 1}, "cost": {"max_dollars": 0}}}'
        )
        self.watch("--config", str(self.config), steps=(
            self.pass_deadline, _IDLE_SENTINEL, _costly_response(),
        ))
        self.assertEqual(self.count("DurationWatcher:"), 1)
        self.assertEqual(self.count("CostWatcher:"), 1)
        self.mocks["kill"].assert_not_called()


class TestPerSessionMaxKills(WatchdogLimitTest):
    """per_session_max lowers a cost limit that kills and replaces any other."""

    def setUp(self):
        super().setUp()
        self.config = self.root / "watch.json"
        # The standard project config, read from the working directory.
        (self.root / ".agent-strace.json").write_text('{"budget": {"per_session_max": 5}}')

    def test_it_replaces_a_max_cost_that_only_alerts(self):
        """Guard: passes on main too. There is no alert at $2, as on main."""
        self.watch("--max-cost", "2", steps=(_three_dollars(), _three_dollars()))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $6.00 (threshold: $5.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_it_never_raises_a_max_cost_that_kills(self):
        self.watch("--max-cost", "2", "--on-violation", "kill", steps=(_three_dollars(),))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $3.00 (threshold: $2.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_it_never_raises_a_config_file_cost_limit_that_kills(self):
        self.config.write_text(
            '{"watchers": {"cost": {"max_dollars": 2}, "retry": {"alert": "kill"}}}'
        )
        self.watch("--config", str(self.config), steps=(_three_dollars(),))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $3.00 (threshold: $2.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_it_replaces_a_config_file_cost_limit_that_only_alerts(self):
        """Guard: passes on main too."""
        self.config.write_text(
            '{"watchers": {"cost": {"max_dollars": 2}, "retry": {"alert": "terminal"}}}'
        )
        self.watch("--config", str(self.config), steps=(_three_dollars(), _three_dollars()))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $6.00 (threshold: $5.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_the_config_file_per_session_max_applies_without_weekly(self):
        self.config.write_text('{"budget": {"per_session_max": 2}}')
        self.watch("--config", str(self.config), steps=(_three_dollars(),))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $3.00 (threshold: $2.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_a_built_in_budget_never_raises_it(self):
        self.watch("--rules", "budget:100", steps=(_three_dollars(), _three_dollars()))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $6.00 (threshold: $5.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_a_lower_built_in_budget_lowers_it_and_still_kills(self):
        """Guard: passes on main too."""
        self.watch("--rules", "budget:2", steps=(_three_dollars(),))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $3.00 (threshold: $2.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)


class TestBuiltInLimitsNeverRaiseAKill(WatchdogLimitTest):
    """--rules budget:N and timeout:D lower a limit that kills, but never
    raise it, and the limit still kills."""

    def setUp(self):
        super().setUp()
        self.config = self.root / "watch.json"

    def test_budget_n_never_raises_budget(self):
        self.watch("--budget", "5", "--rules", "budget:10", steps=(
            _three_dollars(), _three_dollars(),
        ))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $6.00 (threshold: $5.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_budget_n_never_raises_a_max_cost_that_kills(self):
        self.watch("--max-cost", "5", "--on-violation", "kill", "--rules", "budget:10", steps=(
            _three_dollars(), _three_dollars(),
        ))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $6.00 (threshold: $5.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_budget_n_never_raises_a_config_file_cost_limit_that_kills(self):
        self.config.write_text(
            '{"watchers": {"cost": {"max_dollars": 5}, "retry": {"alert": "kill"}}}'
        )
        self.watch("--config", str(self.config), "--rules", "budget:10", steps=(
            _three_dollars(), _three_dollars(),
        ))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $6.00 (threshold: $5.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_timeout_d_never_raises_timeout_for_an_agent_that_writes_no_events(self):
        # The kill comes on a poll cycle, and watch stops once the kill grace
        # has passed, not at the 1 h deadline.
        out = self.watch("--timeout", "1m", "--rules", "timeout:1h", max_idle_seconds=0.05, steps=(
            self.pass_deadline, _IDLE_SENTINEL,
            self.idle, _IDLE_SENTINEL,
            self.pass_grace, _IDLE_SENTINEL,
            _IDLE_SENTINEL,
        ))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertIn("DurationWatcher: 61s elapsed (threshold: 60s)", self.alerts())
        self.assertIn("stopping", out)
        self.assertEqual(self.yielded, 3)

    def test_timeout_d_never_raises_a_max_duration_that_kills(self):
        self.watch(
            "--max-duration", "60", "--on-violation", "kill", "--rules", "timeout:1h",
            steps=(self.pass_deadline, _tool_call()),
        )
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertIn("DurationWatcher: 61s elapsed (threshold: 60s)", self.alerts())

    def test_timeout_d_never_raises_a_config_file_duration_limit_that_kills(self):
        self.config.write_text(
            '{"watchers": {"duration": {"max_minutes": 1}, "retry": {"alert": "kill"}}}'
        )
        self.watch("--config", str(self.config), "--rules", "timeout:1h", steps=(
            self.pass_deadline, _tool_call(),
        ))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertIn("DurationWatcher: 61s elapsed (threshold: 60s)", self.alerts())

    def test_a_lower_budget_n_lowers_budget_and_still_kills(self):
        """Guard: passes on main too."""
        self.watch("--budget", "10", "--rules", "budget:5", steps=(
            _three_dollars(), _three_dollars(),
        ))
        self.assertEqual(self.cost_alerts(), ["CostWatcher: $6.00 (threshold: $5.0)"])
        self.mocks["kill"].assert_called_once_with(PID, None)

    def test_a_lower_timeout_d_lowers_timeout_and_still_kills(self):
        """Guard: passes on main too."""
        self.watch("--timeout", "2m", "--rules", "timeout:1m", steps=(
            self.pass_deadline, _IDLE_SENTINEL, _IDLE_SENTINEL,
        ))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertIn("DurationWatcher: 61s elapsed (threshold: 60s)", self.alerts())

    def test_a_built_in_replaces_limits_that_only_alert(self):
        """Guard: passes on main too. There is no alert at $6 or 61 s."""
        self.watch("--max-cost", "5", "--max-duration", "60", "--rules", "budget:10,timeout:1h",
                   steps=(self.pass_deadline, _three_dollars(), _three_dollars()))
        self.assertEqual(self.cost_alerts(), [])
        self.assertEqual(self.count("DurationWatcher:"), 0)
        self.mocks["kill"].assert_not_called()


# ---------------------------------------------------------------------------
# per_session_max and command-line cost limits
# ---------------------------------------------------------------------------

class TestPerSessionMaxCostLimit(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = Path(tmp.name) / "traces"
        TraceStore(self.trace_dir, redact=False).create_session(
            SessionMeta(session_id=SESSION, started_at=1.0),
        )

    def cost_limit(self, *flags, per_session_max=5.0):
        argv = ["--trace-dir", str(self.trace_dir), "watch", SESSION, *flags]
        budget = ProjectBudgetConfig(per_session_max=per_session_max)
        with patch("agent_trace.watch.load_project_budget_config", return_value=budget), \
             patch("agent_trace.watch.watch_session") as watch_session_mock:
            self.assertEqual(cmd_watch(build_parser().parse_args(argv)), 0)
        return watch_session_mock.call_args.args[2].max_cost_dollars

    def test_per_session_max_never_raises_a_cost_limit_that_kills(self):
        for flags in (("--budget", "2"), ("--max-cost", "2", "--on-violation", "kill")):
            with self.subTest(flags=flags):
                self.assertEqual(self.cost_limit(*flags), 2.0)

    def test_per_session_max_caps_or_replaces_the_cost_limit(self):
        """Guard: passes on main too. A cost limit that only alerts or pauses,
        and the default, are replaced."""
        for flags in (
            (),
            ("--max-cost", "2"),
            ("--max-cost", "2", "--on-violation", "file"),
            ("--on-violation", "kill"),
            ("--budget", "7"),
            ("--max-cost", "7"),
            ("--max-cost", "7", "--on-violation", "kill"),
        ):
            with self.subTest(flags=flags):
                self.assertEqual(self.cost_limit(*flags), 5.0)

    def test_the_kill_action_alone_does_not_cap_per_session_max_at_the_default(self):
        """Guard: passes on main too. Without --budget, --max-cost or
        max_dollars, the $10 default is not a set limit, so a per_session_max
        above it replaces it, even when the action is kill."""
        config = self.trace_dir.parent / "watch.json"
        config.write_text(json.dumps({"watchers": {"retry": {"alert": "kill"}}}))
        for flags in (("--on-violation", "kill"), ("--config", str(config)), ()):
            with self.subTest(flags=flags):
                self.assertEqual(self.cost_limit(*flags, per_session_max=20.0), 20.0)

    def test_per_session_max_lowers_only_a_config_file_cost_limit_that_kills(self):
        config = self.trace_dir.parent / "watch.json"
        for alert, cost_limit in (
            ("kill", 2.0),
            ("terminal", 5.0),
            ("file", 5.0),
            ("pause", 5.0),
            ("require_approval", 5.0),
        ):
            with self.subTest(alert=alert):
                config.write_text(json.dumps(
                    {"watchers": {"cost": {"max_dollars": 2}, "retry": {"alert": alert}}},
                ))
                self.assertEqual(self.cost_limit("--config", str(config)), cost_limit)


class TestConfigFileProjectBudget(unittest.TestCase):
    """The --config file's budget keys replace the standard project config
    when the file sets weekly or per_session_max."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.root)
        self.trace_dir = self.root / "traces"
        TraceStore(self.trace_dir, redact=False).create_session(
            SessionMeta(session_id=SESSION, started_at=1.0),
        )
        self.config = self.root / "watch.json"

    def standard_project_config(self, budget):
        (self.root / ".agent-strace.json").write_text(json.dumps({"budget": budget}))

    def run_with(self, config):
        """Return the cost limit and the project budget watch gets."""
        self.config.write_text(json.dumps(config))
        argv = ["--trace-dir", str(self.trace_dir), "watch", SESSION, "--config", str(self.config)]
        with patch("agent_trace.watch.watch_session") as watch_session_mock:
            self.assertEqual(cmd_watch(build_parser().parse_args(argv)), 0)
        call = watch_session_mock.call_args
        return call.args[2].max_cost_dollars, call.kwargs["project_budget_config"]

    def test_per_session_max_without_weekly_is_used(self):
        for config in ({"budget": {"per_session_max": 2}}, {"per_session_max": 2}):
            with self.subTest(config=config):
                limit, _budget = self.run_with(config)
                self.assertEqual(limit, 2.0)

    def test_either_key_replaces_the_standard_project_config(self):
        self.standard_project_config({"weekly": 1000, "per_session_max": 1})
        cases = (
            ({"budget": {"per_session_max": 2}}, 2.0),
            # A key that is present counts as set, even when it turns the
            # budget off.
            ({"budget": {"weekly": 0}}, 10.0),
            ({"budget": {"per_session_max": None}}, 10.0),
        )
        for config, cost_limit in cases:
            with self.subTest(config=config):
                limit, budget = self.run_with(config)
                self.assertEqual(limit, cost_limit)
                self.assertEqual(budget.weekly, 0.0)

    def test_without_either_key_the_standard_project_config_is_read(self):
        """Guard: passes on main too."""
        self.standard_project_config({"weekly": 1000, "per_session_max": 1})
        for config in ({}, {"budget": {"warn_at": 0.5}}):
            with self.subTest(config=config):
                limit, budget = self.run_with(config)
                self.assertEqual(limit, 1.0)
                self.assertEqual(budget.weekly, 1000.0)

    def test_a_weekly_budget_in_the_config_file_is_used(self):
        """Guard: passes on main too."""
        self.standard_project_config({"weekly": 1000, "per_session_max": 1})
        limit, budget = self.run_with({"budget": {"weekly": 50}})
        self.assertEqual(limit, 10.0)
        self.assertEqual(budget.weekly, 50.0)


# ---------------------------------------------------------------------------
# Bad limit values are refused at startup
# ---------------------------------------------------------------------------

class TestLimitFlagChecks(unittest.TestCase):
    """cmd_watch refuses a limit that never fires or fires at once, with one
    line and status 1, before it opens the store."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.root)
        self.trace_dir = self.root / "traces"
        TraceStore(self.trace_dir, redact=False).create_session(
            SessionMeta(session_id=SESSION, started_at=1.0),
        )
        # A writable TraceStore writes this file back, so a refusal that
        # opened the store first changes the snapshot.
        (self.trace_dir / ".gitignore").unlink(missing_ok=True)
        self.config = self.root / "watch.json"
        self.config.write_text("{}")
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_STRACE_TENANT_ID", None)

    def _snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(self.root.rglob("*"))
        }

    def _run(self, *flags, config=False):
        argv = ["--trace-dir", str(self.trace_dir), "watch", SESSION, *flags]
        if config:
            argv += ["--config", str(self.config)]
        out, err = io.StringIO(), io.StringIO()
        with patch("agent_trace.watch.watch_session") as watch_session_mock, \
             redirect_stdout(out), redirect_stderr(err):
            status = cmd_watch(build_parser().parse_args(argv))
        return status, out.getvalue(), err.getvalue(), watch_session_mock

    def _assert_refused(self, flags, line):
        for config in (False, True):
            with self.subTest(flags=flags, config=config):
                before = self._snapshot()
                status, out, err, watch_session_mock = self._run(*flags, config=config)
                self.assertEqual((status, out, err), (1, "", line))
                watch_session_mock.assert_not_called()
                self.assertEqual(self._snapshot(), before)

    def test_bad_timeout_values_are_refused(self):
        finite = "[watch] invalid --timeout value: it must be a finite number of seconds\n"
        positive = "[watch] invalid --timeout value: it must be more than 0 seconds\n"
        # --timeout "$T" with T unset must not drop the kill.
        empty = "[watch] invalid --timeout value: empty duration\n"
        for flags, line in (
            (("--timeout", "nan"), finite),
            (("--timeout", "inf"), finite),
            (("--timeout=-inf",), finite),
            (("--timeout", "1e400"), finite),
            (("--timeout", "0"), positive),
            (("--timeout", "-5"), positive),
            (("--timeout", ""), empty),
            (("--timeout=",), empty),
            (("--timeout", "  "), empty),
        ):
            self._assert_refused(flags, line)

    def test_timeout_text_the_parser_does_not_use_is_refused(self):
        # Main read these as 300, 300, 1, 3, 3630, 181, 18000.5, 5 and 300
        # seconds.
        for value in ("-5m", "+5m", "1w", "3y", "1h30", "1e3m", "0.5.5h", "5 apples", "5ms"):
            self._assert_refused(
                (f"--timeout={value}",),
                f"[watch] invalid --timeout value: cannot parse duration: {value!r}\n",
            )

    def test_bad_max_duration_values_are_refused(self):
        for value, problem in (
            ("0", "it must be more than 0 seconds"),
            ("-5", "it must be more than 0 seconds"),
            ("1" + "0" * 400, "it must be a finite number of seconds"),
        ):
            self._assert_refused(
                ("--max-duration", value), f"[watch] invalid --max-duration value: {problem}\n",
            )

    def test_bad_budget_and_max_cost_values_are_refused(self):
        for flag in ("--budget", "--max-cost"):
            finite = f"[watch] invalid {flag} value: it must be a finite number of dollars\n"
            negative = f"[watch] invalid {flag} value: it must not be negative\n"
            for flags, line in (
                ((flag, "nan"), finite),
                ((flag, "inf"), finite),
                ((f"{flag}=-inf",), finite),
                ((flag, "1e400"), finite),
                ((flag, "-1"), negative),
                ((f"{flag}=-0.5",), negative),
            ):
                self._assert_refused(flags, line)

    def test_zero_budget_and_max_cost_are_accepted(self):
        """Guard: passes on main too."""
        for flags in (("--budget", "0"), ("--max-cost", "0")):
            for config in (False, True):
                with self.subTest(flags=flags, config=config):
                    status, _, err, watch_session_mock = self._run(*flags, config=config)
                    self.assertEqual(status, 0, err)
                    watch_session_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Live: the real CLI kills an agent that writes no events
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_SIGNALS, "--agent-pid needs POSIX signals")
class TestTimeoutKillsSilentAgentLive(unittest.TestCase):
    @staticmethod
    def _stop(proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    def test_cli_timeout_kills_an_agent_that_writes_no_events(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        trace_dir = root / "traces"
        TraceStore(trace_dir, redact=False).create_session(
            SessionMeta(session_id=SESSION, started_at=time.time()),
        )
        agent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        self.addCleanup(self._stop, agent)
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(agent_trace.__file__).resolve().parents[1]),
            HOME=str(root),
            DO_NOT_TRACK="1",
            AGENT_STRACE_TELEMETRY="0",
            PYTHONUNBUFFERED="1",
        )
        env.pop("AGENT_STRACE_TENANT_ID", None)
        log = root / "watch.log"
        with open(log, "wb") as log_file:
            watch = subprocess.Popen(
                [
                    sys.executable, "-m", "agent_trace.cli", "--trace-dir", str(trace_dir),
                    "watch", SESSION, "--agent-pid", str(agent.pid), "--timeout", "1s",
                ],
                cwd=root, env=env, stdout=log_file, stderr=subprocess.STDOUT,
            )
        self.addCleanup(self._stop, watch)

        try:
            returncode = agent.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.fail(
                "watch --timeout 1s left a silent agent running for 15 s:\n"
                + log.read_text(errors="replace")
            )
        self.assertEqual(returncode, -signal.SIGTERM)
        watch.send_signal(signal.SIGINT)
        watch.wait(timeout=15)
        output = log.read_text(errors="replace")
        self.assertIn("DurationWatcher:", output)
        self.assertIn(f"Killing agent process {agent.pid}", output)


if __name__ == "__main__":
    unittest.main()
