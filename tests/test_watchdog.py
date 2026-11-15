"""Tests for watchdog mode: --timeout, --budget, --on-death, post-mortem JSON."""

import io
import json
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


@unittest.skipUnless(HAS_SIGNALS, "--agent-pid needs POSIX signals")
class WatchdogLimitTest(unittest.TestCase):
    """Run cmd_watch from parsed flags with the event stream and signals faked.

    watch() takes steps: events and _IDLE_SENTINEL are what the tail yields,
    one per poll cycle, and callables run between them. The fake agent PID
    answers signal 0 until agent_exits() runs.
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
        self.loop_state = None
        self.yielded = 0
        self.mocks = {}
        for name, target, kwargs in (
            ("pause", "agent_trace.watch._pause_process", {}),
            ("resume", "agent_trace.watch._resume_process", {}),
            ("kill", "agent_trace.watch._kill_process", {}),
            ("os_kill", "os.kill", {"side_effect": self._signal}),
            ("identity", "agent_trace.watch.process_identity", {"return_value": None}),
            ("terminal", "agent_trace.watch._alert_terminal", {}),
        ):
            p = patch(target, **kwargs)
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)

    def _signal(self, pid, sig):
        if not self.agent_alive:
            raise ProcessLookupError(pid)

    def pass_deadline(self):
        """Move the watch start back so a 60 s --timeout has passed."""
        self.loop_state.start_time -= 61

    def pass_grace(self):
        """Move the watch start back past the 10 s kill grace too."""
        self.loop_state.start_time -= 11

    def agent_exits(self):
        self.agent_alive = False

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


class TestTimeoutKills(WatchdogLimitTest):
    def test_timeout_kills_an_agent_that_writes_no_events(self):
        self.watch("--timeout", "1m", steps=(
            self.pass_deadline, _IDLE_SENTINEL, _IDLE_SENTINEL,
        ))
        self.mocks["kill"].assert_called_once_with(PID, None)
        self.assertIn("DurationWatcher: 61s elapsed (threshold: 60s)", self.alerts())
        self.assertIn(f"Killing agent process {PID}", self.alerts())

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

    def cost_limit(self, *flags):
        argv = ["--trace-dir", str(self.trace_dir), "watch", SESSION, *flags]
        budget = ProjectBudgetConfig(per_session_max=5.0)
        with patch("agent_trace.watch.load_project_budget_config", return_value=budget), \
             patch("agent_trace.watch.watch_session") as watch_session_mock:
            self.assertEqual(cmd_watch(build_parser().parse_args(argv)), 0)
        return watch_session_mock.call_args.args[2].max_cost_dollars

    def test_per_session_max_never_raises_a_command_line_cost_limit(self):
        for flags in (("--budget", "2"), ("--max-cost", "2")):
            with self.subTest(flags=flags):
                self.assertEqual(self.cost_limit(*flags), 2.0)

    def test_per_session_max_caps_or_replaces_the_cost_limit(self):
        """Guard: passes on main too."""
        for flags in ((), ("--budget", "7"), ("--max-cost", "7")):
            with self.subTest(flags=flags):
                self.assertEqual(self.cost_limit(*flags), 5.0)


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
        for flags, line in (
            (("--timeout", "nan"), finite),
            (("--timeout", "inf"), finite),
            (("--timeout=-inf",), finite),
            (("--timeout", "1e400"), finite),
            (("--timeout", "0"), positive),
            (("--timeout", "-5"), positive),
            (("--timeout=-5m",), positive),
        ):
            self._assert_refused(flags, line)

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
