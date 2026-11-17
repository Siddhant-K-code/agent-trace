"""Tests for a watch kill: SIGTERM, then the post-mortem and --on-death, once.

Watchers and --rules rules kill through one path. SIGCONT follows SIGTERM,
so a stopped agent can act on it. The post-mortem and --on-death follow
only a SIGTERM that watch sent, and after it watch does not kill, pause or
resume the agent again.
"""

import io
import json
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace import watch
from agent_trace.approval import approve_request, deny_request, list_requests, load_request
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.watch import (
    NannyRule,
    WatcherConfig,
    WatchState,
    _check_pause_file,
    _dispatch_alert,
    _dispatch_nanny_rule,
    watch_session,
)

SESSION = "kill-session"
PID = 424242
WEBHOOK = "http://hooks.example/x"
ON_DEATH = "restart-agent --from {post_mortem_path}"
COST_MSG = "CostWatcher: $3.00 (threshold: $2.0)"
RETRY_MSG = "RetryWatcher: command ran 6 times: pytest"
# The agent writes the command, so an alert can quote escape codes and line breaks.
RAW_MSG = "RetryWatcher: command ran 6 times: ls \x1b[2K\n2026-01-01T00:00:00Z  FORGED"
ALREADY = f"Agent pid {PID} was already killed; no signal sent"


def _rule_msg(action: str = "kill") -> str:
    return f"NannyRule 'cost cap': cost_usd > 0.5 → {action}"


def _dollars3() -> TraceEvent:
    """An LLM response with 1M input tokens: $3.00 at the sonnet price."""
    return TraceEvent(
        event_type=EventType.LLM_RESPONSE, timestamp=time.time(), session_id=SESSION,
        data={"input_tokens": 1_000_000, "output_tokens": 0},
    )


class KillTestCase(unittest.TestCase):
    """Record each signal, post-mortem, --on-death run, line and alert, in order.

    os.kill is stubbed, so no process is signalled, and the thread that
    sends SIGKILL 5 seconds after SIGTERM is never started. The post-mortem
    is written to a real store.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name, redact=False)
        self.store.create_session(SessionMeta(session_id=SESSION, started_at=1.0))
        self.post_mortem = self.store._session_dir(SESSION) / "watchdog-postmortem.json"
        self.marker = Path(tmp.name) / ".pause-request"
        self.config = WatcherConfig(
            on_violation="kill", webhook_url=WEBHOOK, on_death_cmd=ON_DEATH,
            alert_log=str(Path(tmp.name) / "alerts.log"),
        )
        self.out = io.StringIO()
        self.calls: list[tuple] = []
        self.reset()

        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("AGENT_STRACE_SLACK_WEBHOOK", "AGENT_STRACE_TEAMS_WEBHOOK"):
            os.environ.pop(name, None)

        record = self.calls.append
        real_write_post_mortem = watch._write_watchdog_postmortem

        def os_kill(pid, sig):
            if sig == 0:  # checks that the process exists; sends no signal
                return
            record(("signal", pid, sig))
            if sig == signal.SIGTERM and self.kill_error is not None:
                raise self.kill_error

        def pid_reused(pid, start_time=None):
            if isinstance(self.reused, list):
                return self.reused.pop(0)
            return self.reused

        def write_post_mortem(*args, **kwargs):
            path = real_write_post_mortem(*args, **kwargs)
            record(("post-mortem", path))
            return path

        class SigkillTimer:
            """Stands in for the thread that sends SIGKILL 5 seconds later."""

            def __init__(self, target, daemon):
                pass

            def start(self):
                record(("sigkill-timer",))

        stubs = {
            "os.kill": os_kill,
            "agent_trace.watch.pid_reused": pid_reused,
            "agent_trace.watch.process_identity": lambda pid: "proc:1",
            "agent_trace.watch._alert_terminal": lambda msg, *a, **k: record(("line", msg)),
            "agent_trace.watch._alert_file": lambda msg, *a, **k: record(("alert-log", msg)),
            "agent_trace.watch._alert_webhook": lambda msg, url: record(("webhook", msg)),
            # --on-death runs its command through Popen(cmd, shell=True).
            "agent_trace.watch.subprocess.Popen": lambda cmd, **k: record(("on-death", cmd)),
            "agent_trace.watch._write_watchdog_postmortem": write_post_mortem,
        }
        for target, stub in stubs.items():
            patcher = patch(target, side_effect=stub)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("threading.Thread", SigkillTimer)
        patcher.start()
        self.addCleanup(patcher.stop)

    def reset(self, agent_pid: int | None = PID) -> None:
        """Start again with an agent that watch has not signalled."""
        self.calls.clear()
        self.state = WatchState(agent_pid=agent_pid, session_id=SESSION)
        self.post_mortem.unlink(missing_ok=True)
        self.kill_error: Exception | None = None
        self.reused: bool | list[bool] = False

    def recorded(self, step: str) -> list:
        """What each *step* was given, in order: signals, paths, commands or lines."""
        return [call[-1] for call in self.calls if call[0] == step]

    def watcher(self, message: str = COST_MSG, action: str | None = None) -> None:
        _dispatch_alert(
            message, self.config, self.state, action=action, store=self.store,
            session_id=SESSION,
        )

    def rule(self, action: str = "kill") -> None:
        rule = NannyRule(name="cost cap", condition="cost_usd > 0.5", action=action)
        _dispatch_nanny_rule(rule, _dollars3(), self.store, SESSION, self.state, self.config)

    def kill(self, path: str) -> str:
        """Kill through *path* ("watcher" or "rule"); return the alert message."""
        if path == "watcher":
            self.watcher()
            return COST_MSG
        self.rule()
        return _rule_msg()

    def tick(self) -> None:
        """One idle poll: approval decisions and the VS Code pause marker."""
        _check_pause_file(self.store, self.state, self.out)

    def reason(self) -> str:
        return json.loads(self.post_mortem.read_text())["reason"]

    def killed_steps(self, message: str) -> list[tuple]:
        """What one kill alert that sends SIGTERM records, in order."""
        return [
            ("line", message),
            ("alert-log", message),
            ("line", f"Killing agent process {PID}"),
            ("signal", PID, signal.SIGTERM),
            ("signal", PID, signal.SIGCONT),
            ("sigkill-timer",),
            ("post-mortem", self.post_mortem),
            ("line", f"Post-mortem written to {self.post_mortem}"),
            ("on-death", f"restart-agent --from {self.post_mortem}"),
            ("webhook", message),
        ]


class TestKill(KillTestCase):
    def test_rule_kill_runs_on_death_once_with_the_post_mortem(self):
        self.rule()
        self.assertEqual(self.recorded("signal"), [signal.SIGTERM, signal.SIGCONT])
        self.assertEqual(self.recorded("post-mortem"), [self.post_mortem])
        self.assertEqual(self.reason(), _rule_msg())
        self.assertEqual(self.recorded("on-death"), [f"restart-agent --from {self.post_mortem}"])

    def test_order(self):
        # The alert line, the alert log, SIGTERM, the post-mortem, --on-death
        # with its path, then the network.
        for path in ("watcher", "rule"):
            with self.subTest(path=path):
                self.reset()
                message = self.kill(path)
                self.assertEqual(self.calls, self.killed_steps(message))

    def test_post_mortem_keeps_the_raw_text(self):
        # Guard: unchanged from main. The post-mortem is JSON, which escapes
        # the agent's text.
        self.watcher(RAW_MSG)
        post_mortem = json.loads(self.post_mortem.read_text())
        self.assertEqual(post_mortem["reason"], RAW_MSG)
        self.assertIn(f" Reason: {RAW_MSG}. Resume ", post_mortem["recovery_context"])


class TestNotKilled(KillTestCase):
    """Watch did not send SIGTERM, so it writes no post-mortem and runs no --on-death."""

    def check(self, path, line, signals, *, agent_pid=PID, reused=False, kill_error=None):
        self.reset(agent_pid)
        self.reused, self.kill_error = reused, kill_error
        self.kill(path)
        self.assertEqual(self.recorded("signal"), signals)
        self.assertNotIn(("sigkill-timer",), self.calls)
        self.assertEqual(self.recorded("post-mortem"), [])
        self.assertFalse(self.post_mortem.exists())
        self.assertEqual(self.recorded("on-death"), [])
        lines = self.recorded("line")
        self.assertEqual([text for text in lines if text.startswith(line)], [lines[-1]])

    def no_pid(self, path):
        self.check(path, "Cannot kill the agent: no agent PID is known", [], agent_pid=None)

    def reused_pid(self, path):
        self.check(
            path, f"Agent pid {PID}: the PID now belongs to another process", [], reused=True,
        )

    def reused_at_sigterm(self, path):
        # The check before the kill passes; the one in _kill_process does not.
        self.check(
            path, f"Agent pid {PID} was not killed: the PID now belongs to another process",
            [], reused=[False, True],
        )

    def no_such_process(self, path):
        self.check(
            path, f"Agent pid {PID} was not killed: no process has that PID",
            [signal.SIGTERM], kill_error=ProcessLookupError(),
        )

    def not_permitted(self, path):
        self.check(
            path, f"Agent pid {PID} was not killed: not allowed to signal it",
            [signal.SIGTERM], kill_error=PermissionError(),
        )

    def test_watcher_without_agent_pid(self):
        self.no_pid("watcher")

    def test_rule_without_agent_pid(self):
        self.no_pid("rule")

    def test_watcher_with_reused_pid(self):
        self.reused_pid("watcher")

    def test_rule_with_reused_pid(self):
        self.reused_pid("rule")

    def test_watcher_with_pid_reused_at_sigterm(self):
        self.reused_at_sigterm("watcher")

    def test_rule_with_pid_reused_at_sigterm(self):
        self.reused_at_sigterm("rule")

    def test_watcher_sigterm_finds_no_process(self):
        self.no_such_process("watcher")

    def test_rule_sigterm_finds_no_process(self):
        self.no_such_process("rule")

    def test_watcher_sigterm_not_permitted(self):
        self.not_permitted("watcher")

    def test_rule_sigterm_not_permitted(self):
        self.not_permitted("rule")


class TestKillPausedAgent(KillTestCase):
    """SIGCONT follows SIGTERM, so a stopped agent can act on it, and the
    kill ends the pause that watch holds."""

    def test_kill_continues_an_agent_paused_by_a_rule(self):
        self.rule("pause")
        self.assertEqual(self.recorded("signal"), [signal.SIGSTOP])
        self.calls.clear()
        self.kill("watcher")
        self.assertEqual(self.calls, self.killed_steps(COST_MSG))
        self.assertFalse(self.state.paused)

    def test_kill_leaves_a_pending_approval_request_pending(self):
        # Deciding it later neither prints a pause line nor sends a signal.
        for decision in ("approve", "deny"):
            with self.subTest(decision=decision):
                self.reset()
                self.out = io.StringIO()
                self.rule("require_approval")
                request_id = self.state.approval_request_id
                self.assertEqual(self.recorded("signal"), [signal.SIGSTOP])
                self.calls.clear()
                self.kill("watcher")
                self.assertEqual(self.calls, self.killed_steps(COST_MSG))
                self.assertFalse(self.state.paused)
                self.assertIsNone(self.state.approval_request_id)
                self.assertEqual(load_request(self.store, request_id).state, "pending")

                self.calls.clear()
                if decision == "approve":
                    approve_request(self.store, request_id, resume_agent=False)
                else:
                    deny_request(self.store, request_id, kill_agent=False)
                self.tick()
                self.assertEqual(self.calls, [])
                self.assertEqual(self.out.getvalue(), "")
                self.assertFalse(self.state.paused)


class TestKillOnce(KillTestCase):
    """After SIGTERM, later alerts are still shown, logged and sent, but
    watch does not kill, pause or resume the agent again."""

    def run_watch(self, events: list[TraceEvent], max_cost: float) -> None:
        self.config.max_cost_dollars = max_cost
        rule = NannyRule(name="cost cap", condition="cost_usd > 0.5", action="kill")
        with patch("agent_trace.watch._tail_events", return_value=iter(events)):
            watch_session(
                self.store, SESSION, self.config, out=self.out, nanny_rules=[rule],
                agent_pid=PID,
            )

    def test_watcher_and_rule_kill_on_the_same_event(self):
        self.run_watch([_dollars3()], max_cost=2.0)
        rule_msg = _rule_msg()
        self.assertEqual(self.calls, self.killed_steps(COST_MSG) + [
            ("line", rule_msg),
            ("alert-log", rule_msg),
            ("line", ALREADY),
            ("webhook", rule_msg),
        ])
        self.assertEqual(self.reason(), COST_MSG)

    def test_rule_kill_then_watcher_kill_on_a_later_event(self):
        self.run_watch([_dollars3(), _dollars3()], max_cost=5.0)
        cost_msg = "CostWatcher: $6.00 (threshold: $5.0)"
        self.assertEqual(self.calls, self.killed_steps(_rule_msg()) + [
            ("line", cost_msg),
            ("alert-log", cost_msg),
            ("line", ALREADY),
            ("webhook", cost_msg),
        ])
        self.assertEqual(self.reason(), _rule_msg())

    def test_later_kill(self):
        for first, later in (("rule", "watcher"), ("watcher", "rule")):
            with self.subTest(first=first, later=later):
                self.reset()
                first_msg = self.kill(first)
                self.calls.clear()
                later_msg = self.kill(later)
                self.assertEqual(self.calls, [
                    ("line", later_msg),
                    ("alert-log", later_msg),
                    ("line", ALREADY),
                    ("webhook", later_msg),
                ])
                self.assertEqual(self.reason(), first_msg)

    def test_later_pause(self):
        for path, message in (("watcher", RETRY_MSG), ("rule", _rule_msg("pause"))):
            with self.subTest(path=path):
                self.reset()
                self.kill("watcher")
                self.calls.clear()
                if path == "watcher":
                    self.watcher(RETRY_MSG, action="pause")
                else:
                    self.rule("pause")
                self.assertEqual(self.calls, [
                    ("line", message),
                    ("alert-log", message),
                    ("line", ALREADY),
                    ("webhook", message),
                ])
                self.assertFalse(self.state.paused)

    def test_later_require_approval(self):
        for path, message in (("watcher", RETRY_MSG), ("rule", _rule_msg("require_approval"))):
            with self.subTest(path=path):
                self.reset()
                self.kill("rule")
                self.calls.clear()
                if path == "watcher":
                    self.watcher(RETRY_MSG, action="require_approval")
                else:
                    self.rule("require_approval")
                self.assertEqual(self.calls, [
                    ("line", message),
                    ("line", f"{ALREADY} and no approval request created"),
                ])
                self.assertEqual(list_requests(self.store), [])
                self.assertFalse(self.state.paused)

    def test_editor_pause_and_resume_after_a_kill(self):
        self.kill("watcher")
        self.marker.write_text(SESSION)
        self.tick()
        self.marker.unlink()
        self.tick()
        self.assertEqual(self.recorded("signal"), [signal.SIGTERM, signal.SIGCONT])
        self.assertFalse(self.state.paused)

    def test_editor_resume_after_a_kill_sends_no_second_sigcont(self):
        self.marker.write_text(SESSION)
        self.tick()
        self.kill("rule")
        self.assertFalse(self.state.paused)
        self.marker.unlink()
        self.tick()
        self.assertEqual(
            self.recorded("signal"), [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT],
        )

    def test_later_alerts_that_signal_nothing_are_unchanged(self):
        # Guard: unchanged from main. They get no already-killed line.
        for path, message, expected in (
            ("rule", _rule_msg("alert"), ["line", "webhook"]),
            ("watcher", RETRY_MSG, ["line", "alert-log", "webhook"]),
        ):
            with self.subTest(path=path):
                self.reset()
                self.kill("rule")
                self.calls.clear()
                if path == "watcher":
                    self.watcher(RETRY_MSG, action="file")
                else:
                    self.rule("alert")
                self.assertEqual(self.calls, [(step, message) for step in expected])


class TestDryRunKill(KillTestCase):
    def test_dry_run_kill_is_unchanged_and_is_not_the_kill(self):
        # Guard: unchanged from main.
        for path in ("watcher", "rule"):
            with self.subTest(path=path):
                self.reset()
                if path == "watcher":
                    message = COST_MSG
                    _dispatch_alert(
                        message, self.config, self.state, dry_run=True, store=self.store,
                        session_id=SESSION,
                    )
                else:
                    message = _rule_msg()
                    rule = NannyRule(name="cost cap", condition="cost_usd > 0.5", action="kill")
                    _dispatch_nanny_rule(
                        rule, _dollars3(), self.store, SESSION, self.state, self.config,
                        dry_run=True,
                    )
                self.assertEqual(self.calls, [
                    ("line", message),
                    ("alert-log", message),
                    ("line", "[dry-run] would kill agent process"),
                    ("webhook", message),
                ])
                self.assertFalse(self.post_mortem.exists())
                # A dry run is not the kill: a pause still stops the agent.
                self.watcher(RETRY_MSG, action="pause")
                self.assertEqual(self.recorded("signal"), [signal.SIGSTOP])


if __name__ == "__main__":
    unittest.main()
