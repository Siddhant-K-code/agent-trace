"""Tests for a watch kill: SIGTERM, then the post-mortem and --on-death, once.

Watchers and --rules rules kill through one path. SIGCONT follows SIGTERM,
so a stopped agent can act on it. The post-mortem and --on-death follow
only a SIGTERM that watch sent, and after it watch does not kill, pause or
resume the agent again.

On POSIX a kill also reaches the processes the agent started. They are
listed once, before the agent's SIGTERM, and get SIGTERM, SIGCONT and the
SIGKILL 5 seconds later. watch_session waits for that SIGKILL before it
returns.
"""

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:  # keeps product telemetry off in this process and its children
    from _no_telemetry import without_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests._no_telemetry import without_telemetry

from agent_trace import watch
from agent_trace.approval import approve_request, deny_request, list_requests, load_request
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.process_identity import ProcessEntry, ProcessListError, process_identity
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

    os.kill is stubbed, so no process is signalled. The agent has no child
    processes. The thread that sends SIGKILL 5 seconds after SIGTERM is
    never started, and watch_session's wait for it is recorded, not run.
    The post-mortem is written to a real store.
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
            "agent_trace.watch.list_processes": lambda: {},
            "agent_trace.watch._wait_for_sigkill": lambda out: record(("sigkill-wait",)),
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
        for target, new in (("threading.Thread", SigkillTimer),
                            ("agent_trace.watch._sigkill_steps", [])):
            patcher = patch(target, new)
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
            ("sigkill-wait",),
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
            ("sigkill-wait",),
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


# ---------------------------------------------------------------------------
# The agent's child processes, with a fake process table
# ---------------------------------------------------------------------------

# Fake PIDs. Linux and macOS never assign a PID above 2**22, so none of
# them is this test's own PID.
AGENT = 5_000_000
CHILD_A = 5_000_001
CHILD_B = 5_000_002
GRANDCHILD = 5_000_011
NEW_A = 5_000_021
NEW_B = 5_000_022


def _table(*rows: tuple[int, int], changed: tuple = ()) -> dict:
    """PID → (parent PID, identity). Each identity is "id:PID", or "id:other" if changed."""
    return {
        pid: ProcessEntry(ppid, "id:other" if pid in changed else f"id:{pid}")
        for pid, ppid in rows
    }


def _tree() -> dict:
    """The agent, its two children and a grandchild. Watch runs elsewhere."""
    return _table(
        (1, 0), (AGENT, 1), (CHILD_A, AGENT), (CHILD_B, AGENT), (GRANDCHILD, CHILD_A),
        (os.getpid(), 1),
    )


@unittest.skipUnless(os.name == "posix", "a kill reaches child processes on POSIX only")
class TestKillChildProcesses(unittest.TestCase):
    """_kill_process with a fake process table. No process is signalled.

    os.kill is stubbed. The SIGKILL step is run at once, in the test.
    """

    def setUp(self):
        self.calls: list[tuple] = []
        self.lines: list[str] = []
        self.errors: dict[tuple[int, int], OSError] = {}
        self.reused: set[int] = set()
        self.tables: list = [_tree()]
        self.steps: list = []
        record = self.calls.append

        def os_kill(pid, sig):
            record(("signal", pid, sig))
            if (pid, sig) in self.errors:
                raise self.errors[(pid, sig)]

        def pid_reused(pid, start_time=None):
            record(("reused?", pid))
            return pid in self.reused

        def list_processes():
            record(("list",))
            table = self.tables.pop(0)
            if isinstance(table, Exception):
                raise table
            return table

        class Thread:
            """The SIGKILL step's thread, never started."""

            def __init__(self, target, daemon):
                pass

            def start(self):
                pass

        for target, new in (
            ("os.kill", os_kill),
            ("agent_trace.watch.pid_reused", pid_reused),
            ("agent_trace.watch.list_processes", list_processes),
            ("agent_trace.watch._alert_terminal", self.lines.append),
            ("agent_trace.watch._sigkill_steps", self.steps),
            ("agent_trace.watch.time.sleep", lambda seconds: None),
            ("threading.Thread", Thread),
        ):
            patcher = patch(target, new)
            patcher.start()
            self.addCleanup(patcher.stop)

    def kill(self) -> bool:
        return watch._kill_process(AGENT, "id:agent")

    def run_step(self, table: dict) -> list[tuple]:
        """Run the SIGKILL step with *table* as the new listing; return its calls."""
        [step] = self.steps
        self.tables.append(table)
        self.calls.clear()
        step.run()
        return self.calls

    def sent(self, sig: int) -> list[int]:
        return [call[1] for call in self.calls if call[0] == "signal" and call[2] == sig]

    def not_killed(self, line: str, calls: list[tuple]) -> None:
        self.assertFalse(self.kill())
        self.assertEqual(self.calls, calls)
        self.assertEqual(self.lines, [line])
        self.assertEqual(self.steps, [])

    def test_each_signal_goes_to_the_agent_then_its_child_processes(self):
        children = [CHILD_A, CHILD_B, GRANDCHILD]
        self.assertTrue(self.kill())
        self.assertEqual(self.calls, [
            ("reused?", AGENT),
            ("signal", AGENT, 0),
            ("list",),
            ("signal", AGENT, signal.SIGTERM),
            *[("signal", pid, signal.SIGTERM) for pid in children],
            ("reused?", AGENT),
            ("signal", AGENT, signal.SIGCONT),
            *[("signal", pid, signal.SIGCONT) for pid in children],
        ])
        self.assertEqual(
            self.lines, [f"Killing 3 child processes: {CHILD_A}, {CHILD_B}, {GRANDCHILD}"],
        )
        self.assertEqual(self.run_step(_tree()), [
            ("reused?", AGENT),
            ("signal", AGENT, signal.SIGKILL),
            ("list",),
            *[("signal", pid, signal.SIGKILL) for pid in children],
        ])
        self.assertEqual(len(self.lines), 1)

    def test_sigkill_goes_only_to_a_listed_process_that_is_unchanged(self):
        self.kill()
        # CHILD_A's PID now names another process and CHILD_B is gone. NEW_A
        # and NEW_B started after the kill's listing.
        table = _table(
            (1, 0), (AGENT, 1), (CHILD_A, 1), (GRANDCHILD, 1), (NEW_A, AGENT),
            (NEW_B, GRANDCHILD), changed=(CHILD_A,),
        )
        self.assertEqual(self.run_step(table), [
            ("reused?", AGENT),
            ("signal", AGENT, signal.SIGKILL),
            ("list",),
            ("signal", GRANDCHILD, signal.SIGKILL),
        ])

    def test_a_child_process_that_cannot_be_signalled(self):
        # One is not ours to signal and one exited: only the first gets a line.
        self.errors[(CHILD_B, signal.SIGTERM)] = PermissionError()
        self.errors[(GRANDCHILD, signal.SIGTERM)] = ProcessLookupError()
        self.assertTrue(self.kill())
        self.assertEqual(self.sent(signal.SIGTERM), [AGENT, CHILD_A, CHILD_B, GRANDCHILD])
        self.assertEqual(self.sent(signal.SIGCONT), [AGENT, CHILD_A])
        self.assertEqual(self.lines, [
            f"Killing 1 child process: {CHILD_A}",
            f"Child process {CHILD_B} not killed: not allowed to signal it",
        ])
        self.run_step(_tree())
        self.assertEqual(self.sent(signal.SIGKILL), [AGENT, CHILD_A])

    def test_child_processes_that_cannot_be_signalled(self):
        for pid in (CHILD_A, CHILD_B):
            self.errors[(pid, signal.SIGTERM)] = PermissionError()
        self.assertTrue(self.kill())
        self.assertEqual(self.lines, [
            f"Killing 1 child process: {GRANDCHILD}",
            f"Child processes {CHILD_A}, {CHILD_B} not killed: not allowed to signal them",
        ])

    def test_the_line_names_ten_pids_at_most(self):
        children = [AGENT + n for n in range(1, 13)]
        self.tables = [_table((AGENT, 1), *[(pid, AGENT) for pid in children])]
        self.assertTrue(self.kill())
        shown = ", ".join(str(pid) for pid in children[:10])
        self.assertEqual(self.lines, [f"Killing 12 child processes: {shown} and 2 more"])

    def test_listing_fails(self):
        self.tables = [ProcessListError("ps did not finish within 2s")]
        self.assertTrue(self.kill())
        self.assertEqual(self.calls, [
            ("reused?", AGENT),
            ("signal", AGENT, 0),
            ("list",),
            ("signal", AGENT, signal.SIGTERM),
            ("reused?", AGENT),
            ("signal", AGENT, signal.SIGCONT),
        ])
        self.assertEqual(self.lines, [
            "Could not list the agent's child processes, so they were not killed: "
            "ps did not finish within 2s",
        ])
        self.assertEqual(self.run_step(_tree()), [
            ("reused?", AGENT), ("signal", AGENT, signal.SIGKILL),
        ])

    def test_reused_pid_lists_nothing(self):
        self.reused.add(AGENT)
        self.not_killed(
            f"Agent pid {AGENT} was not killed: the PID now belongs to another process",
            [("reused?", AGENT)],
        )

    def test_agent_that_cannot_be_signalled_lists_nothing(self):
        for error, reason in (
            (PermissionError(), "not allowed to signal it"),
            (ProcessLookupError(), "no process has that PID"),
        ):
            with self.subTest(reason=reason):
                self.calls.clear()
                self.lines.clear()
                for sig in (0, signal.SIGTERM, signal.SIGCONT, signal.SIGKILL):
                    self.errors[(AGENT, sig)] = error
                self.assertFalse(self.kill())
                self.assertEqual(self.lines, [f"Agent pid {AGENT} was not killed: {reason}"])
                self.assertNotIn(("list",), self.calls)
                # One attempt, at the agent, and nothing after it.
                self.assertEqual([c[1] for c in self.calls if c[0] == "signal"], [AGENT])
                self.assertEqual(self.steps, [])

    def test_failed_sigterm_signals_no_child_process(self):
        # The agent exited between the check and SIGTERM.
        self.errors[(AGENT, signal.SIGTERM)] = ProcessLookupError()
        self.assertFalse(self.kill())
        self.assertEqual(
            self.lines, [f"Agent pid {AGENT} was not killed: no process has that PID"],
        )
        sent = [c for c in self.calls if c[0] == "signal" and c[2] != 0]
        self.assertEqual(sent, [("signal", AGENT, signal.SIGTERM)])
        self.assertEqual(self.steps, [])

    def test_watch_and_the_processes_above_it_are_left_out(self):
        # Watch runs under a shell the agent started, and has a child of its own.
        shell, watch_child = AGENT + 3, AGENT + 4
        me = os.getpid()
        self.tables = [_table(
            (1, 0), (AGENT, 1), (shell, AGENT), (me, shell), (watch_child, me), (CHILD_B, AGENT),
        )]
        self.assertTrue(self.kill())
        self.assertEqual(self.sent(signal.SIGTERM), [AGENT, CHILD_B])
        self.assertEqual(self.lines, [f"Killing 1 child process: {CHILD_B}"])


class _FakeStep:
    def __init__(self, due: float, pids: list[int], done=None):
        self.due = due
        self.done = done or threading.Event()
        self.pids = pids

    def settled(self) -> bool:
        return self.done.is_set()

    def running(self) -> list[int]:
        return self.pids


class _Interrupted:
    """A SIGKILL step's done event whose wait gets Ctrl-C."""

    def is_set(self) -> bool:
        return False

    def wait(self, timeout=None):
        raise KeyboardInterrupt


class TestWaitForSigkill(unittest.TestCase):
    """watch_session does not return before a kill's SIGKILL step."""

    def wait(self, steps: list, notice_after: float = 1.0, after_due: float = 2.0):
        """Run the wait for *steps*; return its output and how long it took."""
        registry = list(steps)
        out = io.StringIO()
        with patch.object(watch, "_sigkill_steps", registry), \
                patch.object(watch, "_SIGKILL_WAIT_NOTICE_AFTER", notice_after), \
                patch.object(watch, "_SIGKILL_WAIT_AFTER_DUE", after_due):
            start = time.monotonic()
            watch._wait_for_sigkill(out)
            elapsed = time.monotonic() - start
        self.assertEqual(registry, [])
        return out.getvalue(), elapsed

    def test_no_kill_no_wait(self):
        output, elapsed = self.wait([])
        self.assertEqual(output, "")
        self.assertLess(elapsed, 0.1)

    def test_waits_for_sigkill_and_says_why(self):
        step = _FakeStep(time.monotonic() + 0.6, [AGENT])
        timer = threading.Timer(0.3, step.done.set)
        timer.start()
        self.addCleanup(timer.cancel)
        output, elapsed = self.wait([step], notice_after=0.1)
        self.assertEqual(
            output,
            "[watch] Waiting 1s to send SIGKILL to processes that did not exit on SIGTERM; "
            "press Ctrl-C to stop waiting.\n",
        )
        self.assertGreaterEqual(elapsed, 0.25)
        self.assertLess(elapsed, 0.6)

    def test_the_wait_ends_after_the_due_time(self):
        step = _FakeStep(time.monotonic() + 0.1, [AGENT])
        output, elapsed = self.wait([step], after_due=0.2)
        self.assertEqual(output, "")
        self.assertGreaterEqual(elapsed, 0.25)
        self.assertLess(elapsed, 1.0)

    def test_ctrl_c_ends_the_wait(self):
        step = _FakeStep(time.monotonic() + 5, [AGENT, CHILD_A], _Interrupted())
        output, elapsed = self.wait([step])
        self.assertEqual(
            output, f"\n[watch] Stopped waiting before SIGKILL. Still running: {AGENT}, {CHILD_A}\n",
        )
        self.assertLess(elapsed, 1.0)

    def test_a_kill_is_settled_once_its_processes_are_gone(self):
        step = watch._SigkillStep(AGENT, "id:agent", [(CHILD_A, "id:a")], time.monotonic())
        alive = {AGENT, CHILD_A}

        def os_kill(pid, sig):
            if pid not in alive:
                raise ProcessLookupError

        with patch("os.kill", os_kill):
            self.assertFalse(step.settled())
            alive.discard(AGENT)
            self.assertEqual(step.running(), [CHILD_A])
            self.assertFalse(step.settled())
            alive.clear()
            self.assertTrue(step.settled())


# ---------------------------------------------------------------------------
# The agent's child processes, live
# ---------------------------------------------------------------------------

_TREE_SCRIPT = '''\
"""A fake agent and the processes it starts, for tests/test_watch_kill.py.

Usage: python tree.py FOLDER ROLE [ARGS]
Each process writes its PID to FOLDER/ROLE once it is ready. None of them
runs for more than 30 seconds.
"""
import os
import signal
import subprocess
import sys
import time

FOLDER, ROLE, ARGS = sys.argv[1], sys.argv[2], sys.argv[3:]


def write(name, text):
    tmp = os.path.join(FOLDER, name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, os.path.join(FOLDER, name))


def start(role, *args, **kwargs):
    return subprocess.Popen([sys.executable, __file__, FOLDER, role, *args], **kwargs)


def wait_for(names):
    deadline = time.monotonic() + 20
    while not all(os.path.exists(os.path.join(FOLDER, name)) for name in names):
        if time.monotonic() > deadline:
            sys.exit(1)
        time.sleep(0.01)


def write_mark(signum, frame):
    write("mark", "SIGTERM")
    os._exit(0)


def exit_slowly(signum, frame):
    # Like a server that shuts down: it still runs while watch prints its lines.
    time.sleep(1)
    os._exit(0)


def run_hook(signum, frame):
    # Like a SessionEnd hook: the agent starts it on SIGTERM and waits.
    start("hook").wait()
    os._exit(0)


if ROLE == "launcher":
    # Start the agent and exit, so the agent's parent is not the test.
    start(*ARGS)
    sys.exit(0)
if ROLE == "hook":
    write("hook", str(os.getpid()))
    time.sleep(1)
    write("hook-mark", "done")
    sys.exit(0)
if ROLE in ("agent", "hooked-agent"):
    if ROLE == "hooked-agent":
        signal.signal(signal.SIGTERM, run_hook)
    for child in ARGS:
        start(child, *(["--token", "secret-token-123"] if child == "secret" else []))
    wait_for(ARGS + (["own-session"] if "middle" in ARGS else []))
    write("agent", str(os.getpid()))
else:
    if ROLE == "ignores-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif ROLE == "stopped":
        signal.signal(signal.SIGTERM, write_mark)
    elif ROLE == "secret":
        signal.signal(signal.SIGTERM, exit_slowly)
    elif ROLE == "middle":
        start("own-session", start_new_session=True)
    write(ROLE, str(os.getpid()))
time.sleep(30)
'''


def _wait_until(check, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _state(pid: int) -> str:
    """Return the state letter of *pid*, such as S, T or Z, or "" once it is gone."""
    if os.path.exists("/proc/self/stat"):
        try:
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rpartition(")")[2].split()[0]
        except (OSError, IndexError):
            return ""
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5,
    )
    return result.stdout.strip()[:1]


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait()


class _Tree:
    """Processes started from _TREE_SCRIPT, by role.

    Cleanup sends SIGKILL to each one that still has the identity it had
    when it started, so a failed test leaves no process behind and signals
    no process it did not start.
    """

    def __init__(self, test: unittest.TestCase):
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        self.script = self.folder / "tree.py"
        self.script.write_text(_TREE_SCRIPT)
        self.pids: dict[str, int] = {}
        self.identities: dict[str, str] = {}
        test.addCleanup(self.kill_leftovers)

    def command(self, role: str, *args: str) -> list[str]:
        return [sys.executable, str(self.script), str(self.folder), role, *args]

    def start(self, agent_role: str, *children: str) -> None:
        """Start an agent whose parent is not this test; wait until all of it runs."""
        launcher = subprocess.Popen(
            self.command("launcher", agent_role, *children), start_new_session=True,
        )
        launcher.wait(timeout=20)
        self.read("agent")
        for role in children + (("own-session",) if "middle" in children else ()):
            self.read(role)

    def read(self, role: str, timeout: float = 20.0) -> int:
        path = self.folder / role
        if not _wait_until(path.exists, timeout):
            raise AssertionError(f"the {role} process did not start")
        pid = int(path.read_text())
        identity = process_identity(pid)
        if identity is None:
            raise AssertionError(f"no identity for the {role} process {pid}")
        self.pids[role] = pid
        self.identities[role] = identity
        return pid

    def gone(self, role: str) -> bool:
        pid = self.pids[role]
        return process_identity(pid) != self.identities[role] or _state(pid) in ("", "Z")

    def kill_leftovers(self) -> None:
        for role, pid in self.pids.items():
            if process_identity(pid) == self.identities[role]:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass


@unittest.skipUnless(os.name == "posix", "a kill reaches child processes on POSIX only")
class TestKillProcessTreeLive(unittest.TestCase):
    """Real kills of a fake agent and the processes it started."""

    def test_kill_ends_every_process_the_agent_started(self):
        tree = _Tree(self)
        tree.start("agent", "ignores-term", "middle", "stopped")
        pids = tree.pids
        # The grandchild has its own session, as a command run in a pty does.
        own = pids["own-session"]
        self.assertEqual(os.getsid(own), own)
        self.assertNotEqual(os.getsid(own), os.getsid(pids["agent"]))
        os.kill(pids["stopped"], signal.SIGSTOP)
        self.assertTrue(_wait_until(lambda: _state(pids["stopped"]) == "T", 5))

        lines: list[str] = []
        steps: list = []
        with patch.object(watch, "_SIGKILL_DELAY", 3.0), \
                patch.object(watch, "_sigkill_steps", steps), \
                patch.object(watch, "_alert_terminal", lines.append):
            self.assertTrue(watch._kill_process(pids["agent"], tree.identities["agent"]))
            [step] = steps
            # The stopped child ran its SIGTERM handler before SIGKILL was
            # due, so it got SIGCONT.
            self.assertTrue(_wait_until((tree.folder / "mark").exists, 2))
            self.assertFalse(step.done.is_set())
            ended = ("agent", "middle", "own-session", "stopped")
            self.assertTrue(_wait_until(lambda: all(tree.gone(role) for role in ended), 2))
            # Only SIGKILL ends the child that ignores SIGTERM.
            self.assertFalse(tree.gone("ignores-term"))
            self.assertFalse(step.done.is_set())
            self.assertTrue(step.done.wait(10))
            self.assertTrue(_wait_until(lambda: tree.gone("ignores-term"), 2))
        first = sorted(pids[role] for role in ("ignores-term", "middle", "stopped"))
        listed = ", ".join(str(pid) for pid in first + [own])
        self.assertEqual(lines, [f"Killing 4 child processes: {listed}"])

    def test_a_process_the_agent_starts_on_sigterm_is_not_signalled(self):
        # Watch lists the processes once, before SIGTERM. The hook the agent
        # starts on SIGTERM runs to its end, even after the agent's SIGKILL.
        tree = _Tree(self)
        tree.start("hooked-agent", "plain")
        lines: list[str] = []
        steps: list = []
        with patch.object(watch, "_SIGKILL_DELAY", 0.5), \
                patch.object(watch, "_sigkill_steps", steps), \
                patch.object(watch, "_alert_terminal", lines.append):
            self.assertTrue(watch._kill_process(tree.pids["agent"], tree.identities["agent"]))
            tree.read("hook", 5)
            self.assertTrue(steps[0].done.wait(10))
            self.assertTrue(_wait_until(lambda: tree.gone("agent"), 2))
            self.assertTrue(_wait_until((tree.folder / "hook-mark").exists, 5))
        self.assertEqual(lines, [f"Killing 1 child process: {tree.pids['plain']}"])

    def test_watch_sends_sigkill_before_it_returns_on_session_end(self):
        tree = _Tree(self)
        tree.start("agent", "ignores-term")
        store = TraceStore(tree.folder / "traces", redact=False)
        store.create_session(SessionMeta(session_id=SESSION, started_at=time.time()))
        config = WatcherConfig(
            on_violation="kill", alert_log=str(tree.folder / "alerts.log"),
        )
        rule = NannyRule(name="cost cap", condition="cost_usd > 0.5", action="kill")
        end = TraceEvent(
            event_type=EventType.SESSION_END, timestamp=time.time(), session_id=SESSION,
        )
        out = io.StringIO()
        lines: list[str] = []
        with patch.dict(os.environ), \
                patch.object(watch, "_SIGKILL_DELAY", 1.5), \
                patch.object(watch, "_sigkill_steps", []), \
                patch.object(watch, "_alert_terminal", lines.append), \
                patch.object(watch, "_tail_events", return_value=iter([_dollars3(), end])):
            for name in ("AGENT_STRACE_SLACK_WEBHOOK", "AGENT_STRACE_TEAMS_WEBHOOK"):
                os.environ.pop(name, None)
            watch_session(
                store, SESSION, config, out=out, nanny_rules=[rule],
                agent_pid=tree.pids["agent"],
            )
            # The session ended right after the kill, and SIGKILL still came first.
            self.assertTrue(_wait_until(lambda: tree.gone("ignores-term"), 0.5))
        self.assertIn(f"Killing 1 child process: {tree.pids['ignores-term']}", lines)
        self.assertIn("[watch] Session ended", out.getvalue())

    def test_no_command_line_in_the_output_or_the_alert_log(self):
        # The real CLI, with an agent whose child has a token in its arguments.
        # The child takes 1s to exit on SIGTERM, so it still runs while watch
        # prints its lines.
        tree = _Tree(self)
        agent = subprocess.Popen(tree.command("agent", "secret"))
        self.addCleanup(_stop, agent)
        tree.read("agent")
        secret = tree.read("secret")
        trace_dir = tree.folder / "traces"
        TraceStore(trace_dir, redact=False).create_session(
            SessionMeta(session_id=SESSION, started_at=time.time()),
        )
        env = without_telemetry(dict(
            os.environ,
            PYTHONPATH=str(Path(watch.__file__).resolve().parents[1]),
            HOME=str(tree.folder),
            PYTHONUNBUFFERED="1",
        ))
        for name in (
            "AGENT_STRACE_TENANT_ID", "AGENT_STRACE_SLACK_WEBHOOK", "AGENT_STRACE_TEAMS_WEBHOOK",
        ):
            env.pop(name, None)
        log = tree.folder / "watch.log"
        with open(log, "wb") as log_file:
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "agent_trace.entry", "--trace-dir", str(trace_dir),
                    "watch", SESSION, "--agent-pid", str(agent.pid), "--timeout", "1s",
                ],
                cwd=tree.folder, env=env, stdout=log_file, stderr=subprocess.STDOUT,
            )
        self.addCleanup(_stop, proc)
        self.assertEqual(agent.wait(timeout=15), -signal.SIGTERM)
        self.assertTrue(_wait_until(lambda: tree.gone("secret"), 5))
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)
        output = log.read_text(errors="replace")
        alerts = (tree.folder / ".agent-traces" / "alerts.log").read_text(errors="replace")
        self.assertIn(f"Killing 1 child process: {secret}", output)
        self.assertIn("DurationWatcher", alerts)
        for text in (output, alerts):
            self.assertNotIn("secret-token-123", text)


if __name__ == "__main__":
    unittest.main()
