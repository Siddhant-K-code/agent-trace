"""Tests for process start-time identity and PID reuse detection."""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from agent_trace import process_identity as pi
from agent_trace.approval import (
    ApprovalRequest,
    create_approval_request,
    deny_request,
    load_request,
    save_request,
)
from agent_trace.models import SessionMeta
from agent_trace.store import TraceStore
from agent_trace.watch import WatcherConfig, WatchState, _dispatch_alert

# Linux /proc/PID/stat with a command name holding spaces and ")".
# Field 22 (starttime) is 4242.
PROC_STAT = (
    b"1234 (my) agent) S 1 1234 1234 0 -1 4194560 100 0 0 0 1 2 0 0 "
    b"20 0 1 0 4242 1000000 100 18446744073709551615\n"
)


class TestProcStartTime(unittest.TestCase):
    def test_reads_field_22(self):
        with patch("builtins.open", mock_open(read_data=PROC_STAT)):
            self.assertEqual(pi._proc_start_time(1234), "proc:4242")

    def test_missing_process_is_unknown(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            self.assertIsNone(pi._proc_start_time(1234))

    def test_truncated_stat_is_unknown(self):
        with patch("builtins.open", mock_open(read_data=b"1234 (agent) S 1")):
            self.assertIsNone(pi._proc_start_time(1234))


class TestPsStartTime(unittest.TestCase):
    def _run(self, returncode=0, stdout=""):
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")

    def test_reads_lstart_with_c_locale(self):
        with patch("subprocess.run", return_value=self._run(stdout=" Wed Nov 11  9:01:02 2026\n")) as run:
            self.assertEqual(pi._ps_start_time(1234), "ps:Wed Nov 11 9:01:02 2026")
        self.assertEqual(run.call_args.args[0], ["ps", "-o", "lstart=", "-p", "1234"])
        self.assertEqual(run.call_args.kwargs["env"]["LC_ALL"], "C")
        self.assertEqual(run.call_args.kwargs["env"]["TZ"], "UTC0")

    def test_timezone_does_not_change_identity(self):
        if os.name != "posix" or os.path.exists("/proc/self/stat"):
            self.skipTest("ps start time is used only without /proc")
        first = pi._ps_start_time(os.getpid())
        with patch.dict(os.environ, {"TZ": "Asia/Tokyo"}):
            self.assertEqual(pi._ps_start_time(os.getpid()), first)

    def test_missing_process_is_unknown(self):
        with patch("subprocess.run", return_value=self._run(returncode=1)):
            self.assertIsNone(pi._ps_start_time(1234))

    def test_ps_failure_is_unknown(self):
        for exc in (FileNotFoundError("ps"), subprocess.TimeoutExpired("ps", 2)):
            with self.subTest(exc=exc), patch("subprocess.run", side_effect=exc):
                self.assertIsNone(pi._ps_start_time(1234))


class TestProcessIdentity(unittest.TestCase):
    def test_invalid_pids_are_unknown(self):
        for pid in (0, -1, None, "1234", True):
            with self.subTest(pid=pid):
                self.assertIsNone(pi.process_identity(pid))

    def test_uses_proc_when_mounted(self):
        with patch("os.name", "posix"), patch("os.path.exists", return_value=True), \
             patch.object(pi, "_proc_start_time", return_value="proc:1") as proc, \
             patch.object(pi, "_ps_start_time") as ps:
            self.assertEqual(pi.process_identity(1234), "proc:1")
        proc.assert_called_once_with(1234)
        ps.assert_not_called()

    def test_uses_ps_without_proc(self):
        with patch("os.name", "posix"), patch("os.path.exists", return_value=False), \
             patch.object(pi, "_ps_start_time", return_value="ps:x") as ps:
            self.assertEqual(pi.process_identity(1234), "ps:x")
        ps.assert_called_once_with(1234)

    def test_windows_is_unknown(self):
        with patch("os.name", "nt"), patch.object(pi, "_ps_start_time") as ps, \
             patch.object(pi, "_proc_start_time") as proc:
            self.assertIsNone(pi.process_identity(1234))
        ps.assert_not_called()
        proc.assert_not_called()


class TestPidReused(unittest.TestCase):
    def check(self, recorded, current):
        with patch.object(pi, "process_identity", return_value=current):
            return pi.pid_reused(1234, recorded)

    def test_different_start_time_is_reuse(self):
        self.assertTrue(self.check("proc:100", "proc:200"))

    def test_same_start_time_is_not_reuse(self):
        self.assertFalse(self.check("proc:100", "proc:100"))

    def test_unknown_identity_keeps_old_behaviour(self):
        for recorded, current in (
            ("", "proc:1"), (None, "proc:1"), (123, "proc:1"), ("junk", "proc:1"),
            ("proc:1", None),
        ):
            with self.subTest(recorded=recorded, current=current):
                self.assertFalse(self.check(recorded, current))

    def test_different_sources_are_not_compared(self):
        self.assertFalse(self.check("ps:Wed Nov 11 09:01:02 2026", "proc:1"))


def _proc_stat(pid: int, ppid: int, start: int, name: str = "sh") -> bytes:
    """A /proc/PID/stat line with parent *ppid* and start time *start*."""
    return (
        f"{pid} ({name}) S {ppid} {pid} {pid} 0 -1 4194560 100 0 0 0 1 2 0 0 "
        f"20 0 1 0 {start} 1000000 100 18446744073709551615\n"
    ).encode()


class TestListProc(unittest.TestCase):
    """The Linux listing, read from a /proc-like folder."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def add(self, pid: int, ppid: int, start: int, name: str = "sh") -> None:
        (self.root / str(pid)).mkdir()
        (self.root / str(pid) / "stat").write_bytes(_proc_stat(pid, ppid, start, name))

    def test_reads_parent_and_identity(self):
        self.add(1, 0, 1)
        self.add(100, 1, 500, name="my) agent")
        self.add(200, 100, 600)
        (self.root / "self").mkdir()
        (self.root / "meminfo").write_text("MemTotal: 1 kB\n")
        self.assertEqual(pi._list_proc(str(self.root)), {
            1: pi.ProcessEntry(0, "proc:1"),
            100: pi.ProcessEntry(1, "proc:500"),
            200: pi.ProcessEntry(100, "proc:600"),
        })

    def test_process_that_ends_during_the_scan_is_skipped(self):
        self.add(100, 1, 500)
        (self.root / "200").mkdir()  # listed, but its stat file is gone
        self.add(300, 100, 700)
        self.add(400, 100, 800)
        real_open = open

        def open_stat(path, *args, **kwargs):
            if path == os.path.join(str(self.root), "400", "stat"):
                raise ProcessLookupError(3, "No such process")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", open_stat):
            self.assertEqual(sorted(pi._list_proc(str(self.root))), [100, 300])

    def test_malformed_entry_is_skipped(self):
        self.add(100, 1, 500)
        for pid, raw in ((200, b"200 (sh) S 1"), (300, b"garbage")):
            (self.root / str(pid)).mkdir()
            (self.root / str(pid) / "stat").write_bytes(raw)
        self.assertEqual(sorted(pi._list_proc(str(self.root))), [100])

    def test_unreadable_or_empty_folder_raises(self):
        for root, reason in ((self.root / "missing", "cannot read"), (self.root, "lists no process")):
            with self.subTest(reason=reason), self.assertRaises(pi.ProcessListError) as caught:
                pi._list_proc(str(root))
            self.assertIn(reason, str(caught.exception))


class TestListPs(unittest.TestCase):
    """The listing without /proc: one ps call."""

    def _run(self, stdout="", returncode=0):
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")

    def test_reads_parent_and_identity(self):
        stdout = (
            "    1     0 Mon Jun  2 08:00:00 2026\n"
            "  100     1 Mon Jun  2 09:01:02 2026\n"
            "  200   100 Mon Jun  2 09:01:03 2026\n"
            "junk\n"
        )
        with patch("subprocess.run", return_value=self._run(stdout)) as run:
            table = pi._list_ps()
        self.assertEqual(table, {
            1: pi.ProcessEntry(0, "ps:Mon Jun 2 08:00:00 2026"),
            100: pi.ProcessEntry(1, "ps:Mon Jun 2 09:01:02 2026"),
            200: pi.ProcessEntry(100, "ps:Mon Jun 2 09:01:03 2026"),
        })
        self.assertEqual(run.call_args.args[0], ["ps", "-A", "-o", "pid=,ppid=,lstart="])
        env = run.call_args.kwargs["env"]
        self.assertEqual((env["LC_ALL"], env["LANG"], env["TZ"]), ("C", "C", "UTC0"))
        self.assertEqual(run.call_args.kwargs["timeout"], 2.0)

    def test_identity_is_the_one_ps_start_time_reads(self):
        lstart = "Wed Nov 11  9:01:02 2026"
        with patch("subprocess.run", return_value=self._run(f"  100     1 {lstart}\n")):
            listed = pi._list_ps()[100].identity
        with patch("subprocess.run", return_value=self._run(f" {lstart}\n")):
            self.assertEqual(pi._ps_start_time(100), listed)

    def test_failure_raises_with_the_reason(self):
        for outcome, reason in (
            (subprocess.TimeoutExpired("ps", 2), "ps did not finish within 2s"),
            (FileNotFoundError(2, "No such file or directory"),
             "cannot run ps: No such file or directory"),
            (self._run(returncode=1), "ps exited with status 1"),
            (self._run("\n"), "ps listed no process"),
        ):
            with self.subTest(reason=reason):
                kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {
                    "return_value": outcome}
                with patch("subprocess.run", **kwargs), \
                     self.assertRaises(pi.ProcessListError) as caught:
                    pi._list_ps()
                self.assertEqual(str(caught.exception), reason)


def _table(*rows: tuple[int, int]) -> dict[int, pi.ProcessEntry]:
    """A process table from (pid, ppid) rows. Each identity is "id:<pid>"."""
    return {pid: pi.ProcessEntry(ppid, f"id:{pid}") for pid, ppid in rows}


class TestDescendants(unittest.TestCase):
    AGENT = 100
    WATCH = 900

    def pids(self, table, own=WATCH):
        found = pi.descendants(table, self.AGENT, own)
        self.assertEqual([identity for _, identity in found], [f"id:{pid}" for pid, _ in found])
        return [pid for pid, _ in found]

    def test_parents_come_before_their_children(self):
        # Only the parent PID links a process to the agent, so a grandchild
        # in its own session or process group is found too.
        table = _table(
            (1, 0), (100, 1), (300, 200), (200, 100), (150, 100), (400, 300), (900, 1),
            (950, 1),
        )
        found = self.pids(table)
        self.assertEqual(found, [150, 200, 300, 400])
        for pid in found:
            parent = table[pid].ppid
            self.assertTrue(parent == self.AGENT or found.index(parent) < found.index(pid))

    def test_watch_and_what_it_started_are_left_out(self):
        # watch (900) runs in a shell (300) that the agent started, and
        # started ps (901), which started 902.
        table = _table(
            (1, 0), (100, 1), (200, 100), (300, 100), (900, 300), (901, 900), (902, 901),
        )
        self.assertEqual(self.pids(table), [200])

    def test_agent_under_watch_gets_nothing(self):
        # Every process under the agent is then one that watch started.
        table = _table((1, 0), (900, 1), (100, 900), (200, 100))
        self.assertEqual(self.pids(table), [])
        self.assertEqual(pi.descendants(table, 900, 900), [])

    def test_pid_1_and_lower_are_never_listed(self):
        table = _table((0, 100), (1, 100), (100, 50), (50, 1), (200, 100))
        self.assertEqual(self.pids(table), [200])

    def test_parent_cycle_ends(self):
        table = _table((100, 300), (200, 100), (300, 200), (400, 300), (500, 600), (600, 500))
        self.assertEqual(self.pids(table), [200, 300, 400])
        self.assertEqual(self.pids(table, own=500), [200, 300, 400])


@unittest.skipUnless(os.name == "posix", "the process table is read on POSIX only")
class TestListRealProcesses(unittest.TestCase):
    def setUp(self):
        if pi.process_identity(os.getpid()) is None:
            self.skipTest("process start time is not readable here")
        self.child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

        def cleanup():
            if self.child.poll() is None:
                self.child.kill()
            self.child.wait()

        self.addCleanup(cleanup)

    def test_listed_identity_is_process_identity(self):
        entry = pi.list_processes()[self.child.pid]
        self.assertEqual(entry, pi.ProcessEntry(os.getpid(), pi.process_identity(self.child.pid)))

    @unittest.skipUnless(shutil.which("ps"), "needs ps")
    def test_ps_listing_matches_ps_start_time(self):
        # On Linux list_processes reads /proc, so this covers the ps path there.
        entry = pi._list_ps()[self.child.pid]
        self.assertEqual(entry, pi.ProcessEntry(os.getpid(), pi._ps_start_time(self.child.pid)))


@unittest.skipUnless(os.name == "posix", "start-time identity is POSIX only")
class TestRealProcessReuse(unittest.TestCase):
    """Record a real child, let it exit, then point its record at a new process.

    The OS cannot be made to reuse a PID on demand, so the reused PID is
    faked: the record keeps the exited child's start time but names a live
    child started later, exactly what a reused PID looks like.
    """

    def setUp(self):
        if pi.process_identity(os.getpid()) is None:
            self.skipTest("process start time is not readable here")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TraceStore(self._tmp.name, redact=False)

    def _sleeper(self) -> subprocess.Popen:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

        def cleanup():
            if child.poll() is None:
                child.kill()
            child.wait()

        self.addCleanup(cleanup)
        return child

    def _exited_child_identity(self) -> str:
        old = subprocess.Popen([sys.executable, "-c", "pass"])
        identity = pi.process_identity(old.pid)
        self.assertIsNotNone(identity)
        old.wait()
        # ps reports start times to the second; make the next start differ.
        time.sleep(1.1)
        return identity

    def test_live_child_matches_its_own_identity(self):
        child = self._sleeper()
        identity = pi.process_identity(child.pid)
        self.assertFalse(pi.pid_reused(child.pid, identity))

    def test_deny_does_not_kill_process_with_reused_pid(self):
        old_identity = self._exited_child_identity()
        stranger = self._sleeper()
        self.assertTrue(pi.pid_reused(stranger.pid, old_identity))
        req = create_approval_request(
            self.store, "s1", "rule", agent_pid=stranger.pid,
            agent_start_time=old_identity,
        )
        ok, msg = deny_request(self.store, req.request_id)
        self.assertTrue(ok)
        self.assertIn(f"agent {stranger.pid} was not signalled", msg)
        self.assertIn("now belongs to another process", msg)
        self.assertEqual(load_request(self.store, req.request_id).state, "denied")
        time.sleep(0.2)
        self.assertIsNone(stranger.poll(), "the unrelated process was signalled")

    def test_deny_still_kills_the_recorded_agent(self):
        agent = self._sleeper()
        req = create_approval_request(
            self.store, "s1", "rule", agent_pid=agent.pid,
            agent_start_time=pi.process_identity(agent.pid),
        )
        ok, msg = deny_request(self.store, req.request_id)
        self.assertTrue(msg.endswith(f"agent {agent.pid} terminated"), msg)
        self.assertEqual(agent.wait(timeout=10), -signal.SIGTERM)

    def test_old_record_without_identity_still_signals(self):
        agent = self._sleeper()
        req = ApprovalRequest(agent_pid=agent.pid)
        save_request(self.store, req)
        deny_request(self.store, req.request_id)
        self.assertEqual(agent.wait(timeout=10), -signal.SIGTERM)

    @unittest.skipUnless(hasattr(signal, "SIGSTOP"), "needs SIGSTOP")
    def test_watch_does_not_stop_or_kill_process_with_reused_pid(self):
        old_identity = self._exited_child_identity()
        stranger = self._sleeper()
        self.store.create_session(SessionMeta(session_id="s1", started_at=1.0))
        config = WatcherConfig(alert_log=str(Path(self._tmp.name) / "alerts.log"))
        for action in ("pause", "kill"):
            with self.subTest(action=action):
                state = WatchState(
                    agent_pid=stranger.pid, agent_start_time=old_identity, session_id="s1",
                )
                with patch("agent_trace.watch._alert_terminal") as terminal:
                    _dispatch_alert("rule fired", config, state, action=action,
                                    store=self.store, session_id="s1")
                said = " ".join(str(c.args[0]) for c in terminal.call_args_list)
                self.assertIn("no agent was signalled", said)
                self.assertIsNone(state.agent_pid)
                self.assertFalse(state.paused)
        # A stopped process would not answer; a killed one would have exited.
        time.sleep(0.2)
        self.assertIsNone(stranger.poll())
        state_letter = subprocess.run(
            ["ps", "-o", "state=", "-p", str(stranger.pid)],
            capture_output=True, text=True,
        ).stdout.strip()
        self.assertNotIn("T", state_letter)
        self.assertIn("now belongs to another process",
                      (Path(self._tmp.name) / "alerts.log").read_text())


if __name__ == "__main__":
    unittest.main()
