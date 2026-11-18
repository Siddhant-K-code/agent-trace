"""Runs the command for ``agent-strace auto`` and stops it with agent-strace.

On POSIX the command runs in a process group of its own, so a signal meant
for agent-strace never reaches it directly and every signal it gets is
delivered once:

- SIGTERM, SIGINT and SIGHUP sent to agent-strace, by PID or to its process
  group, are passed on to the command's whole process group. agent-strace
  then waits until every process in that group has exited, so the hook
  directory outlives all of them.
- When agent-strace runs in the foreground of its controlling terminal, the
  command's group is made the terminal's foreground group, the way a shell
  starts a job. Ctrl-C, Ctrl-\\ and Ctrl-Z then go to the command alone.
  When the command stops, agent-strace takes the terminal back and stops
  with the same signal, so the shell sees its job stop; ``fg`` and ``bg``
  resume both.

The handlers are installed before the hook directory is created and before
the command starts. A stop signal agent-strace was started with ignored
stays ignored. A process that leaves the command's process group (with
``setsid`` or ``setpgid``) is neither signalled nor waited for, and SIGKILL
sent to agent-strace or to its process group does not reach the command.

On Windows, and outside the main thread, where signal handlers cannot be
installed, the command is run with :func:`subprocess.run` as before.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any

# Signals that stop ``agent-strace auto``. Each is passed on to the command.
STOP_SIGNALS = ("SIGTERM", "SIGINT", "SIGHUP")
# Seconds between checks on the command while agent-strace waits for it.
POLL_INTERVAL = 0.05
# Stops a shell follows: agent-strace stops along with the command.
_JOB_CONTROL_STOPS = frozenset(
    getattr(signal, name) for name in ("SIGTSTP", "SIGTTIN", "SIGTTOU") if hasattr(signal, name)
)


class _Terminal:
    """agent-strace's controlling terminal, for handing it to the command."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    @classmethod
    def open(cls) -> _Terminal | None:
        try:
            return cls(os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY))
        except OSError:
            return None

    def close(self) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass

    def foreground(self) -> int | None:
        try:
            return os.tcgetpgrp(self._fd)
        except OSError:
            return None

    def set_foreground(self, pgid: int) -> None:
        # A background process that changes the foreground group gets
        # SIGTTOU unless it blocks it.
        try:
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTTOU})
        except (AttributeError, OSError, ValueError):
            old_mask = None
        try:
            os.tcsetpgrp(self._fd, pgid)
        except OSError:
            pass
        finally:
            if old_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    def give(self, pgid: int) -> None:
        """Make *pgid* the foreground group if agent-strace's group is."""
        if self.foreground() == os.getpgrp():
            self.set_foreground(pgid)

    def take_back(self, pgid: int) -> None:
        """Make agent-strace's group the foreground group if *pgid* is."""
        if self.foreground() == pgid:
            self.set_foreground(os.getpgrp())


class CommandRunner:
    """Runs the command and passes on the stop signals agent-strace gets.

    Use it as a context manager around everything that must be undone when
    agent-strace is stopped, so the handlers are in place before the hook
    directory exists. A signal that arrives before the command starts stops
    agent-strace without starting it.
    """

    def __init__(self) -> None:
        self._managed = False
        self._previous: dict[int, Any] = {}
        self._received: list[int] = []
        self._pending: list[int] = []
        self._forwarded: list[int] = []
        self._pgid: int | None = None
        self._stopped = False

    def __enter__(self) -> CommandRunner:
        if os.name != "posix" or threading.current_thread() is not threading.main_thread():
            return self
        self._managed = True
        for name in STOP_SIGNALS:
            signum = getattr(signal, name)
            try:
                if signal.getsignal(signum) == signal.SIG_IGN:
                    continue
                self._previous[signum] = signal.signal(signum, self._on_signal)
            except (OSError, ValueError):
                pass
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for signum, handler in self._previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, TypeError, ValueError):
                pass
        self._previous = {}

    def _on_signal(self, signum: int, frame: Any) -> None:
        self._received.append(signum)
        if self._pgid is None:
            self._pending.append(signum)
        else:
            self._forward(signum)

    def _forward(self, signum: int) -> None:
        self._forwarded.append(signum)
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, signum)
            if self._stopped:
                os.killpg(self._pgid, signal.SIGCONT)
        except OSError:
            pass

    def run(self, command: list[str], env: Mapping[str, str]) -> int:
        """Run *command* to completion and return agent-strace's exit code.

        That is the command's exit code, or 128 plus the signal number when
        a signal killed the command or agent-strace passed a stop signal on
        to it, as a shell reports it.
        """
        if not self._managed:
            return subprocess.run(command, env=env).returncode
        if self._received:
            return 128 + self._received[0]
        terminal = _Terminal.open()
        proc: subprocess.Popen | None = None
        try:
            proc = _spawn_in_own_group(command, env)
            self._pgid = proc.pid
            # Signals that arrived while the command was starting. The
            # command is in its own group, so none of them reached it.
            while self._pending:
                self._forward(self._pending.pop(0))
            if terminal is not None:
                terminal.give(proc.pid)
            status = self._wait(proc, terminal)
            if self._forwarded:
                self._wait_for_group(proc.pid)
        except BaseException:
            if proc is not None and proc.returncode is None:
                _kill_group(proc)
            raise
        finally:
            if terminal is not None:
                if proc is not None:
                    terminal.take_back(proc.pid)
                terminal.close()
        if self._forwarded:
            return 128 + self._forwarded[0]
        code = os.waitstatus_to_exitcode(status)
        return 128 - code if code < 0 else code

    def _wait(self, proc: subprocess.Popen, terminal: _Terminal | None) -> int:
        """Wait for the command to exit and return its wait status.

        The wait is polled: Python runs signal handlers only in the main
        thread, and a signal the kernel delivers to another thread does not
        interrupt a blocking wait.
        """
        flags = os.WNOHANG | os.WUNTRACED | getattr(os, "WCONTINUED", 0)
        while True:
            pid, status = os.waitpid(proc.pid, flags)
            if pid == 0:
                time.sleep(POLL_INTERVAL)
            elif os.WIFSTOPPED(status):
                self._stopped = True
                stopsig = os.WSTOPSIG(status)
                if terminal is not None and stopsig in _JOB_CONTROL_STOPS:
                    self._on_command_stopped(stopsig, proc.pid, terminal)
            elif getattr(os, "WIFCONTINUED", lambda _: False)(status):
                self._stopped = False
            else:
                proc.returncode = os.waitstatus_to_exitcode(status)
                return status

    def _on_command_stopped(self, stopsig: int, pgid: int, terminal: _Terminal) -> None:
        """Follow the command's job-control stop, as a shell job would."""
        if stopsig != signal.SIGTSTP and terminal.foreground() in (pgid, os.getpgrp()):
            # It touched the terminal while agent-strace still held it.
            terminal.give(pgid)
            self._continue(pgid)
            return
        terminal.take_back(pgid)
        # Stop agent-strace's own group, as Ctrl-Z would have before the
        # command had a group of its own, so the shell sees its job stop.
        # The kernel discards this signal when no shell can resume the
        # group (an orphaned group), and the command is simply continued.
        try:
            os.killpg(os.getpgrp(), stopsig)
        except OSError:
            pass
        # Continued, by ``fg`` or ``bg``.
        terminal.give(pgid)
        self._continue(pgid)

    def _continue(self, pgid: int) -> None:
        try:
            os.killpg(pgid, signal.SIGCONT)
        except OSError:
            pass
        self._stopped = False

    def _wait_for_group(self, pgid: int) -> None:
        """Wait until no process is left in the command's process group."""
        while True:
            # Reap members that were orphaned and handed to agent-strace,
            # which happens when it is PID 1 or a child subreaper.
            try:
                while os.waitpid(-pgid, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            try:
                os.killpg(pgid, 0)
            except OSError:
                # ESRCH: the group is empty. EPERM: what is left cannot be
                # signalled by agent-strace either, so waiting cannot end it.
                return
            time.sleep(POLL_INTERVAL)


def _spawn_in_own_group(command: list[str], env: Mapping[str, str]) -> subprocess.Popen:
    if sys.version_info >= (3, 11):
        return subprocess.Popen(command, env=env, process_group=0)
    return subprocess.Popen(command, env=env, preexec_fn=os.setpgrp)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait()
    except Exception:
        pass
