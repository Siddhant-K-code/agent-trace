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
  resume both. When Ctrl-C or Ctrl-\\ kills the command, agent-strace
  removes the hook directory and sends the same signal to its own group,
  so a calling shell script is interrupted as it would be without
  agent-strace. A small observer process in the command's group tells the
  terminal's signal apart from one the command sent itself.

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
import select
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
# Seconds to wait for the observer to start, and to report.
OBSERVER_TIMEOUT = 10
# Signals the terminal sends that agent-strace passes on to its own group
# when they kill the command.
_INTERRUPTS = frozenset(
    getattr(signal, name) for name in ("SIGINT", "SIGQUIT") if hasattr(signal, name)
)
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


# Run in the command's process group. SIGINT and SIGQUIT stay blocked, so
# one the group gets stays pending and cannot kill the observer. When its
# stdin is closed, the observer reports the pending ones followed by "done"
# and exits. No handler runs and no signal is taken, so an interrupt sent
# before stdin is closed is always reported. (sigwait is not used: on macOS
# it can return a wrong signal number when several signals are pending.)
_OBSERVER = """\
import os, signal
interrupts = {signal.SIGINT, signal.SIGQUIT}
signal.pthread_sigmask(signal.SIG_BLOCK, interrupts)
for signum in interrupts:
    # An ignored signal is discarded even while blocked.
    signal.signal(signum, signal.SIG_DFL)
os.write(1, b"ready\\n")
while os.read(0, 1024):
    pass
pending = sorted(signal.sigpending() & interrupts)
os.write(1, b"".join(b"%d\\n" % signum for signum in pending) + b"done\\n")
os._exit(0)
"""


class _SignalObserver:
    """Tells which interrupts the terminal sent to the command's group.

    A terminal signals every process in its foreground group, so the
    observer gets Ctrl-C and Ctrl-\\ along with the command, and not a
    signal the command sends itself.
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._output = b""

    @classmethod
    def start(cls, pgid: int) -> _SignalObserver | None:
        if not sys.executable:
            return None
        if sys.version_info >= (3, 11):
            group: dict[str, Any] = {"process_group": pgid}
        else:
            group = {"preexec_fn": lambda: os.setpgid(0, pgid)}
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _OBSERVER],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, **group,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        observer = cls(proc)
        if b"ready" not in observer._read_until(b"ready"):
            observer.close()
            return None
        return observer

    def _read_until(self, marker: bytes) -> list[bytes]:
        assert self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + OBSERVER_TIMEOUT
        while marker not in self._output.split(b"\n"):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                break
            data = os.read(fd, 1024)
            if not data:
                break
            self._output += data
        return self._output.split(b"\n")

    def interrupts(self) -> set[int]:
        """Stop the observer and return the interrupts it got."""
        self._close_stdin()
        lines = self._read_until(b"done")
        self.close()
        return {int(line) for line in lines if line.isdigit()}

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.kill()
            except OSError:
                pass
        self._close_stdin()
        self._proc.wait()
        if self._proc.stdout is not None:
            self._proc.stdout.close()

    def _close_stdin(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except OSError:
                pass


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
        # The terminal interrupt that killed the command, if any.
        self._interrupt: int | None = None

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
        if self._interrupt is not None:
            _interrupt_own_group(self._interrupt)

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
        observer: _SignalObserver | None = None
        try:
            proc = _spawn_in_own_group(command, env)
            self._pgid = proc.pid
            # Signals that arrived while the command was starting. The
            # command is in its own group, so none of them reached it.
            while self._pending:
                self._forward(self._pending.pop(0))
            if terminal is not None and terminal.foreground() == os.getpgrp():
                observer = _SignalObserver.start(proc.pid)
                terminal.give(proc.pid)
            status = self._wait(proc, terminal)
            if observer is not None:
                # The command has exited, so a signal the terminal sent to
                # its group is already pending in the observer.
                interrupts = observer.interrupts()
                observer = None
                self._interrupt = _terminal_interrupt(status, interrupts, self._forwarded)
            if self._forwarded:
                self._wait_for_group(proc.pid)
        except BaseException:
            if proc is not None and proc.returncode is None:
                _kill_group(proc)
            raise
        finally:
            if observer is not None:
                observer.close()
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


def _terminal_interrupt(status: int, observed: set[int], forwarded: list[int]) -> int | None:
    """Return the interrupt from the terminal that killed the command, if any.

    *status* is the command's wait status and *observed* the interrupts the
    observer saw. A signal agent-strace forwarded also reaches the observer,
    so after a forwarded signal there is no terminal interrupt to pass on.
    """
    if forwarded or not os.WIFSIGNALED(status):
        return None
    signum = os.WTERMSIG(status)
    return signum if signum in observed & _INTERRUPTS else None


def _interrupt_own_group(signum: int) -> None:
    """Send the terminal's interrupt on to agent-strace's own group.

    A shell running agent-strace in a script stops the script only if it
    got the interrupt itself, as it would have if agent-strace had not
    handed the terminal to the command. The handler agent-strace started
    with decides what happens to agent-strace itself: an ignored signal
    stays ignored and a handler set by the embedding program runs.
    Python's own KeyboardInterrupt handler is replaced by the default
    action, so agent-strace dies of the signal without a traceback.
    """
    try:
        if signal.getsignal(signum) is signal.default_int_handler:
            signal.signal(signum, signal.SIG_DFL)
        os.killpg(os.getpgrp(), signum)
    except (OSError, ValueError):
        pass


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait()
    except Exception:
        pass
