"""Tests for ``agent-strace auto`` stopped by a signal (agent_trace.auto_runner)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import _no_telemetry  # noqa: F401
except ImportError:
    from tests import _no_telemetry  # noqa: F401

from agent_trace import auto_bootstrap, auto_runner

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
POSIX = os.name == "posix"

_STRIPPED_ENV = (
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "AGENT_STRACE_AUTO_INSTRUMENT",
    "AGENT_TRACE_DIR",
    "AGENT_STRACE_ENDPOINT",
    "AGENT_STRACE_WORKSPACE",
    "AGENT_STRACE_TENANT_ID",
)

# Records the stop signals it gets, then exits $CHILD_EXIT. It waits up to
# $CHILD_WAIT seconds for a signal, and $CHILD_CLEANUP seconds after one.
SIGNAL_CHILD = textwrap.dedent('''\
    import json
    import os
    import signal
    import sys
    import time

    received = []


    def on_signal(signum, frame):
        received.append(signum)


    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        signal.signal(getattr(signal, name), on_signal)
    bootstrap = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    ready = os.environ["CHILD_READY"]
    with open(ready + ".tmp", "w") as handle:
        handle.write(str(os.getpid()))
    os.replace(ready + ".tmp", ready)
    deadline = time.monotonic() + float(os.environ.get("CHILD_WAIT", "30"))
    while not received and time.monotonic() < deadline:
        time.sleep(0.05)
    # Count any copy of the signal that arrives late.
    time.sleep(float(os.environ.get("CHILD_CLEANUP", "0.5")))
    with open(os.environ["CHILD_OUT"], "w") as handle:
        json.dump({
            "signals": received,
            "bootstrap_existed": bool(bootstrap) and os.path.isdir(bootstrap),
        }, handle)
    sys.exit(int(os.environ.get("CHILD_EXIT", "0")))
''')

# Starts SIGNAL_CHILD as its own child and dies on the first stop signal,
# leaving that grandchild behind.
GROUP_PARENT = textwrap.dedent('''\
    import os
    import subprocess
    import sys
    import time

    subprocess.Popen([sys.executable, os.environ["GRANDCHILD"]])
    while not os.path.exists(os.environ["CHILD_READY"]):
        time.sleep(0.05)
    with open(os.environ["PARENT_READY"], "w") as handle:
        handle.write(str(os.getpid()))
    time.sleep(60)
''')

# Holds back the command's start until the test has typed Ctrl-C on the
# terminal, so the terminal's SIGINT arrives while agent-strace is in
# subprocess.Popen.
HOLD_POPEN = textwrap.dedent('''\
    import os
    import subprocess
    import time

    _real_popen = subprocess.Popen


    def _held_popen(args, *rest, **kwargs):
        proc = _real_popen(args, *rest, **kwargs)
        if os.environ["HOLD_FOR"] in map(str, args):
            while not os.path.exists(os.environ["CHILD_READY"]):
                time.sleep(0.01)
            open(os.environ["SPAWNED"], "w").close()
            while not os.path.exists(os.environ["GO"]):
                time.sleep(0.01)
            time.sleep(1)
        return proc


    subprocess.Popen = _held_popen
''')


def _wrapper_code(prelude: str = "") -> str:
    """Code for ``python -c`` that runs the CLI from this checkout."""
    return "\n".join([
        prelude,
        "import sys",
        f"sys.path.insert(0, {str(SRC_DIR)!r})",
        "from agent_trace.cli import main",
        "main()",
    ])


def _wait_for_file(path: Path, proc: subprocess.Popen | None = None, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if proc is not None and proc.poll() is not None:
            raise AssertionError(f"exited with {proc.returncode} before {path.name} appeared")
        if time.monotonic() > deadline:
            raise AssertionError(f"{path.name} did not appear")
        time.sleep(0.05)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reset_sigint() -> None:
    # The test runner may have been started with SIGINT ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)


class AutoTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.tmpdir = self.tmp / "tmpdir"
        self.tmpdir.mkdir()
        self.child = self.tmp / "signal_child.py"
        self.child.write_text(SIGNAL_CHILD)
        self.child_ready = self.tmp / "child.ready"
        self.child_out = self.tmp / "child.json"
        self.addCleanup(self.kill_child)

    def env(self, **extra: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        env.update(
            AGENT_STRACE_TELEMETRY="0",
            DO_NOT_TRACK="1",
            HOME=str(self.tmp),
            AGENT_TRACE_DIR=str(self.tmp / "traces"),
            TMPDIR=str(self.tmpdir),
            TEMP=str(self.tmpdir),
            TMP=str(self.tmpdir),
            CHILD_READY=str(self.child_ready),
            CHILD_OUT=str(self.child_out),
        )
        env.update(extra)
        return env

    def kill_child(self) -> None:
        """Kill the command at the end of the test if it was left running."""
        try:
            os.kill(int(self.child_ready.read_text()), signal.SIGKILL)
        except (OSError, ValueError):
            pass

    def child_result(self) -> dict:
        self.assertTrue(self.child_out.exists(), "the command did not finish its own exit")
        return json.loads(self.child_out.read_text())

    def leftover_bootstrap_dirs(self) -> list[str]:
        return [n for n in os.listdir(self.tmpdir) if n.startswith(auto_bootstrap.BOOTSTRAP_PREFIX)]

    def assert_stopped_cleanly(self, signals: list[int]) -> None:
        result = self.child_result()
        self.assertEqual(result["signals"], signals)
        # agent-strace waited for the command before removing the directory.
        self.assertTrue(result["bootstrap_existed"])
        self.assertEqual(self.leftover_bootstrap_dirs(), [])


@unittest.skipUnless(POSIX, "POSIX signals")
class TestAutoStopSignals(AutoTestCase):
    """agent-strace auto without a terminal, stopped by a signal."""

    def start_auto(self, command: list[str], ignored=(), **extra: str) -> subprocess.Popen:
        def preexec():
            _reset_sigint()
            for signum in ignored:
                signal.signal(signum, signal.SIG_IGN)

        # A new session has no controlling terminal: every signal sent to
        # agent-strace comes from this test.
        proc = subprocess.Popen(
            [sys.executable, "-c", _wrapper_code(), "auto", "--", *command],
            env=self.env(**extra), cwd=str(self.tmp), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True, preexec_fn=preexec,
        )
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        return proc

    def test_stop_signal_is_passed_on_and_bootstrap_removed(self):
        for signum, code in ((signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)):
            with self.subTest(signal=signum.name):
                self.child_ready.unlink(missing_ok=True)
                self.child_out.unlink(missing_ok=True)
                proc = self.start_auto([sys.executable, str(self.child)])
                _wait_for_file(self.child_ready, proc)
                proc.send_signal(signum)
                _, stderr = proc.communicate(timeout=60)
                self.assertEqual(proc.returncode, code, stderr)
                self.assertNotIn("Traceback", stderr)
                self.assert_stopped_cleanly([signum])

    def test_signal_to_the_process_group_reaches_the_command_once(self):
        # A supervisor stopping agent-strace's whole process group.
        proc = self.start_auto([sys.executable, str(self.child)])
        _wait_for_file(self.child_ready, proc)
        os.killpg(proc.pid, signal.SIGTERM)
        _, stderr = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 143, stderr)
        self.assert_stopped_cleanly([signal.SIGTERM])

    def test_grandchild_is_stopped_and_waited_for(self):
        parent = self.tmp / "group_parent.py"
        parent.write_text(GROUP_PARENT)
        parent_ready = self.tmp / "parent.ready"
        proc = self.start_auto(
            [sys.executable, str(parent)],
            GRANDCHILD=str(self.child), PARENT_READY=str(parent_ready), CHILD_CLEANUP="1.5",
        )
        _wait_for_file(parent_ready, proc)
        grandchild = int(self.child_ready.read_text())
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 143, stderr)
        self.assertFalse(_alive(grandchild), "the grandchild outlived agent-strace")
        self.assert_stopped_cleanly([signal.SIGTERM])

    def test_ignored_signal_stays_ignored(self):
        proc = self.start_auto(
            [sys.executable, str(self.child)], ignored=(signal.SIGHUP,), CHILD_WAIT="1.5")
        _wait_for_file(self.child_ready, proc)
        proc.send_signal(signal.SIGHUP)
        _, stderr = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, stderr)
        self.assertEqual(self.child_result()["signals"], [])
        self.assertEqual(self.leftover_bootstrap_dirs(), [])

    def test_command_killed_by_signal_exits_128_plus_signal(self):
        proc = self.start_auto(
            [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"])
        _, stderr = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 128 + signal.SIGKILL, stderr)
        self.assertEqual(self.leftover_bootstrap_dirs(), [])


# Writes its pid to $CHILD_READY and sleeps, with the default SIGINT action.
SLEEPER = (
    "import os, signal, time\n"
    "signal.signal(signal.SIGINT, signal.SIG_DFL)\n"
    "ready = os.environ['CHILD_READY']\n"
    "open(ready + '.tmp', 'w').write(str(os.getpid()))\n"
    "os.replace(ready + '.tmp', ready)\n"
    "time.sleep(30)\n"
)


class TerminalTestCase(AutoTestCase):
    """Runs programs as the session leader of a new pseudo terminal."""

    def start_on_terminal(self, argv: list[str], env: dict[str, str]) -> tuple[subprocess.Popen, int]:
        """Start *argv* as the session leader of a new pseudo terminal."""
        import fcntl
        import termios

        master, slave = os.openpty()
        attrs = termios.tcgetattr(slave)
        attrs[3] |= termios.ISIG
        attrs[6][termios.VINTR] = b"\x03"
        termios.tcsetattr(slave, termios.TCSANOW, attrs)

        def preexec():
            _reset_sigint()
            # Make the pty the controlling terminal; the new session's
            # leader is in its foreground group.
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        try:
            proc = subprocess.Popen(
                argv, env=env, cwd=str(self.tmp),
                stdin=slave, stdout=slave, stderr=slave,
                start_new_session=True, preexec_fn=preexec,
            )
        finally:
            os.close(slave)
        self.output = bytearray()

        def drain():
            while True:
                try:
                    data = os.read(master, 4096)
                except OSError:
                    return
                if not data:
                    return
                self.output += data

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()

        def close():
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=60)
            self.kill_child()
            reader.join(timeout=10)
            os.close(master)
        self.addCleanup(close)
        return proc, master

    def wait_for_foreground_change(self, proc: subprocess.Popen, master: int) -> None:
        """Wait until agent-strace has handed the terminal to the command."""
        deadline = time.monotonic() + 30
        while os.tcgetpgrp(master) == proc.pid:
            if proc.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(f"the terminal was not handed over\n{bytes(self.output)!r}")
            time.sleep(0.02)

    def finish(self, proc: subprocess.Popen) -> int:
        code = proc.wait(timeout=60)
        self.assertNotIn(b"Traceback", bytes(self.output), bytes(self.output))
        return code


@unittest.skipUnless(POSIX and hasattr(os, "openpty"), "POSIX terminals")
class TestAutoOnTerminal(TerminalTestCase):
    """agent-strace auto in the foreground of a real (pseudo) terminal."""

    def start_auto(self, prelude: str = "", **extra: str) -> tuple[subprocess.Popen, int]:
        return self.start_on_terminal(
            [sys.executable, "-c", _wrapper_code(prelude),
             "auto", "--", sys.executable, str(self.child)],
            self.env(**extra),
        )

    def test_signal_sent_to_agent_strace_reaches_the_command(self):
        for signum, code in ((signal.SIGINT, 130), (signal.SIGHUP, 129), (signal.SIGTERM, 143)):
            with self.subTest(signal=signum.name):
                self.child_ready.unlink(missing_ok=True)
                self.child_out.unlink(missing_ok=True)
                proc, _ = self.start_auto()
                _wait_for_file(self.child_ready, proc)
                os.kill(proc.pid, signum)
                self.assertEqual(self.finish(proc), code, bytes(self.output))
                self.assert_stopped_cleanly([signum])

    def test_ctrl_c_reaches_the_command_once(self):
        proc, master = self.start_auto(CHILD_EXIT="5")
        _wait_for_file(self.child_ready, proc)
        self.wait_for_foreground_change(proc, master)
        os.write(master, b"\x03")
        # The command handled Ctrl-C and chose its own exit code.
        self.assertEqual(self.finish(proc), 5, bytes(self.output))
        self.assert_stopped_cleanly([signal.SIGINT])

    def test_ctrl_c_runs_the_handler_agent_strace_started_with(self):
        handled = self.tmp / "handled"
        prelude = (
            "import os, signal\n"
            "signal.signal(signal.SIGINT, lambda signum, frame: open("
            f"{str(handled)!r}, 'a').write(str(signum)))\n"
        )
        argv = [sys.executable, "-c", _wrapper_code(prelude),
                "auto", "--", sys.executable, "-c", SLEEPER]
        proc, master = self.start_on_terminal(argv, self.env())
        _wait_for_file(self.child_ready, proc)
        self.wait_for_foreground_change(proc, master)
        os.write(master, b"\x03")
        # The command died of Ctrl-C; agent-strace passes the interrupt on to
        # its own group, where the embedding program's handler takes it.
        self.assertEqual(self.finish(proc), 130, bytes(self.output))
        self.assertEqual(handled.read_text(), str(int(signal.SIGINT)))
        self.assertEqual(self.leftover_bootstrap_dirs(), [])

    def test_ctrl_c_while_the_command_starts_reaches_it_once(self):
        spawned = self.tmp / "spawned"
        go = self.tmp / "go"
        proc, master = self.start_auto(
            prelude=HOLD_POPEN, HOLD_FOR=str(self.child), SPAWNED=str(spawned), GO=str(go))
        _wait_for_file(spawned, proc)
        # agent-strace is still inside subprocess.Popen: the command has
        # started, and the terminal has not been handed to it yet.
        os.write(master, b"\x03")
        go.touch()
        self.assertEqual(self.finish(proc), 130, bytes(self.output))
        self.assert_stopped_cleanly([signal.SIGINT])


# Runs ``agent-strace auto`` twice in a loop, as a shell script would.
SHELL_LOOP = (
    'for i in 1 2; do "$PYTHON" "$AUTO" auto -- "$@"; echo "rc=$?"; '
    'export CHILD_WAIT=0; done; echo LOOP-END'
)


@unittest.skipUnless(POSIX and hasattr(os, "openpty"), "POSIX terminals")
@unittest.skipUnless(shutil.which("bash"), "bash is not installed")
class TestAutoInShellScriptOnTerminal(TerminalTestCase):
    """A shell script running agent-strace auto, interrupted from its terminal."""

    def start_loop(self, *command: str, **extra: str) -> tuple[subprocess.Popen, int]:
        auto = self.tmp / "auto.py"
        auto.write_text(_wrapper_code())
        env = self.env(PYTHON=sys.executable, AUTO=str(auto), **extra)
        return self.start_on_terminal(
            [shutil.which("bash"), "-c", SHELL_LOOP, "loop", *command], env)

    def loop_output(self) -> str:
        return bytes(self.output).decode(errors="replace")

    # Only Ctrl-C here: on macOS a process that dies of SIGQUIT (Ctrl-\)
    # opens a crash report. TestTerminalInterrupt covers SIGQUIT.
    def test_ctrl_c_stops_the_script(self):
        proc, master = self.start_loop(sys.executable, "-c", SLEEPER)
        _wait_for_file(self.child_ready, proc)
        self.wait_for_foreground_change(proc, master)
        os.write(master, b"\x03")
        proc.wait(timeout=60)
        time.sleep(0.2)
        self.assertEqual(proc.returncode, -signal.SIGINT, self.loop_output())
        self.assertNotIn("LOOP-END", self.loop_output())
        self.assertEqual(self.leftover_bootstrap_dirs(), [])

    def test_command_that_handles_ctrl_c_lets_the_script_go_on(self):
        proc, master = self.start_loop(sys.executable, str(self.child), CHILD_CLEANUP="0")
        _wait_for_file(self.child_ready, proc)
        self.wait_for_foreground_change(proc, master)
        os.write(master, b"\x03")
        self.assertEqual(self.finish(proc), 0, self.loop_output())
        time.sleep(0.2)
        self.assertEqual(self.loop_output().count("rc=0"), 2, self.loop_output())
        self.assertIn("LOOP-END", self.loop_output())
        self.assertEqual(self.leftover_bootstrap_dirs(), [])

    def test_command_that_signals_itself_lets_the_script_go_on(self):
        proc, _ = self.start_loop(
            sys.executable, "-c",
            "import os, signal\n"
            "signal.signal(signal.SIGINT, signal.SIG_DFL)\n"
            "os.kill(os.getpid(), signal.SIGINT)\n",
        )
        self.assertEqual(proc.wait(timeout=60), 0, self.loop_output())
        time.sleep(0.2)
        self.assertEqual(self.loop_output().count("rc=130"), 2, self.loop_output())
        self.assertIn("LOOP-END", self.loop_output())
        self.assertEqual(self.leftover_bootstrap_dirs(), [])


@unittest.skipUnless(POSIX, "POSIX signals")
class TestTerminalInterrupt(unittest.TestCase):
    """Which interrupts agent-strace passes on to its own process group.

    These cover SIGQUIT without letting any process die of it.
    """

    def test_observer_reports_interrupts_sent_to_its_group(self):
        observer = auto_runner._SignalObserver.start(0)
        self.assertIsNotNone(observer)
        self.addCleanup(observer.close)
        # The observer blocks both signals, so neither kills it.
        os.killpg(observer._proc.pid, signal.SIGINT)
        os.killpg(observer._proc.pid, signal.SIGQUIT)
        self.assertEqual(observer.interrupts(), {signal.SIGINT, signal.SIGQUIT})

    def test_observer_never_loses_an_interrupt_sent_just_before_it_stops(self):
        # SIGINT, SIGQUIT and the stop request from interrupts() come back to
        # back. The observer blocks both signals, so neither kills it.
        lost = []
        for attempt in range(200):
            observer = auto_runner._SignalObserver.start(0)
            self.assertIsNotNone(observer)
            try:
                os.killpg(observer._proc.pid, signal.SIGINT)
                os.killpg(observer._proc.pid, signal.SIGQUIT)
                got = observer.interrupts()
            finally:
                observer.close()
            if got != {signal.SIGINT, signal.SIGQUIT}:
                lost.append((attempt, sorted(got)))
        self.assertEqual(lost, [])

    def test_only_an_observed_interrupt_that_killed_the_command_is_passed_on(self):
        both = {signal.SIGINT, signal.SIGQUIT}
        for signum in (signal.SIGINT, signal.SIGQUIT):
            with self.subTest(signal=signum.name):
                # A wait status for "killed by signal N" is N.
                self.assertEqual(auto_runner._terminal_interrupt(signum, both, []), signum)
                # The command signalled itself: the observer saw nothing.
                self.assertIsNone(auto_runner._terminal_interrupt(signum, set(), []))
                # agent-strace forwarded a signal, which the observer also saw.
                self.assertIsNone(auto_runner._terminal_interrupt(signum, both, [signal.SIGHUP]))
                # The command handled the interrupt and exited.
                self.assertIsNone(auto_runner._terminal_interrupt(0, both, []))
        self.assertIsNone(auto_runner._terminal_interrupt(signal.SIGTERM, {signal.SIGTERM}, []))

    def test_interrupt_goes_to_agent_strace_group_with_its_handler_kept(self):
        for signum in (signal.SIGINT, signal.SIGQUIT):
            with self.subTest(signal=signum.name):
                previous = signal.signal(signum, signal.SIG_IGN)
                self.addCleanup(signal.signal, signum, previous)
                with mock.patch.object(auto_runner.os, "killpg") as killpg:
                    auto_runner._interrupt_own_group(signum)
                killpg.assert_called_once_with(os.getpgrp(), signum)
                self.assertIs(signal.getsignal(signum), signal.SIG_IGN)

    def test_python_keyboard_interrupt_handler_becomes_the_default_action(self):
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        self.addCleanup(signal.signal, signal.SIGINT, previous)
        with mock.patch.object(auto_runner.os, "killpg") as killpg:
            auto_runner._interrupt_own_group(signal.SIGINT)
        killpg.assert_called_once_with(os.getpgrp(), signal.SIGINT)
        self.assertIs(signal.getsignal(signal.SIGINT), signal.SIG_DFL)


@unittest.skipUnless(POSIX, "POSIX signals")
class TestCommandRunner(AutoTestCase):
    def setUp(self):
        super().setUp()
        # The test runner may be in the foreground of a terminal; keep it.
        patcher = mock.patch.object(auto_runner._Terminal, "open", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_signal_before_the_command_starts_skips_it(self):
        marker = self.tmp / "started"
        before = signal.getsignal(signal.SIGTERM)
        with auto_runner.CommandRunner() as runner:
            signal.raise_signal(signal.SIGTERM)
            code = runner.run([sys.executable, "-c", f"open({str(marker)!r}, 'w')"], self.env())
        self.assertEqual(code, 143)
        self.assertFalse(marker.exists())
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_signal_while_the_command_starts_is_passed_on_once(self):
        real_popen = subprocess.Popen

        def popen_then_signal(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            _wait_for_file(self.child_ready, proc)
            signal.raise_signal(signal.SIGTERM)
            return proc

        with mock.patch.object(auto_runner.subprocess, "Popen", popen_then_signal):
            with auto_runner.CommandRunner() as runner:
                code = runner.run([sys.executable, str(self.child)], self.env())
        self.assertEqual(code, 143)
        self.assertEqual(self.child_result()["signals"], [signal.SIGTERM])

    def test_command_runs_in_its_own_process_group(self):
        with auto_runner.CommandRunner() as runner:
            code = runner.run(
                [sys.executable, "-c",
                 f"import os, sys; sys.exit(os.getpgrp() == {os.getpgrp()})"],
                self.env())
        self.assertEqual(code, 0)


class TestAutoRemovesBootstrapDir(AutoTestCase):
    def test_bootstrap_dir_removed_and_exit_code_kept(self):
        proc = subprocess.run(
            [sys.executable, "-c", _wrapper_code(),
             "auto", "--", sys.executable, "-c", "import sys; sys.exit(3)"],
            env=self.env(), cwd=str(self.tmp), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertEqual(os.listdir(self.tmpdir), [])


if __name__ == "__main__":
    unittest.main()
