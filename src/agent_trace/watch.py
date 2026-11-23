"""Live session monitoring with circuit breakers and rule-based kill switch.

Tails the active session's events.ndjson and triggers alerts when
configurable thresholds are exceeded. Zero new dependencies — stdlib only.

Rule-based nanny (--rules rules.yaml):
  Evaluates declarative rules on every event. Supports pause (SIGSTOP/SIGCONT)
  and kill (SIGTERM) actions with optional notifications. A kill writes a
  postmortem once watch has sent SIGTERM.

Watchdog mode (--timeout / --budget):
  Enforces a wall-clock timeout and/or token-cost ceiling. When either limit
  is breached the agent process is killed, whatever --on-violation says, and
  a structured post-mortem JSON file is written to the session directory.
  Killing needs --agent-pid; without it watch alerts and warns once. An
  optional --on-death command is invoked with the post-mortem path
  substituted for {post_mortem_path}. --max-duration and --max-cost set the
  same limits but act with --on-violation, like the other checks.

  Time counts from when watch starts. A duration limit that kills, from
  --timeout, or from --max-duration, watchers.duration.max_minutes or a
  built-in timeout:D while the action is kill, is checked on every poll
  cycle, so an agent that writes no events is still killed at the deadline.
  Any other duration limit is checked when an event arrives.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .cost import TokenCounter, _usage_dollars
from .compaction import CompactionCheckpointWatcher
from .event_data import (
    EDIT_TOOLS,
    PATH_KEYS,
    NonObjectEventDataError,
    UnusableTimestampError,
    event_arguments,
    left_out_event_warning,
    left_out_timestamp_warning,
    patch_paths,
    skipped_event_warning,
    str_field,
    tool_arguments,
)
from .http_post import (
    no_redirect_opener as _stream_opener,
    otlp_traces_url as _otlp_traces_url,
    post_error_detail as _stream_error_detail,
    url_endpoint as _stream_endpoint,
    url_problem as _stream_to_problem,
)
from .inert import inert
from .models import EventType, TraceEvent, usable_time
from .otlp_proto import (
    CONTENT_TYPES as _OTLP_CONTENT_TYPES,
    PROTOCOL_JSON as _OTLP_JSON,
    EncodeError as _OtlpEncodeError,
    encode_body as _otlp_body,
    resolve_protocol as _resolve_otlp_protocol,
    unsupported_media_hint as _unsupported_media_hint,
)
from .postmortem import clear_heartbeat, write_heartbeat
from .process_identity import (
    PID_REUSED_REASON,
    ProcessListError,
    descendants,
    list_processes,
    pid_reused,
    process_identity,
)
from .project_budget import (
    ProjectBudgetConfig,
    ProjectBudgetConfigError,
    ProjectBudgetStatus,
    format_project_budget_status,
    load_project_budget_config,
    mapping_from_pairs,
    project_budget_keys,
    project_budget_status,
)
from .session_paths import absolute_cwd, resolve_path
from .store import TraceStore, validate_optional_tenant_id


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

# The units _parse_duration accepts, case-insensitive, in seconds.
_DURATION_UNITS = {
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), 1),
    **dict.fromkeys(("m", "min", "mins", "minute", "minutes"), 60),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), 3600),
    **dict.fromkeys(("d", "day", "days"), 86400),
}
# One number and its unit, with optional spaces around them.
_DURATION_PART = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*([a-z]+)\s*")


def _parse_duration(value: str) -> float:
    """Parse a human-readable duration string to seconds.

    A bare number is seconds: 90. Anything else must be one or more groups
    of a number and a unit, with optional spaces: 30s, 5m, 2h, 1d, 1h30m,
    1h 30m, 30mins, 2hr, 5 minutes. The units, case-insensitive, are s,
    sec, secs, second, seconds, m, min, mins, minute, minutes, h, hr, hrs,
    hour, hours, d, day, days. Raises ValueError for anything else, such
    as 1w, 1h30, 1e3m, -5m or 5 apples, rather than guess.
    """
    value = value.strip()
    if not value:
        raise ValueError("empty duration")

    # bare number → seconds
    try:
        return float(value)
    except ValueError:
        pass

    text = value.lower()
    total = 0.0
    pos = 0
    while pos < len(text):
        m = _DURATION_PART.match(text, pos)
        if m is None or m.group(2) not in _DURATION_UNITS:
            raise ValueError(f"cannot parse duration: {value!r}")
        total += float(m.group(1)) * _DURATION_UNITS[m.group(2)]
        pos = m.end()
    if total == 0.0:
        raise ValueError(f"cannot parse duration: {value!r}")
    return total


def _limit_seconds_problem(seconds: float) -> str | None:
    """Return why *seconds* cannot be a --timeout or --max-duration, or None.

    A NaN or infinite limit never fires, and a limit of 0 or less fires at
    the first check.
    """
    try:
        seconds = float(seconds)
    except OverflowError:
        return "it must be a finite number of seconds"
    if not math.isfinite(seconds):
        return "it must be a finite number of seconds"
    if seconds <= 0:
        return "it must be more than 0 seconds"
    return None


def _limit_dollars_problem(dollars: float) -> str | None:
    """Return why *dollars* cannot be a --budget or --max-cost, or None.

    0 is allowed: any spend crosses it.
    """
    try:
        dollars = float(dollars)
    except OverflowError:
        return "it must be a finite number of dollars"
    if not math.isfinite(dollars):
        return "it must be a finite number of dollars"
    if dollars < 0:
        return "it must not be negative"
    return None


def _read_limit_flags(args: argparse.Namespace) -> tuple[float | None, float | None]:
    """Check --timeout, --max-duration, --budget and --max-cost.

    Returns the --timeout seconds and the --budget dollars, each None when
    the flag is not given. Raises ValueError naming the first bad flag.
    """
    timeout_seconds: float | None = None
    timeout_str = getattr(args, "timeout", None)
    # An empty value is refused too: --timeout "$T" with T unset must not
    # drop the kill.
    if timeout_str is not None:
        try:
            timeout_seconds = _parse_duration(timeout_str)
        except ValueError as exc:
            raise ValueError(f"invalid --timeout value: {exc}") from None
        problem = _limit_seconds_problem(timeout_seconds)
        if problem:
            raise ValueError(f"invalid --timeout value: {problem}")
    max_duration = getattr(args, "max_duration", None)
    if max_duration is not None:
        problem = _limit_seconds_problem(max_duration)
        if problem:
            raise ValueError(f"invalid --max-duration value: {problem}")

    budget: float | None = None
    budget_str = getattr(args, "budget", None)
    if budget_str is not None:
        try:
            budget = float(budget_str)
        except ValueError:
            raise ValueError(f"invalid --budget value: {budget_str!r}") from None
        problem = _limit_dollars_problem(budget)
        if problem:
            raise ValueError(f"invalid --budget value: {problem}")
    max_cost = getattr(args, "max_cost", None)
    if max_cost is not None:
        problem = _limit_dollars_problem(max_cost)
        if problem:
            raise ValueError(f"invalid --max-cost value: {problem}")
    return timeout_seconds, budget


# ---------------------------------------------------------------------------
# Alert actions
# ---------------------------------------------------------------------------

def _alert_terminal(message: str, out: TextIO = sys.stderr, *, hint: str = "") -> None:
    """Show *message* as one alert line on *out*, and *hint* on the next line.

    The message can hold the agent's text, so it goes through inert(): an
    escape code or line break shows as an escape and cannot hide or split
    the alert.
    """
    out.write(f"[watch] ⚠️  {inert(message)}\n")
    if hint:
        out.write(f"  {inert(hint)}\n")
    out.flush()


def _alert_file(message: str, log_path: str = ".agent-traces/alerts.log") -> None:
    """Append *message* to the alert log as one line. Never raises.

    The message goes through inert(), so a line break in the agent's text
    cannot forge a log entry. A failed write prints one line to stderr and
    the alert goes on, so an unwritable log cannot stop a pause or kill.
    """
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as f:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            f.write(f"{ts}  {inert(message)}\n")
    except (OSError, ValueError) as exc:
        # ValueError covers a NUL in the path. inert() already wrote a lone
        # surrogate in the message as \ud800, so it no longer raises here.
        sys.stderr.write(f"[watch] Could not write alert log: {exc}\n")
        sys.stderr.flush()


# Seconds to wait for the --webhook answer.
_WEBHOOK_TIMEOUT = 5.0


def _alert_webhook(message: str, url: str) -> None:
    """POST *message* as JSON to the --webhook URL. Never raises.

    Redirects are not followed and any answer other than 2xx is a failure.
    A failure prints one line to stderr that names only the URL's
    scheme://host[:port], because the path and query can hold a token.
    """
    problem = _stream_to_problem(url)
    if problem:
        sys.stderr.write(f"[watch] Webhook URL not used: {problem}\n")
        sys.stderr.flush()
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": message, "source": "agent-strace"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _stream_opener().open(req, timeout=_WEBHOOK_TIMEOUT):
            pass
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            exc.close()
        endpoint = _stream_endpoint(url)
        target = f" to {endpoint}" if endpoint else ""
        sys.stderr.write(f"[watch] Webhook POST{target} failed: {_stream_error_detail(exc)}\n")
        sys.stderr.flush()


_SIGKILL_DELAY = 5.0
"""Seconds from a kill's SIGTERM to its SIGKILL. Tests shorten it."""

# How long watch_session waits past a due SIGKILL for it to be sent: the
# step lists the processes first, and ps may take up to its 2s timeout.
_SIGKILL_WAIT_AFTER_DUE = 2.0
# A wait that lasts this long gets one line that says why and how to stop it.
_SIGKILL_WAIT_NOTICE_AFTER = 1.0
# How many PIDs one line names before "and N more".
_PIDS_SHOWN = 10


class _SigkillStep:
    """The SIGKILL that follows a kill's SIGTERM by _SIGKILL_DELAY seconds.

    It goes to the agent unless its PID was reused, as before, and to each
    child process that got SIGTERM and whose identity in a new listing is
    still the recorded one. It prints nothing.
    """

    def __init__(
        self, pid: int, start_time: str | None, children: list[tuple[int, str]],
        sigterm_at: float,
    ):
        self.pid = pid
        self.start_time = start_time
        self.children = children
        self.due = sigterm_at + _SIGKILL_DELAY
        self.done = threading.Event()

    def run(self) -> None:
        try:
            time.sleep(max(0.0, self.due - time.monotonic()))
            self._send()
        finally:
            self.done.set()

    def _send(self) -> None:
        if not pid_reused(self.pid, self.start_time):
            try:
                os.kill(self.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if not self.children:
            return
        try:
            table = list_processes()
        except ProcessListError:
            return
        for pid, identity in self.children:
            entry = table.get(pid)
            if entry is not None and entry.identity == identity:
                _send_signal(pid, signal.SIGKILL)

    def running(self) -> list[int]:
        """Return the PIDs this kill signalled that some process still has."""
        pids = [self.pid] + [pid for pid, _ in self.children]
        return [pid for pid in pids if _send_signal(pid, 0) != _NO_SUCH_PROCESS]

    def settled(self) -> bool:
        """Return True once SIGKILL was sent or every signalled process is gone."""
        return self.done.is_set() or not self.running()


# Kills whose SIGKILL step watch_session waits for before it returns.
_sigkill_steps: list[_SigkillStep] = []


def _pid_list(pids: list[int]) -> str:
    shown = ", ".join(str(pid) for pid in pids[:_PIDS_SHOWN])
    more = len(pids) - _PIDS_SHOWN
    return f"{shown} and {more} more" if more > 0 else shown


def _kill_process(pid: int, start_time: str | None = None) -> bool:
    """Send SIGTERM, then SIGCONT, and SIGKILL 5s later from a background thread.

    SIGCONT lets a stopped agent act on SIGTERM. A stopped process does not
    run its SIGTERM handler until it is continued, and on Linux even the
    default action waits for that: only SIGKILL ends a stopped process at
    once. The agent may have been stopped by watch (a pause or
    require_approval rule, the editor's Pause) or from outside watch
    (Ctrl-Z, kill -STOP), so SIGCONT follows every SIGTERM, as in systemd
    and daemontools.

    The agent's child processes get the same signals: every process whose
    parent chain leads to *pid*, in any process group or session. They are
    listed once, just before the agent's SIGTERM, and get SIGTERM and
    SIGCONT right after the agent. SIGKILL goes only to a listed process
    whose identity is still the listed one. A process started after the
    listing, such as one the agent starts to shut down, is never
    signalled. Neither is PID 1, watch itself, a process watch started, or
    a process between the agent and watch when watch runs under the
    agent. One line names the child processes that got SIGTERM, one names
    those it could not reach, and one says when they could not be listed.
    Their command lines are never printed. Without POSIX signals only
    *pid* is signalled.

    Return True if SIGTERM was sent to the agent. If it was not, print one
    line that says why, list nothing, send no other signal and return
    False. A SIGCONT that fails is ignored and does not change the result.

    ``start_time`` is the agent's identity from :func:`process_identity`.
    No signal is sent once the PID belongs to another process, so an
    agent that exits on SIGTERM cannot have its PID reused and then
    continued or SIGKILLed.

    watch_session waits for the SIGKILL step before it returns; see
    _wait_for_sigkill.
    """
    posix = os.name == "posix"
    reason = PID_REUSED_REASON if pid_reused(pid, start_time) else None
    if reason is None and posix:
        # Signal 0 makes the checks SIGTERM would make, so a kill that
        # cannot reach the agent does not list its child processes.
        reason = _send_signal(pid, 0)
    children: list[tuple[int, str]] = []
    list_error = ""
    if reason is None and posix:
        try:
            children = descendants(list_processes(), pid, os.getpid())
        except ProcessListError as exc:
            list_error = str(exc)
    if reason is None:
        reason = _send_signal(pid, signal.SIGTERM)
    sigterm_at = time.monotonic()
    if reason is not None:
        _alert_terminal(f"Agent pid {pid} was not killed: {reason}")
        return False

    termed: list[tuple[int, str]] = []
    refused: list[int] = []
    for child, identity in children:
        child_reason = _send_signal(child, signal.SIGTERM)
        if child_reason is None:
            termed.append((child, identity))
        elif child_reason != _NO_SUCH_PROCESS:
            refused.append(child)

    # After SIGTERM, never before, so each process handles the pending
    # SIGTERM as soon as it runs again.
    sigcont = getattr(signal, "SIGCONT", None)
    if sigcont is not None:
        if not pid_reused(pid, start_time):
            _send_signal(pid, sigcont)
        for child, _ in termed:
            _send_signal(child, sigcont)

    if termed:
        noun = "child process" if len(termed) == 1 else "child processes"
        _alert_terminal(f"Killing {len(termed)} {noun}: {_pid_list([c for c, _ in termed])}")
    if refused:
        if len(refused) == 1:
            _alert_terminal(f"Child process {refused[0]} not killed: {_NOT_ALLOWED}")
        else:
            _alert_terminal(
                f"Child processes {_pid_list(refused)} not killed: not allowed to signal them"
            )
    if list_error:
        _alert_terminal(
            f"Could not list the agent's child processes, so they were not killed: {list_error}"
        )

    step = _SigkillStep(pid, start_time, termed, sigterm_at)
    if posix:
        _sigkill_steps.append(step)
    threading.Thread(target=step.run, daemon=True).start()
    return True


def _wait_for_sigkill(out: TextIO) -> None:
    """Wait for the SIGKILL step of each kill, so watch does not end first.

    Once watch exits, the thread that sends SIGKILL is gone. The wait ends
    when SIGKILL was sent, when every process the kill signalled is gone,
    or _SIGKILL_WAIT_AFTER_DUE seconds after SIGKILL was due. If it lasts
    _SIGKILL_WAIT_NOTICE_AFTER seconds, one line says why and that Ctrl-C
    stops it. Ctrl-C ends the wait without a traceback, and one line names
    the signalled PIDs that are still in use.
    """
    steps = list(_sigkill_steps)
    _sigkill_steps.clear()
    if not steps:
        return
    deadline = max(step.due for step in steps) + _SIGKILL_WAIT_AFTER_DUE
    notice_at: float | None = time.monotonic() + _SIGKILL_WAIT_NOTICE_AFTER
    try:
        while True:
            steps = [step for step in steps if not step.settled()]
            now = time.monotonic()
            if not steps or now >= deadline:
                return
            if notice_at is not None and now >= notice_at:
                seconds = max(1, math.ceil(max(step.due for step in steps) - now))
                out.write(
                    f"[watch] Waiting {seconds}s to send SIGKILL to processes that did "
                    "not exit on SIGTERM; press Ctrl-C to stop waiting.\n"
                )
                out.flush()
                notice_at = None
            steps[0].done.wait(min(0.1, deadline - now))
    except KeyboardInterrupt:
        running = [pid for step in steps if not step.done.is_set() for pid in step.running()]
        line = "\n[watch] Stopped waiting before SIGKILL."
        if running:
            line += f" Still running: {_pid_list(running)}"
        out.write(line + "\n")
        out.flush()


_NO_SUCH_PROCESS = "no process has that PID"
_NOT_ALLOWED = "not allowed to signal it"


def _send_signal(pid: int, sig: int) -> str | None:
    """Send *sig* to *pid*. Return None if it was sent, else why it was not."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return _NO_SUCH_PROCESS
    except PermissionError:
        return _NOT_ALLOWED
    return None


def _pause_process(pid: int) -> str | None:
    """Send SIGSTOP. Return None if it was sent, else why it was not."""
    return _send_signal(pid, signal.SIGSTOP)


def _resume_process(pid: int) -> str | None:
    """Send SIGCONT. Return None if it was sent, else why it was not."""
    return _send_signal(pid, signal.SIGCONT)


def _report(msg: str, out: TextIO | None = None, alert_log: str | None = None) -> None:
    """Show *msg* on *out* (the terminal when None) and add it to *alert_log*.

    Each output is one line, with *msg* made inert like an alert.
    """
    if out is None:
        _alert_terminal(msg)
    else:
        out.write(f"[watch] {inert(msg)}\n")
        out.flush()
    if alert_log:
        _alert_file(msg, alert_log)


def _pause_agent(
    state: "WatchState", out: TextIO | None = None, alert_log: str | None = None,
) -> bool:
    """SIGSTOP the agent. Return True if the signal was sent.

    If it was not, say why and return False. Callers record a pause owner
    only on True, so watch never holds a pause the agent did not get.
    """
    pid = state.agent_pid
    problem = _pause_process(pid)
    if problem is None:
        return True
    _report(f"Agent pid {pid} was not paused: {problem}. Watch holds no pause.", out, alert_log)
    return False


def _resume_agent(
    state: "WatchState", out: TextIO | None = None, alert_log: str | None = None,
) -> None:
    """SIGCONT the agent and drop the pause watch holds.

    The pause is dropped even when SIGCONT was not sent: the PID is gone,
    or watch may no longer signal it, so watch could never release that
    pause and would keep the idle timeout stopped for good. Watch says so
    and, when the process still exists, how to resume it by hand.
    """
    pid = state.agent_pid
    problem = _resume_process(pid)
    state.pause_owner = None
    state.approval_request_id = None
    if problem is None:
        return
    msg = f"Agent pid {pid} was not resumed: {problem}. Pause cleared."
    if problem == _NOT_ALLOWED:
        msg += f" If the agent is still stopped, run 'kill -CONT {pid}' as its owner."
    _report(msg, out, alert_log)


def validate_agent_pid(pid: int) -> str | None:
    """Return why *pid* cannot be used as the agent PID, or None if it can.

    The check sends signal 0, which tests for the process without affecting it.
    """
    if not hasattr(signal, "SIGSTOP"):
        return "--agent-pid is not supported on this platform (no SIGSTOP)"
    if pid <= 1:
        return f"--agent-pid must be greater than 1, got {pid}"
    if pid == os.getpid():
        return f"--agent-pid {pid} is this watch process"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return f"no process with PID {pid}"
    except PermissionError:
        return f"not allowed to signal PID {pid}"
    except OSError as exc:
        return f"cannot check PID {pid}: {exc}"
    return None


NO_AGENT_PID_HINT = (
    "no agent PID is known. Restart watch with --agent-pid PID "
    "to let it pause, resume and kill the agent. This warning is shown once."
)


def _agent_pid_reused(
    state: "WatchState", out: TextIO | None = None, alert_log: str | None = None,
    *, signalling: bool = True,
) -> bool:
    """Return True, and stop using the agent PID, if another process now has it.

    Call before signalling the agent. When the start time recorded at watch
    startup can be read and differs, the agent has exited and the OS gave
    its PID to another process: watch says so, records it in *alert_log*,
    marks the agent exited, drops the PID and any pause it held, and never
    signals that PID again.
    When either start time is unknown this returns False and watch signals
    as before. Pass *signalling* False when watch was not about to signal,
    so the message does not say a signal was held back.
    """
    pid = state.agent_pid
    if not pid or not pid_reused(pid, state.agent_start_time):
        return False
    held_back = ", so no agent was signalled" if signalling else ""
    msg = (
        f"Agent pid {pid}: {PID_REUSED_REASON} (the agent exited and its PID "
        f"was reused){held_back}. Watch will not signal PID "
        f"{pid} again; restart watch with --agent-pid PID for a new agent."
    )
    if state.paused:
        msg += " Pause cleared."
    _report(msg, out, alert_log)
    state.agent_exited = True
    state.agent_pid = None
    state.agent_start_time = None
    state.pause_owner = None
    state.approval_request_id = None
    # The message above already explains why watch cannot signal.
    state.warned_no_agent_pid = True
    return True


def _warn_no_agent_pid(state: "WatchState", what: str, out: TextIO | None = None) -> None:
    """Say once that watch wanted to signal the agent but has no PID."""
    if state.warned_no_agent_pid:
        return
    state.warned_no_agent_pid = True
    _report(f"Cannot {what}: {NO_AGENT_PID_HINT}", out)


# ---------------------------------------------------------------------------
# Reading --config and --rules files
# ---------------------------------------------------------------------------

def _shown_path(path: str) -> str:
    """Return *path* as a refusal shows it: as given when it is printable."""
    return path if path and path.isprintable() else repr(path)


def _read_utf8_file(path: str, error: type[ValueError]) -> str:
    """Return the text of the file at *path*.

    Raises *error* when the file cannot be read, is not UTF-8, or starts with
    a byte order mark.
    """
    if not path:
        raise error("the path is empty")
    try:
        raw = Path(path).read_bytes()
    except (OSError, ValueError) as exc:
        detail = getattr(exc, "strerror", None)
        raise error(f"cannot read it ({detail})" if detail else "cannot read it") from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise error("it is not valid UTF-8") from None
    if text.startswith("\ufeff"):
        # json.loads and the rules reader do not skip a byte order mark, so
        # a clear line beats the parse error it would cause. The project
        # budget config files that project_budget.py reads may start with one.
        raise error("it starts with a byte order mark; save it without one")
    return text


def _parse_json_text(text: str, error: type[ValueError], **kwargs: Any) -> Any:
    """Return json.loads(text, **kwargs), raising *error* when it fails."""
    try:
        return json.loads(text, **kwargs)
    except json.JSONDecodeError as exc:
        raise error(
            f"it is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
        ) from None
    except RecursionError:
        raise error("it is nested too deeply") from None
    except ValueError:
        # json.loads raises a plain ValueError for an integer with more
        # digits than int() accepts.
        raise error("it has a number with too many digits") from None


# ---------------------------------------------------------------------------
# Nanny rules (--rules rules.yaml)
# ---------------------------------------------------------------------------

class WatchRulesError(ValueError):
    """A watch --rules value or rules file that watch refuses to use.

    The message says what is wrong and, for a rule, names its position and
    name. It never repeats a notify value.
    """


# The metrics WatchState.nanny_metrics returns.
_NANNY_METRICS = ("files_modified", "cost_usd", "consecutive_test_failures", "duration_minutes")
_NANNY_OPERATORS = (">", ">=", "<", "<=", "==")
# The actions _dispatch_alert carries out for a rule. alert and terminal both
# print the alert and take no other action.
_NANNY_ACTIONS = ("alert", "terminal", "file", "pause", "kill", "require_approval")
_NANNY_RULE_KEYS = ("name", "condition", "action", "notify")
_NANNY_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]*)?|\.[0-9]+", re.ASCII)


def _parse_nanny_condition(condition: str) -> tuple[str, str, int | float | None, str]:
    """Return (metric, operator, threshold, pattern) for a rule condition.

    The whole condition must be ``METRIC OP NUMBER`` or
    ``file_path matches "GLOB"``. Raises WatchRulesError otherwise, so a rule
    is never loaded with a condition it cannot evaluate.
    """
    if not isinstance(condition, str):
        raise WatchRulesError("condition must be a string")
    cond = condition.strip()

    match = re.fullmatch(r"file_path\s+matches\b\s*(.*)", cond, re.ASCII | re.DOTALL)
    if match:
        glob = match.group(1)
        quote = glob[:1]
        if (
            quote not in ("'", '"') or len(glob) < 3 or not glob.endswith(quote)
            or quote in glob[1:-1]
        ):
            raise WatchRulesError(
                'condition must be file_path matches "GLOB" with a non-empty GLOB '
                "in quotes and nothing after it"
            )
        return "file_path", "matches", None, glob[1:-1]

    match = re.fullmatch(r"(\w+)\s*([<>=!]+)\s*(.*)", cond, re.ASCII | re.DOTALL)
    if not match:
        raise WatchRulesError('condition must be METRIC OP NUMBER or file_path matches "GLOB"')
    metric, op, raw = match.groups()
    if metric not in _NANNY_METRICS:
        raise WatchRulesError(
            f"condition has an unknown metric {metric!r}; use "
            "files_modified, cost_usd, consecutive_test_failures or duration_minutes"
        )
    if op not in _NANNY_OPERATORS:
        raise WatchRulesError(
            f"condition has an unknown operator {op!r}; use >, >=, <, <= or =="
        )
    threshold: int | float | None = None
    if _NANNY_NUMBER.fullmatch(raw):
        if "." in raw:
            number = float(raw)
            threshold = number if math.isfinite(number) else None
        else:
            try:
                threshold = int(raw)
            except ValueError:
                threshold = None  # more digits than int() accepts
    if threshold is None:
        raise WatchRulesError(
            "condition must end with one plain number, such as 5 or 0.50, "
            "and nothing after it"
        )
    return metric, op, threshold, ""


@dataclass
class NannyRule:
    """A declarative rule evaluated on every event.

    The constructor raises WatchRulesError for a condition it cannot parse.
    """
    name: str
    condition: str          # e.g. "files_modified > 20", "cost_usd > 5.00"
    action: str             # one of _NANNY_ACTIONS
    notify: str = ""        # e.g. "slack:#alerts", "email:me@example.com"
    fired: bool = field(default=False, compare=False)

    # Parsed condition components (set by _parse_condition)
    _metric: str = field(default="", init=False, repr=False)
    _op: str = field(default="", init=False, repr=False)
    _threshold: Any = field(default=None, init=False, repr=False)
    _pattern: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._parse_condition()

    def _parse_condition(self) -> None:
        """Parse condition string into metric, operator, threshold."""
        self._metric, self._op, self._threshold, self._pattern = (
            _parse_nanny_condition(self.condition)
        )

    @property
    def reads_paths(self) -> bool:
        """True for a ``file_path matches`` rule, which reads the paths events name."""
        return self._metric == "file_path" and self._op == "matches"

    def evaluate(
        self,
        metrics: dict[str, Any],
        event: TraceEvent | None = None,
        cwd: str = "",
    ) -> bool:
        """Return True if this rule's condition is satisfied.

        A ``file_path matches`` rule tests each path the event names (see
        _event_paths), as given and resolved against *cwd*, the session's
        working directory.
        """
        if self.reads_paths and event:
            for path in _event_paths(event):
                if fnmatch.fnmatch(resolve_path(path, cwd), self._pattern) or fnmatch.fnmatch(
                    path, self._pattern,
                ):
                    return True
            return False

        if not self._metric or self._threshold is None:
            return False

        value = metrics.get(self._metric)
        if value is None:
            return False

        op = self._op
        t = self._threshold
        if op == ">":
            return value > t
        if op == ">=":
            return value >= t
        if op == "<":
            return value < t
        if op == "<=":
            return value <= t
        if op == "==":
            return value == t
        return False


class _RulesObject(dict):
    """An object from a rules file that remembers the first key given twice."""

    duplicate_key: str | None = None


def _rules_object(pairs: list[tuple[str, Any]]) -> _RulesObject:
    """json object_pairs_hook that records a duplicate key instead of hiding it."""
    obj = _RulesObject()
    for key, value in pairs:
        if key in obj and obj.duplicate_key is None:
            obj.duplicate_key = key
        obj[key] = value
    return obj


def _nanny_rule_from_data(item: Any, position: int) -> NannyRule:
    """Return the NannyRule for entry *position* (1-based) of a rules list."""
    if not isinstance(item, dict):
        raise WatchRulesError(f"rule {position} must be an object")
    name = item.get("name", "unnamed")
    if not isinstance(name, str):
        raise WatchRulesError(f"rule {position}: name must be a string")
    label = f"rule {position} {name!r}" if "name" in item and name else f"rule {position}"

    def refuse(problem: str) -> WatchRulesError:
        return WatchRulesError(f"{label}: {problem}")

    duplicate = getattr(item, "duplicate_key", None)
    if duplicate is not None:
        raise refuse(f"it has {duplicate!r} twice")
    for key in item:
        if key not in _NANNY_RULE_KEYS:
            raise refuse(
                f"unknown key {key!r}; a rule has only name, condition, action and notify"
            )
    if "condition" not in item:
        raise refuse("it has no condition")
    try:
        _parse_nanny_condition(item["condition"])
    except WatchRulesError as exc:
        raise refuse(str(exc)) from None
    if "action" not in item:
        raise refuse("it has no action")
    action = item["action"]
    if not isinstance(action, str) or action not in _NANNY_ACTIONS:
        raise refuse(f"action must be {_one_of(_NANNY_ACTIONS)}")
    notify = item.get("notify", "")
    if not isinstance(notify, str):
        raise refuse("notify must be a string")
    return NannyRule(name=name, condition=item["condition"], action=action, notify=notify)


def _nanny_rules_from_data(data: Any) -> list[NannyRule]:
    """Return the rules in a parsed rules file, refusing anything else in it."""
    if not isinstance(data, dict):
        raise WatchRulesError("the top level must be a JSON object")
    for key in data:
        if key != "rules":
            raise WatchRulesError(
                f"the top level has an unknown key {key!r}; it must hold only rules"
            )
    if getattr(data, "duplicate_key", None) is not None:
        raise WatchRulesError("the top level has 'rules' twice")
    if "rules" not in data:
        raise WatchRulesError("the top level has no rules key")
    items = data["rules"]
    if not isinstance(items, list) or not items:
        raise WatchRulesError("rules must be a non-empty list")
    return [_nanny_rule_from_data(item, position) for position, item in enumerate(items, 1)]


def _load_nanny_rules(path: str) -> list[NannyRule]:
    """Load the rules in a --rules file.

    A file whose name ends in .json, or whose text starts with { or [, is
    read as JSON; any other file is read as the YAML subset that
    _parse_rules_yaml accepts. Raises WatchRulesError when the file cannot be
    read or holds anything watch cannot use, so no rule is dropped or changed.
    """
    if path and os.path.isdir(path):
        raise WatchRulesError("it is a directory, not a rules file")
    text = _read_utf8_file(path, WatchRulesError)
    if not text.strip():
        raise WatchRulesError("it is empty")
    if path.lower().endswith(".json") or text.lstrip(" \t\r\n")[:1] in ("{", "["):
        data = _parse_json_text(text, WatchRulesError, object_pairs_hook=_rules_object)
    else:
        data = _parse_rules_yaml(text)
    return _nanny_rules_from_data(data)


_YAML_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)[ \t]*:(?:[ \t]+(.*))?", re.ASCII)
_YAML_DOCUMENT_START = re.compile(r"---(?:[ \t]+#.*)?")
# YAML reads an unquoted value that starts with one of these as a flow
# collection, anchor, alias, tag, block scalar or reserved character.
_YAML_INDICATORS = "[]{},&*!|>%@`"


def _yaml_scalar(text: str | None, line: int) -> str | None:
    """Return one value of the rules-file YAML subset; None for empty or null.

    A value in matching quotes loses that one pair of quotes. In single
    quotes '' stands for one quote; in double quotes the only escapes are
    \\" and \\\\. An unquoted value ends before a comment that starts with
    whitespace and #.
    """
    if not text or text.startswith("#"):
        return None
    quote = text[0]
    if quote in ("'", '"'):
        chars: list[str] = []
        index = 1
        while True:
            if index >= len(text):
                raise WatchRulesError(f"line {line}: the closing quote is missing")
            char = text[index]
            if char == quote:
                if quote == "'" and text[index + 1:index + 2] == "'":
                    chars.append("'")
                    index += 2
                    continue
                break
            if char == "\\" and quote == '"':
                escaped = text[index + 1:index + 2]
                if escaped not in ('"', "\\"):
                    raise WatchRulesError(
                        f'line {line}: in double quotes a backslash may only escape " or \\; '
                        "use single quotes for other backslashes"
                    )
                chars.append(escaped)
                index += 2
                continue
            chars.append(char)
            index += 1
        if not re.fullmatch(r"(?:[ \t]+(?:#.*)?)?", text[index + 1:], re.DOTALL):
            raise WatchRulesError(f"line {line}: only a comment may follow the closing quote")
        return "".join(chars)
    comment = re.search(r"[ \t]#", text)
    if comment:
        text = text[:comment.start()].rstrip(" \t")
    if (
        text[0] in _YAML_INDICATORS or text in ("-", "?", ":")
        or text[:2] in ("- ", "? ", ": ", "-\t", "?\t", ":\t")
    ):
        raise WatchRulesError(
            f"line {line}: a value cannot start with {text[0]!r} unless it is quoted"
        )
    if ": " in text or ":\t" in text or text.endswith(":"):
        raise WatchRulesError(
            f"line {line}: a value cannot contain ': ' or end with ':' unless it is quoted"
        )
    return None if text in ("~", "null", "Null", "NULL") else text


def _yaml_rule_pair(rule: _RulesObject, text: str, line: int) -> None:
    """Add one key: value line of the YAML subset to *rule*."""
    pair = _YAML_PAIR.fullmatch(text)
    if pair is None:
        raise WatchRulesError(f"line {line}: expected key: value")
    key = pair.group(1)
    if key in rule and rule.duplicate_key is None:
        rule.duplicate_key = key
    rule[key] = _yaml_scalar(pair.group(2), line)


def _parse_rules_yaml(text: str) -> _RulesObject:
    """Parse a rules file written in the YAML subset that watch reads.

    The subset is a top-level rules: key holding a block list of rules, each
    a block mapping of keys to one-line values. Blank lines, comment lines
    and a leading --- are allowed. Raises WatchRulesError for the first line
    outside the subset instead of skipping it.
    """
    top = _RulesObject()
    rules: list[Any] = []
    item_indent = 0
    key_indent: int | None = None
    seen_content = False
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for number, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip(" \t")
        body = line.lstrip(" \t")
        if not body or body.startswith("#"):
            continue
        lead = line[:len(line) - len(body)]
        if "\t" in lead:
            raise WatchRulesError(f"line {number}: indent with spaces, not tabs")
        indent = len(lead)
        first_line, seen_content = not seen_content, True

        if "rules" not in top:
            if first_line and indent == 0 and _YAML_DOCUMENT_START.fullmatch(body):
                continue
            if indent:
                raise WatchRulesError(f"line {number}: unexpected indentation")
            pair = _YAML_PAIR.fullmatch(body)
            if pair is None:
                raise WatchRulesError(f"line {number}: expected rules:")
            if pair.group(1) != "rules":
                raise WatchRulesError(
                    f"line {number}: the top level has an unknown key "
                    f"{pair.group(1)!r}; it must hold only rules"
                )
            value = pair.group(2)
            if value is not None and not value.startswith("#"):
                raise WatchRulesError(
                    f"line {number}: list the rules on the lines below rules:"
                )
            top["rules"] = None
            continue

        if body == "-" or body.startswith(("- ", "-\t")):
            if not rules:
                item_indent = indent
            elif indent != item_indent:
                raise WatchRulesError(f"line {number}: unexpected indentation")
            rest = body[1:]
            entry = rest.lstrip(" ")
            if entry.startswith("\t"):
                raise WatchRulesError(f"line {number}: indent with spaces, not tabs")
            top["rules"] = rules
            if not entry or entry.startswith("#"):
                # A bare "-": the rule's keys start on the next line.
                rules.append(None)
                key_indent = None
                continue
            rules.append(_RulesObject())
            key_indent = indent + 1 + len(rest) - len(entry)
            _yaml_rule_pair(rules[-1], entry, number)
            continue

        if indent == 0:
            pair = _YAML_PAIR.fullmatch(body)
            if pair is None:
                raise WatchRulesError(f"line {number}: expected - at the start of a rule")
            if pair.group(1) == "rules":
                raise WatchRulesError(f"line {number}: the top level has 'rules' twice")
            raise WatchRulesError(
                f"line {number}: the top level has an unknown key "
                f"{pair.group(1)!r}; it must hold only rules"
            )
        if not rules:
            raise WatchRulesError(f"line {number}: expected - at the start of a rule")
        if key_indent is None:
            if indent <= item_indent:
                raise WatchRulesError(f"line {number}: unexpected indentation")
            key_indent = indent
            rules[-1] = _RulesObject()
        elif indent != key_indent:
            raise WatchRulesError(f"line {number}: unexpected indentation")
        _yaml_rule_pair(rules[-1], body, number)
    return top


# ---------------------------------------------------------------------------
# Watcher state machines
# ---------------------------------------------------------------------------

@dataclass
class OperationRule:
    """A per-operation enforcement rule."""
    tool_name: str          # e.g. "bash", "write", "*"
    pattern: str            # glob pattern matched against command/path
    action: str             # "alert" | "block"
    reason: str = ""        # human-readable explanation

    def matches(self, tool: str, target: str) -> bool:
        import fnmatch
        tool_match = (
            self.tool_name == "*"
            or self.tool_name.lower() == tool.lower()
        )
        if not tool_match:
            return False
        return fnmatch.fnmatch(target.lower(), self.pattern.lower()) or self.pattern == "*"


# ---------------------------------------------------------------------------
# watch --config file checks
# ---------------------------------------------------------------------------

# Actions _dispatch_alert takes; watchers.retry.alert sets it for every watcher.
_WATCH_ACTIONS = ("terminal", "file", "kill", "pause", "require_approval")
_OPERATION_RULE_ACTIONS = ("alert", "block")
# The built-in rule names check_event looks for in built_in_rules.
_CONFIG_BUILT_IN_RULES = ("mcp-poisoning", "loop", "cognitive-debt")


class WatchConfigError(ValueError):
    """A watch --config file that watch refuses to use.

    The message says what is wrong and, for a key, names its path and what
    it must be. It never repeats a value from the file.
    """


def _one_of(choices: tuple[str, ...]) -> str:
    if len(choices) == 2:
        return f"{choices[0]} or {choices[1]}"
    return "one of " + ", ".join(choices)


def _config_object(parent: dict, key: str, path: str) -> dict:
    """Return parent[key], which must be a JSON object; {} when it is absent."""
    if key not in parent:
        return {}
    value = parent[key]
    if not isinstance(value, dict):
        raise WatchConfigError(f"{path} must be a JSON object")
    return value


def _config_whole_number(
    parent: dict, key: str, path: str, minimum: int, maximum: int | None = None,
) -> int | None:
    """Return parent[key] as an int, or None when it is absent.

    A float with no fraction, such as 2.0, counts as a whole number.
    """
    if key not in parent:
        return None
    value = parent[key]
    number: int | None = None
    if isinstance(value, int) and not isinstance(value, bool):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    if number is not None and number >= minimum and (maximum is None or number <= maximum):
        return number
    if maximum is None:
        raise WatchConfigError(f"{path} must be a whole number, {minimum} or more")
    raise WatchConfigError(f"{path} must be a whole number from {minimum} to {maximum}")


def _config_number(
    parent: dict, key: str, path: str, minimum: int, maximum: int | None = None,
) -> float | None:
    """Return parent[key] as a finite float, or None when it is absent."""
    if key not in parent:
        return None
    value = parent[key]
    number: float | None = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:
            number = None
    if (
        number is not None and math.isfinite(number) and number >= minimum
        and (maximum is None or number <= maximum)
    ):
        return number
    if maximum is None:
        raise WatchConfigError(f"{path} must be a finite number, {minimum} or more")
    raise WatchConfigError(f"{path} must be a number from {minimum} to {maximum}")


def _config_positive_number(parent: dict, key: str, path: str) -> float | None:
    """Return parent[key] as a finite float more than 0, or None when it is absent."""
    problem = f"{path} must be a finite number more than 0"
    try:
        number = _config_number(parent, key, path, 0)
    except WatchConfigError:
        raise WatchConfigError(problem) from None
    if number is not None and number <= 0:
        raise WatchConfigError(problem)
    return number


def _config_string(parent: dict, key: str, path: str, non_empty: bool = False) -> str | None:
    """Return parent[key], which must be a string, or None when it is absent."""
    if key not in parent:
        return None
    value = parent[key]
    if not isinstance(value, str) or (non_empty and not value):
        raise WatchConfigError(f"{path} must be a {'non-empty ' if non_empty else ''}string")
    return value


def _config_choice(parent: dict, key: str, path: str, choices: tuple[str, ...]) -> str | None:
    """Return parent[key], which must be one of *choices*, or None when it is absent."""
    if key not in parent:
        return None
    value = parent[key]
    if not isinstance(value, str) or value not in choices:
        raise WatchConfigError(f"{path} must be {_one_of(choices)}")
    return value


def _config_operation_rules(d: dict) -> list[OperationRule]:
    if "operation_rules" not in d:
        return []
    items = d["operation_rules"]
    if not isinstance(items, list):
        raise WatchConfigError("operation_rules must be a JSON array")
    rules: list[OperationRule] = []
    for index, item in enumerate(items):
        path = f"operation_rules[{index}]"
        if not isinstance(item, dict):
            raise WatchConfigError(f"{path} must be a JSON object")
        tool = _config_string(item, "tool", f"{path}.tool")
        pattern = _config_string(item, "pattern", f"{path}.pattern")
        action = _config_choice(item, "action", f"{path}.action", _OPERATION_RULE_ACTIONS)
        reason = _config_string(item, "reason", f"{path}.reason")
        rules.append(OperationRule(
            tool_name="*" if tool is None else tool,
            pattern="*" if pattern is None else pattern,
            action=action or "alert",
            reason=reason or "",
        ))
    return rules


def _config_built_in_rules(d: dict) -> set[str] | None:
    if "built_in_rules" not in d:
        return None
    items = d["built_in_rules"]
    if not isinstance(items, list):
        raise WatchConfigError("built_in_rules must be a JSON array")
    for index, item in enumerate(items):
        if not isinstance(item, str) or item not in _CONFIG_BUILT_IN_RULES:
            raise WatchConfigError(
                f"built_in_rules[{index}] must be {_one_of(_CONFIG_BUILT_IN_RULES)}"
            )
    return set(items)


@dataclass
class WatcherConfig:
    max_retries: int = 5
    max_cost_dollars: float = 10.0
    max_duration_seconds: float = 1800.0
    loop_threshold: int = 3
    loop_window: int = 10
    loop_sequence_length: int = 3
    loop_max_repeats: int = 3
    scope_policy: str = ".agent-scope.json"
    on_violation: str = "terminal"   # one of _WATCH_ACTIONS
    webhook_url: str = ""
    alert_log: str = ".agent-traces/alerts.log"
    # Per-operation enforcement rules
    operation_rules: list[OperationRule] = field(default_factory=list)
    # Token budget threshold (1–100, percentage of context window)
    max_context_pct: int = 90
    # Watchdog: command to run once, after watch sends the agent SIGTERM,
    # with {post_mortem_path} substituted
    on_death_cmd: str = ""
    # Watchdog: --timeout and --budget kill whatever on_violation says. Only
    # the command line sets these; a config file never does.
    kill_on_timeout: bool = False
    kill_on_budget: bool = False
    # True when watchers.cost.max_dollars, --max-cost or --budget sets
    # max_cost_dollars, so it is not the default. per_session_max and a
    # built-in budget:N lower a set limit that kills, and replace any other
    # (see _cost_limit_kills).
    cost_limit_set: bool = False
    # True when watchers.duration.max_minutes, --max-duration, --timeout or
    # a built-in timeout:D sets max_duration_seconds, so it is not the
    # default. timeout:D lowers a set limit that kills, and replaces any
    # other (see _apply_builtin_rules and _duration_limit_kills).
    duration_limit_set: bool = False
    # Set when the config file sets the project budget key weekly or
    # per_session_max, even to null or 0. It then replaces the standard
    # project config for the run (see cmd_watch).
    project_budget: ProjectBudgetConfig | None = None
    # Built-in rule names enabled via --rules mcp-poisoning,budget:$5,...
    built_in_rules: set[str] = field(default_factory=set)
    cognitive_debt_threshold: float | None = None

    @classmethod
    def from_dict(cls, d: Any) -> "WatcherConfig":
        """Build a config from a parsed watch --config file.

        Raises WatchConfigError for the first known key that has the wrong
        type or an impossible value. The budget keys are checked as
        project_budget.py checks a project config file, without showing
        values. Other unknown keys are ignored.
        """
        if not isinstance(d, dict):
            raise WatchConfigError("the top level must be a JSON object")
        watchers = _config_object(d, "watchers", "watchers")
        retry_cfg = _config_object(watchers, "retry", "watchers.retry")
        cost_cfg = _config_object(watchers, "cost", "watchers.cost")
        dur_cfg = _config_object(watchers, "duration", "watchers.duration")
        loop_cfg = _config_object(watchers, "loop", "watchers.loop")
        scope_cfg = _config_object(watchers, "scope", "watchers.scope")
        webhook = _config_object(d, "webhook", "webhook")

        max_retries = _config_whole_number(retry_cfg, "max", "watchers.retry.max", 0)
        on_violation = _config_choice(retry_cfg, "alert", "watchers.retry.alert", _WATCH_ACTIONS)
        max_dollars = _config_number(cost_cfg, "max_dollars", "watchers.cost.max_dollars", 0)
        # A time limit of 0 fires at the first event, so it is refused like
        # --max-duration 0. A cost limit of 0 is kept: any spend crosses it.
        max_minutes = _config_positive_number(
            dur_cfg, "max_minutes", "watchers.duration.max_minutes",
        )
        loop = {
            key: _config_whole_number(loop_cfg, key, f"watchers.loop.{key}", 1)
            for key in ("identical_calls", "threshold", "max_repeats", "window", "sequence_length")
        }
        policy = _config_string(scope_cfg, "policy", "watchers.scope.policy", non_empty=True)
        webhook_url = _config_string(webhook, "url", "webhook.url")
        operation_rules = _config_operation_rules(d)
        max_context_pct = _config_whole_number(d, "max_context_pct", "max_context_pct", 1, 100)
        built_in_rules = _config_built_in_rules(d)
        debt_threshold = None
        if d.get("cognitive_debt_threshold") is not None:
            debt_threshold = _config_number(
                d, "cognitive_debt_threshold", "cognitive_debt_threshold", 0, 1,
            )

        values = {
            "max_retries": max_retries,
            "on_violation": on_violation,
            "max_cost_dollars": max_dollars,
            "max_duration_seconds": None if max_minutes is None else max_minutes * 60,
            "loop_threshold": next(
                (loop[key] for key in ("identical_calls", "threshold", "max_repeats")
                 if loop[key] is not None),
                None,
            ),
            "loop_window": loop["window"],
            "loop_sequence_length": loop["sequence_length"],
            "loop_max_repeats": loop["max_repeats"],
            "scope_policy": policy,
            "webhook_url": webhook_url,
            "operation_rules": operation_rules,
            "max_context_pct": max_context_pct,
            "built_in_rules": built_in_rules,
            "cognitive_debt_threshold": debt_threshold,
        }
        # Absent keys keep the dataclass defaults.
        config = cls(**{key: value for key, value in values.items() if value is not None})
        config.cost_limit_set = max_dollars is not None
        config.duration_limit_set = max_minutes is not None
        try:
            budget_keys = project_budget_keys(d, show_values=False, mapping="JSON object")
        except ValueError as exc:
            raise WatchConfigError(str(exc)) from None
        if "weekly" in budget_keys or "per_session_max" in budget_keys:
            config.project_budget = ProjectBudgetConfig.from_keys(budget_keys)
        return config

    @classmethod
    def load(cls, path: str) -> "WatcherConfig":
        """Read a watch --config file.

        Raises WatchConfigError when the file cannot be read, is not UTF-8
        JSON, or fails from_dict.
        """
        text = _read_utf8_file(path, WatchConfigError)
        # The hook lets from_dict refuse a budget key given twice.
        return cls.from_dict(
            _parse_json_text(text, WatchConfigError, object_pairs_hook=mapping_from_pairs)
        )


# ---------------------------------------------------------------------------
# Push-based event streaming
# ---------------------------------------------------------------------------

# Seconds one stream POST may take before it fails.
_STREAM_POST_TIMEOUT = 10.0
# Seconds EventStreamer.stop() waits for the worker. It is longer than one
# POST may take, so a slow collector still receives the last batch.
_STREAM_STOP_TIMEOUT = 15.0
# Once stop() has waited this many seconds, it says how long it will wait.
_STREAM_STOP_NOTICE_AFTER = 1.0
# Longest single wait for the next event. queue.get raises OverflowError for
# a timeout above threading.TIMEOUT_MAX, so a huge flush interval must never
# reach it unchanged.
_STREAM_MAX_QUEUE_WAIT = 60.0


@dataclass
class StreamConfig:
    """Configuration for push-based event streaming.

    NDJSON mode POSTs to ``url`` unchanged. OTLP mode (``otlp=True``) treats
    ``url`` as a base URL: trailing slashes are dropped from its path and
    ``/v1/traces`` is appended unless the path already ends with it. The
    query string is kept. Stream messages name only the URL's
    scheme://host[:port], never its user info, path, query or fragment.
    """
    url: str = ""                    # NDJSON endpoint or OTLP/HTTP base URL
    batch_size: int = 10             # max events per POST
    flush_interval: float = 1.0     # max seconds between flushes
    headers: dict[str, str] = field(default_factory=dict)
    # Filter: only stream these event types (empty = all)
    event_types: list[str] = field(default_factory=list)
    # When True, flush as OTLP spans instead of raw NDJSON
    otlp: bool = False
    # OTel service name used in OTLP spans
    service_name: str = "agent-trace"
    # OTLP/HTTP encoding: http/json or http/protobuf
    protocol: str = _OTLP_JSON

    @classmethod
    def from_url(cls, url: str, headers: dict[str, str] | None = None,
                 otlp: bool = False) -> "StreamConfig":
        return cls(url=url, headers=headers or {}, otlp=otlp)


def _stream_queue_wait(interval: float, elapsed: float) -> float:
    """Return how long the stream worker may wait for the next event.

    The wait never exceeds _STREAM_MAX_QUEUE_WAIT, so an infinite, NaN or
    huge flush interval means "send on batch size or at stop" instead of
    killing the worker.
    """
    remaining = interval - elapsed
    if not remaining < _STREAM_MAX_QUEUE_WAIT:
        return _STREAM_MAX_QUEUE_WAIT
    return max(0.01, remaining)


def _stream_unconfirmed_line(count: int) -> str:
    """Return the line stop() writes when it stops waiting for the worker."""
    noun = "1 event was" if count == 1 else f"{count} events were"
    return f"[stream] Stopped waiting; {noun} not confirmed sent.\n"


def _stream_flush_interval_problem(seconds: float) -> str | None:
    """Return why *seconds* cannot be a --stream-flush-interval, or None."""
    if math.isnan(seconds) or math.isinf(seconds):
        return "it must be a finite number of seconds"
    if seconds < 0:
        return "it must not be negative"
    return None


class EventStreamer:
    """Background thread that batches and POSTs events to a remote URL.

    Thread-safe: events are enqueued from the watch loop and flushed by a
    dedicated daemon thread. A batch is sent when it holds batch_size events,
    when flush_interval seconds have passed since the last flush, and at
    stop(). An infinite, NaN or huge flush_interval sends only on batch size
    and at stop().

    Each POST may take up to _STREAM_POST_TIMEOUT seconds and never follows
    a redirect. A failed POST, a 3xx, or a status other than 200 or 202
    gives one line on stderr, and the worker goes on with the next batch,
    even when the URL cannot be used at all. These lines name only the
    URL's scheme://host[:port] (see :func:`_stream_endpoint`) and never
    quote an exception message that can repeat the URL.

    NDJSON mode sends every event with its data as stored. OTLP mode
    converts each session's events to spans and sends them as OTLP JSON or
    protobuf (``StreamConfig.protocol``). The first 415 answer in OTLP mode
    adds a line naming the other --protocol. OTLP mode leaves out each event
    whose timestamp is not a Unix time in seconds, or whose data is not a
    JSON object, when the converter would read it, with one line on stderr
    per event, and still sends the other events of the batch.
    """

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self._queue: queue.Queue[TraceEvent | None] = queue.Queue()
        # Set by the worker after the first 415 hint, so it is shown once.
        self._media_hint_shown = False
        # _handed counts the events enqueue() accepted. _settled counts those
        # whose POST finished or that a stderr line already reported. Both
        # change under _lock, so stop() can count the events not confirmed
        # sent without counting a reported event twice.
        self._lock = threading.Lock()
        self._handed = 0
        self._settled = 0
        self._abandoned = False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, event: TraceEvent) -> None:
        """Add an event to the outbound queue (non-blocking)."""
        if self.config.event_types:
            if event.event_type.value not in self.config.event_types:
                return
        with self._lock:
            self._handed += 1
        self._queue.put(event)

    def stop(self) -> None:
        """Send the remaining events and wait for the worker to finish.

        Waits up to _STREAM_STOP_TIMEOUT seconds, longer than one POST may
        take. If the wait lasts _STREAM_STOP_NOTICE_AFTER seconds, one line
        says how much longer it may last and that Ctrl-C stops it. If the
        worker is still busy when the wait ends or Ctrl-C stops it, one line
        gives the number of events not confirmed sent, and the worker writes
        nothing more. Ctrl-C during the wait never raises KeyboardInterrupt.
        """
        start = time.monotonic()
        limit = _STREAM_STOP_TIMEOUT
        notice_after = min(_STREAM_STOP_NOTICE_AFTER, limit)
        try:
            self._queue.put(None)  # sentinel: send what is left, then exit
            if not self._wait_until(start + notice_after) and limit > notice_after:
                seconds = max(1, math.ceil(limit - (time.monotonic() - start)))
                sys.stderr.write(
                    f"[stream] Waiting up to {seconds}s more for the last events "
                    "to be sent; press Ctrl-C to stop waiting.\n"
                )
                self._wait_until(start + limit)
        except KeyboardInterrupt:
            pass
        if self._done.is_set():
            self._thread.join()
            return
        with self._lock:
            self._abandoned = True
            pending = self._handed - self._settled
        if pending > 0:
            sys.stderr.write(_stream_unconfirmed_line(pending))

    def _wait_until(self, deadline: float) -> bool:
        """Wait for the worker until *deadline*; return True if it finished.

        Waiting in short steps lets Ctrl-C stop the wait at once on every
        platform.
        """
        while not self._done.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._done.wait(min(remaining, 0.1))
        return True

    def _worker(self) -> None:
        try:
            self._send_batches()
        finally:
            self._done.set()

    def _send_batches(self) -> None:
        batch: list[TraceEvent] = []
        last_flush = time.time()

        while True:
            interval = self.config.flush_interval
            try:
                timeout = _stream_queue_wait(interval, time.time() - last_flush)
                item = self._queue.get(timeout=timeout)
                if item is None:
                    # Sentinel: flush remaining and exit
                    if batch:
                        self._flush(batch)
                    return
                batch.append(item)
            except queue.Empty:
                pass

            now = time.time()
            should_flush = (
                len(batch) >= self.config.batch_size
                or (batch and now - last_flush >= interval)
            )
            if should_flush:
                self._flush(batch)
                batch = []
                last_flush = now

    def _report(self, text: str | None, count: int) -> None:
        """Count *count* events as settled and write *text* to stderr.

        Both happen under the lock, so stop() never counts an event that a
        line already reported. Nothing is written once stop() gave up.
        """
        with self._lock:
            self._settled += count
            if text and not self._abandoned:
                sys.stderr.write(text)

    def _post_line(self, label: str, outcome: str) -> str:
        endpoint = _stream_endpoint(self.config.url)
        target = f"{label} to {endpoint}" if endpoint else label
        return f"[stream] {target} {outcome}\n"

    def _post(self, body: bytes, content_type: str, *, otlp: bool) -> int:
        """POST *body* without following redirects and return the status.

        Raises for a URL that cannot be used, for a network error and for
        every status outside 2xx, 3xx included.
        """
        url = _otlp_traces_url(self.config.url) if otlp else self.config.url
        headers = {"Content-Type": content_type}
        headers.update(self.config.headers)
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with _stream_opener().open(request, timeout=_STREAM_POST_TIMEOUT) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            exc.close()
            raise

    def _flush(self, events: list[TraceEvent]) -> None:
        """POST a batch of events as NDJSON or OTLP spans."""
        if not events:
            return
        if self.config.otlp:
            self._flush_otlp(events)
        else:
            self._flush_ndjson(events)

    def _flush_ndjson(self, events: list[TraceEvent]) -> None:
        line = None
        try:
            body = "\n".join(e.to_json() for e in events).encode("utf-8")
            status = self._post(body, "application/x-ndjson", otlp=False)
            if status not in (200, 202):
                line = self._post_line("POST", f"returned {status}")
        except Exception as exc:
            line = self._post_line("POST", f"failed: {_stream_error_detail(exc)}")
        self._report(line, len(events))

    def _flush_otlp(self, events: list[TraceEvent]) -> None:
        """Convert events to OTLP GenAI spans and POST them to the traces URL.

        Each session in the batch is converted and sent on its own. Events
        whose timestamp the converter would read and that is not a Unix time
        in seconds, and events whose non-object data it would read, are left
        out (see :func:`_otlp_readable_events`), and a session with no event left
        sends nothing. If the conversion still fails, one line on stderr
        gives the number of events not sent and the exception class, never
        its message, and later batches are still sent.
        """
        try:
            from .otlp import session_to_otlp_genai
            from .models import SessionMeta as _SessionMeta
        except ImportError:
            self._flush_ndjson(events)
            return

        # Group by session_id; build a minimal SessionMeta for each group
        by_session: dict[str, list[TraceEvent]] = {}
        for ev in events:
            by_session.setdefault(ev.session_id, []).append(ev)

        for sid, evs in by_session.items():
            meta = _SessionMeta(agent_name=self.config.service_name)
            meta.session_id = sid
            meta.tenant_id = next((ev.tenant_id for ev in evs if ev.tenant_id), "")
            try:
                kept, left_out_lines = _otlp_readable_events(evs)
            except Exception as exc:
                self._report(_otlp_conversion_failed_line(len(evs), exc), len(evs))
                continue
            self._report(left_out_lines, len(evs) - len(kept))
            if not kept:
                continue
            try:
                # A batch is not the whole session, so its root cannot say
                # how the session ended.
                payload = session_to_otlp_genai(
                    meta, kept, service_name=self.config.service_name,
                    root_outcome=False,
                )
            except Exception as exc:
                self._report(_otlp_conversion_failed_line(len(kept), exc), len(kept))
                continue
            line = None
            protocol = self.config.protocol
            try:
                body = _otlp_body(payload, protocol)
                status = self._post(body, _OTLP_CONTENT_TYPES[protocol], otlp=True)
                if status not in (200, 202):
                    line = self._post_line("OTLP POST", f"returned {status}")
            except _OtlpEncodeError as exc:
                line = self._post_line("OTLP POST", f"failed: {exc}")
            except Exception as exc:
                line = self._post_line("OTLP POST", f"failed: {_stream_error_detail(exc)}")
                if (isinstance(exc, urllib.error.HTTPError) and exc.code == 415
                        and not self._media_hint_shown):
                    self._media_hint_shown = True
                    line += f"[stream] {_unsupported_media_hint(protocol)}\n"
            self._report(line, len(kept))


def _otlp_readable_events(events: list[TraceEvent]) -> tuple[list[TraceEvent], str]:
    """Return the events of one session that the OTLP converter can read.

    ``session_to_otlp_genai`` reads the timestamps of the events of the
    ``OTLP_GENAI_TIMESTAMP_EVENT_TYPES`` types. Each of those whose timestamp
    is not a Unix time in seconds is left out first. The converter also reads
    the data of the events that ``otlp_genai_data_indexes`` names. Each of
    those whose data is not a JSON object is left out. Leaving an event out
    changes how the others pair, which can make the converter read another
    event, so the data check repeats on the events that are left. Events
    whose data the converter never reads are kept. The second value is the
    stderr text for the events left out: one line per event, in batch order.
    An event left out for its timestamp gets only the timestamp line.
    """
    from .otlp import OTLP_GENAI_TIMESTAMP_EVENT_TYPES, otlp_genai_data_indexes

    unusable_times = {
        position
        for position, event in enumerate(events)
        if event.event_type in OTLP_GENAI_TIMESTAMP_EVENT_TYPES
        and usable_time(event.timestamp) is None
    }
    positions = [
        position for position in range(len(events)) if position not in unusable_times
    ]
    while True:
        kept = [events[position] for position in positions]
        unreadable = {
            index
            for index in otlp_genai_data_indexes(kept)
            if not isinstance(kept[index].data, Mapping)
        }
        if not unreadable:
            break
        positions = [
            position for index, position in enumerate(positions) if index not in unreadable
        ]
    lines = "".join(
        left_out_timestamp_warning("stream", events[position])
        if position in unusable_times
        else left_out_event_warning("stream", events[position])
        for position in sorted(set(range(len(events))).difference(positions))
    )
    return kept, lines


def _otlp_conversion_failed_line(count: int, exc: Exception) -> str:
    """Return the stderr line for events the OTLP converter could not convert.

    It names the exception class but never its message, which can quote
    event data.
    """
    if count == 1:
        noun, result = "1 event", "it was"
    else:
        noun, result = f"{count} events", "they were"
    return (
        f"[stream] Could not convert {noun} to OTLP spans "
        f"({type(exc).__name__}); {result} not sent.\n"
    )


PAUSE_EDITOR = "editor"      # .pause-request marker; released when it is removed
PAUSE_RULE = "rule"          # rule action "pause"; released by a human
PAUSE_APPROVAL = "approval"  # rule action "require_approval"; released by the decision


@dataclass
class WatchState:
    """Mutable state accumulated across events."""
    # Retry tracking: command → count
    command_counts: Counter = field(default_factory=Counter)
    # Cost accumulation
    estimated_cost: float = 0.0
    token_counter: TokenCounter = field(default_factory=TokenCounter)
    # Session start time
    start_time: float = field(default_factory=time.time)
    # Recent event sequence for loop detection (circular buffer)
    recent_events: deque = field(default_factory=lambda: deque(maxlen=500))
    # Violations already fired (to avoid duplicate alerts)
    fired: set = field(default_factory=set)
    # Agent PID from --agent-pid. Without it watch can alert but not signal.
    agent_pid: int | None = None
    # Identity of agent_pid at startup (see process_identity.py); None if unknown
    agent_start_time: str | None = None
    # True once watch has said it cannot signal because no PID is known
    warned_no_agent_pid: bool = False
    # True once watch has sent the agent SIGTERM. Apart from the SIGCONT and
    # SIGKILL that _kill_process sends with it, watch then sends that PID no
    # other signal and creates no approval request, so the post-mortem and
    # --on-death happen once.
    agent_killed: bool = False
    # True once the agent PID is gone or names another process (see _agent_exited)
    agent_exited: bool = False
    # time.monotonic() of the last identity check in _agent_exited, or None
    agent_identity_checked_at: float | None = None
    # Token budget watcher (lazy-initialised on first LLM_REQUEST)
    token_budget_watcher: object | None = None
    # Nanny rule metrics
    files_modified: int = 0          # distinct files written/edited
    files_modified_set: set = field(default_factory=set)
    consecutive_test_failures: int = 0
    duration_minutes: float = 0.0
    # Who holds the current SIGSTOP: None, PAUSE_EDITOR, PAUSE_RULE or
    # PAUSE_APPROVAL. Only the owner's release path may resume the agent.
    pause_owner: str | None = None
    approval_request_id: str | None = None
    # True once an editor pause request could not SIGSTOP the agent. Watch
    # does not retry until the marker is removed, so it says so once.
    editor_pause_failed: bool = False
    session_id: str = ""
    # Working directory of the latest SESSION_START that names an absolute
    # one. Relative paths in tool calls resolve against it.
    session_cwd: str = ""
    mcp_scan_events: list[TraceEvent] = field(default_factory=list)

    @property
    def paused(self) -> bool:
        """True when watch has SIGSTOP'd the agent and not yet released it."""
        return self.pause_owner is not None

    def nanny_metrics(self) -> dict:
        """Return current metric snapshot for nanny rule evaluation."""
        elapsed = time.time() - self.start_time
        return {
            "files_modified": self.files_modified,
            "cost_usd": self.estimated_cost,
            "consecutive_test_failures": self.consecutive_test_failures,
            "duration_minutes": elapsed / 60.0,
        }


# FILE_WRITE and FILE_READ events name their file under these keys.
_FILE_EVENT_PATH_KEYS = ("path", "file_path", "uri")
# Editor tools that only read the file when their command is "view".
_VIEW_COMMAND_TOOLS = frozenset({"str_replace_editor", "str_replace_based_edit_tool"})


def _event_paths(event: TraceEvent) -> list[str]:
    """Return the file paths an event names, as stored, in order, once each.

    A tool call names each PATH_KEYS argument and, for apply_patch with the
    patch stored as text, each file the patch adds, updates, deletes or moves
    to. A FILE_WRITE or FILE_READ event names its path, file_path or uri.
    Other events name none.
    """
    if not isinstance(event.data, Mapping):
        return []
    if event.event_type in (EventType.FILE_WRITE, EventType.FILE_READ):
        values = [event.data.get(key) for key in _FILE_EVENT_PATH_KEYS]
    elif event.event_type == EventType.TOOL_CALL:
        args, raw = tool_arguments(event)
        values = [args.get(key) for key in PATH_KEYS]
        if raw is not None and str_field(event.data, "tool_name", "").lower() == "apply_patch":
            values.extend(patch_paths(raw))
    else:
        return []
    return list(dict.fromkeys(v for v in values if isinstance(v, str) and v))


def _is_edit(event: TraceEvent) -> bool:
    """Return True if *event* writes, edits, creates or deletes the files it names.

    That is a FILE_WRITE event, or a call to an EDIT_TOOLS tool or
    str_replace. An editor call whose command is "view" only reads. Shell
    commands are never edits, even when they write files.
    """
    if event.event_type == EventType.FILE_WRITE:
        return True
    if event.event_type != EventType.TOOL_CALL or not isinstance(event.data, Mapping):
        return False
    name = str_field(event.data, "tool_name", "").lower()
    if name in _VIEW_COMMAND_TOOLS:
        return tool_arguments(event)[0].get("command") != "view"
    return name in EDIT_TOOLS or name == "str_replace"


def _event_key(event: TraceEvent) -> str:
    """Stable string key for an event (for loop detection).

    A tool call's key holds a hash of its arguments as stored, under
    ``arguments`` or else ``tool_input``, so calls with different arguments
    get different keys. Arguments stored as text, such as an apply_patch
    patch, are hashed as that text. A call with neither key hashes ``{}``.
    """
    if event.event_type == EventType.TOOL_CALL:
        data = event.data if isinstance(event.data, Mapping) else {}
        name = str_field(data, "tool_name", "?")
        if "arguments" in data:
            stored = data["arguments"]
        else:
            stored = data.get("tool_input", {})
        if isinstance(stored, str):
            arg_text = stored
        else:
            arg_text = json.dumps(stored, sort_keys=True, default=str)
        # JSON text can hold a lone surrogate, which strict UTF-8 refuses.
        arg_hash = hashlib.sha256(arg_text.encode("utf-8", "surrogatepass")).hexdigest()[:12]
        args, raw = tool_arguments(event)
        preview = str(
            args.get("command")
            or args.get("file_path")
            or args.get("path")
            or args.get("url")
            or ""
        )
        if not preview and raw is not None and name.lower() == "apply_patch":
            preview = next(iter(patch_paths(raw)), "")
        return f"{name}:{preview[:40]}:{arg_hash}"
    return event.event_type.value


def _detect_spin_loop(
    recent: deque,
    threshold: int,
    window: int,
) -> str | None:
    """Detect repeated identical tool calls within the last window events."""
    threshold = max(2, int(threshold))
    window = max(threshold, int(window))
    items = [item for item in list(recent)[-window:] if ":" in str(item)]
    if len(items) < threshold:
        return None

    counts = Counter(items)
    key, count = counts.most_common(1)[0]
    if count >= threshold:
        display = str(key).rsplit(":", 1)[0]
        return f"identical tool call repeated {count} times in last {window} events ({display})"
    return None


def _detect_loop(
    recent: deque,
    seq_len: int,
    max_repeats: int,
) -> str | None:
    """Return a description if a repeating sequence is detected, else None."""
    items = list(recent)
    if len(items) < seq_len * 2:
        return None

    # Check if the last seq_len items repeat max_repeats times
    tail = items[-seq_len:]
    count = 1
    pos = len(items) - seq_len * 2
    while pos >= 0:
        window = items[pos:pos + seq_len]
        if window == tail:
            count += 1
            pos -= seq_len
        else:
            break

    if count >= max_repeats:
        seq_str = "→".join(tail)
        return f"detected loop ({seq_str}) × {count}"
    return None


# ---------------------------------------------------------------------------
# Alert dispatcher
# ---------------------------------------------------------------------------

def _send_network_alerts(
    message: str,
    config: WatcherConfig,
    action: str,
    notify: str = "",
    session_id: str = "",
) -> None:
    """Send an alert to --webhook and, when their env vars are set, Slack and Teams.

    Called last, after the pause or kill: each channel can take seconds.
    A failed channel prints one line and the next one is still tried.
    """
    if config.webhook_url:
        _alert_webhook(message, config.webhook_url)

    # Route to Slack/Teams via notify.py when env vars or rule hint is set
    has_slack = bool(os.environ.get("AGENT_STRACE_SLACK_WEBHOOK"))
    has_teams = bool(os.environ.get("AGENT_STRACE_TEAMS_WEBHOOK"))
    if has_slack or has_teams:
        try:
            from .notify import notify_violation as _notify_violation
            _notify_violation(
                rule_name=notify or "watch-rule",
                tool_name="",
                session_id=session_id,
                action=action or "alert",
            )
        except Exception as exc:
            # notify reports its own delivery failures; this is anything else.
            sys.stderr.write(f"[watch] Slack/Teams notification failed: {type(exc).__name__}\n")
            sys.stderr.flush()
    elif notify and notify.startswith("slack:") and config.webhook_url:
        # Legacy: route slack: hint through the generic webhook URL
        _alert_webhook(f"[{notify}] {message}", config.webhook_url)


def _say_already_killed(state: WatchState, extra: str = "") -> None:
    """Say that watch skipped a signal because it already killed the agent."""
    who = f"Agent pid {state.agent_pid}" if state.agent_pid else "The agent"
    _alert_terminal(f"{who} was already killed; no signal sent{extra}")


def _kill_agent(
    message: str,
    config: WatcherConfig,
    state: WatchState,
    store: TraceStore | None,
    session_id: str,
) -> None:
    """SIGTERM the agent, then write the post-mortem and run --on-death.

    The post-mortem and --on-death follow only when SIGTERM was sent, and
    run once: after this, watch does not kill, pause or resume the agent
    again. The kill also ends any pause that watch holds: _kill_process
    sends SIGCONT right after SIGTERM. A pending approval request stays
    pending.
    """
    _alert_terminal(f"Killing agent process {state.agent_pid}")
    if not _kill_process(state.agent_pid, state.agent_start_time):
        return
    state.agent_killed = True
    # Keeping the pause would print "agent stays stopped" or "pause
    # released" when the request is decided later, and hold the idle stop
    # until the agent exits.
    state.pause_owner = None
    state.approval_request_id = None
    pm_path: Path | None = None
    if store and session_id:
        pm_path = _write_watchdog_postmortem(
            store, session_id, state, reason=message, config=config
        )
        if pm_path:
            _alert_terminal(f"Post-mortem written to {pm_path}")
    if config.on_death_cmd:
        _invoke_on_death(config.on_death_cmd, pm_path)


def _dispatch_alert(
    message: str,
    config: WatcherConfig,
    state: WatchState,
    action: str | None = None,
    notify: str = "",
    dry_run: bool = False,
    store: TraceStore | None = None,
    session_id: str = "",
) -> None:
    """Show, log and act on one alert, then send it over the network.

    Order: the terminal line, the alert log (file, kill and pause actions),
    the pause or kill (or the dry-run line), then the --webhook POST and
    Slack/Teams. A kill sends SIGTERM and, only if it was sent, writes the
    post-mortem (when *store* and *session_id* are given) and runs
    --on-death with its path. No channel failure raises, so none can stop
    or delay the pause or kill. A require_approval action sends nothing
    over the network.

    Once watch has sent SIGTERM it does not kill, pause or gate the agent
    again: a later kill, pause or require_approval alert says the agent was
    already killed instead, sends no signal and creates no approval request.
    """
    action = action or config.on_violation
    # terminal is always shown regardless of action so the operator watching
    # the process sees every alert; file/webhook are additive channels
    _alert_terminal(message)
    if action == "require_approval":
        rule_name = notify or "watch-rule"
        if dry_run:
            _alert_terminal(
                f"[dry-run] would pause agent and require approval for rule '{rule_name}'"
            )
            return
        if state.agent_killed:
            _say_already_killed(state, " and no approval request created")
            return
        if state.pause_owner == PAUSE_APPROVAL:
            _alert_terminal(
                f"[HITL] Approval {state.approval_request_id} is already pending; "
                "not creating another request"
            )
            return
        # Pause before creating the request so an approval cannot land
        # before the SIGSTOP and leave the agent stopped. If SIGSTOP was not
        # sent the request is still created, but watch holds no pause.
        pause_failed = False
        if state.agent_pid and not state.paused and not _agent_pid_reused(
            state, alert_log=config.alert_log,
        ):
            pause_failed = not _pause_agent(state, alert_log=config.alert_log)
        elif not state.agent_pid:
            _warn_no_agent_pid(state, f"pause the agent for approval rule '{rule_name}'")
        req = None
        sid = session_id or state.session_id
        if store and sid:
            try:
                from .approval import create_approval_request
                req = create_approval_request(
                    store,
                    session_id=sid,
                    rule_name=rule_name,
                    tool_name="",
                    agent_pid=state.agent_pid or 0,
                    agent_start_time=state.agent_start_time or "",
                )
                _alert_terminal(
                    f"[HITL] Approval required for rule '{rule_name}'. "
                    f"Request ID: {req.request_id}",
                    hint=f"Run: agent-strace approval approve {req.request_id}",
                )
            except Exception as exc:
                _alert_terminal(f"[HITL] Could not create approval request: {exc}")
        if state.agent_pid and not pause_failed:
            if req is not None:
                state.pause_owner = PAUSE_APPROVAL
                state.approval_request_id = req.request_id
            elif state.pause_owner is None:
                # No request to decide on, so leave it releasable by a human.
                state.pause_owner = PAUSE_RULE
        return

    if action in ("file", "kill", "pause"):
        _alert_file(message, config.alert_log)

    if dry_run:
        _alert_terminal(f"[dry-run] would {action} agent process")
        _send_network_alerts(message, config, action, notify=notify, session_id=session_id)
        return

    if action in ("pause", "kill") and state.agent_killed:
        _say_already_killed(state)
    elif action in ("pause", "kill"):
        if action == "kill" or not state.paused:
            _agent_pid_reused(state, alert_log=config.alert_log)
        if not state.agent_pid:
            _warn_no_agent_pid(state, f"{action} the agent")
        elif action == "kill":
            _kill_agent(message, config, state, store, session_id)
        elif not state.paused:
            _alert_terminal(f"Pausing agent process {state.agent_pid} (SIGSTOP)")
            if _pause_agent(state, alert_log=config.alert_log):
                state.pause_owner = PAUSE_RULE

    _send_network_alerts(message, config, action, notify=notify, session_id=session_id)


def _write_watchdog_postmortem(
    store: TraceStore,
    session_id: str,
    state: WatchState,
    reason: str,
    config: WatcherConfig | None = None,
) -> Path | None:
    """Write a structured JSON post-mortem to the session directory.

    Returns the path written, or None on failure.
    """
    try:
        events = store.load_events(session_id)
        meta = store.load_meta(session_id)
    except Exception:
        return None

    last_tool_call = None
    last_llm_response = None
    for ev in reversed(events):
        if last_tool_call is None and ev.event_type == EventType.TOOL_CALL:
            last_tool_call = ev.data
        if last_llm_response is None and ev.event_type == EventType.LLM_RESPONSE:
            last_llm_response = ev.data
        if last_tool_call and last_llm_response:
            break

    elapsed = time.time() - state.start_time
    pm = {
        "session_id": session_id,
        "terminated_at": time.time(),
        "reason": reason,
        "elapsed_seconds": round(elapsed, 2),
        "cost_at_death": round(state.estimated_cost, 6),
        "last_tool_call": last_tool_call,
        "last_llm_response": last_llm_response,
        "recovery_context": (
            f"Session {session_id} was terminated after {elapsed:.0f}s "
            f"(${state.estimated_cost:.4f} spent). "
            f"Reason: {reason}. "
            "Resume from the last tool call above."
        ),
    }
    if config is not None:
        pm["max_cost_dollars"] = config.max_cost_dollars
        pm["budget_at_death"] = config.max_cost_dollars

    pm_path = store._session_dir(session_id) / "watchdog-postmortem.json"
    try:
        pm_path.write_text(json.dumps(pm, indent=2), newline="\n")
        return pm_path
    except Exception:
        return None


def _invoke_on_death(on_death_cmd: str, pm_path: Path | None) -> None:
    """Run the --on-death command with {post_mortem_path} substituted."""
    if not on_death_cmd:
        return
    path_str = str(pm_path) if pm_path else ""
    cmd = on_death_cmd.replace("{post_mortem_path}", path_str)
    try:
        subprocess.Popen(cmd, shell=True)
    except Exception as exc:
        sys.stderr.write(f"[watch] on-death command failed: {exc}\n")


def _dispatch_nanny_rule(
    rule: NannyRule,
    event: TraceEvent,
    store: TraceStore,
    session_id: str,
    state: WatchState,
    config: WatcherConfig,
    dry_run: bool = False,
) -> None:
    """Fire a nanny rule: alert, pause, require approval, or kill.

    A kill takes the same path as a watcher kill, so the post-mortem and
    --on-death come from _dispatch_alert, after SIGTERM is sent.
    """
    msg = f"NannyRule '{rule.name}': {rule.condition} → {rule.action}"
    _dispatch_alert(
        msg, config, state, action=rule.action, notify=rule.notify,
        dry_run=dry_run, store=store, session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Per-event check
# ---------------------------------------------------------------------------

def _check_duration(config: WatcherConfig, state: WatchState) -> str | None:
    """Return the DurationWatcher violation once the time limit has passed.

    Returns None before the limit and after the violation has fired once.
    """
    elapsed = time.time() - state.start_time
    if elapsed > config.max_duration_seconds and "duration" not in state.fired:
        state.fired.add("duration")
        return (
            f"DurationWatcher: {elapsed:.0f}s elapsed "
            f"(threshold: {config.max_duration_seconds:.0f}s)"
        )
    return None


def check_event(
    event: TraceEvent,
    config: WatcherConfig,
    state: WatchState,
) -> list[str]:
    """Update state and return list of violation messages (may be empty)."""
    violations: list[str] = []

    # --- Cost accumulation ---
    state.estimated_cost += _usage_dollars(state.token_counter.usage(event), "sonnet")

    # --- Track the session's working directory ---
    if event.event_type == EventType.SESSION_START and isinstance(event.data, Mapping):
        _cwd = absolute_cwd(event.data.get("cwd"))
        if _cwd:
            state.session_cwd = _cwd

    # --- Track files modified (for nanny rules) ---
    if _is_edit(event):
        for _path in _event_paths(event):
            state.files_modified_set.add(resolve_path(_path, state.session_cwd))
        state.files_modified = len(state.files_modified_set)

    # --- Track consecutive test failures (for nanny rules) ---
    if event.event_type == EventType.TOOL_RESULT:
        _content = str(event.data.get("content", ""))
        _is_test_fail = any(
            kw in _content.lower()
            for kw in ("failed", "error", "assertion", "traceback", "exit code 1")
        )
        if _is_test_fail:
            state.consecutive_test_failures += 1
        elif not event.data.get("is_error", False):
            # Non-error result resets the counter (is_error absent = success)
            state.consecutive_test_failures = 0

    # --- Loop detection ---
    key = _event_key(event)
    state.recent_events.append(key)

    # --- Retry detection (bash commands) ---
    if event.event_type == EventType.TOOL_CALL:
        name = str_field(event.data, "tool_name", "").lower()
        args = event_arguments(event)
        if name == "bash":
            cmd = str(args.get("command", "")).strip()
            if cmd:
                state.command_counts[cmd] += 1
                count = state.command_counts[cmd]
                if count > config.max_retries:
                    key_id = f"retry:{cmd}"
                    if key_id not in state.fired:
                        state.fired.add(key_id)
                        violations.append(
                            f"RetryWatcher: command ran {count} times: {cmd[:60]}"
                        )

    # --- Cost threshold ---
    if state.estimated_cost > config.max_cost_dollars:
        key_id = "cost"
        if key_id not in state.fired:
            state.fired.add(key_id)
            violations.append(
                f"CostWatcher: ${state.estimated_cost:.2f} (threshold: ${config.max_cost_dollars})"
            )

    # --- Live cognitive debt threshold ---
    if "cognitive-debt" in config.built_in_rules and state.files_modified > 0:
        threshold = config.cognitive_debt_threshold
        if threshold is None:
            threshold = 0.8
        live_score = 1.0
        if live_score > threshold:
            key_id = "cognitive-debt"
            if key_id not in state.fired:
                state.fired.add(key_id)
                violations.append(
                    f"CognitiveDebtWatcher: live debt score {live_score:.2f} "
                    f"({state.files_modified} modified files, no review yet; threshold: {threshold:g})"
                )

    # --- Duration threshold ---
    duration_msg = _check_duration(config, state)
    if duration_msg:
        violations.append(duration_msg)

    # --- Spin loop detection: repeated identical tool calls in a window ---
    spin_msg = _detect_spin_loop(
        state.recent_events,
        config.loop_threshold,
        config.loop_window,
    )
    if spin_msg:
        key_id = f"spin-loop:{spin_msg.rsplit('(', 1)[-1][:80]}"
        if key_id not in state.fired:
            state.fired.add(key_id)
            violations.append(f"LoopWatcher: {spin_msg}")

    # --- Sequence loop detection: repeated event sequence ---
    loop_msg = _detect_loop(
        state.recent_events,
        config.loop_sequence_length,
        config.loop_max_repeats,
    )
    if loop_msg:
        # Key on the sequence pattern only (strip the " × N" count suffix)
        # so dedup fires once per unique loop, not once per repeat increment.
        seq_part = loop_msg.split(" × ")[0]
        key_id = f"loop:{seq_part[:60]}"
        if key_id not in state.fired:
            state.fired.add(key_id)
            violations.append(f"LoopWatcher: {loop_msg}")

    # --- Scope check (file edits) ---
    if _is_edit(event):
        scope_path = Path(config.scope_policy)
        if scope_path.exists():
            try:
                from .audit import Policy, _glob_match
                policy = Policy.load(scope_path)
                if policy and policy.file_write_deny:
                    for path in _event_paths(event):
                        resolved = resolve_path(path, state.session_cwd)
                        if not (
                            _glob_match(resolved, policy.file_write_deny)
                            or _glob_match(path, policy.file_write_deny)
                        ):
                            continue
                        key_id = f"scope:{resolved}"
                        if key_id not in state.fired:
                            state.fired.add(key_id)
                            violations.append(f"ScopeWatcher: write to {resolved} denied by policy")
            except Exception:
                pass

    # --- Token budget ---
    if event.event_type == EventType.LLM_REQUEST:
        if state.token_budget_watcher is None:
            try:
                from .token_budget import TokenBudgetWatcher
                threshold = (
                    getattr(config, "max_context_pct", 90) / 100.0
                    if getattr(config, "max_context_pct", None) else 0.9
                )
                state.token_budget_watcher = TokenBudgetWatcher(threshold=threshold)
            except ImportError:
                state.token_budget_watcher = False  # module not yet available
        if state.token_budget_watcher:
            msg = state.token_budget_watcher.update(event)
            if msg:
                violations.append(msg)

    # --- Per-operation enforcement rules ---
    if event.event_type == EventType.TOOL_CALL and config.operation_rules:
        tool_name = str_field(event.data, "tool_name", "").lower()
        args = event_arguments(event)
        # Derive the target string: command for bash, path for file ops
        if tool_name == "bash":
            target = str(args.get("command", "")).strip()
        else:
            target = str(
                args.get("file_path") or args.get("path") or args.get("pattern") or ""
            ).strip()

        for rule in config.operation_rules:
            if rule.matches(tool_name, target):
                reason_suffix = f": {rule.reason}" if rule.reason else ""
                msg = (
                    f"OperationWatcher: {rule.action} {tool_name}"
                    f" '{target[:60]}'{reason_suffix}"
                )
                key_id = f"op:{rule.tool_name}:{rule.pattern}:{target[:40]}"
                if key_id not in state.fired:
                    state.fired.add(key_id)
                    violations.append(msg)

    # --- Built-in MCP poisoning detection ---
    if "mcp-poisoning" in config.built_in_rules:
        state.mcp_scan_events.append(event)
        state.mcp_scan_events = state.mcp_scan_events[-80:]
        try:
            from .mcp_scan import scan_live_event
            findings = scan_live_event(
                state.mcp_scan_events,
                event,
                event.session_id,
                project_root=".",
            )
        except Exception:
            findings = []
        for finding in findings:
            detail_key = json.dumps(finding.details, sort_keys=True, default=str)
            key_id = f"mcp:{finding.kind}:{finding.tool_name}:{detail_key}"
            if key_id in state.fired:
                continue
            state.fired.add(key_id)
            violations.append(
                f"McpPoisoningWatcher: {finding.severity.upper()} {finding.kind}: "
                f"{finding.message}"
            )

    return violations


# check_event reads the data of these event types as soon as they arrive.
_LIVE_DATA_EVENT_TYPES = frozenset({
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.FILE_WRITE,
})
# CompactionCheckpointWatcher.update reads the data of these event types.
_CHECKPOINT_DATA_EVENT_TYPES = frozenset({
    EventType.LLM_REQUEST,
    EventType.LLM_RESPONSE,
    EventType.ASSISTANT_RESPONSE,
})


def _live_checks_read_data(
    event: TraceEvent,
    state: WatchState,
    checkpoint_mode: bool = False,
    path_rules: bool = False,
) -> bool:
    """Return True if checking *event* now would read its data as an object.

    path_rules: a ``file_path matches`` rule that has not fired is active, so
    the paths of FILE_READ events are read too.
    """
    if event.event_type in _LIVE_DATA_EVENT_TYPES:
        return True
    if path_rules and event.event_type == EventType.FILE_READ:
        return True
    if checkpoint_mode and event.event_type in _CHECKPOINT_DATA_EVENT_TYPES:
        return True
    if event.event_type == EventType.LLM_REQUEST:
        watcher = state.token_budget_watcher
        if watcher is None:
            # check_event creates the watcher, which reads the model name.
            return True
        if watcher:
            # The watcher skips the data only once it knows the model and
            # that model has no known context limit.
            return (
                not getattr(watcher, "model", "")
                or getattr(watcher, "context_limit", None) is not None
            )
    return False


# ---------------------------------------------------------------------------
# File tailer
# ---------------------------------------------------------------------------

_IDLE_SENTINEL = object()  # yielded when poll_interval elapses with no new event


class _StartupCwd:
    """First item _tail_events yields when the file names a working directory.

    cwd is the absolute cwd of the latest SESSION_START written before watch
    started.
    """

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd


def _startup_session_cwd(events_file: Path, end: int) -> str:
    """Return the cwd of the latest SESSION_START in the first *end* bytes.

    Only lines that hold ``"session_start"`` are parsed, so a large file costs
    one read. A line that runs past *end* is not read; the tail starts there.
    Returns "" when no SESSION_START names an absolute cwd.
    """
    cwd = ""
    read = 0
    try:
        with open(events_file, "rb") as f:
            for line in f:
                read += len(line)
                if read > end:
                    break
                if b'"session_start"' not in line:
                    continue
                try:
                    event = TraceEvent.from_json(line.decode("utf-8"))
                except Exception:
                    continue
                if event.event_type == EventType.SESSION_START and isinstance(event.data, Mapping):
                    cwd = absolute_cwd(event.data.get("cwd")) or cwd
    except OSError:
        return ""
    return cwd


def _editor_wants_pause(store: "TraceStore", session_id: str) -> bool | None:
    """Read the .pause-request marker written by the VS Code extension.

    Returns True when the marker asks to pause *session_id*, False when it is
    absent or names another session, and None when it cannot be read.
    An empty marker (for example from ``touch``) applies to any session.
    """
    try:
        content = (store.base_dir / ".pause-request").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return not content or not session_id or content == session_id


def _check_approval_decision(store: "TraceStore", state: "WatchState", out: "TextIO") -> None:
    """Release an approval-owned pause once the request has been decided.

    ``approval approve`` sends SIGCONT from another process, so watch only
    clears its own pause state. ``approval deny`` sends SIGTERM (unless
    ``--no-kill``), but an agent that handles SIGTERM stays stopped until it
    gets SIGCONT. Watch cannot tell which deny was used, so a denied pause
    stays in place as a rule pause that a human can release.
    """
    if state.pause_owner != PAUSE_APPROVAL or not state.approval_request_id:
        return
    try:
        from .approval import load_request
        req = load_request(store, state.approval_request_id)
    except Exception:
        return
    # A missing or half-written file reads as None; keep waiting.
    if req is None or req.state == "pending":
        return
    state.approval_request_id = None
    # The request file is in the trace directory, which the agent can write.
    request_id = inert(req.request_id)
    if req.state == "denied":
        out.write(
            f"[watch] Approval {request_id} denied; agent stays stopped. "
            f"Run 'kill -CONT {state.agent_pid}' or Pause then Resume in the editor to release it\n"
        )
        state.pause_owner = PAUSE_RULE
    else:
        out.write(f"[watch] Approval {request_id} {req.state}; pause released\n")
        state.pause_owner = None
    out.flush()


def _check_agent_alive(state: "WatchState", out: "TextIO") -> None:
    """Drop pause state once the paused agent process has exited."""
    if not state.paused or not state.agent_pid:
        return
    try:
        os.kill(state.agent_pid, 0)
    except ProcessLookupError:
        out.write(f"[watch] Agent pid {state.agent_pid} exited; pause cleared\n")
        out.flush()
        state.pause_owner = None
        state.approval_request_id = None
        return
    except OSError:
        return
    _agent_pid_reused(state, out)


def _report_pause_left_held(
    store: "TraceStore", state: "WatchState", out: "TextIO", alert_log: str | None,
) -> None:
    """Say how to release the pause watch holds as it exits.

    Watch never resumes the agent on its own: a pause is a safety stop that
    only a human releases. VS Code Resume works only while watch runs, so on
    every exit with a pause still held, watch prints and logs the PID, who
    paused it and how to resume it. Nothing is said when watch killed the
    agent, the agent has exited, or its PID now names another process.
    """
    if not state.paused or state.agent_killed or not state.agent_pid:
        return
    _check_approval_decision(store, state, out)
    if not state.paused:
        return
    pid = state.agent_pid
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        # PermissionError: the PID now belongs to another user's process.
        return
    except OSError:
        pass
    if _agent_pid_reused(state, out, alert_log, signalling=False):
        return
    if state.pause_owner == PAUSE_APPROVAL:
        request_id = state.approval_request_id
        msg = (
            f"Watch stopped while agent pid {pid} is paused for approval "
            f"{request_id}. It stays stopped until the request is decided: "
            f"run 'agent-strace approval approve {request_id}' to resume it."
        )
    else:
        who = "the editor" if state.pause_owner == PAUSE_EDITOR else "a watch rule"
        msg = (
            f"Watch stopped while agent pid {pid} is paused by {who}. It stays "
            f"stopped, and VS Code Resume needs a running watch: run "
            f"'kill -CONT {pid}' to resume it."
        )
    _report(msg, out, alert_log)


def _check_pause_file(store: "TraceStore", state: "WatchState", out: "TextIO") -> None:
    """Apply approval decisions and the editor's .pause-request marker.

    Marker present → SIGSTOP the agent and record the editor as pause owner.
    Marker removed → SIGCONT, but only if the editor owns the pause. A rule
    pause is handed to the editor when the marker appears, so pausing and
    then resuming from the editor releases it. An approval pause is never
    released by the marker; the approval decision ends it, and so does a
    kill. Once watch has killed the agent the marker is ignored: watch sends
    no SIGSTOP or SIGCONT for it.
    If SIGSTOP cannot be sent, watch says so once, holds no pause, and
    tries again only after the marker is removed and written again.
    """
    _check_approval_decision(store, state, out)
    _check_agent_alive(state, out)
    if state.agent_killed:
        return
    wants_pause = _editor_wants_pause(store, state.session_id)
    if wants_pause is None:
        return
    if not state.agent_pid:
        if wants_pause:
            _warn_no_agent_pid(state, "pause the agent for the editor", out)
        return

    if not wants_pause:
        state.editor_pause_failed = False
    elif state.editor_pause_failed and state.pause_owner is None:
        # SIGSTOP failed for this marker; wait for it to be removed.
        return

    signals = (state.pause_owner is None) if wants_pause else (state.pause_owner == PAUSE_EDITOR)
    if signals and _agent_pid_reused(state, out):
        return

    if wants_pause:
        if state.pause_owner is None:
            out.write(f"[watch] Pause requested by editor - SIGSTOP pid {state.agent_pid}\n")
            out.flush()
            if _pause_agent(state, out):
                state.pause_owner = PAUSE_EDITOR
            else:
                state.editor_pause_failed = True
                out.write("[watch] Resume in the editor, then Pause, to try again\n")
                out.flush()
        elif state.pause_owner == PAUSE_RULE:
            out.write("[watch] Editor took over the rule pause; resume from the editor to continue\n")
            out.flush()
            state.pause_owner = PAUSE_EDITOR
    elif state.pause_owner == PAUSE_EDITOR:
        out.write(f"[watch] Resume requested by editor - SIGCONT pid {state.agent_pid}\n")
        out.flush()
        _resume_agent(state, out)


# ---------------------------------------------------------------------------
# Watchdog limits (--timeout / --budget)
# ---------------------------------------------------------------------------

# Seconds watch keeps running past the deadline of a duration limit that
# kills (see _duration_limit_kills) while the agent lives, so the SIGKILL
# that _kill_process sends 5 seconds after SIGTERM is not lost when watch
# would otherwise stop for lack of events.
_TIMEOUT_KILL_GRACE = 10.0

# Seconds between the identity checks of a live agent PID in _agent_exited.
# On macOS each check runs ps.
_IDENTITY_RECHECK_SECONDS = 5.0


def _agent_exited(
    state: WatchState, out: TextIO | None = None, alert_log: str | None = None,
    *, recheck_identity: bool = False,
) -> bool:
    """Return True once the agent process is known to have exited.

    Signal 0 tests for the process without affecting it. A zombie still
    answers it, so an agent that exited but was not yet reaped counts as
    running. When signal 0 finds a process, its start time is compared with
    the one recorded at startup, at most every _IDENTITY_RECHECK_SECONDS
    unless *recheck_identity* is True: a PID the OS gave to another process
    counts as exited, and watch says so once (see _agent_pid_reused) and
    drops the PID. Once the agent has exited the answer is kept.
    """
    if state.agent_exited:
        return True
    if not state.agent_pid:
        return False
    try:
        os.kill(state.agent_pid, 0)
    except (ProcessLookupError, PermissionError):
        # PermissionError: the PID now belongs to another user's process.
        state.agent_exited = True
        return True
    except OSError:
        return False
    now = time.monotonic()
    last = state.agent_identity_checked_at
    if not recheck_identity and last is not None and now - last < _IDENTITY_RECHECK_SECONDS:
        return False
    state.agent_identity_checked_at = now
    return _agent_pid_reused(state, out, alert_log, signalling=False)


def _check_timeout(
    config: WatcherConfig, state: WatchState, out: TextIO | None = None,
) -> str | None:
    """Return the DurationWatcher violation once a deadline that kills passes.

    The deadline is that of a duration limit that kills: --timeout, or
    --max-duration, watchers.duration.max_minutes or a built-in timeout:D
    while the action is kill (see _duration_limit_kills). Called on every
    poll cycle, so an agent that writes no events is still killed at the
    deadline. Returns None after the violation has fired, once the agent has
    exited, and for any other duration limit, which is checked only when an
    event arrives (see check_event).
    """
    if not _duration_limit_kills(config) or "duration" in state.fired:
        return None
    if time.time() - state.start_time <= config.max_duration_seconds:
        return None
    # Check the identity now, so a reused PID gets no stale alert.
    if _agent_exited(state, out, config.alert_log, recheck_identity=True):
        return None
    return _check_duration(config, state)


def _timeout_holds_watch(
    config: WatcherConfig, state: WatchState, now: float, out: TextIO | None = None,
) -> bool:
    """Return True while the idle stop must wait for a deadline that kills.

    With a duration limit that kills (see _duration_limit_kills) and
    --agent-pid, watch keeps running while the agent lives, until
    _TIMEOUT_KILL_GRACE seconds after the deadline. Without an agent PID
    watch cannot tell whether the agent runs, so it does not wait.
    """
    if not _duration_limit_kills(config) or not state.agent_pid:
        return False
    deadline = state.start_time + config.max_duration_seconds
    if now > deadline + _TIMEOUT_KILL_GRACE:
        return False
    return not _agent_exited(state, out, config.alert_log)


def _limit_action(
    message: str,
    config: WatcherConfig,
    state: WatchState,
    project_budget_config: ProjectBudgetConfig | None,
) -> str | None:
    """Return "kill" when the violation *message* must kill the agent.

    --timeout and --budget kill whatever --on-violation says, and so does a
    project budget's per_session_max. A timeout does not kill an agent that
    has already exited. None means the --on-violation action applies.
    """
    if message.startswith("DurationWatcher:"):
        if config.kill_on_timeout and not state.agent_exited:
            return "kill"
    elif message.startswith("CostWatcher:"):
        if config.kill_on_budget or (
            project_budget_config and project_budget_config.per_session_max
        ):
            return "kill"
    return None


def _cost_limit_kills(config: WatcherConfig, per_session_max: float | None = None) -> bool:
    """Return True when the cost limit is one the user set that kills.

    That is a limit from --budget or per_session_max, or from --max-cost or
    watchers.cost.max_dollars while the action is kill. The default $10 is
    not set by the user, so it does not count, even with the kill action.
    """
    return bool(
        config.kill_on_budget
        or per_session_max
        or (config.cost_limit_set and config.on_violation == "kill")
    )


def _duration_limit_kills(config: WatcherConfig) -> bool:
    """Return True when the duration limit is one the user set that kills.

    That is a limit from --timeout, or from --max-duration,
    watchers.duration.max_minutes or a built-in timeout:D while the action
    is kill. The default 1800 seconds is not set by the user, so it does not
    count, even with the kill action.
    """
    return config.kill_on_timeout or (
        config.duration_limit_set and config.on_violation == "kill"
    )


def _file_identity(stat: os.stat_result) -> tuple[int, int]:
    return stat.st_dev, stat.st_ino


class _EventsTail:
    """Reads the complete lines appended to an events file since the last read.

    The file is opened for each read and closed before the lines are
    returned, so a tag_session rewrite can replace it on Windows too.

    tag_session replaces the file with one that holds the same lines in the
    same order, each rewritten with the tenant, so byte offsets move. The
    tail therefore counts the non-blank lines it has read. When the path
    names another file (device or inode changed), or the bytes just before
    the read offset are no longer the last line read, it skips that many
    non-blank lines from the start of the file and reads on from there.
    When the file holds fewer lines than were read, as after a truncation,
    it starts after the last complete line and replays nothing, so no rule
    counts an event twice.

    Only newline-terminated lines are read. A line still being written stays
    in the file until its newline arrives.
    """

    def __init__(self, path: Path, identity: tuple[int, int]) -> None:
        self.path = path
        self.identity = identity
        self.offset = 0
        self.lines = 0
        self.last_line = b""

    def skip_to(self, f, end: int | None = None, lines: int | None = None) -> None:
        """Move past complete lines from the start of *f*.

        Stops at *end* bytes or after *lines* non-blank lines, whichever is
        given, and never inside a line without its newline.
        """
        f.seek(0)
        offset = count = 0
        last = b""
        for line in f:
            if lines is not None and count >= lines:
                break
            if not line.endswith(b"\n") or (end is not None and offset + len(line) > end):
                break
            offset += len(line)
            last = line
            if line.strip():
                count += 1
        self.offset, self.lines, self.last_line = offset, count, last

    def _unchanged(self, f) -> bool:
        if not self.last_line:
            return True
        start = self.offset - len(self.last_line)
        if start < 0:
            return False
        f.seek(start)
        return f.read(len(self.last_line)) == self.last_line

    def read(self) -> list[bytes]:
        """Return the non-blank complete lines added since the last read."""
        try:
            f = open(self.path, "rb")
        except OSError:
            # The path may be missing for a moment while it is replaced.
            return []
        with f:
            identity = _file_identity(os.fstat(f.fileno()))
            if identity != self.identity or not self._unchanged(f):
                self.identity = identity
                self.skip_to(f, lines=self.lines)
            f.seek(self.offset)
            data = f.read()
        complete = data[: data.rfind(b"\n") + 1]
        if not complete:
            return []
        self.offset += len(complete)
        found = []
        for line in complete.split(b"\n")[:-1]:
            self.last_line = line + b"\n"
            if line.strip():
                self.lines += 1
                found.append(line)
        return found


def _tail_events(events_file: Path, poll_interval: float = 0.5):
    """Generator that yields TraceEvent objects or _IDLE_SENTINEL each poll cycle.

    Yields _IDLE_SENTINEL when no new line arrived during the poll interval,
    allowing callers to implement idle-timeout logic without blocking.

    Events already in the file are not yielded. When one of them is a
    SESSION_START with an absolute cwd, the first item is a _StartupCwd with
    the latest such cwd. It is read up to the offset the tail starts from, so
    no event falls between the two. A last line without its newline yet is
    not read at startup; the tail yields it once it is complete.

    The tail keeps reading when the file is replaced, as tag_session does;
    see _EventsTail.
    """
    with open(events_file, "rb") as f:
        end = f.seek(0, 2)
        tail = _EventsTail(events_file, _file_identity(os.fstat(f.fileno())))
        cwd = _startup_session_cwd(events_file, end)
        tail.skip_to(f, end=end)
    if cwd:
        yield _StartupCwd(cwd)
    while True:
        lines = tail.read()
        if not lines:
            time.sleep(poll_interval)
            yield _IDLE_SENTINEL
            continue
        for line in lines:
            try:
                yield TraceEvent.from_json(line.decode("utf-8").strip())
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public watch loop
# ---------------------------------------------------------------------------

def watch_session(
    store: TraceStore,
    session_id: str,
    config: WatcherConfig,
    out: TextIO = sys.stderr,
    poll_interval: float = 0.5,
    max_idle_seconds: float = 300.0,
    nanny_rules: list[NannyRule] | None = None,
    dry_run: bool = False,
    stream_config: StreamConfig | None = None,
    project_budget_config: ProjectBudgetConfig | None = None,
    checkpoint_watcher: CompactionCheckpointWatcher | None = None,
    agent_pid: int | None = None,
) -> None:
    """Watch a session's event stream and fire alerts on violations.

    agent_pid: process that pause, kill and approval actions signal. Without
    it watch still alerts but warns once instead of signalling.

    A duration limit that kills (see _duration_limit_kills), such as
    --timeout, is checked on every poll cycle and kills the agent, so an
    agent that writes no events is still killed. With agent_pid, the idle
    stop waits while the agent process runs, until _TIMEOUT_KILL_GRACE
    seconds after the deadline.

    Watch never resumes the agent as it exits. If it still holds a pause,
    it says how to release it (see _report_pause_left_held).

    After a kill, watch does not return before the SIGKILL that follows
    SIGTERM was sent, or nothing it signalled is left; see
    _wait_for_sigkill.

    stream_config: when set, events are pushed in real-time to the configured
    URL as they arrive (push-based streaming). The startup line names only
    the URL's scheme://host[:port]. When watching ends, EventStreamer.stop()
    waits for the last POST; Ctrl-C during that wait ends it without a
    traceback.

    An event whose data is not a JSON object is still counted and passed to
    the streamer. When a check would read that data, watch writes one warning
    to *out* and skips the checks for that event. Compaction checkpoints
    leave such events out. NDJSON streaming sends them with their data as
    stored; OTLP streaming leaves them out when its converter would read the
    data and writes one line to stderr for each (see EventStreamer).
    """
    events_file = store._session_file(session_id, "events.ndjson", "event stream")
    if not events_file.exists():
        out.write(f"[watch] events file not found: {inert(events_file)}\n")
        return

    # Temporal's OTel interceptor can pass the activity's W3C traceparent to
    # the agent process.  Persist it before tailing so later/offline exports
    # remain children of the activity span even after the environment is gone.
    try:
        from .temporal import attach_temporal_trace_context

        temporal_context = attach_temporal_trace_context(store, session_id)
    except ValueError as exc:
        out.write(f"[watch] Invalid Temporal trace context: {inert(exc)}\n")
        temporal_context = None
    if temporal_context is not None:
        out.write(
            "[watch] Attached Temporal parent span "
            f"{temporal_context.parent_span_id}\n"
        )

    state = WatchState(
        start_time=time.time(), session_id=session_id, agent_pid=agent_pid,
        agent_start_time=process_identity(agent_pid) if agent_pid else None,
    )

    mode = " [dry-run]" if dry_run else ""
    out.write(f"[watch] Monitoring session {inert(session_id[:12])}...{mode}\n")
    if agent_pid:
        out.write(f"[watch] Agent pid {agent_pid}: pause, kill and approval actions will signal it\n")
    if nanny_rules:
        out.write(f"[watch] {len(nanny_rules)} nanny rule(s) active\n")
    if stream_config and stream_config.url:
        endpoint = _stream_endpoint(stream_config.url)
        out.write(f"[watch] Streaming events to {endpoint}\n" if endpoint
                  else "[watch] Streaming events\n")
    if checkpoint_watcher:
        try:
            checkpoint_path = checkpoint_watcher.checkpoint_current()
            if checkpoint_path:
                out.write(f"[watch] Compaction checkpoint written to {inert(checkpoint_path)}\n")
        except OSError as exc:
            out.write(f"[watch] Could not write compaction checkpoint: {exc}\n")
    out.flush()

    project_budget_base_spend = 0.0
    if project_budget_config and project_budget_config.enabled:
        budget_status = project_budget_status(
            store,
            project_budget_config,
            exclude_session_id=session_id,
        )
        project_budget_base_spend = budget_status.spent
        if budget_status.should_warn:
            msg = format_project_budget_status(budget_status)
            state.fired.add("project-budget")
            _report(msg, out, config.alert_log)

    streamer: EventStreamer | None = None
    if stream_config and stream_config.url:
        streamer = EventStreamer(stream_config)

    last_event_time = time.time()
    last_heartbeat_time = 0.0
    event_count = 0
    clean_session_end = False

    try:
        for item in _tail_events(events_file, poll_interval=poll_interval):
            if isinstance(item, _StartupCwd):
                state.session_cwd = item.cwd
                continue

            now = time.time()
            if now - last_heartbeat_time >= 5.0:
                write_heartbeat(store, session_id, agent_pid=state.agent_pid)
                last_heartbeat_time = now

            # A duration limit that kills, such as --timeout, is wall-clock
            # time, so check it on every poll cycle: an agent that hangs and
            # writes nothing is still killed.
            timeout_msg = _check_timeout(config, state, out)
            if timeout_msg:
                _dispatch_alert(
                    timeout_msg, config, state, action="kill", dry_run=dry_run,
                    store=store, session_id=session_id,
                )

            # Idle timeout: checked on every poll cycle (including when no
            # event arrived), so it fires reliably after max_idle_seconds.
            # A paused agent writes no events, so the clock stops while
            # watch holds a pause and restarts when it is released. With a
            # duration limit that kills, watch does not stop before the
            # deadline while the agent process runs (see
            # _timeout_holds_watch).
            if state.paused:
                last_event_time = now
            elif now - last_event_time > max_idle_seconds and not _timeout_holds_watch(
                config, state, now, out,
            ):
                out.write(f"[watch] No events for {max_idle_seconds:.0f}s - stopping\n")
                break

            if item is _IDLE_SENTINEL:
                # Approval decisions and the VS Code .pause-request marker
                _check_pause_file(store, state, out)
                continue

            event: TraceEvent = item  # type: ignore[assignment]
            event_count += 1
            last_event_time = time.time()
            _check_approval_decision(store, state, out)

            if not isinstance(event.data, Mapping) and _live_checks_read_data(
                event, state, checkpoint_mode=checkpoint_watcher is not None,
                path_rules=any(
                    rule.reads_paths and not rule.fired for rule in nanny_rules or ()
                ),
            ):
                out.write(skipped_event_warning("watch", event))
                out.flush()
                if streamer:
                    streamer.enqueue(event)
                continue

            # A checkpoint reads the data of every event it keeps.
            if checkpoint_watcher and isinstance(event.data, Mapping):
                try:
                    checkpoint_path = checkpoint_watcher.update(event)
                    if checkpoint_path:
                        out.write(
                            f"[watch] Compaction checkpoint written to {inert(checkpoint_path)}\n"
                        )
                        out.flush()
                except UnusableTimestampError:
                    out.write(
                        left_out_timestamp_warning("watch", event, "compaction checkpoint")
                    )
                    out.flush()
                except OSError as exc:
                    out.write(f"[watch] Could not write compaction checkpoint: {exc}\n")
                    out.flush()

            # Push event to stream if configured
            if streamer:
                streamer.enqueue(event)

            violations = check_event(event, config, state)
            for msg in violations:
                _dispatch_alert(
                    msg, config, state,
                    action=_limit_action(msg, config, state, project_budget_config),
                    dry_run=dry_run, store=store, session_id=session_id,
                )

            if project_budget_config and project_budget_config.enabled:
                budget_status = ProjectBudgetStatus(
                    config=project_budget_config,
                    spent=project_budget_base_spend + state.estimated_cost,
                    window_start=time.time() - 7 * 24 * 60 * 60,
                    window_end=time.time(),
                )
                if budget_status.should_warn and "project-budget" not in state.fired:
                    state.fired.add("project-budget")
                    msg = format_project_budget_status(budget_status)
                    _report(msg, out, config.alert_log)
                    if config.webhook_url:
                        _alert_webhook(msg, config.webhook_url)

            # --- Nanny rule evaluation ---
            if nanny_rules:
                metrics = state.nanny_metrics()
                for rule in nanny_rules:
                    if rule.fired:
                        continue
                    if rule.evaluate(metrics, event, cwd=state.session_cwd):
                        rule.fired = True
                        _dispatch_nanny_rule(
                            rule, event, store, session_id,
                            state, config, dry_run=dry_run,
                        )

            if event.event_type == EventType.SESSION_END:
                clean_session_end = True
                clear_heartbeat(store, session_id)
                out.write(f"[watch] Session ended ({event_count} events, ${state.estimated_cost:.4f})\n")
                break

    except KeyboardInterrupt:
        out.write("\n[watch] Stopped.\n")
    finally:
        _report_pause_left_held(store, state, out, config.alert_log)
        if streamer:
            streamer.stop()
        _wait_for_sigkill(out)
        if clean_session_end:
            clear_heartbeat(store, session_id)


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_watch(args: argparse.Namespace) -> int:
    # Check the stream flags before anything is read or written, so a bad
    # value costs one line and exit status 1.
    stream_url = getattr(args, "stream_to", None)
    if stream_url is not None:
        problem = _stream_to_problem(stream_url)
        if problem:
            sys.stderr.write(f"[watch] invalid --stream-to value: {problem}\n")
            return 1
        problem = _stream_flush_interval_problem(
            getattr(args, "stream_flush_interval", 2.0)
        )
        if problem:
            sys.stderr.write(f"[watch] invalid --stream-flush-interval value: {problem}\n")
            return 1
    # --protocol, then the OTEL protocol variables, choose the OTLP encoding.
    # The variables are read only for --stream-otlp: NDJSON is not OTLP.
    stream_protocol = _OTLP_JSON
    protocol_flag = getattr(args, "protocol", None)
    if stream_url is not None and getattr(args, "stream_otlp", False):
        try:
            stream_protocol = _resolve_otlp_protocol(protocol_flag, os.environ)
        except ValueError as exc:
            sys.stderr.write(f"[watch] {exc}\n")
            return 1
    elif protocol_flag is not None:
        sys.stderr.write("[watch] --protocol is used only with --stream-to and --stream-otlp\n")
        return 1

    max_context_pct = getattr(args, "max_context_pct", None)
    if max_context_pct is not None and not 1 <= max_context_pct <= 100:
        sys.stderr.write(
            "[watch] invalid --max-context-pct value: it must be a whole number from 1 to 100\n"
        )
        return 1

    # A limit that can never fire, or fires at the first check, is refused.
    try:
        timeout_seconds, budget = _read_limit_flags(args)
    except ValueError as exc:
        sys.stderr.write(f"[watch] {exc}\n")
        return 1

    # --timeout and --budget replace --max-duration and --max-cost, and they
    # also kill the agent whatever --on-violation says.
    max_duration = getattr(args, "max_duration", None)
    if timeout_seconds is not None:
        max_duration = timeout_seconds
    max_cost = getattr(args, "max_cost", None)
    if budget is not None:
        max_cost = budget

    # A --config file that cannot be used is refused, not replaced by the
    # defaults, so the limits it sets are never silently dropped.
    config_path = getattr(args, "config", None)
    if config_path is not None:
        try:
            config = WatcherConfig.load(config_path)
        except WatchConfigError as exc:
            sys.stderr.write(f"[watch] invalid --config file {_shown_path(config_path)}: {exc}\n")
            return 1
    else:
        config = WatcherConfig()

    # A --rules value that cannot be used is refused the same way, so watch
    # never runs with fewer or different rules than the user wrote.
    nanny_rules: list[NannyRule] | None = None
    builtin_rules: list[tuple[str, Any]] | None = None
    rules_value = getattr(args, "rules", None)
    if rules_value is not None:
        try:
            nanny_rules, builtin_rules = _load_watch_rules(rules_value)
        except WatchRulesError as exc:
            sys.stderr.write(f"[watch] {exc}\n")
            return 1

    # Flags given on the command line override the file. Their argparse
    # defaults are None, so an absent flag keeps the file's value.
    if getattr(args, "max_retries", None) is not None:
        config.max_retries = int(args.max_retries)
    if max_cost is not None:
        config.max_cost_dollars = max_cost
        config.cost_limit_set = True
    if max_duration is not None:
        config.max_duration_seconds = max_duration
        config.duration_limit_set = True
    config.kill_on_timeout = timeout_seconds is not None
    config.kill_on_budget = budget is not None
    if getattr(args, "on_violation", None) is not None:
        config.on_violation = args.on_violation
    if getattr(args, "webhook", None) is not None:
        config.webhook_url = args.webhook
    if getattr(args, "on_death", None) is not None:
        config.on_death_cmd = args.on_death
    if getattr(args, "policy", None) is not None:
        config.scope_policy = args.policy or ".agent-scope.json"
    if max_context_pct is not None:
        config.max_context_pct = int(max_context_pct)
    if getattr(args, "loop_threshold", None) is not None:
        config.loop_threshold = int(getattr(args, "loop_threshold"))
    if getattr(args, "loop_window", None) is not None:
        config.loop_window = int(getattr(args, "loop_window"))

    # Check the URL alerts will be POSTed to before the agent pid is checked,
    # the session is looked up or anything is written. Without this check a
    # bad URL fails at the first alert. The line never quotes the URL: its
    # path and query can hold a token. A given --webhook overrides the file.
    if config.webhook_url:
        problem = _stream_to_problem(config.webhook_url)
        if problem:
            if getattr(args, "webhook", None) is None and config_path is not None:
                shown = config_path if config_path.isprintable() else repr(config_path)
                source = f"webhook.url value in {shown}"
            else:
                source = "--webhook value"
            sys.stderr.write(f"[watch] invalid {source}: {problem}\n")
            return 1

    # The --config file's budget keys, checked when it was read, replace the
    # standard project config when the file sets weekly or per_session_max.
    # A project config that cannot be used is refused before the store is
    # opened, so nothing is written.
    project_budget_config = config.project_budget
    if project_budget_config is None:
        try:
            project_budget_config = load_project_budget_config()
        except ProjectBudgetConfigError as exc:
            sys.stderr.write(f"[watch] {exc}\n")
            return 1

    store = TraceStore(args.trace_dir)

    per_session_max = project_budget_config.per_session_max
    if per_session_max:
        # per_session_max kills (see _limit_action). It may lower a cost
        # limit that kills, from --budget, or from --max-cost or the file
        # with the kill action, but never raises it. It replaces a cost
        # limit that only alerts or pauses, and the default.
        if _cost_limit_kills(config):
            config.max_cost_dollars = min(config.max_cost_dollars, per_session_max)
        else:
            config.max_cost_dollars = per_session_max

    # Built-in rules are applied last. budget:N and timeout:D lower a limit
    # that kills, such as per_session_max, but never raise it; they replace
    # any other limit.
    if builtin_rules:
        _apply_builtin_rules(builtin_rules, config, per_session_max)

    dry_run = getattr(args, "dry_run", False)

    agent_pid = getattr(args, "agent_pid", None)
    if agent_pid is not None:
        error = validate_agent_pid(agent_pid)
        if error:
            sys.stderr.write(f"[watch] {error}\n")
            return 1

    session_id = getattr(args, "session_id", None)
    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1

    full_id = store.find_session(session_id)
    if not full_id:
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    try:
        requested_tenant = getattr(args, "tenant_id", None)
        tenant_id = validate_optional_tenant_id(
            "" if requested_tenant is None else requested_tenant
        )
        if tenant_id == "":
            tenant_id = validate_optional_tenant_id(
                os.environ.get("AGENT_STRACE_TENANT_ID", "").strip()
            )
        if tenant_id:
            tagged_events = store.tag_session(full_id, tenant_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # The error can quote an event ID from the session's events file.
        sys.stderr.write(f"[watch] could not assign tenant: {inert(exc)}\n")
        return 1
    if tenant_id:
        sys.stderr.write(
            f"[watch] Tenant {tenant_id} assigned to "
            f"{tagged_events} existing event(s).\n"
        )

    checkpoint_watcher: CompactionCheckpointWatcher | None = None
    if getattr(args, "compaction_checkpoint", False):
        try:
            checkpoint_watcher = CompactionCheckpointWatcher(
                store,
                full_id,
                checkpoint_at=float(getattr(args, "checkpoint_at", 0.80)),
            )
        except (NonObjectEventDataError, UnusableTimestampError):
            raise
        except ValueError as exc:
            sys.stderr.write(f"[watch] invalid checkpoint threshold: {exc}\n")
            return 1

    # Build StreamConfig if --stream-to was provided (checked above)
    stream_cfg: StreamConfig | None = None
    if stream_url:
        stream_cfg = StreamConfig(
            url=stream_url,
            batch_size=getattr(args, "stream_batch_size", 10),
            flush_interval=getattr(args, "stream_flush_interval", 2.0),
            otlp=getattr(args, "stream_otlp", False),
            protocol=stream_protocol,
        )

    watch_session(
        store,
        full_id,
        config,
        nanny_rules=nanny_rules,
        dry_run=dry_run,
        stream_config=stream_cfg,
        project_budget_config=project_budget_config,
        checkpoint_watcher=checkpoint_watcher,
        agent_pid=agent_pid,
    )
    return 0


# The built-in rules a --rules value can list, as name or name:value.
_BUILTIN_RULE_NAMES = ("mcp-poisoning", "loop", "budget", "timeout", "cognitive-debt")
_WHOLE_NUMBER = re.compile(r"[0-9]+", re.ASCII)


def _builtin_rule_value(name: str, value: str | None, item: str) -> Any:
    """Return the checked value of one built-in rule; None when it has none.

    *value* is the text after the colon, or None when there is no colon.
    """
    def refuse(problem: str) -> WatchRulesError:
        return WatchRulesError(f"{item!r}: {problem}")

    if name == "mcp-poisoning":
        if value is not None:
            raise refuse("mcp-poisoning takes no value")
        return None
    if name == "loop":
        if value is None:
            return None
        threshold, slash, window = value.partition("/")
        parts = [part.strip() for part in ((threshold, window) if slash else (threshold,))]
        numbers: list[int] = []
        for part in parts:
            if not _WHOLE_NUMBER.fullmatch(part):
                break
            try:
                numbers.append(int(part))
            except ValueError:  # more digits than int() accepts
                break
        if len(numbers) == len(parts) and min(numbers) >= 1:
            return numbers[0], numbers[1] if slash else None
        raise refuse("N and W in loop:N and loop:N/W must be whole numbers, 1 or more")
    if name == "budget":
        text = (value or "").strip()
        try:
            dollars = float(text[1:] if text.startswith("$") else text)
        except ValueError:
            dollars = math.nan
        if not (math.isfinite(dollars) and dollars >= 0):
            raise refuse("N in budget:N must be a finite number, 0 or more")
        return dollars
    if name == "timeout":
        try:
            seconds = _parse_duration(value or "")
        except ValueError:
            seconds = math.nan
        if not (math.isfinite(seconds) and seconds > 0):
            raise refuse("D in timeout:D must be a duration more than 0, such as 30m or 2h")
        return seconds
    # cognitive-debt with no value keeps the configured threshold.
    if value is None:
        return None
    try:
        threshold_value = float(value)
    except ValueError:
        threshold_value = math.nan
    if not (math.isfinite(threshold_value) and 0 <= threshold_value <= 1):
        raise refuse("N in cognitive-debt:N must be a number from 0 to 1")
    return threshold_value


def _parse_builtin_rules(spec: str) -> list[tuple[str, Any]]:
    """Return (name, value) for each item of a comma-separated --rules list.

    Raises WatchRulesError, naming the item, for an empty item, an item that
    is not a built-in rule, a rule listed twice, or a value the rule cannot
    use.
    """
    items = [item.strip() for item in spec.split(",")]
    if not any(items):
        raise WatchRulesError("it lists no rules")
    parsed: list[tuple[str, Any]] = []
    for position, item in enumerate(items, 1):
        if not item:
            raise WatchRulesError(f"item {position} is empty")
        name, colon, value = item.partition(":")
        if name not in _BUILTIN_RULE_NAMES:
            if os.path.exists(item):
                raise WatchRulesError(
                    f"{item!r} is not a built-in rule; --rules takes one rules file "
                    "or a list of built-in rules, not both"
                )
            raise WatchRulesError(
                f"{item!r} is not a built-in rule or an existing file; the built-in "
                "rules are mcp-poisoning, loop, budget, timeout and cognitive-debt"
            )
        # A second value would replace the first without a word.
        if any(seen == name for seen, _ in parsed):
            raise WatchRulesError(f"{item!r} lists {name} again; list each built-in rule once")
        parsed.append((name, _builtin_rule_value(name, value.strip() if colon else None, item)))
    return parsed


def _apply_builtin_rules(
    parsed: list[tuple[str, Any]],
    config: WatcherConfig,
    per_session_max: float | None = None,
) -> None:
    """Apply the rules _parse_builtin_rules returned to config, in order.

    budget:N and timeout:D may lower a limit the user set that kills, but
    never raise it, and the limit still kills. They replace any other limit,
    and the defaults. *per_session_max* is the project budget's: it makes
    the cost limit one that kills. timeout:D sets the duration limit, so
    with the kill action it is a limit that kills (see
    _duration_limit_kills).
    """
    cost_kills = _cost_limit_kills(config, per_session_max)
    duration_kills = _duration_limit_kills(config)
    for name, value in parsed:
        # The rule's value is first, so min keeps it unless the limit is
        # lower: a NaN limit never wins.
        if name == "budget":
            config.max_cost_dollars = (
                min(value, config.max_cost_dollars) if cost_kills else value
            )
        elif name == "timeout":
            config.max_duration_seconds = (
                min(value, config.max_duration_seconds) if duration_kills else value
            )
            config.duration_limit_set = True
        else:
            if name == "loop" and value is not None:
                config.loop_threshold, window = value
                if window is not None:
                    config.loop_window = window
            elif name == "cognitive-debt" and value is not None:
                config.cognitive_debt_threshold = value
            config.built_in_rules.add(name)


def _apply_builtin_rules_spec(
    spec: str,
    config: WatcherConfig,
    per_session_max: float | None = None,
) -> None:
    """Apply comma-separated built-in watch rules to config, as cmd_watch does.

    Raises WatchRulesError, before changing config, for an item that is not
    a built-in rule or has a value the rule cannot use.
    """
    _apply_builtin_rules(_parse_builtin_rules(spec), config, per_session_max)


def _looks_like_file(item: str) -> bool:
    """True when a --rules item reads as a file name rather than a rule."""
    return (
        "/" in item or "\\" in item
        or item.lower().endswith((".json", ".yaml", ".yml"))
    )


def _load_watch_rules(
    value: str,
) -> tuple[list[NannyRule] | None, list[tuple[str, Any]] | None]:
    """Check a --rules value before watch reads the session.

    An existing path, or a single item that reads as a file name, is a rules
    file: the result is (rules, None). Anything else is a comma-separated
    list of built-in rules: the result is (None, parsed rules), applied later
    by _apply_builtin_rules. Raises WatchRulesError, with the line cmd_watch
    prints after "[watch] ", for a value watch cannot use.
    """
    if not value.strip():
        raise WatchRulesError("invalid --rules value: it is empty")
    items = [item.strip() for item in value.split(",")]
    names_file = (
        len(items) == 1 and _looks_like_file(items[0])
        and items[0].partition(":")[0] not in _BUILTIN_RULE_NAMES
    )
    if os.path.exists(value) or names_file:
        try:
            return _load_nanny_rules(value), None
        except WatchRulesError as exc:
            raise WatchRulesError(
                f"invalid --rules file {_shown_path(value)}: {exc}"
            ) from None
    try:
        return None, _parse_builtin_rules(value)
    except WatchRulesError as exc:
        raise WatchRulesError(f"invalid --rules value: {exc}") from None
