"""Live session monitoring with circuit breakers.

Tails the active session's events.ndjson and triggers alerts when
configurable thresholds are exceeded. Zero new dependencies — stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TextIO

from .cost import _dollars, _event_tokens
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Alert actions
# ---------------------------------------------------------------------------

def _alert_terminal(message: str, out: TextIO = sys.stderr) -> None:
    out.write(f"[watch] ⚠️  {message}\n")
    out.flush()


def _alert_file(message: str, log_path: str = ".agent-traces/alerts.log") -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        f.write(f"{ts}  {message}\n")


def _alert_webhook(message: str, url: str) -> None:
    payload = json.dumps({"text": message, "source": "agent-strace"}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:
        pass  # webhook failures are non-fatal


def _kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------------------
# Watcher state machines
# ---------------------------------------------------------------------------

@dataclass
class WatcherConfig:
    max_retries: int = 5
    max_cost_dollars: float = 10.0
    max_duration_seconds: float = 1800.0
    loop_sequence_length: int = 3
    loop_max_repeats: int = 3
    scope_policy: str = ".agent-scope.json"
    on_violation: str = "terminal"   # terminal | file | kill
    webhook_url: str = ""
    alert_log: str = ".agent-traces/alerts.log"

    @classmethod
    def from_dict(cls, d: dict) -> "WatcherConfig":
        watchers = d.get("watchers", {})
        retry_cfg = watchers.get("retry", {})
        cost_cfg = watchers.get("cost", {})
        dur_cfg = watchers.get("duration", {})
        loop_cfg = watchers.get("loop", {})
        scope_cfg = watchers.get("scope", {})
        webhook = d.get("webhook", {})
        return cls(
            max_retries=int(retry_cfg.get("max", 5)),
            max_cost_dollars=float(cost_cfg.get("max_dollars", 10.0)),
            max_duration_seconds=float(dur_cfg.get("max_minutes", 30)) * 60,
            loop_sequence_length=int(loop_cfg.get("sequence_length", 3)),
            loop_max_repeats=int(loop_cfg.get("max_repeats", 3)),
            scope_policy=str(scope_cfg.get("policy", ".agent-scope.json")),
            on_violation=str(retry_cfg.get("alert", "terminal")),
            webhook_url=str(webhook.get("url", "")),
        )

    @classmethod
    def load(cls, path: str) -> "WatcherConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except Exception:
            return cls()


@dataclass
class WatchState:
    """Mutable state accumulated across events."""
    # Retry tracking: command → count
    command_counts: Counter = field(default_factory=Counter)
    # Cost accumulation
    estimated_cost: float = 0.0
    # Session start time
    start_time: float = field(default_factory=time.time)
    # Recent event sequence for loop detection (circular buffer)
    recent_events: deque = field(default_factory=lambda: deque(maxlen=30))
    # Violations already fired (to avoid duplicate alerts)
    fired: set = field(default_factory=set)
    # Agent PID (from session meta, if available)
    agent_pid: int | None = None


def _event_key(event: TraceEvent) -> str:
    """Stable string key for an event (for loop detection)."""
    if event.event_type == EventType.TOOL_CALL:
        name = event.data.get("tool_name", "?")
        args = event.data.get("arguments", {}) or {}
        cmd = str(args.get("command", args.get("file_path", "")))[:40]
        return f"{name}:{cmd}"
    return event.event_type.value


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

def _dispatch_alert(
    message: str,
    config: WatcherConfig,
    state: WatchState,
    action: str | None = None,
) -> None:
    action = action or config.on_violation
    _alert_terminal(message)
    if action == "file":
        _alert_file(message, config.alert_log)
    if config.webhook_url:
        _alert_webhook(message, config.webhook_url)
    if action == "kill" and state.agent_pid:
        _alert_terminal(f"Killing agent process {state.agent_pid}")
        _kill_process(state.agent_pid)


# ---------------------------------------------------------------------------
# Per-event check
# ---------------------------------------------------------------------------

def check_event(
    event: TraceEvent,
    config: WatcherConfig,
    state: WatchState,
) -> list[str]:
    """Update state and return list of violation messages (may be empty)."""
    violations: list[str] = []

    # --- Cost accumulation ---
    inp, out = _event_tokens(event)
    state.estimated_cost += _dollars(inp, out, "sonnet")

    # --- Loop detection ---
    key = _event_key(event)
    state.recent_events.append(key)

    # --- Retry detection (bash commands) ---
    if event.event_type == EventType.TOOL_CALL:
        name = event.data.get("tool_name", "").lower()
        args = event.data.get("arguments", {}) or {}
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
        key_id = f"cost:{int(state.estimated_cost)}"
        if key_id not in state.fired:
            state.fired.add(key_id)
            violations.append(
                f"CostWatcher: ${state.estimated_cost:.2f} (threshold: ${config.max_cost_dollars})"
            )

    # --- Duration threshold ---
    elapsed = time.time() - state.start_time
    if elapsed > config.max_duration_seconds:
        key_id = "duration"
        if key_id not in state.fired:
            state.fired.add(key_id)
            violations.append(
                f"DurationWatcher: {elapsed:.0f}s elapsed (threshold: {config.max_duration_seconds:.0f}s)"
            )

    # --- Loop detection ---
    loop_msg = _detect_loop(
        state.recent_events,
        config.loop_sequence_length,
        config.loop_max_repeats,
    )
    if loop_msg:
        key_id = f"loop:{loop_msg[:40]}"
        if key_id not in state.fired:
            state.fired.add(key_id)
            violations.append(f"LoopWatcher: {loop_msg}")

    # --- Scope check (file operations) ---
    if event.event_type == EventType.TOOL_CALL:
        scope_path = Path(config.scope_policy)
        if scope_path.exists():
            try:
                from .audit import Policy, _glob_match
                policy = Policy.load(scope_path)
                if policy:
                    name = event.data.get("tool_name", "").lower()
                    args = event.data.get("arguments", {}) or {}
                    path = str(args.get("file_path") or args.get("path") or "")
                    if path and name in ("write", "edit", "create"):
                        if policy.file_write_deny and _glob_match(path, policy.file_write_deny):
                            key_id = f"scope:{path}"
                            if key_id not in state.fired:
                                state.fired.add(key_id)
                                violations.append(f"ScopeWatcher: write to {path} denied by policy")
            except Exception:
                pass

    return violations


# ---------------------------------------------------------------------------
# File tailer
# ---------------------------------------------------------------------------

def _tail_events(events_file: Path, poll_interval: float = 0.5):
    """Generator that yields new TraceEvent lines as they appear."""
    with open(events_file, "r", encoding="utf-8") as f:
        # Skip existing content
        f.seek(0, 2)
        while True:
            line = f.readline()
            if line:
                line = line.strip()
                if line:
                    try:
                        yield TraceEvent.from_json(line)
                    except Exception:
                        pass
            else:
                time.sleep(poll_interval)


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
) -> None:
    """Watch a session's event stream and fire alerts on violations."""
    events_file = store._session_dir(session_id) / "events.ndjson"
    if not events_file.exists():
        out.write(f"[watch] events file not found: {events_file}\n")
        return

    state = WatchState(start_time=time.time())

    # Try to read agent PID from meta
    try:
        meta = store.load_meta(session_id)
        # PID not stored in meta currently; placeholder for future use
    except Exception:
        pass

    out.write(f"[watch] Monitoring session {session_id[:12]}...\n")
    out.flush()

    last_event_time = time.time()
    event_count = 0

    try:
        for event in _tail_events(events_file, poll_interval=poll_interval):
            event_count += 1
            last_event_time = time.time()

            violations = check_event(event, config, state)
            for msg in violations:
                _dispatch_alert(msg, config, state)

            if event.event_type == EventType.SESSION_END:
                out.write(f"[watch] Session ended ({event_count} events, ${state.estimated_cost:.4f})\n")
                break

            # Idle timeout
            if time.time() - last_event_time > max_idle_seconds:
                out.write(f"[watch] No events for {max_idle_seconds:.0f}s — stopping\n")
                break

    except KeyboardInterrupt:
        out.write("\n[watch] Stopped.\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_watch(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    # Load config file if provided
    config_path = getattr(args, "config", None)
    if config_path:
        config = WatcherConfig.load(config_path)
    else:
        config = WatcherConfig(
            max_retries=getattr(args, "max_retries", 5),
            max_cost_dollars=getattr(args, "max_cost", 10.0),
            max_duration_seconds=getattr(args, "max_duration", 1800),
            on_violation=getattr(args, "on_violation", "terminal"),
            webhook_url=getattr(args, "webhook", "") or "",
        )

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

    watch_session(store, full_id, config)
    return 0
