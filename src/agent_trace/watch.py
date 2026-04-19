"""Live session monitoring with circuit breakers and rule-based kill switch.

Tails the active session's events.ndjson and triggers alerts when
configurable thresholds are exceeded. Zero new dependencies — stdlib only.

Rule-based nanny (--rules rules.yaml):
  Evaluates declarative rules on every event. Supports pause (SIGSTOP/SIGCONT)
  and kill (SIGTERM) actions with optional notifications. Auto-generates a
  postmortem when a kill action fires.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import signal
import sys
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

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
    """Send SIGTERM; escalate to SIGKILL after 5s in a background thread."""
    import threading

    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    def _escalate() -> None:
        time.sleep(5)
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    t = threading.Thread(target=_escalate, daemon=True)
    t.start()


def _pause_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGSTOP)
    except (ProcessLookupError, PermissionError):
        pass


def _resume_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGCONT)
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------------------
# Nanny rules (--rules rules.yaml)
# ---------------------------------------------------------------------------

@dataclass
class NannyRule:
    """A declarative rule evaluated on every event."""
    name: str
    condition: str          # e.g. "files_modified > 20", "cost_usd > 5.00"
    action: str             # "pause" | "kill" | "alert"
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
        cond = self.condition.strip()

        # file_path matches "/etc/**"
        m = re.match(r'file_path\s+matches\s+"([^"]+)"', cond)
        if not m:
            m = re.match(r"file_path\s+matches\s+'([^']+)'", cond)
        if m:
            self._metric = "file_path"
            self._op = "matches"
            self._pattern = m.group(1)
            return

        # numeric comparisons: metric op value
        m = re.match(r'(\w+)\s*(>=|<=|>|<|==)\s*([0-9.]+)', cond)
        if m:
            self._metric = m.group(1)
            self._op = m.group(2)
            raw = m.group(3)
            self._threshold = float(raw) if "." in raw else int(raw)

    def evaluate(self, metrics: dict[str, Any], event: TraceEvent | None = None) -> bool:
        """Return True if this rule's condition is satisfied."""
        if self._metric == "file_path" and self._op == "matches" and event:
            if event.event_type == EventType.TOOL_CALL:
                args = event.data.get("arguments", {}) or {}
                path = str(
                    args.get("file_path") or args.get("path") or ""
                )
                return bool(path and fnmatch.fnmatch(path, self._pattern))
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


def _load_nanny_rules(path: str) -> list[NannyRule]:
    """Load rules from a YAML or JSON file. Returns [] on error."""
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text()
    try:
        # Try JSON first (no extra dep)
        data = json.loads(text)
    except json.JSONDecodeError:
        # Minimal YAML parser for simple rule files (no PyYAML required)
        data = _parse_simple_yaml(text)

    rules: list[NannyRule] = []
    for r in data.get("rules", []):
        try:
            rules.append(NannyRule(
                name=r.get("name", "unnamed"),
                condition=r.get("condition", ""),
                action=r.get("action", "alert"),
                notify=r.get("notify", ""),
            ))
        except Exception:
            pass
    return rules


def _parse_simple_yaml(text: str) -> dict:
    """Parse a minimal subset of YAML sufficient for rules files.

    Handles: top-level 'rules:' list with string scalar fields.
    Does not handle anchors, multi-line values, or complex types.
    """
    result: dict = {"rules": []}
    current_rule: dict | None = None
    in_rules = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()

        if stripped.startswith("#") or not stripped:
            continue

        if stripped == "rules:":
            in_rules = True
            continue

        if not in_rules:
            continue

        # New list item
        if stripped.startswith("- "):
            if current_rule is not None:
                result["rules"].append(current_rule)
            # Inline key: value after the dash
            rest = stripped[2:].strip()
            current_rule = {}
            if ":" in rest:
                k, _, v = rest.partition(":")
                current_rule[k.strip()] = v.strip().strip('"').strip("'")
        elif stripped.startswith("-") and stripped == "-":
            if current_rule is not None:
                result["rules"].append(current_rule)
            current_rule = {}
        elif current_rule is not None and ":" in stripped:
            k, _, v = stripped.partition(":")
            current_rule[k.strip()] = v.strip().strip('"').strip("'")

    if current_rule is not None:
        result["rules"].append(current_rule)

    return result


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
    # Per-operation enforcement rules
    operation_rules: list[OperationRule] = field(default_factory=list)
    # Token budget threshold (1–100, percentage of context window)
    max_context_pct: int = 90

    @classmethod
    def from_dict(cls, d: dict) -> "WatcherConfig":
        watchers = d.get("watchers", {})
        retry_cfg = watchers.get("retry", {})
        cost_cfg = watchers.get("cost", {})
        dur_cfg = watchers.get("duration", {})
        loop_cfg = watchers.get("loop", {})
        scope_cfg = watchers.get("scope", {})
        webhook = d.get("webhook", {})

        # Parse per-operation rules
        rules: list[OperationRule] = []
        for rule_dict in d.get("operation_rules", []):
            rules.append(OperationRule(
                tool_name=rule_dict.get("tool", "*"),
                pattern=rule_dict.get("pattern", "*"),
                action=rule_dict.get("action", "alert"),
                reason=rule_dict.get("reason", ""),
            ))

        return cls(
            max_retries=int(retry_cfg.get("max", 5)),
            max_cost_dollars=float(cost_cfg.get("max_dollars", 10.0)),
            max_duration_seconds=float(dur_cfg.get("max_minutes", 30)) * 60,
            loop_sequence_length=int(loop_cfg.get("sequence_length", 3)),
            loop_max_repeats=int(loop_cfg.get("max_repeats", 3)),
            scope_policy=str(scope_cfg.get("policy", ".agent-scope.json")),
            on_violation=str(retry_cfg.get("alert", "terminal")),
            webhook_url=str(webhook.get("url", "")),
            operation_rules=rules,
            max_context_pct=int(d.get("max_context_pct", 90)),
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
    # Token budget watcher (lazy-initialised on first LLM_REQUEST)
    token_budget_watcher: object | None = None
    # Nanny rule metrics
    files_modified: int = 0          # distinct files written/edited
    files_modified_set: set = field(default_factory=set)
    consecutive_test_failures: int = 0
    duration_minutes: float = 0.0
    paused: bool = False             # True when agent is SIGSTOP'd

    def nanny_metrics(self) -> dict:
        """Return current metric snapshot for nanny rule evaluation."""
        elapsed = time.time() - self.start_time
        return {
            "files_modified": self.files_modified,
            "cost_usd": self.estimated_cost,
            "consecutive_test_failures": self.consecutive_test_failures,
            "duration_minutes": elapsed / 60.0,
        }


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
    notify: str = "",
    dry_run: bool = False,
) -> None:
    action = action or config.on_violation
    # terminal is always shown regardless of action so the operator watching
    # the process sees every alert; file/webhook are additive channels
    _alert_terminal(message)
    if action in ("file", "kill", "pause"):
        _alert_file(message, config.alert_log)
    if config.webhook_url:
        _alert_webhook(message, config.webhook_url)
    # notify field from nanny rules (e.g. "slack:#alerts")
    if notify and notify.startswith("slack:") and config.webhook_url:
        _alert_webhook(f"[{notify}] {message}", config.webhook_url)

    if dry_run:
        _alert_terminal(f"[dry-run] would {action} agent process")
        return

    if action == "pause" and state.agent_pid and not state.paused:
        _alert_terminal(f"Pausing agent process {state.agent_pid} (SIGSTOP)")
        _pause_process(state.agent_pid)
        state.paused = True
    elif action == "kill" and state.agent_pid:
        _alert_terminal(f"Killing agent process {state.agent_pid}")
        _kill_process(state.agent_pid)


def _dispatch_nanny_rule(
    rule: NannyRule,
    event: TraceEvent,
    store: TraceStore,
    session_id: str,
    state: WatchState,
    config: WatcherConfig,
    dry_run: bool = False,
) -> None:
    """Fire a nanny rule: alert, pause, or kill."""
    msg = f"NannyRule '{rule.name}': {rule.condition} → {rule.action}"
    _dispatch_alert(msg, config, state, action=rule.action, notify=rule.notify, dry_run=dry_run)

    # Auto-generate postmortem on kill
    if rule.action == "kill" and not dry_run:
        try:
            from .postmortem import generate_postmortem, format_postmortem
            pm = generate_postmortem(store, session_id)
            pm_path = Path(config.alert_log).parent / f"postmortem-{session_id[:12]}.md"
            pm_path.write_text(format_postmortem(pm))
            _alert_terminal(f"Postmortem written to {pm_path}")
        except Exception:
            pass


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

    # --- Track files modified (for nanny rules) ---
    if event.event_type == EventType.TOOL_CALL:
        _name = event.data.get("tool_name", "").lower()
        _args = event.data.get("arguments", {}) or {}
        if _name in ("write", "edit", "create", "str_replace"):
            _path = str(_args.get("file_path") or _args.get("path") or "")
            if _path and _path not in state.files_modified_set:
                state.files_modified_set.add(_path)
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
        key_id = "cost"
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
        # Key on the sequence pattern only (strip the " × N" count suffix)
        # so dedup fires once per unique loop, not once per repeat increment.
        seq_part = loop_msg.split(" × ")[0]
        key_id = f"loop:{seq_part[:60]}"
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
        tool_name = event.data.get("tool_name", "").lower()
        args = event.data.get("arguments", {}) or {}
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

    return violations


# ---------------------------------------------------------------------------
# File tailer
# ---------------------------------------------------------------------------

_IDLE_SENTINEL = object()  # yielded when poll_interval elapses with no new event


def _check_pause_file(store: "TraceStore", state: "WatchState", out: "TextIO") -> None:
    """Honour a .pause-request file written by the VS Code extension.

    Presence of the file → SIGSTOP the agent (if a PID is known).
    Absence after a pause → SIGCONT to resume.
    """
    pause_file = store.base_dir / ".pause-request"
    wants_pause = pause_file.exists()

    if wants_pause and not state.paused and state.agent_pid:
        out.write(f"[watch] Pause requested by editor — SIGSTOP pid {state.agent_pid}\n")
        out.flush()
        _pause_process(state.agent_pid)
        state.paused = True
    elif not wants_pause and state.paused and state.agent_pid:
        out.write(f"[watch] Resume requested by editor — SIGCONT pid {state.agent_pid}\n")
        out.flush()
        _resume_process(state.agent_pid)
        state.paused = False


def _tail_events(events_file: Path, poll_interval: float = 0.5):
    """Generator that yields TraceEvent objects or _IDLE_SENTINEL each poll cycle.

    Yields _IDLE_SENTINEL when no new line arrived during the poll interval,
    allowing callers to implement idle-timeout logic without blocking.
    """
    with open(events_file, "r", encoding="utf-8") as f:
        # Skip existing content — start from the end of the file
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
                yield _IDLE_SENTINEL


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
) -> None:
    """Watch a session's event stream and fire alerts on violations."""
    events_file = store._session_dir(session_id) / "events.ndjson"
    if not events_file.exists():
        out.write(f"[watch] events file not found: {events_file}\n")
        return

    state = WatchState(start_time=time.time())

    mode = " [dry-run]" if dry_run else ""
    out.write(f"[watch] Monitoring session {session_id[:12]}...{mode}\n")
    if nanny_rules:
        out.write(f"[watch] {len(nanny_rules)} nanny rule(s) active\n")
    out.flush()

    last_event_time = time.time()
    event_count = 0

    try:
        for item in _tail_events(events_file, poll_interval=poll_interval):
            # Idle timeout: checked on every poll cycle (including when no
            # event arrived), so it fires reliably after max_idle_seconds.
            if time.time() - last_event_time > max_idle_seconds:
                out.write(f"[watch] No events for {max_idle_seconds:.0f}s - stopping\n")
                break

            if item is _IDLE_SENTINEL:
                # Check for pause-request signal file written by the VS Code extension
                _check_pause_file(store, state, out)
                continue

            event: TraceEvent = item  # type: ignore[assignment]
            event_count += 1
            last_event_time = time.time()

            violations = check_event(event, config, state)
            for msg in violations:
                _dispatch_alert(msg, config, state, dry_run=dry_run)

            # --- Nanny rule evaluation ---
            if nanny_rules:
                metrics = state.nanny_metrics()
                for rule in nanny_rules:
                    if rule.fired:
                        continue
                    if rule.evaluate(metrics, event):
                        rule.fired = True
                        _dispatch_nanny_rule(
                            rule, event, store, session_id,
                            state, config, dry_run=dry_run,
                        )

            if event.event_type == EventType.SESSION_END:
                out.write(f"[watch] Session ended ({event_count} events, ${state.estimated_cost:.4f})\n")
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

    # Load nanny rules if --rules provided
    nanny_rules: list[NannyRule] | None = None
    rules_path = getattr(args, "rules", None)
    if rules_path:
        nanny_rules = _load_nanny_rules(rules_path)
        if not nanny_rules:
            sys.stderr.write(f"[watch] Warning: no rules loaded from {rules_path}\n")

    dry_run = getattr(args, "dry_run", False)

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

    watch_session(store, full_id, config, nanny_rules=nanny_rules, dry_run=dry_run)
    return 0
