"""Live session monitoring with circuit breakers and rule-based kill switch.

Tails the active session's events.ndjson and triggers alerts when
configurable thresholds are exceeded. Zero new dependencies — stdlib only.

Rule-based nanny (--rules rules.yaml):
  Evaluates declarative rules on every event. Supports pause (SIGSTOP/SIGCONT)
  and kill (SIGTERM) actions with optional notifications. Auto-generates a
  postmortem when a kill action fires.

Watchdog mode (--timeout / --budget):
  Enforces a wall-clock timeout and/or token-cost ceiling. When either limit
  is breached the agent process is terminated and a structured post-mortem
  JSON file is written to the session directory. An optional --on-death
  command is invoked with the post-mortem path substituted for
  {post_mortem_path}.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
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
# Duration parsing
# ---------------------------------------------------------------------------

def _parse_duration(value: str) -> float:
    """Parse a human-readable duration string to seconds.

    Accepts: 30s, 5m, 2h, 1h30m, 90 (bare number = seconds).
    """
    value = value.strip()
    if not value:
        raise ValueError("empty duration")

    # bare number → seconds
    try:
        return float(value)
    except ValueError:
        pass

    total = 0.0
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*([smhd]?)")
    for m in pattern.finditer(value.lower()):
        num, unit = float(m.group(1)), m.group(2)
        if unit == "d":
            total += num * 86400
        elif unit == "h":
            total += num * 3600
        elif unit == "m":
            total += num * 60
        else:  # 's' or no unit
            total += num
    if total == 0.0:
        raise ValueError(f"cannot parse duration: {value!r}")
    return total


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
    action: str             # "pause" | "kill" | "alert" | "require_approval"
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
    # Watchdog: command to run after kill, with {post_mortem_path} substituted
    on_death_cmd: str = ""

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


# ---------------------------------------------------------------------------
# Push-based event streaming
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    """Configuration for push-based event streaming."""
    url: str = ""                    # webhook URL or OTLP collector endpoint
    batch_size: int = 10             # max events per POST
    flush_interval: float = 1.0     # max seconds between flushes
    headers: dict[str, str] = field(default_factory=dict)
    # Filter: only stream these event types (empty = all)
    event_types: list[str] = field(default_factory=list)
    # When True, flush as OTLP JSON spans instead of raw NDJSON
    otlp: bool = False
    # OTel service name used in OTLP spans
    service_name: str = "agent-trace"

    @classmethod
    def from_url(cls, url: str, headers: dict[str, str] | None = None,
                 otlp: bool = False) -> "StreamConfig":
        return cls(url=url, headers=headers or {}, otlp=otlp)


class EventStreamer:
    """Background thread that batches and POSTs events to a remote URL.

    Thread-safe: events are enqueued from the watch loop and flushed by a
    dedicated daemon thread. Failures are logged to stderr but never block
    the watch loop.
    """

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self._queue: queue.Queue[TraceEvent | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, event: TraceEvent) -> None:
        """Add an event to the outbound queue (non-blocking)."""
        if self.config.event_types:
            if event.event_type.value not in self.config.event_types:
                return
        self._queue.put(event)

    def stop(self) -> None:
        """Signal the worker to flush and exit."""
        self._queue.put(None)  # sentinel
        self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        batch: list[TraceEvent] = []
        last_flush = time.time()

        while True:
            try:
                timeout = max(0.01, self.config.flush_interval - (time.time() - last_flush))
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
                or (batch and now - last_flush >= self.config.flush_interval)
            )
            if should_flush:
                self._flush(batch)
                batch = []
                last_flush = now

    def _flush(self, events: list[TraceEvent]) -> None:
        """POST a batch of events as NDJSON or OTLP JSON spans."""
        if not events or not self.config.url:
            return
        if self.config.otlp:
            self._flush_otlp(events)
        else:
            self._flush_ndjson(events)

    def _flush_ndjson(self, events: list[TraceEvent]) -> None:
        body = "\n".join(e.to_json() for e in events).encode("utf-8")
        headers = {"Content-Type": "application/x-ndjson"}
        headers.update(self.config.headers)
        req = urllib.request.Request(
            self.config.url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status not in (200, 202):
                    sys.stderr.write(
                        f"[stream] POST to {self.config.url} returned {resp.status}\n"
                    )
        except Exception as exc:
            sys.stderr.write(f"[stream] POST to {self.config.url} failed: {exc}\n")

    def _flush_otlp(self, events: list[TraceEvent]) -> None:
        """Convert events to OTLP GenAI spans and POST to /v1/traces."""
        import json as _json
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

        url = self.config.url.rstrip("/") + "/v1/traces"
        for sid, evs in by_session.items():
            meta = _SessionMeta(agent_name=self.config.service_name)
            meta.session_id = sid
            try:
                payload = session_to_otlp_genai(
                    meta, evs, service_name=self.config.service_name
                )
            except Exception:
                continue
            body = _json.dumps(payload).encode("utf-8")
            req_headers = {"Content-Type": "application/json"}
            req_headers.update(self.config.headers)
            req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status not in (200, 202):
                        sys.stderr.write(
                            f"[stream] OTLP POST to {url} returned {resp.status}\n"
                        )
            except Exception as exc:
                sys.stderr.write(f"[stream] OTLP POST to {url} failed: {exc}\n")


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
    store: TraceStore | None = None,
    session_id: str = "",
) -> None:
    action = action or config.on_violation
    # terminal is always shown regardless of action so the operator watching
    # the process sees every alert; file/webhook are additive channels
    _alert_terminal(message)
    if action == "require_approval":
        # Pause the agent and create an approval request
        if state.agent_pid and not state.paused:
            _pause_process(state.agent_pid)
            state.paused = True
        if store and state.session_id:
            try:
                from .approval import create_approval_request, poll_for_decision
                req = create_approval_request(
                    store,
                    session_id=state.session_id,
                    rule_name=notify or "watch-rule",
                    tool_name="",
                    agent_pid=state.agent_pid or 0,
                )
                _alert_terminal(
                    f"[HITL] Approval required for rule '{notify or 'watch-rule'}'. "
                    f"Request ID: {req.request_id}\n"
                    f"  Run: agent-strace approval approve {req.request_id}"
                )
            except Exception:
                pass
        return

    if action in ("file", "kill", "pause"):
        _alert_file(message, config.alert_log)
    if config.webhook_url:
        _alert_webhook(message, config.webhook_url)

    # Route to Slack/Teams via notify.py when env vars or rule hint is set
    try:
        from .notify import notify as _notify, notify_violation as _notify_violation
        import os as _os
        has_slack = bool(_os.environ.get("AGENT_STRACE_SLACK_WEBHOOK"))
        has_teams = bool(_os.environ.get("AGENT_STRACE_TEAMS_WEBHOOK"))
        if has_slack or has_teams:
            _notify_violation(
                rule_name=notify or "watch-rule",
                tool_name="",
                session_id=session_id,
                action=action or "alert",
            )
        elif notify and notify.startswith("slack:") and config.webhook_url:
            # Legacy: route slack: hint through the generic webhook URL
            _alert_webhook(f"[{notify}] {message}", config.webhook_url)
    except Exception:
        pass

    if dry_run:
        _alert_terminal(f"[dry-run] would {action} agent process")
        return

    if action == "pause" and state.agent_pid and not state.paused:
        _alert_terminal(f"Pausing agent process {state.agent_pid} (SIGSTOP)")
        _pause_process(state.agent_pid)
        state.paused = True
    elif action == "kill" and state.agent_pid:
        _alert_terminal(f"Killing agent process {state.agent_pid}")
        # Write watchdog post-mortem JSON before killing
        pm_path: Path | None = None
        if store and session_id:
            pm_path = _write_watchdog_postmortem(store, session_id, state, reason=message)
            if pm_path:
                _alert_terminal(f"Post-mortem written to {pm_path}")
        if config.on_death_cmd:
            _invoke_on_death(config.on_death_cmd, pm_path)
        _kill_process(state.agent_pid)


def _write_watchdog_postmortem(
    store: TraceStore,
    session_id: str,
    state: WatchState,
    reason: str,
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

    pm_path = store._session_dir(session_id) / "watchdog-postmortem.json"
    try:
        pm_path.write_text(json.dumps(pm, indent=2))
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
    """Fire a nanny rule: alert, pause, or kill."""
    msg = f"NannyRule '{rule.name}': {rule.condition} → {rule.action}"
    _dispatch_alert(msg, config, state, action=rule.action, notify=rule.notify, dry_run=dry_run)

    # Auto-generate postmortem on kill
    if rule.action == "kill" and not dry_run:
        pm_path = _write_watchdog_postmortem(store, session_id, state, reason=msg)
        if pm_path:
            _alert_terminal(f"Post-mortem written to {pm_path}")
        on_death = getattr(config, "on_death_cmd", "")
        if on_death:
            _invoke_on_death(on_death, pm_path)


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
    stream_config: StreamConfig | None = None,
) -> None:
    """Watch a session's event stream and fire alerts on violations.

    stream_config: when set, events are pushed in real-time to the configured
    URL as they arrive (push-based streaming).
    """
    events_file = store._session_dir(session_id) / "events.ndjson"
    if not events_file.exists():
        out.write(f"[watch] events file not found: {events_file}\n")
        return

    state = WatchState(start_time=time.time())

    mode = " [dry-run]" if dry_run else ""
    out.write(f"[watch] Monitoring session {session_id[:12]}...{mode}\n")
    if nanny_rules:
        out.write(f"[watch] {len(nanny_rules)} nanny rule(s) active\n")
    if stream_config and stream_config.url:
        out.write(f"[watch] Streaming events to {stream_config.url}\n")
    out.flush()

    streamer: EventStreamer | None = None
    if stream_config and stream_config.url:
        streamer = EventStreamer(stream_config)

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

            # Push event to stream if configured
            if streamer:
                streamer.enqueue(event)

            violations = check_event(event, config, state)
            for msg in violations:
                _dispatch_alert(
                    msg, config, state, dry_run=dry_run,
                    store=store, session_id=session_id,
                )

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
    finally:
        if streamer:
            streamer.stop()


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
        # --timeout is a friendlier alias for --max-duration
        max_duration = getattr(args, "max_duration", 1800)
        timeout_str = getattr(args, "timeout", None)
        if timeout_str:
            try:
                max_duration = _parse_duration(timeout_str)
            except ValueError as exc:
                sys.stderr.write(f"[watch] invalid --timeout value: {exc}\n")
                return 1

        # --budget is a friendlier alias for --max-cost
        max_cost = getattr(args, "max_cost", 10.0)
        budget_str = getattr(args, "budget", None)
        if budget_str is not None:
            try:
                max_cost = float(budget_str)
            except ValueError:
                sys.stderr.write(f"[watch] invalid --budget value: {budget_str!r}\n")
                return 1

        config = WatcherConfig(
            max_retries=getattr(args, "max_retries", 5),
            max_cost_dollars=max_cost,
            max_duration_seconds=max_duration,
            on_violation=getattr(args, "on_violation", "terminal"),
            webhook_url=getattr(args, "webhook", "") or "",
            on_death_cmd=getattr(args, "on_death", "") or "",
            scope_policy=getattr(args, "policy", None) or ".agent-scope.json",
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

    # Build StreamConfig if --stream-to was provided
    stream_cfg: StreamConfig | None = None
    stream_url = getattr(args, "stream_to", None)
    if stream_url:
        stream_cfg = StreamConfig(
            url=stream_url,
            batch_size=getattr(args, "stream_batch_size", 10),
            flush_interval=getattr(args, "stream_flush_interval", 2.0),
            otlp=getattr(args, "stream_otlp", False),
        )

    watch_session(store, full_id, config, nanny_rules=nanny_rules, dry_run=dry_run,
                  stream_config=stream_cfg)
    return 0
