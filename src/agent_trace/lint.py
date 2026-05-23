"""Static behaviour analysis: agent-strace lint.

Analyses a session's event stream and flags known bad patterns — tool loops,
reasoning spirals, budget proximity, context saturation, redundant reads,
error-retry loops, and no-output sessions.

Each rule is a pure function:
    rule(events: list[TraceEvent], config: dict) -> list[LintResult]

Rules are independent — a failure in one never prevents others from running.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TextIO

from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class LintLevel:
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class LintResult:
    rule: str
    level: str          # INFO | WARN | ERROR
    message: str
    line_start: int | None = None   # 1-based event index where the issue starts
    line_end: int | None = None


@dataclass
class LintReport:
    session_id: str
    findings: list[LintResult] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return sum(1 for f in self.findings if f.level == LintLevel.ERROR)

    @property
    def warnings(self) -> int:
        return sum(1 for f in self.findings if f.level == LintLevel.WARN)

    @property
    def infos(self) -> int:
        return sum(1 for f in self.findings if f.level == LintLevel.INFO)


# ---------------------------------------------------------------------------
# Default rule configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "tool-loop": {
        "enabled": True,
        "level": LintLevel.WARN,
        "threshold": 5,        # same tool N+ times consecutively
    },
    "reasoning-spiral": {
        "enabled": True,
        "level": LintLevel.WARN,
        "threshold": 3,        # N+ consecutive LLM calls with no tool call
    },
    "budget-proximity": {
        "enabled": True,
        "level": LintLevel.ERROR,
        "threshold": 0.90,     # fraction of budget ceiling
    },
    "context-saturation": {
        "enabled": True,
        "level": LintLevel.INFO,
        "threshold": 0.80,     # fraction of context window
    },
    "redundant-read": {
        "enabled": True,
        "level": LintLevel.INFO,
        "threshold": 3,        # same file read N+ times
    },
    "error-retry-loop": {
        "enabled": True,
        "level": LintLevel.WARN,
        "threshold": 3,        # same tool errored and retried N+ times
    },
    "no-output": {
        "enabled": True,
        "level": LintLevel.WARN,
    },
}

_WRITE_TOOLS = {"write", "edit", "create", "str_replace", "str_replace_based_edit_tool",
                "multiedit", "notebook_edit"}


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _rule_tool_loop(events: list[TraceEvent], cfg: dict) -> list[LintResult]:
    """Same tool called N+ times consecutively."""
    threshold = cfg.get("threshold", 5)
    level = cfg.get("level", LintLevel.WARN)
    results: list[LintResult] = []

    run_tool: str | None = None
    run_start = 0
    run_len = 0

    for i, ev in enumerate(events):
        if ev.event_type != EventType.TOOL_CALL:
            if run_len >= threshold:
                results.append(LintResult(
                    rule="tool-loop",
                    level=level,
                    message=(
                        f"{run_tool!r} called {run_len} times consecutively "
                        f"(events {run_start + 1}–{run_start + run_len}). Possible loop."
                    ),
                    line_start=run_start + 1,
                    line_end=run_start + run_len,
                ))
            run_tool = None
            run_len = 0
            continue

        tool = ev.data.get("tool_name", "")
        if tool == run_tool:
            run_len += 1
        else:
            if run_len >= threshold:
                results.append(LintResult(
                    rule="tool-loop",
                    level=level,
                    message=(
                        f"{run_tool!r} called {run_len} times consecutively "
                        f"(events {run_start + 1}–{run_start + run_len}). Possible loop."
                    ),
                    line_start=run_start + 1,
                    line_end=run_start + run_len,
                ))
            run_tool = tool
            run_start = i
            run_len = 1

    # flush trailing run
    if run_len >= threshold:
        results.append(LintResult(
            rule="tool-loop",
            level=level,
            message=(
                f"{run_tool!r} called {run_len} times consecutively "
                f"(events {run_start + 1}–{run_start + run_len}). Possible loop."
            ),
            line_start=run_start + 1,
            line_end=run_start + run_len,
        ))

    return results


def _rule_reasoning_spiral(events: list[TraceEvent], cfg: dict) -> list[LintResult]:
    """N+ consecutive LLM calls with no tool call between them."""
    threshold = cfg.get("threshold", 3)
    level = cfg.get("level", LintLevel.WARN)
    results: list[LintResult] = []

    llm_types = {EventType.LLM_REQUEST, EventType.LLM_RESPONSE,
                 EventType.ASSISTANT_RESPONSE}

    run_start = 0
    run_len = 0

    for i, ev in enumerate(events):
        if ev.event_type in llm_types:
            if run_len == 0:
                run_start = i
            run_len += 1
        elif ev.event_type == EventType.TOOL_CALL:
            if run_len >= threshold:
                results.append(LintResult(
                    rule="reasoning-spiral",
                    level=level,
                    message=(
                        f"{run_len} consecutive LLM calls with no tool call "
                        f"(events {run_start + 1}–{run_start + run_len}). "
                        "Agent may be over-reasoning."
                    ),
                    line_start=run_start + 1,
                    line_end=run_start + run_len,
                ))
            run_len = 0

    if run_len >= threshold:
        results.append(LintResult(
            rule="reasoning-spiral",
            level=level,
            message=(
                f"{run_len} consecutive LLM calls with no tool call "
                f"(events {run_start + 1}–{run_start + run_len}). "
                "Agent may be over-reasoning."
            ),
            line_start=run_start + 1,
            line_end=run_start + run_len,
        ))

    return results


def _rule_budget_proximity(events: list[TraceEvent], cfg: dict) -> list[LintResult]:
    """Session cost exceeded N% of watchdog budget ceiling."""
    threshold = cfg.get("threshold", 0.90)
    level = cfg.get("level", LintLevel.ERROR)

    # Find budget ceiling from session events (written by watchdog)
    budget: float | None = None
    for ev in events:
        if ev.event_type == EventType.SESSION_START:
            budget = ev.data.get("budget_dollars") or ev.data.get("max_cost_dollars")
            if budget:
                break

    if not budget:
        return []

    # Estimate cost from token events
    total_tokens = sum(
        len(json.dumps(ev.data)) // 4
        for ev in events
        if ev.event_type in {EventType.LLM_REQUEST, EventType.LLM_RESPONSE,
                              EventType.ASSISTANT_RESPONSE, EventType.USER_PROMPT}
    )
    # Use sonnet pricing: $3/M input, $15/M output — rough average $9/M
    estimated_cost = total_tokens * 9.0 / 1_000_000

    if estimated_cost >= budget * threshold:
        pct = estimated_cost / budget * 100
        return [LintResult(
            rule="budget-proximity",
            level=level,
            message=(
                f"Session reached {pct:.0f}% of a ${budget:.2f} budget ceiling "
                f"(estimated ${estimated_cost:.4f} spent). "
                "Consider raising or splitting the task."
            ),
        )]
    return []


def _rule_context_saturation(events: list[TraceEvent], cfg: dict) -> list[LintResult]:
    """Input tokens exceeded N% of model context window."""
    threshold = cfg.get("threshold", 0.80)
    level = cfg.get("level", LintLevel.INFO)
    # Common context window sizes by model hint
    context_window = 200_000  # claude default; conservative

    results: list[LintResult] = []
    cumulative_input = 0

    for i, ev in enumerate(events):
        if ev.event_type in {EventType.LLM_REQUEST, EventType.USER_PROMPT}:
            tokens = len(json.dumps(ev.data)) // 4
            cumulative_input += tokens
            if cumulative_input >= context_window * threshold:
                pct = cumulative_input / context_window * 100
                results.append(LintResult(
                    rule="context-saturation",
                    level=level,
                    message=(
                        f"Input tokens exceeded {threshold * 100:.0f}% of model context window "
                        f"at event {i + 1} (~{cumulative_input:,} tokens, "
                        f"{pct:.0f}% of {context_window:,})."
                    ),
                    line_start=i + 1,
                ))
                break  # report once

    return results


def _rule_redundant_read(events: list[TraceEvent], cfg: dict) -> list[LintResult]:
    """Same file read N+ times in a session."""
    threshold = cfg.get("threshold", 3)
    level = cfg.get("level", LintLevel.INFO)

    from collections import Counter
    read_counts: Counter = Counter()

    for ev in events:
        if ev.event_type not in {EventType.TOOL_CALL, EventType.FILE_READ}:
            continue
        tool = ev.data.get("tool_name", "").lower()
        if tool in ("read", "read_file", "file_read") or ev.event_type == EventType.FILE_READ:
            path = (
                ev.data.get("arguments", {}).get("file_path")
                or ev.data.get("arguments", {}).get("path")
                or ev.data.get("path")
                or ev.data.get("file_path")
            )
            if path:
                read_counts[str(path)] += 1

    results: list[LintResult] = []
    for path, count in read_counts.items():
        if count >= threshold:
            results.append(LintResult(
                rule="redundant-read",
                level=level,
                message=(
                    f"{path!r} read {count} times in this session. "
                    "Consider caching or restructuring the prompt."
                ),
            ))
    return results


def _rule_error_retry_loop(events: list[TraceEvent], cfg: dict) -> list[LintResult]:
    """Same tool errored and was retried N+ times."""
    threshold = cfg.get("threshold", 3)
    level = cfg.get("level", LintLevel.WARN)

    from collections import Counter
    error_counts: Counter = Counter()
    last_tool: str | None = None

    for ev in events:
        if ev.event_type == EventType.TOOL_CALL:
            last_tool = ev.data.get("tool_name", "")
        elif ev.event_type == EventType.ERROR:
            if last_tool:
                error_counts[last_tool] += 1
        elif ev.event_type == EventType.TOOL_RESULT:
            # tool_result with is_error flag
            if ev.data.get("is_error") and last_tool:
                error_counts[last_tool] += 1

    results: list[LintResult] = []
    for tool, count in error_counts.items():
        if count >= threshold:
            results.append(LintResult(
                rule="error-retry-loop",
                level=level,
                message=(
                    f"{tool!r} errored and was retried {count} times. "
                    "Check tool implementation or input validation."
                ),
            ))
    return results


def _rule_no_output(events: list[TraceEvent], cfg: dict) -> list[LintResult]:
    """Session completed with no write or file-modifying tool calls."""
    level = cfg.get("level", LintLevel.WARN)

    has_session_end = any(e.event_type == EventType.SESSION_END for e in events)
    if not has_session_end:
        return []  # session still in progress

    has_write = any(
        e.event_type == EventType.TOOL_CALL
        and e.data.get("tool_name", "").lower() in _WRITE_TOOLS
        for e in events
    )
    has_file_write = any(e.event_type == EventType.FILE_WRITE for e in events)

    if not has_write and not has_file_write:
        return [LintResult(
            rule="no-output",
            level=level,
            message=(
                "Session completed with no Write or file-modifying tool calls. "
                "Agent may have only read/reasoned without producing output."
            ),
        )]
    return []


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

_RULES: dict[str, Callable[[list[TraceEvent], dict], list[LintResult]]] = {
    "tool-loop": _rule_tool_loop,
    "reasoning-spiral": _rule_reasoning_spiral,
    "budget-proximity": _rule_budget_proximity,
    "context-saturation": _rule_context_saturation,
    "redundant-read": _rule_redundant_read,
    "error-retry-loop": _rule_error_retry_loop,
    "no-output": _rule_no_output,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(config_path: str | None) -> dict:
    """Load rule config from file, merging with defaults."""
    config = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}

    if not config_path:
        # Look for .agent-strace-lint.json in cwd
        default = Path(".agent-strace-lint.json")
        if default.exists():
            config_path = str(default)

    if config_path:
        try:
            overrides = json.loads(Path(config_path).read_text())
            for rule_name, rule_cfg in overrides.items():
                if rule_name in config:
                    config[rule_name].update(rule_cfg)
                else:
                    config[rule_name] = rule_cfg
        except Exception as exc:
            sys.stderr.write(f"[lint] Warning: could not load config {config_path}: {exc}\n")

    return config


# ---------------------------------------------------------------------------
# Core lint function
# ---------------------------------------------------------------------------

def lint_session(
    store: TraceStore,
    session_id: str,
    config: dict | None = None,
) -> LintReport:
    """Run all enabled rules against a session. Returns a LintReport."""
    if config is None:
        config = DEFAULT_CONFIG

    events = store.load_events(session_id)
    report = LintReport(session_id=session_id)

    for rule_name, rule_fn in _RULES.items():
        rule_cfg = config.get(rule_name, {})
        if not rule_cfg.get("enabled", True):
            continue
        try:
            findings = rule_fn(events, rule_cfg)
            report.findings.extend(findings)
        except Exception as exc:
            sys.stderr.write(f"[lint] Rule {rule_name!r} failed: {exc}\n")

    return report


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_LEVEL_ORDER = {LintLevel.ERROR: 0, LintLevel.WARN: 1, LintLevel.INFO: 2}


def format_report(report: LintReport, out: TextIO = sys.stdout) -> None:
    """Print a lint report in human-readable form."""
    if not report.findings:
        out.write(f"✓  No issues found in session {report.session_id[:12]}\n")
        return

    # Sort: ERROR first, then WARN, then INFO
    sorted_findings = sorted(report.findings, key=lambda f: _LEVEL_ORDER.get(f.level, 99))

    for f in sorted_findings:
        loc = ""
        if f.line_start is not None:
            loc = f" (event {f.line_start}" + (f"–{f.line_end}" if f.line_end and f.line_end != f.line_start else "") + ")"
        out.write(f"{f.level:<5}  {f.rule:<22}  {f.message}{loc}\n")

    out.write(
        f"\n{report.errors} error(s), {report.warnings} warning(s), "
        f"{report.infos} info(s)."
    )
    if report.errors == 0 and report.warnings == 0:
        out.write(" Exit code: 0\n")
    else:
        out.write(" Use --strict for non-zero exit on warnings.\n")


def format_report_json(report: LintReport) -> str:
    return json.dumps({
        "session_id": report.session_id,
        "errors": report.errors,
        "warnings": report.warnings,
        "infos": report.infos,
        "findings": [
            {
                "rule": f.rule,
                "level": f.level,
                "message": f.message,
                "line_start": f.line_start,
                "line_end": f.line_end,
            }
            for f in report.findings
        ],
    }, indent=2)


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_lint(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    config = _load_config(getattr(args, "config", None))
    strict = getattr(args, "strict", False)
    fmt = getattr(args, "format", "text")
    lint_all = getattr(args, "all", False)
    since_str = getattr(args, "since", None)

    # Resolve session IDs
    session_ids: list[str] = []

    if lint_all:
        sessions = store.list_sessions()
        if since_str:
            import time as _time
            cutoff = _parse_since(since_str)
            sessions = [s for s in sessions if s.started_at >= cutoff]
        session_ids = [s.session_id for s in sessions]
        if not session_ids:
            sys.stderr.write("No sessions found.\n")
            return 1
    else:
        raw_id = getattr(args, "session_id", None)
        if not raw_id:
            latest = store.get_latest_session_id()
            if not latest:
                sys.stderr.write("No sessions found.\n")
                return 1
            raw_id = latest
        full_id = store.find_session(raw_id)
        if not full_id:
            sys.stderr.write(f"Session not found: {raw_id}\n")
            return 1
        session_ids = [full_id]

    reports: list[LintReport] = []
    for sid in session_ids:
        reports.append(lint_session(store, sid, config))

    if fmt == "json":
        if len(reports) == 1:
            sys.stdout.write(format_report_json(reports[0]) + "\n")
        else:
            sys.stdout.write(json.dumps([
                json.loads(format_report_json(r)) for r in reports
            ], indent=2) + "\n")
    else:
        for report in reports:
            if len(reports) > 1:
                sys.stdout.write(f"\n── Session {report.session_id[:12]} ──\n")
            format_report(report, sys.stdout)

    # Exit code
    total_errors = sum(r.errors for r in reports)
    total_warnings = sum(r.warnings for r in reports)

    if strict and (total_errors > 0 or total_warnings > 0):
        return 1
    if total_errors > 0:
        return 1
    return 0


def _parse_since(s: str) -> float:
    """Parse a duration string like '7d', '24h' into a Unix timestamp cutoff."""
    import time as _time
    s = s.strip()
    multipliers = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    if s[-1] in multipliers:
        try:
            return _time.time() - float(s[:-1]) * multipliers[s[-1]]
        except ValueError:
            pass
    # Try ISO date
    try:
        import datetime
        dt = datetime.datetime.fromisoformat(s)
        return dt.timestamp()
    except ValueError:
        pass
    raise ValueError(f"Cannot parse --since value: {s!r}")
