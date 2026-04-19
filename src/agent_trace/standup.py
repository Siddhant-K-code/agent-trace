"""Agent standup: plain-English narrative of what the agent did.

Generates a structured standup report from a session's trace data.
Uses the existing explain.py and postmortem.py analysis — no LLM call
required for the structured output. An optional --llm flag can be added
later to generate a narrative summary.

Usage:
    agent-strace standup
    agent-strace standup --session <id>
    agent-strace standup --no-llm
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import TextIO

from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ApproachAttempt:
    description: str
    abandoned: bool
    failure_reason: str = ""
    retries: int = 0


@dataclass
class UncertaintySignal:
    description: str
    location: str = ""   # file:line if available


@dataclass
class ReviewItem:
    description: str
    severity: str = "normal"   # "normal" | "high"


@dataclass
class StandupReport:
    session_id: str
    duration_seconds: float
    cost_usd: float
    # What the agent did
    files_read: int
    files_modified: int
    files_modified_list: list[str]
    new_dependencies: list[str]
    approaches: list[ApproachAttempt]
    # What it was uncertain about
    uncertainties: list[UncertaintySignal]
    # What to review
    review_items: list[ReviewItem]
    # Stats
    tool_calls: int
    llm_requests: int
    context_resets: int
    retries: int
    errors: int


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _extract_new_deps(events: list[TraceEvent]) -> list[str]:
    """Detect new package installations from tool calls."""
    deps: list[str] = []
    for e in events:
        if e.event_type != EventType.TOOL_CALL:
            continue
        tool = e.data.get("tool_name", "").lower()
        args = e.data.get("arguments", {}) or {}
        cmd = str(args.get("command") or args.get("cmd") or "")
        if tool in ("bash", "run", "execute", "shell") or "command" in args:
            if any(pm in cmd for pm in ("npm install", "pip install", "yarn add",
                                         "go get", "cargo add", "gem install")):
                # Extract package name(s) from command
                parts = cmd.split()
                for i, part in enumerate(parts):
                    if part in ("install", "add", "get") and i + 1 < len(parts):
                        pkg = parts[i + 1]
                        if not pkg.startswith("-"):
                            deps.append(pkg)
    return list(dict.fromkeys(deps))  # deduplicate, preserve order


def _extract_uncertainties(events: list[TraceEvent]) -> list[UncertaintySignal]:
    """Find TODO/FIXME/uncertainty signals in written file content."""
    signals: list[UncertaintySignal] = []
    uncertainty_patterns = [
        "TODO", "FIXME", "not sure", "unclear", "might", "may need",
        "edge case", "double-check", "verify", "assumption",
    ]
    for e in events:
        if e.event_type != EventType.TOOL_CALL:
            continue
        tool = e.data.get("tool_name", "").lower()
        if tool not in ("write", "edit", "str_replace", "create"):
            continue
        args = e.data.get("arguments", {}) or {}
        content = str(args.get("new_str") or args.get("content") or "")
        path = str(args.get("file_path") or args.get("path") or "")
        for line_no, line in enumerate(content.splitlines(), 1):
            for pat in uncertainty_patterns:
                if pat.lower() in line.lower():
                    signals.append(UncertaintySignal(
                        description=line.strip()[:100],
                        location=f"{path}:{line_no}" if path else "",
                    ))
                    break
        if len(signals) >= 5:
            break
    return signals


def _extract_approaches(events: list[TraceEvent]) -> list[ApproachAttempt]:
    """Detect multiple approaches from retry patterns and error sequences."""
    approaches: list[ApproachAttempt] = []
    consecutive_errors = 0
    last_tool = ""
    retry_count = 0

    for e in events:
        if e.event_type == EventType.TOOL_RESULT:
            content = str(e.data.get("content", ""))
            is_error = e.data.get("is_error", False) or any(
                kw in content.lower() for kw in ("error", "failed", "traceback")
            )
            if is_error:
                consecutive_errors += 1
                retry_count += 1
            else:
                if consecutive_errors >= 2:
                    # Detected an abandoned approach
                    approaches.append(ApproachAttempt(
                        description=f"Approach using {last_tool or 'unknown tool'}",
                        abandoned=True,
                        failure_reason=f"Failed {consecutive_errors} times",
                        retries=consecutive_errors,
                    ))
                consecutive_errors = 0

        if e.event_type == EventType.TOOL_CALL:
            last_tool = e.data.get("tool_name", "")

    if not approaches:
        approaches.append(ApproachAttempt(
            description="Single approach (no retries detected)",
            abandoned=False,
            retries=retry_count,
        ))

    return approaches


def _extract_review_items(
    events: list[TraceEvent],
    files_modified: list[str],
    new_deps: list[str],
) -> list[ReviewItem]:
    """Build a list of things the reviewer should check."""
    items: list[ReviewItem] = []

    # New dependencies are always worth reviewing
    for dep in new_deps:
        items.append(ReviewItem(
            description=f"New dependency added: {dep}",
            severity="high",
        ))

    # Files with large changes
    for e in events:
        if e.event_type != EventType.TOOL_CALL:
            continue
        tool = e.data.get("tool_name", "").lower()
        if tool not in ("write", "edit", "str_replace", "create"):
            continue
        args = e.data.get("arguments", {}) or {}
        content = str(args.get("new_str") or args.get("content") or "")
        path = str(args.get("file_path") or args.get("path") or "")
        if len(content.splitlines()) > 100 and path:
            items.append(ReviewItem(
                description=f"Large change: {path} ({len(content.splitlines())} lines)",
                severity="high",
            ))

    # Behavior-changing patterns
    behavior_patterns = [
        ("retry", "Retry logic changed"),
        ("timeout", "Timeout value changed"),
        ("auth", "Authentication logic modified"),
        ("password", "Password/credential handling changed"),
        ("migration", "Database migration included"),
    ]
    all_content = " ".join(
        str(e.data.get("arguments", {}).get("new_str") or
            e.data.get("arguments", {}).get("content") or "")
        for e in events if e.event_type == EventType.TOOL_CALL
    ).lower()
    for pattern, description in behavior_patterns:
        if pattern in all_content:
            items.append(ReviewItem(description=description, severity="normal"))

    return items[:8]  # cap at 8 items


def _count_context_resets(events: list[TraceEvent]) -> int:
    import time as _time
    resets = 0
    last_ts: float | None = None
    for e in events:
        if e.event_type == EventType.LLM_REQUEST:
            if last_ts is not None and (e.timestamp - last_ts) > 120:
                resets += 1
            last_ts = e.timestamp
    return resets


def analyse_standup(store: TraceStore, session_id: str) -> StandupReport:
    """Build a standup report from a session's trace data."""
    from .cost import estimate_cost

    meta = store.load_meta(session_id)
    events = store.load_events(session_id)

    try:
        cost = estimate_cost(store, session_id).total_cost
    except Exception:
        cost = 0.0

    # Files read and modified
    files_read_set: set[str] = set()
    files_modified_set: set[str] = set()
    for e in events:
        if e.event_type == EventType.TOOL_CALL:
            tool = e.data.get("tool_name", "").lower()
            args = e.data.get("arguments", {}) or {}
            path = str(args.get("file_path") or args.get("path") or "")
            if path:
                if tool in ("read", "read_file", "view"):
                    files_read_set.add(path)
                elif tool in ("write", "edit", "create", "str_replace"):
                    files_modified_set.add(path)
        elif e.event_type == EventType.FILE_READ:
            path = str(e.data.get("path") or "")
            if path:
                files_read_set.add(path)
        elif e.event_type == EventType.FILE_WRITE:
            path = str(e.data.get("path") or "")
            if path:
                files_modified_set.add(path)

    # Duration
    ts_list = [e.timestamp for e in events]
    duration = (max(ts_list) - min(ts_list)) if len(ts_list) >= 2 else 0.0

    # Retries
    retries = sum(
        1 for e in events
        if e.event_type == EventType.TOOL_RESULT and e.data.get("is_error", False)
    )

    new_deps = _extract_new_deps(events)
    uncertainties = _extract_uncertainties(events)
    approaches = _extract_approaches(events)
    files_modified_list = sorted(files_modified_set)
    review_items = _extract_review_items(events, files_modified_list, new_deps)

    return StandupReport(
        session_id=session_id,
        duration_seconds=duration,
        cost_usd=cost,
        files_read=len(files_read_set),
        files_modified=len(files_modified_set),
        files_modified_list=files_modified_list,
        new_dependencies=new_deps,
        approaches=approaches,
        uncertainties=uncertainties,
        review_items=review_items,
        tool_calls=meta.tool_calls,
        llm_requests=meta.llm_requests,
        context_resets=_count_context_resets(events),
        retries=retries,
        errors=meta.errors,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_standup(report: StandupReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    sep = "─" * 55

    dur_m = int(report.duration_seconds // 60)
    dur_s = int(report.duration_seconds % 60)
    dur_str = f"{dur_m}m {dur_s:02d}s" if dur_m else f"{dur_s}s"

    w(f"\nSession: {report.session_id[:16]}  ({dur_str} · ${report.cost_usd:.4f})\n")
    w(f"{sep}\n\n")

    # What the agent did
    w("What the agent did:\n")
    w(f"  - Read {report.files_read} file(s), modified {report.files_modified}\n")
    if report.files_modified_list:
        for path in report.files_modified_list[:6]:
            w(f"      {path}\n")
        if len(report.files_modified_list) > 6:
            w(f"      ... and {len(report.files_modified_list) - 6} more\n")

    for approach in report.approaches:
        if approach.abandoned:
            w(f"  - Tried and abandoned: {approach.description} "
              f"({approach.failure_reason})\n")
        elif approach.retries > 0:
            w(f"  - {approach.description} ({approach.retries} retries)\n")
        else:
            w(f"  - {approach.description}\n")

    if report.new_dependencies:
        w(f"  - Added {len(report.new_dependencies)} new dependency(ies): "
          f"{', '.join(report.new_dependencies[:4])}\n")

    w("\n")

    # What it was uncertain about
    if report.uncertainties:
        w("What it was uncertain about:\n")
        for u in report.uncertainties[:4]:
            loc = f"  [{u.location}]" if u.location else ""
            w(f"  - {u.description[:80]}{loc}\n")
        w("\n")

    # What to review
    if report.review_items:
        w("What to review carefully:\n")
        for item in report.review_items:
            icon = "❗" if item.severity == "high" else "  "
            w(f"  {icon} {item.description}\n")
        w("\n")

    # Stats
    w(f"Stats: {report.tool_calls} tool calls · "
      f"{report.context_resets} context reset(s) · "
      f"{report.retries} retries · "
      f"{report.errors} error(s)\n")
    w(f"{sep}\n\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_standup(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

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

    report = analyse_standup(store, full_id)
    format_standup(report)
    return 0
