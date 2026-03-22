"""Session explanation: group events into logical phases and detect retries.

Groups a flat event stream into human-readable phases based on user prompt
boundaries, tool call intent, and retry patterns.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, TextIO

from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    name: str
    index: int                          # 1-based phase number
    start_offset: float                 # seconds from session start
    end_offset: float
    events: list[TraceEvent] = field(default_factory=list)
    failed: bool = False
    retry_count: int = 0
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_offset - self.start_offset

    @property
    def event_count(self) -> int:
        return len(self.events)


@dataclass
class ExplainResult:
    session_id: str
    total_duration: float       # seconds
    total_events: int
    phases: list[Phase]
    total_retries: int
    wasted_seconds: float       # time spent in failed phases


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

def _tool_command(event: TraceEvent) -> str:
    """Return a normalised command string for a tool_call event."""
    args = event.data.get("arguments", {})
    name = event.data.get("tool_name", "").lower()
    if name == "bash":
        return str(args.get("command", "")).strip()
    if name in ("read", "write", "edit"):
        return str(args.get("file_path", "")).strip()
    return ""


def _is_error(event: TraceEvent) -> bool:
    return event.event_type == EventType.ERROR


def _phase_label(events: list[TraceEvent], index: int) -> str:
    """Derive a short label for a phase from its events."""
    for e in events:
        if e.event_type == EventType.USER_PROMPT:
            text = e.data.get("prompt", "")
            label = text[:50].replace("\n", " ").strip()
            if len(text) > 50:
                label += "..."
            return label
    # Fall back to dominant tool type
    tool_names = [
        e.data.get("tool_name", "")
        for e in events
        if e.event_type == EventType.TOOL_CALL
    ]
    if tool_names:
        dominant = max(set(tool_names), key=tool_names.count)
        return dominant.capitalize()
    return f"Phase {index}"


def build_phases(events: list[TraceEvent], base_ts: float) -> list[Phase]:
    """Split events into phases, separated by USER_PROMPT boundaries."""
    if not events:
        return []

    phases: list[Phase] = []
    current: list[TraceEvent] = []

    def _flush(idx: int) -> None:
        if not current:
            return
        start = current[0].timestamp - base_ts
        end = current[-1].timestamp - base_ts
        p = Phase(
            name="",
            index=idx,
            start_offset=max(start, 0.0),
            end_offset=max(end, 0.0),
            events=list(current),
        )
        p.name = _phase_label(current, idx)
        _annotate_phase(p)
        phases.append(p)
        current.clear()

    idx = 1
    for event in events:
        # New user prompt starts a new phase (except at the very beginning)
        if event.event_type == EventType.USER_PROMPT and current:
            _flush(idx)
            idx += 1
        current.append(event)

    _flush(idx)
    return phases


def _annotate_phase(phase: Phase) -> None:
    """Populate files, commands, retry count, and failed flag on a phase."""
    seen_commands: list[str] = []
    has_error = False

    for event in phase.events:
        if event.event_type == EventType.ERROR:
            has_error = True

        elif event.event_type == EventType.TOOL_CALL:
            name = event.data.get("tool_name", "").lower()
            args = event.data.get("arguments", {})

            if name == "bash":
                cmd = str(args.get("command", "")).strip()
                if cmd:
                    phase.commands.append(cmd)
                    seen_commands.append(cmd)

            elif name == "read":
                path = str(args.get("file_path", "")).strip()
                if path and path not in phase.files_read:
                    phase.files_read.append(path)

            elif name in ("write", "edit"):
                path = str(args.get("file_path", "")).strip()
                if path and path not in phase.files_written:
                    phase.files_written.append(path)

    # Retry detection: same command appears more than once in this phase
    from collections import Counter
    counts = Counter(seen_commands)
    phase.retry_count = sum(c - 1 for c in counts.values() if c > 1)
    phase.failed = has_error


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_offset(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}m {s:02d}s"


def format_explain(result: ExplainResult, out: TextIO = sys.stdout) -> None:
    """Write a human-readable explain report to *out*."""
    w = out.write

    w(f"\nSession: {result.session_id} "
      f"({_fmt_duration(result.total_duration)}, {result.total_events} events)\n\n")

    for phase in result.phases:
        status = ""
        if phase.failed and phase.retry_count > 0:
            status = " — FAILED"
        elif phase.failed:
            status = " — FAILED"

        time_range = (
            f"{_fmt_offset(phase.start_offset)}–{_fmt_offset(phase.end_offset)}"
        )
        w(f"Phase {phase.index}: {phase.name}{status} "
          f"({time_range}, {phase.event_count} events)\n")

        if phase.files_read:
            preview = phase.files_read[:5]
            suffix = f" +{len(phase.files_read) - 5} more" if len(phase.files_read) > 5 else ""
            w(f"  Read: {', '.join(preview)}{suffix}\n")

        if phase.files_written:
            preview = phase.files_written[:5]
            suffix = f" +{len(phase.files_written) - 5} more" if len(phase.files_written) > 5 else ""
            w(f"  Wrote: {', '.join(preview)}{suffix}\n")

        for i, cmd in enumerate(phase.commands[:8]):
            cmd_display = cmd[:100] + ("..." if len(cmd) > 100 else "")
            suffix = ""
            if i > 0:
                # Check if this looks like a retry of a previous command
                prev_cmds = phase.commands[:i]
                if any(cmd == c for c in prev_cmds):
                    suffix = "  ← retry"
                elif phase.failed and i == len(phase.commands) - 1:
                    suffix = "  ← retry"
            w(f"  Ran: {cmd_display}{suffix}\n")

        if len(phase.commands) > 8:
            w(f"  ... and {len(phase.commands) - 8} more commands\n")

        w("\n")

    # Summary line
    files_read_total = sum(len(p.files_read) for p in result.phases)
    files_written_total = sum(len(p.files_written) for p in result.phases)
    w(f"Files touched: {files_read_total} read, {files_written_total} written\n")

    if result.total_retries > 0:
        pct = (result.wasted_seconds / result.total_duration * 100) if result.total_duration else 0
        w(f"Retries: {result.total_retries} "
          f"(wasted {_fmt_duration(result.wasted_seconds)}, {pct:.0f}% of session)\n")

    w("\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_session(
    store: TraceStore,
    session_id: str,
) -> ExplainResult:
    """Build an ExplainResult for *session_id*."""
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)

    base_ts = events[0].timestamp if events else meta.started_at
    total_duration = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0.0

    phases = build_phases(events, base_ts)

    total_retries = sum(p.retry_count for p in phases)
    wasted_seconds = sum(p.duration for p in phases if p.failed)

    return ExplainResult(
        session_id=meta.session_id,
        total_duration=total_duration,
        total_events=len(events),
        phases=phases,
        total_retries=total_retries,
        wasted_seconds=wasted_seconds,
    )


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_explain(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    full_id = store.find_session(session_id)
    if not full_id:
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    result = explain_session(store, full_id)
    format_explain(result)
    return 0
