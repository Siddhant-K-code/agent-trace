"""Session timeline view.

Renders a structured, chronological view of a session grouped into phases.
Shows tool calls, file operations, LLM requests, errors, retries, and a
wasted-spend callout for failed phases.

Builds on explain.py (phase detection) and cost.py (token/cost estimation).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TextIO

from .cost import DEFAULT_MODEL, PRICING, _dollars, _estimate_tokens, _event_tokens
from .explain import ExplainResult, Phase, explain_session
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimelineEntry:
    """One rendered line in the timeline."""
    offset: float           # seconds from session start
    status: str             # "ok", "fail", "info"
    label: str              # short description
    detail: str = ""        # optional second line
    duration_ms: float | None = None
    tokens: int = 0
    cost: float = 0.0


@dataclass
class TimelinePhase:
    index: int
    name: str
    start_offset: float
    end_offset: float
    entries: list[TimelineEntry] = field(default_factory=list)
    failed: bool = False
    retry_count: int = 0
    wasted_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class TimelineResult:
    session_id: str
    started_at: float
    total_duration: float       # seconds
    total_events: int
    phases: list[TimelinePhase]
    total_cost: float
    wasted_cost: float
    error_count: int
    retry_count: int


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _fmt_offset(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _tool_label(event: TraceEvent) -> tuple[str, str]:
    """Return (label, detail) for a tool_call event."""
    name = event.data.get("tool_name", "?")
    args = event.data.get("arguments", {})
    n = name.lower()

    if n == "bash":
        cmd = str(args.get("command", "")).strip()
        detail = cmd[:120] + ("..." if len(cmd) > 120 else "")
        return f"Run {name}", detail

    if n == "read":
        path = str(args.get("file_path", "")).strip()
        return f"Read {path}", ""

    if n in ("write", "edit"):
        path = str(args.get("file_path", "")).strip()
        lines = args.get("new_string", args.get("content", ""))
        line_count = str(lines).count("\n") + 1 if lines else 0
        detail = f"+{line_count} lines" if line_count > 1 else ""
        return f"Write {path}", detail

    if n == "glob":
        return f"Glob {args.get('pattern', '')}", ""

    if n == "grep":
        return f"Grep /{args.get('pattern', '')}/", str(args.get("path", ""))

    if n == "webfetch":
        return f"Fetch {args.get('url', '')[:80]}", ""

    if n == "websearch":
        return f"Search \"{args.get('query', '')[:80]}\"", ""

    # Generic: show first string arg
    for k, v in args.items():
        if isinstance(v, str) and v:
            return f"{name} {k}={v[:60]}", ""

    return name, ""


def _phase_cost(phase: Phase, model: str) -> float:
    inp = out = 0
    for e in phase.events:
        i, o = _event_tokens(e)
        inp += i
        out += o
    return _dollars(inp, out, model)


def _build_entries(phase: Phase, base_ts: float, model: str) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = []

    # Track tool_call events so we can pair them with tool_results for retry detection
    pending_calls: dict[str, TraceEvent] = {}   # event_id -> tool_call event
    call_counts: dict[str, int] = {}            # normalised key -> count

    for event in phase.events:
        offset = event.timestamp - base_ts
        inp, out = _event_tokens(event)
        tokens = inp + out
        cost = _dollars(inp, out, model)

        if event.event_type == EventType.TOOL_CALL:
            label, detail = _tool_label(event)
            pending_calls[event.event_id] = event

            # Retry detection: same tool+args key seen before
            key = f"{event.data.get('tool_name','')}:{json.dumps(event.data.get('arguments',{}), sort_keys=True)}"
            call_counts[key] = call_counts.get(key, 0) + 1
            retry_tag = f" (attempt {call_counts[key]})" if call_counts[key] > 1 else ""

            entries.append(TimelineEntry(
                offset=offset,
                status="info",
                label=label + retry_tag,
                detail=detail,
                duration_ms=event.duration_ms,
                tokens=tokens,
                cost=cost,
            ))

        elif event.event_type == EventType.TOOL_RESULT:
            # Mark the matching call entry as ok/fail based on result
            result_text = str(event.data.get("result", ""))
            is_error = event.data.get("is_error", False) or "error" in result_text.lower()[:50]
            # Update the last entry for this tool if it's still "info"
            for entry in reversed(entries):
                if entry.status == "info" and entry.label.split(" (attempt")[0]:
                    entry.status = "fail" if is_error else "ok"
                    if event.duration_ms:
                        entry.duration_ms = event.duration_ms
                    break

        elif event.event_type == EventType.LLM_REQUEST:
            model_name = event.data.get("model", "")
            msg_count = event.data.get("message_count", 0)
            detail = f"{model_name}, {msg_count} messages" if model_name else f"{msg_count} messages"
            entries.append(TimelineEntry(
                offset=offset,
                status="info",
                label="LLM request",
                detail=detail,
                tokens=tokens,
                cost=cost,
            ))

        elif event.event_type == EventType.LLM_RESPONSE:
            tok = event.data.get("total_tokens", tokens)
            entries.append(TimelineEntry(
                offset=offset,
                status="ok",
                label="LLM response",
                detail=f"{tok} tokens",
                duration_ms=event.duration_ms,
                tokens=tok,
                cost=cost,
            ))

        elif event.event_type == EventType.FILE_READ:
            uri = event.data.get("uri", "")
            entries.append(TimelineEntry(
                offset=offset,
                status="ok",
                label=f"Read {uri}",
            ))

        elif event.event_type == EventType.FILE_WRITE:
            uri = event.data.get("uri", "")
            entries.append(TimelineEntry(
                offset=offset,
                status="ok",
                label=f"Write {uri}",
            ))

        elif event.event_type == EventType.ERROR:
            msg = event.data.get("message", "") or event.data.get("error", "")
            tool = event.data.get("tool_name", "")
            label = f"Error: {tool}" if tool else "Error"
            entries.append(TimelineEntry(
                offset=offset,
                status="fail",
                label=label,
                detail=str(msg)[:120],
            ))

        elif event.event_type == EventType.DECISION:
            choice = event.data.get("choice", "")
            reason = event.data.get("reason", "")
            entries.append(TimelineEntry(
                offset=offset,
                status="info",
                label=f"Decision: {choice}",
                detail=reason[:80] if reason else "",
            ))

        elif event.event_type == EventType.USER_PROMPT:
            text = event.data.get("prompt", "")
            preview = text[:100].replace("\n", " ")
            if len(text) > 100:
                preview += "..."
            entries.append(TimelineEntry(
                offset=offset,
                status="info",
                label="User prompt",
                detail=f'"{preview}"',
            ))

        elif event.event_type == EventType.ASSISTANT_RESPONSE:
            text = event.data.get("text", "")
            preview = text[:100].replace("\n", " ")
            if len(text) > 100:
                preview += "..."
            entries.append(TimelineEntry(
                offset=offset,
                status="ok",
                label="Response",
                detail=f'"{preview}"' if preview else "",
                tokens=tokens,
                cost=cost,
            ))

    return entries


def build_timeline(
    store: TraceStore,
    session_id: str,
    model: str = DEFAULT_MODEL,
) -> TimelineResult:
    """Build a structured timeline for *session_id*."""
    explain = explain_session(store, session_id)
    meta = store.load_meta(session_id)
    base_ts = meta.started_at if meta else (explain.phases[0].events[0].timestamp if explain.phases else 0.0)

    total_cost = 0.0
    wasted_cost = 0.0
    error_count = 0
    retry_count = 0
    phases: list[TimelinePhase] = []

    for phase in explain.phases:
        phase_cost = _phase_cost(phase, model)
        entries = _build_entries(phase, base_ts, model)

        # Count errors and retries from entries
        phase_errors = sum(1 for e in entries if e.status == "fail")
        # Retry count: entries with "(attempt N)" where N > 1
        phase_retries = sum(
            1 for e in entries
            if "(attempt " in e.label and not e.label.endswith("(attempt 1)")
        )

        wasted = phase_cost if phase.failed else 0.0

        phases.append(TimelinePhase(
            index=phase.index,
            name=phase.name,
            start_offset=phase.start_offset,
            end_offset=phase.end_offset,
            entries=entries,
            failed=phase.failed,
            retry_count=phase_retries,
            wasted_cost=wasted,
            total_cost=phase_cost,
        ))

        total_cost += phase_cost
        wasted_cost += wasted
        error_count += phase_errors
        retry_count += phase_retries

    return TimelineResult(
        session_id=session_id,
        started_at=base_ts,
        total_duration=explain.total_duration,
        total_events=explain.total_events,
        phases=phases,
        total_cost=total_cost,
        wasted_cost=wasted_cost,
        error_count=error_count,
        retry_count=retry_count,
    )


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------

def _fmt_duration(ms: float | None) -> str:
    if ms is None:
        return ""
    if ms < 1000:
        return f" ({ms:.0f}ms)"
    return f" ({ms / 1000:.2f}s)"


def _status_icon(status: str) -> str:
    return {"ok": "✓", "fail": "✗", "info": "·"}.get(status, "·")


def format_timeline(result: TimelineResult, out: TextIO = sys.stdout) -> None:
    """Write a human-readable timeline to *out*."""
    w = out.write

    started = datetime.fromtimestamp(result.started_at, tz=timezone.utc)
    dur_m = int(result.total_duration) // 60
    dur_s = int(result.total_duration) % 60
    dur_str = f"{dur_m}m {dur_s:02d}s" if dur_m else f"{dur_s}s"

    cost_str = f"${result.total_cost:.4f}" if result.total_cost else ""
    err_str = f"{result.error_count} error{'s' if result.error_count != 1 else ''}" if result.error_count else ""
    meta_parts = [p for p in [dur_str, cost_str, err_str] if p]

    w(f"\nSession: {result.session_id} | "
      f"{started.strftime('%Y-%m-%d %H:%M')} | "
      f"{' | '.join(meta_parts)}\n\n")

    for phase in result.phases:
        time_range = f"{_fmt_offset(phase.start_offset)}–{_fmt_offset(phase.end_offset)}"
        failed_tag = " — FAILED" if phase.failed else ""
        cost_tag = f"  ${phase.total_cost:.4f}" if phase.total_cost else ""
        w(f"Phase {phase.index}: {phase.name}{failed_tag} ({time_range}){cost_tag}\n")

        for entry in phase.entries:
            icon = _status_icon(entry.status)
            dur = _fmt_duration(entry.duration_ms)
            cost = f"  ${entry.cost:.4f}" if entry.cost >= 0.0001 else ""
            w(f"  {icon} {entry.label}{dur}{cost}\n")
            if entry.detail:
                w(f"      {entry.detail}\n")

        if phase.retry_count:
            w(f"\n  ⚠ {phase.retry_count} retr{'ies' if phase.retry_count != 1 else 'y'} in this phase\n")

        w("\n")

    if result.wasted_cost > 0 and result.total_cost > 0:
        wasted_pct = result.wasted_cost / result.total_cost * 100
        w(f"⚠ Wasted spend: {result.retry_count} retr{'ies' if result.retry_count != 1 else 'y'} "
          f"on failed phases = ~${result.wasted_cost:.4f} ({wasted_pct:.0f}% of session cost)\n\n")


def format_timeline_json(result: TimelineResult, out: TextIO = sys.stdout) -> None:
    """Write the timeline as JSON to *out*."""
    def _phase_dict(p: TimelinePhase) -> dict:
        return {
            "index": p.index,
            "name": p.name,
            "start_offset": round(p.start_offset, 3),
            "end_offset": round(p.end_offset, 3),
            "failed": p.failed,
            "retry_count": p.retry_count,
            "total_cost": round(p.total_cost, 6),
            "wasted_cost": round(p.wasted_cost, 6),
            "entries": [
                {
                    "offset": round(e.offset, 3),
                    "status": e.status,
                    "label": e.label,
                    **({"detail": e.detail} if e.detail else {}),
                    **({"duration_ms": e.duration_ms} if e.duration_ms is not None else {}),
                    **({"tokens": e.tokens} if e.tokens else {}),
                    **({"cost": round(e.cost, 6)} if e.cost else {}),
                }
                for e in p.entries
            ],
        }

    out.write(json.dumps({
        "session_id": result.session_id,
        "started_at": result.started_at,
        "total_duration": round(result.total_duration, 3),
        "total_events": result.total_events,
        "total_cost": round(result.total_cost, 6),
        "wasted_cost": round(result.wasted_cost, 6),
        "error_count": result.error_count,
        "retry_count": result.retry_count,
        "phases": [_phase_dict(p) for p in result.phases],
    }, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_timeline(args: argparse.Namespace) -> int:
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

    model = getattr(args, "model", DEFAULT_MODEL) or DEFAULT_MODEL
    fmt = getattr(args, "format", "text") or "text"

    result = build_timeline(store, full_id, model=model)

    if fmt == "json":
        format_timeline_json(result)
    else:
        format_timeline(result)

    return 0
