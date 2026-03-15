"""Trace replay and display.

Renders a captured trace as a human-readable timeline.
Supports filtering by event type and time range.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from typing import TextIO

from .models import EventType, TraceEvent, SessionMeta
from .store import TraceStore


# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


EVENT_COLORS = {
    EventType.SESSION_START: C.GREEN,
    EventType.SESSION_END: C.GREEN,
    EventType.TOOL_CALL: C.CYAN,
    EventType.TOOL_RESULT: C.BLUE,
    EventType.LLM_REQUEST: C.MAGENTA,
    EventType.LLM_RESPONSE: C.MAGENTA,
    EventType.FILE_READ: C.YELLOW,
    EventType.FILE_WRITE: C.YELLOW,
    EventType.DECISION: C.WHITE,
    EventType.ERROR: C.RED,
    EventType.USER_PROMPT: C.GREEN,
    EventType.ASSISTANT_RESPONSE: C.MAGENTA,
}

EVENT_ICONS = {
    EventType.SESSION_START: "▶",
    EventType.SESSION_END: "■",
    EventType.TOOL_CALL: "→",
    EventType.TOOL_RESULT: "←",
    EventType.LLM_REQUEST: "⬆",
    EventType.LLM_RESPONSE: "⬇",
    EventType.FILE_READ: "📖",
    EventType.FILE_WRITE: "📝",
    EventType.DECISION: "◆",
    EventType.ERROR: "✗",
    EventType.USER_PROMPT: "👤",
    EventType.ASSISTANT_RESPONSE: "🤖",
}


def _format_timestamp(ts: float, base_ts: float | None = None) -> str:
    """Format timestamp as relative offset or absolute time."""
    if base_ts is not None:
        offset = ts - base_ts
        if offset < 60:
            return f"+{offset:6.2f}s"
        minutes = int(offset // 60)
        seconds = offset % 60
        return f"+{minutes}m{seconds:05.2f}s"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%H:%M:%S.%f")[:-3]


def _format_duration(ms: float | None) -> str:
    if ms is None:
        return ""
    if ms < 1000:
        return f" ({ms:.0f}ms)"
    return f" ({ms / 1000:.2f}s)"


def format_event(event: TraceEvent, base_ts: float | None = None) -> str:
    """Format a single event as a colored terminal line."""
    color = EVENT_COLORS.get(event.event_type, C.WHITE)
    icon = EVENT_ICONS.get(event.event_type, " ")
    ts = _format_timestamp(event.timestamp, base_ts)
    duration = _format_duration(event.duration_ms)

    parts = [
        f"{C.GRAY}{ts}{C.RESET}",
        f"{color}{icon}{C.RESET}",
    ]

    if event.event_type == EventType.TOOL_CALL:
        name = event.data.get("tool_name", "?")
        args = event.data.get("arguments", {})
        arg_keys = ", ".join(args.keys()) if args else ""
        parts.append(f"{color}{C.BOLD}tool_call{C.RESET} {C.WHITE}{name}{C.RESET}")
        if arg_keys:
            parts.append(f"{C.DIM}({arg_keys}){C.RESET}")

    elif event.event_type == EventType.TOOL_RESULT:
        preview = event.data.get("content_preview", "")
        types = event.data.get("content_types", [])
        type_str = ",".join(types) if types else ""
        parts.append(f"{color}tool_result{C.RESET}")
        if type_str:
            parts.append(f"{C.DIM}[{type_str}]{C.RESET}")
        parts.append(f"{duration}")
        if preview:
            parts.append(f"\n{C.GRAY}{'':>14}  {preview[:120]}{C.RESET}")

    elif event.event_type == EventType.LLM_REQUEST:
        model = event.data.get("model", "")
        count = event.data.get("message_count", 0)
        parts.append(f"{color}{C.BOLD}llm_request{C.RESET}")
        if model:
            parts.append(f"{C.DIM}{model}{C.RESET}")
        parts.append(f"{C.DIM}({count} messages){C.RESET}")

    elif event.event_type == EventType.LLM_RESPONSE:
        tokens = event.data.get("total_tokens", 0)
        parts.append(f"{color}llm_response{C.RESET}")
        if tokens:
            parts.append(f"{C.DIM}({tokens} tokens){C.RESET}")
        parts.append(f"{duration}")

    elif event.event_type == EventType.FILE_READ:
        uri = event.data.get("uri", "")
        parts.append(f"{color}file_read{C.RESET} {C.DIM}{uri}{C.RESET}")

    elif event.event_type == EventType.FILE_WRITE:
        uri = event.data.get("uri", "")
        parts.append(f"{color}file_write{C.RESET} {C.DIM}{uri}{C.RESET}")

    elif event.event_type == EventType.ERROR:
        msg = event.data.get("message", "")
        code = event.data.get("code", "")
        parts.append(f"{color}{C.BOLD}error{C.RESET} {C.RED}{msg}{C.RESET}")
        if code:
            parts.append(f"{C.DIM}(code: {code}){C.RESET}")

    elif event.event_type == EventType.SESSION_START:
        cmd = event.data.get("command", [])
        parts.append(f"{color}{C.BOLD}session_start{C.RESET}")
        if cmd:
            parts.append(f"{C.DIM}{' '.join(cmd)}{C.RESET}")

    elif event.event_type == EventType.SESSION_END:
        exit_code = event.data.get("exit_code", "?")
        parts.append(f"{color}{C.BOLD}session_end{C.RESET}")
        parts.append(f"{C.DIM}exit={exit_code}{C.RESET}")
        parts.append(f"{duration}")

    elif event.event_type == EventType.DECISION:
        choice = event.data.get("choice", "")
        reason = event.data.get("reason", "")
        parts.append(f"{color}{C.BOLD}decision{C.RESET} {C.WHITE}{choice}{C.RESET}")
        if reason:
            parts.append(f"\n{C.GRAY}{'':>14}  reason: {reason[:120]}{C.RESET}")

    elif event.event_type == EventType.USER_PROMPT:
        prompt = event.data.get("prompt", "")
        preview = prompt[:150]
        if len(prompt) > 150:
            preview += "..."
        parts.append(f"{color}{C.BOLD}user_prompt{C.RESET}")
        parts.append(f"\n{C.GRAY}{'':>14}  \"{preview}\"{C.RESET}")

    elif event.event_type == EventType.ASSISTANT_RESPONSE:
        text = event.data.get("text", "")
        preview = text[:150]
        if len(text) > 150:
            preview += "..."
        parts.append(f"{color}{C.BOLD}assistant_response{C.RESET}")
        if preview:
            parts.append(f"\n{C.GRAY}{'':>14}  \"{preview}\"{C.RESET}")

    return " ".join(parts)


def format_summary(meta: SessionMeta) -> str:
    """Format session summary."""
    started = datetime.fromtimestamp(meta.started_at, tz=timezone.utc)
    duration = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0

    lines = [
        f"",
        f"{C.BOLD}Session Summary{C.RESET}",
        f"{C.GRAY}{'─' * 50}{C.RESET}",
        f"  Session:    {meta.session_id}",
        f"  Started:    {started.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"  Duration:   {duration:.2f}s",
        f"  Tool calls: {meta.tool_calls}",
        f"  LLM reqs:   {meta.llm_requests}",
        f"  Errors:     {meta.errors}",
    ]

    if meta.agent_name:
        lines.insert(4, f"  Agent:      {meta.agent_name}")
    if meta.command:
        lines.insert(4, f"  Command:    {meta.command}")

    lines.append(f"{C.GRAY}{'─' * 50}{C.RESET}")
    return "\n".join(lines)


def replay_session(
    store: TraceStore,
    session_id: str,
    event_filter: set[EventType] | None = None,
    speed: float = 1.0,
    live: bool = False,
    out: TextIO = sys.stdout,
) -> None:
    """Replay a trace session to the terminal."""
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)

    if not events:
        out.write(f"No events found for session {session_id}\n")
        return

    if event_filter:
        events = [e for e in events if e.event_type in event_filter]

    base_ts = events[0].timestamp if events else None

    out.write(format_summary(meta) + "\n\n")

    prev_ts = base_ts
    for event in events:
        if live and prev_ts and speed > 0:
            delay = (event.timestamp - prev_ts) / speed
            if delay > 0:
                time.sleep(min(delay, 2.0))  # cap at 2s between events
            prev_ts = event.timestamp

        out.write(format_event(event, base_ts) + "\n")

    out.write("\n")


def list_sessions(store: TraceStore, out: TextIO = sys.stdout) -> None:
    """List all captured sessions."""
    sessions = store.list_sessions()

    if not sessions:
        out.write("No traces found.\n")
        return

    out.write(f"\n{C.BOLD}Captured Sessions{C.RESET}\n")
    out.write(f"{C.GRAY}{'─' * 70}{C.RESET}\n")
    out.write(
        f"  {C.DIM}{'ID':<18} {'Started':<22} {'Duration':>10} {'Tools':>6} {'LLM':>5} {'Err':>4}{C.RESET}\n"
    )
    out.write(f"{C.GRAY}{'─' * 70}{C.RESET}\n")

    for meta in sessions:
        started = datetime.fromtimestamp(meta.started_at, tz=timezone.utc)
        duration = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0
        err_color = C.RED if meta.errors > 0 else C.DIM

        out.write(
            f"  {meta.session_id:<18} "
            f"{started.strftime('%Y-%m-%d %H:%M:%S'):<22} "
            f"{duration:>9.1f}s "
            f"{meta.tool_calls:>6} "
            f"{meta.llm_requests:>5} "
            f"{err_color}{meta.errors:>4}{C.RESET}\n"
        )

    out.write(f"{C.GRAY}{'─' * 70}{C.RESET}\n")
    out.write(f"  {len(sessions)} session(s)\n\n")
