"""Causal chain tracing: why did a specific event happen?

Walks backwards from a target event through causal links to find the
chain of events that led to it, terminating at a user_prompt or
session_start.

Causal link rules:
  - tool_call after error        → caused by the error (retry)
  - file_write after file_read   → read informed the write (same file)
  - tool_call referencing a path from a prior tool_result → result informed call
  - any event after user_prompt  → prompt caused it
  - tool_result links to its tool_call via parent_id
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import TextIO

from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CausalLink:
    event: TraceEvent
    reason: str          # human-readable explanation of the causal link
    event_index: int     # 0-based index in the original event list


@dataclass
class CausalChain:
    target_index: int
    links: list[CausalLink]   # ordered root → target


# ---------------------------------------------------------------------------
# Causal link detection
# ---------------------------------------------------------------------------

def _event_paths(event: TraceEvent) -> set[str]:
    """Extract file paths referenced in an event's data."""
    paths: set[str] = set()
    data = event.data
    for key in ("file_path", "uri", "path"):
        val = data.get(key, "")
        if val:
            paths.add(str(val))
    args = data.get("arguments", {})
    if isinstance(args, dict):
        for key in ("file_path", "path", "uri"):
            val = args.get(key, "")
            if val:
                paths.add(str(val))
    result = data.get("result", "") or data.get("content_preview", "")
    return paths


def _result_text(event: TraceEvent) -> str:
    return str(
        event.data.get("result", "")
        or event.data.get("content_preview", "")
        or event.data.get("text", "")
    )


def build_causal_chain(events: list[TraceEvent], target_index: int) -> CausalChain:
    """Trace backwards from events[target_index] to find the causal chain."""
    if not events or target_index < 0 or target_index >= len(events):
        return CausalChain(target_index=target_index, links=[])

    # Index events by event_id for parent_id lookups
    by_id: dict[str, tuple[int, TraceEvent]] = {
        e.event_id: (i, e) for i, e in enumerate(events)
    }

    visited: set[int] = set()
    chain: list[CausalLink] = []

    def _walk(idx: int, reason: str) -> None:
        if idx in visited or idx < 0:
            return
        visited.add(idx)
        event = events[idx]
        chain.append(CausalLink(event=event, reason=reason, event_index=idx))

        # Terminate at root causes
        if event.event_type in (EventType.USER_PROMPT, EventType.SESSION_START):
            return

        # 1. Follow parent_id link (tool_result → tool_call, llm_response → llm_request)
        if event.parent_id and event.parent_id in by_id:
            parent_idx, parent_event = by_id[event.parent_id]
            _walk(parent_idx, f"← parent of #{idx + 1}")
            return

        # 2. Scan backwards for the most recent causal predecessor
        target_paths = _event_paths(event)
        target_tool = event.data.get("tool_name", "").lower()

        for prev_idx in range(idx - 1, -1, -1):
            prev = events[prev_idx]

            # Error → next tool_call is a retry
            if (prev.event_type == EventType.ERROR
                    and event.event_type == EventType.TOOL_CALL):
                _walk(prev_idx, f"retry after error at #{prev_idx + 1}")
                return

            # tool_result containing a path that this tool_call references
            if (prev.event_type == EventType.TOOL_RESULT and target_paths):
                result_text = _result_text(prev)
                if any(p in result_text for p in target_paths):
                    _walk(prev_idx, f"result at #{prev_idx + 1} referenced path")
                    return

            # file_read → file_write of same file
            if (prev.event_type in (EventType.TOOL_CALL, EventType.FILE_READ)
                    and event.event_type in (EventType.TOOL_CALL, EventType.FILE_WRITE)):
                prev_paths = _event_paths(prev)
                if target_paths & prev_paths:
                    _walk(prev_idx, f"read at #{prev_idx + 1} informed write")
                    return

            # user_prompt is always a root cause
            if prev.event_type == EventType.USER_PROMPT:
                _walk(prev_idx, f"prompt at #{prev_idx + 1} triggered this")
                return

        # Fallback: link to session_start or first event
        _walk(0, "session start")

    _walk(target_index, "target event")

    # Reverse so chain reads root → target
    chain.reverse()
    return CausalChain(target_index=target_index, links=chain)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _event_summary(event: TraceEvent, index: int) -> str:
    etype = event.event_type.value
    data = event.data

    if event.event_type == EventType.TOOL_CALL:
        name = data.get("tool_name", "?")
        args = data.get("arguments", {})
        detail = ""
        if isinstance(args, dict):
            if "command" in args:
                detail = f"  $ {str(args['command'])[:60]}"
            elif "file_path" in args:
                detail = f"  {args['file_path']}"
        return f"#{index + 1:>3}  tool_call: {name}{detail}"

    if event.event_type == EventType.TOOL_RESULT:
        preview = _result_text(event)[:60]
        return f"#{index + 1:>3}  tool_result: {preview}"

    if event.event_type == EventType.ERROR:
        msg = (data.get("message", "") or data.get("error", ""))[:60]
        return f"#{index + 1:>3}  error: {msg}"

    if event.event_type == EventType.USER_PROMPT:
        prompt = data.get("prompt", "")[:60]
        return f"#{index + 1:>3}  user_prompt: \"{prompt}\""

    if event.event_type == EventType.ASSISTANT_RESPONSE:
        text = data.get("text", "")[:60]
        return f"#{index + 1:>3}  assistant_response: \"{text}\""

    if event.event_type in (EventType.FILE_READ, EventType.FILE_WRITE):
        uri = data.get("uri", data.get("file_path", ""))
        return f"#{index + 1:>3}  {etype}: {uri}"

    return f"#{index + 1:>3}  {etype}"


def format_why(
    chain: CausalChain,
    events: list[TraceEvent],
    out: TextIO = sys.stdout,
) -> None:
    w = out.write

    if not chain.links:
        w(f"No causal chain found for event #{chain.target_index + 1}.\n")
        return

    target = events[chain.target_index]
    w(f"\nWhy did event #{chain.target_index + 1} happen?\n\n")
    w(f"  {_event_summary(target, chain.target_index)}\n\n")

    if len(chain.links) <= 1:
        w("  No prior causal events found.\n\n")
        return

    w("Causal chain (root → target):\n\n")
    for i, link in enumerate(chain.links):
        prefix = "  " + ("← " if i > 0 else "  ")
        w(f"{prefix}{_event_summary(link.event, link.event_index)}\n")
        if i < len(chain.links) - 1 and link.reason:
            w(f"       ({link.reason})\n")

    w("\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_why(args: argparse.Namespace) -> int:
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

    events = store.load_events(full_id)
    if not events:
        sys.stderr.write("No events in session.\n")
        return 1

    # event_number is 1-based
    event_number = args.event_number
    if event_number < 1 or event_number > len(events):
        sys.stderr.write(
            f"Event number must be between 1 and {len(events)}.\n"
        )
        return 1

    chain = build_causal_chain(events, event_number - 1)
    format_why(chain, events)
    return 0
