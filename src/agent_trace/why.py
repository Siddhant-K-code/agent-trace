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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TextIO

from .event_data import require_object_event_data
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

class _NonObjectData(Exception):
    """The causal walk read an event whose data is not an object."""

    def __init__(self, event: TraceEvent) -> None:
        super().__init__()
        self.event = event


def _data(event: TraceEvent) -> Mapping:
    """Return the data of an event the causal walk reads."""
    if not isinstance(event.data, Mapping):
        raise _NonObjectData(event)
    return event.data


def _event_paths(event: TraceEvent) -> set[str]:
    """Extract file paths referenced in an event's data."""
    paths: set[str] = set()
    data = _data(event)
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
    return paths


def _result_text(data: Mapping) -> str:
    return str(
        data.get("result", "")
        or data.get("content_preview", "")
        or data.get("text", "")
    )


def build_causal_chain(events: list[TraceEvent], target_index: int) -> CausalChain:
    """Trace backwards from events[target_index] to find the causal chain.

    Raises NonObjectEventDataError when the walk reaches an event whose data
    it reads and that data is not an object.
    """
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

        for prev_idx in range(idx - 1, -1, -1):
            prev = events[prev_idx]

            # Error → retry: only fire when the target tool_call repeats the
            # same tool name AND command as the call that caused the error.
            # This avoids attributing unrelated commands (e.g. git status
            # after a failed pytest) as retries.
            if (prev.event_type == EventType.ERROR
                    and event.event_type == EventType.TOOL_CALL):
                # Find the tool_call that caused the error (skip tool_results)
                causing_idx = prev_idx - 1
                while causing_idx >= 0 and events[causing_idx].event_type == EventType.TOOL_RESULT:
                    causing_idx -= 1
                if causing_idx >= 0 and events[causing_idx].event_type == EventType.TOOL_CALL:
                    causing = _data(events[causing_idx])
                    target = _data(event)
                    same_tool = causing.get("tool_name", "") == target.get("tool_name", "")
                    # For Bash, also require the same command string
                    causing_args = causing.get("arguments", {}) or {}
                    target_args = target.get("arguments", {}) or {}
                    if same_tool and causing.get("tool_name", "").lower() == "bash":
                        same_cmd = causing_args.get("command", "") == target_args.get("command", "")
                        if same_cmd:
                            _walk(prev_idx, f"retry after error at #{prev_idx + 1}")
                            return
                    elif same_tool:
                        _walk(prev_idx, f"retry after error at #{prev_idx + 1}")
                        return

            # tool_result containing a path that this tool_call references
            if (prev.event_type == EventType.TOOL_RESULT and target_paths):
                result_text = _result_text(_data(prev))
                if any(p in result_text for p in target_paths):
                    _walk(prev_idx, f"result at #{prev_idx + 1} referenced path")
                    return

            # file_read → file_write of same file: only match actual write ops
            if (prev.event_type in (EventType.TOOL_CALL, EventType.FILE_READ)
                    and event.event_type == EventType.FILE_WRITE):
                prev_paths = _event_paths(prev)
                if target_paths & prev_paths:
                    _walk(prev_idx, f"read at #{prev_idx + 1} informed write")
                    return

            # Write tool calls (Write/Edit) via tool_call event type
            if (prev.event_type in (EventType.TOOL_CALL, EventType.FILE_READ)
                    and event.event_type == EventType.TOOL_CALL):
                tool = _data(event).get("tool_name", "").lower()
                if tool in ("write", "edit", "create"):
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

    try:
        _walk(target_index, "target event")
    except _NonObjectData as exc:
        position = next(i for i, event in enumerate(events) if event is exc.event)
        require_object_event_data(events, indexes=[position])
        raise

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
        preview = _result_text(data)[:60]
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


# _event_summary reads the data of events of these types.
SUMMARY_DATA_EVENT_TYPES = frozenset({
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.ERROR,
    EventType.USER_PROMPT,
    EventType.ASSISTANT_RESPONSE,
    EventType.FILE_READ,
    EventType.FILE_WRITE,
})


def _summarized_indexes(chain: CausalChain) -> list[int]:
    """Return the indexes of the events format_why summarizes."""
    if not chain.links:
        return []
    indexes = {chain.target_index}
    if len(chain.links) > 1:
        indexes.update(link.event_index for link in chain.links)
    return sorted(indexes)


def format_why(
    chain: CausalChain,
    events: list[TraceEvent],
    out: TextIO = sys.stdout,
) -> None:
    """Write the causal chain.

    Raises NonObjectEventDataError before writing anything when a summarized
    event's data is not an object.
    """
    require_object_event_data(
        events,
        event_types=SUMMARY_DATA_EVENT_TYPES,
        indexes=_summarized_indexes(chain),
    )
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
