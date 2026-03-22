"""Import Claude Code native JSONL session logs.

Claude Code stores session logs as JSONL files in:
    ~/.claude/projects/<encoded-project-path>/<session-id>.jsonl

Each line is a JSON object with fields like:
    type, message, uuid, parentUuid, sessionId, timestamp, etc.

This module parses those logs and converts them into agent-trace's
TraceEvent/SessionMeta format, so they can be replayed, exported,
and analyzed with all existing agent-trace commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


def cmd_import(args: argparse.Namespace) -> int:
    """Import a Claude Code JSONL session log (CLI handler)."""
    store = TraceStore(args.trace_dir)

    if args.discover:
        sessions = discover_claude_sessions(args.claude_dir)
        if not sessions:
            sys.stderr.write("No Claude Code sessions found.\n")
            return 1

        sys.stdout.write(f"\nFound {len(sessions)} Claude Code sessions:\n\n")
        for s in sessions:
            sys.stdout.write(
                f"  {s['session_id'][:12]}  {s['size_kb']:>6} KB  {s['project']}\n"
            )
        sys.stdout.write("\nImport with: agent-strace import <path-to-session.jsonl>\n")
        return 0

    if not args.path:
        sys.stderr.write("Usage: agent-strace import <session.jsonl>\n")
        sys.stderr.write("       agent-strace import --discover\n")
        return 1

    try:
        session_id = import_jsonl(args.path, store=store)
    except FileNotFoundError as e:
        sys.stderr.write(f"{e}\n")
        return 1

    meta = store.load_meta(session_id)
    events = store.load_events(session_id)

    sys.stderr.write(
        f"Imported session {session_id}\n"
        f"  {meta.tool_calls} tool calls, "
        f"{meta.llm_requests} LLM requests, "
        f"{meta.total_tokens} tokens\n"
        f"  {len(events)} events\n"
        f"  Replay with: agent-strace replay {session_id}\n"
    )
    return 0


def _parse_iso_timestamp(ts: str) -> float:
    """Convert ISO 8601 timestamp to Unix epoch seconds."""
    from datetime import datetime

    if not ts:
        return 0.0
    try:
        # Handle Z suffix
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return dt.timestamp()
    except (ValueError, OSError):
        return 0.0


def _extract_tool_calls(content: list[dict]) -> list[dict[str, Any]]:
    """Extract tool_use blocks from message content."""
    calls = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            calls.append(
                {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                    "caller": block.get("caller", {}),
                }
            )
    return calls


def _extract_tool_results(content: list[dict]) -> list[dict[str, Any]]:
    """Extract tool_result blocks from message content."""
    results = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            text_parts = []
            bc = block.get("content", "")
            if isinstance(bc, str):
                text_parts.append(bc)
            elif isinstance(bc, list):
                for sub in bc:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        text_parts.append(sub.get("text", ""))
            results.append(
                {
                    "tool_use_id": block.get("tool_use_id", ""),
                    "content": "\n".join(text_parts),
                }
            )
    return results


def _extract_text(content: Any) -> str:
    """Extract text from message content (string or content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def import_jsonl(
    path: str | Path,
    store: TraceStore | None = None,
    trace_dir: str = ".agent-traces",
) -> str:
    """Import a Claude Code JSONL session log into agent-trace format.

    Parameters
    ----------
    path : str or Path
        Path to the .jsonl session file.
    store : TraceStore, optional
        Existing store to import into. Creates one if not provided.
    trace_dir : str
        Directory for trace storage (used if store is None).

    Returns
    -------
    str
        The session ID of the imported session.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Session log not found: {path}")

    if store is None:
        store = TraceStore(trace_dir)

    # First pass: extract metadata from first entry
    session_id = ""
    extra_meta: dict[str, str] = {}
    first_ts = 0.0
    last_ts = 0.0
    entries: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip queue operations
            if raw.get("type") in ("queue-operation",):
                continue

            entries.append(raw)

            ts = _parse_iso_timestamp(raw.get("timestamp", ""))
            if first_ts == 0.0 and ts > 0:
                first_ts = ts
            if ts > 0:
                last_ts = ts

            if not session_id:
                session_id = raw.get("sessionId", "")
                extra_meta["git_branch"] = raw.get("gitBranch", "")
                extra_meta["version"] = raw.get("version", "")

    if not session_id:
        session_id = path.stem

    # Create session
    meta = SessionMeta(
        session_id=session_id,
        started_at=first_ts,
        agent_name="claude-code",
        command=f"imported from {path.name} (branch: {extra_meta.get('git_branch', '')}, v{extra_meta.get('version', '')})",
    )
    store.create_session(meta)

    # Second pass: convert entries to TraceEvents
    for raw in entries:
        entry_type = raw.get("type", "")
        ts = _parse_iso_timestamp(raw.get("timestamp", ""))
        msg = raw.get("message", {})
        if not isinstance(msg, dict):
            continue

        content = msg.get("content", "")
        usage = msg.get("usage", {})
        model = msg.get("model", "")
        is_sidechain = raw.get("isSidechain", False)

        # User entry
        if entry_type == "user":
            text = _extract_text(content)

            # Check for tool results in user messages
            if isinstance(content, list):
                tool_results = _extract_tool_results(content)
                for tr in tool_results:
                    _c = tr["content"]
                    preview = (_c[:2000] + "..." if len(_c) > 2000 else _c) if _c else ""
                    event = TraceEvent(
                        event_type=EventType.TOOL_RESULT,
                        timestamp=ts,
                        session_id=meta.session_id,
                        data={
                            "tool_use_id": tr["tool_use_id"],
                            "content_preview": preview,
                        },
                    )
                    store.append_event(meta.session_id, event)

                # Also check toolUseResult — only if no content-block tool_results
                # were already emitted to avoid duplicate TOOL_RESULT events.
                tr_data = raw.get("toolUseResult")
                if tr_data and isinstance(tr_data, dict) and not tool_results:
                    stdout = tr_data.get("stdout", "") or ""
                    stderr = tr_data.get("stderr", "") or ""
                    if stdout or stderr:
                        result_text = stdout[:500]
                        if stderr:
                            result_text += f" [stderr: {stderr[:200]}]"
                        event = TraceEvent(
                            event_type=EventType.TOOL_RESULT,
                            timestamp=ts,
                            session_id=meta.session_id,
                            data={
                                "result": result_text,
                                "content_types": ["text"],
                            },
                        )
                        store.append_event(meta.session_id, event)

            if text and not text.startswith("{"):
                event = TraceEvent(
                    event_type=EventType.USER_PROMPT,
                    timestamp=ts,
                    session_id=meta.session_id,
                    data={"prompt": text[:2000]},
                )
                store.append_event(meta.session_id, event)

        # Assistant entry
        elif entry_type == "assistant":
            text = _extract_text(content)

            # Extract tool calls
            if isinstance(content, list):
                tool_calls = _extract_tool_calls(content)
                for tc in tool_calls:
                    tool_data: dict[str, Any] = {
                        "tool_name": tc["name"],
                        "arguments": tc["input"],
                        "request_id": tc["id"],
                    }
                    # Tag subagent calls
                    if is_sidechain:
                        tool_data["is_sidechain"] = True
                    caller = tc.get("caller", {})
                    if caller.get("type"):
                        tool_data["caller_type"] = caller["type"]
                    # Extract subagent info from Agent tool
                    if tc["name"] == "Agent":
                        inp = tc["input"]
                        if inp.get("subagent_type"):
                            tool_data["subagent_type"] = inp["subagent_type"]

                    event = TraceEvent(
                        event_type=EventType.TOOL_CALL,
                        timestamp=ts,
                        session_id=meta.session_id,
                        data=tool_data,
                    )
                    store.append_event(meta.session_id, event)
                    meta.tool_calls += 1

            # Log assistant text whenever present, including messages that also
            # contain tool calls (Claude Code often emits reasoning text alongside
            # a tool_use block in the same message).
            if text:
                event = TraceEvent(
                    event_type=EventType.ASSISTANT_RESPONSE,
                    timestamp=ts,
                    session_id=meta.session_id,
                    data={
                        "text": text[:2000],
                        "model": model,
                    },
                )
                store.append_event(meta.session_id, event)

            # Track token usage
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                meta.total_tokens += (
                    input_tokens
                    + output_tokens
                    + usage.get("cache_creation_input_tokens", 0)
                    + usage.get("cache_read_input_tokens", 0)
                )
                meta.llm_requests += 1

        # System entry (e.g., turn duration)
        elif entry_type == "system":
            subtype = raw.get("subtype", "")
            if subtype == "turn_duration":
                duration_ms = raw.get("durationMs", 0)
                if duration_ms:
                    meta.total_duration_ms += duration_ms

    # Finalize session
    meta.ended_at = last_ts if last_ts > 0 else meta.started_at
    if meta.total_duration_ms == 0 and meta.ended_at > meta.started_at:
        meta.total_duration_ms = (meta.ended_at - meta.started_at) * 1000

    # Add session end event
    store.append_event(
        meta.session_id,
        TraceEvent(
            event_type=EventType.SESSION_END,
            timestamp=meta.ended_at,
            session_id=meta.session_id,
            data={
                "duration_ms": meta.total_duration_ms,
                "tool_calls": meta.tool_calls,
                "llm_requests": meta.llm_requests,
                "total_tokens": meta.total_tokens,
                "source": str(path),
            },
        ),
    )
    store.update_meta(meta)

    return meta.session_id


def discover_claude_sessions(
    claude_dir: str | Path = "~/.claude",
) -> list[dict[str, Any]]:
    """Discover all Claude Code session JSONL files.

    Parameters
    ----------
    claude_dir : str or Path
        Claude config directory (default: ~/.claude).

    Returns
    -------
    list[dict]
        List of session info dicts with 'path', 'project', 'session_id'.
    """
    claude_dir = Path(claude_dir).expanduser().resolve()
    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        return []

    sessions = []
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue

        project_name = _decode_project_path(project_dir.name)

        for jsonl_file in sorted(project_dir.glob("*.jsonl")):
            # Claude Code names files <session-uuid>.jsonl, so the stem is the ID.
            sessions.append(
                {
                    "path": jsonl_file,
                    "project": project_name,
                    "session_id": jsonl_file.stem,
                    "size_kb": jsonl_file.stat().st_size // 1024,
                }
            )

    return sessions


def _decode_project_path(encoded: str) -> str:
    """Decode Claude Code's encoded project directory name.

    Claude Code encodes a path by replacing each '/' with '-' and prepending '-'.
    e.g. '-home-user-proj-myapp' -> '/home/user/proj/myapp'
    """
    if not encoded.startswith("-"):
        return encoded
    return encoded.replace("-", "/")
