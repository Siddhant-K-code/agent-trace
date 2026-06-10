"""Claude Code hooks integration.

Captures every tool call Claude Code makes — not just MCP calls.
Uses Claude Code's hooks system (PreToolUse, PostToolUse, SessionStart,
SessionEnd) to trace Bash, Edit, Write, Read, Agent, and all other tools.

Usage:
    # In .claude/settings.json or ~/.claude/settings.json:
    {
      "hooks": {
        "PreToolUse": [{
          "matcher": "",
          "hooks": [{"type": "command", "command": "agent-strace hook pre-tool"}]
        }],
        "PostToolUse": [{
          "matcher": "",
          "hooks": [{"type": "command", "command": "agent-strace hook post-tool"}]
        }],
        "PostToolUseFailure": [{
          "matcher": "",
          "hooks": [{"type": "command", "command": "agent-strace hook post-tool-failure"}]
        }],
        "SessionStart": [{
          "hooks": [{"type": "command", "command": "agent-strace hook session-start"}]
        }],
        "SessionEnd": [{
          "hooks": [{"type": "command", "command": "agent-strace hook session-end"}]
        }]
      }
    }

The hook script reads JSON from stdin (provided by Claude Code), converts
it to a TraceEvent, and appends it to the active session's trace store.

Session state is tracked via a file at .agent-traces/.active-session so
that PreToolUse and PostToolUse hooks (which run as separate processes)
can find the current session.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .models import EventType, SessionMeta, TraceEvent
from .attribution import collect_attribution
from .redact import redact_data, redact_data_with_status, redaction_enabled
from .store import TraceStore

# Pending tool calls are tracked in a file so separate hook processes
# can link PostToolUse back to PreToolUse for latency measurement.
# The file is keyed by event_id (not tool name) so concurrent calls to
# the same tool don't overwrite each other.
_PENDING_FILE = ".pending-calls.json"

# Each Claude Code session gets its own state files, derived from the
# session ID passed in the SessionStart payload.  This prevents concurrent
# agents sharing the same AGENT_TRACE_DIR from corrupting each other.
_CLAUDE_SESSION_ID_ENV = "AGENT_TRACE_CLAUDE_SESSION_ID"


def _get_store_dir() -> str:
    return os.environ.get("AGENT_TRACE_DIR", ".agent-traces")


def _get_store() -> TraceStore:
    return TraceStore(_get_store_dir())


def _get_remote_endpoint() -> str:
    """Return AGENT_STRACE_ENDPOINT if set, else empty string."""
    return os.environ.get("AGENT_STRACE_ENDPOINT", "").rstrip("/")


def _write_event(store: TraceStore, session_id: str, event: TraceEvent) -> None:
    """Write an event to local store or remote collector, depending on env."""
    endpoint = _get_remote_endpoint()
    if endpoint:
        if redaction_enabled():
            event.data, changed = redact_data_with_status(event.data)
            if changed:
                if not event.redacted:
                    sys.stderr.write("agent-strace: redacted secrets from trace event\n")
                event.redacted = True
        from .server import send_event_to_endpoint
        send_event_to_endpoint(event, endpoint)
    else:
        store.append_event(session_id, event)


def _state_suffix() -> str:
    """Return a filename suffix scoped to the current Claude Code session.

    Claude Code sets AGENT_TRACE_CLAUDE_SESSION_ID (written by handle_session_start).
    Using it as a suffix isolates concurrent agents that share AGENT_TRACE_DIR.
    """
    sid = os.environ.get(_CLAUDE_SESSION_ID_ENV, "")
    return f".{sid}" if sid else ""


def _active_session_path() -> Path:
    return Path(_get_store_dir()) / f".active-session{_state_suffix()}"


def _pending_calls_path() -> Path:
    suffix = _state_suffix()
    name = _PENDING_FILE.replace(".json", f"{suffix}.json")
    return Path(_get_store_dir()) / name


def _read_active_session() -> str | None:
    path = _active_session_path()
    if path.exists():
        return path.read_text().strip()
    return None


def _resolve_session_id(input_data: dict) -> str | None:
    """Return the agent-trace session ID for this hook invocation.

    Claude Code passes session_id in every hook payload. We derive our
    session ID from it the same way handle_session_start does (first 16
    chars). This is the reliable path — it works even when the env var
    AGENT_TRACE_CLAUDE_SESSION_ID is not set (i.e. in all hook processes
    other than the SessionStart process that wrote it).

    Falls back to _read_active_session() for hooks that don't carry a
    session_id (e.g. SessionEnd).
    """
    raw = input_data.get("session_id", "")
    if raw:
        return raw[:16]
    return _read_active_session()


def _write_active_session(session_id: str) -> None:
    path = _active_session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(session_id)


def _clear_active_session() -> None:
    path = _active_session_path()
    if path.exists():
        path.unlink()


def _read_pending_calls() -> dict:
    path = _pending_calls_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _write_pending_calls(calls: dict) -> None:
    path = _pending_calls_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calls))


def _read_stdin() -> dict:
    """Read JSON from stdin (provided by Claude Code)."""
    try:
        data = sys.stdin.read()
        if not data.strip():
            return {}
        return json.loads(data)
    except (json.JSONDecodeError, OSError):
        return {}


def _should_redact() -> bool:
    return (
        redaction_enabled()
        and os.environ.get("AGENT_TRACE_REDACT", "").lower() in ("1", "true", "yes")
    )


def handle_session_start(input_data: dict) -> None:
    """Handle SessionStart hook event."""
    store = _get_store()
    redact = _should_redact()

    session_id = input_data.get("session_id", "")
    attr = collect_attribution()
    meta = SessionMeta(
        agent_name="claude-code",
        command=f"claude-code ({input_data.get('source', 'startup')})",
        attribution=attr.to_dict(),
    )
    # Use Claude Code's session ID as part of our session ID for correlation
    if session_id:
        meta.session_id = session_id[:16]

    store.create_session(meta)

    if session_id:
        os.environ[_CLAUDE_SESSION_ID_ENV] = session_id

    _write_active_session(meta.session_id)
    _write_pending_calls({})

    event_data = {
        "mode": "claude-code-hooks",
        "source": input_data.get("source", "startup"),
        "model": input_data.get("model", ""),
    }
    if redact:
        event_data = redact_data(event_data)

    _write_event(store, 
        meta.session_id,
        TraceEvent(
            event_type=EventType.SESSION_START,
            session_id=meta.session_id,
            data=event_data,
        ),
    )


def handle_session_end(input_data: dict) -> None:
    """Handle SessionEnd hook event."""
    store = _get_store()
    session_id = _resolve_session_id(input_data)
    if not session_id:
        return

    meta = store.load_meta(session_id)
    if meta:
        meta.ended_at = time.time()
        meta.total_duration_ms = (meta.ended_at - meta.started_at) * 1000

        _write_event(store, 
            session_id,
            TraceEvent(
                event_type=EventType.SESSION_END,
                session_id=session_id,
                data={"duration_ms": meta.total_duration_ms},
            ),
        )
        store.update_meta(meta)

    _clear_active_session()


def handle_pre_tool(input_data: dict) -> None:
    """Handle PreToolUse hook event. Logs tool_call and tracks pending calls."""
    store = _get_store()
    session_id = _resolve_session_id(input_data)
    if not session_id:
        return

    redact = _should_redact()
    tool_name = input_data.get("tool_name", "unknown")
    tool_input = input_data.get("tool_input", {})

    event_data = {
        "tool_name": tool_name,
        "arguments": tool_input,
    }
    if redact:
        event_data = redact_data(event_data)

    event = TraceEvent(
        event_type=EventType.TOOL_CALL,
        session_id=session_id,
        data=event_data,
    )
    _write_event(store, session_id, event)

    # Update meta
    meta = store.load_meta(session_id)
    if meta:
        meta.tool_calls += 1
        store.update_meta(meta)

    # Track pending call for latency measurement.
    # Keyed by event_id (not tool_name) so concurrent calls to the same
    # tool don't overwrite each other.
    pending = _read_pending_calls()
    pending[event.event_id] = {
        "tool_name": tool_name,
        "timestamp": event.timestamp,
    }
    _write_pending_calls(pending)


def handle_user_prompt(input_data: dict) -> None:
    """Handle UserPromptSubmit hook event. Logs the user's prompt."""
    store = _get_store()
    session_id = _resolve_session_id(input_data)
    if not session_id:
        return

    redact = _should_redact()
    prompt = input_data.get("prompt", "")

    event_data = {"prompt": prompt}
    if redact:
        event_data = redact_data(event_data)

    _write_event(store, 
        session_id,
        TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=session_id,
            data=event_data,
        ),
    )


def handle_stop(input_data: dict) -> None:
    """Handle Stop hook event. Logs the assistant's final response."""
    store = _get_store()
    session_id = _resolve_session_id(input_data)
    if not session_id:
        return

    # Skip if this is a recursive stop hook call
    if input_data.get("stop_hook_active"):
        return

    redact = _should_redact()
    text = input_data.get("last_assistant_message", "")

    if not text:
        return

    event_data = {"text": text}
    if redact:
        event_data = redact_data(event_data)

    _write_event(store, 
        session_id,
        TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE,
            session_id=session_id,
            data=event_data,
        ),
    )


def handle_post_tool(input_data: dict, failed: bool = False) -> None:
    """Handle PostToolUse / PostToolUseFailure hook event."""
    store = _get_store()
    session_id = _resolve_session_id(input_data)
    if not session_id:
        return

    redact = _should_redact()
    tool_name = input_data.get("tool_name", "unknown")
    tool_output = input_data.get("tool_output", "")

    if failed:
        event_type = EventType.ERROR
        error_msg = str(tool_output)[:500] if tool_output else "tool call failed"
        event_data = {
            "tool_name": tool_name,
            "error": error_msg,
        }
    else:
        event_type = EventType.TOOL_RESULT
        # Truncate large outputs
        output_str = str(tool_output)
        if len(output_str) > 1000:
            output_str = output_str[:1000] + "... (truncated)"
        event_data = {
            "tool_name": tool_name,
            "result": output_str,
        }

    if redact:
        event_data = redact_data(event_data)

    event = TraceEvent(
        event_type=event_type,
        session_id=session_id,
        data=event_data,
    )

    # Link to the earliest pending call for this tool name, then remove it.
    # Pending entries are keyed by event_id so concurrent same-tool calls
    # don't collide; we match by tool_name in the value and pick the oldest.
    pending = _read_pending_calls()
    match_id = None
    match_ts = float("inf")
    for eid, info in pending.items():
        if info.get("tool_name") == tool_name and info["timestamp"] < match_ts:
            match_id = eid
            match_ts = info["timestamp"]
    if match_id:
        call_info = pending.pop(match_id)
        event.parent_id = match_id
        event.duration_ms = (event.timestamp - call_info["timestamp"]) * 1000
        _write_pending_calls(pending)

    _write_event(store, session_id, event)

    # Update meta on errors
    if failed:
        meta = store.load_meta(session_id)
        if meta:
            meta.errors += 1
            store.update_meta(meta)


def hook_main(args: list[str]) -> None:
    """Entry point for `agent-strace hook <event>` CLI command."""
    if not args:
        sys.stderr.write("Usage: agent-strace hook <event>\n")
        sys.stderr.write("Events: session-start, session-end, pre-tool, post-tool, post-tool-failure, user-prompt, stop\n")
        sys.exit(1)

    event = args[0]
    input_data = _read_stdin()

    handlers = {
        "session-start": handle_session_start,
        "session-end": handle_session_end,
        "pre-tool": handle_pre_tool,
        "post-tool": lambda d: handle_post_tool(d, failed=False),
        "post-tool-failure": lambda d: handle_post_tool(d, failed=True),
        "user-prompt": handle_user_prompt,
        "stop": handle_stop,
    }

    handler = handlers.get(event)
    if not handler:
        sys.stderr.write(f"Unknown hook event: {event}\n")
        sys.stderr.write(f"Valid events: {', '.join(handlers.keys())}\n")
        sys.exit(1)

    try:
        handler(input_data)
    except Exception as e:
        # Hooks must not crash Claude Code. Log and exit cleanly.
        sys.stderr.write(f"agent-strace hook error: {e}\n")
        sys.exit(0)
