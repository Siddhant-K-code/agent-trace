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
from .redact import redact_data
from .store import TraceStore

# Pending tool calls are tracked in a file so separate hook processes
# can link PostToolUse back to PreToolUse for latency measurement.
_PENDING_FILE = ".pending-calls.json"


def _get_store_dir() -> str:
    return os.environ.get("AGENT_TRACE_DIR", ".agent-traces")


def _get_store() -> TraceStore:
    return TraceStore(_get_store_dir())


def _active_session_path() -> Path:
    return Path(_get_store_dir()) / ".active-session"


def _pending_calls_path() -> Path:
    return Path(_get_store_dir()) / _PENDING_FILE


def _read_active_session() -> str | None:
    path = _active_session_path()
    if path.exists():
        return path.read_text().strip()
    return None


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
    return os.environ.get("AGENT_TRACE_REDACT", "").lower() in ("1", "true", "yes")


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
    _write_active_session(meta.session_id)
    _write_pending_calls({})

    event_data = {
        "mode": "claude-code-hooks",
        "source": input_data.get("source", "startup"),
        "model": input_data.get("model", ""),
    }
    if redact:
        event_data = redact_data(event_data)

    store.append_event(
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
    session_id = _read_active_session()
    if not session_id:
        return

    meta = store.load_meta(session_id)
    if meta:
        meta.ended_at = time.time()
        meta.total_duration_ms = (meta.ended_at - meta.started_at) * 1000

        store.append_event(
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
    session_id = _read_active_session()
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
    store.append_event(session_id, event)

    # Update meta
    meta = store.load_meta(session_id)
    if meta:
        meta.tool_calls += 1
        store.update_meta(meta)

    # Track pending call for latency measurement
    pending = _read_pending_calls()
    pending[tool_name] = {
        "event_id": event.event_id,
        "timestamp": event.timestamp,
    }
    _write_pending_calls(pending)


def handle_user_prompt(input_data: dict) -> None:
    """Handle UserPromptSubmit hook event. Logs the user's prompt."""
    store = _get_store()
    session_id = _read_active_session()
    if not session_id:
        return

    redact = _should_redact()
    prompt = input_data.get("prompt", "")

    event_data = {"prompt": prompt}
    if redact:
        event_data = redact_data(event_data)

    store.append_event(
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
    session_id = _read_active_session()
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

    store.append_event(
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
    session_id = _read_active_session()
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

    # Link to pending call for latency
    pending = _read_pending_calls()
    call_info = pending.pop(tool_name, None)
    if call_info:
        event.parent_id = call_info["event_id"]
        event.duration_ms = (event.timestamp - call_info["timestamp"]) * 1000
        _write_pending_calls(pending)

    store.append_event(session_id, event)

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
