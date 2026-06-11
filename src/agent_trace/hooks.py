"""Agent CLI hooks integration.

Captures hook-visible tool calls from supported agent CLIs, not just MCP
calls. Uses provider hook systems such as Claude Code, OpenAI Codex, Gemini,
Cursor, and GitHub Copilot.

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
_CODEX_SESSION_ID_ENV = "AGENT_TRACE_CODEX_SESSION_ID"
_GEMINI_SESSION_ID_ENV = "AGENT_TRACE_GEMINI_SESSION_ID"
_CURSOR_SESSION_ID_ENV = "AGENT_TRACE_CURSOR_SESSION_ID"
_COPILOT_SESSION_ID_ENV = "AGENT_TRACE_COPILOT_SESSION_ID"

_PROVIDER_ENV = {
    "claude": _CLAUDE_SESSION_ID_ENV,
    "codex": _CODEX_SESSION_ID_ENV,
    "gemini": _GEMINI_SESSION_ID_ENV,
    "cursor": _CURSOR_SESSION_ID_ENV,
    "copilot": _COPILOT_SESSION_ID_ENV,
}

_PROVIDER_AGENT = {
    "claude": "claude-code",
    "codex": "openai-codex",
    "gemini": "gemini-cli",
    "cursor": "cursor-agent",
    "copilot": "github-copilot",
}


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


def _provider_env(provider: str = "claude") -> str:
    return _PROVIDER_ENV.get(provider, _CLAUDE_SESSION_ID_ENV)


def _state_suffix(provider: str = "claude") -> str:
    """Return a filename suffix scoped to the current hook provider session.

    Providers pass session_id on every hook payload. handle_session_start also
    sets a provider-specific env var for same-process tests and manual use.
    """
    sid = os.environ.get(_provider_env(provider), "")
    return f".{sid}" if sid else ""


def _active_session_path(provider: str = "claude") -> Path:
    return Path(_get_store_dir()) / f".active-session{_state_suffix(provider)}"


def _pending_calls_path(provider: str = "claude") -> Path:
    suffix = _state_suffix(provider)
    name = _PENDING_FILE.replace(".json", f"{suffix}.json")
    return Path(_get_store_dir()) / name


def _read_active_session(provider: str = "claude") -> str | None:
    path = _active_session_path(provider)
    if path.exists():
        return path.read_text().strip()
    return None


def _resolve_session_id(input_data: dict, provider: str = "claude") -> str | None:
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
    return _read_active_session(provider)


def _write_active_session(session_id: str, provider: str = "claude") -> None:
    path = _active_session_path(provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(session_id)


def _clear_active_session(provider: str = "claude") -> None:
    path = _active_session_path(provider)
    if path.exists():
        path.unlink()


def _read_pending_calls(provider: str = "claude") -> dict:
    path = _pending_calls_path(provider)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _write_pending_calls(calls: dict, provider: str = "claude") -> None:
    path = _pending_calls_path(provider)
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


def _normalise_payload(input_data: dict, provider: str, event: str) -> dict:
    """Map provider-specific hook payloads to the Claude-shaped fields."""
    data = dict(input_data)
    if provider not in ("codex", "gemini", "cursor", "copilot"):
        return data

    data.setdefault("session_id", data.get("sessionId") or "")
    data.setdefault("turn_id", data.get("turnId") or "")
    data.setdefault("tool_use_id", data.get("toolUseId") or "")

    if event in {"pre-tool", "post-tool", "post-tool-failure"}:
        tool = data.get("tool")
        if isinstance(tool, dict):
            data.setdefault("tool_name", tool.get("name") or tool.get("tool_name") or "")
            data.setdefault("tool_input", tool.get("input") or tool.get("arguments") or {})
            data.setdefault("tool_output", tool.get("output") or tool.get("response") or "")
        if data.get("toolName") and not data.get("tool_name"):
            data["tool_name"] = data.get("toolName")
        if data.get("toolArgs") is not None and "tool_input" not in data:
            data["tool_input"] = data.get("toolArgs")
        if (data.get("toolResult") is not None or data.get("textResultForLlm") is not None) and "tool_output" not in data:
            data["tool_output"] = data.get("toolResult", data.get("textResultForLlm", ""))
        command = data.get("command")
        if command and not data.get("tool_name"):
            data.setdefault("tool_name", "shell")
            data.setdefault("tool_input", {"command": command})
        if data.get("file_path") or data.get("path"):
            data.setdefault("tool_name", data.get("tool_name") or "file_edit")
            data.setdefault("tool_input", {
                "file_path": data.get("file_path") or data.get("path"),
            })
        data.setdefault("tool_input", data.get("input") or data.get("arguments") or {})
        data.setdefault("tool_output", data.get("tool_response", data.get("output", "")))

    if provider in ("gemini", "cursor", "copilot"):
        if event == "session-start":
            data.setdefault("source", data.get("hook_event_name", "startup"))
        if event == "user-prompt":
            input_value = data.get("input", {})
            prompt = input_value.get("prompt", "") if isinstance(input_value, dict) else input_value
            data.setdefault("prompt", data.get("user_prompt") or data.get("prompt") or data.get("initialPrompt") or prompt or "")
        if event == "stop":
            data.setdefault("last_assistant_message", data.get("prompt_response", ""))

    if event == "stop" and data.get("last_assistant_message") is None:
        data["last_assistant_message"] = data.get("assistant_message") or data.get("message") or ""

    return data


def handle_session_start(input_data: dict, provider: str = "claude") -> None:
    """Handle SessionStart hook event."""
    store = _get_store()
    redact = _should_redact()

    session_id = input_data.get("session_id", "")
    agent_name = _PROVIDER_AGENT.get(provider, "agent-cli")
    attr = collect_attribution()
    meta = SessionMeta(
        agent_name=agent_name,
        command=f"{agent_name} ({input_data.get('source', 'startup')})",
        attribution=attr.to_dict(),
    )
    # Use Claude Code's session ID as part of our session ID for correlation
    if session_id:
        meta.session_id = session_id[:16]

    store.create_session(meta)

    if session_id:
        os.environ[_provider_env(provider)] = session_id

    _write_active_session(meta.session_id, provider=provider)
    _write_pending_calls({}, provider=provider)

    event_data = {
        "mode": f"{agent_name}-hooks",
        "provider": provider,
        "source": input_data.get("source", "startup"),
        "model": input_data.get("model", ""),
    }
    for key in ("cwd", "transcript_path", "permission_mode"):
        if input_data.get(key) not in (None, ""):
            event_data[key] = input_data.get(key)
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


def handle_session_end(input_data: dict, provider: str = "claude") -> None:
    """Handle SessionEnd hook event."""
    store = _get_store()
    session_id = _resolve_session_id(input_data, provider=provider)
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

    _clear_active_session(provider=provider)


def handle_pre_tool(input_data: dict, provider: str = "claude") -> None:
    """Handle PreToolUse hook event. Logs tool_call and tracks pending calls."""
    store = _get_store()
    session_id = _resolve_session_id(input_data, provider=provider)
    if not session_id:
        return

    redact = _should_redact()
    tool_name = input_data.get("tool_name", "unknown")
    tool_input = input_data.get("tool_input", {})

    event_data = {
        "tool_name": tool_name,
        "arguments": tool_input,
    }
    for key in ("tool_use_id", "turn_id", "permission_mode"):
        if input_data.get(key) not in (None, ""):
            event_data[key] = input_data.get(key)
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
    pending = _read_pending_calls(provider=provider)
    pending[event.event_id] = {
        "tool_name": tool_name,
        "tool_use_id": input_data.get("tool_use_id", ""),
        "timestamp": event.timestamp,
    }
    _write_pending_calls(pending, provider=provider)


def handle_user_prompt(input_data: dict, provider: str = "claude") -> None:
    """Handle UserPromptSubmit hook event. Logs the user's prompt."""
    store = _get_store()
    session_id = _resolve_session_id(input_data, provider=provider)
    if not session_id:
        return

    redact = _should_redact()
    prompt = input_data.get("prompt", "")

    event_data = {"prompt": prompt}
    for key in ("turn_id", "permission_mode"):
        if input_data.get(key) not in (None, ""):
            event_data[key] = input_data.get(key)
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


def handle_stop(input_data: dict, provider: str = "claude") -> None:
    """Handle Stop hook event. Logs the assistant's final response."""
    store = _get_store()
    session_id = _resolve_session_id(input_data, provider=provider)
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
    for key in ("turn_id", "permission_mode"):
        if input_data.get(key) not in (None, ""):
            event_data[key] = input_data.get(key)
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


def handle_file_write(input_data: dict, provider: str = "claude") -> None:
    """Handle provider file-edit hooks as file_write events."""
    store = _get_store()
    session_id = _resolve_session_id(input_data, provider=provider)
    if not session_id:
        return

    redact = _should_redact()
    data = {
        "path": input_data.get("file_path") or input_data.get("path") or input_data.get("uri") or "",
    }
    for key in ("diff", "patch", "change_summary", "turn_id", "permission_mode"):
        if input_data.get(key) not in (None, ""):
            data[key] = input_data.get(key)
    if redact:
        data = redact_data(data)

    _write_event(
        store,
        session_id,
        TraceEvent(
            event_type=EventType.FILE_WRITE,
            session_id=session_id,
            data=data,
        ),
    )


def handle_post_tool(input_data: dict, failed: bool = False, provider: str = "claude") -> None:
    """Handle PostToolUse / PostToolUseFailure hook event."""
    store = _get_store()
    session_id = _resolve_session_id(input_data, provider=provider)
    if not session_id:
        return

    redact = _should_redact()
    tool_name = input_data.get("tool_name", "unknown")
    tool_output = input_data.get("tool_output", input_data.get("tool_response", ""))

    if provider in ("codex", "gemini", "cursor", "copilot") and not failed:
        if isinstance(tool_output, dict):
            exit_code = tool_output.get("exit_code")
            failed = (
                bool(tool_output.get("error"))
                or tool_output.get("success") is False
                or (isinstance(exit_code, int) and exit_code != 0)
            )

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
    for key in ("tool_use_id", "turn_id", "permission_mode"):
        if input_data.get(key) not in (None, ""):
            event_data[key] = input_data.get(key)

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
    pending = _read_pending_calls(provider=provider)
    match_id = None
    match_ts = float("inf")
    tool_use_id = input_data.get("tool_use_id", "")
    for eid, info in pending.items():
        if tool_use_id and info.get("tool_use_id") == tool_use_id:
            match_id = eid
            break
        if not tool_use_id and info.get("tool_name") == tool_name and info["timestamp"] < match_ts:
            match_id = eid
            match_ts = info["timestamp"]
    if match_id:
        call_info = pending.pop(match_id)
        event.parent_id = match_id
        event.duration_ms = (event.timestamp - call_info["timestamp"]) * 1000
        _write_pending_calls(pending, provider=provider)

    _write_event(store, session_id, event)

    # Update meta on errors
    if failed:
        meta = store.load_meta(session_id)
        if meta:
            meta.errors += 1
            store.update_meta(meta)


def hook_main(args: list[str]) -> None:
    """Entry point for `agent-strace hook <event>` CLI command."""
    provider = "claude"
    rest = list(args)
    if rest[:1] == ["--provider"] and len(rest) >= 2:
        provider = rest[1]
        rest = rest[2:]
    elif rest and rest[0] in _PROVIDER_AGENT and len(rest) >= 2:
        provider = rest[0]
        rest = rest[1:]

    if not args:
        sys.stderr.write("Usage: agent-strace hook <event>\n")
        sys.stderr.write("Events: session-start, session-end, pre-tool, post-tool, post-tool-failure, user-prompt, stop\n")
        sys.exit(1)

    if provider not in _PROVIDER_AGENT:
        sys.stderr.write(f"Unknown hook provider: {provider}\n")
        sys.exit(1)

    if not rest:
        sys.stderr.write("Usage: agent-strace hook [--provider claude|codex|gemini|cursor|copilot] <event>\n")
        sys.exit(1)

    aliases = {
        "before-tool": "pre-tool",
        "before-tool-call": "pre-tool",
        "after-tool": "post-tool",
        "after-tool-call": "post-tool",
        "before-agent": "user-prompt",
        "before-prompt": "user-prompt",
        "after-agent": "stop",
        "before-submit-prompt": "user-prompt",
        "before-shell-execution": "pre-tool",
        "after-shell-execution": "post-tool",
        "after-file-edit": "file-write",
        "after-agent-response": "stop",
        "SessionStart": "session-start",
        "SessionEnd": "session-end",
        "UserPromptSubmit": "user-prompt",
        "PreToolUse": "pre-tool",
        "PostToolUse": "post-tool",
        "PostToolUseFailure": "post-tool-failure",
        "AgentStop": "stop",
        "agent-stop": "stop",
    }
    event = aliases.get(rest[0], rest[0])
    input_data = _normalise_payload(_read_stdin(), provider, event)

    handlers = {
        "session-start": lambda d: handle_session_start(d, provider=provider),
        "session-end": lambda d: handle_session_end(d, provider=provider),
        "pre-tool": lambda d: handle_pre_tool(d, provider=provider),
        "post-tool": lambda d: handle_post_tool(d, failed=False, provider=provider),
        "post-tool-failure": lambda d: handle_post_tool(d, failed=True, provider=provider),
        "user-prompt": lambda d: handle_user_prompt(d, provider=provider),
        "file-write": lambda d: handle_file_write(d, provider=provider),
        "stop": lambda d: handle_stop(d, provider=provider),
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
