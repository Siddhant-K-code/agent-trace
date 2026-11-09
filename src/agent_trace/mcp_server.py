"""MCP server: expose the agent-trace session store as queryable MCP tools.

Implements the Model Context Protocol over stdio (JSON-RPC 2.0).
No external dependencies; uses only stdlib.

Tools exposed:
  list_sessions        list captured sessions with metadata
  get_session          full event stream for a session
  search_events        filter events by tool, file path, exit code, or time range
  get_session_summary  plain-English phase breakdown (wraps explain_session)
  diff_sessions        compare two sessions: what changed between runs
  health               evidence health for a session (wraps agent-strace health)
  skills               skill adoption report (wraps agent-strace skills)

Usage:
  agent-strace mcp                  # stdio transport (default)
  agent-strace mcp --trace-dir DIR  # custom trace directory

Client setup for Claude Code, Copilot CLI, Codex, and Gemini CLI: docs/mcp.md
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from typing import Any

from . import __version__
from .cost import parse_since
from .explain import explain_session, format_explain
from .health import assess_session, load_capture_matrix
from .skills import build_aggregate_report, build_session_report
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "name": "list_sessions",
        "description": (
            "List captured agent sessions with metadata: session ID, start time, "
            "tool call count, LLM requests, errors, total tokens, and estimated cost."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of sessions to return (default: 20).",
                    "default": 20,
                },
                "agent": {
                    "type": "string",
                    "description": "Filter by agent name (substring match).",
                    "default": "",
                },
            },
        },
    },
    {
        "name": "get_session",
        "description": (
            "Return the full event stream for a session as structured JSON. "
            "Includes every tool call, LLM request, file read/write, and error."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID or unique prefix.",
                },
                "event_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter to these event types only. "
                        "Valid values: tool_call, tool_result, llm_request, llm_response, "
                        "file_read, file_write, error, user_prompt, assistant_response. "
                        "Empty list returns all events."
                    ),
                    "default": [],
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "search_events",
        "description": (
            "Search events across one or all sessions. Filter by tool name, "
            "file path substring, exit code, or time range."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID or prefix. Omit to search all sessions.",
                    "default": "",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Filter tool_call events by tool name (case-insensitive substring).",
                    "default": "",
                },
                "file_path": {
                    "type": "string",
                    "description": "Filter file_read/file_write events by path substring.",
                    "default": "",
                },
                "exit_code": {
                    "type": "integer",
                    "description": "Filter tool_result events by exit code.",
                },
                "has_error": {
                    "type": "boolean",
                    "description": "If true, return only error events.",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum events to return (default: 50).",
                    "default": 50,
                },
            },
        },
    },
    {
        "name": "get_session_summary",
        "description": (
            "Return a plain-English summary of what the agent did in a session: "
            "phases, files touched, commands run, retries, and wasted time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID or unique prefix.",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "diff_sessions",
        "description": (
            "Compare two sessions and return a structured diff: "
            "which tools were added/removed, file overlap, cost delta, "
            "error delta, and token delta."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_a": {
                    "type": "string",
                    "description": "First session ID or prefix (the 'before' session).",
                },
                "session_b": {
                    "type": "string",
                    "description": "Second session ID or prefix (the 'after' session).",
                },
            },
            "required": ["session_a", "session_b"],
        },
    },
    {
        "name": "health",
        "description": (
            "Report evidence health for a session: which agent/provider recorded it, "
            "whether it was finalized, redaction state, known capture blind spots, "
            "and structural problems such as orphaned tool results. "
            "Same data as 'agent-strace health --format json'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID or unique prefix. Omit for the latest session.",
                    "default": "",
                },
            },
        },
    },
    {
        "name": "skills",
        "description": (
            "Report agent skill (SKILL.md) adoption: which skills are available, "
            "which were loaded, follow-up signals after a load, and which never loaded. "
            "Without session_id, aggregates sessions since 'since'. "
            "Same data as 'agent-strace skills --format json', except command_match "
            "reports counts only. Prompt text, SKILL.md bodies, and command "
            "signatures taken from them are never returned."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID or prefix for a single-session report. Omit to aggregate.",
                    "default": "",
                },
                "since": {
                    "type": "string",
                    "description": "Aggregate window, e.g. 7d, 24h, or 2026-01-01 (default: 30d).",
                    "default": "30d",
                },
                "workspace": {
                    "type": "string",
                    "description": (
                        "Repository path used to discover repo skills when a session has no "
                        "recorded cwd. Defaults to the server's working directory."
                    ),
                    "default": "",
                },
                "skills_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra skill directories to scan.",
                    "default": [],
                },
                "min_sessions": {
                    "type": "integer",
                    "description": "Minimum sessions before a skill is reported as never loaded (default: 5).",
                    "default": 5,
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_COST_PER_1K = 0.003  # rough blended cost estimate


def _session_to_dict(meta) -> dict:
    cost = meta.total_tokens / 1_000 * _COST_PER_1K
    return {
        "session_id": meta.session_id,
        "started_at": meta.started_at,
        "agent_name": meta.agent_name or "",
        "command": meta.command or "",
        "tool_calls": meta.tool_calls,
        "llm_requests": meta.llm_requests,
        "errors": meta.errors,
        "total_tokens": meta.total_tokens,
        "total_duration_ms": meta.total_duration_ms,
        "estimated_cost_usd": round(cost, 4),
    }


def _event_to_dict(ev: TraceEvent) -> dict:
    return {
        "event_id": ev.event_id,
        "event_type": ev.event_type.value,
        "timestamp": ev.timestamp,
        "session_id": ev.session_id or "",
        "parent_id": ev.parent_id or "",
        "duration_ms": ev.duration_ms,
        "data": ev.data,
    }


def _tool_list_sessions(store: TraceStore, args: dict) -> str:
    limit = int(args.get("limit") or 20)
    agent_filter = str(args.get("agent") or "").lower()
    sessions = store.list_sessions()
    if agent_filter:
        sessions = [s for s in sessions if agent_filter in (s.agent_name or "").lower()]
    sessions = sessions[:limit]
    result = [_session_to_dict(s) for s in sessions]
    return json.dumps({"sessions": result, "count": len(result)}, indent=2)


def _tool_get_session(store: TraceStore, args: dict) -> str:
    session_id = str(args.get("session_id") or "")
    if not session_id:
        return json.dumps({"error": "session_id is required"})
    full_id = store.find_session(session_id)
    if not full_id:
        return json.dumps({"error": f"session not found: {session_id}"})

    meta = store.load_meta(full_id)
    events = store.load_events(full_id)

    type_filter: list[str] = [t.lower() for t in (args.get("event_types") or [])]
    if type_filter:
        events = [e for e in events if e.event_type.value in type_filter]

    return json.dumps({
        "session": _session_to_dict(meta),
        "events": [_event_to_dict(e) for e in events],
        "event_count": len(events),
    }, indent=2)


def _tool_search_events(store: TraceStore, args: dict) -> str:
    session_id = str(args.get("session_id") or "")
    tool_name = str(args.get("tool_name") or "").lower()
    file_path = str(args.get("file_path") or "").lower()
    exit_code = args.get("exit_code")
    has_error = bool(args.get("has_error"))
    limit = int(args.get("limit") or 50)

    if session_id:
        full_id = store.find_session(session_id)
        if not full_id:
            return json.dumps({"error": f"session not found: {session_id}"})
        session_ids = [full_id]
    else:
        sessions = store.list_sessions()
        session_ids = [s.session_id for s in sessions]

    matches: list[dict] = []
    for sid in session_ids:
        try:
            events = store.load_events(sid)
        except Exception:
            continue
        for ev in events:
            if has_error and ev.event_type != EventType.ERROR:
                continue
            if tool_name and ev.event_type == EventType.TOOL_CALL:
                if tool_name not in str(ev.data.get("tool_name", "")).lower():
                    continue
            elif tool_name:
                continue
            if file_path:
                if ev.event_type not in (EventType.FILE_READ, EventType.FILE_WRITE):
                    continue
                path = str(ev.data.get("path", ev.data.get("file_path", ""))).lower()
                if file_path not in path:
                    continue
            if exit_code is not None:
                if ev.event_type != EventType.TOOL_RESULT:
                    continue
                if ev.data.get("exit_code") != exit_code:
                    continue
            d = _event_to_dict(ev)
            d["_session_id"] = sid
            matches.append(d)
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break

    return json.dumps({"events": matches, "count": len(matches)}, indent=2)


def _tool_get_session_summary(store: TraceStore, args: dict) -> str:
    session_id = str(args.get("session_id") or "")
    if not session_id:
        return json.dumps({"error": "session_id is required"})
    full_id = store.find_session(session_id)
    if not full_id:
        return json.dumps({"error": f"session not found: {session_id}"})

    result = explain_session(store, full_id)
    buf = io.StringIO()
    format_explain(result, out=buf)
    return buf.getvalue()


def _tool_diff_sessions(store: TraceStore, args: dict) -> str:
    sid_a = str(args.get("session_a") or "")
    sid_b = str(args.get("session_b") or "")
    if not sid_a or not sid_b:
        return json.dumps({"error": "session_a and session_b are required"})

    full_a = store.find_session(sid_a)
    full_b = store.find_session(sid_b)
    if not full_a:
        return json.dumps({"error": f"session not found: {sid_a}"})
    if not full_b:
        return json.dumps({"error": f"session not found: {sid_b}"})

    meta_a = store.load_meta(full_a)
    meta_b = store.load_meta(full_b)
    events_a = store.load_events(full_a)
    events_b = store.load_events(full_b)

    def _tools(events: list[TraceEvent]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in events:
            if e.event_type == EventType.TOOL_CALL:
                name = e.data.get("tool_name", "unknown")
                counts[name] = counts.get(name, 0) + 1
        return counts

    def _files(events: list[TraceEvent]) -> set[str]:
        paths: set[str] = set()
        for e in events:
            if e.event_type in (EventType.FILE_READ, EventType.FILE_WRITE):
                p = e.data.get("path", e.data.get("file_path", ""))
                if p:
                    paths.add(p)
        return paths

    def _errors(events: list[TraceEvent]) -> list[str]:
        return [e.data.get("message", "") for e in events if e.event_type == EventType.ERROR]

    tools_a = _tools(events_a)
    tools_b = _tools(events_b)
    files_a = _files(events_a)
    files_b = _files(events_b)
    errors_a = _errors(events_a)
    errors_b = _errors(events_b)

    all_tools = set(tools_a) | set(tools_b)
    tool_diff = {
        t: {"session_a": tools_a.get(t, 0), "session_b": tools_b.get(t, 0)}
        for t in sorted(all_tools)
        if tools_a.get(t, 0) != tools_b.get(t, 0)
    }

    cost_a = meta_a.total_tokens / 1_000 * _COST_PER_1K
    cost_b = meta_b.total_tokens / 1_000 * _COST_PER_1K

    return json.dumps({
        "session_a": full_a,
        "session_b": full_b,
        "tool_call_diff": tool_diff,
        "files_only_in_a": sorted(files_a - files_b),
        "files_only_in_b": sorted(files_b - files_a),
        "files_in_both": sorted(files_a & files_b),
        "token_delta": meta_b.total_tokens - meta_a.total_tokens,
        "cost_delta_usd": round(cost_b - cost_a, 4),
        "error_count_a": len(errors_a),
        "error_count_b": len(errors_b),
        "duration_delta_ms": meta_b.total_duration_ms - meta_a.total_duration_ms,
        "errors_a": errors_a[:10],
        "errors_b": errors_b[:10],
    }, indent=2)


def _tool_health(store: TraceStore, args: dict) -> str:
    requested = str(args.get("session_id") or "")
    if requested:
        session_id = store.find_session(requested)
        if not session_id:
            return json.dumps({"error": f"session not found: {requested}"})
    else:
        session_id = store.get_latest_session_id()
        if not session_id:
            return json.dumps({"error": "no sessions found"})

    matrix = load_capture_matrix()
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    report = assess_session(store, session_id, meta, events, matrix)
    return json.dumps(report.to_dict(), indent=2)


def _redact_command_matches(value: Any) -> Any:
    """Replace command_match signature lists with counts.

    Signatures are matched against commands written in SKILL.md bodies, so
    the MCP tool never returns them.
    """
    if isinstance(value, list):
        return [_redact_command_matches(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if key == "command_match" and isinstance(item, dict) and "by_signature" in item:
            out[key] = {
                "count": item.get("count", 0),
                "distinct_signatures": len(item.get("by_signature") or {}),
            }
        else:
            out[key] = _redact_command_matches(item)
    return out


def _tool_skills(store: TraceStore, args: dict) -> str:
    workspace = str(args.get("workspace") or "")
    extra_dirs = [str(d) for d in (args.get("skills_dirs") or [])]
    try:
        raw_min = args.get("min_sessions")
        min_sessions = 5 if raw_min is None else int(raw_min)
    except (TypeError, ValueError):
        return json.dumps({"error": "min_sessions must be an integer"})
    if min_sessions < 1:
        return json.dumps({"error": "min_sessions must be at least 1"})

    requested = str(args.get("session_id") or "")
    if requested:
        session_id = store.find_session(requested)
        if not session_id:
            return json.dumps({"error": f"session not found: {requested}"})
        report = build_session_report(store, session_id, workspace=workspace, extra_dirs=extra_dirs)
        return json.dumps(_redact_command_matches(report), indent=2)

    try:
        since = parse_since(str(args.get("since") or "30d"))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    report = build_aggregate_report(
        store, since=since, workspace=workspace, extra_dirs=extra_dirs, min_sessions=min_sessions,
    )
    return json.dumps(_redact_command_matches(report), indent=2)


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 dispatcher
# ---------------------------------------------------------------------------

def _ok(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


_DISPATCH = {
    "list_sessions": _tool_list_sessions,
    "get_session": _tool_get_session,
    "search_events": _tool_search_events,
    "get_session_summary": _tool_get_session_summary,
    "diff_sessions": _tool_diff_sessions,
    "health": _tool_health,
    "skills": _tool_skills,
}


def _tool_result(text: str, is_error: bool) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _reports_error(text: str) -> bool:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and set(payload) == {"error"}


def _handle(store: TraceStore, request: Any) -> dict | None:
    if not isinstance(request, dict):
        return _err(None, -32600, "invalid request: expected a JSON object")

    is_notification = "id" not in request
    req_id = request.get("id")
    if not is_notification and not (
        req_id is None or isinstance(req_id, str)
        or (isinstance(req_id, (int, float)) and not isinstance(req_id, bool))
    ):
        return _err(None, -32600, "invalid request: id must be a string, number, or null")

    method = request.get("method")
    if request.get("jsonrpc") != "2.0" or not isinstance(method, str):
        if is_notification:
            return None
        return _err(req_id, -32600, "invalid request: jsonrpc must be \"2.0\" and method a string")

    params = request.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        if is_notification:
            return None
        return _err(req_id, -32602, "invalid params: params must be an object")

    if is_notification:
        return None

    if method == "initialize":
        return _ok(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "agent-trace", "version": __version__},
        })

    if method == "ping":
        return _ok(req_id, {})

    if method == "tools/list":
        return _ok(req_id, {"tools": _TOOLS})

    if method == "tools/call":
        name = params.get("name")
        if not isinstance(name, str):
            return _err(req_id, -32602, "invalid params: name must be a string")
        tool_args = params.get("arguments", {})
        if tool_args is None:
            tool_args = {}
        if not isinstance(tool_args, dict):
            return _err(req_id, -32602, "invalid params: arguments must be an object")

        fn = _DISPATCH.get(name)
        if fn is None:
            return _err(req_id, -32602, f"unknown tool: {name}")

        try:
            text = fn(store, tool_args)
        except Exception as exc:
            return _ok(req_id, _tool_result(f"{name} failed: {type(exc).__name__}: {exc}", True))

        return _ok(req_id, _tool_result(text, _reports_error(text)))

    return _err(req_id, -32601, f"method not found: {method}")


# ---------------------------------------------------------------------------
# Stdio transport loop
# ---------------------------------------------------------------------------

def run_stdio(store: TraceStore) -> None:
    """Read JSON-RPC requests from stdin, write responses to stdout."""
    stdin = sys.stdin
    stdout = sys.stdout

    # Use binary mode for reliable line reading across platforms
    if hasattr(stdin, "buffer"):
        reader = io.TextIOWrapper(stdin.buffer, encoding="utf-8", newline="\n")
    else:
        reader = stdin

    for raw_line in reader:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            response = _err(None, -32700, f"parse error: {exc}")
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
            continue

        if isinstance(request, list):
            response = _err(None, -32600, "invalid request: batch requests are not supported")
        else:
            try:
                response = _handle(store, request)
            except Exception as exc:
                req_id = request.get("id") if isinstance(request, dict) else None
                response = _err(req_id, -32603, f"internal error: {type(exc).__name__}: {exc}")
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_mcp(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir, read_only=True)
    sys.stderr.write(
        f"agent-trace MCP server started (trace_dir={args.trace_dir})\n"
        "Waiting for JSON-RPC requests on stdin...\n"
    )
    try:
        run_stdio(store)
    except KeyboardInterrupt:
        pass
    return 0
