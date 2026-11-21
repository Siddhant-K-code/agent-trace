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
import os
import stat
import sys
from pathlib import Path
from typing import Any, Iterator

from .models import EventType, SessionMeta, TraceEvent, usable_number, usable_time
from .redact import redact_data
from .store import TraceStore, resolve_trace_dir, validate_session_id

_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _token_count(value: object) -> int:
    """Return ``value`` if it is an int >= 0 and not a bool, else 0.

    This is the hook's rule (``hooks._token_count``).
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


# A line nested more than this many levels deep is skipped, the limit
# collector_client.py uses. The store's redactor needs 2 frames per level on
# Python 3.12 and later and 3 on 3.10 and 3.11, so on 3.10 and 3.11 data about
# 328 levels deep fails, after the session is created.
MAX_LINE_DEPTH = 64


def _too_deep(entry: dict[str, Any]) -> bool:
    """Return True if ``entry`` nests objects and arrays more than MAX_LINE_DEPTH levels deep.

    ``entry`` is level 1. The walk uses a stack, not recursion: on 3.12 and
    later, json.loads parses lines far deeper than Python can recurse.
    """
    stack = [(entry, 1)]
    while stack:
        node, depth = stack.pop()
        for child in node.values() if isinstance(node, dict) else node:
            if isinstance(child, (dict, list)):
                if depth == MAX_LINE_DEPTH:
                    return True
                stack.append((child, depth + 1))
    return False


def _redact_text(text: str, redact: bool) -> tuple[str, bool]:
    """Return ``text`` with its secrets redacted when ``redact`` is set, and whether any were.

    Call this on the whole text before cutting it, as the hooks do: a secret
    cut in half no longer matches its pattern, so the store cannot find it.
    """
    if not redact or not text:
        return text, False
    redacted = redact_data(text)
    return redacted, redacted != text


def _usable_session_id(value: object) -> str:
    """Return ``value`` if it is a string the store accepts as a session ID, else ""."""
    if not isinstance(value, str):
        return ""
    try:
        return validate_session_id(value)
    except ValueError:
        return ""


def _resolve(path: str | Path) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except RuntimeError:
        # Python 3.12 and older raise this on a symlink loop.
        return Path(os.path.abspath(Path(path).expanduser()))


class _UnreadableLog(OSError):
    """The session log exists but cannot be opened or read."""


def _log_lines(path: Path) -> Iterator[str]:
    """Yield the lines of the log. Only errors from opening and reading it are converted."""
    try:
        # A byte that is not UTF-8 becomes a lone surrogate, so one bad line
        # does not stop the read. The caller skips that line.
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            yield from f
    except FileNotFoundError:
        raise FileNotFoundError(f"Session log not found: {path}") from None
    except OSError as e:
        raise _UnreadableLog(e.errno, e.strerror or str(e), e.filename) from e


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
        session_id, written = _import_jsonl(args.path, store)
    except FileNotFoundError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    except _UnreadableLog as e:
        sys.stderr.write(f"Cannot read session log {args.path}: {e.strerror}\n")
        return 1
    except ValueError as e:
        sys.stderr.write(f"Cannot import {args.path}: {e}\n")
        return 1

    if not written:
        sys.stderr.write(
            f"Session {session_id} is already imported; nothing written.\n"
            f"  To import it again, delete {store.base_dir / session_id} first.\n"
        )
        return 0

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


def _parse_iso_timestamp(ts: object) -> float:
    """Convert an ISO 8601 timestamp to Unix epoch seconds.

    Returns 0.0 for a value that is not a non-empty string, for text that does
    not parse, and for a time that is not a Unix time in seconds, such as a
    time before 1970.
    """
    from datetime import datetime

    if not isinstance(ts, str) or not ts:
        return 0.0
    try:
        # Handle Z suffix
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return usable_time(dt.timestamp()) or 0.0
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
                        text = sub.get("text", "")
                        text_parts.append(text if isinstance(text, str) else "")
            results.append(
                {
                    "tool_use_id": block.get("tool_use_id", ""),
                    "content": "\n".join(text_parts),
                    "is_error": block.get("is_error") is True,
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
                text = block.get("text", "")
                parts.append(text if isinstance(text, str) else "")
        return "\n".join(parts)
    return ""


def _explicit_sidechain_id(raw: dict[str, Any]) -> str:
    """Return a provider-supplied sidechain identity when one is available."""
    for key in ("agentId", "agent_id", "sidechainId", "sidechain_id"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _is_compaction_summary(raw: dict[str, Any], text: str, pending: bool) -> bool:
    """Recognise Claude Code's synthetic post-compaction user message."""
    if raw.get("isCompactSummary") is True or raw.get("isCompactionSummary") is True:
        return True
    lowered = text.lstrip().lower()
    if lowered.startswith(
        "this session is being continued from a previous conversation that ran out of context"
    ):
        return True
    return pending and bool(text)


def import_jsonl(
    path: str | Path,
    store: TraceStore | None = None,
    trace_dir: str | None = None,
    *,
    merge: bool = False,
) -> str:
    """Import a Claude Code JSONL session log into agent-trace format.

    If the store already holds a session with the log's ID, nothing is
    written and that ID is returned, so importing a file again does not
    duplicate its events. With ``merge=True`` the existing session's
    metadata is replaced and this log's events are appended to it instead;
    ``skills --from-claude`` uses this to merge transcripts that share a
    session ID.

    Parameters
    ----------
    path : str or Path
        Path to the .jsonl session file.
    store : TraceStore, optional
        Existing store to import into. Creates one if not provided.
    trace_dir : str, optional
        Directory for trace storage (used if store is None). Defaults to
        ``$AGENT_TRACE_DIR`` when set and non-empty, else ``.agent-traces``.

    Returns
    -------
    str
        The session ID of the imported session.

    Raises
    ------
    FileNotFoundError
        The log does not exist.
    OSError
        The log cannot be opened or read.
    ValueError
        Neither the log nor its file name holds a usable session ID, or the
        store refuses the session.
    """
    if store is None:
        store = TraceStore(resolve_trace_dir(trace_dir))
    return _import_jsonl(path, store, merge=merge)[0]


def _import_jsonl(
    path: str | Path, store: TraceStore, *, merge: bool = False
) -> tuple[str, bool]:
    """Import ``path`` into ``store``; returns ``(session_id, written)``.

    Events are built in memory and written only after the whole log has been
    converted, so a log that fails to parse leaves no partial session behind.
    """
    path = _resolve(path)

    # First pass: extract metadata from first entry
    session_id = ""
    cwd = ""
    extra_meta: dict[str, str] = {}
    first_ts = 0.0
    last_ts = 0.0
    entries: list[dict[str, Any]] = []

    for line in _log_lines(path):
        line = line.strip()
        if not line:
            continue
        try:
            line.encode("utf-8")
        except UnicodeEncodeError:
            continue  # Not UTF-8, such as a last line cut mid-character.
        try:
            raw = json.loads(line)
        except (ValueError, RecursionError):
            continue  # Not JSON, nested too deep, or a number too long to convert.
        if not isinstance(raw, dict) or _too_deep(raw):
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

        if not cwd and isinstance(raw.get("cwd"), str):
            cwd = raw["cwd"]

        if not session_id:
            session_id = _usable_session_id(raw.get("sessionId"))
            extra_meta["git_branch"] = raw.get("gitBranch", "")
            extra_meta["version"] = raw.get("version", "")

    # Only Claude's own sessionId becomes gen_ai.conversation.id; the file-name
    # fallback below is not a native conversation ID.
    conversation_id = session_id
    if not session_id:
        session_id = _usable_session_id(path.stem)
    if not session_id:
        raise ValueError("no usable session ID in the log or the file name")

    # Create session
    meta = SessionMeta(
        session_id=session_id,
        started_at=first_ts,
        agent_name="claude-code",
        command=f"imported from {path.name} (branch: {extra_meta.get('git_branch', '')}, v{extra_meta.get('version', '')})",
        conversation_id=conversation_id,
    )
    events: list[TraceEvent] = []

    # Always open the session, since a session_end is always written below.
    start_data: dict[str, Any] = {"mode": "claude-jsonl-import", "provider": "claude"}
    if cwd:
        start_data["cwd"] = cwd
    events.append(
        TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=first_ts,
            session_id=meta.session_id,
            data=start_data,
        ),
    )

    # UUID ancestry keeps separate Claude sidechains separate even when the
    # source version does not include an explicit agentId on every entry.
    stream_by_uuid: dict[str, str] = {}
    # Events whose text was redacted before it was cut. The store cannot flag
    # them itself when the cut lands inside a redaction marker.
    redacted_events: list[TraceEvent] = []
    pending_compaction: set[str] = set()
    pending_context: dict[str, str] = {}

    # Claude Code writes one line per content block of a response, and each
    # line repeats the response's usage. Count it once, on the last line.
    usage_line: dict[str, int] = {}
    for index, raw in enumerate(entries):
        msg = raw.get("message")
        if raw.get("type") != "assistant" or not isinstance(msg, dict):
            continue
        message_id = msg.get("id")
        usage = msg.get("usage")
        if (
            isinstance(message_id, str) and message_id
            and isinstance(usage, dict) and usage
        ):
            usage_line[message_id] = index

    # Second pass: convert entries to TraceEvents
    for index, raw in enumerate(entries):
        entry_type = raw.get("type", "")
        ts = _parse_iso_timestamp(raw.get("timestamp", ""))
        msg = raw.get("message", {})
        if not isinstance(msg, dict):
            msg = {}

        content = msg.get("content", "")
        usage = msg.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        message_id = msg.get("id")
        usage_deferred = (
            isinstance(message_id, str) and usage_line.get(message_id, index) != index
        )
        if usage_deferred:
            usage = {}
        model = msg.get("model", "")
        is_sidechain = raw.get("isSidechain") is True
        raw_uuid = raw.get("uuid", "")
        if not isinstance(raw_uuid, str):
            raw_uuid = ""
        # Claude Code writes "parentUuid": null on the first entry of a conversation.
        parent_uuid = raw.get("parentUuid", "")
        if not isinstance(parent_uuid, str):
            parent_uuid = ""
        if is_sidechain:
            explicit_id = _explicit_sidechain_id(raw)
            inherited = stream_by_uuid.get(parent_uuid, "")
            if explicit_id:
                context_stream = f"sidechain:{explicit_id}"
            elif inherited and inherited != "main":
                context_stream = inherited
            else:
                context_stream = f"sidechain:{raw_uuid or parent_uuid or 'unknown'}"
        else:
            context_stream = "main"
        if raw_uuid:
            stream_by_uuid[raw_uuid] = context_stream

        provenance: dict[str, Any] = {"context_stream": context_stream}
        if is_sidechain:
            provenance["is_sidechain"] = True
        if raw_uuid:
            provenance["source_uuid"] = raw_uuid
        if parent_uuid:
            provenance["source_parent_uuid"] = parent_uuid
        explicit_id = _explicit_sidechain_id(raw)
        if explicit_id:
            provenance["source_agent_id"] = explicit_id

        # User entry
        if entry_type == "user":
            text = _extract_text(content)

            # Check for tool results in user messages
            if isinstance(content, list):
                tool_results = _extract_tool_results(content)
                for tr in tool_results:
                    _c, hit = _redact_text(tr["content"], store.redact)
                    preview = (_c[:2000] + "..." if len(_c) > 2000 else _c) if _c else ""
                    event = TraceEvent(
                        event_type=EventType.TOOL_RESULT,
                        timestamp=ts,
                        session_id=meta.session_id,
                        data={
                            "tool_use_id": tr["tool_use_id"],
                            "content_preview": preview,
                            **({"is_error": True} if tr["is_error"] else {}),
                            **provenance,
                        },
                    )
                    if hit:
                        redacted_events.append(event)
                    events.append(event)

                # Also check toolUseResult — only if no content-block tool_results
                # were already emitted to avoid duplicate TOOL_RESULT events.
                tr_data = raw.get("toolUseResult")
                if tr_data and isinstance(tr_data, dict) and not tool_results:
                    stdout = tr_data.get("stdout", "")
                    if not isinstance(stdout, str):
                        stdout = ""
                    stderr = tr_data.get("stderr", "")
                    if not isinstance(stderr, str):
                        stderr = ""
                    if stdout or stderr:
                        stdout, out_hit = _redact_text(stdout, store.redact)
                        stderr, err_hit = _redact_text(stderr, store.redact)
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
                                **provenance,
                            },
                        )
                        if out_hit or err_hit:
                            redacted_events.append(event)
                        events.append(event)

            if text and not text.startswith("{"):
                is_summary = _is_compaction_summary(
                    raw, text, context_stream in pending_compaction,
                )
                prompt, hit = _redact_text(text, store.redact)
                prompt_data: dict[str, Any] = {
                    "prompt": prompt[:2000],
                    **provenance,
                }
                if is_summary:
                    prompt_data.update({
                        "is_compaction_summary": True,
                        "context_summary": text,
                    })
                    pending_context[context_stream] = text
                    pending_compaction.discard(context_stream)
                event = TraceEvent(
                    event_type=EventType.USER_PROMPT,
                    timestamp=ts,
                    session_id=meta.session_id,
                    data=prompt_data,
                )
                if hit:
                    redacted_events.append(event)
                events.append(event)

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
                        **provenance,
                    }
                    caller = tc.get("caller", {})
                    if isinstance(caller, dict) and caller.get("type"):
                        tool_data["caller_type"] = caller["type"]
                    # Extract subagent info from Agent tool
                    if tc["name"] == "Agent":
                        inp = tc["input"]
                        if isinstance(inp, dict) and inp.get("subagent_type"):
                            tool_data["subagent_type"] = inp["subagent_type"]

                    event = TraceEvent(
                        event_type=EventType.TOOL_CALL,
                        timestamp=ts,
                        session_id=meta.session_id,
                        data=tool_data,
                    )
                    events.append(event)
                    meta.tool_calls += 1

            # Log assistant text whenever present, including messages that also
            # contain tool calls (Claude Code often emits reasoning text alongside
            # a tool_use block in the same message).
            if text or usage:
                hit = False
                response_data: dict[str, Any] = {
                    "model": model,
                    **provenance,
                }
                if text:
                    response_text, hit = _redact_text(text, store.redact)
                    response_data["text"] = response_text[:2000]
                if isinstance(usage, dict) and usage:
                    response_data["usage"] = dict(usage)
                # A post-compaction summary belongs with the response's usage.
                summary = (
                    "" if usage_deferred else pending_context.pop(context_stream, "")
                )
                if summary:
                    response_data["context_summary"] = summary
                event = TraceEvent(
                    event_type=EventType.ASSISTANT_RESPONSE,
                    timestamp=ts,
                    session_id=meta.session_id,
                    data=response_data,
                )
                if hit:
                    redacted_events.append(event)
                events.append(event)

            # Track token usage
            if usage:
                meta.total_tokens += sum(_token_count(usage.get(key)) for key in _TOKEN_KEYS)
                meta.llm_requests += 1

        # System entry (e.g., turn duration)
        elif entry_type == "system":
            subtype = raw.get("subtype", "")
            if not isinstance(subtype, str):
                subtype = ""
            if subtype == "turn_duration":
                duration_ms = raw.get("durationMs", 0)
                if usable_number(duration_ms):
                    meta.total_duration_ms += duration_ms
            elif subtype in {"compact_boundary", "compaction_boundary"}:
                summary = _extract_text(raw.get("content", ""))
                if not summary:
                    summary = _extract_text(msg.get("content", ""))
                if summary:
                    pending_context[context_stream] = summary
                    pending_compaction.discard(context_stream)
                else:
                    pending_compaction.add(context_stream)

    # Finalize session
    meta.ended_at = last_ts if last_ts > 0 else meta.started_at
    if meta.total_duration_ms == 0 and meta.ended_at > meta.started_at:
        meta.total_duration_ms = (meta.ended_at - meta.started_at) * 1000

    # Add session end event
    events.append(
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
    if merge:
        store.create_session(meta)
    elif store.create_session(meta, if_missing=True) is None:
        return meta.session_id, False
    for event in redacted_events:
        event.redacted = True
        sys.stderr.write("agent-strace: redacted secrets from trace event\n")
    store.append_events(meta.session_id, events)
    return meta.session_id, True


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
    projects_dir = _resolve(claude_dir) / "projects"
    try:
        project_dirs = sorted(projects_dir.iterdir())
    except OSError:
        return []

    sessions = []
    for project_dir in project_dirs:
        try:
            if not project_dir.is_dir():
                continue
            jsonl_files = sorted(project_dir.glob("*.jsonl"))
        except OSError:
            continue

        project_name = _decode_project_path(project_dir.name)

        for jsonl_file in jsonl_files:
            try:
                st = jsonl_file.stat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            # Claude Code names files <session-uuid>.jsonl, so the stem is the ID.
            sessions.append(
                {
                    "path": jsonl_file,
                    "project": project_name,
                    "session_id": jsonl_file.stem,
                    "size_kb": st.st_size // 1024,
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
