"""Subagent tracing: correlate nested agent sessions into a parent-child tree.

Subagent sessions are linked via SessionMeta.parent_session_id and
parent_event_id. This module provides:

  - Tree building: reconstruct the full agent call tree from the store
  - Tree-aware replay: render the tree with inline subagent expansion
  - Aggregated stats: roll up tool calls, tokens, errors across the tree
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import TextIO

from .event_data import require_object_event_data
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore

MAX_DEPTH = 5  # configurable guard against runaway recursion


# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------

@dataclass
class SessionNode:
    meta: SessionMeta
    events: list[TraceEvent]
    children: list[SessionNode] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return self.meta.depth


@dataclass
class AggregatedStats:
    session_count: int = 0
    tool_calls: int = 0
    llm_requests: int = 0
    errors: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------

def build_tree(store: TraceStore, root_session_id: str) -> SessionNode:
    """Build a SessionNode tree rooted at *root_session_id*.

    Discovers child sessions by scanning all sessions for ones whose
    parent_session_id matches a session already in the tree.
    Depth is bounded by MAX_DEPTH.
    """
    try:
        root_meta = store.load_meta(root_session_id)
    except FileNotFoundError as exc:
        raise KeyError(f"Session not found in store: {root_session_id}") from exc
    tenant_id = root_meta.tenant_id
    all_meta = store.list_sessions()

    # Index by session_id for fast lookup
    meta_by_id: dict[str, SessionMeta] = {m.session_id: m for m in all_meta}

    if root_meta.parent_session_id and root_meta.parent_session_id in meta_by_id:
        parent = meta_by_id[root_meta.parent_session_id]
        if parent.tenant_id != tenant_id:
            raise ValueError("cross-tenant parent link rejected")

    # Index children by parent_session_id
    children_of: dict[str, list[SessionMeta]] = {}
    for m in all_meta:
        if m.parent_session_id:
            children_of.setdefault(m.parent_session_id, []).append(m)

    def _build(session_id: str, current_depth: int) -> SessionNode:
        if session_id not in meta_by_id:
            raise KeyError(f"Session not found in store: {session_id}")
        meta = meta_by_id[session_id]
        events = store.load_events(session_id)
        node = SessionNode(meta=meta, events=events)

        if current_depth < MAX_DEPTH:
            for child_meta in sorted(
                children_of.get(session_id, []),
                key=lambda m: m.started_at,
            ):
                if child_meta.tenant_id != tenant_id:
                    raise ValueError(
                        f"cross-tenant child link rejected for session {child_meta.session_id}"
                    )
                node.children.append(_build(child_meta.session_id, current_depth + 1))
        elif children_of.get(session_id):
            sys.stderr.write(
                f"agent-strace: subagent tree truncated at depth {MAX_DEPTH}"
                f" for session {session_id[:12]}\n"
            )

        return node

    return _build(root_session_id, 0)


def aggregate_stats(node: SessionNode) -> AggregatedStats:
    """Recursively aggregate stats across the full session tree."""
    stats = AggregatedStats(
        session_count=1,
        tool_calls=node.meta.tool_calls,
        llm_requests=node.meta.llm_requests,
        errors=node.meta.errors,
        total_tokens=node.meta.total_tokens,
        # Duration is wall-clock: subagents run within parent time, so we take
        # the max of the root's own duration and each child subtree's duration.
        total_duration_ms=node.meta.total_duration_ms,
    )
    for child in node.children:
        child_stats = aggregate_stats(child)
        stats.session_count += child_stats.session_count
        stats.tool_calls += child_stats.tool_calls
        stats.llm_requests += child_stats.llm_requests
        stats.errors += child_stats.errors
        stats.total_tokens += child_stats.total_tokens
        # Duration is wall-clock: subagents run within parent time, so take
        # the running max across all children (not the root's fixed value).
        stats.total_duration_ms = max(
            stats.total_duration_ms, child_stats.total_duration_ms
        )
    return stats


# ---------------------------------------------------------------------------
# In-session subagents
# ---------------------------------------------------------------------------
# Codex runs a subagent inside the parent session: its tool events and its
# subagent_start/subagent_stop events carry the subagent's ``agent_id`` and
# ``agent_type`` instead of belonging to a child session.

_IN_SESSION_SUBAGENT_EVENT_TYPES = frozenset({
    EventType.TOOL_CALL,
    EventType.ERROR,
    EventType.SUBAGENT_START,
    EventType.SUBAGENT_STOP,
})


@dataclass
class InSessionSubagent:
    agent_id: str
    agent_type: str = ""
    tool_calls: int = 0
    errors: int = 0


def _event_agent_id(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    agent_id = data.get("agent_id")
    return agent_id if isinstance(agent_id, str) else ""


def short_agent_id(agent_id: str) -> str:
    """Return the last 8 characters of *agent_id*.

    Codex agent IDs are UUIDv7, whose leading characters encode the creation
    time and repeat across subagents started close together; the trailing
    characters are random.
    """
    return agent_id[-8:]


def subagent_label(data: object) -> str:
    """Return ``"<agent_type> <short id>"`` for an attributed event, else ``""``."""
    agent_id = _event_agent_id(data)
    if not agent_id:
        return ""
    agent_type = data.get("agent_type")  # type: ignore[union-attr]
    short = short_agent_id(agent_id)
    return f"{agent_type} {short}" if isinstance(agent_type, str) and agent_type else short


def in_session_subagents(events: list[TraceEvent]) -> list[InSessionSubagent]:
    """Group a session's attributed events by ``agent_id``, in first-seen order.

    Events whose data is not a JSON object are skipped.
    """
    by_id: dict[str, InSessionSubagent] = {}
    for event in events:
        if event.event_type not in _IN_SESSION_SUBAGENT_EVENT_TYPES:
            continue
        agent_id = _event_agent_id(event.data)
        if not agent_id:
            continue
        entry = by_id.setdefault(agent_id, InSessionSubagent(agent_id=agent_id))
        agent_type = event.data.get("agent_type")
        if not entry.agent_type and isinstance(agent_type, str):
            entry.agent_type = agent_type
        if event.event_type == EventType.TOOL_CALL:
            entry.tool_calls += 1
        elif event.event_type == EventType.ERROR:
            entry.errors += 1
    return list(by_id.values())


def describe_subagent(sub: InSessionSubagent) -> str:
    """Return ``"<agent_type> <short id>  N tools[  ✗ N errors]"``."""
    kind = f"{sub.agent_type} " if sub.agent_type else ""
    errors = f"  ✗ {sub.errors} errors" if sub.errors else ""
    return f"{kind}{short_agent_id(sub.agent_id)}  {sub.tool_calls} tools{errors}"


def _subagent_summary_line(sub: InSessionSubagent) -> str:
    return f"subagent {describe_subagent(sub)}"


def _returned_preview(event: TraceEvent, limit: int = 80) -> str:
    """Return a one-line preview of what a subagent returned, else ``""``."""
    if event.event_type != EventType.SUBAGENT_STOP or not isinstance(event.data, dict):
        return ""
    text = event.data.get("text")
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    return text[:limit] + "..." if len(text) > limit else text


# ---------------------------------------------------------------------------
# Tree-aware replay formatting
# ---------------------------------------------------------------------------

def _fmt_offset(base_ts: float, ts: float) -> str:
    offset = max(0.0, ts - base_ts)
    if offset < 60:
        return f"+{offset:5.2f}s"
    m = int(offset) // 60
    s = offset % 60
    return f"+{m}m{s:04.1f}s"


def _indent(depth: int, last_child: bool = False) -> str:
    if depth == 0:
        return ""
    connector = "└─ " if last_child else "├─ "
    return "│  " * (depth - 1) + connector


# format_tree reads the data of events of these types.
TREE_DATA_EVENT_TYPES = frozenset({
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.ERROR,
    EventType.USER_PROMPT,
    EventType.ASSISTANT_RESPONSE,
    EventType.SUBAGENT_START,
    EventType.SUBAGENT_STOP,
})


def require_tree_event_data(node: SessionNode, expand: bool = True) -> None:
    """Raise NonObjectEventDataError if format_tree would read non-object data.

    Checks every session format_tree renders: *node* and, when *expand* is
    set, each child it inlines under the tool_call that spawned it.
    """
    require_object_event_data(
        node.events,
        event_types=TREE_DATA_EVENT_TYPES,
        session_id=node.meta.session_id,
    )
    if not expand:
        return
    children_by_event = {
        c.meta.parent_event_id: c for c in node.children if c.meta.parent_event_id
    }
    for event in node.events:
        if event.event_type == EventType.TOOL_CALL and event.event_id in children_by_event:
            require_tree_event_data(children_by_event[event.event_id], expand)


def format_tree(
    node: SessionNode,
    base_ts: float | None = None,
    out: TextIO = sys.stdout,
    expand: bool = True,
    last_child: bool = False,
) -> None:
    """Render the session tree to *out*.

    Parameters
    ----------
    node : SessionNode
        Root of the tree to render.
    base_ts : float, optional
        Timestamp origin for relative offsets. Defaults to root session start.
    expand : bool
        If True, inline subagent events under their parent tool_call.
    last_child : bool
        Whether this node is the last child of its parent (affects tree chars).
    """
    if base_ts is None:
        base_ts = node.meta.started_at

    indent = _indent(node.depth, last_child=last_child)
    w = out.write

    # Session header
    w(f"{indent}▶ session_start  {node.meta.session_id[:12]}"
      f"  agent={node.meta.agent_name or 'unknown'}"
      f"  depth={node.depth}\n")

    # Build a lookup of child sessions by the parent_event_id that spawned them
    children_by_event: dict[str, SessionNode] = {
        c.meta.parent_event_id: c for c in node.children if c.meta.parent_event_id
    }

    for event in node.events:
        ts_str = _fmt_offset(base_ts, event.timestamp)
        etype = event.event_type.value

        if event.event_type == EventType.TOOL_CALL:
            tool_name = event.data.get("tool_name", "?")
            args = event.data.get("arguments", {})
            detail = ""
            if tool_name.lower() == "bash":
                cmd = str(args.get("command", ""))
                detail = f"  $ {cmd[:80]}{'...' if len(cmd) > 80 else ''}"
            elif tool_name.lower() in ("read", "write", "edit"):
                detail = f"  {args.get('file_path', '')}"
            elif tool_name.lower() in ("agent", "task"):
                prompt = str(args.get("prompt", ""))
                detail = f"  \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\""

            subagent_tag = ""
            if event.data.get("is_sidechain"):
                subagent_tag = "  [sidechain]"
            if event.data.get("subagent_type"):
                subagent_tag += f"  [{event.data['subagent_type']}]"
            label = subagent_label(event.data)
            if label:
                subagent_tag += f"  [subagent {label}]"

            w(f"{indent}{ts_str}  → tool_call  {tool_name}{subagent_tag}{detail}\n")

            # Inline expand subagent if this tool_call spawned one
            if expand and event.event_id in children_by_event:
                child = children_by_event[event.event_id]
                is_last = child == node.children[-1] if node.children else True
                format_tree(child, base_ts=base_ts, out=out, expand=expand,
                            last_child=is_last)

        elif event.event_type == EventType.TOOL_RESULT:
            preview = (event.data.get("result", "") or
                       event.data.get("content_preview", ""))[:80]
            label = subagent_label(event.data)
            w(f"{indent}{ts_str}  ← tool_result"
              f"{'  [subagent ' + label + ']' if label else ''}"
              f"{'  ' + preview if preview else ''}\n")

        elif event.event_type == EventType.ERROR:
            msg = (event.data.get("message", "") or
                   event.data.get("error", ""))[:80]
            label = subagent_label(event.data)
            w(f"{indent}{ts_str}  ✗ error"
              f"{'  [subagent ' + label + ']' if label else ''}  {msg}\n")

        elif event.event_type == EventType.USER_PROMPT:
            prompt = event.data.get("prompt", "")[:80]
            w(f"{indent}{ts_str}  👤 \"{prompt}\"\n")

        elif event.event_type == EventType.ASSISTANT_RESPONSE:
            text = event.data.get("text", "")[:80]
            w(f"{indent}{ts_str}  🤖 \"{text}\"\n")

        elif event.event_type in (EventType.SUBAGENT_START, EventType.SUBAGENT_STOP):
            agent = event.data.get("agent_type") or event.data.get("agent_name") or ""
            child = str(event.data.get("child_session_id", ""))[:12]
            agent_id = _event_agent_id(event.data)
            icon = "⤷" if event.event_type == EventType.SUBAGENT_START else "⤶"
            w(f"{indent}{ts_str}  {icon} {event.event_type.value}"
              f"{'  ' + agent if agent else ''}"
              f"{'  child=' + child if child else ''}"
              f"{'  agent=' + short_agent_id(agent_id) if agent_id else ''}\n")
            returned = _returned_preview(event)
            if returned:
                w(f"{indent}{' ' * len(ts_str)}    returned: \"{returned}\"\n")

        elif event.event_type == EventType.SESSION_END:
            w(f"{indent}{ts_str}  ■ session_end\n")

    w("\n")


def format_tree_summary(
    node: SessionNode,
    out: TextIO = sys.stdout,
    last_child: bool = False,
) -> None:
    """Print a compact tree structure showing session hierarchy."""
    w = out.write
    indent = _indent(node.depth, last_child=last_child)

    duration = node.meta.total_duration_ms / 1000 if node.meta.total_duration_ms else 0
    w(f"{indent}{node.meta.session_id[:12]}"
      f"  {duration:.1f}s"
      f"  {node.meta.tool_calls} tools"
      f"  {node.meta.total_tokens:,} tokens"
      f"{'  ✗ ' + str(node.meta.errors) + ' errors' if node.meta.errors else ''}\n")

    _write_subagent_lines(node, out)
    for i, child in enumerate(node.children):
        format_tree_summary(child, out=out, last_child=(i == len(node.children) - 1))


def _write_subagent_lines(node: SessionNode, out: TextIO) -> None:
    """Write one line per in-session subagent, before any child sessions."""
    subagents = in_session_subagents(node.events)
    for i, sub in enumerate(subagents):
        last = i == len(subagents) - 1 and not node.children
        out.write(f"{_indent(node.depth + 1, last_child=last)}{_subagent_summary_line(sub)}\n")


def _status(meta: SessionMeta) -> str:
    if meta.errors:
        return "error"
    return "ok"


def _duration_seconds(meta: SessionMeta) -> float:
    if meta.total_duration_ms:
        return meta.total_duration_ms / 1000
    if meta.ended_at and meta.started_at:
        return max(0.0, meta.ended_at - meta.started_at)
    return 0.0


def _node_cost(store: TraceStore, session_id: str) -> float:
    try:
        from .cost import estimate_cost
        return estimate_cost(store, session_id).total_cost
    except Exception:
        return 0.0


def _cost_map(store: TraceStore, node: SessionNode) -> dict[str, float]:
    costs = {node.meta.session_id: _node_cost(store, node.meta.session_id)}
    for child in node.children:
        costs.update(_cost_map(store, child))
    return costs


def tree_to_dict(
    node: SessionNode,
    costs: dict[str, float] | None = None,
) -> dict:
    costs = costs or {}
    data = {
        "session_id": node.meta.session_id,
        "agent": node.meta.agent_name or node.meta.command or "",
        "depth": node.depth,
        "parent_session_id": node.meta.parent_session_id,
        "parent_event_id": node.meta.parent_event_id,
        "cost_usd": round(costs.get(node.meta.session_id, 0.0), 6),
        "tool_calls": node.meta.tool_calls,
        "llm_requests": node.meta.llm_requests,
        "errors": node.meta.errors,
        "status": _status(node.meta),
        "duration_s": round(_duration_seconds(node.meta), 3),
        "children": [tree_to_dict(child, costs) for child in node.children],
    }
    subagents = in_session_subagents(node.events)
    if subagents:
        data["subagents"] = [asdict(sub) for sub in subagents]
    return data


def format_session_tree(
    node: SessionNode,
    costs: dict[str, float] | None = None,
    out: TextIO = sys.stdout,
    last_child: bool = False,
) -> None:
    """Print a compact tree with per-session cost, calls, status, and duration."""
    costs = costs or {}
    indent = _indent(node.depth, last_child=last_child)
    duration = _duration_seconds(node.meta)
    status = "✗" if node.meta.errors else "✓"
    agent = node.meta.agent_name or node.meta.command or "session"
    out.write(
        f"{indent}{node.meta.session_id[:12]}  "
        f"{agent[:28]:<28}  "
        f"${costs.get(node.meta.session_id, 0.0):.4f}  "
        f"{node.meta.tool_calls} tools  "
        f"{duration:.1f}s  "
        f"{status}"
    )
    if node.meta.errors:
        out.write(f"  {node.meta.errors} errors")
    out.write("\n")

    _write_subagent_lines(node, out)
    for i, child in enumerate(node.children):
        format_session_tree(
            child,
            costs=costs,
            out=out,
            last_child=(i == len(node.children) - 1),
        )


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------

def cmd_replay_tree(args: argparse.Namespace) -> int:
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

    tree = build_tree(store, full_id)
    expand = not getattr(args, "tree_only", False)

    if getattr(args, "tree", False):
        stats = aggregate_stats(tree)
        sys.stdout.write(f"\nSession tree for {full_id[:12]}\n\n")
        format_tree_summary(tree)
        sys.stdout.write(
            f"\nTotal: {stats.session_count} sessions, "
            f"{stats.tool_calls} tool calls, "
            f"{stats.llm_requests} LLM requests, "
            f"{stats.total_tokens:,} tokens"
            f"{', ' + str(stats.errors) + ' errors' if stats.errors else ''}\n\n"
        )
    else:
        require_tree_event_data(tree, expand=expand)
        format_tree(tree, expand=expand)

    return 0


def cmd_stats_tree(args: argparse.Namespace) -> int:
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

    tree = build_tree(store, full_id)
    stats = aggregate_stats(tree)

    sys.stdout.write(f"\nAggregated stats for {full_id[:12]} (including subagents)\n\n")
    sys.stdout.write(f"  Sessions:      {stats.session_count}\n")
    sys.stdout.write(f"  Tool calls:    {stats.tool_calls}\n")
    sys.stdout.write(f"  LLM requests:  {stats.llm_requests}\n")
    sys.stdout.write(f"  Total tokens:  {stats.total_tokens:,}\n")
    sys.stdout.write(f"  Errors:        {stats.errors}\n")
    sys.stdout.write(f"  Duration:      {stats.total_duration_ms / 1000:.1f}s\n\n")

    if tree.children:
        sys.stdout.write("Session tree:\n\n")
        format_tree_summary(tree)
        sys.stdout.write("\n")

    return 0


def cmd_tree(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    session_id = args.session_id or store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    full_id = store.find_session(session_id)
    if not full_id:
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    try:
        tree = build_tree(store, full_id)
    except KeyError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    costs = _cost_map(store, tree)
    stats = aggregate_stats(tree)
    if getattr(args, "format", "text") == "json":
        data = {
            "root_session_id": full_id,
            "total_sessions": stats.session_count,
            "total_cost_usd": round(sum(costs.values()), 6),
            "total_tool_calls": stats.tool_calls,
            "total_llm_requests": stats.llm_requests,
            "total_errors": stats.errors,
            "tree": tree_to_dict(tree, costs),
        }
        sys.stdout.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return 0

    sys.stdout.write(f"\nSession tree for {full_id[:12]}\n\n")
    format_session_tree(tree, costs=costs)
    sys.stdout.write(
        f"\nTotal: {stats.session_count} sessions, "
        f"${sum(costs.values()):.4f}, "
        f"{stats.tool_calls} tool calls, "
        f"{stats.llm_requests} LLM requests"
        f"{', ' + str(stats.errors) + ' errors' if stats.errors else ''}\n\n"
    )
    return 0
