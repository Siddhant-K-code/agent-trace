"""Subagent tracing: correlate nested agent sessions into a parent-child tree.

Subagent sessions are linked via SessionMeta.parent_session_id and
parent_event_id. This module provides:

  - Tree building: reconstruct the full agent call tree from the store
  - Tree-aware replay: render the tree with inline subagent expansion
  - Aggregated stats: roll up tool calls, tokens, errors across the tree
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import TextIO

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
    all_meta = store.list_sessions()

    # Index by session_id for fast lookup
    meta_by_id: dict[str, SessionMeta] = {m.session_id: m for m in all_meta}

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
                node.children.append(_build(child_meta.session_id, current_depth + 1))

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
        # Keep the root's own duration as the floor; a child subtree longer
        # than the root would indicate clock skew — still take the max.
        stats.total_duration_ms = max(
            node.meta.total_duration_ms, child_stats.total_duration_ms
        )
    return stats


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


def _indent(depth: int) -> str:
    if depth == 0:
        return ""
    return "│  " * (depth - 1) + "├─ "


def format_tree(
    node: SessionNode,
    base_ts: float | None = None,
    out: TextIO = sys.stdout,
    expand: bool = True,
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
    """
    if base_ts is None:
        base_ts = node.meta.started_at

    indent = _indent(node.depth)
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
            elif tool_name.lower() == "agent":
                prompt = str(args.get("prompt", ""))
                detail = f"  \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\""

            subagent_tag = ""
            if event.data.get("is_sidechain"):
                subagent_tag = "  [sidechain]"
            if event.data.get("subagent_type"):
                subagent_tag += f"  [{event.data['subagent_type']}]"

            w(f"{indent}{ts_str}  → tool_call  {tool_name}{subagent_tag}{detail}\n")

            # Inline expand subagent if this tool_call spawned one
            if expand and event.event_id in children_by_event:
                child = children_by_event[event.event_id]
                format_tree(child, base_ts=base_ts, out=out, expand=expand)

        elif event.event_type == EventType.TOOL_RESULT:
            preview = (event.data.get("result", "") or
                       event.data.get("content_preview", ""))[:80]
            w(f"{indent}{ts_str}  ← tool_result"
              f"{'  ' + preview if preview else ''}\n")

        elif event.event_type == EventType.ERROR:
            msg = (event.data.get("message", "") or
                   event.data.get("error", ""))[:80]
            w(f"{indent}{ts_str}  ✗ error  {msg}\n")

        elif event.event_type == EventType.USER_PROMPT:
            prompt = event.data.get("prompt", "")[:80]
            w(f"{indent}{ts_str}  👤 \"{prompt}\"\n")

        elif event.event_type == EventType.ASSISTANT_RESPONSE:
            text = event.data.get("text", "")[:80]
            w(f"{indent}{ts_str}  🤖 \"{text}\"\n")

        elif event.event_type == EventType.SESSION_END:
            w(f"{indent}{ts_str}  ■ session_end\n")

    w("\n")


def format_tree_summary(node: SessionNode, out: TextIO = sys.stdout) -> None:
    """Print a compact tree structure showing session hierarchy."""
    w = out.write
    indent = "  " * node.depth

    duration = node.meta.total_duration_ms / 1000 if node.meta.total_duration_ms else 0
    w(f"{indent}{'└─' if node.depth > 0 else '  '} {node.meta.session_id[:12]}"
      f"  {duration:.1f}s"
      f"  {node.meta.tool_calls} tools"
      f"  {node.meta.total_tokens:,} tokens"
      f"{'  ✗ ' + str(node.meta.errors) + ' errors' if node.meta.errors else ''}\n")

    for child in node.children:
        format_tree_summary(child, out=out)


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
