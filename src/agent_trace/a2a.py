"""A2A (Agent-to-Agent) protocol support.

Extends agent-trace to capture agent-to-agent calls as first-class events,
with cross-agent trace correlation. Intercepts A2A HTTP calls via the
existing proxy infrastructure and links sub-sessions via parent_session_id.

A2A calls are stored as tool_call events with ``event_subtype: a2a_call``:
  - agent_id: identifier of the called agent
  - task: the task description sent to the sub-agent
  - response: the sub-agent's response payload
  - sub_session_id: session ID of the called agent (if it runs agent-trace)
  - duration_ms: round-trip time
  - cost_usd: estimated cost of the sub-agent call

Usage:
    agent-strace a2a-tree <session-id>   # show the full agent call graph
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, TextIO

from .event_data import require_object_event_data
from .inert import inert
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# A2A event type (extends EventType without modifying the enum)
# ---------------------------------------------------------------------------

A2A_CALL = "a2a_call"   # event_type value for agent-to-agent calls

# A2A protocol detection patterns (Google A2A spec, April 2026)
A2A_CONTENT_TYPES = {
    "application/json",
    "application/a2a+json",
}

A2A_PATH_PATTERNS = [
    "/a2a/",
    "/agent/",
    "/.well-known/agent.json",
]

A2A_HEADER_PATTERNS = [
    "x-a2a-",
    "x-agent-",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class A2ACallEvent:
    """Structured representation of an A2A call extracted from a TraceEvent."""
    event_id: str
    session_id: str
    timestamp: float
    agent_id: str
    agent_url: str
    task: str
    response: dict[str, Any]
    sub_session_id: str       # empty if called agent doesn't run agent-trace
    duration_ms: float
    cost_usd: float
    success: bool
    error: str = ""

    @classmethod
    def from_trace_event(cls, event: TraceEvent) -> "A2ACallEvent | None":
        """Extract A2ACallEvent from a raw TraceEvent, or None if not an A2A call."""
        if event.data.get("event_subtype") != A2A_CALL:
            return None
        d = event.data
        return cls(
            event_id=event.event_id,
            session_id=event.session_id,
            timestamp=event.timestamp,
            agent_id=d.get("agent_id", ""),
            agent_url=d.get("agent_url", ""),
            task=d.get("task", ""),
            response=d.get("response", {}),
            sub_session_id=d.get("sub_session_id", ""),
            duration_ms=d.get("duration_ms", 0.0),
            cost_usd=d.get("cost_usd", 0.0),
            success=d.get("success", True),
            error=d.get("error", ""),
        )


@dataclass
class AgentNode:
    """Node in the agent call graph."""
    session_id: str
    agent_id: str
    depth: int
    cost_usd: float
    duration_ms: float
    tool_calls: int
    children: list["AgentNode"] = field(default_factory=list)
    a2a_calls: list[A2ACallEvent] = field(default_factory=list)
    # A2A calls in the parent session that name this session as their
    # sub-session. 0 for the root and for a child linked only by its meta.
    delegations: int = 0
    # "" for a node shown in full. "cycle" for a session that is already on
    # the path from the root, "shared" for one already shown in full under
    # another parent. A reference has no children and is not counted twice.
    ref: str = ""


@dataclass
class A2ATreeReport:
    root: AgentNode
    total_cost: float
    total_agents: int
    max_depth: int


# ---------------------------------------------------------------------------
# A2A call detection (for http_proxy.py integration)
# ---------------------------------------------------------------------------

def is_a2a_request(method: str, path: str, headers: dict[str, str], body: bytes) -> bool:
    """Return True if an HTTP request looks like an A2A protocol call."""
    # Path pattern match
    path_lower = path.lower()
    if any(pat in path_lower for pat in A2A_PATH_PATTERNS):
        return True

    # A2A-specific headers
    headers_lower = {k.lower(): v for k, v in headers.items()}
    if any(any(h in k for h in A2A_HEADER_PATTERNS) for k in headers_lower):
        return True

    # Content-type check for POST
    if method.upper() == "POST":
        ct = headers_lower.get("content-type", "")
        if any(a2a_ct in ct for a2a_ct in A2A_CONTENT_TYPES):
            return True

        # Body heuristic: A2A task objects have a "task" or "message" key
        if body:
            try:
                payload = json.loads(body)
                if isinstance(payload, dict) and (
                    "task" in payload or
                    ("jsonrpc" in payload and payload.get("method") in ("tasks/send", "tasks/get"))
                ):
                    return True
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

    return False


def make_a2a_event(
    session_id: str,
    agent_id: str,
    agent_url: str,
    task: str,
    response: dict,
    sub_session_id: str = "",
    duration_ms: float = 0.0,
    cost_usd: float = 0.0,
    success: bool = True,
    error: str = "",
) -> TraceEvent:
    """Create a TraceEvent representing an A2A call."""
    return TraceEvent(
        event_type=EventType.TOOL_CALL,   # stored as TOOL_CALL with subtype
        session_id=session_id,
        data={
            "event_subtype": A2A_CALL,
            "agent_id": agent_id,
            "agent_url": agent_url,
            "task": task,
            "response": response,
            "sub_session_id": sub_session_id,
            "duration_ms": duration_ms,
            "cost_usd": cost_usd,
            "success": success,
            "error": error,
        },
    )


# ---------------------------------------------------------------------------
# Sub-session linking
# ---------------------------------------------------------------------------

def link_sub_session(
    store: TraceStore,
    parent_session_id: str,
    parent_event_id: str,
    child_session_id: str,
    depth: int | None = None,
) -> None:
    """Update child session meta to link it to its parent A2A call.

    *depth* defaults to the parent's depth plus one. The parent's depth is
    read and the child's meta is changed under one store lock (see
    ``TraceStore.link_parent``), so updates the child agent writes to its
    own meta at the same time are kept. Does nothing if the store refuses
    the link.
    """
    try:
        if depth is None:
            store.link_parent(child_session_id, parent_session_id, parent_event_id)
            return

        def _link(meta: SessionMeta) -> None:
            meta.parent_session_id = parent_session_id
            meta.parent_event_id = parent_event_id
            meta.depth = depth

        store.mutate_meta(child_session_id, _link)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Call graph construction
# ---------------------------------------------------------------------------

def _extract_a2a_calls(events: list[TraceEvent]) -> list[A2ACallEvent]:
    calls = []
    for e in events:
        call = A2ACallEvent.from_trace_event(e)
        if call:
            calls.append(call)
    return calls


def build_a2a_tree(store: TraceStore, root_session_id: str) -> A2ATreeReport:
    """Build the full agent call graph rooted at root_session_id.

    Raises NonObjectEventDataError, naming the session, when an event in a
    session of the tree has data that is not an object.
    """
    # A session already on the path from the root is a cycle. A session
    # already shown in full elsewhere (a diamond) is a shared reference.
    on_path: set[str] = set()
    built: dict[str, AgentNode] = {}

    def _child_ids(sid: str, a2a_calls: list[A2ACallEvent]) -> dict[str, int]:
        """Return each child session once, in first-seen order, with the
        number of A2A calls that name it."""
        counts: dict[str, int] = {}
        for call in a2a_calls:
            if call.sub_session_id:
                counts[call.sub_session_id] = counts.get(call.sub_session_id, 0) + 1
        # Also find sessions that declare this as parent
        for m in store.list_sessions():
            if m.parent_session_id == sid and m.session_id not in counts:
                counts[m.session_id] = 0
        return counts

    def _build_node(sid: str, depth: int, delegations: int) -> AgentNode:
        if sid in on_path or sid in built:
            target = built[sid]
            return AgentNode(
                session_id=sid, agent_id=target.agent_id, depth=depth,
                cost_usd=0, duration_ms=0, tool_calls=0,
                delegations=delegations,
                ref="cycle" if sid in on_path else "shared",
            )

        try:
            meta = store.load_meta(sid)
            events = store.load_events(sid)
        except Exception:
            node = AgentNode(session_id=sid, agent_id="[not found]", depth=depth,
                             cost_usd=0, duration_ms=0, tool_calls=0,
                             delegations=delegations)
            built[sid] = node
            return node

        # _extract_a2a_calls reads the data of every event.
        require_object_event_data(events, session_id=sid)
        a2a_calls = _extract_a2a_calls(events)

        try:
            from .cost import estimate_cost
            cost = estimate_cost(store, sid).total_cost
        except Exception:
            cost = 0.0

        node = AgentNode(
            session_id=sid,
            agent_id=meta.agent_name or sid[:12],
            depth=depth,
            cost_usd=cost,
            duration_ms=meta.total_duration_ms,
            tool_calls=meta.tool_calls,
            a2a_calls=a2a_calls,
            delegations=delegations,
        )
        built[sid] = node

        on_path.add(sid)
        try:
            for child_id, count in _child_ids(sid, a2a_calls).items():
                node.children.append(_build_node(child_id, depth + 1, count))
        finally:
            on_path.discard(sid)

        return node

    root = _build_node(root_session_id, 0, 0)

    # References add nothing: their session is counted where it is in full.
    def _total_cost(node: AgentNode) -> float:
        return node.cost_usd + sum(_total_cost(c) for c in node.children)

    def _total_agents(node: AgentNode) -> int:
        if node.ref:
            return 0
        return 1 + sum(_total_agents(c) for c in node.children)

    def _max_depth(node: AgentNode) -> int:
        return max([node.depth] + [_max_depth(c) for c in node.children if not c.ref])

    return A2ATreeReport(
        root=root,
        total_cost=_total_cost(root),
        total_agents=_total_agents(root),
        max_depth=_max_depth(root),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_a2a_tree(report: A2ATreeReport, out: TextIO = sys.stdout) -> None:
    """Write the agent call graph to *out*.

    Each value from a trace or a session goes through inert(), so an escape
    code or a line break in it shows as text. In a task, each run of
    whitespace becomes one space first.
    """
    w = out.write
    sep = "─" * 55

    w(f"\nAgent Call Graph\n{sep}\n")
    w(f"Total agents: {report.total_agents}  "
      f"Total cost: ${report.total_cost:.4f}  "
      f"Max depth: {report.max_depth}\n{sep}\n\n")

    def _render(node: AgentNode, prefix: str = "", is_last: bool = True) -> None:
        connector = "└── " if is_last else "├── "
        times = [f"{node.delegations} delegations"] if node.delegations > 1 else []
        if node.ref == "cycle":
            label, details = f"[cycle] → {inert(node.agent_id)}", times
        elif node.ref == "shared":
            label, details = inert(node.agent_id), ["shown above"] + times
        else:
            dur = f"{node.duration_ms/1000:.1f}s" if node.duration_ms else "?"
            label = inert(node.agent_id)
            details = [f"{inert(node.tool_calls)} calls", f"${node.cost_usd:.4f}", dur] + times
        w(f"{prefix}{connector}{label}"
          f"{'  [' + ' · '.join(details) + ']' if details else ''}\n")
        if node.ref:
            return

        child_prefix = prefix + ("    " if is_last else "│   ")

        for i, call in enumerate(node.a2a_calls):
            is_last_call = (i == len(node.a2a_calls) - 1) and not node.children
            call_connector = "└── " if is_last_call else "├── "
            status = "✅" if call.success else "❌"
            w(f"{child_prefix}{call_connector}a2a → {inert(call.agent_id)}  "
              f"{status}  ${call.cost_usd:.4f}  {call.duration_ms:.0f}ms\n")
            if call.task:
                task = " ".join(str(call.task).split())
                w(f"{child_prefix}{'    ' if is_last_call else '│   '}"
                  f"task: {inert(task[:60])}\n")

        for i, child in enumerate(node.children):
            _render(child, child_prefix, is_last=(i == len(node.children) - 1))

    _render(report.root)
    w(f"\n{sep}\n\n")


# ---------------------------------------------------------------------------
# OTLP span export for A2A calls
# ---------------------------------------------------------------------------

def a2a_calls_to_otlp_spans(
    report: A2ATreeReport,
    service_name: str = "agent-trace",
) -> list[dict]:
    """Convert A2A call graph to OTLP-compatible span dicts.

    Each session gets one span. A cycle or a shared sub-session adds a link
    on its session's span to the span of the agent that called it, with
    ``agent.link`` set to ``cycle`` or ``shared``.
    """
    spans = []
    trace_id = report.root.session_id[:32].ljust(32, "0")
    span_by_session: dict[str, dict] = {}
    links: list[tuple[str, dict]] = []

    def _node_to_spans(node: AgentNode, parent_span_id: str = "") -> None:
        if node.ref:
            links.append((node.session_id, {
                "traceId": trace_id,
                "spanId": parent_span_id,
                "attributes": [
                    {"key": "agent.link", "value": {"stringValue": node.ref}},
                    {"key": "agent.delegations", "value": {"intValue": node.delegations}},
                ],
            }))
            return
        span_id = node.session_id[:16]
        span: dict = {
            "traceId": trace_id,
            "spanId": span_id,
            "name": f"agent:{node.agent_id}",
            "kind": 3,  # CLIENT
            "startTimeUnixNano": 0,
            "endTimeUnixNano": int(node.duration_ms * 1_000_000),
            "attributes": [
                {"key": "agent.id", "value": {"stringValue": node.agent_id}},
                {"key": "agent.session_id", "value": {"stringValue": node.session_id}},
                {"key": "agent.cost_usd", "value": {"doubleValue": node.cost_usd}},
                {"key": "agent.tool_calls", "value": {"intValue": node.tool_calls}},
                {"key": "agent.depth", "value": {"intValue": node.depth}},
                {"key": "agent.delegations", "value": {"intValue": node.delegations}},
            ],
        }
        if parent_span_id:
            span["parentSpanId"] = parent_span_id
        spans.append(span)
        span_by_session[node.session_id] = span

        for call in node.a2a_calls:
            call_span_id = call.event_id[:16]
            spans.append({
                "traceId": trace_id,
                "spanId": call_span_id,
                "parentSpanId": span_id,
                "name": f"a2a_call:{call.agent_id}",
                "kind": 3,
                "startTimeUnixNano": int(call.timestamp * 1_000_000_000),
                "endTimeUnixNano": int((call.timestamp + call.duration_ms / 1000) * 1_000_000_000),
                "attributes": [
                    {"key": "a2a.agent_id", "value": {"stringValue": call.agent_id}},
                    {"key": "a2a.agent_url", "value": {"stringValue": call.agent_url}},
                    {"key": "a2a.task", "value": {"stringValue": call.task[:200]}},
                    {"key": "a2a.success", "value": {"boolValue": call.success}},
                    {"key": "a2a.cost_usd", "value": {"doubleValue": call.cost_usd}},
                ],
                "status": {"code": 0 if call.success else 2},  # UNSET or ERROR
            })

        for child in node.children:
            _node_to_spans(child, span_id)

    _node_to_spans(report.root)
    for session_id, link in links:
        span_by_session[session_id].setdefault("links", []).append(link)
    return spans


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_a2a_tree(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    session_id = getattr(args, "session_id", None)
    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1

    full_id = store.find_session(session_id)
    if not full_id:
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    report = build_a2a_tree(store, full_id)

    fmt = getattr(args, "format", "text") or "text"
    if fmt == "json":
        import json as _json
        spans = a2a_calls_to_otlp_spans(report)
        sys.stdout.write(_json.dumps({"spans": spans}, indent=2) + "\n")
    else:
        format_a2a_tree(report)

    return 0
