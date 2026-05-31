"""W3C Trace Context propagation (traceparent / tracestate).

Implements the W3C Trace Context Level 1 spec (https://www.w3.org/TR/trace-context/)
using stdlib only.

Format:
    traceparent: 00-<trace-id>-<parent-span-id>-<flags>
    tracestate:  agent-trace=<session-id>

Usage — injecting into outbound HTTP headers:
    headers = inject_traceparent(headers, session_id, event_id)

Usage — extracting from inbound HTTP headers:
    ctx = extract_traceparent(headers)
    if ctx:
        session_id = ctx.get("at_session_id")   # from tracestate
        trace_id   = ctx["trace_id"]
        parent_id  = ctx["parent_id"]
"""

from __future__ import annotations

import re
import uuid

# W3C traceparent regex: version-traceid-parentid-flags
_TRACEPARENT_RE = re.compile(
    r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)

# tracestate key used to carry the agent-trace session ID
_TRACESTATE_KEY = "agent-trace"


def _new_trace_id() -> str:
    """Generate a 32-hex-char W3C trace ID."""
    return uuid.uuid4().hex + uuid.uuid4().hex[:0]  # 32 chars from one UUID4


def _session_to_span_id(session_id: str) -> str:
    """Derive a stable 16-hex-char span ID from a session/event ID."""
    # Take the first 16 hex chars; pad or truncate as needed
    clean = re.sub(r"[^0-9a-f]", "", session_id.lower())
    return (clean + "0" * 16)[:16]


def inject_traceparent(
    headers: dict[str, str],
    session_id: str,
    event_id: str = "",
    trace_id: str = "",
) -> dict[str, str]:
    """Return a copy of *headers* with W3C traceparent/tracestate injected.

    If *trace_id* is provided (extracted from an upstream traceparent), it is
    reused so the full distributed trace stays on one trace ID.  Otherwise a
    new trace ID is generated from the session ID.
    """
    headers = dict(headers)

    if not trace_id:
        # Derive a deterministic trace ID from the session so replays are stable
        clean = re.sub(r"[^0-9a-f]", "", session_id.lower())
        trace_id = (clean + "0" * 32)[:32]

    span_id = _session_to_span_id(event_id or session_id)
    headers["traceparent"] = f"00-{trace_id}-{span_id}-01"

    # Carry the agent-trace session ID in tracestate so downstream agents
    # can set parent_session_id without any out-of-band coordination.
    existing = headers.get("tracestate", "")
    new_entry = f"{_TRACESTATE_KEY}={session_id}"
    if existing:
        # Prepend (highest priority vendor first per spec)
        headers["tracestate"] = f"{new_entry},{existing}"
    else:
        headers["tracestate"] = new_entry

    return headers


def extract_traceparent(headers: dict[str, str]) -> dict[str, str] | None:
    """Parse W3C traceparent/tracestate from *headers*.

    Returns a dict with keys:
        trace_id        — 32-hex W3C trace ID
        parent_id       — 16-hex parent span ID
        flags           — 2-hex trace flags
        at_session_id   — agent-trace session ID from tracestate (may be "")

    Returns None if no valid traceparent header is present.
    """
    raw = headers.get("traceparent") or headers.get("Traceparent") or ""
    if not raw:
        return None

    m = _TRACEPARENT_RE.match(raw.strip().lower())
    if not m:
        return None

    version, trace_id, parent_id, flags = m.groups()

    # Extract agent-trace session ID from tracestate
    at_session_id = ""
    tracestate = headers.get("tracestate") or headers.get("Tracestate") or ""
    for entry in tracestate.split(","):
        entry = entry.strip()
        if entry.startswith(f"{_TRACESTATE_KEY}="):
            at_session_id = entry[len(_TRACESTATE_KEY) + 1:]
            break

    return {
        "version": version,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "flags": flags,
        "at_session_id": at_session_id,
    }
