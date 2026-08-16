"""Temporal trace-context integration.

Temporal's OpenTelemetry interceptor propagates W3C trace context to
activities.  Agent processes can expose that context through
``AGENT_STRACE_TEMPORAL_TRACE_PARENT`` (or the shorter
``TEMPORAL_TRACE_PARENT`` compatibility name).  This module stores the
upstream trace and span IDs on the session and emits GenAI OTLP/HTTP JSON
spans beneath the Temporal activity span.

No Temporal or OpenTelemetry SDK is required.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from .models import SessionMeta, TraceEvent
from .otlp import export_otlp_payload, session_to_otlp_genai
from .propagation import extract_traceparent
from .store import TraceStore


TEMPORAL_TRACE_PARENT_ENV = "AGENT_STRACE_TEMPORAL_TRACE_PARENT"
TEMPORAL_TRACE_PARENT_FALLBACK_ENV = "TEMPORAL_TRACE_PARENT"

_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_FLAGS_RE = re.compile(r"^[0-9a-f]{2}$")


@dataclass(frozen=True)
class TemporalTraceContext:
    """The upstream Temporal activity's W3C trace context."""

    trace_id: str
    parent_span_id: str
    trace_flags: str
    source_env: str = TEMPORAL_TRACE_PARENT_ENV


def _validate_ids(trace_id: str, parent_span_id: str, trace_flags: str) -> None:
    """Validate the W3C fields needed by the OTLP exporter."""
    if not _TRACE_ID_RE.fullmatch(trace_id) or trace_id == "0" * 32:
        raise ValueError("traceparent contains an invalid trace ID")
    if not _SPAN_ID_RE.fullmatch(parent_span_id) or parent_span_id == "0" * 16:
        raise ValueError("traceparent contains an invalid parent span ID")
    if not _FLAGS_RE.fullmatch(trace_flags):
        raise ValueError("traceparent contains invalid trace flags")


def parse_temporal_traceparent(
    value: str,
    source_env: str = TEMPORAL_TRACE_PARENT_ENV,
) -> TemporalTraceContext:
    """Parse a Temporal W3C ``traceparent`` environment value.

    ``ValueError`` is raised for malformed or unusable trace context so a
    configured activity never silently exports into a separate trace.
    """
    parsed = extract_traceparent({"traceparent": value})
    if parsed is None or parsed["version"] == "ff":
        raise ValueError(f"{source_env} is not a valid W3C traceparent")

    trace_id = parsed["trace_id"]
    parent_span_id = parsed["parent_id"]
    trace_flags = parsed["flags"]
    _validate_ids(trace_id, parent_span_id, trace_flags)
    return TemporalTraceContext(
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        trace_flags=trace_flags,
        source_env=source_env,
    )


def get_temporal_trace_context(
    environ: Mapping[str, str] | None = None,
) -> TemporalTraceContext | None:
    """Read Temporal trace context from the supported environment names.

    The agent-strace-specific name takes precedence when both are set.
    """
    env = os.environ if environ is None else environ
    for name in (TEMPORAL_TRACE_PARENT_ENV, TEMPORAL_TRACE_PARENT_FALLBACK_ENV):
        value = env.get(name, "").strip()
        if value:
            return parse_temporal_traceparent(value, source_env=name)
    return None


def attach_temporal_trace_context(
    store: TraceStore,
    session_id: str,
    environ: Mapping[str, str] | None = None,
) -> TemporalTraceContext | None:
    """Persist the current Temporal activity context on a stored session."""
    context = get_temporal_trace_context(environ)
    if context is None:
        return None

    meta = store.load_meta(session_id)
    meta.trace_id = context.trace_id
    meta.parent_span_id = context.parent_span_id
    meta.trace_flags = context.trace_flags
    store.update_meta(meta)
    return context


def session_to_temporal_otlp(
    meta: SessionMeta,
    events: list[TraceEvent],
    service_name: str = "agent-trace",
) -> dict:
    """Convert a session into OTLP JSON parented to a Temporal activity."""
    trace_id = meta.trace_id.lower()
    parent_span_id = meta.parent_span_id.lower()
    trace_flags = (meta.trace_flags or "01").lower()
    if not trace_id or not parent_span_id:
        raise ValueError(
            "session has no Temporal trace context; run watch with "
            f"{TEMPORAL_TRACE_PARENT_ENV} set"
        )
    _validate_ids(trace_id, parent_span_id, trace_flags)

    payload = session_to_otlp_genai(
        meta,
        events,
        service_name=service_name,
        parent_span_id=parent_span_id,
        parent_trace_id=trace_id,
    )

    # OTLP Span.flags carries the W3C sampled/debug bits.  Apply it to the
    # complete local subtree so collectors preserve the upstream decision.
    flags = int(trace_flags, 16)
    for resource_spans in payload.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                span["flags"] = flags
    return payload


def export_temporal_otlp(
    store: TraceStore,
    session_id: str,
    endpoint: str,
    headers: dict[str, str] | None = None,
    service_name: str = "agent-trace",
) -> bool:
    """Export one Temporal-parented session via OTLP/HTTP JSON."""
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    if not events:
        sys.stderr.write(f"No events for session {session_id}\n")
        return False
    try:
        payload = session_to_temporal_otlp(meta, events, service_name=service_name)
    except ValueError as exc:
        sys.stderr.write(f"Temporal export failed: {exc}\n")
        return False
    return export_otlp_payload(
        payload,
        endpoint,
        headers=headers,
        event_count=len(events),
    )
