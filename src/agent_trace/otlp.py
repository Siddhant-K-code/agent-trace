"""OTLP/HTTP JSON exporter.

Converts agent-trace sessions to OpenTelemetry spans and sends them
to any OTLP-compatible collector over HTTP/JSON. Zero dependencies.

Each agent-trace session becomes an OTel trace. Tool calls become spans.
User prompts and assistant responses become events on the root span.

Works with: Datadog, Honeycomb, New Relic, Splunk, Grafana Tempo, Jaeger.

Usage:
    agent-strace export <session-id> --format otlp --endpoint http://localhost:4318
    agent-strace export <session-id> --format otlp --endpoint https://api.honeycomb.io
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
import urllib.error
from typing import Any

from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# OTel semantic conventions for GenAI (opentelemetry-semantic-conventions 1.27+)
# https://opentelemetry.io/docs/specs/semconv/gen-ai/
# ---------------------------------------------------------------------------

_SEMCONV_SYSTEM = "gen_ai.system"
_SEMCONV_OP = "gen_ai.operation.name"
_SEMCONV_MODEL = "gen_ai.request.model"
_SEMCONV_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_SEMCONV_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_SEMCONV_TOOL_NAME = "gen_ai.tool.name"
_SEMCONV_TOOL_CALL_ID = "gen_ai.tool.call.id"
_SEMCONV_FINISH_REASON = "gen_ai.response.finish_reasons"


def _to_trace_id(session_id: str) -> str:
    """Convert session ID to a 32-hex-char trace ID."""
    h = hashlib.sha256(session_id.encode()).hexdigest()
    return h[:32]


def _to_span_id(event_id: str) -> str:
    """Convert event ID to a 16-hex-char span ID."""
    h = hashlib.sha256(event_id.encode()).hexdigest()
    return h[:16]


def _ts_to_nanos(ts: float) -> str:
    """Convert Unix timestamp to nanoseconds as string."""
    return str(int(ts * 1_000_000_000))


def _duration_to_nanos(ms: float | None) -> int:
    """Convert milliseconds to nanoseconds."""
    if ms is None or ms <= 0:
        return 1_000_000  # 1ms default
    return int(ms * 1_000_000)


def _make_attributes(data: dict) -> list[dict]:
    """Convert a flat dict to OTel attribute format."""
    attrs = []
    for key, value in data.items():
        if isinstance(value, bool):
            attrs.append({"key": key, "value": {"boolValue": value}})
        elif isinstance(value, int):
            attrs.append({"key": key, "value": {"intValue": str(value)}})
        elif isinstance(value, float):
            attrs.append({"key": key, "value": {"doubleValue": value}})
        elif isinstance(value, str):
            attrs.append({"key": key, "value": {"stringValue": value}})
        elif isinstance(value, dict):
            attrs.append({"key": key, "value": {"stringValue": json.dumps(value)}})
        elif isinstance(value, list):
            attrs.append({"key": key, "value": {"stringValue": json.dumps(value)}})
        else:
            attrs.append({"key": key, "value": {"stringValue": str(value)}})
    return attrs


def _make_event(name: str, timestamp: float, data: dict) -> dict:
    """Create an OTel span event."""
    return {
        "timeUnixNano": _ts_to_nanos(timestamp),
        "name": name,
        "attributes": _make_attributes(data),
    }


def session_to_otlp(
    meta: SessionMeta,
    events: list[TraceEvent],
    service_name: str = "agent-trace",
    parent_span_id: str = "",
    parent_trace_id: str = "",
) -> dict:
    """Convert an agent-trace session to OTLP JSON trace format.

    Returns the full ExportTraceServiceRequest body ready to POST.

    Args:
        parent_span_id: When set, the root span is linked as a child of this
            span (used for subagent hierarchy).
        parent_trace_id: When set, the root span shares this trace ID so all
            subagent spans appear in the same trace in the backend.
    """
    # Use parent trace ID for subagents so they appear in the same trace
    trace_id = parent_trace_id if parent_trace_id else _to_trace_id(meta.session_id)

    # Root span covers the entire session
    root_span_id = _to_span_id(f"root-{meta.session_id}")
    root_start = _ts_to_nanos(meta.started_at)
    root_end = _ts_to_nanos(meta.ended_at or (meta.started_at + (meta.total_duration_ms or 0) / 1000))

    # Detect provider from agent name for semantic conventions
    agent_lower = (meta.agent_name or "").lower()
    if "claude" in agent_lower or "anthropic" in agent_lower:
        gen_ai_system = "anthropic"
    elif "gpt" in agent_lower or "openai" in agent_lower:
        gen_ai_system = "openai"
    elif "gemini" in agent_lower or "google" in agent_lower:
        gen_ai_system = "google"
    else:
        gen_ai_system = "unknown"

    root_attrs = _make_attributes({
        # Standard OTel resource attributes
        "agent.name": meta.agent_name or "unknown",
        "agent.command": meta.command or "",
        "agent.session_id": meta.session_id,
        "agent.tool_calls": meta.tool_calls,
        "agent.llm_requests": meta.llm_requests,
        "agent.errors": meta.errors,
        "agent.depth": meta.depth,
        # GenAI semantic conventions
        _SEMCONV_SYSTEM: gen_ai_system,
        _SEMCONV_OP: "agent.session",
    })

    # Collect span events (user prompts, assistant responses, decisions)
    # and child spans (tool calls, errors)
    root_events = []
    child_spans = []

    # Track tool_call events so we can pair them with tool_result
    pending_calls: dict[str, TraceEvent] = {}

    for event in events:
        if event.event_type == EventType.USER_PROMPT:
            root_events.append(_make_event(
                "user_prompt",
                event.timestamp,
                {"prompt": event.data.get("prompt", "")},
            ))

        elif event.event_type == EventType.ASSISTANT_RESPONSE:
            text = event.data.get("text", "")
            if len(text) > 500:
                text = text[:500] + "..."
            root_events.append(_make_event(
                "assistant_response",
                event.timestamp,
                {"text": text},
            ))

        elif event.event_type == EventType.DECISION:
            root_events.append(_make_event(
                "decision",
                event.timestamp,
                event.data,
            ))

        elif event.event_type == EventType.TOOL_CALL:
            pending_calls[event.event_id] = event

        elif event.event_type == EventType.TOOL_RESULT:
            # Find the matching tool_call
            call_event = None
            if event.parent_id and event.parent_id in pending_calls:
                call_event = pending_calls.pop(event.parent_id)

            tool_name = event.data.get("tool_name", "") or (
                call_event.data.get("tool_name", "tool") if call_event else "tool"
            )
            span_start = call_event.timestamp if call_event else event.timestamp
            duration_ns = _duration_to_nanos(event.duration_ms)

            span_attrs = {
                _SEMCONV_TOOL_NAME: tool_name,
                _SEMCONV_OP: "tool.call",
                _SEMCONV_TOOL_CALL_ID: (call_event.event_id if call_event else event.event_id),
            }
            if call_event:
                args = call_event.data.get("arguments", {})
                if args:
                    for k, v in args.items():
                        span_attrs[f"tool.input.{k}"] = str(v)[:200]

            result = event.data.get("result", "")
            if result:
                span_attrs["tool.output"] = str(result)[:500]

            child_spans.append({
                "traceId": trace_id,
                "spanId": _to_span_id(call_event.event_id if call_event else event.event_id),
                "parentSpanId": root_span_id,
                "name": f"tool/{tool_name}",
                "kind": 3,  # SPAN_KIND_CLIENT
                "startTimeUnixNano": _ts_to_nanos(span_start),
                "endTimeUnixNano": _ts_to_nanos(span_start + duration_ns / 1_000_000_000),
                "attributes": _make_attributes(span_attrs),
                "status": {"code": 1},  # STATUS_CODE_OK
            })

        elif event.event_type == EventType.ERROR:
            # Find the matching tool_call
            call_event = None
            if event.parent_id and event.parent_id in pending_calls:
                call_event = pending_calls.pop(event.parent_id)

            tool_name = event.data.get("tool_name", "error")
            error_msg = event.data.get("error", "") or event.data.get("message", "")
            span_start = call_event.timestamp if call_event else event.timestamp
            duration_ns = _duration_to_nanos(event.duration_ms)

            span_attrs = {"tool.name": tool_name}
            if error_msg:
                span_attrs["error.message"] = str(error_msg)[:500]
            if call_event:
                args = call_event.data.get("arguments", {})
                if args:
                    for k, v in args.items():
                        span_attrs[f"tool.input.{k}"] = str(v)[:200]

            child_spans.append({
                "traceId": trace_id,
                "spanId": _to_span_id(call_event.event_id if call_event else event.event_id),
                "parentSpanId": root_span_id,
                "name": tool_name,
                "kind": 1,
                "startTimeUnixNano": _ts_to_nanos(span_start),
                "endTimeUnixNano": _ts_to_nanos(span_start + duration_ns / 1_000_000_000),
                "attributes": _make_attributes(span_attrs),
                "status": {"code": 2, "message": str(error_msg)[:200]},  # STATUS_CODE_ERROR
                "events": [_make_event("exception", event.timestamp, {
                    "exception.message": str(error_msg)[:500],
                })],
            })

        elif event.event_type == EventType.LLM_REQUEST:
            llm_attrs: dict = {_SEMCONV_OP: "chat"}
            model = event.data.get("model", "")
            if model:
                llm_attrs[_SEMCONV_MODEL] = model
            inp_tok = event.data.get("input_tokens", 0)
            if inp_tok:
                llm_attrs[_SEMCONV_INPUT_TOKENS] = inp_tok
            root_events.append(_make_event("gen_ai.system.message", event.timestamp, llm_attrs))

        elif event.event_type == EventType.LLM_RESPONSE:
            llm_attrs = {_SEMCONV_OP: "chat"}
            out_tok = event.data.get("output_tokens", 0)
            if out_tok:
                llm_attrs[_SEMCONV_OUTPUT_TOKENS] = out_tok
            finish = event.data.get("stop_reason", event.data.get("finish_reason", ""))
            if finish:
                llm_attrs[_SEMCONV_FINISH_REASON] = finish
            root_events.append(_make_event("gen_ai.assistant.message", event.timestamp, llm_attrs))

    # Emit any unmatched tool_calls as spans (no result received)
    for call_event in pending_calls.values():
        tool_name = call_event.data.get("tool_name", "tool")
        child_spans.append({
            "traceId": trace_id,
            "spanId": _to_span_id(call_event.event_id),
            "parentSpanId": root_span_id,
            "name": tool_name,
            "kind": 1,
            "startTimeUnixNano": _ts_to_nanos(call_event.timestamp),
            "endTimeUnixNano": _ts_to_nanos(call_event.timestamp + 0.001),
            "attributes": _make_attributes({
                "tool.name": tool_name,
                **{f"tool.input.{k}": str(v)[:200] for k, v in call_event.data.get("arguments", {}).items()},
            }),
            "status": {"code": 0},  # STATUS_CODE_UNSET
        })

    # Build root span
    root_span: dict = {
        "traceId": trace_id,
        "spanId": root_span_id,
        "name": f"agent-session ({meta.agent_name or 'agent'})",
        "kind": 1,
        "startTimeUnixNano": root_start,
        "endTimeUnixNano": root_end,
        "attributes": root_attrs,
        "events": root_events,
        "status": {"code": 2 if meta.errors > 0 else 1},
    }
    # Link to parent span for subagent hierarchy
    if parent_span_id:
        root_span["parentSpanId"] = parent_span_id

    all_spans = [root_span] + child_spans

    return {
        "resourceSpans": [{
            "resource": {
                "attributes": _make_attributes({
                    "service.name": service_name,
                    "service.version": "agent-trace",
                    "agent.session_id": meta.session_id,
                }),
            },
            "scopeSpans": [{
                "scope": {
                    "name": "agent-trace",
                },
                "spans": all_spans,
            }],
        }],
    }


def tree_to_otlp(
    store: TraceStore,
    root_session_id: str,
    service_name: str = "agent-trace",
) -> dict:
    """Export a full subagent tree as a single OTLP trace.

    All sessions in the tree share the same trace_id. Each subagent session's
    root span is parented to the tool_call span that spawned it.
    """
    from .subagent import build_tree, SessionNode

    tree = build_tree(store, root_session_id)
    root_trace_id = _to_trace_id(root_session_id)
    all_spans: list[dict] = []

    def _collect(node: SessionNode, parent_span_id: str = "") -> None:
        meta = node.meta
        events = node.events
        payload = session_to_otlp(
            meta, events, service_name,
            parent_span_id=parent_span_id,
            parent_trace_id=root_trace_id,
        )
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        all_spans.extend(spans)

        # The root span of this session is the first span
        session_root_span_id = _to_span_id(f"root-{meta.session_id}")

        for child in node.children:
            # Parent span for child = the tool_call span that spawned it
            spawning_span_id = (
                _to_span_id(child.meta.parent_event_id)
                if child.meta.parent_event_id
                else session_root_span_id
            )
            _collect(child, parent_span_id=spawning_span_id)

    _collect(tree)

    root_meta = store.load_meta(root_session_id)
    return {
        "resourceSpans": [{
            "resource": {
                "attributes": _make_attributes({
                    "service.name": service_name,
                    "service.version": "agent-trace",
                    "agent.session_id": root_session_id,
                }),
            },
            "scopeSpans": [{
                "scope": {"name": "agent-trace"},
                "spans": all_spans,
            }],
        }],
    }


def export_otlp(
    store: TraceStore,
    session_id: str,
    endpoint: str,
    headers: dict[str, str] | None = None,
    service_name: str = "agent-trace",
) -> bool:
    """Export a session to an OTLP/HTTP endpoint.

    Args:
        store: TraceStore to load session from
        session_id: Session to export
        endpoint: OTLP collector URL (e.g. http://localhost:4318)
        headers: Extra HTTP headers (for auth tokens, API keys)
        service_name: OTel service name

    Returns:
        True if export succeeded
    """
    meta = store.load_meta(session_id)
    if not meta:
        sys.stderr.write(f"Session {session_id} not found\n")
        return False

    events = store.load_events(session_id)
    if not events:
        sys.stderr.write(f"No events for session {session_id}\n")
        return False

    payload = session_to_otlp(meta, events, service_name)
    body = json.dumps(payload).encode("utf-8")

    # POST to /v1/traces
    url = endpoint.rstrip("/") + "/v1/traces"

    req_headers = {
        "Content-Type": "application/json",
    }
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            if status in (200, 202):
                sys.stderr.write(f"Exported {len(events)} events to {url} (HTTP {status})\n")
                return True
            else:
                sys.stderr.write(f"OTLP export returned HTTP {status}\n")
                return False
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"OTLP export failed: HTTP {e.code} {e.reason}\n")
        body = e.read().decode("utf-8", errors="replace")[:200]
        if body:
            sys.stderr.write(f"  {body}\n")
        return False
    except urllib.error.URLError as e:
        sys.stderr.write(f"OTLP export failed: {e.reason}\n")
        return False
