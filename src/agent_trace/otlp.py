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

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from .event_data import object_field, str_field
from .models import EventType, SessionMeta, TraceEvent
from .pairing import (
    LLM_REQUEST_TYPES,
    LLM_RESPONSE_TYPES,
    TOOL_REQUEST_TYPES,
    TOOL_RESPONSE_TYPES,
    Pairer,
    event_model,
    llm_pairer,
    tool_pairer,
)
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
_SEMCONV_CONVERSATION_ID = "gen_ai.conversation.id"
_SEMCONV_ERROR_TYPE = "error.type"

# OpenInference compatibility attributes. These are emitted additively by the
# GenAI exporter so the same OTLP/HTTP JSON payload remains useful in Phoenix.
_OI_SPAN_KIND = "openinference.span.kind"
_OI_SESSION_ID = "session.id"
_OI_LLM_PROVIDER = "llm.provider"
_OI_LLM_MODEL = "llm.model_name"
_OI_INPUT_TOKENS = "llm.token_count.prompt"
_OI_OUTPUT_TOKENS = "llm.token_count.completion"
_OI_TOTAL_TOKENS = "llm.token_count.total"
_OI_TOOL_NAME = "tool.name"
_OI_INPUT_VALUE = "input.value"
_OI_INPUT_MIME_TYPE = "input.mime_type"
_OI_OUTPUT_VALUE = "output.value"
_OI_OUTPUT_MIME_TYPE = "output.mime_type"

_INPUT_TOKEN_KEYS = (
    "input_tokens",
    "prompt_tokens",
    "inputTokens",
    "promptTokenCount",
    "prompt_token_count",
)
_OUTPUT_TOKEN_KEYS = (
    "output_tokens",
    "completion_tokens",
    "outputTokens",
    "candidatesTokenCount",
    "completion_token_count",
)
_USAGE_KEYS = ("usage", "usage_metadata", "usageMetadata", "token_usage")
_EVENT_PROVENANCE_KEYS = (
    "provider",
    "capture_method",
    "context_stream",
    "is_sidechain",
    "source_uuid",
    "source_parent_uuid",
    "source_agent_id",
    "source",
    "hook_event",
    "agent_id",
    "agent_type",
)


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


# ---------------------------------------------------------------------------
# Failed and cancelled model requests
# ---------------------------------------------------------------------------

# error.type when nothing more specific was recorded: the GenAI semconv
# fallback value.
ERROR_TYPE_OTHER = "_OTHER"
# error.type for a cancelled request that recorded no error type. The GenAI
# semconv names no span status for a cancelled request, but ends its span and
# reports the finish reason "error" for it, so it exports as an error.
ERROR_TYPE_CANCELLED = "cancelled"
_FINISH_REASON_ERROR = "error"
_ERROR_MESSAGE_LIMIT = 200
_FAILED_STATUSES = frozenset({"failed"})
_CANCELLED_STATUSES = frozenset({"cancelled", "canceled"})
_ERROR_TYPE_KEYS = ("error_type", "exception_type")
_ERROR_CODE_KEYS = ("type", "code")


@dataclass(frozen=True)
class LLMFailure:
    """How a model request ended when it did not complete."""

    error_type: str
    """Low-cardinality value for the ``error.type`` attribute."""
    message: str = ""
    """Recorded error text, at most 200 characters. Empty when there is none."""
    cancelled: bool = False


def _error_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        message = value.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str) if value else ""
    return str(value)


def _recorded_error_type(data: Mapping[str, Any], error: Any) -> str:
    for key in _ERROR_TYPE_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:_ERROR_MESSAGE_LIMIT]
    if isinstance(error, Mapping):
        for key in _ERROR_CODE_KEYS:
            value = error.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                return str(value).strip()[:_ERROR_MESSAGE_LIMIT]
    return ""


def llm_failure(event: TraceEvent) -> LLMFailure | None:
    """Return how the model request that ``event`` ends failed, or ``None``.

    ``event`` is an ``llm_response``, or an ``error`` linked to the
    ``llm_request`` it ends. An error always fails. A response is cancelled
    when its ``status`` is ``cancelled`` and failed when its ``status`` is
    ``failed`` or it carries a non-empty ``error``. Any other response,
    including one with no ``status``, completed and returns ``None``.

    ``error_type`` is the recorded ``error_type`` or ``exception_type``, else
    the ``type`` or ``code`` of an ``error`` object, else ``cancelled`` for a
    cancelled request and ``_OTHER`` for a failed one.
    """
    data = event.data if isinstance(event.data, Mapping) else {}
    error = data.get("error")
    text = _error_text(error)
    cancelled = False
    if event.event_type == EventType.ERROR:
        text = text or _error_text(data.get("message"))
    else:
        status = data.get("status")
        status = status.strip().lower() if isinstance(status, str) else ""
        cancelled = status in _CANCELLED_STATUSES
        if not cancelled and status not in _FAILED_STATUSES and not text:
            return None
    error_type = _recorded_error_type(data, error) or (
        ERROR_TYPE_CANCELLED if cancelled else ERROR_TYPE_OTHER
    )
    return LLMFailure(error_type, text[:_ERROR_MESSAGE_LIMIT], cancelled)


def _failure_status(failure: LLMFailure) -> dict:
    """Return the OTLP ERROR status for a failed or cancelled model request."""
    if failure.message:
        return {"code": 2, "message": failure.message}  # STATUS_CODE_ERROR
    return {"code": 2}


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


def _canonical_provider(value: object) -> str:
    """Return the provider name used by the project's GenAI convention."""
    raw = str(value or "").strip().lower()
    if "bedrock" in raw or raw.startswith("aws"):
        return "aws.bedrock"
    if "claude" in raw or "anthropic" in raw:
        return "anthropic"
    if "gpt" in raw or "openai" in raw or raw.startswith(("o1", "o3", "o4")):
        return "openai"
    if "gemini" in raw or "google" in raw:
        return "google"
    return "unknown"


def _provider_value(
    meta: SessionMeta,
    model: str = "",
    *event_data: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the raw provider evidence and its canonical GenAI value."""
    for data in event_data:
        for key in ("provider", "model_provider", "system"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), _canonical_provider(value)
    if model:
        canonical = _canonical_provider(model)
        if canonical != "unknown":
            return canonical, canonical
    attribution = meta.attribution if isinstance(meta.attribution, dict) else {}
    raw = attribution.get("agent_provider") or meta.agent_name
    return str(raw or "unknown"), _canonical_provider(raw)


def _number(data: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key not in data or isinstance(data[key], bool):
            continue
        try:
            value = int(data[key])
        except (TypeError, ValueError, OverflowError):
            continue
        if value >= 0:
            return value
    return None


def _token_count(data: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    direct = _number(data, keys)
    if direct is not None:
        return direct
    candidates = [data.get(key) for key in _USAGE_KEYS]
    response_metadata = data.get("response_metadata")
    if isinstance(response_metadata, Mapping):
        candidates.append(response_metadata.get("token_usage"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            nested = _number(candidate, keys)
            if nested is not None:
                return nested
    return None


def _float_value(data: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in data or isinstance(data[key], bool):
            continue
        try:
            value = float(data[key])
        except (TypeError, ValueError, OverflowError):
            continue
        if value >= 0 and math.isfinite(value):
            return value
    return None


def _cost_attributes(
    request_data: Mapping[str, Any],
    response_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve recorded USD cost without estimating missing values."""
    input_cost = _float_value(
        request_data,
        ("input_cost_usd", "prompt_cost_usd"),
    )
    if input_cost is None:
        input_cost = _float_value(
            response_data,
            ("input_cost_usd", "prompt_cost_usd"),
        )
    output_cost = _float_value(
        response_data,
        ("output_cost_usd", "completion_cost_usd"),
    )
    if output_cost is None:
        output_cost = _float_value(
            request_data,
            ("output_cost_usd", "completion_cost_usd"),
        )
    total_cost = _float_value(
        response_data,
        ("cost_usd", "total_cost_usd", "estimated_cost_usd"),
    )
    if total_cost is None:
        total_cost = _float_value(
            request_data,
            ("cost_usd", "total_cost_usd", "estimated_cost_usd"),
        )
    if total_cost is None and (input_cost is not None or output_cost is not None):
        total_cost = (input_cost or 0.0) + (output_cost or 0.0)

    attrs: dict[str, Any] = {}
    if input_cost is not None:
        attrs["agent.strace.cost.input_usd"] = input_cost
        attrs["llm.cost.prompt"] = input_cost
    if output_cost is not None:
        attrs["agent.strace.cost.output_usd"] = output_cost
        attrs["llm.cost.completion"] = output_cost
    if total_cost is not None:
        attrs["agent.strace.cost.total_usd"] = total_cost
        attrs["llm.cost.total"] = total_cost
    return attrs


def _event_trace_attributes(event: TraceEvent, role: str) -> dict[str, Any]:
    """Return additive redaction, provenance, and hash-chain attributes."""
    prefix = f"agent.strace.{role}"
    data = event.data if isinstance(event.data, Mapping) else {}
    attrs: dict[str, Any] = {
        f"{prefix}.event_id": event.event_id,
        f"{prefix}.redacted": event.redacted,
    }
    if event.parent_id:
        attrs[f"{prefix}.parent_event_id"] = event.parent_id
    if event.prev_hash:
        attrs[f"{prefix}.previous_hash"] = event.prev_hash
    provenance = {
        key: data[key]
        for key in _EVENT_PROVENANCE_KEYS
        if key in data
    }
    if provenance:
        attrs[f"{prefix}.provenance"] = provenance
    return attrs


def _conversation_attributes(meta: SessionMeta) -> dict[str, str]:
    """Return gen_ai.conversation.id when the agent supplied its own session ID.

    The GenAI conventions make the attribute conditionally required only when
    an ID is readily available and say instrumentations SHOULD NOT invent one,
    so sessions without a native ID omit it instead of reusing session_id.
    """
    conversation_id = getattr(meta, "conversation_id", "")
    if isinstance(conversation_id, str) and conversation_id:
        return {_SEMCONV_CONVERSATION_ID: conversation_id}
    return {}


def _openinference_metadata(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Pack agent-strace attributes into OpenInference metadata."""
    return {
        "agent_strace": {
            key.removeprefix("agent.strace."): value
            for key, value in attrs.items()
            if key.startswith("agent.strace.")
        }
    }


def _evidence_attributes(
    meta: SessionMeta,
    events: list[TraceEvent],
) -> dict[str, Any]:
    """Summarize observed structural evidence gaps.

    Gaps are assessed without event data. The event types the hook provider
    sends before its session start, and the result types it records without
    requests, come from its capture matrix entry, as in ``health`` and
    ``validate``; identifying the provider reads only the hook-written
    session_start and stop events, and ignores data that is not an object.
    """
    from .evidence_health import assess_evidence_health
    from .health import declared_capture

    declared = declared_capture(meta, events)
    structural_events = [replace(event, data={}) for event in events]
    result = assess_evidence_health(
        structural_events,
        session_finalized=True if meta.ended_at is not None else None,
        allowed_pre_start_event_types=declared.events_before_start,
        allowed_unrequested_result_types=declared.results_without_requests,
    )
    reasons = [
        {
            "code": reason.code,
            "kind": reason.kind.value,
            "event_id": reason.event_id,
            "message": reason.message,
        }
        for reason in result.reasons
    ]
    return {
        "agent.strace.evidence.schema_version": result.schema_version,
        "agent.strace.evidence.status": result.status.value,
        "agent.strace.evidence.gap_count": len(reasons),
        "agent.strace.evidence.gap_codes": [reason["code"] for reason in reasons],
        "agent.strace.evidence.gaps": reasons,
    }


def _io_value(value: object, limit: int) -> tuple[str, str]:
    """Serialize an OpenInference input or output value with a MIME type."""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if len(text) <= limit:
            return text, "application/json"
        return text[:limit], "text/plain"
    return str(value)[:limit], "text/plain"


# session_to_otlp reads the data of every event of these types. It reads a
# tool_call only when otlp_data_indexes says so.
OTLP_DATA_EVENT_TYPES = frozenset({
    EventType.USER_PROMPT,
    EventType.ASSISTANT_RESPONSE,
    EventType.DECISION,
    EventType.TOOL_RESULT,
    EventType.ERROR,
    EventType.LLM_REQUEST,
    EventType.LLM_RESPONSE,
})


def otlp_data_indexes(events: list[TraceEvent]) -> list[int]:
    """Return the 0-based indexes of the events whose data session_to_otlp reads.

    It pairs events with the same pairing.py rules as the converter. A
    tool_call is read when a tool_result or an error pairs with it or when it
    is still unpaired at the end, but not when a later tool_call with the
    same ID replaces it before that.
    """
    read: set[int] = set()
    calls: Pairer[int] = tool_pairer()
    for index, event in enumerate(events):
        if event.event_type in TOOL_REQUEST_TYPES:
            calls.add(event, index)
        elif event.event_type in OTLP_DATA_EVENT_TYPES:
            read.add(index)
            if event.event_type in TOOL_RESPONSE_TYPES:
                call_index = calls.match(event)
                if call_index is not None:
                    read.add(call_index)
    read.update(calls.unmatched())
    return sorted(read)


def otlp_genai_data_indexes(events: list[TraceEvent]) -> list[int]:
    """Return the 0-based indexes of the events whose data session_to_otlp_genai reads.

    It pairs events with the same pairing.py rules as the converter. A
    tool_call is read when a tool_result pairs with it or when it is still
    unpaired at the end, but not when an error pairs with it. An llm_request
    is read when an llm_response or a linked error ends it (see
    Pairer.match_parent) or when it is still unpaired at the end. Pairing an
    llm_response without a matching parent_id also reads the model of each
    pending llm_request; a request whose data is not an object counts as
    naming no model.
    """
    read: set[int] = set()
    requests: Pairer[int] = llm_pairer()
    calls: Pairer[int] = tool_pairer()
    for index, event in enumerate(events):
        if event.event_type in (EventType.USER_PROMPT, EventType.ASSISTANT_RESPONSE):
            read.add(index)
        elif event.event_type in LLM_REQUEST_TYPES:
            requests.add(event, index)
        elif event.event_type in LLM_RESPONSE_TYPES:
            read.add(index)
            request_index = requests.match(event)
            if request_index is not None:
                read.add(request_index)
        elif event.event_type in TOOL_REQUEST_TYPES:
            calls.add(event, index)
        elif event.event_type == EventType.TOOL_RESULT:
            read.add(index)
            call_index = calls.match(event)
            if call_index is not None:
                read.add(call_index)
        elif event.event_type == EventType.ERROR:
            read.add(index)
            request_index = requests.match_parent(event)
            if request_index is not None:
                read.add(request_index)
            else:
                calls.match(event)
    read.update(requests.unmatched())
    read.update(calls.unmatched())
    return sorted(read)


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
        **_conversation_attributes(meta),
    })

    # Collect span events (user prompts, assistant responses, decisions)
    # and child spans (tool calls, errors)
    root_events = []
    child_spans = []

    # Pair tool_call events with their tool_result or error, and LLM
    # requests with their responses (see pairing.py)
    calls: Pairer[TraceEvent] = tool_pairer()
    requests: Pairer[TraceEvent] = llm_pairer()

    for event in events:
        if event.event_type == EventType.USER_PROMPT:
            root_events.append(_make_event(
                "user_prompt",
                event.timestamp,
                {"prompt": event.data.get("prompt", "")},
            ))

        elif event.event_type == EventType.ASSISTANT_RESPONSE:
            text = str_field(event.data, "text", "")
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
            calls.add(event)

        elif event.event_type == EventType.TOOL_RESULT:
            call_event = calls.match(event)

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
                args = object_field(call_event.data, "arguments", {})
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
            # An error linked to an LLM request ends that request; it still
            # exports as an error span below.
            requests.match_parent(event)
            call_event = calls.match(event)

            tool_name = event.data.get("tool_name", "error")
            error_msg = event.data.get("error", "") or event.data.get("message", "")
            span_start = call_event.timestamp if call_event else event.timestamp
            duration_ns = _duration_to_nanos(event.duration_ms)

            span_attrs = {"tool.name": tool_name}
            if error_msg:
                span_attrs["error.message"] = str(error_msg)[:500]
            if call_event:
                args = object_field(call_event.data, "arguments", {})
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
            requests.add(event)
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
            if requests.match(event) is None:
                # A response with no request (Codex records each model
                # request this way) carries its own model and input count.
                model = event_model(event.data)
                if model:
                    llm_attrs[_SEMCONV_MODEL] = model
                inp_tok = event.data.get("input_tokens", 0)
                if inp_tok:
                    llm_attrs[_SEMCONV_INPUT_TOKENS] = inp_tok
            out_tok = event.data.get("output_tokens", 0)
            if out_tok:
                llm_attrs[_SEMCONV_OUTPUT_TOKENS] = out_tok
            finish = event.data.get("stop_reason", event.data.get("finish_reason", ""))
            if finish:
                llm_attrs[_SEMCONV_FINISH_REASON] = finish
            failure = llm_failure(event)
            if failure is not None:
                # The classic export has no span per model request, so a
                # failed or cancelled one is marked on its event instead.
                if not finish:
                    llm_attrs[_SEMCONV_FINISH_REASON] = _FINISH_REASON_ERROR
                llm_attrs[_SEMCONV_ERROR_TYPE] = failure.error_type
                if failure.message:
                    llm_attrs["error.message"] = failure.message
            root_events.append(_make_event("gen_ai.assistant.message", event.timestamp, llm_attrs))

    # Emit any unmatched tool_calls as spans (no result received)
    for call_event in calls.unmatched():
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
                **{f"tool.input.{k}": str(v)[:200] for k, v in object_field(call_event.data, "arguments", {}).items()},
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

    if meta.tenant_id:
        tenant_attributes = _make_attributes({"tenant_id": meta.tenant_id})
        for span in [root_span] + child_spans:
            span.setdefault("attributes", []).extend(tenant_attributes)

    all_spans = [root_span] + child_spans

    return {
        "resourceSpans": [{
            "resource": {
                "attributes": _make_attributes({
                    "service.name": service_name,
                    "service.version": "agent-trace",
                    "agent.session_id": meta.session_id,
                    **({"tenant_id": meta.tenant_id} if meta.tenant_id else {}),
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
                    **({"tenant_id": root_meta.tenant_id} if root_meta.tenant_id else {}),
                }),
            },
            "scopeSpans": [{
                "scope": {"name": "agent-trace"},
                "spans": all_spans,
            }],
        }],
    }


def _genai_llm_span(
    meta: SessionMeta,
    trace_id: str,
    root_span_id: str,
    req_event: TraceEvent | None,
    event: TraceEvent,
) -> dict:
    """Build the gen_ai.client.operation span for one model request.

    ``event`` ends the request: an ``llm_response``, or an ``error`` linked
    to ``req_event``. A failed or cancelled request (see :func:`llm_failure`)
    gets ERROR status and ``error.type``; a completed one keeps OK status.
    """
    span_start = req_event.timestamp if req_event else event.timestamp
    span_end = event.timestamp

    model = event.data.get("model", "") or (
        req_event.data.get("model", "") if req_event else ""
    )
    req_data = req_event.data if req_event else {}
    raw_llm_provider, system = _provider_value(
        meta,
        str(model),
        event.data,
        req_data,
    )

    llm_attrs: dict = {
        _SEMCONV_SYSTEM: system,
        _SEMCONV_OP: "chat",
        _OI_SPAN_KIND: "LLM",
        _OI_SESSION_ID: meta.session_id,
        **_conversation_attributes(meta),
        _OI_LLM_PROVIDER: system if system != "unknown" else raw_llm_provider,
        "agent.strace.provider": raw_llm_provider,
    }
    if model:
        llm_attrs[_SEMCONV_MODEL] = model
        llm_attrs[_OI_LLM_MODEL] = model
    if req_event:
        max_tok = req_event.data.get("max_tokens")
        if max_tok:
            llm_attrs["gen_ai.request.max_tokens"] = max_tok
            llm_attrs["llm.invocation_parameters"] = json.dumps(
                {"max_tokens": max_tok},
                sort_keys=True,
                separators=(",", ":"),
            )

    inp_tok = _token_count(req_data, _INPUT_TOKEN_KEYS)
    if inp_tok is None:
        inp_tok = _token_count(event.data, _INPUT_TOKEN_KEYS)
    out_tok = _token_count(event.data, _OUTPUT_TOKEN_KEYS)
    if out_tok is None:
        out_tok = _token_count(req_data, _OUTPUT_TOKEN_KEYS)
    if inp_tok is not None:
        llm_attrs[_SEMCONV_INPUT_TOKENS] = inp_tok
        llm_attrs[_OI_INPUT_TOKENS] = inp_tok
    if out_tok is not None:
        llm_attrs[_SEMCONV_OUTPUT_TOKENS] = out_tok
        llm_attrs[_OI_OUTPUT_TOKENS] = out_tok
    if inp_tok is not None or out_tok is not None:
        llm_attrs[_OI_TOTAL_TOKENS] = (inp_tok or 0) + (out_tok or 0)
    finish = event.data.get("stop_reason") or event.data.get("finish_reason", "")
    if finish:
        llm_attrs[_SEMCONV_FINISH_REASON] = finish
        llm_attrs["agent.strace.finish_reason"] = finish
    failure = llm_failure(event)
    if failure is not None:
        if not finish:
            # The GenAI conventions report "error" when a failed or
            # cancelled generation returned no finish reason.
            llm_attrs[_SEMCONV_FINISH_REASON] = _FINISH_REASON_ERROR
        llm_attrs[_SEMCONV_ERROR_TYPE] = failure.error_type
    llm_attrs.update(_cost_attributes(req_data, event.data))
    if req_event:
        llm_attrs.update(_event_trace_attributes(req_event, "request"))
    llm_attrs.update(_event_trace_attributes(event, "response"))
    llm_attrs["metadata"] = _openinference_metadata(llm_attrs)

    span: dict = {
        "traceId": trace_id,
        "spanId": _to_span_id(req_event.event_id if req_event else event.event_id),
        "parentSpanId": root_span_id,
        "name": "gen_ai.client.operation",
        "kind": 3,  # SPAN_KIND_CLIENT
        "startTimeUnixNano": _ts_to_nanos(span_start),
        "endTimeUnixNano": _ts_to_nanos(max(span_end, span_start + 0.001)),
        "attributes": _make_attributes(llm_attrs),
    }
    if event.event_type == EventType.ERROR:
        span["events"] = [_make_event("exception", event.timestamp, {
            "exception.type": (
                str_field(event.data, "error_type", "")
                or str_field(event.data, "exception_type", "")
                or "Error"
            ),
            "exception.message": (
                _error_text(event.data.get("error"))
                or _error_text(event.data.get("message"))
            )[:500],
            "exception.escaped": False,
            **_event_trace_attributes(event, "response"),
        })]
    if failure is None:
        span["status"] = {"code": 1}  # STATUS_CODE_OK
    else:
        span["status"] = _failure_status(failure)
    return span


def session_to_otlp_genai(
    meta: SessionMeta,
    events: list[TraceEvent],
    service_name: str = "agent-trace",
    parent_span_id: str = "",
    parent_trace_id: str = "",
) -> dict:
    """Convert a session to OTLP JSON using strict OTel GenAI semantic conventions.

    Differences from session_to_otlp():
    - LLM request/response pairs become gen_ai.client.operation child spans
      (not just events on the root span). A failed or cancelled model
      request, or an error linked to an llm_request, gets ERROR status and
      error.type
    - Root span carries gen_ai.agent.id and gen_ai.agent.name
    - Root, LLM and tool spans carry gen_ai.conversation.id when the agent
      supplied its own session ID
    - Error events use the OTel exception event format
    - Tool calls carry gen_ai.tool.name and gen_ai.tool.call.id
    - gen_ai.system is derived from the model name when possible

    See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """
    trace_id = parent_trace_id if parent_trace_id else _to_trace_id(meta.session_id)
    root_span_id = _to_span_id(f"root-{meta.session_id}")
    root_start = _ts_to_nanos(meta.started_at)
    root_end = _ts_to_nanos(
        meta.ended_at or (meta.started_at + (meta.total_duration_ms or 0) / 1000)
    )

    raw_provider, gen_ai_system = _provider_value(meta)
    root_attr_values: dict[str, Any] = {
        # GenAI agent attributes
        "gen_ai.agent.id": meta.session_id,
        "gen_ai.agent.name": meta.agent_name or "agent",
        _SEMCONV_SYSTEM: gen_ai_system,
        _SEMCONV_OP: "agent.session",
        # OpenInference compatibility attributes
        _OI_SPAN_KIND: "AGENT",
        _OI_SESSION_ID: meta.session_id,
        **_conversation_attributes(meta),
        # Standard resource attributes
        "agent.session_id": meta.session_id,
        "agent.tool_calls": meta.tool_calls,
        "agent.llm_requests": meta.llm_requests,
        "agent.errors": meta.errors,
        # agent-strace preservation attributes
        "agent.strace.provider": raw_provider,
        "agent.strace.redacted": meta.redacted or any(event.redacted for event in events),
        "agent.strace.redacted_event_count": sum(event.redacted for event in events),
        **_evidence_attributes(meta, events),
    }
    if meta.attribution:
        root_attr_values["agent.strace.provenance"] = meta.attribution
    if meta.parent_session_id:
        root_attr_values["agent.strace.parent_session_id"] = meta.parent_session_id
    if meta.parent_event_id:
        root_attr_values["agent.strace.parent_event_id"] = meta.parent_event_id
    if meta.depth:
        root_attr_values["agent.strace.depth"] = meta.depth
    root_attr_values["metadata"] = _openinference_metadata(root_attr_values)
    root_attrs = _make_attributes(root_attr_values)

    root_events: list[dict] = []
    child_spans: list[dict] = []

    # Pair LLM requests and tool calls with their responses (see pairing.py)
    requests: Pairer[TraceEvent] = llm_pairer()
    calls: Pairer[TraceEvent] = tool_pairer()

    for event in events:
        if event.event_type == EventType.USER_PROMPT:
            event_attrs = {
                "gen_ai.prompt": str_field(event.data, "prompt", "")[:1000],
                **_event_trace_attributes(event, "event"),
            }
            event_attrs["metadata"] = _openinference_metadata(event_attrs)
            root_events.append(_make_event(
                "gen_ai.user.message",
                event.timestamp,
                event_attrs,
            ))

        elif event.event_type == EventType.ASSISTANT_RESPONSE:
            text = str_field(event.data, "text", "")
            event_attrs = {
                "gen_ai.completion": text[:1000],
                **_event_trace_attributes(event, "event"),
            }
            event_attrs["metadata"] = _openinference_metadata(event_attrs)
            root_events.append(_make_event(
                "gen_ai.assistant.message",
                event.timestamp,
                event_attrs,
            ))

        elif event.event_type == EventType.LLM_REQUEST:
            requests.add(event)

        elif event.event_type == EventType.LLM_RESPONSE:
            req_event = requests.match(event)
            child_spans.append(_genai_llm_span(
                meta, trace_id, root_span_id, req_event, event,
            ))

        elif event.event_type == EventType.TOOL_CALL:
            calls.add(event)

        elif event.event_type == EventType.TOOL_RESULT:
            call_event = calls.match(event)

            tool_name = event.data.get("tool_name", "") or (
                call_event.data.get("tool_name", "tool") if call_event else "tool"
            )
            span_start = call_event.timestamp if call_event else event.timestamp
            duration_ns = _duration_to_nanos(event.duration_ms)

            tool_attrs: dict = {
                _SEMCONV_TOOL_NAME: tool_name,
                _SEMCONV_OP: "execute",
                _SEMCONV_TOOL_CALL_ID: (
                    call_event.event_id if call_event else event.event_id
                ),
                _OI_SPAN_KIND: "TOOL",
                _OI_SESSION_ID: meta.session_id,
                **_conversation_attributes(meta),
                _OI_TOOL_NAME: tool_name,
            }
            if call_event:
                arguments = call_event.data.get("arguments", {})
                if isinstance(arguments, Mapping):
                    for k, v in arguments.items():
                        tool_attrs[f"gen_ai.tool.input.{k}"] = str(v)[:200]
                if arguments:
                    input_value, input_mime = _io_value(arguments, 1000)
                    tool_attrs[_OI_INPUT_VALUE] = input_value
                    tool_attrs[_OI_INPUT_MIME_TYPE] = input_mime
                tool_attrs.update(_event_trace_attributes(call_event, "request"))
            result = event.data.get("result", "")
            if result:
                tool_attrs["gen_ai.tool.output"] = str(result)[:500]
                output_value, output_mime = _io_value(result, 500)
                tool_attrs[_OI_OUTPUT_VALUE] = output_value
                tool_attrs[_OI_OUTPUT_MIME_TYPE] = output_mime
            tool_attrs.update(_event_trace_attributes(event, "response"))
            tool_attrs["metadata"] = _openinference_metadata(tool_attrs)

            child_spans.append({
                "traceId": trace_id,
                "spanId": _to_span_id(
                    call_event.event_id if call_event else event.event_id
                ),
                "parentSpanId": root_span_id,
                "name": f"gen_ai.tool.call/{tool_name}",
                "kind": 3,  # SPAN_KIND_CLIENT
                "startTimeUnixNano": _ts_to_nanos(span_start),
                "endTimeUnixNano": _ts_to_nanos(
                    span_start + duration_ns / 1_000_000_000
                ),
                "attributes": _make_attributes(tool_attrs),
                "status": {"code": 1},
            })

        elif event.event_type == EventType.ERROR:
            req_event = requests.match_parent(event)
            if req_event is not None:
                # An error linked to an LLM request ends that request, so it
                # exports as one failed model call instead of a tool error.
                child_spans.append(_genai_llm_span(
                    meta, trace_id, root_span_id, req_event, event,
                ))
                continue

            call_event = calls.match(event)

            tool_name = event.data.get("tool_name", "error")
            error_msg = event.data.get("error", "") or event.data.get("message", "")
            span_start = call_event.timestamp if call_event else event.timestamp
            duration_ns = _duration_to_nanos(event.duration_ms)

            exc_event = _make_event("exception", event.timestamp, {
                "exception.type": event.data.get("error_type", "Error"),
                "exception.message": str(error_msg)[:500],
                "exception.escaped": False,
                **_event_trace_attributes(event, "response"),
            })

            error_attrs: dict[str, Any] = {
                _SEMCONV_TOOL_NAME: tool_name,
                _SEMCONV_OP: "execute",
                _SEMCONV_TOOL_CALL_ID: (
                    call_event.event_id if call_event else event.event_id
                ),
                _OI_SPAN_KIND: "TOOL",
                _OI_SESSION_ID: meta.session_id,
                **_conversation_attributes(meta),
                _OI_TOOL_NAME: tool_name,
            }
            if call_event and isinstance(call_event.data, Mapping):
                arguments = call_event.data.get("arguments", {})
                if isinstance(arguments, Mapping):
                    for k, v in arguments.items():
                        error_attrs[f"gen_ai.tool.input.{k}"] = str(v)[:200]
                if arguments:
                    input_value, input_mime = _io_value(arguments, 1000)
                    error_attrs[_OI_INPUT_VALUE] = input_value
                    error_attrs[_OI_INPUT_MIME_TYPE] = input_mime
                error_attrs.update(_event_trace_attributes(call_event, "request"))
            error_attrs.update(_event_trace_attributes(event, "response"))
            error_attrs["metadata"] = _openinference_metadata(error_attrs)

            child_spans.append({
                "traceId": trace_id,
                "spanId": _to_span_id(
                    call_event.event_id if call_event else event.event_id
                ),
                "parentSpanId": root_span_id,
                "name": f"gen_ai.tool.call/{tool_name}",
                "kind": 3,
                "startTimeUnixNano": _ts_to_nanos(span_start),
                "endTimeUnixNano": _ts_to_nanos(
                    span_start + duration_ns / 1_000_000_000
                ),
                "attributes": _make_attributes(error_attrs),
                "events": [exc_event],
                "status": {"code": 2, "message": str(error_msg)[:200]},
            })

    # Emit unmatched LLM requests as minimal spans
    for req_event in requests.unmatched():
        model = req_event.data.get("model", "")
        raw_llm_provider, system = _provider_value(
            meta,
            str(model),
            req_event.data,
        )
        llm_attrs: dict[str, Any] = {
            _SEMCONV_SYSTEM: system,
            _SEMCONV_OP: "chat",
            _OI_SPAN_KIND: "LLM",
            _OI_SESSION_ID: meta.session_id,
            **_conversation_attributes(meta),
            _OI_LLM_PROVIDER: system if system != "unknown" else raw_llm_provider,
            "agent.strace.provider": raw_llm_provider,
            **_event_trace_attributes(req_event, "request"),
        }
        if model:
            llm_attrs[_SEMCONV_MODEL] = model
            llm_attrs[_OI_LLM_MODEL] = model
        inp_tok = _token_count(req_event.data, _INPUT_TOKEN_KEYS)
        out_tok = _token_count(req_event.data, _OUTPUT_TOKEN_KEYS)
        if inp_tok is not None:
            llm_attrs[_SEMCONV_INPUT_TOKENS] = inp_tok
            llm_attrs[_OI_INPUT_TOKENS] = inp_tok
        if out_tok is not None:
            llm_attrs[_SEMCONV_OUTPUT_TOKENS] = out_tok
            llm_attrs[_OI_OUTPUT_TOKENS] = out_tok
        if inp_tok is not None or out_tok is not None:
            llm_attrs[_OI_TOTAL_TOKENS] = (inp_tok or 0) + (out_tok or 0)
        llm_attrs.update(_cost_attributes(req_event.data, {}))
        llm_attrs["metadata"] = _openinference_metadata(llm_attrs)
        child_spans.append({
            "traceId": trace_id,
            "spanId": _to_span_id(req_event.event_id),
            "parentSpanId": root_span_id,
            "name": "gen_ai.client.operation",
            "kind": 3,
            "startTimeUnixNano": _ts_to_nanos(req_event.timestamp),
            "endTimeUnixNano": _ts_to_nanos(req_event.timestamp + 0.001),
            "attributes": _make_attributes(llm_attrs),
            "status": {"code": 0},
        })

    # Emit unmatched tool calls
    for call_event in calls.unmatched():
        tool_name = call_event.data.get("tool_name", "tool")
        tool_attrs: dict[str, Any] = {
            _SEMCONV_TOOL_NAME: tool_name,
            _SEMCONV_OP: "execute",
            _SEMCONV_TOOL_CALL_ID: call_event.event_id,
            _OI_SPAN_KIND: "TOOL",
            _OI_SESSION_ID: meta.session_id,
            **_conversation_attributes(meta),
            _OI_TOOL_NAME: tool_name,
            **_event_trace_attributes(call_event, "request"),
        }
        arguments = call_event.data.get("arguments", {})
        if isinstance(arguments, Mapping):
            for k, v in arguments.items():
                tool_attrs[f"gen_ai.tool.input.{k}"] = str(v)[:200]
        if arguments:
            input_value, input_mime = _io_value(arguments, 1000)
            tool_attrs[_OI_INPUT_VALUE] = input_value
            tool_attrs[_OI_INPUT_MIME_TYPE] = input_mime
        tool_attrs["metadata"] = _openinference_metadata(tool_attrs)
        child_spans.append({
            "traceId": trace_id,
            "spanId": _to_span_id(call_event.event_id),
            "parentSpanId": root_span_id,
            "name": f"gen_ai.tool.call/{tool_name}",
            "kind": 3,
            "startTimeUnixNano": _ts_to_nanos(call_event.timestamp),
            "endTimeUnixNano": _ts_to_nanos(call_event.timestamp + 0.001),
            "attributes": _make_attributes(tool_attrs),
            "status": {"code": 0},
        })

    root_span: dict = {
        "traceId": trace_id,
        "spanId": root_span_id,
        "name": f"gen_ai.agent.session ({meta.agent_name or 'agent'})",
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": root_start,
        "endTimeUnixNano": root_end,
        "attributes": root_attrs,
        "events": root_events,
        "status": {"code": 2 if meta.errors > 0 else 1},
    }
    if parent_span_id:
        root_span["parentSpanId"] = parent_span_id

    if meta.tenant_id:
        tenant_attributes = _make_attributes({"tenant_id": meta.tenant_id})
        for span in [root_span] + child_spans:
            span.setdefault("attributes", []).extend(tenant_attributes)

    return {
        "resourceSpans": [{
            "resource": {
                "attributes": _make_attributes({
                    "service.name": service_name,
                    "service.version": "agent-trace",
                    "agent.session_id": meta.session_id,
                    **({"tenant_id": meta.tenant_id} if meta.tenant_id else {}),
                }),
            },
            "scopeSpans": [{
                "scope": {"name": "agent-trace", "version": "genai-semconv-1.27"},
                "spans": [root_span] + child_spans,
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
    return export_otlp_payload(
        payload,
        endpoint,
        headers=headers,
        event_count=len(events),
    )


def export_otlp_payload(
    payload: dict,
    endpoint: str,
    headers: dict[str, str] | None = None,
    event_count: int = 0,
) -> bool:
    """Send a prepared ExportTraceServiceRequest via OTLP/HTTP JSON."""
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
                sys.stderr.write(f"Exported {event_count} events to {url} (HTTP {status})\n")
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
