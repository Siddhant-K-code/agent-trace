"""OTLP/HTTP JSON exporter.

Converts agent-trace sessions to OpenTelemetry spans and sends them
to any OTLP-compatible collector over HTTP/JSON. Zero dependencies.

Each agent-trace session becomes an OTel trace. Tool calls become spans.
User prompts and assistant responses become events on the root span. A model
response that ``agent-strace import`` read from a transcript, with its
recorded model and token usage, also becomes a ``chat {model}`` span.

Works with: Datadog, Honeycomb, New Relic, Splunk, Grafana Tempo, Jaeger.

Usage:
    agent-strace export <session-id> --format otlp --endpoint http://localhost:4318
    agent-strace export <session-id> --format otlp --endpoint https://api.honeycomb.io
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from . import __version__
from .cost import (
    TokenUsage,
    is_bedrock_model,
    reasoning_output_tokens,
    recorded_response_indexes,
    usage_detail,
)
from .event_data import (
    OTLP_LATEST_TIME,
    OTLP_MAX_UNIX_NANOS,
    NonObjectEventDataError,
    UnusableTimestampError,
    event_id_of,
    is_redacted,
    object_field,
    otlp_unix_nanos,
    parent_id_of,
    str_field,
)
# ERROR_TYPE_* and LLMFailure stay importable from here.
from .failures import (
    ERROR_TYPE_CANCELLED,
    ERROR_TYPE_OTHER,
    LLMFailure,
    error_text as _error_text,
    last_request_failure,
    llm_failure,
)
from .models import (
    EventType,
    SessionMeta,
    TraceEvent,
    usable_count,
    usable_number,
    usable_object,
    usable_text,
    usable_time,
)
from .http_post import no_redirect_opener, otlp_traces_url, post_error_detail, url_endpoint
from .otlp_proto import (
    CONTENT_TYPES,
    PROTOCOL_JSON,
    EncodeError,
    encode_body,
    unsupported_media_hint,
)
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
from .session_outcome import behavior_analysis, recorded_session_status
from .store import TraceStore, validate_optional_tenant_id

if TYPE_CHECKING:
    from .eval.results import StoredEvaluation
    from .subagent import SessionNode


# ---------------------------------------------------------------------------
# OTel semantic conventions for GenAI. ``session_to_otlp_genai`` follows
# https://github.com/open-telemetry/semantic-conventions-genai release
# v0.1.0 (2026-11-21); see docs/production.md.
# ---------------------------------------------------------------------------

# Deprecated: renamed to gen_ai.provider.name in semantic-conventions v1.37.0.
# The GenAI format still emits it beside gen_ai.provider.name.
_SEMCONV_SYSTEM = "gen_ai.system"
_SEMCONV_PROVIDER = "gen_ai.provider.name"
_SEMCONV_OP = "gen_ai.operation.name"
_SEMCONV_MODEL = "gen_ai.request.model"
_SEMCONV_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_SEMCONV_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_SEMCONV_REASONING_TOKENS = "gen_ai.usage.reasoning.output_tokens"
_SEMCONV_AGENT_NAME = "gen_ai.agent.name"
_SEMCONV_AGENT_ID = "gen_ai.agent.id"

_OP_CHAT = "chat"
_OP_EXECUTE_TOOL = "execute_tool"
_OP_INVOKE_AGENT = "invoke_agent"

# Well-known gen_ai.provider.name values.
_WELL_KNOWN_PROVIDERS = frozenset({
    "anthropic", "aws.bedrock", "azure.ai.inference", "azure.ai.openai",
    "cohere", "deepseek", "gcp.gemini", "gcp.gen_ai", "gcp.vertex_ai", "groq",
    "ibm.watsonx.ai", "mistral", "moonshot_ai", "ollama", "openai",
    "perplexity", "x_ai",
})
_MISTRAL_MARKERS = (
    "mistral", "mixtral", "codestral", "devstral", "magistral", "ministral",
    "pixtral",
)
# Agent products that can run models from more than one provider: the hook
# ``--provider`` keys and agent names of Codex, Copilot and Cursor. They name
# no gen_ai.provider.name, so the model decides instead.
_AGENT_PRODUCTS = frozenset({
    "codex", "openai-codex", "copilot", "github-copilot", "cursor",
    "cursor-agent",
})
_SEMCONV_CACHE_READ_TOKENS = "gen_ai.usage.cache_read.input_tokens"
_SEMCONV_CACHE_WRITE_TOKENS = "gen_ai.usage.cache_write.input_tokens"
_SEMCONV_TOOL_NAME = "gen_ai.tool.name"
_SEMCONV_TOOL_CALL_ID = "gen_ai.tool.call.id"
_SEMCONV_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
_SEMCONV_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
_SEMCONV_INPUT_MESSAGES = "gen_ai.input.messages"
_SEMCONV_OUTPUT_MESSAGES = "gen_ai.output.messages"
_SEMCONV_FINISH_REASON = "gen_ai.response.finish_reasons"
_SEMCONV_RESPONSE_ID = "gen_ai.response.id"
_SEMCONV_RESPONSE_MODEL = "gen_ai.response.model"
_SEMCONV_SERVER_ADDRESS = "server.address"
_SEMCONV_SERVER_PORT = "server.port"
_SEMCONV_CONVERSATION_ID = "gen_ai.conversation.id"
_SEMCONV_CONVERSATION_COMPACTED = "gen_ai.conversation.compacted"
_SEMCONV_SKILL_NAME = "gen_ai.skill.name"
_SEMCONV_ERROR_TYPE = "error.type"
# The gen_ai.evaluation.result event (development stability in v0.1.0).
_SEMCONV_EVALUATION_RESULT = "gen_ai.evaluation.result"
_SEMCONV_EVALUATION_NAME = "gen_ai.evaluation.name"
_SEMCONV_EVALUATION_SCORE_VALUE = "gen_ai.evaluation.score.value"
_SEMCONV_EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
_SEMCONV_EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"
# The only gen_ai.evaluation.score.label values the export writes.
EVALUATION_LABEL_PASS = "pass"
EVALUATION_LABEL_FAIL = "fail"

# OpenInference compatibility attributes. These are emitted additively by the
# GenAI exporter so the same OTLP/HTTP JSON payload remains useful in Phoenix.
_OI_SPAN_KIND = "openinference.span.kind"
_OI_SESSION_ID = "session.id"
_OI_LLM_PROVIDER = "llm.provider"
_OI_LLM_MODEL = "llm.model_name"
_OI_INPUT_TOKENS = "llm.token_count.prompt"
_OI_OUTPUT_TOKENS = "llm.token_count.completion"
_OI_TOTAL_TOKENS = "llm.token_count.total"
_OI_FINISH_REASON = "llm.finish_reason"
_OI_CACHE_READ_TOKENS = "llm.token_count.prompt_details.cache_read"
_OI_CACHE_WRITE_TOKENS = "llm.token_count.prompt_details.cache_write"
_OI_TOOL_NAME = "tool.name"
_OI_INPUT_VALUE = "input.value"
_OI_INPUT_MIME_TYPE = "input.mime_type"
_OI_OUTPUT_VALUE = "output.value"
_OI_OUTPUT_MIME_TYPE = "output.mime_type"

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
    """Convert session ID to a 32-hex-char trace ID.

    A stored ID can hold a lone surrogate, which strict UTF-8 refuses.
    surrogatepass encodes it to bytes that no other ID encodes to. Every
    other ID keeps its bytes, so its hash does not change.
    """
    h = hashlib.sha256(session_id.encode("utf-8", "surrogatepass")).hexdigest()
    return h[:32]


def _to_span_id(event_id: str) -> str:
    """Convert event ID to a 16-hex-char span ID.

    Encodes the ID as _to_trace_id does.
    """
    h = hashlib.sha256(event_id.encode("utf-8", "surrogatepass")).hexdigest()
    return h[:16]


def _event_span_ids(
    session_id: str, events: list[TraceEvent],
) -> Callable[[TraceEvent], str]:
    """Return a function that gives each event in ``events`` its span ID.

    An event with an ``event_id`` gets ``_to_span_id(event_id)``, so a link
    to that ID, such as a subagent's ``parent_event_id``, finds its span. An
    event whose ``event_id`` is empty or not a string gets an ID made from the
    session, its position in ``events`` and its timestamp, so two such events
    never share one.
    """
    positions = {id(event): position for position, event in enumerate(events)}

    def span_id(event: TraceEvent) -> str:
        event_id = event_id_of(event)
        if event_id:
            return _to_span_id(event_id)
        position = positions.get(id(event))
        return _to_span_id(f"event-{session_id}-{position}-{event.timestamp!r}")

    return span_id


def _ts_to_nanos(ts: float) -> str:
    """Convert Unix timestamp to nanoseconds as string.

    Raises UnusableTimestampError for a value that is not an int or a float,
    because the multiply would repeat a str or a list a billion times. Also
    raises it when the nanoseconds fall outside the 0 to 2**64 - 1 that OTLP
    carries. The OTLP exports refuse such a time before they convert, so this
    is a last guard for a library caller such as tree_to_otlp.
    """
    nanos = otlp_unix_nanos(ts)
    if nanos is None:
        if usable_time(ts) is not None:
            raise UnusableTimestampError(
                f"timestamp is past the latest time OTLP can carry ({OTLP_LATEST_TIME})"
            )
        raise UnusableTimestampError("timestamp must be a Unix time in seconds")
    return str(nanos)


def _one_ms_end_nanos(start: float) -> str:
    """Return the end of a span that starts at *start* and lasts 1ms.

    A span that starts less than 1ms before the latest time OTLP can carry
    ends at that time.
    """
    end = start + 0.001
    if otlp_unix_nanos(end) is None and otlp_unix_nanos(start) is not None:
        return str(OTLP_MAX_UNIX_NANOS)
    return _ts_to_nanos(end)


def _duration_end_nanos(start: float, duration_ns: int) -> str:
    """Return the end of a span that starts at *start* and lasts *duration_ns*.

    A duration whose span would end past the latest time OTLP can carry reads
    as missing, so the span gets the 1ms default of :func:`_duration_to_nanos`.
    """
    end = start + duration_ns / 1_000_000_000
    if otlp_unix_nanos(end) is None and otlp_unix_nanos(start) is not None:
        return _one_ms_end_nanos(start)
    return _ts_to_nanos(end)


def _llm_span_end_nanos(start: float, end: float) -> str:
    """Return the end of a model request span from *start* to *end*.

    The span lasts at least 1ms (see :func:`_one_ms_end_nanos`).
    """
    if start + 0.001 > end:
        return _one_ms_end_nanos(start)
    return _ts_to_nanos(end)


def _root_end_nanos(meta: SessionMeta) -> str:
    """Return the end of the root span: ``ended_at``, or else ``started_at``
    plus ``total_duration_ms``.

    A total whose end would be past the latest time OTLP can carry reads as
    missing, so the root span then ends at ``started_at``.
    """
    if meta.ended_at:
        return _ts_to_nanos(meta.ended_at)
    end = meta.started_at + (usable_number(meta.total_duration_ms) or 0) / 1000
    if otlp_unix_nanos(end) is None and otlp_unix_nanos(meta.started_at) is not None:
        return _ts_to_nanos(meta.started_at)
    return _ts_to_nanos(end)


# ---------------------------------------------------------------------------
# Failed and cancelled model requests
# ---------------------------------------------------------------------------

# The failure predicate lives in failures.py, shared with the session error
# counts; the names stay importable from here.
_FINISH_REASON_ERROR = "error"


def _failure_status(failure: LLMFailure) -> dict:
    """Return the OTLP ERROR status for a failed or cancelled model request."""
    if failure.message:
        return {"code": 2, "message": failure.message}  # STATUS_CODE_ERROR
    return {"code": 2}


def _duration_to_nanos(ms: Any) -> int:
    """Convert milliseconds to nanoseconds.

    A duration that is missing, zero or not a usable number (see
    :func:`usable_number`) gives the 1ms default. So does one too large for a
    float once it is in nanoseconds.
    """
    if not usable_number(ms):
        return 1_000_000  # 1ms default
    nanos = ms * 1_000_000
    if nanos > sys.float_info.max:
        return 1_000_000
    return int(nanos)


class _StringArray(tuple):
    """An attribute value that OTLP carries as an array of strings.

    ``_make_attributes`` writes other lists as one JSON string.
    """


def _finish_reasons(value: object) -> _StringArray:
    """Return a recorded finish reason as ``gen_ai.response.finish_reasons``.

    The GenAI conventions type the attribute as string[], one entry per
    generation. A single recorded reason becomes a one-item array, and a
    recorded list keeps its order; an entry that is not a string is kept as
    its JSON text.
    """
    items = value if isinstance(value, (list, tuple)) else (value,)
    return _StringArray(
        item if isinstance(item, str) else json.dumps(item, default=str)
        for item in items
    )


def _response_server_attributes(
    req_data: Mapping[str, Any],
    resp_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Return ``gen_ai.response.id``, ``gen_ai.response.model``,
    ``server.address`` and ``server.port`` for one model request.

    Each comes only from what a capture path recorded: ``response_id`` and
    ``response_model`` on the response, ``server_address`` and
    ``server_port`` on the request. The OpenAI and Anthropic SDK integrations
    record them. A value that is not a non-empty string is left out, and so is
    a port that is not an integer from 1 to 65535 or that has no address.
    """
    attrs: dict[str, Any] = {}
    for key, attr in (
        ("response_id", _SEMCONV_RESPONSE_ID),
        ("response_model", _SEMCONV_RESPONSE_MODEL),
    ):
        value = resp_data.get(key)
        if isinstance(value, str) and value:
            attrs[attr] = value
    address = req_data.get("server_address")
    if isinstance(address, str) and address:
        attrs[_SEMCONV_SERVER_ADDRESS] = address
        port = req_data.get("server_port")
        if isinstance(port, int) and not isinstance(port, bool) and 0 < port < 65536:
            attrs[_SEMCONV_SERVER_PORT] = port
    return attrs


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
        elif isinstance(value, _StringArray):
            values = [{"stringValue": item} for item in value]
            attrs.append({"key": key, "value": {"arrayValue": {"values": values}}})
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


# The longest gen_ai.evaluation.explanation the export writes, in characters.
_EVALUATION_EXPLANATION_LIMIT = 1000


def _evaluation_event(evaluation: StoredEvaluation, time_unix_nano: str) -> dict:
    """Return the gen_ai.evaluation.result event for one stored eval result.

    The score, the pass/fail label and the reason come from the stored
    result as they are; nothing here judges the session.
    """
    attrs: dict[str, Any] = {
        _SEMCONV_EVALUATION_NAME: evaluation.scorer,
        _SEMCONV_EVALUATION_SCORE_VALUE: float(evaluation.score),
        _SEMCONV_EVALUATION_SCORE_LABEL: (
            EVALUATION_LABEL_PASS if evaluation.passed else EVALUATION_LABEL_FAIL
        ),
    }
    if evaluation.reason:
        attrs[_SEMCONV_EVALUATION_EXPLANATION] = (
            evaluation.reason[:_EVALUATION_EXPLANATION_LIMIT]
        )
    return {
        "timeUnixNano": time_unix_nano,
        "name": _SEMCONV_EVALUATION_RESULT,
        "attributes": _make_attributes(attrs),
    }


def _canonical_provider(value: object) -> str:
    """Return the well-known ``gen_ai.provider.name`` value for *value*.

    *value* is a provider name or a model name, such as a LiteLLM model
    string. A serving platform (Bedrock, Azure, Vertex AI, Groq, Ollama) wins
    over the model family, so ``bedrock/anthropic.claude-3`` is
    ``aws.bedrock``. Google values that name neither Vertex AI nor the Gemini
    API are ``gcp.gen_ai``. Anything else is ``unknown``.
    """
    raw = str(value or "").strip().lower()
    if raw in _WELL_KNOWN_PROVIDERS:
        return raw
    if "bedrock" in raw or raw.startswith(("aws", "amazon.")):
        return "aws.bedrock"
    if raw.startswith("azure_ai") or ("azure" in raw and "inference" in raw):
        return "azure.ai.inference"
    if raw == "azure" or raw.startswith("azure/") or ("azure" in raw and "openai" in raw):
        return "azure.ai.openai"
    if "vertex" in raw or "aiplatform" in raw:
        return "gcp.vertex_ai"
    if (
        raw.startswith("gemini/")
        or "generativelanguage" in raw
        or "aistudio" in raw
        or "ai studio" in raw
    ):
        return "gcp.gemini"
    if "watsonx" in raw:
        return "ibm.watsonx.ai"
    if "groq" in raw:
        return "groq"
    if "ollama" in raw:
        return "ollama"
    if "gemini" in raw or "google" in raw:
        return "gcp.gen_ai"
    if "claude" in raw or "anthropic" in raw:
        return "anthropic"
    if "gpt" in raw or "openai" in raw or raw.startswith(("o1", "o3", "o4")):
        return "openai"
    if any(marker in raw for marker in _MISTRAL_MARKERS):
        return "mistral"
    if "deepseek" in raw:
        return "deepseek"
    if (
        "grok" in raw
        or raw in {"xai", "x-ai", "x.ai"}
        or raw.startswith(("xai/", "xai-", "x-ai/", "x-ai-", "x.ai/"))
    ):
        return "x_ai"
    if "cohere" in raw:
        return "cohere"
    if "perplexity" in raw:
        return "perplexity"
    if "moonshot" in raw or "kimi" in raw:
        return "moonshot_ai"
    return "unknown"


def _provider_value(
    meta: SessionMeta,
    model: str = "",
    *event_data: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the raw provider evidence and its canonical GenAI value.

    A recorded provider wins over the model, unless it names a multi-model
    agent product (``_AGENT_PRODUCTS``), as the Codex hook does on every
    model request. Then the model decides, and the product stays the raw
    evidence; with no recognizable model the value is ``unknown``. A model
    that :func:`agent_trace.cost.is_bedrock_model` accepts, such as a Bedrock
    model ID, inference-profile ID or ARN, is ``aws.bedrock``.
    """
    product = ""
    for data in event_data:
        for key in ("provider", "model_provider", "system"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                if value.strip().lower() in _AGENT_PRODUCTS:
                    product = product or value.strip()
                    continue
                return value.strip(), _canonical_provider(value)
    if model:
        canonical = "aws.bedrock" if is_bedrock_model(model) else _canonical_provider(model)
        if canonical != "unknown":
            return product or canonical, canonical
    if product:
        return product, "unknown"
    attribution = meta.attribution if isinstance(meta.attribution, dict) else {}
    raw = attribution.get("agent_provider") or usable_text(meta.agent_name)
    if isinstance(raw, str) and raw.strip().lower() in _AGENT_PRODUCTS:
        return raw, "unknown"
    return str(raw or "unknown"), _canonical_provider(raw)


def model_request_usage(
    request_data: Any,
    response_data: Any,
) -> tuple[TokenUsage, bool, bool]:
    """Return ``(usage, has_input, has_output)`` for one model request.

    Counts are read with the cost report's provider normalization
    (:func:`agent_trace.cost.usage_detail`), so the input is the whole input,
    cache reads and writes included, whichever way the provider counts it.
    The input side (the input and its cache reads and writes) comes from the
    request when it recorded an input count, otherwise from the response. The
    output comes from the response when it recorded one, otherwise from the
    request.
    """
    request, request_input, request_output = usage_detail(request_data)
    response, response_input, response_output = usage_detail(response_data)
    usage = request if request_input else response
    output = response.output_tokens if response_output else request.output_tokens
    return (
        replace(usage, output_tokens=output),
        request_input or response_input,
        request_output or response_output,
    )


def _cache_attributes(usage: TokenUsage, openinference: bool = False) -> dict[str, int]:
    """Return the cache read and write counts of *usage* that are not zero.

    The GenAI conventions record them "when available and non-zero". They are
    part of ``gen_ai.usage.input_tokens``, not added to it.
    """
    attrs: dict[str, int] = {}
    if usage.cache_read_tokens:
        attrs[_SEMCONV_CACHE_READ_TOKENS] = usage.cache_read_tokens
        if openinference:
            attrs[_OI_CACHE_READ_TOKENS] = usage.cache_read_tokens
    if usage.cache_write_tokens:
        attrs[_SEMCONV_CACHE_WRITE_TOKENS] = usage.cache_write_tokens
        if openinference:
            attrs[_OI_CACHE_WRITE_TOKENS] = usage.cache_write_tokens
    return attrs


def _token_attributes(
    usage: TokenUsage,
    has_input: bool,
    has_output: bool,
) -> dict[str, int]:
    """Return the GenAI and OpenInference token attributes of one model request."""
    attrs: dict[str, int] = {}
    if has_input:
        attrs[_SEMCONV_INPUT_TOKENS] = usage.input_tokens
        attrs[_OI_INPUT_TOKENS] = usage.input_tokens
        attrs.update(_cache_attributes(usage, openinference=True))
    if has_output:
        attrs[_SEMCONV_OUTPUT_TOKENS] = usage.output_tokens
        attrs[_OI_OUTPUT_TOKENS] = usage.output_tokens
    if has_input or has_output:
        attrs[_OI_TOTAL_TOKENS] = usage.input_tokens + usage.output_tokens
    return attrs


def _reasoning_attributes(output_data: Any, usage: TokenUsage, has_output: bool) -> dict[str, int]:
    """Return ``gen_ai.usage.reasoning.output_tokens`` when it is not zero.

    *output_data* is the event the output count was read from. The reasoning
    count is part of ``gen_ai.usage.output_tokens``, not added to it.
    """
    if not has_output:
        return {}
    reasoning = min(reasoning_output_tokens(output_data), usage.output_tokens)
    return {_SEMCONV_REASONING_TOKENS: reasoning} if reasoning else {}


def _model_span_name(model: object) -> str:
    """``{gen_ai.operation.name} {gen_ai.request.model}``, or the operation."""
    return f"{_OP_CHAT} {model}" if isinstance(model, str) and model else _OP_CHAT


def _tool_span_name(tool_name: object, skill_name: str = "") -> str:
    """``execute_tool {gen_ai.tool.name}``, then ``{gen_ai.skill.name}`` if any."""
    if skill_name:
        return f"{_OP_EXECUTE_TOOL} {tool_name} {skill_name}"
    return f"{_OP_EXECUTE_TOOL} {tool_name}"


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
        f"{prefix}.event_id": event_id_of(event),
        f"{prefix}.redacted": is_redacted(event),
    }
    parent_id = parent_id_of(event)
    if parent_id:
        attrs[f"{prefix}.parent_event_id"] = parent_id
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


# Recorded session statuses that end the root span in ERROR. Each is its own
# error.type.
_FAILED_SESSION_STATUSES = frozenset({"timeout", "terminated"})


def _tool_error_type(event: TraceEvent) -> str:
    """Return the error.type of a failed tool outcome with object data."""
    return (
        str_field(event.data, "error_type").strip()
        or str_field(event.data, "exception_type").strip()
        or ERROR_TYPE_OTHER
    )


def _outcome_event(event: TraceEvent) -> TraceEvent:
    """Return ``event`` with object data and a number timestamp.

    Data that is not an object reads as empty. An unusable timestamp becomes
    NaN, which orders neither before nor after another event.
    """
    data = event.data if isinstance(event.data, Mapping) else {}
    timestamp = usable_number(event.timestamp)
    if data is event.data and timestamp is not None:
        return event
    return replace(
        event,
        data=data,
        timestamp=math.nan if timestamp is None else timestamp,
    )


def _root_outcome(events: list[TraceEvent]) -> tuple[dict[str, Any], dict]:
    """Return the session root's outcome attributes and its span status.

    The root is ERROR when the recorded session status is timeout or
    terminated, the last model request failed, or a failed tool operation
    never recovered. Its error.type is then the session status, else the
    failed request's error type, else the error type of the last unresolved
    failure. Any other root is UNSET, never OK: the export does not claim
    the session succeeded. Failed spans below the root keep their own status.
    """
    usable = [_outcome_event(event) for event in events]
    analysis = behavior_analysis(usable)
    behavior = analysis.summary
    session_status = recorded_session_status(usable)
    request_failure = last_request_failure(usable)
    attributes: dict[str, Any] = {
        "agent.strace.behavior.failed_operation_count": behavior.failed_operation_count,
        "agent.strace.behavior.retry_count": behavior.retry_count,
        "agent.strace.behavior.recovered_operation_count": behavior.recovered_operation_count,
        "agent.strace.behavior.unresolved_failed_operation_count": (
            behavior.unresolved_failed_operation_count
        ),
        "agent.strace.session_status": session_status,
        "agent.strace.last_model_request_failed": request_failure is not None,
    }
    if session_status in _FAILED_SESSION_STATUSES:
        error_type = session_status
    elif request_failure is not None:
        error_type = request_failure.error_type
    elif analysis.unresolved_failures:
        error_type = _tool_error_type(usable[analysis.unresolved_failures[-1]])
    else:
        return attributes, {"code": 0}  # STATUS_CODE_UNSET
    # Recording Errors: error.type MUST be set when the operation fails.
    attributes[_SEMCONV_ERROR_TYPE] = error_type
    return attributes, {"code": 2}  # STATUS_CODE_ERROR


def _io_value(value: object, limit: int) -> tuple[str, str]:
    """Serialize an OpenInference input or output value with a MIME type."""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if len(text) <= limit:
            return text, "application/json"
        return text[:limit], "text/plain"
    return str(value)[:limit], "text/plain"


# The GenAI content attributes are JSON text, the way the OTel Python GenAI
# instrumentations write them on spans. Each carries only text that the older
# keys beside it already carry, within the same limits.
_MESSAGE_TEXT_LIMIT = 1000
_TOOL_ARGUMENTS_LIMIT = 1000
_TOOL_ARGUMENT_VALUE_LIMIT = 200
_TOOL_RESULT_LIMIT = 500


def _compact_json(value: object, sort_keys: bool) -> str:
    return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"))


def _json_string(text: str, limit: int) -> str:
    """Return the longest leading part of ``text`` whose JSON string literal
    fits in ``limit`` characters, as that literal."""
    low, high = 0, min(len(text), max(limit - 2, 0))
    while low < high:
        middle = (low + high + 1) // 2
        if len(_compact_json(text[:middle], False)) <= limit:
            low = middle
        else:
            high = middle - 1
    return _compact_json(text[:low], False)


def _cut_json(value: object, limit: int, sort_keys: bool) -> tuple[str, bool] | None:
    """Return the leading part of ``value`` that fits in ``limit`` characters
    as compact JSON, and whether that is all of it.

    Members of an object or array are kept in order while they fit; the
    first that does not is cut the same way, and the rest are dropped. A
    string keeps its longest leading part that fits. None means not even an
    empty value of the same kind fits.
    """
    text = _compact_json(value, sort_keys)
    if len(text) <= limit:
        return text, True
    if isinstance(value, str):
        return (_json_string(value, limit), False) if limit >= 2 else None
    if isinstance(value, dict):
        keys = sorted(value) if sort_keys else list(value)
        members = [
            (_compact_json(key if isinstance(key, str) else str(key), False) + ":", value[key])
            for key in keys
        ]
        opening, closing = "{", "}"
    elif isinstance(value, list):
        members = [("", item) for item in value]
        opening, closing = "[", "]"
    else:
        return None
    if limit < 2:
        return None
    parts: list[str] = []
    used = 2
    for prefix, member in members:
        prefix = ("," if parts else "") + prefix
        cut = _cut_json(member, limit - used - len(prefix), sort_keys)
        if cut is None:
            break
        parts.append(prefix + cut[0])
        used += len(prefix) + len(cut[0])
        if not cut[1]:
            break
    return opening + "".join(parts) + closing, False


def _json_content(value: object, limit: int) -> str:
    """Return tool call arguments or a tool result for its GenAI content
    attribute, in at most ``limit`` characters.

    A dict or list is compact JSON. A string that starts with ``{`` or ``[``
    is kept when it is JSON that fits. Either kind of JSON that does not fit
    is cut with its structure kept (see :func:`_cut_json`), so it stays valid
    JSON. A string that starts with ``{`` or ``[`` but is not JSON, such as
    the Python text of a dict a hook stored, becomes a JSON string, so a
    reader that parses such values gets one. Any other value is its text, cut
    at ``limit``, as before.
    """
    if isinstance(value, (dict, list)):
        parsed: object = value
        sort_keys = True
    elif isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            return _json_string(value, limit)
        if len(value) <= limit:
            return value
        sort_keys = False
    else:
        return str(value)[:limit]
    try:
        cut = _cut_json(parsed, limit, sort_keys)
    except RecursionError:
        cut = None
    if cut is not None:
        return cut[0]
    text = value if isinstance(value, str) else _compact_json(value, True)
    return _json_string(text, limit)


def _json_text(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _text_message(role: str, texts: list[str]) -> dict:
    """Return a semconv chat message with one TextPart per text."""
    return {"role": role, "parts": [{"type": "text", "content": text} for text in texts]}


def _root_messages(events: list[TraceEvent]) -> dict[str, str]:
    """Return ``gen_ai.input.messages`` and ``gen_ai.output.messages`` for the root.

    The input holds every user prompt and assistant reply up to and including
    the last user_prompt, in event order. The output holds the replies after
    it as one assistant message with one text part per reply; with no
    user_prompt, every reply is output. Each text is the one the
    ``gen_ai.user.message`` and ``gen_ai.assistant.message`` events carry, cut
    at 1000 characters. A text that is empty or not a string is skipped, as a
    missing one is. An attribute with no message is left out.
    """
    turns: list[tuple[str, str]] = []
    boundary = 0
    for event in events:
        if event.event_type == EventType.USER_PROMPT:
            text = str_field(event.data, "prompt", "")[:_MESSAGE_TEXT_LIMIT]
            if text:
                turns.append(("user", text))
            boundary = len(turns)
        elif event.event_type == EventType.ASSISTANT_RESPONSE:
            text = str_field(event.data, "text", "")[:_MESSAGE_TEXT_LIMIT]
            if text:
                turns.append(("assistant", text))
    attrs: dict[str, str] = {}
    if boundary:
        attrs[_SEMCONV_INPUT_MESSAGES] = _json_text([
            _text_message(role, [text]) for role, text in turns[:boundary]
        ])
    replies = [text for _, text in turns[boundary:]]
    if replies:
        attrs[_SEMCONV_OUTPUT_MESSAGES] = _json_text([_text_message("assistant", replies)])
    return attrs


def _tool_input_attributes(arguments: object) -> dict[str, str]:
    """Return the GenAI tool input keys for a tool call's recorded ``arguments``.

    A mapping gives one ``gen_ai.tool.input.<key>`` per argument, cut at 200
    characters. Any truthy value gives ``input.value`` and its MIME type, and
    ``gen_ai.tool.call.arguments``. For a mapping that is the ``input.value``
    text when it is the whole JSON object. A mapping too long for
    ``input.value`` gives the JSON object of the per-key values instead, so
    the attribute stays an object and carries no new text. Any other value
    gives the text :func:`_json_content` returns, which is valid JSON when
    the value is JSON.
    """
    attrs: dict[str, str] = {}
    if isinstance(arguments, Mapping):
        for k, v in arguments.items():
            attrs[f"gen_ai.tool.input.{k}"] = str(v)[:_TOOL_ARGUMENT_VALUE_LIMIT]
    if arguments:
        input_value, input_mime = _io_value(arguments, _TOOL_ARGUMENTS_LIMIT)
        attrs[_OI_INPUT_VALUE] = input_value
        attrs[_OI_INPUT_MIME_TYPE] = input_mime
        if not isinstance(arguments, Mapping):
            attrs[_SEMCONV_TOOL_CALL_ARGUMENTS] = _json_content(
                arguments, _TOOL_ARGUMENTS_LIMIT,
            )
        elif input_mime == "application/json":
            attrs[_SEMCONV_TOOL_CALL_ARGUMENTS] = input_value
        else:
            attrs[_SEMCONV_TOOL_CALL_ARGUMENTS] = json.dumps(
                {
                    str(k): str(v)[:_TOOL_ARGUMENT_VALUE_LIMIT]
                    for k, v in arguments.items()
                },
                sort_keys=True,
                separators=(",", ":"),
            )
    return attrs


def _error_tool_name(event: TraceEvent, call_event: TraceEvent | None) -> Any:
    """Return the tool name for an error span.

    That is the error's own ``tool_name``, else the paired tool_call's, else
    ``error``. The ``@trace_tool`` decorator, for one, records the name only
    on the call. A failed tool_result with neither is named ``tool``, as an
    unpaired result that did not fail is: it is a tool's outcome, not a call
    to a tool named error.
    """
    name = event.data.get("tool_name", "")
    if not name and call_event is not None and isinstance(call_event.data, Mapping):
        name = call_event.data.get("tool_name", "")
    if name:
        return name
    return "tool" if event.event_type == EventType.TOOL_RESULT else "error"


def _tool_call_id(call_event: TraceEvent | None, event: TraceEvent | None = None) -> str:
    """Return the gen_ai.tool.call.id of a tool span.

    That is the model's own ID for the call, the ``tool_use_id`` that the
    hooks and ``agent-strace import`` store, taken from the call, else from
    its outcome. Without one it is the call's event ID, else the outcome's.
    """
    for source in (call_event, event):
        if source is not None and isinstance(source.data, Mapping):
            value = source.data.get("tool_use_id")
            if isinstance(value, str) and value.strip():
                return value
    return event_id_of(call_event if call_event else event)


def _failed_tool_result(event: TraceEvent) -> bool:
    """Return whether ``event`` is a tool_result that records a failure.

    A Claude Code transcript marks a failed tool result ``is_error: true``,
    and ``agent-strace import`` keeps the flag. Such a result exports like an
    error that ends its tool call, with the result text as the message.
    """
    return event.event_type == EventType.TOOL_RESULT and event.data.get("is_error") is True


def _tool_error_message(event: TraceEvent) -> Any:
    """Return the message of an error, or the text of a failed tool_result."""
    message = event.data.get("error", "") or event.data.get("message", "")
    if not message and event.event_type == EventType.TOOL_RESULT:
        message = event.data.get("result", "")
    return message


_LINK_TYPE = "agent.strace.link.type"


class _RetryLinks:
    """Collect the ``retry_of`` span links of one payload.

    ``add`` records the events one span was built from. ``apply`` links each
    span whose events name a ``retry_of`` event ID to the span built from that
    event, when that is another span of the same payload. The first span to
    record an event ID keeps it. Event data that is not an object is skipped.
    """

    def __init__(self) -> None:
        self._span_of: dict[str, dict] = {}
        self._retries: list[tuple[dict, list[str]]] = []

    def add(self, span: dict, *events: TraceEvent | None) -> dict:
        retry_ids: list[str] = []
        for event in events:
            if event is None:
                continue
            event_id = event_id_of(event)
            if event_id:
                self._span_of.setdefault(event_id, span)
            if isinstance(event.data, Mapping):
                retry_of = event.data.get("retry_of")
                if isinstance(retry_of, str) and retry_of and retry_of not in retry_ids:
                    retry_ids.append(retry_of)
        if retry_ids:
            self._retries.append((span, retry_ids))
        return span

    def apply(self) -> None:
        for span, retry_ids in self._retries:
            links: list[dict] = []
            linked: set[str] = set()
            for retry_of in retry_ids:
                target = self._span_of.get(retry_of)
                if target is None or target is span or target["spanId"] in linked:
                    continue
                linked.add(target["spanId"])
                links.append({
                    "traceId": target["traceId"],
                    "spanId": target["spanId"],
                    "attributes": _make_attributes({_LINK_TYPE: "retry_of"}),
                })
            if links:
                span["links"] = links


# ---------------------------------------------------------------------------
# Model responses read from a transcript
# ---------------------------------------------------------------------------

# Marks a span whose start time was not recorded but inferred from other
# events, so its duration is not a measured latency.
_TIMING = "agent.strace.timing"
_TIMING_INFERRED = "inferred"
# Events that give the model new input. The next model request of the same
# context stream cannot have started before them.
_MODEL_INPUT_EVENT_TYPES = frozenset({EventType.USER_PROMPT, EventType.TOOL_RESULT})


def _imported_response_starts(events: list[TraceEvent]) -> dict[int, float]:
    """Return the inferred span start of each recorded model response.

    The responses are the ``assistant_response`` events that
    :func:`agent_trace.cost.recorded_response_indexes` names: one per model
    message whose usage ``agent-strace import`` read from a Claude Code
    transcript. A transcript line has one timestamp, so the request's start
    was not recorded. Each span ends at its response and starts at the latest
    earlier event, in trace order, of the same ``context_stream`` that is a
    ``user_prompt``, a ``tool_result`` or another of these responses. With no
    such event the span has zero length at the response. A start later than
    the response is the response's time.

    A ``<synthetic>`` message, which Claude Code writes itself for an API
    error without sending a model request, gets no span and moves no start.

    The result maps ``id()`` of each response event to its start.
    """
    # Claude Code writes "<synthetic>" messages itself, with no model request.
    counted = {
        index for index in recorded_response_indexes(events)
        if event_model(events[index].data) != "<synthetic>"
    }
    if not counted:
        return {}
    starts: dict[int, float] = {}
    latest: dict[str, float] = {}
    for index, event in enumerate(events):
        if index not in counted and event.event_type not in _MODEL_INPUT_EVENT_TYPES:
            continue
        data = event.data if isinstance(event.data, Mapping) else {}
        stream = str_field(data, "context_stream", "")
        if index in counted:
            starts[id(event)] = min(latest.get(stream, event.timestamp), event.timestamp)
        latest[stream] = event.timestamp
    return starts


def _imported_response_attributes(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the recorded response model and ID, and the inferred timing mark.

    ``gen_ai.response.id`` is the transcript ``message.id`` that import stores
    as ``message_id``. A session imported before import stored it has none.
    """
    attrs: dict[str, Any] = {}
    model = event_model(data)
    if model:
        attrs[_SEMCONV_RESPONSE_MODEL] = model
    message_id = str_field(data, "message_id", "")
    if message_id:
        attrs[_SEMCONV_RESPONSE_ID] = message_id
    attrs[_TIMING] = _TIMING_INFERRED
    return attrs


# Events a ``chat`` span is built from: its request, or without one its
# response or the error that ended it.
_CHAT_SPAN_EVENT_TYPES = frozenset({
    EventType.LLM_REQUEST,
    EventType.LLM_RESPONSE,
    EventType.ASSISTANT_RESPONSE,
    EventType.ERROR,
})


def _compacted_context_events(events: list[TraceEvent]) -> set[int]:
    """Return ``id()`` of the events whose model request used a compacted context.

    A context stream (see :func:`agent_trace.compaction._event_stream`) counts
    as compacted from the first recorded sign that its context was compacted:

    - a ``session_start`` whose ``source`` is ``compact``, which Claude Code
      and Codex hooks record when a session continues after compaction (it
      has no ``agent_id``, so it marks the main stream only);
    - a ``user_prompt`` that ``agent-strace import`` marked
      ``is_compaction_summary``, Claude Code's post-compaction summary;
    - an ``assistant_response`` that import gave a ``context_summary``, the
      first response after a Claude Code compaction boundary.

    From there on every ``llm_request``, ``llm_response``,
    ``assistant_response`` and ``error`` of that stream is returned, the
    marking response included. A ``session_start`` with ``source`` resume
    keeps the marks; any other ``session_start`` starts a new context for its
    own stream and clears that stream's mark only, so a compacted subagent
    stays marked when the main conversation starts again. Events whose data
    is not an object are skipped. Nothing else counts, so a request with no
    such sign before it is not returned.
    """
    from .compaction import _event_stream

    compacted: set[str] = set()
    marked: set[int] = set()
    for event in events:
        data = event.data
        if not isinstance(data, Mapping):
            continue
        if event.event_type == EventType.SESSION_START:
            source = data.get("source")
            if source == "compact":
                compacted.add(_event_stream(event))
            elif source != "resume":
                compacted.discard(_event_stream(event))
            continue
        stream = _event_stream(event)
        if (
            event.event_type == EventType.USER_PROMPT
            and data.get("is_compaction_summary") is True
        ) or (
            event.event_type == EventType.ASSISTANT_RESPONSE
            and usable_text(data.get("context_summary"))
        ):
            compacted.add(stream)
        if event.event_type in _CHAT_SPAN_EVENT_TYPES and stream in compacted:
            marked.add(id(event))
    return marked


def _classic_response_span(
    trace_id: str,
    root_span_id: str,
    span_id: str,
    event: TraceEvent,
    start: float,
    gen_ai_system: str,
) -> dict:
    """Build the classic ``chat {model}`` span of one recorded model response.

    It carries the recorded model, token counts and stop reason under the keys
    the classic model request events use. See :func:`_imported_response_starts`
    for its start.
    """
    data = event.data
    model = event_model(data)
    attrs: dict[str, Any] = {_SEMCONV_SYSTEM: gen_ai_system, _SEMCONV_OP: _OP_CHAT}
    if model:
        attrs[_SEMCONV_MODEL] = model
    usage, has_input, has_output = usage_detail(data)
    if has_input:
        attrs[_SEMCONV_INPUT_TOKENS] = usage.input_tokens
        attrs.update(_cache_attributes(usage))
    if has_output:
        attrs[_SEMCONV_OUTPUT_TOKENS] = usage.output_tokens
    finish = data.get("stop_reason") or data.get("finish_reason", "")
    if finish:
        attrs[_SEMCONV_FINISH_REASON] = _finish_reasons(finish)
    attrs.update(_imported_response_attributes(data))
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": root_span_id,
        "name": _model_span_name(model),
        "kind": 3,  # SPAN_KIND_CLIENT
        "startTimeUnixNano": _ts_to_nanos(start),
        "endTimeUnixNano": _ts_to_nanos(event.timestamp),
        "attributes": _make_attributes(attrs),
        "status": {"code": 0},  # STATUS_CODE_UNSET
    }


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

# session_to_otlp reads the timestamps of events of these types, and
# session_to_otlp_genai of the same types but decision. Neither reads the
# timestamp of a tool_result that pairs with a tool_call and did not fail (the
# span starts at the call) or of a tool_call that a later one with the same ID
# replaces before it pairs. session_to_otlp_genai skips such a replaced
# llm_request too.
OTLP_TIMESTAMP_EVENT_TYPES = OTLP_DATA_EVENT_TYPES | {EventType.TOOL_CALL}
OTLP_GENAI_TIMESTAMP_EVENT_TYPES = OTLP_TIMESTAMP_EVENT_TYPES - {EventType.DECISION}


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
    naming no model. With ``skills_and_compaction`` the converter also
    looks at other events, and at a tool_call an error pairs with, but only
    when their data is an object, so they are not listed.
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
    *,
    root_outcome: bool = True,
) -> dict:
    """Convert an agent-trace session to OTLP JSON trace format.

    Returns the full ExportTraceServiceRequest body ready to POST.

    The session's tenant_id goes on the resource and on every span, as
    ``validate_optional_tenant_id`` in store.py returns it. A tenant_id the
    store would refuse raises ValueError.

    Model requests stay events on the root span, except a model response that
    ``agent-strace import`` read from a transcript with recorded usage. That
    one also becomes a ``chat {model}`` CLIENT span with an inferred start
    (see :func:`_imported_response_starts`).

    Args:
        parent_span_id: When set, the root span is linked as a child of this
            span (used for subagent hierarchy).
        parent_trace_id: When set, the root span shares this trace ID so all
            subagent spans appear in the same trace in the backend.
        root_outcome: When True, the root span carries the session's
            outcome attributes and its status follows the session's end
            state (see ``_root_outcome``). Pass False when ``events`` is only
            part of a session: the root then has neither and stays UNSET.
    """
    # Use parent trace ID for subagents so they appear in the same trace
    trace_id = parent_trace_id if parent_trace_id else _to_trace_id(meta.session_id)

    # Root span covers the entire session
    root_span_id = _to_span_id(f"root-{meta.session_id}")
    span_id_of = _event_span_ids(meta.session_id, events)
    root_start = _ts_to_nanos(meta.started_at)
    root_end = _root_end_nanos(meta)

    # Detect provider from agent name for semantic conventions
    agent_lower = usable_text(meta.agent_name).lower()
    if "claude" in agent_lower or "anthropic" in agent_lower:
        gen_ai_system = "anthropic"
    elif "gpt" in agent_lower or "openai" in agent_lower:
        gen_ai_system = "openai"
    elif "gemini" in agent_lower or "google" in agent_lower:
        gen_ai_system = "google"
    else:
        gen_ai_system = "unknown"

    outcome_attributes, root_status = (
        _root_outcome(events) if root_outcome else ({}, {"code": 0})
    )
    root_attrs = _make_attributes({
        # Standard OTel resource attributes
        "agent.name": usable_text(meta.agent_name) or "unknown",
        "agent.command": usable_text(meta.command),
        "agent.session_id": meta.session_id,
        "agent.tool_calls": usable_count(meta.tool_calls),
        "agent.llm_requests": usable_count(meta.llm_requests),
        "agent.errors": usable_count(meta.errors),
        "agent.depth": usable_count(meta.depth),
        # GenAI semantic conventions
        _SEMCONV_SYSTEM: gen_ai_system,
        _SEMCONV_OP: "agent.session",
        **_conversation_attributes(meta),
        **outcome_attributes,
    })

    # Collect span events (user prompts, assistant responses, decisions)
    # and child spans (tool calls, errors)
    root_events = []
    child_spans = []
    retry_links = _RetryLinks()

    # Pair tool_call events with their tool_result or error, and LLM
    # requests with their responses (see pairing.py)
    calls: Pairer[TraceEvent] = tool_pairer()
    requests: Pairer[TraceEvent] = llm_pairer()
    # Model responses read from a transcript, which recorded no request
    response_starts = _imported_response_starts(events)

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
            inferred_start = response_starts.get(id(event))
            if inferred_start is not None:
                child_spans.append(retry_links.add(_classic_response_span(
                    trace_id,
                    root_span_id,
                    span_id_of(event),
                    event,
                    inferred_start,
                    gen_ai_system,
                ), event))

        elif event.event_type == EventType.DECISION:
            root_events.append(_make_event(
                "decision",
                event.timestamp,
                event.data,
            ))

        elif event.event_type == EventType.TOOL_CALL:
            calls.add(event)

        elif event.event_type == EventType.TOOL_RESULT and not _failed_tool_result(event):
            call_event = calls.match(event)

            tool_name = event.data.get("tool_name", "") or (
                call_event.data.get("tool_name", "tool") if call_event else "tool"
            )
            span_start = call_event.timestamp if call_event else event.timestamp
            duration_ns = _duration_to_nanos(event.duration_ms)

            span_attrs = {
                _SEMCONV_TOOL_NAME: tool_name,
                _SEMCONV_OP: "tool.call",
                _SEMCONV_TOOL_CALL_ID: _tool_call_id(call_event, event),
            }
            if call_event:
                args = object_field(call_event.data, "arguments", {})
                if args:
                    for k, v in args.items():
                        span_attrs[f"tool.input.{k}"] = str(v)[:200]

            result = event.data.get("result", "")
            if result:
                span_attrs["tool.output"] = str(result)[:500]

            child_spans.append(retry_links.add({
                "traceId": trace_id,
                "spanId": span_id_of(call_event if call_event else event),
                "parentSpanId": root_span_id,
                "name": f"tool/{tool_name}",
                "kind": 3,  # SPAN_KIND_CLIENT
                "startTimeUnixNano": _ts_to_nanos(span_start),
                "endTimeUnixNano": _duration_end_nanos(span_start, duration_ns),
                "attributes": _make_attributes(span_attrs),
                "status": {"code": 0},  # STATUS_CODE_UNSET
            }, call_event, event))

        elif event.event_type in (EventType.ERROR, EventType.TOOL_RESULT):
            # An error linked to an LLM request ends that request. The classic
            # export has no model request spans, so it still exports as an
            # error span, but not as a tool call. An error with no tool_call
            # and no tool_name of its own, such as "could not start the MCP
            # server", is not a tool call either and gets the same span. A
            # failed tool_result is a tool error, paired or not.
            model_error = (
                event.event_type == EventType.ERROR
                and requests.match_parent(event) is not None
            )
            call_event = calls.match(event)
            model_error = model_error and call_event is None
            lone_error = (
                event.event_type == EventType.ERROR
                and call_event is None
                and not event.data.get("tool_name")
            )

            error_msg = _tool_error_message(event)
            span_start = call_event.timestamp if call_event else event.timestamp
            duration_ns = _duration_to_nanos(event.duration_ms)

            if model_error or lone_error:
                # A lone error's tool_name is absent or empty, so it is named error.
                tool_name = event.data.get("tool_name", "error") if model_error else "error"
                span_name: Any = tool_name
                span_kind = 1  # SPAN_KIND_INTERNAL
                span_attrs = {"tool.name": tool_name}
            else:
                # The same identity as a tool call that returned a result.
                tool_name = _error_tool_name(event, call_event)
                span_name = f"tool/{tool_name}"
                span_kind = 3  # SPAN_KIND_CLIENT
                span_attrs = {
                    _SEMCONV_TOOL_NAME: tool_name,
                    _SEMCONV_OP: "tool.call",
                    _SEMCONV_TOOL_CALL_ID: _tool_call_id(call_event, event),
                    "tool.name": tool_name,
                }
            if error_msg:
                span_attrs["error.message"] = str(error_msg)[:500]
            if call_event:
                args = object_field(call_event.data, "arguments", {})
                if args:
                    for k, v in args.items():
                        span_attrs[f"tool.input.{k}"] = str(v)[:200]

            child_spans.append(retry_links.add({
                "traceId": trace_id,
                "spanId": span_id_of(call_event if call_event else event),
                "parentSpanId": root_span_id,
                "name": span_name,
                "kind": span_kind,
                "startTimeUnixNano": _ts_to_nanos(span_start),
                "endTimeUnixNano": _duration_end_nanos(span_start, duration_ns),
                "attributes": _make_attributes(span_attrs),
                "status": {"code": 2, "message": str(error_msg)[:200]},  # STATUS_CODE_ERROR
                "events": [_make_event("exception", event.timestamp, {
                    "exception.message": str(error_msg)[:500],
                })],
            }, call_event, event))

        elif event.event_type == EventType.LLM_REQUEST:
            requests.add(event)
            llm_attrs: dict = {_SEMCONV_OP: "chat"}
            model = event.data.get("model", "")
            if model:
                llm_attrs[_SEMCONV_MODEL] = model
            usage, _, _ = usage_detail(event.data)
            if usage.input_tokens:
                llm_attrs[_SEMCONV_INPUT_TOKENS] = usage.input_tokens
                llm_attrs.update(_cache_attributes(usage))
            root_events.append(_make_event("gen_ai.system.message", event.timestamp, llm_attrs))

        elif event.event_type == EventType.LLM_RESPONSE:
            llm_attrs = {_SEMCONV_OP: "chat"}
            req_event = requests.match(event)
            req_data = req_event.data if req_event is not None else {}
            if req_event is None:
                # A response with no request (Codex records each model
                # request this way) carries its own model and input count.
                model = event_model(event.data)
                if model:
                    llm_attrs[_SEMCONV_MODEL] = model
            usage, _, _ = model_request_usage(req_data, event.data)
            if not usage_detail(req_data)[1] and usage.input_tokens:
                # The request event carried no input count (the SDK
                # integrations record usage on the response), so the input
                # side is reported here.
                llm_attrs[_SEMCONV_INPUT_TOKENS] = usage.input_tokens
                llm_attrs.update(_cache_attributes(usage))
            if usage.output_tokens:
                llm_attrs[_SEMCONV_OUTPUT_TOKENS] = usage.output_tokens
            finish = event.data.get("stop_reason", event.data.get("finish_reason", ""))
            if finish:
                llm_attrs[_SEMCONV_FINISH_REASON] = _finish_reasons(finish)
            failure = llm_failure(event)
            if failure is not None:
                # The classic export has no span per model request, so a
                # failed or cancelled one is marked on its event instead.
                if not finish:
                    llm_attrs[_SEMCONV_FINISH_REASON] = _finish_reasons(_FINISH_REASON_ERROR)
                llm_attrs[_SEMCONV_ERROR_TYPE] = failure.error_type
                if failure.message:
                    llm_attrs["error.message"] = failure.message
            root_events.append(_make_event("gen_ai.assistant.message", event.timestamp, llm_attrs))

    # Emit any unmatched tool_calls as spans (no result received), with the
    # same identity as a tool call that returned one
    for call_event in calls.unmatched():
        tool_name = call_event.data.get("tool_name", "tool")
        child_spans.append(retry_links.add({
            "traceId": trace_id,
            "spanId": span_id_of(call_event),
            "parentSpanId": root_span_id,
            "name": f"tool/{tool_name}",
            "kind": 3,  # SPAN_KIND_CLIENT
            "startTimeUnixNano": _ts_to_nanos(call_event.timestamp),
            "endTimeUnixNano": _one_ms_end_nanos(call_event.timestamp),
            "attributes": _make_attributes({
                _SEMCONV_TOOL_NAME: tool_name,
                _SEMCONV_OP: "tool.call",
                _SEMCONV_TOOL_CALL_ID: _tool_call_id(call_event),
                "tool.name": tool_name,
                **{f"tool.input.{k}": str(v)[:200] for k, v in object_field(call_event.data, "arguments", {}).items()},
            }),
            "status": {"code": 0},  # STATUS_CODE_UNSET
        }, call_event))
    retry_links.apply()

    # Build root span
    root_span: dict = {
        "traceId": trace_id,
        "spanId": root_span_id,
        "name": f"agent-session ({usable_text(meta.agent_name) or 'agent'})",
        "kind": 1,
        "startTimeUnixNano": root_start,
        "endTimeUnixNano": root_end,
        "attributes": root_attrs,
        "events": root_events,
        "status": root_status,
    }
    # Link to parent span for subagent hierarchy
    if parent_span_id:
        root_span["parentSpanId"] = parent_span_id

    tenant_id = validate_optional_tenant_id(meta.tenant_id)
    if tenant_id:
        tenant_attributes = _make_attributes({"tenant_id": tenant_id})
        for span in [root_span] + child_spans:
            span.setdefault("attributes", []).extend(tenant_attributes)

    all_spans = [root_span] + child_spans

    return {
        "resourceSpans": [{
            "resource": {
                "attributes": _make_attributes({
                    "service.name": service_name,
                    "service.version": __version__,
                    "agent.session_id": meta.session_id,
                    **({"tenant_id": tenant_id} if tenant_id else {}),
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


def node_tree_to_otlp(
    tree: SessionNode,
    service_name: str = "agent-trace",
    *,
    genai: bool = False,
    evaluations: Mapping[str, Sequence[StoredEvaluation]] | None = None,
    skills_and_compaction: bool = False,
) -> dict:
    """Export an already built subagent tree as a single OTLP trace.

    Every session is converted with :func:`session_to_otlp`, or with
    :func:`session_to_otlp_genai` when ``genai`` is true, under the root
    session's trace ID. A subagent's root span is parented to the span of
    the event its ``parent_event_id`` names in the parent session, which the
    hooks set to the spawning tool call. A ``parent_event_id`` that is not a
    non-empty string, or that names no span in the parent's export, parents
    the subagent to the parent session's root span instead. A session that
    appears twice in the tree, which only a parent link cycle produces, is
    exported once. The resource and scope are the root session's.

    With ``genai``, ``evaluations`` maps a session ID to the stored eval
    results that become ``gen_ai.evaluation.result`` events on that
    session's root span, and ``skills_and_compaction`` is passed to every
    session's conversion (see :func:`session_to_otlp_genai`). The classic
    mapping exports neither.
    """
    from .subagent import parent_event_id_of

    convert = session_to_otlp_genai if genai else session_to_otlp
    root_trace_id = _to_trace_id(tree.meta.session_id)
    all_spans: list[dict] = []
    exported: set[str] = set()
    root_payload: dict = {}

    def _collect(node: SessionNode, parent_span_id: str = "") -> None:
        nonlocal root_payload
        meta = node.meta
        if meta.session_id in exported:
            return
        exported.add(meta.session_id)
        extra: dict[str, Any] = {}
        if genai and evaluations:
            extra["evaluations"] = evaluations.get(meta.session_id, ())
        if genai and skills_and_compaction:
            extra["skills_and_compaction"] = True
        payload = convert(
            meta, node.events, service_name,
            parent_span_id=parent_span_id,
            parent_trace_id=root_trace_id,
            **extra,
        )
        if not root_payload:
            root_payload = payload
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        all_spans.extend(spans)
        span_ids = {span["spanId"] for span in spans}
        session_root_span_id = _to_span_id(f"root-{meta.session_id}")

        for child in node.children:
            parent_event_id = parent_event_id_of(child.meta)
            spawning_span_id = session_root_span_id
            if parent_event_id:
                candidate = _to_span_id(parent_event_id)
                if candidate in span_ids:
                    spawning_span_id = candidate
            _collect(child, parent_span_id=spawning_span_id)

    _collect(tree)

    resource_spans = root_payload["resourceSpans"][0]
    return {
        "resourceSpans": [{
            "resource": resource_spans["resource"],
            "scopeSpans": [{
                "scope": resource_spans["scopeSpans"][0]["scope"],
                "spans": all_spans,
            }],
        }],
    }


def tree_to_otlp(
    store: TraceStore,
    root_session_id: str,
    service_name: str = "agent-trace",
) -> dict:
    """Export a full subagent tree as a single classic OTLP trace.

    See :func:`node_tree_to_otlp` for how subagent root spans are parented.
    """
    from .subagent import build_tree

    return node_tree_to_otlp(build_tree(store, root_session_id), service_name)


def tree_to_otlp_genai(
    store: TraceStore,
    root_session_id: str,
    service_name: str = "agent-trace",
) -> dict:
    """Export a full subagent tree as a single OTel GenAI OTLP trace.

    See :func:`node_tree_to_otlp` for how subagent root spans are parented.
    """
    from .subagent import build_tree

    return node_tree_to_otlp(
        build_tree(store, root_session_id), service_name, genai=True,
    )


def _genai_llm_span(
    meta: SessionMeta,
    trace_id: str,
    root_span_id: str,
    req_event: TraceEvent | None,
    event: TraceEvent,
    span_id: str,
    *,
    inferred_start: float | None = None,
    compacted: bool = False,
) -> dict:
    """Build the ``chat`` model request span for one model request.

    ``event`` ends the request: an ``llm_response``, or an ``error`` linked
    to ``req_event``. A failed or cancelled request (see :func:`llm_failure`)
    gets ERROR status and ``error.type``; a completed one keeps OK status.
    ``span_id`` is the span ID of ``req_event``, or of ``event`` without one.

    ``inferred_start`` is set when ``event`` is a model response read from a
    transcript, an ``assistant_response`` with recorded usage and no request
    (see :func:`_imported_response_starts`). The span starts there and may
    have zero length. It carries the recorded response model and ID and
    ``agent.strace.timing=inferred``.

    ``compacted`` sets ``gen_ai.conversation.compacted`` to true (see
    :func:`_compacted_context_events`); it is never set to false.
    """
    span_start = req_event.timestamp if req_event else event.timestamp
    if inferred_start is not None:
        span_start = inferred_start
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
        _SEMCONV_PROVIDER: system,
        _SEMCONV_SYSTEM: system,
        _SEMCONV_OP: _OP_CHAT,
        _OI_SPAN_KIND: "LLM",
        _OI_SESSION_ID: meta.session_id,
        **_conversation_attributes(meta),
        _OI_LLM_PROVIDER: system if system != "unknown" else raw_llm_provider,
        "agent.strace.provider": raw_llm_provider,
    }
    if compacted:
        llm_attrs[_SEMCONV_CONVERSATION_COMPACTED] = True
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
    llm_attrs.update(_response_server_attributes(req_data, event.data))

    usage, has_input, has_output = model_request_usage(req_data, event.data)
    llm_attrs.update(_token_attributes(usage, has_input, has_output))
    # model_request_usage reads the output from the response when it has one.
    output_data = event.data if usage_detail(event.data)[2] else req_data
    llm_attrs.update(_reasoning_attributes(output_data, usage, has_output))
    finish = event.data.get("stop_reason") or event.data.get("finish_reason", "")
    if finish:
        reasons = _finish_reasons(finish)
        llm_attrs[_SEMCONV_FINISH_REASON] = reasons
        llm_attrs["agent.strace.finish_reason"] = finish
        if reasons and reasons[0]:
            # OpenInference types llm.finish_reason as one string. Its
            # instrumentations and GenAI converter take the first choice's.
            llm_attrs[_OI_FINISH_REASON] = reasons[0]
    failure = llm_failure(event)
    if failure is not None:
        if not finish:
            # The GenAI conventions report "error" when a failed or
            # cancelled generation returned no finish reason.
            llm_attrs[_SEMCONV_FINISH_REASON] = _finish_reasons(_FINISH_REASON_ERROR)
        llm_attrs[_SEMCONV_ERROR_TYPE] = failure.error_type
    llm_attrs.update(_cost_attributes(req_data, event.data))
    if req_event:
        llm_attrs.update(_event_trace_attributes(req_event, "request"))
    llm_attrs.update(_event_trace_attributes(event, "response"))
    if inferred_start is not None:
        llm_attrs.update(_imported_response_attributes(event.data))
        span_end_nanos = _ts_to_nanos(span_end)
    else:
        span_end_nanos = _llm_span_end_nanos(span_start, span_end)
    llm_attrs["metadata"] = _openinference_metadata(llm_attrs)

    span: dict = {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": root_span_id,
        "name": _model_span_name(model),
        "kind": 3,  # SPAN_KIND_CLIENT
        "startTimeUnixNano": _ts_to_nanos(span_start),
        "endTimeUnixNano": span_end_nanos,
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
        span["status"] = {"code": 0}  # STATUS_CODE_UNSET
    else:
        span["status"] = _failure_status(failure)
    return span


def session_to_otlp_genai(
    meta: SessionMeta,
    events: list[TraceEvent],
    service_name: str = "agent-trace",
    parent_span_id: str = "",
    parent_trace_id: str = "",
    *,
    root_outcome: bool = True,
    evaluations: Sequence[StoredEvaluation] = (),
    skills_and_compaction: bool = False,
) -> dict:
    """Convert a session to OTLP JSON using the OTel GenAI semantic conventions.

    Follows the semantic-conventions-genai release v0.1.0 (2026-11-21).
    Differences from session_to_otlp():
    - LLM request/response pairs become ``chat {model}`` CLIENT child spans
      (not just events on the root span). A failed or cancelled model
      request, or an error linked to an llm_request, gets ERROR status and
      error.type
    - A model response that ``agent-strace import`` read from a transcript,
      with its recorded model and token usage, becomes a ``chat {model}``
      span too, with an inferred start (see _imported_response_starts)
    - Tool calls become ``execute_tool {tool}`` INTERNAL spans with
      gen_ai.tool.name and gen_ai.tool.call.id, which is the model's ID for
      the call when the trace stored one (``tool_use_id``), else the call's
      event ID; a failed one, an error or a tool_result marked
      ``is_error: true``, gets error.type
    - The root is an ``invoke_agent`` INTERNAL span with gen_ai.agent.name
      when the session names its agent. Its status follows the session's
      end state, and an ERROR root carries error.type (see
      ``_root_outcome``); ``root_outcome=False`` leaves it UNSET for a
      partial batch of events
    - Root, LLM and tool spans carry gen_ai.conversation.id when the agent
      supplied its own session ID
    - Error events use the OTel exception event format
    - gen_ai.provider.name is derived from the provider or model name when
      possible; the deprecated gen_ai.system repeats it
    - The root span carries the recorded prompts and replies as
      gen_ai.input.messages and gen_ai.output.messages, and tool spans carry
      gen_ai.tool.call.arguments and gen_ai.tool.call.result, all as JSON
      strings under the same length limits as the older content keys; a
      tool argument or result that is JSON stays valid JSON when it is cut
    - A chat span carries gen_ai.response.id, gen_ai.response.model,
      server.address and server.port only when the capture path recorded
      them
    - A tool or chat span whose event names another by ``retry_of`` links
      to that event's span
    - Each of ``evaluations``, the results `eval run` or `eval ci` stored in
      the session's eval.json (see eval/results.py), becomes a
      ``gen_ai.evaluation.result`` event on the root span, at the root span's
      end, after the other root events. With none, the payload is the same
      as before evaluations were exported
    - With ``skills_and_compaction``, which ``agent-strace export --format
      otlp-genai`` sets, a dedicated skill tool call that names its skill
      (see :func:`agent_trace.skills.skill_call_name`) gets
      gen_ai.skill.name and the span name ``execute_tool {tool} {skill}``,
      the load skill refinement, whether or not the skill loaded; and a
      chat span whose request used a compacted context (see
      :func:`_compacted_context_events`) gets
      ``gen_ai.conversation.compacted=true``. Without it neither is emitted,
      so Temporal and ``watch --stream-otlp`` spans stay as they were

    The tenant_id works as in session_to_otlp().

    See: https://github.com/open-telemetry/semantic-conventions-genai
    """
    trace_id = parent_trace_id if parent_trace_id else _to_trace_id(meta.session_id)
    root_span_id = _to_span_id(f"root-{meta.session_id}")
    span_id_of = _event_span_ids(meta.session_id, events)
    root_start = _ts_to_nanos(meta.started_at)
    root_end = _root_end_nanos(meta)

    raw_provider, gen_ai_system = _provider_value(meta)
    agent_name = usable_text(meta.agent_name)
    root_attr_values: dict[str, Any] = {
        # GenAI agent attributes
        _SEMCONV_OP: _OP_INVOKE_AGENT,
        **({_SEMCONV_AGENT_NAME: agent_name} if agent_name else {}),
        _SEMCONV_PROVIDER: gen_ai_system,
        _SEMCONV_SYSTEM: gen_ai_system,
        # OpenInference compatibility attributes
        _OI_SPAN_KIND: "AGENT",
        _OI_SESSION_ID: meta.session_id,
        **_conversation_attributes(meta),
        # Standard resource attributes
        "agent.session_id": meta.session_id,
        "agent.tool_calls": usable_count(meta.tool_calls),
        "agent.llm_requests": usable_count(meta.llm_requests),
        "agent.errors": usable_count(meta.errors),
        # agent-strace preservation attributes
        "agent.strace.provider": raw_provider,
        "agent.strace.redacted": (
            meta.redacted is True or any(is_redacted(event) for event in events)
        ),
        "agent.strace.redacted_event_count": sum(is_redacted(event) for event in events),
        **_evidence_attributes(meta, events),
    }
    provenance = usable_object(meta.attribution)
    if provenance:
        root_attr_values["agent.strace.provenance"] = provenance
    if meta.parent_session_id:
        root_attr_values["agent.strace.parent_session_id"] = meta.parent_session_id
    from .subagent import parent_event_id_of

    parent_event_id = parent_event_id_of(meta)
    if parent_event_id:
        root_attr_values["agent.strace.parent_event_id"] = parent_event_id
    depth = usable_count(meta.depth)
    if depth:
        root_attr_values["agent.strace.depth"] = depth
    root_attr_values.update(_root_messages(events))
    outcome_attributes, root_status = (
        _root_outcome(events) if root_outcome else ({}, {"code": 0})
    )
    root_attr_values.update(outcome_attributes)
    root_attr_values["metadata"] = _openinference_metadata(root_attr_values)
    root_attrs = _make_attributes(root_attr_values)

    root_events: list[dict] = []
    child_spans: list[dict] = []
    retry_links = _RetryLinks()

    # Pair LLM requests and tool calls with their responses (see pairing.py)
    requests: Pairer[TraceEvent] = llm_pairer()
    calls: Pairer[TraceEvent] = tool_pairer()
    # Model responses read from a transcript, which recorded no request
    response_starts = _imported_response_starts(events)
    compacted_events = (
        _compacted_context_events(events) if skills_and_compaction else set()
    )

    def skill_name(call_event: TraceEvent | None) -> str:
        if not skills_and_compaction or call_event is None:
            return ""
        from .skills import skill_call_name

        return skill_call_name(call_event)

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
            inferred_start = response_starts.get(id(event))
            if inferred_start is not None:
                child_spans.append(retry_links.add(_genai_llm_span(
                    meta, trace_id, root_span_id, None, event, span_id_of(event),
                    inferred_start=inferred_start,
                    compacted=id(event) in compacted_events,
                ), event))

        elif event.event_type == EventType.LLM_REQUEST:
            requests.add(event)

        elif event.event_type == EventType.LLM_RESPONSE:
            req_event = requests.match(event)
            child_spans.append(retry_links.add(_genai_llm_span(
                meta, trace_id, root_span_id, req_event, event,
                span_id_of(req_event if req_event else event),
                compacted=id(req_event if req_event else event) in compacted_events,
            ), req_event, event))

        elif event.event_type == EventType.TOOL_CALL:
            calls.add(event)

        elif event.event_type == EventType.TOOL_RESULT and not _failed_tool_result(event):
            call_event = calls.match(event)

            tool_name = event.data.get("tool_name", "") or (
                call_event.data.get("tool_name", "tool") if call_event else "tool"
            )
            span_start = call_event.timestamp if call_event else event.timestamp
            duration_ns = _duration_to_nanos(event.duration_ms)

            tool_attrs: dict = {
                _SEMCONV_TOOL_NAME: tool_name,
                _SEMCONV_OP: _OP_EXECUTE_TOOL,
                _SEMCONV_TOOL_CALL_ID: _tool_call_id(call_event, event),
                _OI_SPAN_KIND: "TOOL",
                _OI_SESSION_ID: meta.session_id,
                **_conversation_attributes(meta),
                _OI_TOOL_NAME: tool_name,
            }
            skill = skill_name(call_event)
            if skill:
                tool_attrs[_SEMCONV_SKILL_NAME] = skill
            if call_event:
                tool_attrs.update(_tool_input_attributes(call_event.data.get("arguments", {})))
                tool_attrs.update(_event_trace_attributes(call_event, "request"))
            result = event.data.get("result", "")
            if result:
                tool_attrs["gen_ai.tool.output"] = str(result)[:500]
                output_value, output_mime = _io_value(result, _TOOL_RESULT_LIMIT)
                tool_attrs[_OI_OUTPUT_VALUE] = output_value
                tool_attrs[_OI_OUTPUT_MIME_TYPE] = output_mime
                tool_attrs[_SEMCONV_TOOL_CALL_RESULT] = _json_content(
                    result, _TOOL_RESULT_LIMIT,
                )
            tool_attrs.update(_event_trace_attributes(event, "response"))
            tool_attrs["metadata"] = _openinference_metadata(tool_attrs)

            child_spans.append(retry_links.add({
                "traceId": trace_id,
                "spanId": span_id_of(call_event if call_event else event),
                "parentSpanId": root_span_id,
                "name": _tool_span_name(tool_name, skill),
                "kind": 1,  # SPAN_KIND_INTERNAL
                "startTimeUnixNano": _ts_to_nanos(span_start),
                "endTimeUnixNano": _duration_end_nanos(span_start, duration_ns),
                "attributes": _make_attributes(tool_attrs),
                "status": {"code": 0},  # STATUS_CODE_UNSET
            }, call_event, event))

        elif event.event_type in (EventType.ERROR, EventType.TOOL_RESULT):
            # A failed tool_result ends its tool call like an error does.
            req_event = (
                requests.match_parent(event)
                if event.event_type == EventType.ERROR else None
            )
            if req_event is not None:
                # An error linked to an LLM request ends that request, so it
                # exports as one failed model call instead of a tool error.
                child_spans.append(retry_links.add(_genai_llm_span(
                    meta, trace_id, root_span_id, req_event, event,
                    span_id_of(req_event),
                    compacted=id(req_event) in compacted_events,
                ), req_event, event))
                continue

            call_event = calls.match(event)

            tool_name = _error_tool_name(event, call_event)
            error_msg = _tool_error_message(event)
            span_start = call_event.timestamp if call_event else event.timestamp
            duration_ns = _duration_to_nanos(event.duration_ms)

            exc_event = _make_event("exception", event.timestamp, {
                "exception.type": event.data.get("error_type", "Error"),
                "exception.message": str(error_msg)[:500],
                "exception.escaped": False,
                **_event_trace_attributes(event, "response"),
            })

            error_type = _tool_error_type(event)
            error_attrs: dict[str, Any] = {
                _SEMCONV_TOOL_NAME: tool_name,
                _SEMCONV_OP: _OP_EXECUTE_TOOL,
                _SEMCONV_TOOL_CALL_ID: _tool_call_id(call_event, event),
                _SEMCONV_ERROR_TYPE: error_type,
                _OI_SPAN_KIND: "TOOL",
                _OI_SESSION_ID: meta.session_id,
                **_conversation_attributes(meta),
                _OI_TOOL_NAME: tool_name,
            }
            skill = skill_name(call_event)
            if skill:
                error_attrs[_SEMCONV_SKILL_NAME] = skill
            if call_event and isinstance(call_event.data, Mapping):
                error_attrs.update(_tool_input_attributes(call_event.data.get("arguments", {})))
                error_attrs.update(_event_trace_attributes(call_event, "request"))
            error_attrs.update(_event_trace_attributes(event, "response"))
            error_attrs["metadata"] = _openinference_metadata(error_attrs)

            child_spans.append(retry_links.add({
                "traceId": trace_id,
                "spanId": span_id_of(call_event if call_event else event),
                "parentSpanId": root_span_id,
                "name": _tool_span_name(tool_name, skill),
                "kind": 1,  # SPAN_KIND_INTERNAL
                "startTimeUnixNano": _ts_to_nanos(span_start),
                "endTimeUnixNano": _duration_end_nanos(span_start, duration_ns),
                "attributes": _make_attributes(error_attrs),
                "events": [exc_event],
                "status": {"code": 2, "message": str(error_msg)[:200]},
            }, call_event, event))

    # Emit unmatched LLM requests as minimal spans
    for req_event in requests.unmatched():
        model = req_event.data.get("model", "")
        raw_llm_provider, system = _provider_value(
            meta,
            str(model),
            req_event.data,
        )
        llm_attrs: dict[str, Any] = {
            _SEMCONV_PROVIDER: system,
            _SEMCONV_SYSTEM: system,
            _SEMCONV_OP: _OP_CHAT,
            _OI_SPAN_KIND: "LLM",
            _OI_SESSION_ID: meta.session_id,
            **_conversation_attributes(meta),
            _OI_LLM_PROVIDER: system if system != "unknown" else raw_llm_provider,
            "agent.strace.provider": raw_llm_provider,
            **_event_trace_attributes(req_event, "request"),
        }
        if id(req_event) in compacted_events:
            llm_attrs[_SEMCONV_CONVERSATION_COMPACTED] = True
        if model:
            llm_attrs[_SEMCONV_MODEL] = model
            llm_attrs[_OI_LLM_MODEL] = model
        llm_attrs.update(_response_server_attributes(req_event.data, {}))
        usage, has_input, has_output = usage_detail(req_event.data)
        llm_attrs.update(_token_attributes(usage, has_input, has_output))
        llm_attrs.update(_reasoning_attributes(req_event.data, usage, has_output))
        llm_attrs.update(_cost_attributes(req_event.data, {}))
        llm_attrs["metadata"] = _openinference_metadata(llm_attrs)
        child_spans.append(retry_links.add({
            "traceId": trace_id,
            "spanId": span_id_of(req_event),
            "parentSpanId": root_span_id,
            "name": _model_span_name(model),
            "kind": 3,
            "startTimeUnixNano": _ts_to_nanos(req_event.timestamp),
            "endTimeUnixNano": _one_ms_end_nanos(req_event.timestamp),
            "attributes": _make_attributes(llm_attrs),
            "status": {"code": 0},
        }, req_event))

    # Emit unmatched tool calls
    for call_event in calls.unmatched():
        tool_name = call_event.data.get("tool_name", "tool")
        tool_attrs: dict[str, Any] = {
            _SEMCONV_TOOL_NAME: tool_name,
            _SEMCONV_OP: _OP_EXECUTE_TOOL,
            _SEMCONV_TOOL_CALL_ID: _tool_call_id(call_event),
            _OI_SPAN_KIND: "TOOL",
            _OI_SESSION_ID: meta.session_id,
            **_conversation_attributes(meta),
            _OI_TOOL_NAME: tool_name,
            **_event_trace_attributes(call_event, "request"),
        }
        skill = skill_name(call_event)
        if skill:
            tool_attrs[_SEMCONV_SKILL_NAME] = skill
        tool_attrs.update(_tool_input_attributes(call_event.data.get("arguments", {})))
        tool_attrs["metadata"] = _openinference_metadata(tool_attrs)
        child_spans.append(retry_links.add({
            "traceId": trace_id,
            "spanId": span_id_of(call_event),
            "parentSpanId": root_span_id,
            "name": _tool_span_name(tool_name, skill),
            "kind": 1,  # SPAN_KIND_INTERNAL
            "startTimeUnixNano": _ts_to_nanos(call_event.timestamp),
            "endTimeUnixNano": _one_ms_end_nanos(call_event.timestamp),
            "attributes": _make_attributes(tool_attrs),
            "status": {"code": 0},
        }, call_event))
    retry_links.apply()

    for evaluation in evaluations:
        root_events.append(_evaluation_event(evaluation, root_end))

    root_span: dict = {
        "traceId": trace_id,
        "spanId": root_span_id,
        "name": (
            f"{_OP_INVOKE_AGENT} {agent_name}"
            if agent_name
            else _OP_INVOKE_AGENT
        ),
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": root_start,
        "endTimeUnixNano": root_end,
        "attributes": root_attrs,
        "events": root_events,
        "status": root_status,
    }
    if parent_span_id:
        root_span["parentSpanId"] = parent_span_id

    tenant_id = validate_optional_tenant_id(meta.tenant_id)
    if tenant_id:
        tenant_attributes = _make_attributes({"tenant_id": tenant_id})
        for span in [root_span] + child_spans:
            span.setdefault("attributes", []).extend(tenant_attributes)

    return {
        "resourceSpans": [{
            "resource": {
                "attributes": _make_attributes({
                    "service.name": service_name,
                    "service.version": __version__,
                    "agent.session_id": meta.session_id,
                    **({"tenant_id": tenant_id} if tenant_id else {}),
                }),
            },
            "scopeSpans": [{
                "scope": {"name": "agent-trace", "version": __version__},
                "spans": [root_span] + child_spans,
            }],
        }],
    }


# Read when --agent-id is not given.
AGENT_ID_ENV = "AGENT_STRACE_AGENT_ID"


def resolve_agent_id(flag: str | None, environ: Mapping[str, str]) -> str:
    """Return the ``gen_ai.agent.id`` to send, or "" for none.

    --agent-id (*flag*) wins, then AGENT_STRACE_AGENT_ID. A flag that is
    empty or only spaces raises ValueError with the line to print. A
    variable that is empty or only spaces counts as unset, as an empty
    OpenTelemetry variable does. The value is sent as given.
    """
    if flag is not None:
        if not flag.strip():
            raise ValueError("invalid --agent-id value: it is empty")
        return flag
    value = environ.get(AGENT_ID_ENV, "")
    return value if value.strip() else ""


def set_agent_id(payload: dict, agent_id: str) -> dict:
    """Put ``gen_ai.agent.id`` = *agent_id* on every span of *payload*.

    The ID names an agent registered elsewhere, such as a Foundry external
    agent, which finds its spans by it. agent-strace never derives one, so
    an empty *agent_id* changes nothing. A span's earlier
    ``gen_ai.agent.id`` is replaced. Returns *payload*.
    """
    if not agent_id:
        return payload
    for resource_spans in payload.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                attributes = [
                    attribute for attribute in span.get("attributes", [])
                    if attribute.get("key") != _SEMCONV_AGENT_ID
                ]
                attributes.extend(_make_attributes({_SEMCONV_AGENT_ID: agent_id}))
                span["attributes"] = attributes
    return payload


def export_otlp(
    store: TraceStore,
    session_id: str,
    endpoint: str,
    headers: dict[str, str] | None = None,
    service_name: str = "agent-trace",
    protocol: str = PROTOCOL_JSON,
    agent_id: str = "",
) -> bool:
    """Export a session to an OTLP/HTTP endpoint.

    Args:
        store: TraceStore to load session from
        session_id: Session to export
        endpoint: OTLP collector URL (e.g. http://localhost:4318)
        headers: Extra HTTP headers (for auth tokens, API keys)
        service_name: OTel service name
        protocol: ``http/json`` or ``http/protobuf``
        agent_id: ``gen_ai.agent.id`` for every span; empty sends none

    Returns:
        True if export succeeded. A ValueError from the conversion, other
        than NonObjectEventDataError and UnusableTimestampError, prints
        one line and returns False.
    """
    meta = store.load_meta(session_id)
    if not meta:
        sys.stderr.write(f"Session {session_id} not found\n")
        return False

    events = store.load_events(session_id)
    if not events:
        sys.stderr.write(f"No events for session {session_id}\n")
        return False

    try:
        payload = session_to_otlp(meta, events, service_name)
    except (NonObjectEventDataError, UnusableTimestampError):
        raise
    except ValueError as exc:
        sys.stderr.write(f"OTLP export failed: {exc}\n")
        return False
    set_agent_id(payload, agent_id)
    return export_otlp_payload(
        payload,
        endpoint,
        headers=headers,
        event_count=len(events),
        protocol=protocol,
    )


# Seconds an export POST may wait to connect and for each read.
_EXPORT_TIMEOUT = 30
# Characters of a failed export's reply body that are shown.
_REPLY_EXCERPT_CHARS = 200


def _reply_excerpt(exc: urllib.error.HTTPError) -> str:
    """Return the start of an error reply body, safe to print on one line.

    At most _REPLY_EXCERPT_CHARS characters, with each line break, ESC or
    other character str.isprintable() rejects replaced by a space, so the
    rest of an escape code, such as [31m, still shows.
    """
    try:
        # UTF-8 needs at most 4 bytes for a character.
        raw = exc.read(_REPLY_EXCERPT_CHARS * 4)
        text = raw.decode("utf-8", errors="replace")[:_REPLY_EXCERPT_CHARS]
    except Exception:
        text = ""
    try:
        exc.close()
    except Exception:
        pass
    return "".join(ch if ch.isprintable() else " " for ch in text).strip()


def export_otlp_payload(
    payload: dict,
    endpoint: str,
    headers: dict[str, str] | None = None,
    event_count: int = 0,
    protocol: str = PROTOCOL_JSON,
) -> bool:
    """POST a prepared ExportTraceServiceRequest via OTLP/HTTP.

    Sends the payload as JSON, or as protobuf when *protocol* is
    ``http/protobuf`` (see :mod:`agent_trace.otlp_proto`).

    Sends to otlp_traces_url(endpoint), the URL watch --stream-to
    --stream-otlp uses, and never follows a redirect. Any 2xx answer is
    success; anything else, including a 3xx, a network error and a timeout,
    prints a failure line and returns False without raising. A 415 answer
    adds a line naming the other --protocol. Lines name only the endpoint's
    scheme://host[:port], because the rest of the URL can carry a token.
    """
    url = otlp_traces_url(endpoint)
    shown = url_endpoint(url)
    target = f" to {shown}" if shown else ""
    req_headers = {"Content-Type": CONTENT_TYPES[protocol]}
    if headers:
        req_headers.update(headers)
    try:
        body = encode_body(payload, protocol)
        request = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
        with no_redirect_opener().open(request, timeout=_EXPORT_TIMEOUT) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        excerpt = _reply_excerpt(exc)
        sys.stderr.write(f"OTLP export{target} failed: {post_error_detail(exc)}\n")
        if excerpt:
            sys.stderr.write(f"  {excerpt}\n")
        if exc.code == 415:
            sys.stderr.write(f"  {unsupported_media_hint(protocol)}\n")
        return False
    except EncodeError as exc:
        sys.stderr.write(f"OTLP export{target} failed: {exc}\n")
        return False
    except Exception as exc:
        sys.stderr.write(f"OTLP export{target} failed: {post_error_detail(exc)}\n")
        return False
    if not 200 <= status < 300:
        # The opener raises HTTPError for every other status; this keeps a
        # non-2xx answer from ever being reported as exported.
        sys.stderr.write(f"OTLP export{target} failed: HTTP {status}\n")
        return False
    sys.stderr.write(f"Exported {event_count} events{target} (HTTP {status})\n")
    return True
