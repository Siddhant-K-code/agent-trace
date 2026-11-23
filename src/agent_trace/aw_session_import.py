"""Import a gh-aw Unified Agent Session file (``aw_session.jsonl``).

gh-aw's conclusion job merges each agentic workflow run's agent, firewall,
MCP and workflow logs into one ``aw_session.jsonl`` file (Unified Agent
Session specification, file-format version 1). Its first record is a
``session.format`` header. ``agent-strace import`` reads such a file with
:func:`import_aw_session`; ``jsonl_import`` sends it here when the first
record is that header.

What is kept:

- ``tool.execution_start`` and ``tool.execution_complete`` become
  ``tool_call`` and ``tool_result`` events, paired by ``toolCallId`` within
  one source file and caller scope. A completion with ``success: false`` (or
  another recorded failure) becomes an ``error`` event, as the hooks record a
  failed tool. A completion that records no outcome stays a ``tool_result``
  with no ``success`` key.
- ``firewall.token_usage`` (and its ``usage.report`` alias) become
  ``assistant_response`` events with the recorded token usage, deduplicated
  by ``requestId``, so ``cost`` and ``export`` use the proxy's counts.
- ``session.result`` snapshots of the main agent become the ``session_end``
  status, turn count and reported cost. Later snapshots replace earlier
  fields; they are never summed.
- ``user.message``, ``assistant.message``, ``assistant.refusal``,
  ``session.error``, ``subagent.started``, ``subagent.completed`` and
  ``subagent.failed`` become the matching agent-strace events.

Every event keeps its record's ``provenance`` (component, phase, path,
index). A record with no usable time gets the specification's ordering
anchor as its time and ``timing: inferred``, or ``0.0`` and
``timing: untimed`` when nothing in the file has a time. Record types that
are not mapped are counted, not stored.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .jsonl_import import (
    _log_lines,
    _redact_text,
    _resolve,
    _too_deep,
    _usable_session_id,
)
from .models import EventType, SessionMeta, TraceEvent, usable_number, usable_time
from .pairing import link_response
from .redact import redact_data
from .store import TraceStore

FORMAT_VERSION = 1
IMPORT_MODE = "aw-session-import"
HEADER_TYPE = "session.format"
TEXT_LIMIT = 2000
# Native provenance strings longer than this are left out, not cut.
_NATIVE_LIMIT = 256
_RESULT_ERROR_LIMIT = 20

TIMING_INFERRED = "inferred"
TIMING_UNTIMED = "untimed"

_TIME_UNITS = {
    "seconds": 1.0,
    "milliseconds": 1e3,
    "microseconds": 1e6,
    "nanoseconds": 1e9,
}
_ISO_TIME = re.compile(
    r"(\d{4}-\d{2}-\d{2})[Tt ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?([Zz]|[+-]\d{2}:?\d{2})"
)
_FAILED_TOOL_STATUSES = frozenset({
    "error", "failed", "failure", "cancelled", "canceled", "timeout", "timed_out",
})
# gh-aw model_costs.cjs: providers whose input count already includes cache
# reads when the record does not say (inputTokensIncludeCache absent).
_CACHE_READS_IN_INPUT = frozenset({
    "", "anthropic", "openai", "azure-openai", "azure_openai", "github-copilot",
})
_USAGE_TYPES = frozenset({"firewall.token_usage", "usage.report"})
# Records read for session metadata only; they are not counted as skipped.
_METADATA_TYPES = frozenset({
    HEADER_TYPE, "session.collection", "workflow.info", "session.init",
    "session.start", "agent.execution",
})
_SUBAGENT_STOP_TYPES = frozenset({"subagent.completed", "subagent.failed"})


class AwSessionFormatError(ValueError):
    """The file is an aw_session file this agent-strace cannot read.

    The header is missing, malformed, or names an unsupported version.
    """


def looks_like_aw_record(record: Mapping[str, Any]) -> bool:
    """Return whether ``record`` has the shape of a unified session event.

    Each event of an aw_session file has a dotted ``type`` and a
    ``provenance`` object naming its source ``component``. Copilot CLI's own
    ``events.jsonl`` uses dotted types too, but has no ``provenance``.
    """
    record_type = record.get("type")
    provenance = record.get("provenance")
    return (
        isinstance(record_type, str)
        and "." in record_type
        and isinstance(provenance, Mapping)
        and isinstance(provenance.get("component"), str)
    )


def check_header(record: Mapping[str, Any]) -> None:
    """Raise :class:`AwSessionFormatError` unless ``record`` is a version 1 header."""
    if record.get("type") != HEADER_TYPE:
        raise AwSessionFormatError(
            "no session.format header; an aw_session.jsonl file starts with one"
        )
    data = record.get("data")
    version = data.get("version") if isinstance(data, Mapping) else None
    if isinstance(version, float) and version.is_integer():
        version = int(version)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AwSessionFormatError(
            "invalid session.format header: data.version must be a whole number of at least 1"
        )
    if version != FORMAT_VERSION:
        shown = str(version) if version < 10**9 else "above 1"
        raise AwSessionFormatError(
            f"unsupported aw_session format version {shown}; "
            f"this agent-strace reads version {FORMAT_VERSION}"
        )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _records(path: Path) -> tuple[list[dict[str, Any]], int, str]:
    """Return the JSON object records of ``path`` in file order.

    Also return how many lines were unusable and a SHA-256 digest of the
    file's lines, which names a session whose records give no ID.
    """
    records: list[dict[str, Any]] = []
    unusable = 0
    digest = hashlib.sha256()
    for line in _log_lines(path):
        digest.update(line.encode("utf-8", "surrogateescape"))
        line = line.strip()
        if not line:
            continue
        try:
            line.encode("utf-8")
            raw = json.loads(line)
        except (UnicodeEncodeError, ValueError, RecursionError):
            unusable += 1
            continue
        if not isinstance(raw, dict) or _too_deep(raw):
            unusable += 1
            continue
        records.append(raw)
    return records, unusable, digest.hexdigest()


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("provenance")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _source_key(prov: Mapping[str, Any]) -> tuple[str, str, str]:
    return (_text(prov.get("component")), _text(prov.get("phase")), _text(prov.get("path")))


def _kept_provenance(prov: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provenance fields an event keeps so evidence can cite its source."""
    kept: dict[str, Any] = {}
    for key in ("component", "phase", "path", "attemptId", "startupAttemptId"):
        value = prov.get(key)
        if isinstance(value, str) and value:
            kept[key] = value
    for key in ("index", "timestampMs"):
        value = prov.get(key)
        if usable_number(value) is not None:
            kept[key] = value
    native = prov.get("native")
    if isinstance(native, bool) or usable_number(native) is not None or (
        isinstance(native, str) and 0 < len(native) <= _NATIVE_LIMIT
    ):
        kept["native"] = native
    return kept


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def _iso_time(value: str) -> float | None:
    """Parse an ISO 8601 time that carries a timezone; a naive time is unusable."""
    match = _ISO_TIME.fullmatch(value.strip())
    if match is None:
        return None
    day, clock, fraction, zone = match.groups()
    fraction = ((fraction or "") + "000000")[:6]
    if zone in ("Z", "z"):
        zone = "+00:00"
    elif ":" not in zone:
        zone = f"{zone[:3]}:{zone[3:]}"
    try:
        parsed = datetime.fromisoformat(f"{day}T{clock}.{fraction}{zone}")
    except ValueError:
        return None
    return usable_time(parsed.timestamp())


def _decimal_digits(value: object) -> bool:
    """Return True for a string of ASCII digits only (``str.isdigit`` also takes "²")."""
    return isinstance(value, str) and value.isascii() and value.isdecimal()


def _numeric_time(value: object, unit: str) -> float | None:
    scale = _TIME_UNITS.get(unit)
    if scale is None:
        return None
    if _decimal_digits(value) and unit == "nanoseconds":
        # OTLP writes Unix nanoseconds as a decimal string.
        try:
            return usable_time(int(value) / scale)  # type: ignore[arg-type]
        except (ValueError, OverflowError):
            return None
    if usable_number(value) is None:
        return None
    return usable_time(value / scale)  # type: ignore[operator]


def _source_units(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    units: dict[tuple[str, str, str], str] = {}
    for record in records:
        if record.get("type") != "session.collection":
            continue
        data = record.get("data")
        sources = data.get("sources") if isinstance(data, Mapping) else None
        for source in sources if isinstance(sources, list) else ():
            if not isinstance(source, Mapping):
                continue
            unit = source.get("timestampUnit")
            if isinstance(unit, str) and unit in _TIME_UNITS:
                units.setdefault(_source_key(source), unit)
    return units


def _observed_time(
    record: Mapping[str, Any],
    prov: Mapping[str, Any],
    units: Mapping[tuple[str, str, str], str],
) -> float | None:
    """Return the record's observed Unix time in seconds, or None.

    The native ``timestamp`` comes first: an ISO time with a timezone, or a
    number in the unit its source declares in ``session.collection`` (native
    agent timestamps are milliseconds). Then ``provenance.timestampMs``, the
    merger's millisecond projection. Units are never guessed from magnitude.
    """
    native = record.get("timestamp")
    if isinstance(native, str) and not _decimal_digits(native):
        observed = _iso_time(native)
        if observed is not None:
            return observed
    elif native is not None:
        unit = units.get(_source_key(prov), "")
        if not unit and prov.get("component") == "agent":
            unit = "milliseconds"
        observed = _numeric_time(native, unit)
        if observed is not None:
            return observed
    stamp = prov.get("timestampMs")
    if usable_number(stamp) is not None:
        return usable_time(stamp / 1000)
    return None


def _event_times(
    records: list[dict[str, Any]],
    units: Mapping[tuple[str, str, str], str],
) -> list[tuple[float, str]]:
    """Return each record's time and timing mark ("" when observed).

    An untimed record gets the merge-only anchor of T-UAS-056: before its
    source's first valid time, that time; after it, the latest time its
    source has shown so far. A source with no time at all follows the latest
    time seen so far in the file. Such times are marked ``inferred``; a record
    with nothing to anchor to gets 0.0 and ``untimed``.
    """
    observed = [
        _observed_time(record, _provenance(record), units) for record in records
    ]
    first_by_source: dict[tuple[str, str, str], float] = {}
    for record, when in zip(records, observed):
        if when is not None:
            first_by_source.setdefault(_source_key(_provenance(record)), when)
    first_in_file = next((when for when in observed if when is not None), None)

    times: list[tuple[float, str]] = []
    latest_by_source: dict[tuple[str, str, str], float] = {}
    latest_in_file: float | None = None
    for record, when in zip(records, observed):
        source = _source_key(_provenance(record))
        if when is not None:
            latest_by_source[source] = max(latest_by_source.get(source, when), when)
            latest_in_file = when if latest_in_file is None else max(latest_in_file, when)
            times.append((when, ""))
            continue
        anchor = latest_by_source.get(source)
        if anchor is None:
            anchor = first_by_source.get(source)
        if anchor is None:
            anchor = latest_in_file if latest_in_file is not None else first_in_file
        if anchor is None:
            times.append((0.0, TIMING_UNTIMED))
        else:
            times.append((anchor, TIMING_INFERRED))
    return times


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

def _compact(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return ""


def _content_value(content: object) -> object:
    """Return the text of message ``content`` (a string, or its text blocks
    joined), or the structured value itself for ``_cut`` to redact."""
    if isinstance(content, list):
        parts = [
            block["text"] for block in content
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ]
        if parts:
            return "\n".join(parts)
    if content is None:
        return ""
    return content


def _cut(value: object, redact: bool) -> tuple[str, bool]:
    """Redact ``value``, then cut its text as the Claude importer does.

    A structured value is redacted before it becomes JSON text, as hooks do:
    once an object is text, its key names no longer mark which values to
    redact. Records nest at most MAX_LINE_DEPTH levels, so the walk is safe.
    """
    hit = False
    if not isinstance(value, str):
        if redact:
            redacted = redact_data(value)
            hit = redacted != value
            value = redacted
        value = _compact(value)
    text, text_hit = _redact_text(value, redact)
    if len(text) > TEXT_LIMIT:
        text = text[:TEXT_LIMIT] + "..."
    return text, hit or text_hit


def _whole_number(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _first_count(data: Mapping[str, Any], *keys: str) -> int | None:
    """Return the first key's count that is an int >= 0; a canonical key wins over its alias."""
    for key in keys:
        if key in data:
            value = data[key]
            if not isinstance(value, bool) and isinstance(value, int) and value >= 0:
                return value
    return None


def normalize_usage(usage: Mapping[str, Any], provider: str) -> tuple[dict[str, int], int] | None:
    """Return ``(usage, whole_input)`` in a shape agent-strace's cost reader knows.

    gh-aw usage carries ``inputTokens``/``outputTokens`` and cache counts
    under camelCase or snake_case names. Whether the input count already
    includes the cache follows gh-aw's ``computeInclusiveInputTokens``:
    ``inputTokensIncludeCache`` when recorded; otherwise providers that bundle
    cache reads into input (Anthropic, OpenAI, Azure OpenAI and GitHub
    Copilot) include reads but not writes, and other providers include
    neither. The result has whole input = input + cache read + cache write,
    with cache reads and writes kept apart, so no token counts twice.
    """
    input_tokens = _first_count(usage, "input_tokens", "inputTokens")
    output_tokens = _first_count(usage, "output_tokens", "outputTokens")
    read = _first_count(
        usage, "cache_read_input_tokens", "cacheReadInputTokens", "cacheReadTokens",
    )
    write = _first_count(
        usage,
        "cache_creation_input_tokens", "cacheCreationInputTokens", "cacheWriteTokens",
    )
    reasoning = _first_count(usage, "reasoning_output_tokens", "reasoningOutputTokens")
    if input_tokens is None and output_tokens is None:
        return None
    includes = usage.get("input_tokens_include_cache", usage.get("inputTokensIncludeCache"))
    shaped: dict[str, int] = {}
    whole_input = 0
    if input_tokens is not None:
        if includes is True:
            shaped["input_tokens"] = input_tokens
            if read is not None:
                shaped["cached_input_tokens"] = read
            if write is not None:
                shaped["cache_write_input_tokens"] = write
            whole_input = input_tokens
        else:
            base = input_tokens
            if includes is not False and _normal_provider(provider) in _CACHE_READS_IN_INPUT:
                base = max(input_tokens - (read or 0), 0)
            shaped["input_tokens"] = base
            if read is not None:
                shaped["cache_read_input_tokens"] = read
            if write is not None:
                shaped["cache_creation_input_tokens"] = write
            whole_input = base + (read or 0) + (write or 0)
    if output_tokens is not None:
        shaped["output_tokens"] = output_tokens
    if reasoning is not None:
        shaped["reasoning_output_tokens"] = reasoning
    return shaped, whole_input


def _normal_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in ("github", "copilot", "github_models"):
        return "github-copilot"
    return normalized


def _native_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    """Return the recorded scalar usage fields, as gh-aw named them."""
    return {
        key: value for key, value in usage.items()
        if isinstance(key, str) and (
            isinstance(value, bool) or usable_number(value) is not None
        )
    }


def _tool_failed(data: Mapping[str, Any]) -> bool | None:
    """Return whether a completion recorded a failure, success, or nothing (None)."""
    success = data.get("success")
    if success is False:
        return True
    if data.get("isError") is True or data.get("is_error") is True:
        return True
    exit_code = _whole_number(data.get("exitCode"))
    if exit_code is not None and exit_code != 0:
        return True
    status = data.get("status")
    if isinstance(status, str) and status.strip().lower() in _FAILED_TOOL_STATUSES:
        return True
    if success is True:
        return False
    return None


def _tool_output(data: Mapping[str, Any], failed: bool) -> object:
    keys = ("error", "output", "result", "structuredContent") if failed else (
        "output", "result", "structuredContent", "error",
    )
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return ""


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class _Builder:
    """Convert the records of one aw_session file into agent-strace events."""

    def __init__(self, store: TraceStore, path: Path) -> None:
        self.store = store
        self.path = path
        self.redact = store.redact
        self.events: list[TraceEvent] = []
        self.redacted: list[TraceEvent] = []
        self.skipped: Counter[str] = Counter()
        self.inferred = 0
        self.untimed = 0
        self.tool_calls = 0
        self.llm_requests = 0
        self.errors = 0
        self.total_tokens = 0
        self.pending_tools: dict[tuple[str, str, str, str], TraceEvent] = {}
        self.request_ids: set[str] = set()
        self.agent_types: dict[str, str] = {}
        self.result: dict[str, Any] = {}
        self.exit_code: int | None = None
        self.execution: dict[str, Any] = {}

    # -- scope -------------------------------------------------------------

    def scope(self, record: Mapping[str, Any], prov: Mapping[str, Any]) -> dict[str, Any]:
        phase = _text(prov.get("phase"))
        stream = "main" if phase in ("", "agent") else phase
        scope: dict[str, Any] = {}
        agent_id = _text(record.get("agentId"))
        if agent_id:
            stream = (
                f"sidechain:{agent_id}" if stream == "main" else f"{stream}:sidechain:{agent_id}"
            )
            scope["is_sidechain"] = True
            scope["source_agent_id"] = agent_id
            scope["agent_id"] = agent_id
            agent_type = self.agent_types.get(agent_id)
            if agent_type:
                scope["agent_type"] = agent_type
        scope["context_stream"] = stream
        return scope

    def event(
        self,
        event_type: EventType,
        when: tuple[float, str],
        data: dict[str, Any],
        prov: Mapping[str, Any],
        record_type: str,
        *,
        hit: bool = False,
    ) -> TraceEvent:
        data["source_type"] = record_type
        kept = _kept_provenance(prov)
        if kept:
            data["provenance"] = kept
        timestamp, timing = when
        if timing:
            data["timing"] = timing
            if timing == TIMING_UNTIMED:
                self.untimed += 1
            else:
                self.inferred += 1
        event = TraceEvent(event_type=event_type, timestamp=timestamp, data=data)
        if hit:
            self.redacted.append(event)
        self.events.append(event)
        return event

    # -- mappings ----------------------------------------------------------

    def tool_start(self, record, data, prov, when, scope) -> None:
        tool_data: dict[str, Any] = {"tool_name": _text(data.get("toolName")) or "unknown"}
        hit = False
        arguments = data.get("input", data.get("parameters"))
        if isinstance(arguments, Mapping):
            # The store redacts structured data when it writes it.
            tool_data["arguments"] = dict(arguments)
        elif arguments is not None:
            tool_data["arguments"], hit = _cut(arguments, self.redact)
        command = data.get("command")
        if isinstance(command, str) and command:
            tool_data["command"], command_hit = _cut(command, self.redact)
            hit = hit or command_hit
        if data.get("inputTruncated") is True:
            tool_data["input_truncated"] = True
        call_id = _text(data.get("toolCallId"))
        if call_id:
            tool_data["tool_use_id"] = call_id
        server = _text(data.get("mcpServerName"))
        if server:
            tool_data["mcp_server_name"] = server
        tool_data.update(scope)
        event = self.event(
            EventType.TOOL_CALL, when, tool_data, prov, record["type"], hit=hit,
        )
        self.tool_calls += 1
        if call_id:
            self.pending_tools[self._tool_key(record, prov, call_id)] = event

    @staticmethod
    def _tool_key(record, prov, call_id: str) -> tuple[str, str, str, str]:
        return (
            _text(prov.get("phase")), _text(prov.get("path")),
            _text(record.get("agentId")), call_id,
        )

    def tool_complete(self, record, data, prov, when, scope) -> None:
        call_id = _text(data.get("toolCallId"))
        call = (
            self.pending_tools.pop(self._tool_key(record, prov, call_id), None)
            if call_id else None
        )
        tool_name = _text(data.get("toolName")) or (
            _text(call.data.get("tool_name")) if call is not None else ""
        ) or "unknown"
        failed = _tool_failed(data)
        text, hit = _cut(_tool_output(data, bool(failed)), self.redact)
        result: dict[str, Any] = {"tool_name": tool_name}
        if failed:
            event_type = EventType.ERROR
            result["error"] = text
            exit_code = _whole_number(data.get("exitCode"))
            if exit_code is not None:
                result["exit_code"] = exit_code
            status = _text(data.get("status"))
            if status:
                result["status"] = status
            self.errors += 1
        else:
            event_type = EventType.TOOL_RESULT
            result["result"] = text
            if failed is False:
                result["success"] = True
        if call_id:
            result["tool_use_id"] = call_id
        if data.get("outputTruncated") is True:
            result["output_truncated"] = True
        result.update(scope)
        event = self.event(event_type, when, result, prov, record["type"], hit=hit)
        if call is not None:
            link_response(event, call.event_id, call.timestamp)
            if (
                event.data.get("timing") or call.data.get("timing")
                or event.duration_ms is None or event.duration_ms < 0
            ):
                event.duration_ms = None
        duration = usable_number(data.get("durationMs"))
        if duration is not None:
            event.duration_ms = duration

    def usage(self, record, data, prov, when, scope, has_firewall: bool) -> None:
        record_type = record["type"]
        request_id = _text(data.get("requestId"))
        if request_id:
            if request_id in self.request_ids:
                self.skipped[f"{record_type} (repeated requestId)"] += 1
                return
        elif record_type == "usage.report" and has_firewall:
            # Without a request ID it cannot be told apart from the firewall
            # record it repeats; overlapping accounting is never summed.
            self.skipped[f"{record_type} (no requestId)"] += 1
            return
        provider = _text(data.get("provider"))
        raw_usage = data.get("usage")
        shaped = normalize_usage(raw_usage, provider) if isinstance(raw_usage, Mapping) else None
        status = _whole_number(data.get("status"))
        if shaped is None and (status is None or status < 400):
            self.skipped[f"{record_type} (no usage)"] += 1
            return
        if request_id:
            self.request_ids.add(request_id)
        response: dict[str, Any] = {}
        model = _text(data.get("model"))
        if model:
            response["model"] = model
        if provider:
            response["provider"] = provider
        if request_id:
            response["message_id"] = request_id
        if shaped is not None:
            usage, whole_input = shaped
            response["usage"] = usage
            response["native_usage"] = _native_usage(raw_usage)  # type: ignore[arg-type]
            self.total_tokens += whole_input + usage.get("output_tokens", 0)
        for key, name in (
            ("purpose", "purpose"), ("path", "api_path"), ("xInitiator", "x_initiator"),
            ("agentName", "caller_agent"), ("callerSource", "caller_source"),
        ):
            value = _text(data.get(key))
            if value:
                response[name] = value
        if status is not None:
            response["http_status"] = status
            if status >= 400:
                response["status"] = "failed"
                response["error_type"] = str(status)
        for key, name in (("aic", "aic"), ("totalAic", "total_aic")):
            if usable_number(data.get(key)) is not None:
                response[name] = data[key]
        response.update(scope)
        event = self.event(EventType.ASSISTANT_RESPONSE, when, response, prov, record_type)
        duration = usable_number(data.get("durationMs"))
        if duration is not None:
            event.duration_ms = duration
        self.llm_requests += 1

    def message(self, event_type, key, record, data, prov, when, scope) -> None:
        text, hit = _cut(_content_value(data.get("content")), self.redact)
        payload: dict[str, Any] = {key: text}
        model = _text(data.get("model"))
        if model and event_type == EventType.ASSISTANT_RESPONSE:
            payload["model"] = model
        payload.update(scope)
        self.event(event_type, when, payload, prov, record["type"], hit=hit)

    def refusal(self, record, data, prov, when, scope) -> None:
        if data.get("partial") is True:
            self.skipped[f"{record['type']} (partial)"] += 1
            return
        text, hit = _cut(_content_value(data.get("content")), self.redact)
        payload: dict[str, Any] = {"text": text}
        reason = _text(data.get("reason"))
        if reason:
            payload["stop_reason"] = reason
        for key, name in (("policyCategory", "policy_category"), ("explanation", "explanation")):
            value = _text(data.get(key))
            if value:
                payload[name], extra = _cut(value, self.redact)
                hit = hit or extra
        payload.update(scope)
        self.event(EventType.ASSISTANT_RESPONSE, when, payload, prov, record["type"], hit=hit)

    def session_error(self, record, data, prov, when, scope) -> None:
        severity = _text(data.get("severity")).lower()
        if severity and severity != "error":
            self.skipped[f"{record['type']} ({severity})"] += 1
            return
        message = data.get("message", data.get("error"))
        text, hit = _cut(_content_value(message), self.redact)
        payload: dict[str, Any] = {"error": text}
        for key in ("errorType", "type", "code", "errorCode"):
            value = data.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and value != "":
                payload["error_type"] = str(value)
                break
        payload.update(scope)
        self.event(EventType.ERROR, when, payload, prov, record["type"], hit=hit)
        self.errors += 1

    def subagent(self, record, data, prov, when, scope) -> None:
        record_type = record["type"]
        agent_id = _text(record.get("agentId")) or _text(data.get("invocationId"))
        payload: dict[str, Any] = {}
        if agent_id:
            payload["agent_id"] = agent_id
        name = _text(data.get("agentName"))
        if name:
            payload["agent_name"] = name
        agent_type = _text(data.get("agentType")) or name
        if agent_type:
            payload["agent_type"] = agent_type
        call_id = _text(data.get("toolCallId")) or _text(data.get("parentToolCallId"))
        if call_id:
            payload["tool_use_id"] = call_id
        model = _text(data.get("resolvedModel")) or _text(data.get("model"))
        if model:
            payload["model"] = model
        if record_type == "subagent.failed":
            payload["outcome"] = _text(data.get("outcome")) or "failed"
        elif _text(data.get("outcome")):
            payload["outcome"] = data["outcome"]
        error_code = data.get("errorCode")
        if isinstance(error_code, (str, int)) and not isinstance(error_code, bool):
            payload["error_code"] = error_code
        for key, name in (
            ("durationMs", "duration_ms"), ("totalTokens", "total_tokens"),
            ("totalToolCalls", "total_tool_calls"),
        ):
            if usable_number(data.get(key)) is not None:
                payload[name] = data[key]
        payload.update(scope)
        event_type = (
            EventType.SUBAGENT_START if record_type == "subagent.started"
            else EventType.SUBAGENT_STOP
        )
        self.event(event_type, when, payload, prov, record_type)

    def session_result(self, record, data) -> None:
        """Keep the latest supplied value of each field: results are snapshots.

        Error entries are distinct observations, so every result's errors are
        kept in order, up to ``_RESULT_ERROR_LIMIT``.
        """
        for key, name in (
            ("status", "status"), ("sourceType", "result_source_type"),
        ):
            value = _text(data.get(key))
            if value:
                self.result[name] = value
        for key, name in (
            ("numTurns", "num_turns"), ("totalCostUsd", "reported_cost_usd"),
            ("durationMs", "duration_ms"),
        ):
            if usable_number(data.get(key)) is not None:
                self.result[name] = data[key]
        usage = data.get("usage")
        if isinstance(usage, Mapping):
            native = _native_usage(usage)
            if native:
                self.result["result_usage"] = native
        errors = data.get("errors")
        if isinstance(errors, list):
            kept: list[str] = self.result.get("errors", [])
            for error in errors:
                if len(kept) >= _RESULT_ERROR_LIMIT:
                    self.result["errors_omitted"] = self.result.get("errors_omitted", 0) + 1
                    continue
                text, hit = _cut(_content_value(error), self.redact)
                if text:
                    kept.append(text)
                if hit:
                    self.result["redacted_errors"] = True
            if kept:
                self.result["errors"] = kept

    def agent_execution(self, data) -> None:
        exit_code = _whole_number(data.get("exitCode"))
        if exit_code is not None:
            self.exit_code = exit_code
        for key, name in (
            ("categories", "error_categories"), ("errorCodes", "error_codes"),
            ("errorTypes", "error_types"),
        ):
            values = data.get(key)
            if isinstance(values, list):
                kept = [
                    value for value in values
                    if isinstance(value, (str, int)) and not isinstance(value, bool)
                ][:20]
                if kept:
                    self.execution[name] = kept


def _main_scope(record: Mapping[str, Any], prov: Mapping[str, Any]) -> bool:
    return not record.get("agentId") and _text(prov.get("phase")) in ("", "agent")


def import_aw_session(
    path: str | Path, store: TraceStore, *, merge: bool = False
) -> tuple[str, bool]:
    """Import an aw_session.jsonl file into ``store``; return ``(session_id, written)``.

    Raises ``FileNotFoundError``, ``OSError`` when the file cannot be read,
    and :class:`AwSessionFormatError` (a ``ValueError``) when its first record
    is not a supported ``session.format`` header. Events are built in memory
    and written together, so a file that fails leaves no partial session.
    ``written`` is False when the session already exists and ``merge`` is
    not set.
    """
    path = _resolve(path)
    records, unusable, digest = _records(path)
    if not records:
        raise AwSessionFormatError(
            "no session.format header; an aw_session.jsonl file starts with one"
        )
    check_header(records[0])
    records = records[1:]
    units = _source_units(records)
    times = _event_times(records, units)

    builder = _Builder(store, path)
    if unusable:
        builder.skipped["(unreadable line)"] += unusable

    # Session identity and metadata come from the main agent's records.
    native_id = ""
    engine = ""
    model = ""
    cwd = ""
    workflow: dict[str, Any] = {}
    for record in records:
        data = record.get("data")
        if not isinstance(data, Mapping):
            continue
        prov = _provenance(record)
        record_type = record.get("type")
        if record_type in ("session.init", "session.start") and _main_scope(record, prov):
            native_id = native_id or _usable_session_id(data.get("sessionId"))
            engine = engine or _text(data.get("sourceEngine"))
            model = model or _text(data.get("model"))
            cwd = cwd or _text(data.get("cwd"))
        elif record_type == "workflow.info" and not workflow:
            workflow = dict(data)
        elif record_type == "subagent.started":
            agent_id = _text(record.get("agentId")) or _text(data.get("invocationId"))
            agent_type = _text(data.get("agentType")) or _text(data.get("agentName"))
            if agent_id and agent_type:
                builder.agent_types.setdefault(agent_id, agent_type)
    engine = engine or _text(workflow.get("engineId"))
    model = model or _text(workflow.get("model"))
    session_id = native_id or f"aw-{digest[:16]}"

    has_firewall = any(record.get("type") == "firewall.token_usage" for record in records)
    for record, when in zip(records, times):
        record_type = record.get("type")
        data = record.get("data")
        if not isinstance(record_type, str) or not record_type:
            builder.skipped["(no type)"] += 1
            continue
        if not isinstance(data, Mapping):
            data = {}
        prov = _provenance(record)
        scope = builder.scope(record, prov)
        if record_type == "tool.execution_start":
            builder.tool_start(record, data, prov, when, scope)
        elif record_type == "tool.execution_complete":
            builder.tool_complete(record, data, prov, when, scope)
        elif record_type in _USAGE_TYPES:
            builder.usage(record, data, prov, when, scope, has_firewall)
        elif record_type == "user.message":
            builder.message(EventType.USER_PROMPT, "prompt", record, data, prov, when, scope)
        elif record_type == "assistant.message":
            builder.message(
                EventType.ASSISTANT_RESPONSE, "text", record, data, prov, when, scope,
            )
        elif record_type == "assistant.refusal":
            builder.refusal(record, data, prov, when, scope)
        elif record_type == "session.error":
            builder.session_error(record, data, prov, when, scope)
        elif record_type == "subagent.started" or record_type in _SUBAGENT_STOP_TYPES:
            builder.subagent(record, data, prov, when, scope)
        elif record_type == "session.result":
            if _main_scope(record, prov):
                builder.session_result(record, data)
            else:
                builder.skipped[f"{record_type} (subagent or other phase)"] += 1
        elif record_type == "agent.execution":
            if _text(prov.get("phase")) in ("", "agent"):
                builder.agent_execution(data)
        elif record_type == HEADER_TYPE:
            # A source-native session.format record never replaces the header.
            builder.skipped[f"{record_type} (after the header)"] += 1
        elif record_type not in _METADATA_TYPES:
            builder.skipped[record_type] += 1

    observed = [when for when, timing in times if not timing]
    started_at = min(observed) if observed else 0.0
    ended_at = max(observed) if observed else 0.0

    meta = SessionMeta(
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at,
        agent_name=engine or "gh-aw",
        command=f"imported from {path.name} (gh-aw aw_session)",
        tool_calls=builder.tool_calls,
        llm_requests=builder.llm_requests,
        errors=builder.errors,
        total_tokens=builder.total_tokens,
        conversation_id=native_id,
    )
    duration = usable_number(builder.result.get("duration_ms"))
    if duration is not None:
        meta.total_duration_ms = duration
    elif ended_at > started_at:
        meta.total_duration_ms = (ended_at - started_at) * 1000

    start_data: dict[str, Any] = {"mode": IMPORT_MODE, "format_version": FORMAT_VERSION}
    if engine:
        # Not "provider": readers take that key on session_start as the hook
        # surface that captured the session, and gh-aw ran no agent-strace hook.
        start_data["engine"] = engine
    if model:
        start_data["model"] = model
    if cwd:
        start_data["cwd"] = cwd
    for key, name in (
        ("workflow", "workflow"), ("repository", "repository"),
        ("triggerType", "trigger_type"), ("cliVersion", "gh_aw_version"),
        ("awfVersion", "awf_version"), ("agentVersion", "agent_version"),
    ):
        value = _text(workflow.get(key))
        if value:
            start_data[name] = value
    run_id = _whole_number(workflow.get("runId"))
    if run_id is not None:
        start_data["run_id"] = run_id

    end_data: dict[str, Any] = {
        "duration_ms": meta.total_duration_ms,
        "tool_calls": meta.tool_calls,
        "llm_requests": meta.llm_requests,
        "total_tokens": meta.total_tokens,
        "source": str(path),
    }
    end_data.update(builder.result)
    if builder.exit_code is not None:
        end_data["exit_code"] = builder.exit_code
    end_data.update(builder.execution)
    if builder.skipped:
        end_data["skipped_records"] = dict(sorted(builder.skipped.items()))
    if builder.inferred:
        end_data["inferred_times"] = builder.inferred
    if builder.untimed:
        end_data["untimed_events"] = builder.untimed

    events = [
        TraceEvent(event_type=EventType.SESSION_START, timestamp=started_at, data=start_data),
        *builder.events,
        TraceEvent(event_type=EventType.SESSION_END, timestamp=ended_at, data=end_data),
    ]
    for event in events:
        event.session_id = session_id

    if merge:
        store.create_session(meta)
    elif store.create_session(meta, if_missing=True) is None:
        return session_id, False
    for event in builder.redacted:
        event.redacted = True
        sys.stderr.write("agent-strace: redacted secrets from trace event\n")
    store.append_events(session_id, events)
    return session_id, True


def import_summary(events: list[TraceEvent]) -> str:
    """Return the extra lines ``agent-strace import`` prints for an aw_session import.

    Empty for any other session.
    """
    if not events or events[0].event_type != EventType.SESSION_START:
        return ""
    if not isinstance(events[0].data, dict) or events[0].data.get("mode") != IMPORT_MODE:
        return ""
    end = next(
        (event for event in reversed(events) if event.event_type == EventType.SESSION_END),
        None,
    )
    data = end.data if end is not None and isinstance(end.data, dict) else {}
    lines = ""
    inferred = data.get("inferred_times")
    if isinstance(inferred, int) and inferred:
        noun = "event has" if inferred == 1 else "events have"
        lines += f"  {inferred} {noun} no recorded time (timing: inferred)\n"
    untimed = data.get("untimed_events")
    if isinstance(untimed, int) and untimed:
        noun = "event has" if untimed == 1 else "events have"
        lines += f"  {untimed} {noun} no time and nothing to order by (timing: untimed)\n"
    skipped = data.get("skipped_records")
    if isinstance(skipped, dict) and skipped:
        counts = sorted(
            ((count, name) for name, count in skipped.items() if isinstance(count, int)),
            key=lambda item: (-item[0], item[1]),
        )
        total = sum(count for count, _ in counts)
        listed = ", ".join(f"{name} {count}" for count, name in counts)
        lines += f"  Skipped {total} records not imported: {listed}\n"
    return lines


__all__ = [
    "AwSessionFormatError",
    "FORMAT_VERSION",
    "check_header",
    "import_aw_session",
    "import_summary",
    "looks_like_aw_record",
    "normalize_usage",
]
