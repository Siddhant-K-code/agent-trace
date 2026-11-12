"""Export eval scores and behavioral metrics to Langfuse and OTLP backends.

Langfuse path:
  - Sessions exported as Langfuse Traces
  - Tool calls exported as Spans (type=tool)
  - LLM requests/responses exported as Generations (with token counts)
  - eval.json judge scores exported as Langfuse Scores attached to the trace

OTLP metrics path:
  - Behavioral metrics (error_rate, retry_rate, cost, blast_radius, eval scores)
    exported as OTLP gauge metrics to any compatible backend

No new dependencies. All HTTP calls use urllib.request.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cost import TokenUsage, infer_provider
from .event_data import number_field, require_object_event_data, str_field
from .models import EventType, SessionMeta, TraceEvent
from .otlp import llm_failure, model_request_usage
from .pairing import (
    LLM_REQUEST_TYPES,
    TOOL_REQUEST_TYPES,
    Pairer,
    event_model,
    llm_pairer,
    tool_pairer,
)
from .store import TraceStore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LangfuseConfig:
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"

    @property
    def auth_header(self) -> str:
        token = base64.b64encode(
            f"{self.public_key}:{self.secret_key}".encode()
        ).decode()
        return f"Basic {token}"

    @property
    def configured(self) -> bool:
        return bool(self.public_key and self.secret_key)


@dataclass
class OtlpMetricsConfig:
    endpoint: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.endpoint)


# ---------------------------------------------------------------------------
# Eval score reading
# ---------------------------------------------------------------------------

@dataclass
class EvalScore:
    judge: str
    score: float
    passed: bool
    threshold: float = 1.0


def _load_eval_scores(store: TraceStore, session_id: str) -> list[EvalScore]:
    eval_path = store.base_dir / session_id / "eval.json"
    if not eval_path.exists():
        return []
    try:
        data = json.loads(eval_path.read_text())
        results = data.get("results") or data.get("judges") or []
        scores = []
        for r in results:
            name = r.get("scorer") or r.get("name") or "unknown"
            score = float(r.get("score", 0.0))
            threshold = float(r.get("threshold", 1.0))
            passed = bool(r.get("passed", score >= threshold))
            scores.append(EvalScore(judge=name, score=score, passed=passed, threshold=threshold))
        return scores
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Langfuse export
# ---------------------------------------------------------------------------

def _lf_post(config: LangfuseConfig, path: str, body: dict) -> bool:
    url = config.host.rstrip("/") + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": config.auth_header,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status in (200, 201, 202)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def _session_to_langfuse_trace(
    meta: SessionMeta,
    events: list[TraceEvent],
) -> dict:
    """Build a Langfuse Trace ingestion body."""
    duration_s = meta.total_duration_ms / 1000 if meta.total_duration_ms else None
    return {
        "id": meta.session_id,
        "name": f"agent-session ({meta.agent_name or 'agent'})",
        "timestamp": _iso(meta.started_at),
        "metadata": {
            "agent_name": meta.agent_name or "",
            "command": meta.command or "",
            "tool_calls": meta.tool_calls,
            "llm_requests": meta.llm_requests,
            "errors": meta.errors,
            "total_tokens": meta.total_tokens,
        },
        "tags": ["agent-strace"],
        **({"duration": duration_s} if duration_s else {}),
    }


def langfuse_data_indexes(events: list[TraceEvent]) -> list[int]:
    """Return the positions of the events whose data the Langfuse export reads.

    Every tool result and LLM response is read. A tool call or LLM request is
    read only when a later result or response pairs with it under the
    pairing.py rules. An error whose parent_id names a pending LLM request
    ends that request (see Pairer.match_parent), and both are read. Any other
    error is not read, but it can pair with a tool call, which then is not
    read either.
    """
    indexes = []
    calls: Pairer[int] = tool_pairer()
    requests: Pairer[int] = llm_pairer()
    for index, event in enumerate(events):
        if event.event_type in TOOL_REQUEST_TYPES:
            calls.add(event, index)
        elif event.event_type in LLM_REQUEST_TYPES:
            requests.add(event, index)
        elif event.event_type == EventType.ERROR:
            request_index = requests.match_parent(event)
            if request_index is not None:
                indexes.extend((index, request_index))
            else:
                calls.match(event)
        elif event.event_type in (EventType.TOOL_RESULT, EventType.LLM_RESPONSE):
            indexes.append(index)
            pairer = calls if event.event_type == EventType.TOOL_RESULT else requests
            parent = pairer.match(event)
            if parent is not None:
                indexes.append(parent)
    return indexes


_ONE_HOUR_CACHE_PROVIDERS = frozenset({"Anthropic", "AWS Bedrock"})


def _usage_details(usage: TokenUsage, model: str) -> dict[str, int]:
    """Return Langfuse ``usageDetails`` for a request that read or wrote a cache.

    Langfuse prices each usage type with the model price of the same key and
    expects exclusive buckets, so ``input`` is the uncached input and the
    cache reads and writes are separate keys. Langfuse lays these keys over
    the legacy ``usage`` object, which still carries the whole input for
    older servers. A request without cache usage returns ``{}`` so its
    generation is unchanged.
    """
    if not usage.cache_read_tokens and not usage.cache_write_tokens:
        return {}
    details = {"input": usage.uncached_input_tokens}
    if usage.cache_read_tokens:
        details["input_cached_tokens"] = usage.cache_read_tokens
    long_writes = 0
    if infer_provider(model) in _ONE_HOUR_CACHE_PROVIDERS:
        # Anthropic and Bedrock price 1-hour cache writes separately.
        # OpenAI's long lifetime is 12 hours, which Langfuse has no key for.
        long_writes = min(usage.cache_write_long_tokens, usage.cache_write_tokens)
    if usage.cache_write_tokens - long_writes:
        details["input_cache_creation"] = usage.cache_write_tokens - long_writes
    if long_writes:
        details["input_cache_creation_1h"] = long_writes
    details["output"] = usage.output_tokens
    details["total"] = usage.input_tokens + usage.output_tokens
    return details


def _generation(
    session_id: str,
    req_ev: TraceEvent | None,
    ev: TraceEvent,
) -> dict:
    """Build the GENERATION for one model request.

    ``ev`` ends the request: an LLM response, or an error linked to
    ``req_ev``. A failed or cancelled request gets level ERROR and a status
    message; a completed one has neither.
    """
    start_ts = req_ev.timestamp if req_ev else ev.timestamp
    usage, _, _ = model_request_usage(req_ev.data if req_ev else {}, ev.data)
    if req_ev:
        model = req_ev.data.get("model", "") or ""
    else:
        # A response with no request (Codex records each model
        # request this way) carries its own model and input count.
        model = event_model(ev.data)
    generation = {
        "id": ev.event_id,
        "traceId": session_id,
        "type": "GENERATION",
        "name": "llm-call",
        "startTime": _iso(start_ts),
        "endTime": _iso(ev.timestamp),
        "model": model or None,
        "usage": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "total": usage.input_tokens + usage.output_tokens,
        },
        "input": req_ev.data.get("prompt") if req_ev else None,
        "output": ev.data.get("text"),
    }
    usage_details = _usage_details(usage, model)
    if usage_details:
        generation["usageDetails"] = usage_details
    failure = llm_failure(ev)
    if failure is not None:
        generation["level"] = "ERROR"
        if failure.message:
            generation["statusMessage"] = failure.message
        elif failure.cancelled:
            generation["statusMessage"] = "cancelled"
        else:
            generation["statusMessage"] = "failed"
        generation["metadata"] = {"error.type": failure.error_type}
    return generation


def _events_to_langfuse_observations(
    session_id: str,
    events: list[TraceEvent],
) -> list[dict]:
    """Convert trace events to Langfuse Span/Generation observations."""
    observations = []
    calls: Pairer[TraceEvent] = tool_pairer()
    requests: Pairer[TraceEvent] = llm_pairer()

    for ev in events:
        if ev.event_type == EventType.TOOL_CALL:
            calls.add(ev)

        elif ev.event_type == EventType.ERROR:
            req_ev = requests.match_parent(ev)
            if req_ev is not None:
                # An error linked to an LLM request ends that request as a
                # failed generation.
                observations.append(_generation(session_id, req_ev, ev))
            else:
                # Other errors are not exported, but an error ends its tool call.
                calls.match(ev)

        elif ev.event_type == EventType.TOOL_RESULT:
            call = calls.match(ev)
            tool_name = ev.data.get("tool_name") or (
                call.data.get("tool_name", "tool") if call else "tool"
            )
            start_ts = call.timestamp if call else ev.timestamp
            end_ts = ev.timestamp
            observations.append({
                "id": ev.event_id,
                "traceId": session_id,
                "type": "SPAN",
                "name": f"tool/{tool_name}",
                "startTime": _iso(start_ts),
                "endTime": _iso(end_ts),
                "input": call.data.get("arguments") if call else None,
                "output": ev.data.get("result"),
                "metadata": {"tool_name": tool_name},
            })

        elif ev.event_type == EventType.LLM_REQUEST:
            requests.add(ev)

        elif ev.event_type == EventType.LLM_RESPONSE:
            observations.append(_generation(session_id, requests.match(ev), ev))

    return [o for o in observations if o]  # drop None entries


def _scores_to_langfuse(
    session_id: str,
    scores: list[EvalScore],
) -> list[dict]:
    """Convert eval scores to Langfuse Score objects."""
    return [
        {
            "traceId": session_id,
            "name": s.judge,
            "value": s.score,
            "comment": f"threshold={s.threshold} passed={s.passed}",
            "source": "API",
        }
        for s in scores
    ]


def export_session_to_langfuse(
    store: TraceStore,
    session_id: str,
    config: LangfuseConfig,
    include_scores: bool = True,
) -> bool:
    """Export one session (trace + observations + scores) to Langfuse."""
    try:
        meta = store.load_meta(session_id)
        events = store.load_events(session_id)
    except Exception as exc:
        sys.stderr.write(f"Failed to load session {session_id}: {exc}\n")
        return False

    # Batch ingestion endpoint
    batch: list[dict] = []

    trace_body = _session_to_langfuse_trace(meta, events)
    batch.append({"type": "trace-create", "body": trace_body})

    for obs in _events_to_langfuse_observations(session_id, events):
        batch.append({"type": "observation-create", "body": obs})

    if include_scores:
        scores = _load_eval_scores(store, session_id)
        for score_body in _scores_to_langfuse(session_id, scores):
            batch.append({"type": "score-create", "body": score_body})

    ok = _lf_post(config, "/api/public/ingestion", {"batch": batch})
    if ok:
        sys.stderr.write(
            f"Langfuse: exported session {session_id[:12]} "
            f"({len(events)} events, {len(batch)} items)\n"
        )
    else:
        sys.stderr.write(f"Langfuse: export failed for session {session_id[:12]}\n")
    return ok


# ---------------------------------------------------------------------------
# OTLP metrics export
# ---------------------------------------------------------------------------

def _otlp_gauge(
    name: str,
    value: float,
    attributes: dict[str, str],
    timestamp_ns: int,
) -> dict:
    """Build a single OTLP gauge data point."""
    return {
        "name": name,
        "gauge": {
            "dataPoints": [{
                "attributes": [
                    {"key": k, "value": {"stringValue": str(v)}}
                    for k, v in attributes.items()
                ],
                "timeUnixNano": str(timestamp_ns),
                "asDouble": float(value),
            }]
        },
    }


# _session_metrics reads the data of every event of these types.
METRICS_DATA_EVENT_TYPES = frozenset({EventType.TOOL_CALL, EventType.FILE_WRITE})


def _session_metrics(
    store: TraceStore,
    meta: SessionMeta,
) -> dict[str, float]:
    """Extract behavioral metrics for a session.

    ``agent_strace.session.cost_usd`` is the `agent-strace cost` total. It is
    left out when the session cannot be priced: its events cannot be read,
    or that command refuses the session.
    """
    from .cost import session_cost

    loaded = True
    try:
        events = store.load_events(meta.session_id)
    except Exception:
        events = []
        loaded = False

    tool_calls = 0
    errors = 0
    retries = 0
    files_written: set[str] = set()
    prev_tool: str | None = None
    run = 0

    for ev in events:
        if ev.event_type == EventType.TOOL_CALL:
            tool_calls += 1
            name = ev.data.get("tool_name", "")
            if name == prev_tool:
                run += 1
                if run >= 2:
                    retries += 1
            else:
                prev_tool = name
                run = 0
        elif ev.event_type == EventType.ERROR:
            errors += 1
        elif ev.event_type == EventType.FILE_WRITE:
            path = str_field(ev.data, "path", "") or str_field(ev.data, "file_path", "")
            if path:
                files_written.add(path)

    metrics: dict[str, float] = {}
    if loaded:
        try:
            metrics["agent_strace.session.cost_usd"] = session_cost(
                events, session_id=meta.session_id
            )
        except ValueError:
            pass
    duration_s = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0.0

    metrics.update({
        "agent_strace.session.error_rate": errors / max(tool_calls, 1),
        "agent_strace.session.retry_rate": retries / max(tool_calls, 1),
        "agent_strace.session.blast_radius": float(len(files_written)),
        "agent_strace.session.duration_s": duration_s,
        "agent_strace.session.tool_calls": float(tool_calls),
    })
    return metrics


def export_metrics_to_otlp(
    store: TraceStore,
    session_ids: list[str],
    config: OtlpMetricsConfig,
    include_scores: bool = True,
) -> bool:
    """Export behavioral metrics and eval scores as OTLP gauge metrics."""
    metrics: list[dict] = []
    ts_ns = int(time.time() * 1_000_000_000)

    for sid in session_ids:
        try:
            meta = store.load_meta(sid)
        except Exception:
            continue

        attrs = {
            "session_id": sid[:12],
            "agent_name": meta.agent_name or "unknown",
        }

        session_m = _session_metrics(store, meta)
        for metric_name, value in session_m.items():
            metrics.append(_otlp_gauge(metric_name, value, attrs, ts_ns))

        if include_scores:
            for score in _load_eval_scores(store, sid):
                score_attrs = {**attrs, "judge": score.judge}
                metrics.append(_otlp_gauge(
                    "agent_strace.eval.score", score.score, score_attrs, ts_ns
                ))

    if not metrics:
        sys.stderr.write("No metrics to export.\n")
        return True

    payload = json.dumps({
        "resourceMetrics": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "agent-strace"}}
                ]
            },
            "scopeMetrics": [{
                "scope": {"name": "agent-strace"},
                "metrics": metrics,
            }],
        }]
    }).encode()

    url = config.endpoint.rstrip("/") + "/v1/metrics"
    req_headers = {"Content-Type": "application/json"}
    req_headers.update(config.headers)

    req = urllib.request.Request(url, data=payload, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            ok = resp.status in (200, 202)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        sys.stderr.write(f"OTLP metrics export failed: {exc}\n")
        return False

    if ok:
        sys.stderr.write(
            f"OTLP metrics: exported {len(metrics)} data points "
            f"for {len(session_ids)} session(s)\n"
        )
    return ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def _loadable_events(store: TraceStore, session_ids: list[str]):
    """Yield (session_id, events) for each session the export can load."""
    for sid in session_ids:
        try:
            store.load_meta(sid)
            events = store.load_events(sid)
        except Exception:
            continue
        yield sid, events


def cmd_export_scores(args) -> int:
    """Handle: agent-strace export --scores --backend langfuse|otlp"""
    import os
    store = TraceStore(args.trace_dir)
    backend = getattr(args, "backend", "langfuse") or "langfuse"
    include_scores = getattr(args, "scores", False)
    include_metrics = getattr(args, "metrics", False)
    since_raw = getattr(args, "since", None)
    session_arg = getattr(args, "session_id", None)

    # Resolve sessions
    if session_arg:
        found = store.find_session(session_arg)
        session_ids = [found] if found else []
    elif since_raw:
        days = float(since_raw.rstrip("d"))
        cutoff = time.time() - days * 86400
        session_ids = [
            m.session_id for m in store.list_sessions()
            if m.started_at >= cutoff
        ]
    else:
        latest = store.get_latest_session_id()
        session_ids = [latest] if latest else []

    if not session_ids:
        sys.stderr.write("No sessions found.\n")
        return 1
    multi = not session_arg and bool(since_raw)

    if backend == "langfuse":
        config = LangfuseConfig(
            public_key=(
                getattr(args, "langfuse_public_key", None)
                or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
            ),
            secret_key=(
                getattr(args, "langfuse_secret_key", None)
                or os.environ.get("LANGFUSE_SECRET_KEY", "")
            ),
            host=(
                getattr(args, "langfuse_host", None)
                or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
            ),
        )
        if not config.configured:
            sys.stderr.write(
                "Langfuse credentials not set. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, "
                "or use --langfuse-public-key / --langfuse-secret-key.\n"
            )
            return 1

        for sid, events in _loadable_events(store, session_ids):
            require_object_event_data(
                events,
                indexes=langfuse_data_indexes(events),
                session_id=sid if multi else None,
            )
        success = 0
        for sid in session_ids:
            if export_session_to_langfuse(store, sid, config, include_scores=include_scores):
                success += 1
        sys.stdout.write(f"Langfuse: {success}/{len(session_ids)} sessions exported.\n")
        return 0 if success == len(session_ids) else 1

    elif backend == "otlp":
        endpoint = (
            getattr(args, "otlp_endpoint", None)
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        )
        if not endpoint:
            sys.stderr.write(
                "OTLP endpoint not set. "
                "Set OTEL_EXPORTER_OTLP_ENDPOINT or use --otlp-endpoint.\n"
            )
            return 1

        headers_raw = (
            getattr(args, "otlp_headers", None)
            or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        )
        headers: dict[str, str] = {}
        if headers_raw:
            for pair in headers_raw.split(","):
                if "=" in pair:
                    k, _, v = pair.partition("=")
                    headers[k.strip()] = v.strip()

        for sid, events in _loadable_events(store, session_ids):
            require_object_event_data(
                events,
                event_types=METRICS_DATA_EVENT_TYPES,
                session_id=sid if multi else None,
            )
        config_otlp = OtlpMetricsConfig(endpoint=endpoint, headers=headers)
        ok = export_metrics_to_otlp(
            store, session_ids, config_otlp, include_scores=include_scores
        )
        return 0 if ok else 1

    else:
        sys.stderr.write(f"Unknown backend: {backend!r}. Use 'langfuse' or 'otlp'.\n")
        return 1
