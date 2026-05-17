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

from .models import EventType, SessionMeta, TraceEvent
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


def _events_to_langfuse_observations(
    session_id: str,
    events: list[TraceEvent],
) -> list[dict]:
    """Convert trace events to Langfuse Span/Generation observations."""
    observations = []
    pending_calls: dict[str, TraceEvent] = {}

    for ev in events:
        if ev.event_type == EventType.TOOL_CALL:
            pending_calls[ev.event_id] = ev

        elif ev.event_type == EventType.TOOL_RESULT:
            call = pending_calls.pop(ev.parent_id, None)
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
            pending_calls[ev.event_id] = ev

        elif ev.event_type == EventType.LLM_RESPONSE:
            req_ev = pending_calls.pop(ev.parent_id, None)
            start_ts = req_ev.timestamp if req_ev else ev.timestamp
            input_tokens = (req_ev.data.get("input_tokens", 0) if req_ev else 0) or 0
            output_tokens = ev.data.get("output_tokens", 0) or 0
            model = (req_ev.data.get("model", "") if req_ev else "") or ""
            observations.append({
                "id": ev.event_id,
                "traceId": session_id,
                "type": "GENERATION",
                "name": "llm-call",
                "startTime": _iso(start_ts),
                "endTime": _iso(ev.timestamp),
                "model": model or None,
                "usage": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
                "input": req_ev.data.get("prompt") if req_ev else None,
                "output": ev.data.get("text"),
            })

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


def _session_metrics(
    store: TraceStore,
    meta: SessionMeta,
) -> dict[str, float]:
    """Extract behavioral metrics for a session."""
    try:
        events = store.load_events(meta.session_id)
    except Exception:
        events = []

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
            path = ev.data.get("path") or ev.data.get("file_path") or ""
            if path:
                files_written.add(path)

    cost = meta.total_tokens / 1_000_000 * 3.0
    duration_s = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0.0

    return {
        "agent_strace.session.cost_usd": cost,
        "agent_strace.session.error_rate": errors / max(tool_calls, 1),
        "agent_strace.session.retry_rate": retries / max(tool_calls, 1),
        "agent_strace.session.blast_radius": float(len(files_written)),
        "agent_strace.session.duration_s": duration_s,
        "agent_strace.session.tool_calls": float(tool_calls),
    }


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

        config_otlp = OtlpMetricsConfig(endpoint=endpoint, headers=headers)
        ok = export_metrics_to_otlp(
            store, session_ids, config_otlp, include_scores=include_scores
        )
        return 0 if ok else 1

    else:
        sys.stderr.write(f"Unknown backend: {backend!r}. Use 'langfuse' or 'otlp'.\n")
        return 1
