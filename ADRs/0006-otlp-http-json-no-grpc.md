# ADR-0006: OTLP Export via HTTP/JSON — No gRPC, No SDK

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

Production observability platforms (Grafana, Honeycomb, Datadog, Jaeger) support OpenTelemetry (OTel) as an ingestion format. Exporting agent traces to these platforms enables correlation with application traces and use of existing dashboards. OTel supports two wire formats: gRPC/Protobuf and HTTP/JSON.

## Decision

OTLP export uses HTTP/JSON (`otlp.py`) via `urllib.request` from the standard library. No `opentelemetry-sdk`, no `grpcio`, no `protobuf`.

**Session-to-trace mapping:**
- Each agent session becomes one OTel trace.
- The session is the root span.
- Each `tool_call`+`tool_result` pair becomes a child span with duration from `parent_id` linking.
- `user_prompt` and `assistant_response` events become OTel span events on the root span (not child spans) — they are conversational events without duration.

**ID generation:**
- OTel trace IDs (32 hex chars) are derived from session IDs via `sha256(session_id)[:32]`.
- OTel span IDs (16 hex chars) are derived from event IDs via `sha256(event_id)[:16]`.
- SHA-256 expansion is deterministic, enabling idempotent re-export.

**Timestamps** are encoded as nanosecond integers represented as strings, per the OTLP JSON spec (to avoid JavaScript's 53-bit integer precision limit).

**OTel attributes** map Python types directly. Dicts and lists are serialized as JSON strings (`stringValue: json.dumps(value)`) because OTel attributes do not support nested structures natively.

When `--endpoint` is omitted, OTLP output is written to stdout as JSON for inspection.

## Consequences

- **Zero new dependencies** — consistent with ADR-0003.
- **All major backends supported** — every production observability platform supports OTLP/HTTP/JSON.
- **Human-readable and debuggable** — HTTP/JSON is inspectable with `curl` and `jq`; gRPC/Protobuf is not.
- **No connection pooling or retry** — `urllib.request` is fire-and-forget. Export failures are logged to stderr but do not affect the trace.
- **Deterministic IDs** enable idempotent re-export without duplicate span creation (assuming the backend deduplicates by trace/span ID).
