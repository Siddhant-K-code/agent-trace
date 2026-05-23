# ADR-0011: OTLP GenAI Semantic Conventions Export Format

**Status:** Accepted  
**Date:** 2026-05  
**Deciders:** Siddhant Khare

## Context

agent-strace already exports OTLP (ADR-0006). The OpenTelemetry GenAI semantic
conventions (`gen_ai.*`) are now the standard attribute set for AI spans and are
natively understood by Datadog LLM Observability, Grafana GenAI dashboards,
Honeycomb, and any OTel-compatible backend.

Without this mapping, agent-strace traces land in production backends as
unrecognized custom spans. They don't get AI-specific dashboards, cost views,
token usage charts, or anomaly detection.

## Decision

Add a `--format otlp-genai` flag to `agent-strace export` that applies the
OTel GenAI semantic conventions mapping. The existing `--format otlp` output
is unchanged for backwards compatibility.

### Mapping

| agent-strace event | OTel GenAI span / attribute |
|---|---|
| `llm_request` + `llm_response` | `gen_ai.client.operation` child span with `gen_ai.request.model`, `gen_ai.request.max_tokens`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons` |
| `tool_call` + `tool_result` | `gen_ai.tool.call/<name>` child span with `gen_ai.tool.name`, `gen_ai.tool.call.id` |
| `user_prompt` | `gen_ai.user.message` event on root span |
| `assistant_response` | `gen_ai.assistant.message` event on root span |
| `error` | OTel `exception` event with `exception.type`, `exception.message` |
| `session_start` / `session_end` | Root span with `gen_ai.agent.id`, `gen_ai.agent.name` |

### Span hierarchy

```
session root span (gen_ai.agent.session)
  ├── gen_ai.client.operation   (one per LLM request/response pair)
  ├── gen_ai.tool.call/<name>   (one per tool call/result pair)
  └── gen_ai.tool.call/<name>   (error variant with exception event)
```

### Key differences from --format otlp

- LLM request/response pairs become proper child spans (not events on root)
- Root span carries `gen_ai.agent.id` and `gen_ai.agent.name`
- Error events use the OTel `exception` event format
- Tool input/output attributes use `gen_ai.tool.input.*` / `gen_ai.tool.output`
- Scope name includes `genai-semconv-1.27` for version tracking

## Consequences

- Traces exported with `--format otlp-genai` appear in Grafana, Datadog, and
  Honeycomb with AI-specific dashboards populated automatically.
- Existing `--format otlp` output is unchanged — no breaking change.
- No new runtime dependencies — the mapping is pure Python stdlib.
- The `gen_ai.system` attribute is derived heuristically from the model name;
  this may be incorrect for custom or fine-tuned models.
