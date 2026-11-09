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
| `session_start` / `session_end` | Root span with `gen_ai.agent.id`, `gen_ai.agent.name`, and `gen_ai.conversation.id` when known |

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

### OpenInference compatibility

The GenAI payload also emits additive OpenInference attributes. Session, LLM,
and tool spans use `openinference.span.kind` values `AGENT`, `LLM`, and `TOOL`.
Provider, model, token, recorded cost, session, tool, input, output, redaction,
provenance, and observed evidence-gap fields remain in the same OTLP/HTTP JSON
payload.

No separate exporter or transport is added. The machine-readable mapping and
its explicit losses live in `examples/otlp/openinference-mapping.json`.

### `gen_ai.conversation.id` (added 2026-11)

The GenAI conventions (now maintained in
`open-telemetry/semantic-conventions-genai`) define `gen_ai.conversation.id`
as "the unique identifier for a conversation (session, thread)", with
Development stability. It is conditionally required on `invoke_agent`,
inference and `execute_tool` spans when the instrumented library has an ID
readily available. Instrumentations SHOULD NOT invent one (a new UUID, the
trace ID, or a content hash).

agent-strace follows that rule:

- The value is the agent's own session ID, stored in full in the new
  optional `SessionMeta.conversation_id` field. Hooks take it from the
  payload `session_id` (Copilot CLI `sessionId` is normalised to the same
  key) only when its first 16 characters equal the local session ID. Claude
  JSONL import takes it from the transcript `sessionId`, not from the file
  name fallback.
- Sessions without a native ID (`record`, proxy, decorator, framework
  integrations, server-created sessions, `backfill`, and sessions recorded
  before this change) omit the attribute. agent-strace never puts its own
  local `session_id` under this name.
- `--format otlp-genai` puts it on the root span and on every LLM and tool
  span, including error and unmatched spans. `--format otlp` puts it on the
  root span only.

Join keys with native agent telemetry:

| Agent | Native attribute | Joins with |
|---|---|---|
| GitHub Copilot CLI | `gen_ai.conversation.id` on `invoke_agent` and `chat` spans | `gen_ai.conversation.id` |
| Claude Code | `session.id` on every span, metric and event (Claude Code does not emit `gen_ai.conversation.id`) | `gen_ai.conversation.id` |

The OpenInference `session.id` that agent-strace emits stays the local
agent-strace session ID (the first 16 characters for hook sessions), so it is
not the same value as Claude Code's `session.id`. Use
`gen_ai.conversation.id` for cross-source joins.

`gen_ai.agent.id` keeps its existing meaning (the local session ID). Copilot
CLI uses `gen_ai.agent.id` for the agent definition (for example
`github.copilot.default`), so that attribute is not a join key.

## Consequences

- Traces exported with `--format otlp-genai` appear in Grafana, Datadog, and
  Honeycomb with AI-specific dashboards populated automatically.
- Existing `--format otlp` output is unchanged — no breaking change.
- No new runtime dependencies — the mapping is pure Python stdlib.
- The `gen_ai.system` attribute is derived heuristically from the model name;
  this may be incorrect for custom or fine-tuned models.
