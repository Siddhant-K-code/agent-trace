# Production tracing (OTLP export)

## Quick demo (no signup required)

Spin up Jaeger locally with one Docker command and a browser trace UI:

```bash
# Start Jaeger (OTLP on 4318, UI on 16686)
docker run --rm -d --name jaeger \
  -p 16686:16686 -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Export your session
agent-strace export --format otlp-genai --endpoint http://localhost:4318

# Open the UI
open http://localhost:16686
```

Select service `agent-trace` in the Jaeger UI to see the full trace.

Export sessions as OpenTelemetry spans to your existing observability stack. Sessions become traces. Tool calls become spans with duration and inputs. Errors get exception events. No new dependencies.

---

## OTel GenAI semantic conventions

Use `--format otlp-genai` to export with [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). This produces AI-native spans that populate token usage charts, cost views, and LLM dashboards automatically.

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:4318
```

| Aspect | `--format otlp` | `--format otlp-genai` |
|---|---|---|
| LLM calls | Events on root span | `gen_ai.client.operation` child spans |
| Tool calls | `tool/<name>` spans | `gen_ai.tool.call/<name>` spans |
| Root span | `agent.name` attribute | `gen_ai.agent.id` + `gen_ai.agent.name` |
| Errors | Custom error span | OTel `exception` event format |
| `gen_ai.conversation.id` | Root span | Root, LLM and tool spans |

`--format otlp` is unchanged for backwards compatibility. See [ADR-0011](../ADRs/0011-otlp-genai-semantic-conventions.md) for design rationale.

### Joining with native agent telemetry

Both formats emit `gen_ai.conversation.id` when the agent supplied its own
session ID: hook sessions whose payload carries `session_id` (Claude Code,
GitHub Copilot CLI, Codex, Gemini CLI, Cursor), and `import` of Claude Code
JSONL transcripts. The value is the agent's full
session ID, not the agent-strace session ID. Sessions without one (`record`,
proxy, SDK integrations, and sessions recorded before this field existed)
omit the attribute.

| Native source | Join on |
|---|---|
| GitHub Copilot CLI OTel (`gen_ai.conversation.id`) | `gen_ai.conversation.id` = `gen_ai.conversation.id` |
| Claude Code OTel (`session.id`) | `gen_ai.conversation.id` = `session.id` |

Do not join on `session.id` from agent-strace: it is the local agent-strace
session ID, which is truncated to 16 characters for hook sessions.

#### Version skew

Sessions with a native ID store it as `conversation_id` in `meta.json`.
Readers and collectors on v0.106.0 or older reject metadata that contains
this key: `list` and other readers on a shared trace store fail to load the
session, and the collector returns 400 on `POST /sessions`. Upgrade the
collector and every reader that shares the trace store before the clients
that write it. Sessions without a native ID do not write the key and stay
readable by older versions.

From this release, readers and the collector ignore metadata and event keys
they do not know and keep them when they rewrite the record, so later
additive fields do not need this ordering.

### OpenInference compatibility

The same `otlp-genai` payload includes additive OpenInference attributes for
Phoenix and other OpenInference consumers. The OTLP span kind stays unchanged.
`openinference.span.kind` identifies the AI operation.

| Span | OTLP kind | OpenInference kind | Key attributes |
|---|---:|---|---|
| Session root | `INTERNAL` (`1`) | `AGENT` | `session.id`, `metadata` |
| LLM call | `CLIENT` (`3`) | `LLM` | `llm.provider`, `llm.model_name`, token counts, recorded cost |
| Tool call | `CLIENT` (`3`) | `TOOL` | `tool.name`, `input.value`, `output.value` |

`agent.strace.*` attributes preserve redaction state, persisted provenance,
event IDs, previous hashes, recorded cost, and observed structural evidence
gaps. `session.id` is copied to every span for backend filtering.

The mapping is not fully lossless:

- Multi-turn user and assistant messages remain root span events.
- Tool input is capped at 1000 characters and tool output at 500 characters.
- Only recorded cost is exported. Missing cost is not estimated.
- Redaction markers and flags survive, but removed secret values do not.
- Observed structural gaps survive. Provider capture-matrix blind spots remain
  in `agent-strace health`.
- Complete hash-chain verification still requires the source NDJSON.

The machine-readable contract is
[`examples/otlp/openinference-mapping.json`](../examples/otlp/openinference-mapping.json).

---

## Per-backend setup

### Phoenix / OpenInference

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:6006
```

The exporter appends `/v1/traces`. Phoenix serves OTLP/HTTP on its UI port
`6006`. OpenInference span kinds and compatibility attributes are included in
the payload.

### Langfuse

Langfuse accepts OTLP/HTTP JSON at `/api/public/otel/v1/traces`. Pass the base
OTLP endpoint because the exporter appends `/v1/traces`.

```bash
AUTH_STRING="$(
  printf '%s' "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" |
  base64 | tr -d '\n'
)"

agent-strace export <session-id> --format otlp-genai \
  --endpoint https://cloud.langfuse.com/api/public/otel \
  --header "Authorization: Basic $AUTH_STRING" \
  --header "x-langfuse-ingestion-version: 4"
```

The OTLP recipe exports traces. Eval scores still use the existing direct
exporter:

```bash
agent-strace export --scores --backend langfuse
```

### Honeycomb

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint https://api.honeycomb.io \
  --header "x-honeycomb-team: $HONEYCOMB_API_KEY" \
  --service-name my-agent
```

Use `https://api.eu1.honeycomb.io` for the EU instance.

### Datadog

Enable the Datadog Agent OTLP HTTP receiver, then send to port `4318`:

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:4318
```

This recipe does not send OTLP JSON to the Datadog logs intake.

### Grafana Tempo / Jaeger

```bash
# Local collector
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:4318
```

### New Relic

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://otlp.nr-data.net \
  --header "api-key: $NEW_RELIC_LICENSE_KEY"
```

### Splunk

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://ingest.<realm>.signalfx.com \
  --header "X-SF-Token: $SPLUNK_ACCESS_TOKEN"
```

### Offline fixture validation

The fixtures contain no credentials and require no network:

```bash
python3 -m json.tool examples/otlp/genai-conformance.json >/dev/null
PYTHONPATH=src python3 -m unittest tests.test_otlp_conformance -v
```

Backend metadata and limitations are in
[`examples/otlp/backend-recipes.json`](../examples/otlp/backend-recipes.json).

---

## Live OTLP streaming

Stream events to an OTLP endpoint in real-time as the session runs, rather than exporting after it ends. Use `watch --stream-to` to enable:

```bash
agent-strace watch --stream-to http://localhost:4318/v1/traces
```

Each tool call is flushed as a span immediately on completion. The session root span is closed when the session ends or the watchdog kills it.

Configure batch size and flush interval:

```bash
agent-strace watch \
  --stream-to https://api.honeycomb.io/v1/traces \
  --stream-batch-size 10 \
  --stream-flush-interval 5
```

| Flag | Default | Description |
|---|---|---|
| `--stream-to URL` | none | OTLP HTTP endpoint to stream to |
| `--stream-batch-size N` | `10` | Flush after N events |
| `--stream-flush-interval S` | `5` | Flush every S seconds regardless of batch size |

---

## Baseline anomaly detection

Build a statistical baseline from completed sessions that started recently and
alert when a new session deviates:

```bash
# Build a baseline from completed sessions that started in the last 30 days
agent-strace baseline update --since 30

# Check a session against the baseline
agent-strace baseline check <session-id>

# Show baseline statistics
agent-strace baseline show
```

Metrics tracked: cost, tool call count, duration, error rate, and LLM request
count. Each metric gets a mean and sample standard deviation. A session is
flagged when any comparable metric exceeds `--sigma` standard deviations from
the mean (default: `2.0`). At least three baseline samples are required before
a metric is compared.

```bash
agent-strace baseline check <session-id> --sigma 2.5
```

Use the same custom profile path when updating, checking, and displaying a
baseline:

```bash
agent-strace baseline update --since 14 --output .agent-traces/ci-baseline.json
agent-strace baseline check <session-id> --baseline .agent-traces/ci-baseline.json
agent-strace baseline show --baseline .agent-traces/ci-baseline.json
```

Use `baseline check` as a CI gate — it exits 1 when the session is anomalous:

```bash
agent-strace baseline check "$SESSION_ID"
```

## EU AI Act Audit Packages

`agent-strace export --format eu-ai-act` creates a local JSON package for
Article 12 logging evidence and Article 13 transparency documentation:

```bash
agent-strace export <session-id> --format eu-ai-act --output compliance-report.json

agent-strace export --all --since 2026-01-01 --until 2026-03-31 \
  --format eu-ai-act --output Q1-2026-audit.json
```

The export includes event summaries, data categories processed, tools and
models observed, human oversight points, and hash-chain integrity metadata.
Verify the exported chain links with:

```bash
agent-strace verify --from-export compliance-report.json
```

Before exporting a store for review, run:

```bash
agent-strace audit-readiness
```

---

## Behavioral metrics

Export per-session behavioral metrics as OTLP gauge metrics:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io

agent-strace export --metrics --backend otlp --since 30d
```

| Metric | Description |
|---|---|
| `agent_strace.session.cost_usd` | Estimated cost per session |
| `agent_strace.session.error_rate` | Errors / tool calls |
| `agent_strace.session.retry_rate` | Consecutive same-tool retries / tool calls |
| `agent_strace.session.blast_radius` | Distinct files written |
| `agent_strace.session.duration_s` | Wall-clock session duration |
| `agent_strace.eval.score` | Judge score per session |

---

## How sessions map to OpenTelemetry

| agent-trace | OpenTelemetry |
|---|---|
| session | trace |
| tool_call + tool_result | span (with duration) |
| error | span with error status + exception event |
| user_prompt | event on root span |
| assistant_response | event on root span |
| session_id | trace ID |
| event_id | span ID |
| parent_id | parent span ID |

### How requests and responses are paired

The OTLP, OTLP GenAI, Temporal and Langfuse exports and `cost` pair events with
the same rules, defined in `src/agent_trace/pairing.py`:

1. A response pairs with the pending request its `parent_id` names.
2. If there is no such request, a `tool_result` or `error` stays unpaired. An
   `llm_response` pairs with the most recent pending `llm_request` that names
   the same model. A request and a response that both name no model count as
   the same model. Only when no pending request names the same model does the
   response pair with the most recent pending request where one side names no
   model. Requests and responses that name different models never pair. This
   covers SDK integrations that record responses without a `parent_id`. The
   most recent request wins because a failed call leaves its request without
   a response.

   A matching model is the only evidence this fallback has, so it always
   beats a missing one. Without that order, a request with no model (for
   example a CrewAI task) would take the response of an older `gpt-4o`
   request, and a CrewAI task result would take a failed `gpt-4o` request.
   The match where one side names no model stays as a last resort because
   some integrations record the model on only one side of a call.
3. A response whose `parent_id` names a request that already paired stays
   unpaired. It never takes another request's place.

Requests still pending at the end are exported as unfinished spans in OTLP and
are not exported to Langfuse.

Recorders set `parent_id` while they write the trace, with rules from the same
module. `record` and `record-http` link a `tool_result` or `error` to the
pending `tool_call` with the same JSON-RPC id, and keep at most 10,000 calls
pending per proxy. Hooks link a result to the pending call with its
`tool_use_id`, or, when the provider sends none, to the oldest pending call of
the same tool name, preferring calls from the same subagent.

---

## Inspect the OTLP payload

```bash
agent-strace export <session-id> --format otlp > trace.json
```
