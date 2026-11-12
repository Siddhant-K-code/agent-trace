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
| Failed or cancelled model request | `error.type` on the `gen_ai.assistant.message` event | ERROR status and `error.type` on the `gen_ai.client.operation` span |
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

Stream spans to an OTLP/HTTP receiver while the session runs, rather than exporting after it ends. Pass the receiver's base URL to `watch --stream-to` and add `--stream-otlp`:

```bash
agent-strace watch --stream-to http://localhost:4318 --stream-otlp
```

Without `--stream-otlp`, `--stream-to` POSTs the raw events as NDJSON to the URL exactly as given. That suits the [agent-strace collector](server.md#live-streaming-from-watch), not an OTLP receiver.

With `--stream-otlp`, watch drops trailing slashes from the URL path, then appends `/v1/traces` unless the path already ends with it. The query string is kept:

| `--stream-to` | Spans are POSTed to |
|---|---|
| `http://localhost:4318` | `http://localhost:4318/v1/traces` |
| `http://localhost:4318/v1/traces` | `http://localhost:4318/v1/traces` |
| `https://otel.example.com/api/public/otel/` | `https://otel.example.com/api/public/otel/v1/traces` |
| `https://otel.example.com/ingest?team=a` | `https://otel.example.com/ingest/v1/traces?team=a` |

Watch sends events in batches: when `--stream-batch-size` events are waiting, when `--stream-flush-interval` seconds have passed since the last POST, and when watch stops. Each POST carries only the spans built from its own batch, so the backend receives the session in pieces:

- Every POST includes the session root span again, with the same trace ID and span ID. A backend may show it more than once or keep only one copy.
- A tool call whose call and result land in different batches arrives as two spans, one from each batch.

For one complete set of spans per session, export it after it ends with `agent-strace export --format otlp-genai` (see [Per-backend setup](#per-backend-setup)).

Watch sends no authentication headers and has no flag to add them, so it cannot stream straight to a hosted backend that needs an API key, such as Honeycomb. A local receiver that needs no key, such as Jaeger or the Datadog Agent's OTLP receiver, works directly. For a hosted backend, run a local [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/) that adds the key and forwards the spans:

```yaml
# otel-collector.yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 127.0.0.1:4318
exporters:
  otlphttp:
    endpoint: https://api.honeycomb.io
    headers:
      x-honeycomb-team: ${env:HONEYCOMB_API_KEY}
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlphttp]
```

```bash
agent-strace watch --stream-to http://127.0.0.1:4318 --stream-otlp
```

| Flag | Default | Description |
|---|---|---|
| `--stream-to URL` | none | `http://` or `https://` URL to stream to |
| `--stream-otlp` | off | POST OTLP/HTTP JSON spans to the `/v1/traces` path under `--stream-to` instead of NDJSON to the URL itself |
| `--stream-batch-size N` | `10` | Send once N events are waiting |
| `--stream-flush-interval S` | `2` | Send waiting events once S seconds have passed since the last POST, even if the batch is not full. `0` sends each event at once |

Watch checks `--stream-to` and `--stream-flush-interval` before it looks up the session or writes anything. It refuses a URL that does not start with `http://` or `https://`, has no host, has a bad port, contains spaces or control characters, or carries a user name or password. It also refuses a flush interval that is negative, infinite or NaN. Each refusal is one line, such as `[watch] invalid --stream-to value: it must not contain a user name or password`, and exit status 1.

A failed POST never stops watch. It prints one line to stderr, such as `[stream] OTLP POST to https://otel.example.com failed: HTTP Error 500: Internal Server Error`, and goes on with the next batch. The failed batch is not sent again. These lines, and the `[watch] Streaming events to` line at startup, name only the URL's scheme, host and port, never its path, query or user info. Watch does not follow redirects: a 3xx answer is a failed POST whose line says so, so pass the final URL. Each POST times out after 10 seconds.

When watch stops, it waits up to 15 seconds for the last POST to finish. If that takes more than a second, it prints `[stream] Waiting up to 14s more for the last events to be sent; press Ctrl-C to stop waiting.` If the wait runs out or you press Ctrl-C, it prints one line such as `[stream] Stopped waiting; 3 events were not confirmed sent.` and exits as it would have otherwise.

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

Cost is the same total that `agent-strace cost` reports with its default
model, including recorded token usage where the agent provides it. A session
whose events cannot be priced adds no cost sample and is not compared on cost.
Older versions saved every cost as 0, so run `agent-strace baseline update`
again after upgrading.

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
| `agent_strace.session.cost_usd` | Session cost, as `agent-strace cost` reports it with its default model |
| `agent_strace.session.error_rate` | Errors / tool calls |
| `agent_strace.session.retry_rate` | Consecutive same-tool retries / tool calls |
| `agent_strace.session.blast_radius` | Distinct files written |
| `agent_strace.session.duration_s` | Wall-clock session duration |
| `agent_strace.eval.score` | Judge score per session |

`agent_strace.session.cost_usd` prices recorded token usage, cache reads and
writes included, at the model's rates. Earlier versions priced each session's
token count at a flat $3 per million. A session whose events cannot be read, or
that `agent-strace cost` refuses, has no `cost_usd` data point; its other
metrics are still exported.

---

## How sessions map to OpenTelemetry

| agent-trace | OpenTelemetry |
|---|---|
| session | trace |
| tool_call + tool_result | span (with duration) |
| error | span with error status + exception event |
| failed or cancelled model request | error status and `error.type`; see [Failed and cancelled model requests](#failed-and-cancelled-model-requests) |
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

The exports also end an `llm_request` at an `error` whose `parent_id` names
it, and export the pair as one failed model call (see
[Failed and cancelled model requests](#failed-and-cancelled-model-requests)).
Only rule 1 applies: such an `error` never pairs with a request by model, and
any other `error` stays on the tool channel. `cost` does not read `error`
events.

Requests still pending at the end are exported as unfinished spans in OTLP and
are not exported to Langfuse.

Recorders set `parent_id` while they write the trace, with rules from the same
module. `record` and `record-http` link a `tool_result` or `error` to the
pending `tool_call` with the same JSON-RPC id, and keep at most 10,000 calls
pending per proxy. Hooks link a result to the pending call with its
`tool_use_id`, or, when the provider sends none, to the oldest pending call of
the same tool name, preferring calls from the same subagent.

### Responses recorded without a request

OpenAI Codex records each model request only as an `llm_response` event, from
its `PostModelRequest` hook. No `llm_request` is recorded, by design: the
Codex [capture matrix](capture-matrix.md#openai-codex) entry declares
`results_without_requests: ["llm_response"]`. Each response carries its own
`model` (when Codex reports one), `status` and token usage. `input_tokens`
already includes cached input and `output_tokens` already includes reasoning.
A failed or cancelled request records no usage.

A response that pairs with no request is treated as the whole model request:

- `--format otlp-genai` and Temporal: the response is its own `chat` span with
  its model and token counts. The root span's `agent.strace.evidence.*`
  attributes read the same capture matrix declaration as `validate`, so
  declared Codex responses are one `declared_results_without_requests`
  provider limitation instead of one `orphan_llm_response` gap each. Responses
  without a request from any other source are still orphans.
- `--format otlp`: the `gen_ai.assistant.message` event carries the
  response's own `gen_ai.request.model` and `gen_ai.usage.input_tokens`. A
  paired response is unchanged, because its request event already carries
  them.
- Langfuse: the generation takes its model and input count from the response.
  A paired response still takes them from its request.
- `cost`, `tree`, `timeline`, `replay`, `watch` and the EU AI Act compliance
  export: once a session has such a response with token counts, every
  `llm_response` counts the tokens it recorded, or none if it recorded none
  (as for a failed or cancelled request), and no other event is counted,
  because its content is already part of a request's input. Nothing is also
  estimated from event size. Sessions without such a response keep the size
  estimate. `watch` counts events as they arrive, so events before the first
  recorded usage keep their size estimate.
- `cost --breakdown provider`: a response that names no model still reports
  its tokens, under the model `(not recorded)` (empty in CSV) and the provider
  `Unknown`, with no estimated price.

Cost estimates price each response's `cached_input_tokens` at the model's
cached input rate and its `cache_write_input_tokens` at the cache write rate,
and the rest of `input_tokens` at the input rate. A model without a published
cache rate, such as the `haiku`, `gpt4` and `gpt4o` choices of `--model`,
bills cached input at its input rate. See [`cost`](commands.md#cost)
for the usage shapes read from each provider.

### Failed and cancelled model requests

A model request failed or was cancelled when either:

- its `llm_response` has `status` `failed` or `cancelled` (also `canceled`,
  in any case), or carries a non-empty `error`. Codex records failed and
  cancelled requests this way. A response with any other `status`, or none,
  completed.
- an `error` event's `parent_id` names its `llm_request`. The SDK
  integrations and `@trace_llm_call` record a model call that raised this way.

The exports report such a request as an error, following the
[OTel GenAI spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
conventions and
[Recording errors](https://opentelemetry.io/docs/specs/semconv/general/recording-errors/):

| Export | Failed or cancelled request |
|---|---|
| `--format otlp-genai`, Temporal, `watch --stream-otlp` | The `gen_ai.client.operation` span has status ERROR with the recorded error text (at most 200 characters) as its description, and `error.type`. A request ended by an `error` is one span with the request's span ID and an `exception` event, with no separate error span. |
| `--format otlp` | There is no span per request, so the `gen_ai.assistant.message` event carries `error.type` and, when recorded, `error.message`. |
| `export --scores --backend langfuse` | The generation has level `ERROR`, `statusMessage` set to the error text (`failed` or `cancelled` when none was recorded), and `error.type` in its metadata. A request ended by an `error` is exported as a generation too. |

`error.type` is the recorded `error_type` or `exception_type`, else the `type`
or `code` of an `error` object, else `cancelled` for a cancelled request and
`_OTHER`, the conventions' fallback value, for a failed one. When no finish
reason was recorded, `gen_ai.response.finish_reasons` is `error`, which the
conventions specify for a generation that failed or was cancelled.

The GenAI conventions name no span status for a cancelled request, but they
treat cancellation as an abnormal end, so the exports report it as an error
with `error.type` `cancelled`.

What does not change:

- A completed request, or a response with no `status`, exports exactly as
  before, with OK status.
- A failed request is not an `error` event. Session `errors`,
  `agent_strace.session.error_rate`, `timeline` and the root span status do
  not count it.
- An `error` with no `parent_id`, as older versions recorded a failed SDK
  call, still exports as an unfinished request span and a separate error
  span.

---

## Inspect the OTLP payload

```bash
agent-strace export <session-id> --format otlp > trace.json
```
