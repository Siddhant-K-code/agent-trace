# Production tracing (OTLP export)

## Quick demo (no signup required)

Spin up Jaeger locally — one Docker command, full trace UI in the browser:

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

`--format otlp` is unchanged for backwards compatibility. See [ADR-0011](../ADRs/0011-otlp-genai-semantic-conventions.md) for design rationale.

---

## Per-backend setup

### Datadog

```bash
# Via the Datadog Agent's OTLP receiver (port 4318)
agent-strace export <session-id> --format otlp \
  --endpoint http://localhost:4318

# Via Datadog's OTLP intake directly
agent-strace export <session-id> --format otlp \
  --endpoint https://http-intake.logs.datadoghq.com:443 \
  --header "DD-API-KEY: $DD_API_KEY"
```

### Honeycomb

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://api.honeycomb.io \
  --header "x-honeycomb-team: $HONEYCOMB_API_KEY" \
  --service-name my-agent
```

### Grafana Tempo / Jaeger

```bash
# Local collector
agent-strace export <session-id> --format otlp \
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

### Langfuse

Sessions appear as Traces, tool calls as Spans, LLM calls as Generations, and eval scores as Langfuse Scores.

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

# Export latest session with eval scores
agent-strace export --scores --backend langfuse

# Export last 7 days
agent-strace export --since 7d --scores --backend langfuse

# Self-hosted Langfuse
agent-strace export --scores --backend langfuse \
  --langfuse-host https://langfuse.your-domain.com
```

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

Build a statistical baseline from recent sessions and alert when a new session deviates:

```bash
# Build baseline from the last 50 sessions
agent-strace baseline update --sessions 50

# Check a session against the baseline
agent-strace baseline check <session-id>

# Show baseline statistics
agent-strace baseline show
```

Metrics tracked: cost, duration, tool call count, error rate, retry rate, blast radius. Each metric gets a mean and standard deviation. A session is flagged when any metric exceeds `--threshold` standard deviations from the mean (default: 2.0).

```bash
agent-strace baseline check <session-id> --threshold 2.5 --format json
```

Use `baseline check` as a CI gate — it exits 1 when the session is anomalous:

```bash
agent-strace baseline check $SESSION_ID || echo "Session outside baseline"
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

---

## Inspect the OTLP payload

```bash
agent-strace export <session-id> --format otlp > trace.json
```
