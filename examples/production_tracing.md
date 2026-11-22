# Production tracing with agent-trace

agent-trace captures sessions locally as NDJSON. The OTLP exporter converts them
to OpenTelemetry spans and sends them to your existing observability stack.

## How it works

```
Claude Code / Cursor / any agent
        ↓
  agent-strace (hooks or proxy)
        ↓
  .agent-traces/ (local NDJSON)
        ↓
  agent-strace export --format otlp
        ↓
  OTLP collector (Datadog, Honeycomb, New Relic, Splunk, Grafana Tempo)
```

Each session becomes an OpenTelemetry trace. Each tool call becomes a span with
duration, input arguments, and output. Errors become spans with error status and
exception events. User prompts and assistant responses become events on the root span.

## Datadog

Datadog accepts OTLP traces via the Datadog Agent or directly via their intake API.

### Via Datadog Agent

The Datadog Agent has a built-in OTLP receiver on port 4318. Enable it in
`datadog.yaml`:

```yaml
otlp_config:
  receiver:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
```

Then export:

```bash
agent-strace export <session-id> --format otlp \
  --endpoint http://localhost:4318 \
  --service-name my-coding-agent
```

### Direct intake

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://http-intake.logs.datadoghq.com:443 \
  --header "DD-API-KEY: $DD_API_KEY" \
  --service-name my-coding-agent
```

Traces appear in Datadog APM under the service name you specify.

## Honeycomb

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://api.honeycomb.io \
  --header "x-honeycomb-team: $HONEYCOMB_API_KEY" \
  --service-name my-coding-agent
```

Honeycomb natively supports OTLP. Traces appear in the dataset matching your
service name. Each tool call is a span you can query, filter, and visualize.

## New Relic

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://otlp.nr-data.net \
  --header "api-key: $NEW_RELIC_LICENSE_KEY" \
  --service-name my-coding-agent
```

For EU region, use `https://otlp.eu01.nr-data.net`.

Traces appear in New Relic's distributed tracing view.

## Splunk Observability Cloud

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://ingest.<realm>.signalfx.com \
  --header "X-SF-Token: $SPLUNK_ACCESS_TOKEN" \
  --service-name my-coding-agent
```

Replace `<realm>` with your Splunk realm (e.g., `us0`, `us1`, `eu0`).

## Grafana Tempo

```bash
# Via a local OpenTelemetry Collector forwarding to Tempo
agent-strace export <session-id> --format otlp \
  --endpoint http://localhost:4318
```

Or send directly to Grafana Cloud. `tr -d '\n'` removes the line breaks GNU
`base64` adds to long output, because a header value must be one line:

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://otlp-gateway-<zone>.grafana.net/otlp \
  --header "Authorization: Basic $(printf '%s' "$GRAFANA_INSTANCE_ID:$GRAFANA_API_KEY" | base64 | tr -d '\n')"
```

## Jaeger

```bash
# Jaeger with OTLP receiver enabled
agent-strace export <session-id> --format otlp \
  --endpoint http://localhost:4318
```

## Automation

Export every session automatically after Claude Code finishes by adding a
`SessionEnd` hook:

```json
{
  "hooks": {
    "SessionEnd": [{
      "hooks": [{
        "type": "command",
        "command": "agent-strace hook session-end && agent-strace export $(cat .agent-traces/.active-session 2>/dev/null || echo latest) --format otlp --endpoint http://localhost:4318"
      }]
    }]
  }
}
```

## Inspecting the OTLP payload

To see what agent-trace sends without hitting a collector, write the payload
to a file. Without `--endpoint`, export prints nothing on stdout.

```bash
agent-strace export <session-id> --format otlp --output trace.json
jq . trace.json
```

Without `--output`, export writes `trace-<12-character session ID>-otlp.json`.

## What you see in your dashboard

A typical agent session trace looks like:

```
agent-session (claude-code)                    [1m52s]
├── Glob                                       [51ms]
├── Glob                                       [51ms]
├── Bash ($ python -m pytest tests/ -v)        [21.6s]  ERROR
├── Bash ($ python3 -m pytest tests/ -v)       [10.7s]  ERROR
├── Bash ($ which pytest)                      [51ms]
├── Read (/pyproject.toml)                     [43ms]
├── Bash ($ pip install -e ".[dev]")           [12.1s]
└── Bash ($ uv run --with pytest pytest)       [5.88s]

Events on root span:
  user_prompt: "how many tests does this project have?"
  assistant_response: "75 tests, all passing in 3.60s."
```

Each span has attributes for tool inputs and outputs. Errors have exception
events with the failure message. You can filter, query, and alert on these
using your existing observability tools.
