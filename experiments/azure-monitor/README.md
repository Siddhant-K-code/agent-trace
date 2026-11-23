# Azure Monitor exporter check (#586)

This directory holds the local check for [#586](https://github.com/Siddhant-K-code/agent-trace/issues/586), which adds `--agent-id` to `export` and `watch --stream-otlp`. Microsoft Foundry matches the traces of a registered external agent by the `gen_ai.agent.id` span attribute it finds in Application Insights. Azure Monitor's coding-agent page sends OTLP through an OpenTelemetry Collector with the `azuremonitorexporter`. This check runs that exporter locally and records what it would send to Application Insights for agent-strace traces. Nothing here is packaged, imported by `src/`, or run by the test suite. The user-facing recipe is [Azure Monitor and Microsoft Foundry](../../docs/integrations.md#azure-monitor-and-microsoft-foundry).

**Neither Application Insights nor Foundry was tested.** No Azure account was used. The exporter's connection string has the all-zero fake instrumentation key, and its `IngestionEndpoint` is a capture server on 127.0.0.1 inside `run.py`. The report shows the envelopes the exporter posts. It does not show what Application Insights stores, how its `customDimensions` look in a query, or whether Foundry shows the traces under the registered agent.

Run on 2026-11-23 with the #586 branch (agent-strace 0.121.1). The sessions are the synthetic ones from [#241](../telemetry-dev/README.md#the-synthetic-session), plus two `run.py` builds itself. [RESULTS.md](RESULTS.md) records what the exporter sent, and [report.json](report.json) is the full output of `run.py` from that run.

## Versions

| Tool | Version |
| --- | --- |
| otelcol-contrib | 0.165.0, release `otelcol-contrib_0.165.0_darwin_arm64.tar.gz` (sha256 `f9a0ed310e9beeb0fabca9d9c24bd0d3467f6339e3207710dbd66a6adf1298f6`, the release's `.sha256` asset); the binary's sha256 is `b19bcb7c192c77126ba530fed7bf361183d125b8a9decdec265cbb68b856cc2b` |
| Python running agent-strace and `run.py` | 3.14.8 |
| OS | macOS 27.2, arm64 |

The exporter's behaviour below was also read in its source at v0.165.0 (`exporter/azuremonitorexporter` in opentelemetry-collector-contrib) and in the `microsoft/ApplicationInsights-Go` v0.4.4 contracts it uses.

## Run it

1. Download the pinned collector into a temporary directory, not the repository. Pick the asset for your machine; this one is macOS arm64.

   ```sh
   OTELCOL_DIR="$(mktemp -d)"
   cd "$OTELCOL_DIR"
   curl -fsSLO https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.165.0/otelcol-contrib_0.165.0_darwin_arm64.tar.gz
   echo "f9a0ed310e9beeb0fabca9d9c24bd0d3467f6339e3207710dbd66a6adf1298f6  otelcol-contrib_0.165.0_darwin_arm64.tar.gz" | shasum -a 256 -c
   tar -xzf otelcol-contrib_0.165.0_darwin_arm64.tar.gz otelcol-contrib
   cd -
   ```

2. From the repository root:

   ```sh
   env -i PATH="$PATH" HOME="$HOME" python3 experiments/azure-monitor/run.py \
     --otelcol "$OTELCOL_DIR/otelcol-contrib" > experiments/azure-monitor/report.json
   ```

3. Delete `$OTELCOL_DIR`.

`run.py` takes about 15 seconds. It starts the capture server and the collector with [collector.yaml](collector.yaml) on free loopback ports, writes every session to a temporary trace store, and runs the real CLI (`python -m agent_trace.entry`) with a temporary `HOME`, telemetry off and no `AGENT_STRACE_*` or `OTEL_*` variable except the one a step sets. It prints a JSON report. The report names temporary paths `<tmpdir>`, the collector's receiver `<otlp-receiver>`, and the capture server `<capture>`. The collector's warnings keep only the level, message and component, not its times or its random instance ID. The collector runs with `TZ=UTC`, because the exporter writes envelope times in the local time zone.

## What run.py sends

The collector has one traces pipeline: the OTLP/HTTP receiver, then two exporters. `azure_monitor` has the defaults. `azure_monitor/spanevents` also has `spaneventsenabled: true`. Each posts to its own path on the capture server, so the report can compare them. Both have `maxbatchinterval: 1s`, so the run doesn't wait for the default 10 seconds.

| Group | Sessions | How |
| --- | --- | --- |
| `am586-genai` | parent with its child | `export --format otlp-genai --include-subagents --agent-id demo-agent-v1 --endpoint … --protocol http/protobuf` |
| `am586-otlp` | parent with its child | `export --format otlp --include-subagents` with `AGENT_STRACE_AGENT_ID=demo-agent-v1` instead of the flag |
| `am586-long` | ten turns | `export --format otlp-genai --agent-id demo-agent-v1`. Each prompt is 1100 `é`, so `gen_ai.input.messages` (prompts cut to 1000 characters, `é` escaped as `\u00e9`) is about 70 KB; one `Bash` call has a 5000-character command |
| `am586-noid` | parent with its child | `export --format otlp-genai --include-subagents`, no agent ID |
| `am586-watch` | parent | `watch --stream-to <otlp-receiver> --stream-otlp --protocol http/protobuf --agent-id demo-agent-v1 --stream-flush-interval 1`, while the events are appended |

The synthetic parent has a system prompt, two LLM requests (the first fails with an overload and is retried), a failed `bash` test run and its retry, a `read_file`, and a `Task` call that starts the child session. Each retry carries a span link to the failed attempt: the model request and the `bash` run in `otlp-genai`, the `bash` run in `otlp`.

For each export, `run.py` also writes the same export with `--output` and compares every span in it with the exporter's envelope for that span, matched by trace ID (`ai.operation.id`) and span ID (`baseData.id`): the telemetry type, the span kind, each attribute (kept, changed or dropped), the properties the exporter adds, span links and span events. It counts `gen_ai.agent.id` on the span envelopes and on the exception and message envelopes made from span events.
