# Phoenix OTLP/HTTP protobuf check (#560)

This directory holds the local Phoenix check for [#560](https://github.com/Siddhant-K-code/agent-trace/issues/560): Phoenix's `/v1/traces` takes only `application/x-protobuf`, so `export --endpoint` and `watch --stream-otlp`, which sent only OTLP JSON, got HTTP 415. Nothing here is packaged, imported by `src/`, or run by the test suite. The user-facing guide is [Phoenix / OpenInference](../../docs/production.md#phoenix--openinference).

Run on 2026-11-23 with the #560 branch on top of `main` at `45eb496` (agent-strace 0.121.1). No key or account is needed: Phoenix runs locally, and the sessions are the synthetic ones from [#241](../telemetry-dev/README.md#the-synthetic-session). [RESULTS.md](RESULTS.md) records what Phoenix stored, and [report.json](report.json) is the full output of `run.py` from that run.

## Versions

| Tool | Version |
| --- | --- |
| Phoenix (`arize-phoenix`, run with `uvx`) | 22.0.1, on its own Python 3.13.15 |
| uv | 0.12.3 |
| Python running agent-strace and `run.py` | 3.14.8 |
| OS | macOS 27.2, arm64 |

## Run it

1. Start Phoenix with an empty working directory. It took about 90 seconds to start. Port 6016 keeps it away from a Phoenix you may run on the default 6006; `PHOENIX_GRPC_PORT` does the same for gRPC, which this check does not use.

   ```sh
   PHOENIX_DIR="$(mktemp -d)"
   env -i PATH="$PATH" HOME="$HOME" PHOENIX_WORKING_DIR="$PHOENIX_DIR" \
     PHOENIX_PORT=6016 PHOENIX_GRPC_PORT=4327 \
     PHOENIX_TELEMETRY_ENABLED=false DO_NOT_TRACK=1 \
     uvx --from arize-phoenix phoenix serve
   ```

2. In another shell, from the repository root, wait for Phoenix, then run the check:

   ```sh
   until curl -fsS http://127.0.0.1:6016/healthz >/dev/null; do sleep 5; done
   curl -s http://127.0.0.1:6016/arize_phoenix_version
   env -i PATH="$PATH" HOME="$HOME" python3 experiments/phoenix/run.py > experiments/phoenix/report.json
   ```

3. Stop Phoenix with Ctrl-C and delete `$PHOENIX_DIR`.

`run.py` takes about a minute. It writes every session to a temporary trace store, runs the real CLI (`python -m agent_trace.entry`) with a temporary `HOME`, telemetry off and no `AGENT_STRACE_*` or `OTEL_*` variable except the one a step sets, and prints a JSON report. It refuses a Phoenix URL that is not a loopback address. The report names temporary paths `<tmpdir>` and the relay `RELAY_URL`, and holds no Phoenix internal IDs.

Run it against a fresh Phoenix. Phoenix keeps the first span it stores with a given span ID and ignores later copies, and the span IDs are deterministic, so a second run against the same working directory compares with the first run's spans.

## What run.py sends

| Group | Sessions | How |
| --- | --- | --- |
| `px560-genai` | parent, child | `export --format otlp-genai --endpoint` first with the default JSON (expect 415), then with `--protocol http/protobuf`; then `--protocol grpc` (expect a refusal) |
| `px560-otlp` | parent, child | `export --format otlp --endpoint` with `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` |
| `px560-tree` | parent with its child | `export --format otlp-genai --include-subagents --endpoint --protocol http/protobuf` |
| `px560-watch` | parent | `watch --stream-to RELAY_URL --stream-otlp --protocol http/protobuf --stream-flush-interval 1`, while the events are appended |
| `px560-watchjson` | parent | The same `watch` with the default JSON (expect 415) |

`watch` posts to a relay inside `run.py`, which records each request and forwards it unchanged to Phoenix, so the report can say what each POST carried. For each sent span, `run.py` reads the payload the CLI builds (`--output`) and compares it with Phoenix's REST span API (`/v1/projects/default/spans`): name, trace ID, parent, start and end to the microsecond (Phoenix rounds nanoseconds), status code and message, attributes and event attributes. Phoenix's GraphQL API gives the span kind, token counts, cost, input and output, and the sessions.
