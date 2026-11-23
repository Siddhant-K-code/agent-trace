# GenAI semantic-conventions conformance

This directory runs OpenTelemetry's GenAI conformance check against
`agent-strace export --format otlp-genai`. The check is the
[semantic-conventions-conformance](https://github.com/open-telemetry/semantic-conventions-conformance)
runner, which sends each scenario's spans to
[Weaver](https://github.com/open-telemetry/weaver) live-check and checks them
against a GenAI registry. [RESULTS.md](RESULTS.md) has the findings, before and
after #563.

It is not part of the package or its tests, and it is left out of the sdist
with the rest of `experiments/`. Nothing here is an agent-strace dependency.

## Pinned versions

| Piece | Version |
|---|---|
| Conformance runner | `open-telemetry/semantic-conventions-conformance` at `65781818cf534d30836c890f642fc8b820e4033a` |
| Weaver | `v0.28.0` release binary |
| GenAI registry | `open-telemetry/semantic-conventions-genai` tag `v0.1.0` (`d0e74ba6c48bcc4ded473b1828a8212f06b23753`, 2026-11-21), and commit `e6658f14c9d9ea0760825f029cb82e7673f48fab` (2026-11-06) for the comparison |
| Python | 3.12 |
| SDKs, `sdk` scenario only | `openai` 3.57.0, `anthropic` 2.12.0 |

The runner pulled in `grpcio` 1.85.3, `opentelemetry-proto` 1.45.2 and
`protobuf` 7.36.2.

SHA-256 of the Weaver `v0.28.0` archives:

```text
b334b37372325f77374b461fd6a11d38bc050122832eb83a9732fcd6831a5ab1  weaver-aarch64-apple-darwin.tar.xz
9878e2d56ad82ae441b0b527377a4f6001ae19f4af362fb0a4d951884f59ab61  weaver-x86_64-unknown-linux-gnu.tar.xz
```

## Set up

`$WORK` is any scratch directory outside the checkout. `$REPO` is this
agent-strace checkout. `$TARGET` is `aarch64-apple-darwin` or
`x86_64-unknown-linux-gnu`.

```bash
export WORK=$(mktemp -d) REPO=$(git rev-parse --show-toplevel)
export TARGET=x86_64-unknown-linux-gnu
cd "$WORK"

git clone https://github.com/open-telemetry/semantic-conventions-conformance
git -C semantic-conventions-conformance checkout 65781818cf534d30836c890f642fc8b820e4033a

git clone https://github.com/open-telemetry/semantic-conventions-genai genai-v0.1.0
git -C genai-v0.1.0 checkout v0.1.0
git clone https://github.com/open-telemetry/semantic-conventions-genai genai-e6658f1
git -C genai-e6658f1 checkout e6658f14c9d9ea0760825f029cb82e7673f48fab

mkdir weaver && cd weaver
gh release download v0.28.0 -R open-telemetry/weaver \
  -p "weaver-$TARGET.tar.xz" -p "weaver-$TARGET.tar.xz.sha256"
shasum -a 256 -c "weaver-$TARGET.tar.xz.sha256"
tar -xf "weaver-$TARGET.tar.xz"
cd "$WORK"

python3.12 -m venv venv
venv/bin/pip install \
  -e semantic-conventions-conformance/tools/runner \
  -e semantic-conventions-conformance/tools/gen-ai/mock-server \
  -e semantic-conventions-conformance/tools/gen-ai/runner \
  openai==3.57.0 anthropic==2.12.0

export PATH="$WORK/weaver/weaver-$TARGET:$WORK/venv/bin:$PATH"
weaver --version   # weaver 0.28.0
```

Compare the checksum with the one above too, not only with the `.sha256` file
from the same release.

## Run

From this directory, with `PATH` set as above:

```bash
cd "$REPO/experiments/semconv-conformance"
for ref in v0.1.0 e6658f1; do
  genai-conformance . --registry "$WORK/genai-$ref/model" --report-only \
    --report-dir "output/after-$ref" --data-file "output/after-$ref/data.json"
done
python3 summarize.py output/after-v0.1.0 output/after-e6658f1
```

The runner starts Weaver and the mock model server, runs each scenario, and
writes one Weaver report per scenario to `output/<run>/<scenario>.json`.
`--report-only` reports violations as warnings (`status: WARN`) instead of
failing; a scenario that crashes still fails the run. `output/` is
git-ignored.

To get the "before" numbers, run the same command on the commit before #563
(`git stash push -- src` on a working tree with the change, then
`git stash pop`), with `--report-dir output/before-$ref`.

### Negative check

```bash
cd negative
genai-conformance . --registry "$WORK/genai-v0.1.0/model" --report-only \
  --report-dir output/negative-v0.1.0 --data-file output/negative-v0.1.0/data.json
python3 ../summarize.py output/negative-v0.1.0
```

It must report `genai_content_schema` on each `invoke_agent` root. If it
does not, the check is not reading the content attributes, and a clean result
from the main run means nothing.

### Read a payload without the runner

```bash
mkdir -p output && python3 scenario.py tree --output output/tree.json
```

`session`, `tree` and `invalid-content` need nothing but this checkout. `sdk`
needs the mock server and the SDKs: start `genai-mock-server --port 8090` and
pass `--base-url http://127.0.0.1:8090`.

## Scenarios

[`conformance.yaml`](conformance.yaml) declares them, and
[`scenario.py`](scenario.py) builds each payload.

- `session`: the synthetic parent and subagent sessions that
  [`experiments/telemetry-dev/generate.py`](../telemetry-dev/generate.py)
  builds, each exported with `--format otlp-genai`, sent in one request as two
  traces. They hold a failed model request and its retry, a tool error and its
  retry (both `retry_of` span links), a redacted argument, and a `Task` call
  that starts the subagent.
- `tree`: the same parent exported with `--include-subagents`, so the
  subagent is in the same trace under the `Task` span.
- `sdk`: [`sdk_calls.py`](sdk_calls.py) turns on `instrument_openai` and
  `instrument_anthropic` and makes real SDK calls to the runner's mock model
  server: OpenAI Chat Completions, streamed and not, the Responses API,
  Anthropic Messages, streamed and not, and a Chat Completions call the server
  fails with HTTP 400. The session they record is exported the same way. This
  is the only scenario whose capture path records a response ID, a response
  model and a server.
- `invalid-content` (in [`negative/`](negative/conformance.yaml)): the
  `session` payload with each `gen_ai.input.messages` replaced by a message
  that has no `parts`.

agent-strace runs from this checkout's `src` in a clean environment (no
`AGENT_STRACE_*`, `AGENT_TRACE_*` or `OTEL_*`, a fresh `HOME`, telemetry off),
so the payload does not depend on local settings.

### OTLP/gRPC shim

agent-strace sends OTLP over HTTP only, as JSON or, with
`--protocol http/protobuf`, protobuf; it has no gRPC exporter
([ADR-0006](../../ADRs/0006-otlp-http-json-no-grpc.md)). Weaver accepts only
OTLP/gRPC, so `scenario.py` converts the JSON payload: the hex trace and span
IDs become base64, `google.protobuf.json_format` parses the
JSON into an `ExportTraceServiceRequest`, and it is sent to
`$OTEL_EXPORTER_OTLP_ENDPOINT` over insecure gRPC. Before sending, it checks
that every span link survived the conversion with the same span, trace and
linked span IDs, and exits 1 if one did not. The runner's reports then show
the links Weaver received: 2 in `session` and in `tree`.

## Reading the results

[`summarize.py`](summarize.py) counts Weaver's advice per span from the
per-scenario reports. The runner's own `data.json` merges findings by span
name across scenarios, so a span that a fix changed does not show there while
another span with the same name still has the finding.
