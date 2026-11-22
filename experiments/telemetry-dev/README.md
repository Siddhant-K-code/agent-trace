# Telemetry.dev OTLP preflight (#241)

This directory holds the Telemetry.dev experiment in [#241](https://github.com/Siddhant-K-code/agent-trace/issues/241). Nothing here is packaged, imported by `src/`, or run by the test suite. The user-facing guide is [docs/telemetry-dev.md](../../docs/telemetry-dev.md).

Prepared on 2026-11-22 with agent-strace 0.120.0. [MAPPING.md](MAPPING.md) holds the doc-based predictions, written before any send. The owner sent the three payloads on 2026-11-22T17:16:23Z (run `20261122T171623Z`, all HTTP 200), and [RESULTS.md](RESULTS.md) records what Telemetry.dev showed: **PARTIAL, bounded**. The committed payloads are the agent-strace 0.120.0 bytes that run sent, before `send.py` gave them new IDs and times.

## Send it (owner)

From the repository root, with a Telemetry.dev key that has **Send telemetry** access:

```sh
read -rs TELEMETRY_DEV_API_KEY && export TELEMETRY_DEV_API_KEY
python3 experiments/telemetry-dev/send.py
unset TELEMETRY_DEV_API_KEY
```

`read -rs` waits for you to paste the key without showing it. `send.py` posts each of the three payloads to `https://api.telemetry.dev/v1/traces` through agent-strace's own exporter, and prints each HTTP status and trace ID. Then fill in [CHECKLIST.md](CHECKLIST.md) from the Telemetry.dev UI.

- `--only payload-otlp-genai.json` sends one payload; repeat the flag for more.
- `--environment NAME` adds `deployment.environment.name` to the resource. Leave it out for a key limited to one environment.
- Exit status: 0 if every payload got a 2xx, 1 if one did not, 2 if the key or an argument is wrong.

The key comes only from the environment. `send.py` refuses to run without it, refuses a key that does not start with `td_live_`, and never prints it: the exporter's output is scrubbed of the key before it is shown. It refuses plain `http` except to a loopback address. The bytes it sent, their sha256 and the ID map go to `sent/<run id>/`, which git ignores and which holds no key.

`send.py` gives every span new IDs and moves the timestamps to the send time. Telemetry.dev hides data older than the project's retention, and treats a body it stored in the last hour as a duplicate. Nothing else in the payload changes.

The CLI can send the same thing, but the key then sits in the command line, where `ps` and shell history can see it:

```sh
agent-strace export <session> --format otlp-genai \
  --endpoint https://api.telemetry.dev --header "Authorization: Bearer $TELEMETRY_DEV_API_KEY"
```

## Files

| File | What it is |
| --- | --- |
| [MAPPING.md](MAPPING.md) | The predicted mapping of every attribute, and the predicted PARTIAL decision with its top risks |
| [CHECKLIST.md](CHECKLIST.md) | What to check in the Telemetry.dev UI, by #241 acceptance criterion, with Predicted and Observed columns |
| [RESULTS.md](RESULTS.md) | The observed results of the real send, the decision, and every difference from the predictions |
| `send.py` | The owner's send script |
| `generate.py` | Builds the synthetic sessions and writes the three payloads |
| `validate.py` | Offline check of the payloads against Telemetry.dev's documented OTLP JSON rules, and of MAPPING.md's coverage |
| `payload-otlp.json` | `agent-strace export --format otlp`: parent and child sessions, 2 traces, 7 spans |
| `payload-otlp-genai.json` | `agent-strace export --format otlp-genai`: parent and child sessions, 2 traces, 10 spans |
| `payload-otlp-genai-tree.json` | The library function `session_to_otlp_genai` with a parent span: 1 trace, 10 spans, the child under the parent's `Task` span |

## The synthetic session

The parent session starts from the corpus fixture `src/agent_trace/corpus/v1/fixture-review-001`, which has a tool failure and retry, a redacted prompt and an orphan tool result as a capture gap. `generate.py` adds a provider and model, a failed model call and its retry, tokens with cache reads and writes, a cost, provenance, and a `Task` call that starts a child subagent session. The fixture itself is not changed. All values are synthetic: `synthetic-user`, `synthetic-host`, `/workspace/project`.

## Regenerate and check

```sh
python3 experiments/telemetry-dev/generate.py
python3 experiments/telemetry-dev/validate.py
```

`generate.py` is deterministic, so the committed payloads should not change unless the exporter does. `validate.py` prints `OK` when the payloads follow the documented rules and MAPPING.md names every key they send.
