# Synthetic reliability examples

Run from the repository root with Python 3.10 or later:

```bash
python examples/reliability/run.py
```

No install, model, account, or network access is needed. The runner uses this
checkout's validator, read-only `TraceStore`, and review projection. It reads
the [shipped corpus](../../src/agent_trace/corpus/v1/), does not write traces,
and does not execute recorded commands or send telemetry.

## What to look for

| Case | Recorded evidence | Expected result |
|---|---|---|
| `complete-current` | Start, linked read result, end | `valid`, `healthy` |
| `resumed-current` | One session, two starts, two ends, linked test results | `valid`, `healthy` |
| `resumed-open` | An earlier end, a resume, successful test results, no final end | `partial`, `incomplete`, `missing_session_end` |
| `incomplete-missing-outcome` | End marked completed, but a tool call has no result | `partial`, `incomplete`, `unpaired_tool_call` |
| `terminal-absent` | Start and prompt, no end | `partial`, `incomplete`, `missing_session_end` |

The resumed case lasts 3,610 seconds from its original start to its final
end. Its end events record cumulative durations of 5 and 3,610 seconds,
following the hook writer's behavior. Do not add them together or count
the resume as another session. The gap between the first end and the resume
is not recorded active work. Even the bounded spans do not measure active
work time.

`resumed-open` retains the first end and its old duration. Successful tool
results and a positive assistant response do not supply the missing final
boundary. The validator must still report incomplete evidence.

## Inspect a case directly

Use the checkout's CLI so you inspect the same code as the runner:

```bash
DO_NOT_TRACK=1 PYTHONPATH=src python -m agent_trace.entry validate src/agent_trace/corpus/v1/resumed-open
```

This command exits `1` because the evidence is partial. The example
runner exits `0` after displaying all cases, including their gaps. Its exit
code is not a completion or quality gate.

## Limits

These are hand-authored synthetic records under the repository's MIT license,
not recordings of a live provider. `healthy` describes recorded structure,
not task success, provider coverage, safety, or production reliability.
No provider capture matrix is claimed. The pack uses the published v1 trace
schema, not a CLI or provider version-compatibility matrix. The resume cases
omit hash links, a supported legacy representation, so they do not prove
integrity. No case contains a private trace, private path, or real identity.
