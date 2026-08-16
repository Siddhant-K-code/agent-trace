# Context compaction analysis

Long agent sessions can exceed a model's working context. Runtimes often
compact the conversation into a shorter summary and continue, which can drop
constraints, decisions, and file context. agent-strace detects that boundary
from recorded input-token counts; it does not call an LLM or send trace data to
a service.

## Inspect a session

```bash
agent-strace compaction SESSION_ID
agent-strace compaction SESSION_ID --diff
agent-strace compaction SESSION_ID --behavior-diff
```

A compaction is recorded when consecutive requests in the same conversation
sidechain show an input-token drop greater than 50%. Change the threshold with
`--compaction-threshold 0.65`. Provider-reported usage is required; estimated
prompt sizes are never labeled or priced as compaction.

The base report shows each boundary, input tokens before and after, dropped
tokens, percentage, and estimated cost of the discarded input. `--diff`
compares reconstructable pre-compaction facts with the recorded summary and
marks likely losses. Constraints and decisions are high risk. The comparison
is deterministic keyword analysis, so treat “likely dropped” as a review cue,
not proof.

`--behavior-diff` compares bounded windows before and after the boundary. It
highlights file re-reads, repeated exploration, tool loops, context saturation,
and increases in the normal lint rule set. `agent-strace lint` exposes the same
signal as `post-compaction-regression` for CI use.

## Imported Claude sessions

Claude JSONL imports retain per-turn token usage, compaction summaries, and
sidechain identity. Import and inspect them offline:

```bash
agent-strace import ~/.claude/projects/example/session.jsonl
agent-strace compaction SESSION_ID --diff --behavior-diff
```

Independent sidechains or subagents are never compared with each other.

## Write a recovery checkpoint while watching

```bash
agent-strace watch SESSION_ID --compaction-checkpoint
agent-strace watch SESSION_ID --compaction-checkpoint --checkpoint-at 0.75
```

When provider usage reaches the configured fraction of the model context,
`watch` writes `.agent-traces/checkpoints/SESSION_ID.md`. The sidecar captures
the task, constraints, files read, decisions, modified files, and current
state. It is written once per growth cycle and rearms after compaction.

Checkpoint files can contain trace-derived project context and should receive
the same access controls as `.agent-traces/`. Retention cleanup removes a
session's checkpoint together with its event data. Checkpoints are additive;
the NDJSON storage format is not changed or rewritten.
