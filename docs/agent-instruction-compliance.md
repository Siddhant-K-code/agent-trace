# AGENTS.md and CLAUDE.md instruction compliance

AgentTrace can track instruction-file versions and compare recorded session
proxies across versions. It also ships an offline instruction compliance
benchmark built on `skills eval`. These surfaces do not prove that an agent
understood every instruction or that an instruction caused an outcome.

## Track instruction versions

Record a baseline snapshot and detect later changes:

```bash
agent-strace config-watch snapshot --label before-change
agent-strace config-watch check
agent-strace config-watch affected --since 7d
```

Compare recorded efficiency proxies across instruction-file versions:

```bash
agent-strace context-score --file AGENTS.md --history 30 --compare
agent-strace context-score --file CLAUDE.md --history 30 --compare --format json
```

`config-watch` hashes watched files, including `AGENTS.md`, `CLAUDE.md`, and
`.github/copilot-instructions.md`. `context-score` groups sessions by snapshots
and compares cost estimates, tool counts, lint findings, and scope-violation
proxies.

## Run the instruction compliance benchmark

The packaged public corpus runs offline and makes no provider request:

```bash
agent-strace skills eval examples/instruction-compliance/cases.json \
  --from-traces examples/instruction-compliance/traces \
  --workspace examples/instruction-compliance/fixture \
  --format markdown
```

Cases can name expected `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, or
`.github/copilot-instructions.md` files and expected file, tool, or test
actions. The report keeps `observed`, `inferred`, `missing`, and `unavailable`
evidence separate. Availability without a recorded read is inferred, not proof
that the provider loaded the file.

The benchmark reuses the skill load and follow analysis. It produces evidence
counts, validation facts, and warnings, not a composite score, grade, or
verdict. See the [instruction compliance benchmark](instruction-compliance.md)
for the case schema and report shape.

## Interpretation limits

- A file hash proves only which bytes were snapshotted.
- Unless a read or load event was recorded, AgentTrace does not know whether
  the provider loaded or read the file.
- Natural-language instructions are not converted into a complete executable
  policy.
- `context-score` is a before-and-after comparison without task-mix controls or
  confidence intervals.
- Lower cost or fewer tool calls can score better even when the agent skipped
  required work.
- The default cost proxy can be estimated from event size instead of recorded
  token usage.
- Post-load validation is associated by event order. It is not causal.
- A controlled should-not-trigger case applies only to its exact prompt and
  fixture.
- Correlation after an instruction change does not establish causation.

For machine-readable checks of captured file, command, or network activity,
use a structured `.agent-scope.json` policy with `agent-strace audit`. Do not
pass an arbitrary Markdown instruction file as the audit policy.

## Related documentation

- [Context score command](commands.md#context-score)
- [Config watch command](commands.md#config-watch)
- [Instruction compliance benchmark](instruction-compliance.md)
- [Skills evaluation command](commands.md#skills-eval)
- [Policy audit](commands.md#audit)
- [Existing AGENTS.md integration notes](agents-md-integration.md)
