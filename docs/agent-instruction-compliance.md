# AGENTS.md and CLAUDE.md instruction compliance

AgentTrace can track instruction-file versions and compare recorded session
proxies across versions. It does not currently prove that an agent read,
understood, or followed every instruction.

## Current tools

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

## Active benchmark work

An instruction-compliance benchmark is in active development. It is not part of
the current release. Until it lands, do not describe `context-score` or
`config-watch` as benchmark-backed instruction adherence tests.

## Interpretation limits

- A file hash proves only which bytes were snapshotted.
- AgentTrace does not know whether the provider loaded or read the file.
- Natural-language instructions are not converted into a complete executable
  policy.
- `context-score` is a before-and-after comparison without task-mix controls or
  confidence intervals.
- Lower cost or fewer tool calls can score better even when the agent skipped
  required work.
- The default cost proxy can be estimated from event size instead of recorded
  token usage.
- Correlation after an instruction change does not establish causation.

For machine-readable checks of captured file, command, or network activity,
use a structured `.agent-scope.json` policy with `agent-strace audit`. Do not
pass an arbitrary Markdown instruction file as the audit policy.

## Related documentation

- [Context score command](commands.md#context-score)
- [Config watch command](commands.md#config-watch)
- [Policy audit](commands.md#audit)
- [Existing AGENTS.md integration notes](agents-md-integration.md)
