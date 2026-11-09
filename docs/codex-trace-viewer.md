# OpenAI Codex trace viewer and verified capture

AgentTrace records Codex command-hook events into a local trace store, then
replays them in the terminal or a self-contained HTML viewer.

## Verification is version-specific

The [capture matrix](capture-matrix.md#openai-codex) is the source of truth for
the Codex version and date last exercised with a real provider run. A
`verified` entry applies to that tested release and fixture, not every Codex
release.

`agent-strace setup --cli codex --verify` performs a static installation check.
It validates the user-level hooks file, the Codex hooks feature setting, and
the local AgentTrace executable. After one harmless local run,
`--verify --live` adds a check of the newest matching recorded session. It does
not start Codex or make a model request.

## Install and check hooks

```bash
agent-strace setup --cli codex
agent-strace setup --cli codex --verify
agent-strace setup --cli codex --verify --live
```

Codex must trust each generated command hook in its Hooks UI. Headless
`codex exec` can skip untrusted hooks without a warning. Re-run setup after
upgrading from an older AgentTrace configuration so newer session-end and
subagent hooks are present. Live verification reports observed event types,
boundaries, redaction, evidence health, and packaged capture-matrix gaps. It
returns non-zero when matching reviewable evidence is unavailable.

## View a trace

```bash
agent-strace list
agent-strace health SESSION_ID
agent-strace replay SESSION_ID
agent-strace replay SESSION_ID --format html -o codex-session.html
agent-strace tree SESSION_ID
agent-strace stats SESSION_ID
```

Codex subagent events and their tool calls are stored in the parent session.
Replay and statistics label them by the provider-supplied agent identity.

## Known capture gaps

- Hidden reasoning is not available.
- Token usage is not present in the observed hook payloads.
- A failed shell command can arrive as a normal tool result rather than an
  error event.
- No recorded event directly links a subagent identity to the `spawn_agent`
  tool call that started it.
- File and command detail depends on provider tool payloads.
- Provider upgrades can change event shape or timing.

See the [capture matrix](capture-matrix.md#openai-codex) for the complete
current list.

## Related documentation

- [Codex setup](setup.md#openai-codex-hooks)
- [Capture matrix](capture-matrix.md#openai-codex)
- [Evidence health](evidence-health.md)
- [Replay command reference](commands.md#replay)
