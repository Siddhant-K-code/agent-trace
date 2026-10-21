# CLI capture matrix

AgentTrace can only report events exposed by a provider's hook surface. This matrix keeps those boundaries explicit instead of treating every adapter as equivalent.

The machine-readable source is [`src/agent_trace/capture_matrix.json`](../src/agent_trace/capture_matrix.json). It ships inside the package so [`agent-strace health`](commands.md#health) can read the same entries this page describes. An entry marked **unverified** is documentation of the current adapter contract, not proof that a specific provider release was exercised in a clean environment.

| Provider | Setup | Session | Prompt/response | Tools | File/command/error | Verification |
|---|---|---|---|---|---|---|
| Claude Code | `agent-strace setup --cli claude` | start + end | captured | captured | provider-defined | unverified |
| OpenAI Codex | `agent-strace setup --cli codex` | start only | captured | captured | provider-defined | unverified |

## Claude Code

- Config: `~/.claude/settings.json`
- Setup only **prints** Claude hook JSON: manually copy or merge the generated `hooks` into `~/.claude/settings.json`; it does not write the file.
- Rollback: remove only the AgentTrace command hooks you manually merged into that file, retaining unrelated hooks.
- Privacy: captured fields still pass through AgentTrace redaction, but AgentTrace cannot expose fields Claude Code never emits.
- Known gap: file and command semantics depend on the hook-visible tool payload.

## OpenAI Codex

- Config: `~/.codex/hooks.json`
- Backup: before overwriting an existing config, setup preserves its original bytes once at `~/.codex/hooks.json.agent-strace.bak`. Repeated setup runs do not overwrite that pre-AgentTrace snapshot.
- Rollback: run `cp ~/.codex/hooks.json.agent-strace.bak ~/.codex/hooks.json`. If no config existed before setup, no backup is created; run `rm ~/.codex/hooks.json` instead.
- Current documented coverage: session start, user prompts, assistant responses, `PreToolUse`/`PostToolUse` tool activity, and `Stop` (captured as a stop event, **not** as session end).
- Known gap: no separate Codex session-end event is claimed; file, command, and error detail depends on provider payloads.

## Machine-readable coverage values

The values below are copied from `src/agent_trace/capture_matrix.json`. The fixture test
checks every provider/capability pair so these examples cannot drift from the
published matrix.

| Provider | session_start | session_end | prompt | response | tool | file | command | error | stop | verification_status |
|---|---|---|---|---|---|---|---|---|---|---|
| claude | captured | captured | captured | captured | captured | provider_defined | provider_defined | provider_defined | captured | unverified |
| codex | captured | unavailable | captured | captured | captured | provider_defined | provider_defined | provider_defined | captured | unverified |

## Verification policy

A provider entry may move to `verified` only when `tested_version`, `last_verified`, and `fixture_test_reference` are all populated from a reproducible verification run. Keeping unknown values explicit prevents stale compatibility claims from looking current.
