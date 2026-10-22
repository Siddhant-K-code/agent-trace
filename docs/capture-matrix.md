# CLI capture matrix

AgentTrace can only report events exposed by a provider's hook surface. This matrix keeps those boundaries explicit instead of treating every adapter as equivalent.

The machine-readable source is [`src/agent_trace/capture_matrix.json`](../src/agent_trace/capture_matrix.json). It ships inside the package so [`agent-strace health`](commands.md#health) can read the same entries this page describes. An entry marked **unverified** is documentation of the current adapter contract, not proof that a specific provider release was exercised in a clean environment. An entry marked **verified** names the provider release that was exercised; other releases may behave differently.

| Provider | Setup | Session | Prompt/response | Tools | File/command/error | Verification |
|---|---|---|---|---|---|---|
| Claude Code | `agent-strace setup --cli claude` | start + end | captured | captured | provider-defined | unverified |
| OpenAI Codex | `agent-strace setup --cli codex` | start only | captured | captured | provider-defined | verified, 0.163.0-alpha.11 |
| GitHub Copilot CLI | `agent-strace setup --cli copilot` | start + end, first prompt arrives before start | prompt only | captured | provider-defined; errors captured | verified, 1.0.94-3 |

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
- Coverage: session start, user prompts, assistant responses, `PreToolUse`/`PostToolUse` tool activity, and `Stop` (captured as a stop event, **not** as session end).
- Verified with `codex-cli 0.163.0-alpha.11` on 2026-10-22: a tool call, a failing shell command and a spawned subagent, run with `codex exec` in a throwaway workspace, an isolated `CODEX_HOME`, a read-only sandbox and no approval-bypass flags. The standalone `codex-cli 0.143.0` available on the same machine could not complete a turn, so it was not verified.
- Hook trust: Codex runs the generated command hooks only after they are reviewed and trusted in `/hooks`. `codex exec` skips untrusted hooks without a warning, so an untrusted install records nothing.
- Known gaps:
  - Codex emits `SessionEnd`, but setup does not register it, so traces have no `session_end` event and `health` never knows whether a Codex session has finished.
  - Subagent hooks arrive under the parent `session_id` with an `agent_id`, so subagent tool calls are recorded in the parent trace. `SubagentStart` and `SubagentStop` are not registered.
  - A failed shell command arrives as a normal `PostToolUse` with the error text in `tool_response`, so it is recorded as a `tool_result`, not an `error`.
  - Hook payloads carry the model name, recorded on the `session_start` event, but no token usage.
  - Shell calls arrive as tool `Bash` with `tool_input.command`; file and command detail depends on those payloads.

## GitHub Copilot CLI

- Config: `~/.copilot/hooks/agent-strace.json`, or `$COPILOT_HOME/hooks/agent-strace.json` when `COPILOT_HOME` is set. The [agent-strace Copilot CLI plugin](copilot-plugin.md) registers the same hooks; use one or the other.
- Rollback: setup writes a dedicated file and leaves other Copilot hook files unchanged. Remove it with `rm ~/.copilot/hooks/agent-strace.json`. If you installed the plugin instead, run `copilot plugin uninstall agent-strace`.
- Coverage: session start and end, user prompts, tool calls and results, failed tools (`postToolUseFailure`, recorded as `error` events) and `agentStop` (recorded as a stop event). Response text is unavailable.
- Verified with Copilot CLI `1.0.94-3` on 2026-10-21: a tool call, a failing tool, a tool denied at the permission prompt and a `task` subagent, through both the plugin hooks and `agent-strace setup --cli copilot`, in `copilot -p` and interactive mode. Each run used a throwaway workspace, an empty `COPILOT_HOME` and no approve-all flags.
- Event order: Copilot fires `userPromptSubmitted` about 0.15 to 0.2 seconds before `sessionStart`, so a main session's trace begins with its first `user_prompt`. The entry declares this with `events_before_start`, and `health` reports it as the provider limitation `declared_events_before_session_start` instead of an invalid boundary.
- Subagents: a subagent started with the `task` tool records prompt, tool and stop events under its own session ID, with no `session_start`, no `session_end` and no link to its parent. Its metadata is written as `github-copilot (no session-start)`, which `health` maps to this entry; the session is `partial`. Copilot's `subagentStop` hook names the child session on the parent, but setup does not register it.
- Known gaps:
  - `agentStop` carries a stop reason and transcript path but no response text, so stop events have empty text.
  - Hook payloads carry no model name or token usage.
  - Tool hooks carry no call ID, so results are paired with calls by tool name.
  - A tool denied at the permission prompt leaves its `tool_call` without a result or error. The `permissionRequest` and `notification` hooks that report the denial are not registered, and neither are `preCompact` or `errorOccurred`.
  - File and command activity is visible only in tool arguments and results; no `file_write` events are recorded.
  - Stores written before agent-strace 0.97.0 have no metadata for subagent sessions. Run `agent-strace backfill-meta --apply` to make them visible.

## Machine-readable coverage values

The values below are copied from `src/agent_trace/capture_matrix.json`. The fixture test
checks every provider/capability pair so these examples cannot drift from the
published matrix.

| Provider | session_start | session_end | prompt | response | tool | file | command | error | stop | verification_status |
|---|---|---|---|---|---|---|---|---|---|---|
| claude | captured | captured | captured | captured | captured | provider_defined | provider_defined | provider_defined | captured | unverified |
| codex | captured | unavailable | captured | captured | captured | provider_defined | provider_defined | provider_defined | captured | verified |
| copilot | captured | captured | captured | unavailable | captured | provider_defined | provider_defined | captured | captured | verified |

An entry may also list `events_before_start`: event types the provider is known to send before its session start hook. `agent-strace health` reports events of those types before the first `session_start` as the provider limitation `declared_events_before_session_start`. Any other event before the start is still the observed `events_before_session_start`, which makes the session `invalid`.

## Verification policy

A provider entry may move to `verified` only when `tested_version`, `last_verified`, and `fixture_test_reference` are all populated from a reproducible verification run. Keeping unknown values explicit prevents stale compatibility claims from looking current.

A verification run uses the provider's real CLI with agent-strace hooks, a throwaway workspace, a temporary trace directory, a harmless prompt and no flags that approve every tool. `tested_version` is the provider's `--version` output and `last_verified` is the date of the run. The hook payloads it records are sanitized into [`tests/fixtures/capture_matrix/`](../tests/fixtures/capture_matrix/): every value that could identify a person, machine, repository or prompt is replaced with a synthetic one, keeping the payload shape and event order. The `fixture_test_reference`, [`tests/test_capture_matrix_fixtures.py`](../tests/test_capture_matrix_fixtures.py), replays each fixture through the hook handlers and checks the resulting trace against the entry's coverage and known gaps. To re-verify a provider, repeat the run, refresh its fixture and update `tested_version` and `last_verified`.
