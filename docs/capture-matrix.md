# CLI capture matrix

AgentTrace can only report events exposed by a provider's hook surface. This matrix keeps those boundaries explicit instead of treating every adapter as equivalent.

The machine-readable source is [`src/agent_trace/capture_matrix.json`](../src/agent_trace/capture_matrix.json). It ships inside the package so [`agent-strace health`](commands.md#health) can read the same entries this page describes. An entry marked **unverified** is documentation of the current adapter contract, not proof that a specific provider release was exercised in a clean environment. An entry marked **verified** names the provider release that was exercised; other releases may behave differently.

| Provider | Setup | Session | Prompt/response | Tools | File/command/error | Verification |
|---|---|---|---|---|---|---|
| Claude Code | `agent-strace setup --cli claude` | start + end | captured | captured | provider-defined | unverified |
| OpenAI Codex | `agent-strace setup --cli codex` | start + end | captured | captured | provider-defined | verified, 0.163.0-alpha.11 |
| GitHub Copilot CLI | `agent-strace setup --cli copilot` | start + end, first prompt can arrive before start | prompt only | captured | provider-defined; errors captured | verified, 1.0.103-0 |

## Claude Code

- Config: `~/.claude/settings.json`
- Setup only **prints** Claude hook JSON: manually copy or merge the generated `hooks` into `~/.claude/settings.json`; it does not write the file.
- Rollback: remove only the AgentTrace command hooks you manually merged into that file, retaining unrelated hooks.
- Privacy: captured fields still pass through AgentTrace redaction, but AgentTrace cannot expose fields Claude Code never emits.
- Known gap: file and command semantics depend on the hook-visible tool payload.

## OpenAI Codex

- Config: `$CODEX_HOME/hooks.json (default ~/.codex/hooks.json)`
- Backup: before overwriting an existing config, setup preserves its original bytes once at `hooks.json.agent-strace.bak` in the same directory. Repeated setup runs do not overwrite that pre-AgentTrace snapshot.
- Rollback: run `cp ~/.codex/hooks.json.agent-strace.bak ~/.codex/hooks.json`, replacing `~/.codex` with `"$CODEX_HOME"` when it is set. If no config existed before setup, no backup is created; remove the generated `hooks.json` instead.
- Coverage: session start and end, user prompts, assistant responses, `PreToolUse`/`PostToolUse` tool activity, `Stop` (recorded as a stop event) and `SubagentStart`/`SubagentStop` (recorded as `subagent_start`/`subagent_stop` events in the parent trace).
- Verified with `codex-cli 0.163.0-alpha.11`: a tool call, a failing shell command and a spawned subagent on 2026-10-22, and `SessionEnd`, `SubagentStart` and `SubagentStop` with a spawned subagent on 2026-10-24. Each run used `codex exec` in a throwaway workspace, an isolated `CODEX_HOME`, a read-only sandbox and no approval-bypass flags. The standalone `codex-cli 0.143.0` available on the same machine could not complete a turn, so it was not verified.
- Hook trust: Codex runs the generated command hooks only after they are reviewed and trusted in `/hooks`. `codex exec` skips untrusted hooks without a warning, so an untrusted install records nothing. Codex trusts each hook separately, so hooks added by a newer setup start untrusted.
- Upgrading: installs made before agent-strace registered `SessionEnd`, `SubagentStart` and `SubagentStop` do not have them. Re-run `agent-strace setup --cli codex`, then trust the new hooks in `/hooks`. Until then, `setup --cli codex --verify` reports them as missing. Sessions recorded without `SessionEnd` keep their active-session markers, so `health` reports them as `session_active`.
- Session end: `SessionEnd` fires once after the last `Stop` of a `codex exec` run and records a `session_end` event, so `health` can tell a finished session from an active one. A resumed run (`codex exec resume`) records a `session_start` with source `resume` and continues the same trace. Codex gives `SessionEnd` hooks a 1 second timeout by default and caps it at 3 seconds; setup sets 3.
- Subagents: a subagent is not a separate session. Its hooks arrive under the parent `session_id`, so everything is recorded in the parent trace:
  - `SubagentStart` and `SubagentStop` are recorded as `subagent_start` and `subagent_stop` events with `agent_id`, `agent_type` and `turn_id`. No child session is created or linked, and these events have no `child_session_id` or `child_link`.
  - The subagent's tool calls are recorded as ordinary `tool_call`/`tool_result` events in the parent trace.
  - Event order observed: the parent's `spawn_agent` tool call and result, `SubagentStart`, the parent's `wait_agent` tool call, the subagent's tool calls, `SubagentStop`, the `wait_agent` result, `Stop`, then `SessionEnd`. In one recording `SubagentStart` fired after the `wait_agent` call instead, so do not rely on that position.
  - Payload fields observed. `SubagentStart`: `session_id` (parent), `agent_id`, `agent_type`, `turn_id` (the subagent's turn), `transcript_path`, `cwd`, `model`, `permission_mode`. `SubagentStop`: the same plus `agent_transcript_path`, `last_assistant_message` and `stop_hook_active`. `SessionEnd`: `session_id`, `reason`, `transcript_path`, `cwd`. Subagent `PreToolUse`/`PostToolUse` payloads also carry `agent_id` and `agent_type`.
- Known gaps:
  - `SessionEnd` always reports reason `other`, so the `session_end` event does not say why the session ended.
  - The subagent's tool calls are recorded without its `agent_id`, no event links a subagent to the `spawn_agent` call that started it (that result names only the task), and the subagent's final message (`last_assistant_message`) is not recorded.
  - A failed shell command arrives as a normal `PostToolUse` with the error text in `tool_response`, so it is recorded as a `tool_result`, not an `error`.
  - Hook payloads carry the model name, recorded on the `session_start` event, but no token usage.
  - Shell calls arrive as tool `Bash` with `tool_input.command`; file and command detail depends on those payloads.

## GitHub Copilot CLI

- Config: `~/.copilot/hooks/agent-strace.json`, or `$COPILOT_HOME/hooks/agent-strace.json` when `COPILOT_HOME` is set. The [agent-strace Copilot CLI plugin](copilot-plugin.md) registers the same hooks; use one or the other.
- Rollback: setup writes a dedicated file and leaves other Copilot hook files unchanged. Remove it with `rm ~/.copilot/hooks/agent-strace.json`. If you installed the plugin instead, run `copilot plugin uninstall agent-strace`.
- Coverage: session start and end, user prompts, tool calls and results, failed tools (`postToolUseFailure`, recorded as `error` events), `agentStop` (recorded as a stop event) and the parent's `subagentStart`/`subagentStop` hooks (recorded as `subagent_start`/`subagent_stop` events). Response text is unavailable.
- Verified with Copilot CLI `1.0.103-0` on 2026-10-22: hook payloads for a tool call, a failing tool, a tool denied at the permission prompt and a `task` subagent were recorded in `copilot -p`, and the `task` subagent run was repeated end to end through both the plugin hooks and `agent-strace setup --cli copilot`. Each run used a throwaway workspace, an isolated temporary `COPILOT_HOME`, a `GH_TOKEN` environment variable and no approve-all flags. Interactive mode was last verified with `1.0.94-3` on 2026-10-21; that recording is kept as a legacy fixture. (`copilot --version` prints `1.0.103`; the session log records `1.0.103-0`.)
- Event order: Copilot can fire `userPromptSubmitted` shortly before `sessionStart`, so a main session's trace may begin with its first `user_prompt`. `1.0.94-3` always did; `1.0.103-0` did in some runs and fired `sessionStart` first in others. The entry declares this with `events_before_start`, and `health` reports it as the provider limitation `declared_events_before_session_start` instead of an invalid boundary.
- Subagents: a subagent started with the `task` tool records prompt, tool and stop events under its own session ID, with no `session_start` and no `session_end`. Its metadata is written as `github-copilot (no session-start)`, which `health` maps to this entry; the session is `partial`. The parent fires `subagentStart` and `subagentStop`:
  - The parent's trace gets `subagent_start` and `subagent_stop` events with `agent_type`, `agent_name`, `stop_reason` (stop only) and, when the payload names the child, `child_session_id`. The `response` text is not recorded; the `task` tool result already carries it.
  - When the payload names the child, the child's `meta.json` gets `parent_session_id`, `depth` (parent depth + 1) and, when exactly one `task` call is pending in the parent, `parent_event_id` pointing at that `tool_call`. Otherwise `parent_event_id` stays empty. [`tree`](commands.md#tree), `replay --tree`, `replay --expand-subagents` and `stats --include-subagents` then show the child under its parent.
  - Only the child's metadata is changed, atomically under the store write lock; its hash-chained `events.ndjson` is never rewritten. A missing child is never created, an existing link to a different parent is kept, and erased, pending-deletion, pending-tag and cross-tenant children are skipped with the same checks as `backfill-meta`. The outcome is recorded as `child_link` on the parent event (`linked`, `already_linked`, `missing_child`, `different_parent`, `erased`, `tenant_mismatch`, ...). Later hook writes to the child's metadata, including a late `sessionStart`, keep the link.
  - Payload fields observed on `1.0.103-0`. `subagentStop`: `sessionId` (parent), `agentId` (child session ID), `agentType`, `agentName`, `response`, `stopReason`, `transcriptPath`, and in some runs `stop_hook_active` and `parentAgentId`. `subagentStart`: `sessionId`, `agentName` and `transcriptPath`; in some runs also `agentId`, `agentType` and `parentAgentId`. When `subagentStart` does not name the child, as in `1.0.94-3`, the link is made at `subagentStop`.
  - Existing installs do not have these hooks: re-run `agent-strace setup --cli copilot`, or update the plugin. Sessions captured before then stay unlinked; `backfill-meta` does not link them.
- Known gaps:
  - `agentStop` carries a stop reason and transcript path but no response text, so stop events have empty text.
  - Hook payloads carry no model name or token usage.
  - Tool hooks carry no call ID, so results are paired with calls by tool name, and a subagent's `parent_event_id` is set only when exactly one `task` call is pending.
  - A tool denied at the permission prompt leaves its `tool_call` without a result or error. The `permissionRequest` and `notification` hooks that report the denial are not registered, and neither are `preCompact` or `errorOccurred`.
  - Child hooks in some `1.0.103-0` runs carry `parentAgentId`; agent-strace links only from the parent's subagent hooks and does not use it.
  - File and command activity is visible only in tool arguments and results; no `file_write` events are recorded.
  - Stores written before agent-strace 0.97.0 have no metadata for subagent sessions. Run `agent-strace backfill-meta --apply` to make them visible.

## Machine-readable coverage values

The values below are copied from `src/agent_trace/capture_matrix.json`. The fixture test
checks every provider/capability pair so these examples cannot drift from the
published matrix.

| Provider | session_start | session_end | prompt | response | tool | file | command | error | stop | verification_status |
|---|---|---|---|---|---|---|---|---|---|---|
| claude | captured | captured | captured | captured | captured | provider_defined | provider_defined | provider_defined | captured | unverified |
| codex | captured | captured | captured | captured | captured | provider_defined | provider_defined | provider_defined | captured | verified |
| copilot | captured | captured | captured | unavailable | captured | provider_defined | provider_defined | captured | captured | verified |

An entry may also list `events_before_start`: event types the provider is known to send before its session start hook. `agent-strace health` reports events of those types before the first `session_start` as the provider limitation `declared_events_before_session_start`. Any other event before the start is still the observed `events_before_session_start`, which makes the session `invalid`.

## Verification policy

A provider entry may move to `verified` only when `tested_version`, `last_verified`, and `fixture_test_reference` are all populated from a reproducible verification run. Keeping unknown values explicit prevents stale compatibility claims from looking current.

A verification run uses the provider's real CLI with agent-strace hooks, a throwaway workspace, a temporary trace directory, a harmless prompt and no flags that approve every tool. `tested_version` is the provider's `--version` output and `last_verified` is the date of the run. The hook payloads it records are sanitized into [`tests/fixtures/capture_matrix/`](../tests/fixtures/capture_matrix/): every value that could identify a person, machine, repository or prompt is replaced with a synthetic one, keeping the payload shape and event order. The `fixture_test_reference`, [`tests/test_capture_matrix_fixtures.py`](../tests/test_capture_matrix_fixtures.py), replays each fixture through the hook handlers and checks the resulting trace against the entry's coverage and known gaps. To re-verify a provider, repeat the run, refresh its fixture and update `tested_version` and `last_verified`.
