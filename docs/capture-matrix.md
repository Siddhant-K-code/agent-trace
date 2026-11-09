# CLI capture matrix

AgentTrace can only report events exposed by a provider's hook surface. This matrix keeps those boundaries explicit instead of treating every adapter as equivalent.

The machine-readable source is [`src/agent_trace/capture_matrix.json`](../src/agent_trace/capture_matrix.json). It ships inside the package so [`agent-strace health`](commands.md#health) can read the same entries this page describes. An entry marked **unverified** is documentation of the current adapter contract, not proof that a specific provider release was exercised in a clean environment. An entry marked **verified** names the provider release that was exercised; other releases may behave differently.

## Verify one local capture

After recording one local Claude Code or Codex session, run:

```bash
agent-strace setup --cli claude --verify --live
agent-strace setup --cli codex --verify --live
```

The command first verifies the provider config and AgentTrace executable. It
then selects the newest matching recorded session, or the session passed with
`--session SESSION_ID`, and applies this packaged matrix through the same
evidence-health assessment used by `agent-strace health`. Its known-gap list
comes directly from the selected provider entry. The verifier does not infer a
gap from a missing event and does not start a provider or model request.

Local live verification and matrix `verification_status` answer different
questions. Live verification proves that this installation recorded one
reviewable session. Matrix status records whether the repository contains a
versioned, reproducible provider fixture for the adapter contract.

| Provider | Setup | Session | Prompt/response | Tools | File/command/error | Verification |
|---|---|---|---|---|---|---|
| Claude Code | `agent-strace setup --cli claude` | start + end | captured | captured | provider-defined | unverified |
| OpenAI Codex | `agent-strace setup --cli codex` | start + end | captured | captured | provider-defined | verified, 0.167.0-alpha.3.2 |
| GitHub Copilot CLI | `agent-strace setup --cli copilot` | start + end, first prompt can arrive before start | prompt only | captured | provider-defined; errors captured | verified, 1.0.103-0 |
| GitHub Copilot cloud agent | `agent-strace setup --cli copilot-cloud` | **partial**: start + end per job, only when streamed to a collector | prompt only | captured | provider-defined; errors captured | unverified |

## Claude Code

- Config: `~/.claude/settings.json`
- Setup only **prints** Claude hook JSON: manually copy or merge the generated `hooks` into `~/.claude/settings.json`; it does not write the file.
- Rollback: remove only the AgentTrace command hooks you manually merged into that file, retaining unrelated hooks.
- Privacy: captured fields still pass through AgentTrace redaction, but AgentTrace cannot expose fields Claude Code never emits.
- Known gap: file and command semantics depend on the hook-visible tool payload.
- Matrix verification: none recorded yet. A run against Claude Code `2.1.222` on 2026-11-09 was not performed because no Claude Code credentials were available in the verification environment, so the entry stays unverified.
- Local verification: accept workspace trust, record one small session, exit it
  cleanly, then run `agent-strace setup --cli claude --verify --live`.

## OpenAI Codex

- Config: `$CODEX_HOME/hooks.json (default ~/.codex/hooks.json)`
- Backup: before overwriting an existing config, setup preserves its original bytes once at `hooks.json.agent-strace.bak` in the same directory. Repeated setup runs do not overwrite that pre-AgentTrace snapshot.
- Rollback: run `cp ~/.codex/hooks.json.agent-strace.bak ~/.codex/hooks.json`, replacing `~/.codex` with `"$CODEX_HOME"` when it is set. If no config existed before setup, no backup is created; remove the generated `hooks.json` instead.
- Coverage: session start and end, user prompts, assistant responses, `PreToolUse`/`PostToolUse` tool activity, `Stop` (recorded as a stop event), `SubagentStart`/`SubagentStop` (recorded as `subagent_start`/`subagent_stop` events in the parent trace) and `PostModelRequest` (one `llm_response` event with token usage per model request).
- Verified with `codex-cli 0.167.0-alpha.3.2` on 2026-11-09: a tool call, a failing shell command and a spawned subagent, each through `SessionEnd`. `codex-cli 0.163.0-alpha.11` was verified on 2026-10-22 and 2026-10-24; that recording is kept as a legacy fixture. Each run used `codex exec` in a throwaway workspace, an isolated `CODEX_HOME`, a read-only sandbox and no approval-bypass flags, with the hooks trusted beforehand. The standalone `codex-cli 0.143.0` available on the same machine could not complete a turn, so it was not verified.
- Hook trust: Codex runs the generated command hooks only after they are reviewed and trusted in `/hooks`. `codex exec` skips untrusted hooks without a warning, so an untrusted install records nothing. Codex trusts each hook separately, so hooks that a newer setup adds or changes start untrusted while the others stay trusted. `setup --cli codex` names the hooks that are new or changed since the previous `hooks.json`. `0.167.0-alpha.3.2` also offers the review when the interactive CLI starts with untrusted hooks.
- Upgrading: installs made before agent-strace registered `SessionEnd`, `SubagentStart`, `SubagentStop` and `PostModelRequest` do not have them. Re-run `agent-strace setup --cli codex`, then trust the new hooks in `/hooks`. Until setup is re-run, `setup --cli codex --verify` reports them as missing. `--verify` cannot see trust: until the new hooks are trusted, `codex exec` skips them, and `setup --cli codex --verify --live` reports a session with no model requests. Sessions recorded without `SessionEnd` keep their active-session markers, so `health` reports them as `session_active`.
- Session end: `SessionEnd` fires once after the last `Stop` of a `codex exec` run and records a `session_end` event, so `health` can tell a finished session from an active one. A resumed run (`codex exec resume`) records a `session_start` with source `resume` and continues the same trace. Codex gives `SessionEnd` hooks a 1 second timeout by default and caps it at 3 seconds; setup sets 3.
- Subagents: a subagent is not a separate session. Its hooks arrive under the parent `session_id`, so everything is recorded in the parent trace:
  - `SubagentStart` and `SubagentStop` are recorded as `subagent_start` and `subagent_stop` events with `agent_id`, `agent_type` and `turn_id`. No child session is created or linked, and these events have no `child_session_id` or `child_link`.
  - `subagent_stop` also records the subagent's final message (`last_assistant_message`) as `text` and its transcript as `agent_transcript_path`, both redacted like the `Stop` fields. The event is recorded even when `stop_hook_active` is true. Session summaries and exports take the session's final answer from response events, never from `subagent_stop`.
  - The subagent's tool calls are recorded as `tool_call`/`tool_result` events in the parent trace with the subagent's `agent_id` and `agent_type`. Parent tool events carry neither. [`replay`](commands.md#replay), `replay --tree`, [`tree`](commands.md#tree), [`stats`](commands.md#stats) and [`explain`](commands.md#explain) mark these events with `[subagent <agent_type> <last 8 characters of agent_id>]` or list the subagent with its tool count.
  - Event order observed: the parent's `spawn_agent` tool call and result, `SubagentStart`, the parent's `wait_agent` tool call, the subagent's tool calls, `SubagentStop`, the `wait_agent` result, `Stop`, then `SessionEnd`. In one `0.163.0-alpha.11` recording `SubagentStart` fired after the `wait_agent` call instead, so do not rely on that position.
  - A subagent's model requests are recorded as `llm_response` events with its `agent_id` and `agent_type`.
  - Payload fields observed. `SubagentStart`: `session_id` (parent), `agent_id`, `agent_type`, `turn_id` (the subagent's turn), `transcript_path`, `cwd`, `model`, `permission_mode`. `SubagentStop`: the same plus `agent_transcript_path`, `last_assistant_message` and `stop_hook_active`. `SessionEnd`: `session_id`, `reason`, `transcript_path`, `cwd`. Subagent `PreToolUse`/`PostToolUse` payloads also carry `agent_id` and `agent_type`. The `spawn_agent` `tool_input` has `task_name` and an encrypted `message`; `0.167.0-alpha.3.2` adds `fork_turns`. Payload keys of the registered hooks were otherwise unchanged from `0.163.0-alpha.11`.
- Model requests: `PostModelRequest` fires after every model request, a subagent's included, on Codex `0.164.0` and later. Older releases ignore the hook in `hooks.json`. Each request is recorded as one `llm_response` event:
  - Fields: `model`, `status` (`completed`, `failed` or `cancelled`), `error` when Codex sends one, `turn_id` and `token_usage` as sent, plus `input_tokens`, `output_tokens` and `total_tokens` copied from it. A subagent's request also has its `agent_id` and `agent_type`.
  - `token_usage` has `input_tokens`, `cached_input_tokens`, `cache_write_input_tokens`, `output_tokens`, `reasoning_output_tokens` and `total_tokens`. `input_tokens` already includes the cached and cache-write tokens, `output_tokens` already includes the reasoning tokens, and `total_tokens` is `input_tokens + output_tokens`. Do not add the parts to the totals.
  - The session's `total_tokens` is the sum of `total_tokens` over its requests. Its `llm_requests` stays 0, because it counts `llm_request` events and this hook records none; `validate` and `backfill` rely on that.
  - A failed request has the error text and no `token_usage`; a cancelled request has neither. Neither is recorded as an `error` event.
  - The hook runs synchronously, so Codex waits for it before acting on each response.
- Hooks setup does not register: `/hooks` in `0.167.0-alpha.3.2` lists 14 hook events and setup registers 9. Recorder-only hooks in the verification runs showed:
  - `PreModelRequest` fired before each model request. It carries only `session_id`, `turn_id`, `cwd`, `transcript_path` and `model`, so setup does not register it: it would add a second process per request and no data beyond the request start time. As a result `llm_response` events have no `parent_id`, no `llm_request` event and no request duration. The entry declares this with `results_without_requests`, and `health` reports one `declared_results_without_requests` provider limitation instead of an orphan per response.
  - `PermissionRequest`, `PreCompact`, `PostCompact` and `Interrupt` did not fire in these runs.
- Known gaps:
  - `SessionEnd` always reports reason `other`, so the `session_end` event does not say why the session ended.
  - No event links a subagent to the `spawn_agent` call that started it (that result names only the task).
  - A failed shell command arrives as a normal `PostToolUse` with the error text in `tool_response` and no exit code, so it is recorded as a `tool_result`, not an `error`.
  - Token usage is recorded only on Codex `0.164.0` and later, and only once `PostModelRequest` is trusted. `llm_response` events have no request event, `parent_id` or duration.
  - Shell calls arrive as tool `Bash` with `tool_input.command`; file and command detail depends on those payloads.
- Local verification: trust every AgentTrace command in `/hooks`, record one
  small session, exit it cleanly, then run
  `agent-strace setup --cli codex --verify --live`.

## GitHub Copilot CLI

- Config: `~/.copilot/hooks/agent-strace.json`, or `$COPILOT_HOME/hooks/agent-strace.json` when `COPILOT_HOME` is set. The [agent-strace Copilot CLI plugin](copilot-plugin.md) registers the same hooks; use one or the other.
- Rollback: setup writes a dedicated file and leaves other Copilot hook files unchanged. Remove it with `rm ~/.copilot/hooks/agent-strace.json`. If you installed the plugin instead, run `copilot plugin uninstall agent-strace`.
- Coverage: session start and end, user prompts, tool calls and results, failed tools (`postToolUseFailure`, recorded as `error` events), `agentStop` (recorded as a stop event) and the parent's `subagentStart`/`subagentStop` hooks (recorded as `subagent_start`/`subagent_stop` events). Response text is unavailable.
- Verified with Copilot CLI `1.0.103-0` on 2026-10-22: hook payloads for a tool call, a failing tool, a tool denied at the permission prompt and a `task` subagent were recorded in `copilot -p`, and the `task` subagent run was repeated end to end through both the plugin hooks and `agent-strace setup --cli copilot`. Each run used a throwaway workspace, an isolated temporary `COPILOT_HOME`, a `GH_TOKEN` environment variable and no approve-all flags. Interactive mode was last verified with `1.0.94-3` on 2026-10-21; that recording is kept as a legacy fixture. (`copilot --version` prints `1.0.103`; the session log records `1.0.103-0`.)
- Event order: Copilot can fire `userPromptSubmitted` shortly before `sessionStart`, so a main session's trace may begin with its first `user_prompt`. `1.0.94-3` always did; `1.0.103-0` did in some runs and fired `sessionStart` first in others. The entry declares this with `events_before_start`, so `health` reports it as the provider limitation `declared_events_before_session_start` and `validate` as an `info` finding, instead of an invalid boundary.
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

## GitHub Copilot cloud agent

Coverage is **partial**, and the entry is **unverified**: no live cloud agent job has been recorded. See [Copilot cloud agent capture](copilot-cloud-agent.md) for setup and the GitHub documentation it relies on.

- Config: `.github/hooks/agent-strace.json` plus `.github/hooks/agent-strace-cloud.sh`, on the repository's default branch. The cloud agent loads only `.github/hooks/*.json` from the cloned repository; local Copilot plugins and `~/.copilot` hooks do not run there.
- Rollback: delete both files from the default branch, then remove the `AGENT_STRACE_ENDPOINT` and `AGENT_STRACE_AUTH_KEY` Agents secrets and the collector firewall allow rule.
- Transport: the sandbox filesystem is discarded when the job ends. Events survive only when each hook streams them to an [`agent-strace server`](server.md) collector named by the `AGENT_STRACE_ENDPOINT` Agents secret and allowed by the cloud agent firewall.
- Privacy: by default events are redacted before they leave the sandbox. Agents secrets are readable by the agent, so the collector key must guard nothing else.
- Captured, when events reach the collector (`captured_signals`):
  - Session start, once per job (sessionStart)
  - The job prompt, at most once per job (userPromptSubmitted)
  - Tool calls with tool name and arguments (preToolUse)
  - Tool results (postToolUse)
  - Failed tool calls (postToolUseFailure)
  - Agent stop reason (agentStop)
  - Subagent start and stop (subagentStart, subagentStop)
  - Session end with reason complete, error, or timeout (sessionEnd)
- Not captured (`not_captured_signals`):
  - Assistant response text and reasoning
  - Model name and token usage
  - errorOccurred, userPromptTransformed, and preCompact events, which fire in cloud agent but are not registered
  - permissionRequest and notification events, which do not fire in cloud agent
  - Setup steps, firewall blocks, and other job log output that does not pass through a hook
  - Any event recorded while AGENT_STRACE_ENDPOINT is unset or the collector is unreachable; it is discarded with the sandbox
  - Every event of a job in which agent-strace is not on PATH and the one-time install fails
  - Links between sessions of separate jobs on the same pull request; each job is recorded as its own session
  - Workflow artifact or pull request attachment upload
- Known gaps: the Copilot CLI gaps for `agentStop` text, model and token usage, tool call IDs, and file and command activity also apply. Recorded sessions name the `github-copilot` agent, so `health` and evidence reports apply the GitHub Copilot CLI entry rather than this one.

## Evidence from other tools

The matrix covers what agent-strace hooks capture. Evidence other tools
recorded is read separately and reported as
[external evidence](evidence-health.md#external-evidence), never as capture
coverage: Entire checkpoints and their push state, Copilot cloud agent
`Agent-Logs-Url` trailers, and Claude Code OpenTelemetry `safety_event`
records. Pass `--commits RANGE` or `--otel-logs FILE` to `health` or
`review`.

## Machine-readable coverage values

The values below are copied from `src/agent_trace/capture_matrix.json`. The fixture test
checks every provider/capability pair so these examples cannot drift from the
published matrix.

| Provider | session_start | session_end | prompt | response | tool | file | command | error | stop | verification_status |
|---|---|---|---|---|---|---|---|---|---|---|
| claude | captured | captured | captured | captured | captured | provider_defined | provider_defined | provider_defined | captured | unverified |
| codex | captured | captured | captured | captured | captured | provider_defined | provider_defined | provider_defined | captured | verified |
| copilot | captured | captured | captured | unavailable | captured | provider_defined | provider_defined | captured | captured | verified |
| copilot-cloud | captured | captured | captured | unavailable | captured | provider_defined | provider_defined | captured | captured | unverified |

An entry with incomplete coverage may also set `coverage_status` to `partial` and list `captured_signals` and `not_captured_signals`: plain-language lists of what the template records and what it cannot. They are documentation for readers and are not read by `health` or `validate`. `copilot-cloud` is the only partial entry.

An entry may also list `events_before_start`: event types the provider is known to send before its session start hook. `agent-strace health` reports events of those types before the first `session_start` as the provider limitation `declared_events_before_session_start`. Any other event before the start is still the observed `events_before_session_start`, which makes the session `invalid`.

An entry may also list `results_without_requests`: result event types the provider records with no request event. `agent-strace health` and the dashboard's evidence review report result events of those types with no `parent_id` once, as the provider limitation `declared_results_without_requests`, instead of one orphan each. A result that names a missing request is still an orphan. Codex lists `llm_response`.

`agent-strace validate` reads only these two fields of the entry, for the provider named by hook-written events. It reports each declared gap as one `info` finding and a label of the same name, `declared_events_before_session_start` or `declared_results_without_requests`, which do not change the status: a Copilot trace that begins with its first `user_prompt` and a Codex trace with model requests stay `valid`. An undeclared event type before the start is still an `events_before_session_start` error. A trace with no hook provider, such as an SDK session or an import, gets no exemption, and neither does any trace when the capture matrix cannot be loaded. `validate` does not read the known gaps.

## Verification policy

A provider entry may move to `verified` only when `tested_version`, `last_verified`, and `fixture_test_reference` are all populated from a reproducible verification run. Keeping unknown values explicit prevents stale compatibility claims from looking current.

A verification run uses the provider's real CLI with agent-strace hooks, a throwaway workspace, a temporary trace directory, a harmless prompt and no flags that approve every tool. `tested_version` is the provider's `--version` output and `last_verified` is the date of the run. The hook payloads it records are sanitized into [`tests/fixtures/capture_matrix/`](../tests/fixtures/capture_matrix/): every value that could identify a person, machine, repository or prompt is replaced with a synthetic one, keeping the payload shape and event order. The `fixture_test_reference`, [`tests/test_capture_matrix_fixtures.py`](../tests/test_capture_matrix_fixtures.py), replays each fixture through the hook handlers and checks the resulting trace against the entry's coverage and known gaps. To re-verify a provider, repeat the run, refresh its fixture and update `tested_version` and `last_verified`.

## Freshness

A verified entry is **stale** when its `last_verified` date is more than 30 days before the check date. The threshold is a date, not a release count, because both providers ship faster than verification runs can follow: GitHub Copilot CLI publishes a stable release every one to three days and Codex a stable minor about once a week, with prereleases in between, so any release-count rule would flag every entry within days. Hook payloads still changed within a month in both providers (`subagentStart` fields and event order between Copilot CLI `1.0.94-3` and `1.0.103-0`, the `fork_turns` input between Codex `0.163.0-alpha.11` and `0.167.0-alpha.3.2`), so a verification older than a month should not keep looking current. The same threshold applies to every provider. Unverified entries have no date to age; they are listed for information and never fail the check.

The [`Capture matrix freshness`](../.github/workflows/capture-matrix-freshness.yml) workflow runs the check every Monday and can be started manually from the Actions tab. It fails when a verified entry is stale or its `last_verified` is not a valid `YYYY-MM-DD` date, and writes the report to the run summary. It does not run on pushes or pull requests, so an aging entry never fails an unrelated change. The [release runbook](releases.md#release-order) includes the same check.

Run it locally from the repository root:

```bash
python scripts/check_capture_matrix_freshness.py           # report only, exits 0
python scripts/check_capture_matrix_freshness.py --check   # exits 1 on a stale or malformed entry
python scripts/check_capture_matrix_freshness.py --format json --today 2026-12-01 --max-age-days 45
```

`--today` defaults to the current UTC date and `--max-age-days` to 30. A `last_verified` date one day after the check date counts as verified that day, which absorbs the difference between a maintainer's local date and UTC; a later date is malformed. Exit status 2 means invalid arguments or an unreadable matrix. The script is repository tooling and is not part of the installed package.

To refresh a stale entry:

1. Repeat the verification run described in [Verification policy](#verification-policy) with the provider's current release.
2. Refresh the provider's fixtures in `tests/fixtures/capture_matrix/`, and update its coverage and known gaps if behavior changed.
3. Set `tested_version` and `last_verified` in `src/agent_trace/capture_matrix.json`, and update the provider's row and "Verified with" text on this page.
4. Run `python -m unittest discover -s tests -p 'test_capture_matrix*.py' -v` and the freshness check.

Do not change `last_verified` without a verification run. If a provider cannot be re-verified, leave its entry stale so the report keeps showing it.
