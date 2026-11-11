# Command reference

Full flag reference for every `agent-strace` command.

---

## Global options

```
agent-strace [--trace-dir DIR] <command> ...
```

| Flag | Default | Description |
|---|---|---|
| `--trace-dir DIR` | `$AGENT_TRACE_DIR`, else `.agent-traces` | Trace store that commands read and write |

The store is chosen in this order:

1. `--trace-dir DIR`, when given.
2. `$AGENT_TRACE_DIR`, when set to a non-empty value. Hooks and auto-instrumentation write to the same directory, so `list`, `replay`, `health`, `mcp` and the other commands see their sessions.
3. `.agent-traces`.

An empty `AGENT_TRACE_DIR` is treated as unset. Relative paths resolve against the current working directory.

---

## Session capture

### `record`
```
agent-strace record [--name NAME] [--parent SESSION] [--no-redact] -- <command>
```
Capture an MCP stdio server session. Wraps `<command>` as a transparent proxy.

| Flag | Description |
|---|---|
| `--name NAME` | Label for the session |
| `--parent SESSION` | Link this session as a child of a parent session |
| `--redact` | Strip secrets before writing to disk; kept for compatibility because this is now the default |
| `--no-redact` | Disable automatic secret redaction |

`AGENT_STRACE_PARENT_SESSION` can also be set by orchestrators that spawn child agents.

### `record-http`
```
agent-strace record-http --url URL [--port N] [--name NAME] [--parent SESSION] [--no-redact]
```
Capture an MCP HTTP/SSE server session. Listens on `--port` (default: 5100) and proxies to `--url`. Requests are handled concurrently, so an open `GET /sse` stream does not block `POST /message`. On Ctrl-C the proxy closes open streams, including TLS handshakes still in progress, waits up to 5 seconds for in-flight requests, then records the session end.

For an `https://` URL the proxy verifies the server certificate and host name with the system CA store, sends the URL host as SNI (none for an IP address) and in the `Host` header. To trust a private CA, set `SSL_CERT_FILE` to its PEM file. A TLS failure, such as an untrusted certificate or a host name mismatch, returns 502 to the agent and records an `error` event.

### `setup`
```
agent-strace setup [--cli claude|codex|gemini|cursor|copilot|copilot-cloud|all]
                    [--redact | --no-redact] [--global] [--verify]
                    [--live] [--session SESSION_ID]
                    [--managed [--executable PATH]]
```
Print or install hooks config for supported agent CLIs. `--cli claude` prints Claude Code settings JSON for `~/.claude/settings.json` (or `$CLAUDE_CONFIG_DIR/settings.json`); `--cli codex` writes `$CODEX_HOME/hooks.json` or `~/.codex/hooks.json`, the directory Codex reads (a set `CODEX_HOME` must already exist; `CODEX_CONFIG_DIR` is ignored), first preserving an existing file once as `hooks.json.agent-strace.bak`; `--cli gemini` writes a Gemini CLI extension under `$GEMINI_CLI_HOME/.gemini/extensions/agent-strace` or `~/.gemini/extensions/agent-strace`, the directory Gemini CLI loads (`GEMINI_CONFIG_DIR` is ignored); `--cli cursor` writes `.cursor/hooks.json` or `$CURSOR_CONFIG_DIR/hooks.json`; `--cli copilot` writes `$COPILOT_HOME/hooks/agent-strace.json` or `~/.copilot/hooks/agent-strace.json`, unless an enabled AgentTrace Copilot CLI plugin under `installed-plugins` already registers the hooks, in which case it writes nothing and prints the plugin path; `--cli copilot-cloud` writes `.github/hooks/agent-strace.json` and `.github/hooks/agent-strace-cloud.sh` at the root of the current git repository for Copilot cloud agent, prints the hooks JSON, and prints an optional `copilot-setup-steps.yml` without writing it (partial and unverified coverage; see [copilot-cloud-agent.md](copilot-cloud-agent.md)); `--cli all` configures all supported local CLIs and excludes `copilot-cloud`. Restore Codex with `cp ~/.codex/hooks.json.agent-strace.bak ~/.codex/hooks.json` (use `$CODEX_HOME` when set); repeated setup runs do not overwrite the backup. Secret redaction is enabled by default; use `--no-redact` only for trusted local traces.

| Flag | Default | Description |
|---|---|---|
| `--cli PROVIDER` | `claude` | Configure `claude`, `codex`, `gemini`, `cursor`, `copilot`, `copilot-cloud`, or `all` (`all` excludes `copilot-cloud`) |
| `--verify` | off | Read-only check of installed hooks and the AgentTrace CLI version; requires `--cli codex` or `--cli claude`; returns non-zero if incomplete or hooks are disabled; does not capture events. With `--cli codex`: **Codex user-level** hooks in `$CODEX_HOME` or `~/.codex` and the `config.toml` `[features]` hooks flag; also fails if a set `CODEX_HOME` is not an existing directory; skips Codex profiles, `-c` overrides, and project, system, or managed config layers. With `--cli claude`: **Claude Code user settings** hooks, plus hook-disabling settings and AgentTrace hooks in project, local, and managed settings (see [setup.md](setup.md)) |
| `--live` | off | With `--verify`, run the static setup check and then inspect the newest recorded session whose hook evidence matches the requested provider. Reports the session, capture method, event types, boundaries, redaction, evidence health, and packaged capture-matrix gaps. Returns non-zero when matching reviewable evidence is unavailable. Supports `claude` and `codex` only and never starts a provider process |
| `--session SESSION_ID` | newest matching | Select a session ID or prefix for `--verify --live` instead of automatic newest-provider selection |
| `--redact` | on | Explicitly enable secret redaction in generated hook commands |
| `--no-redact` | off | Disable automatic secret redaction in generated hook commands |
| `--global` | off | Compatibility flag; Claude setup already targets the user-level settings file |
| `--managed` | off | Print machine-wide hooks for Copilot CLI policy (`/etc/github-copilot/policy.d/agent-strace.json`, or `C:\ProgramData\GitHub\Copilot\policy.d\` on Windows) or a Claude Code managed settings drop-in (`managed-settings.d/agent-strace.json`), and print the target path on standard error. Writes nothing. Hooks run an absolute `agent-strace` path in exec form with `hook --metadata-only`. Requires `--cli copilot` or `--cli claude`; `--no-redact` is rejected. With `--verify`, read-only check of the installed managed files: root ownership and permissions (POSIX), one metadata-only hook per event, managed `disableAllHooks`, and the executable version; warns on user-level AgentTrace hooks and an enabled AgentTrace Copilot CLI plugin. `--live` works with `--cli claude` only. See [Platform-team install](platform-install.md) |
| `--executable PATH` | `agent-strace` on `PATH` | With `--managed`, the absolute path the hooks run |

Use the full live verification after recording one harmless local session:

```bash
agent-strace setup --cli claude --verify --live
agent-strace setup --cli codex --verify --live
```

The static configuration result and the recorded-evidence result are printed
separately. Missing or stale setup, unsupported providers, no matching session,
empty or unreadable evidence, provider mismatch, and `unknown` or `invalid`
evidence health return non-zero with a diagnostic. See
[live capture verification](setup.md#live-capture-verification).

### `hook`
```
agent-strace hook [--provider claude|codex|gemini|cursor|copilot] [--metadata-only] <event>
```
Internal hook receiver used by the configurations generated by `setup`. It
reads one provider hook payload as JSON from standard input and appends the
normalized event to the active session. `--provider` defaults to `claude`.

`--metadata-only` keeps event types, timing, IDs, tool names, model, reasons,
token counts, and the key names of tool input, and replaces every other string
value (prompts, responses, tool input and output, errors, paths, diffs) with
`[OMITTED]`. Secret redaction is always on, whatever `AGENT_TRACE_NO_REDACT`
says, and each event records `"capture": "metadata-only"`. `setup --managed`
uses it. See [Platform-team install](platform-install.md#metadata-only-capture).

| Event | Recorded data |
|---|---|
| `session-start` | Start the provider-scoped session, or continue it on `resume` or `compact` and keep its metadata. Records `redaction`: `on` or `off`, the secret-redaction mode the hook wrote with |
| `user-prompt` | User prompt text and available provider metadata |
| `pre-tool` | Tool name and input before execution, plus `agent_id` and `agent_type` when a Codex subagent made the call |
| `post-tool` | Successful tool result and duration when it can be linked to the call, plus `agent_id` and `agent_type` when a Codex subagent made the call |
| `post-tool-failure` | Failed tool result as an error event |
| `file-write` | Provider-reported file edit |
| `stop` | Assistant response or provider stop metadata |
| `subagent-start` | Subagent start in the parent session: agent type and name, plus the child session ID for GitHub Copilot CLI or the agent ID for Codex |
| `subagent-stop` | Subagent stop in the parent session, with the same fields and the stop reason when the provider sends one. For Codex, also the subagent's final message as `text` and its `agent_transcript_path`, both redacted |
| `session-end` | Finish the active session and clear its hook state |
| `post-model-request` | Codex only: one `llm_response` per model request with the model, status, error, turn ID and token usage, plus `agent_id` and `agent_type` for a subagent's request |

Provider-specific aliases in generated configurations are normalized to these
events. Prefer `agent-strace setup --cli PROVIDER` over invoking `hook`
manually. See [setup.md](setup.md) for provider event coverage and generated
configuration paths.

### `telemetry`
```
agent-strace telemetry [status|enable|disable]
```
Show or change the anonymous product telemetry preference. Telemetry is enabled
by default outside CI. `disable` persists an opt-out and deletes the local
anonymous installation ID. `DO_NOT_TRACK=1` or `AGENT_STRACE_TELEMETRY=0`
always disables collection; `AGENT_STRACE_TELEMETRY=1` explicitly enables it,
including in CI. See [telemetry.md](telemetry.md) for the exact event schema and
maintainer setup.

### `import`
```
agent-strace import <path.jsonl> [--discover]
```
Import a Claude Code JSONL session log. `--discover` lists available sessions in `~/.claude/projects/`. When the log records a working directory, the imported session starts with a `session_start` event holding that `cwd`.

Claude Code writes one log line per content block of a model response and
repeats the response's token usage on each. The import keeps that usage once,
on the response's last line, so token totals, request counts and
[`cost`](#cost) count each response once. Sessions imported by earlier versions
count the usage once per line; import them again to correct their totals.

---

## Replay and inspection

### `replay`
```
agent-strace replay [session-id] [--format terminal|html] [--live] [--speed N]
                    [--filter TYPES] [--limit N] [--expand-subagents] [--tree]
                    [-o FILE]
```

| Flag | Description |
|---|---|
| `--format html` | Export self-contained HTML viewer |
| `--live` | Replay with real-time delays |
| `--speed N` | Speed multiplier for `--live` (default: 1.0) |
| `--filter TYPES` | Comma-separated event types to show |
| `--limit N` | Cap at N events |
| `--expand-subagents` | Inline subagent sessions under the parent tool_call that started them (the child's `parent_event_id`) |
| `--tree` | Show session hierarchy without full replay |

Codex runs subagents inside the parent session. Replay tags each of their
tool events with `[subagent <agent_type> <last 8 characters of agent_id>]`,
shows the full `agent=<agent_id>` on `subagent_start`/`subagent_stop`, and
shows what the subagent returned under `subagent_stop`. `--tree` and
`--expand-subagents` use the same tags and list each in-session subagent with
its tool count under the session.

### `tree`
```
agent-strace tree [session-id] [--format text|json]
```
Show the parent/child session hierarchy for a root session, including per-node cost, tool calls, status, and duration. Parent links come from `record --parent`, `AGENT_STRACE_PARENT_SESSION`, A2A trace propagation, imported trace metadata, or GitHub Copilot CLI's `subagentStart`/`subagentStop` hooks, which link `task` subagent sessions to their parent. Copilot sessions captured before those hooks were registered stay unlinked; see the [capture matrix](capture-matrix.md#github-copilot-cli). Codex subagents run inside the parent session, so they are listed under it as `subagent <agent_type> <short agent_id>  N tools` lines, and `--format json` adds a `subagents` list to that node.

### `list`
```
agent-strace list [--tenant ID]
```
List all captured sessions with ID, timestamp, duration, tool calls, and errors.
`--tenant ID` performs an exact tenant match before listing metadata.

### `inspect`
```
agent-strace inspect <session-id>
```
Dump full session as JSON (meta + events).

### `stats`
```
agent-strace stats [session-id] [--include-subagents]
```
Tool call frequency and timing. When a Codex subagent made tool calls in the session, a Subagent Tool Calls section counts them per subagent. `--include-subagents` rolls up across the full subagent tree.

### `health`
```
agent-strace health [session-id] [--format text|json]
                    [--commits RANGE] [--otel-logs FILE]
```
Assess the [evidence health](evidence-health.md) of a session (default: the
latest). Text output starts with the status, the session, the provider and the
event count. Then it names the observed event types and the redaction state.
Then it lists observed reasons and provider limitations separately, and names
the capture-matrix entry that was applied and its `verification_status`.

```
Evidence health: partial
Session:   abc123def456ghij
Provider:  claude (from hook events)
Finalized: yes
Events:    6
Types:     session_start 1, session_end 1, tool_call 1, tool_result 1, user_prompt 1, assistant_response 1
Redaction: on (recorded by the session start hook)
Redacted:  1 of 6 events
```

| Status | Exit code |
|---|---|
| `healthy`, `partial` | 0 |
| `unknown`, `invalid` | 1 |

A missing session, an empty store or an unreadable session also exits 1.
`partial` passes because some providers can never be `healthy`: every entry
declares known gaps, so their sessions are at least `partial`.

| Flag | Description |
|---|---|
| `--format text\|json` | Output format (default: `text`) |
| `--commits RANGE` | Git revision range, such as `origin/main..HEAD`, to read `Entire-Checkpoint` and `Agent-Logs-Url` trailers from |
| `--otel-logs FILE` | Claude Code OpenTelemetry logs export (OTLP JSON or JSON Lines) to read `safety_event` from |

`--commits` and `--otel-logs` add [external evidence](evidence-health.md#external-evidence)
signals after the health result. Each signal has a source, a status
(`present`, `missing`, `unpushed` or `inconclusive`) and a confidence. They are
reported, not graded: they never change the status, the reasons or the exit
code. A range git cannot read is reported as an inconclusive signal. A range
that is not a revision expression exits 2.

```
External evidence (commits origin/main..HEAD):
  - Entire checkpoint 01JD4S7Y2V9Q8M3K5N6P7R8T0W: unpushed (high). Trailer present in 1 commit, not pushed: the checkpoint ref is local and waiting in Entire's push queue.
  - Copilot agent logs: missing (high). No Agent-Logs-Url trailer; 3 commits read.
  Reported, not graded: never changes a status or the exit code.
```

Each reason has a `kind`:

- `observed`: found in the recorded events, such as `missing_session_end`,
  `unpaired_tool_call` or `session_active`.
- `provider_limitation`: declared for the provider in the packaged
  [capture matrix](capture-matrix.md), never inferred from the trace. Every
  `known_gaps` entry is listed verbatim, followed by one line for each
  `coverage` capability marked `unavailable`. `provider_defined` coverage adds
  no line. When the entry lists `events_before_start` and only those event
  types precede the first session start, `declared_events_before_session_start`
  replaces the observed `events_before_session_start`. GitHub Copilot CLI
  declares `user_prompt`, because it sends the first prompt before its session
  start. When the entry lists `results_without_requests`, result events of
  those types with no `parent_id` are reported once as
  `declared_results_without_requests` instead of one orphan each. Codex
  declares `llm_response`.

The provider comes only from hook-written evidence: the `session_start` and
stop events the hooks record, or the metadata the hooks write when a session's
first hook was not SessionStart. Names set with `record --name`, the SDK or an
importer are not used. When no matrix entry applies, no provider limitations
are assessed and a note says why. This covers sessions recorded by the MCP
proxy, exec wrapper or SDK, imported sessions, providers without an entry
(Gemini CLI, Cursor) and sessions whose events name more than one provider.

Whether the session has finished:

- Finished when its metadata records an end time.
- Active when the matrix marks `session_end` as captured for the provider and a
  hook active-session marker still names the session. Active sessions report
  `session_active` instead of `missing_session_end`. A crashed session can
  leave its marker behind and keep reading as active.
- Otherwise not known, and a missing end marker is reported as observed.

Codex installs made before setup registered its `SessionEnd` hook never end a
session, so those sessions keep their markers and read as active. Re-run
`agent-strace setup --cli codex` and trust the new hooks in `/hooks`.

A resumed or compacted session is assessed as one session. See
[session boundaries](evidence-health.md#session-boundaries).

Event types and redaction are reported as facts. They never change the status,
the reasons or the exit code.

- Event types are counted per type, in a fixed order. Only types present in
  the trace are listed.
- The redaction mode comes from the hook-written `session_start` events. Each
  one records `redaction` as `on` or `off`, including resume and compact
  starts. The mode is `on` or `off` when every session start recorded the same
  value, `mixed` when they differ or only some recorded it, and `not_recorded`
  when none did. Sessions recorded before hooks wrote the field and sessions
  captured by the MCP proxy, exec wrapper, SDK or an importer read as
  `not_recorded`. When a session's first hook was not SessionStart, that
  implicit start counts as `not_recorded`, so a later resume or compact start
  makes the session `mixed`. A startup session start replaces that metadata,
  as GitHub Copilot CLI does after its first prompt. The mode is not recorded
  for each event.
- The redacted count is the number of events whose `redacted` flag is set,
  counted as `share --dry-run` counts them. Redaction sets the flag only when it
  changed something, so zero redacted events does not prove that redaction was
  on. Redacted session metadata is reported separately.

`--format json` prints the evidence-health result and the inputs supplied to it:

| Key | Contents |
|---|---|
| `status`, `reasons`, `provider`, `capture_method`, `capture_matrix_version`, `observed_start`, `observed_end`, `event_count`, `schema_version` | The evidence-health result. Each reason has `code`, `message`, `kind` and `event_id` |
| `session_id` | The resolved session |
| `session_finalized` | `true`, `false` (active) or `null` (not known) |
| `provider_source` | `hook_events`, `hook_metadata` or `null` |
| `event_types` | Count of each observed event type, such as `{"session_start": 1, "tool_call": 2}`. Empty when the session has no events |
| `redaction` | `mode` (`on`, `off`, `mixed` or `not_recorded`), `session_start_modes` (`on`, `off` or `not_recorded` for each session start in order: the implicit start first when the first hook was not SessionStart, then each `session_start` event), `redacted_event_count` and `metadata_redacted` |
| `capture_matrix` | `schema_version`, `version` (SHA-256 of the packaged matrix), the `entry` applied or `null`, its `blind_spots` (`source`, `text`, and `capability` for coverage lines) and a `note` when no entry was applied |
| `external` | Only with `--commits` or `--otel-logs`: `commits`, `otel_logs` (file name), `graded` (always `false`) and `signals`, each with `source` (`entire_checkpoint`, `copilot_agent_logs`, `claude_code_otel` or `commit_trailers`), `subject`, `status`, `confidence`, `detail` and `facts` |

`capture_matrix_version` is empty when no entry was applied, while
`capture_matrix.version` always names the matrix that was consulted. To list
only the gaps found in the trace:

```bash
agent-strace health <session-id> --format json | jq '.reasons[] | select(.kind == "observed")'
```

---

## Understanding sessions

### `explain`
```
agent-strace explain [session-id]
```
Plain-English phase breakdown: what the agent did, files touched, retries, wasted time. Each phase lists the Codex subagents that ran in it with their tool counts. No LLM required.

### `timeline`
```
agent-strace timeline [session-id] [--format text|json] [--model MODEL]
```
Structured phase-by-phase view with tool calls, errors, retries, and cost per phase.
A tool result marks the call its `parent_id` names as ok or failed. A result
without a usable `parent_id` marks the newest call in the phase that has no
result or error yet.

| Flag | Default | Description |
|---|---|---|
| `--format` | `text` | `text` or `json` |
| `--model` | `sonnet` | Pricing model: `sonnet`, `opus`, `haiku`, `gpt4`, `gpt4o` |

### `why`
```
agent-strace why [session-id] <event-number>
```
Trace the causal chain backwards from event `#N`. Run `replay` first to see event numbers.

### `cost`
```
agent-strace cost [session-id] [--model MODEL] [--input-price N] [--output-price N]
agent-strace cost --breakdown provider [--since DATE_OR_DURATION] [--csv] [--live-pricing]
agent-strace cost --tenant ID [--since DATE_OR_DURATION]
```
Token and dollar cost by phase. Flags wasted spend on failed phases.

Tokens are estimated from the size of each event. A session that records each
model request as an `llm_response` with no `llm_request` and its own token
counts, as OpenAI Codex does, is priced from those recorded counts instead,
and no event is also estimated from size. `tree`, `timeline`, `replay`,
`watch` and the EU AI Act compliance export count tokens the same way. See
[Responses recorded without a request](production.md#responses-recorded-without-a-request).

`--breakdown provider` aggregates all stored sessions in the selected window by
inferred provider and model. It uses actual recorded token counters and a bundled,
dated pricing snapshot, so the default path makes no network calls. CSV output has
the stable columns `provider`, `model`, `session_id`, `input_tokens`,
`output_tokens`, and `cost_usd`, followed by `cache_read_tokens` and
`cache_write_tokens`. Usage recorded without a model name is listed
under the model `(not recorded)` (an empty `model` in CSV) and the provider
`Unknown`, with no estimated price.

| Flag | Default | Description |
|---|---|---|
| `--model` | `sonnet` | `sonnet`, `opus`, `haiku`, `gpt4`, `gpt4o` |
| `--input-price` | — | Custom input price per 1M tokens (requires `--output-price`) |
| `--output-price` | — | Custom output price per 1M tokens (requires `--input-price`) |
| `--breakdown provider` | — | Aggregate Anthropic, OpenAI, AWS Bedrock, and Gemini usage |
| `--since DATE_OR_DURATION` | `30d` | ISO date or duration such as `7d`, `24h`, or `2w` |
| `--csv` | off | Emit raw per-session/model rows instead of the dashboard |
| `--live-pricing` | off | Request authoritative live rates; error when compatible provider APIs are unavailable rather than scrape or guess |
| `--tenant ID` | — | Scope an explicit session or aggregate the selected time window to one tenant |

Recorded prompt-cache usage is priced at the model's cache rates: cache reads
(cached input) at the cache read rate and cache writes at the cache write rate.
A write recorded with the long lifetime (Anthropic 1 hour, OpenAI 12 hours) uses
the long write rate. These usage shapes are read:

| Source | Cache fields | Input count |
|---|---|---|
| OpenAI Chat Completions | `prompt_tokens_details`: `cached_tokens`, `cache_write_tokens` | `prompt_tokens` includes them |
| OpenAI Responses | `input_tokens_details`: `cached_tokens`, `cache_write_tokens`, `cache_write_tokens_details` (`ttl_30m_tokens`, `ttl_12h_tokens`) | `input_tokens` includes them |
| OpenAI Codex `token_usage` | `cached_input_tokens`, `cache_write_input_tokens` | `input_tokens` includes them |
| Anthropic Messages, imported Claude Code transcripts | `cache_read_input_tokens`, `cache_creation_input_tokens`, `cache_creation` (`ephemeral_5m_input_tokens`, `ephemeral_1h_input_tokens`) | `input_tokens` excludes them, so they are added |
| Bedrock Converse | `cacheReadInputTokens`, `cacheWriteInputTokens`, `cacheDetails` (`ttl`, `inputTokens`) | `inputTokens` excludes them, so they are added |
| Gemini | `cachedContentTokenCount` | `promptTokenCount` includes them |

`input_tokens` in every report is the whole input, cache reads and writes
included, and the cache parts are reported beside it: in the per-session text
output when non-zero, and in the CSV columns above. Anthropic and Bedrock usage
therefore reports more input tokens than its own `input_tokens` field. Gemini
thinking tokens are counted as output and tool-use prompt tokens as input, as
Gemini bills them. OpenAI and Codex output already includes reasoning tokens.

Cache tokens are billed at the input rate when the model has no published
cache rate (for example the `haiku`, `gpt4` and `gpt4o` choices of `--model`,
retired models, and `--input-price`), and a long write without its own rate
uses the default write rate. Tokens estimated from event size have no cache
split. Pricing estimates exclude batch discounts, regional or service-tier
differences, long-context premiums, cache storage fees, and non-token charges.
When a pricing page lists a temporary promotional price and the price that
follows it, the following price is used. To update the bundled table, check the
official URLs in `PRICING_SOURCES`, update `PRICING` (each key's provider is
derived from its name), advance `PRICING_SNAPSHOT_DATE`, and extend
`tests/test_provider_cost.py` for every added model family.

### `diff`
```
agent-strace diff <session-a> <session-b> [--semantic] [--compare] [--eval-config FILE]
```
Compare two sessions structurally.

| Flag | Description |
|---|---|
| `--semantic` | Compare by outcome, not event order |
| `--compare` | Side-by-side table with verdict (cost, duration, tools, errors) |
| `--eval-config FILE` | With `--semantic`, compare scores from this eval config's `scorers` (default: `.agent-evals.yaml`; skipped if missing) |

### `compare`
```
agent-strace compare [session-id-a] [session-id-b] [--tag TAG] [--format text|json]
```
Regression report with verdict. `--tag` compares the last two sessions whose name contains the tag.

### `token-budget`
```
agent-strace token-budget [session-id] [--warning-threshold FRACTION]
```
Check token usage against the model context limit. The model is detected from the session (default: latest). Warns at `--warning-threshold` of the limit (default: `0.9`).

### `compaction`
```
agent-strace compaction [SESSION_ID] [--diff] [--behavior-diff]
                         [--compaction-threshold RATIO]
```
Detect provider-reported input-token drops between consecutive model requests
and report the context discarded by compaction.

| Flag | Description |
|---|---|
| `SESSION_ID` | Session ID or prefix (default: latest) |
| `--diff` | Classify constraints, decisions, and file context as survived or likely dropped |
| `--behavior-diff` | Compare lint findings, re-reads, loops, and exploration around compaction |
| `--compaction-threshold RATIO` | Minimum token drop ratio (default: `0.50`) |

See the [context compaction guide](compaction.md) for interpretation and live
checkpoint setup.

---

## Control and protection

### `watch`
```
agent-strace watch [session-id] [--agent-pid PID] [--timeout DURATION] [--budget DOLLARS]
                   [--on-violation ACTION]
                   [--on-death CMD] [--policy FILE] [--rules FILE_OR_BUILTINS] [--stream-to URL]
                   [--stream-otlp] [--stream-batch-size N] [--stream-flush-interval S]
                   [--loop-threshold N] [--loop-window N] [--max-context-pct N]
                   [--compaction-checkpoint] [--checkpoint-at RATIO] [--dry-run]
                   [--tenant-id ID]
```
Live session monitor with kill-switch rules.

| Flag | Description |
|---|---|
| `--agent-pid PID` | Agent process that pause, kill and approval actions signal. Watch exits with an error if the PID does not exist or cannot be signalled. Not supported on Windows |
| `--timeout DURATION` | Kill after duration (e.g. `30m`, `2h`) |
| `--budget DOLLARS` | Kill when spend exceeds this many dollars. Pass a plain number such as `5`, not `$5` |
| `--loop-threshold N` | Alert when the same tool call and arguments repeat N times; default is 3 |
| `--loop-window N` | Number of recent events to scan for repeated identical tool calls; default is 10 |
| `--on-violation terminal\|file\|kill` | Action when a rule fires |
| `--on-death CMD` | Command to run after kill (receives `{post_mortem_path}`) |
| `--compaction-checkpoint` | Write a recovery checkpoint before context saturation |
| `--checkpoint-at RATIO` | Context fill ratio that triggers a checkpoint (default: `0.80`) |
| `--policy FILE` | Scope policy file to enforce (default: `.agent-scope.json`) |
| `--rules FILE_OR_BUILTINS` | JSON/YAML rules file, or comma-separated built-ins such as `mcp-poisoning,loop:3/10,budget:5,timeout:30m,cognitive-debt:0.8` |
| `--stream-to URL` | POST events as NDJSON to this `http://` or `https://` URL while watching. Watch refuses at startup, with one line and exit 1, a URL with no host, a bad port, spaces or control characters, or a user name or password. A failed POST prints one line that names only the URL's `scheme://host[:port]`, and watch goes on. Redirects are not followed, so pass the final URL. When watch stops, it waits up to 15 seconds for the last POST; if the wait runs out or Ctrl-C stops it, one line gives the number of events not confirmed sent |
| `--stream-otlp` | With `--stream-to`, POST OTLP/HTTP JSON spans instead of NDJSON. Trailing slashes are dropped from the URL path and `/v1/traces` is appended unless the path already ends with it; the query is kept. Pass the collector base URL, such as `http://localhost:4318`. See [Live OTLP streaming](production.md#live-otlp-streaming) |
| `--stream-batch-size N` | With `--stream-to`, send once N events are waiting (default: `10`) |
| `--stream-flush-interval S` | With `--stream-to`, send waiting events once S seconds have passed since the last POST (default: `2.0`). `0` sends each event at once. A negative, infinite or NaN value is refused at startup |
| `--dry-run` | Evaluate rules without acting |
| `--tenant-id ID` | Tag the session and all events; defaults to `AGENT_STRACE_TENANT_ID` |

Watch does not guess the agent PID. Capture hooks run as children of a
shell, so their parent is not reliably the agent, and signalling the wrong
process is worse than not signalling. Pass `--agent-pid` to let watch stop,
resume and kill the agent:

```bash
claude &                                  # or any agent
agent-strace watch --agent-pid $! --rules .watch-rules.json
```

Without `--agent-pid`, every check still alerts, but `pause`, `kill` and
`require_approval` actions and VS Code Pause cannot stop the agent. The first
time one of them fires, watch prints `Cannot ...: no agent PID is known`
once. A `require_approval` rule still creates the approval request.

**Project budget config** (`.agent-strace.yaml`):
```yaml
budget:
  weekly: 20.00
  warn_at: 0.80
  stop_at: 1.00
  per_session_max: 5.00
```

When this block is present, `watch` checks rolling seven-day spend at startup
and during the session. `warn_at` writes a terminal warning and alert-log entry.
`stop_at` blocks new `record` and `record-http` sessions. `per_session_max`
uses the watchdog cost guard and kills only the over-budget session.

**Rules file format** (`.watch-rules.json`):
```json
{
  "rules": [
    { "name": "cost cap", "condition": "cost_usd > 0.50", "action": "kill" },
    { "name": "protected env", "condition": "file_path matches \"**/production.env\"", "action": "kill" },
    { "name": "large edit", "condition": "files_modified > 30", "action": "pause" }
  ]
}
```

A `pause` rule keeps the agent stopped until a human resumes it: send
`SIGCONT`, or run Pause Agent and then Resume Agent in VS Code. A
`require_approval` rule keeps it stopped until the request is decided; the
VS Code Resume command does not release it. Approving releases the pause.
Denying leaves the agent stopped like a `pause` rule, because an agent that
handles `SIGTERM` only acts on it after `SIGCONT`. While watch holds a pause,
the idle timeout does not count; the pause is dropped if the agent process
exits. `--dry-run` reports a `require_approval` rule without pausing or
creating a request.

Built-in `loop` rule config:
```yaml
watchers:
  loop:
    identical_calls: 3
    window: 10
```

### `mcp-scan`
```
agent-strace mcp-scan [--session ID] [--since DURATION_OR_DATE] [--watch]
                       [--patterns FILE] [--project-root DIR] [--format text|json]
```
Scan recorded sessions for runtime MCP tool poisoning indicators. Checks include suspicious tool description instructions, description hash drift against earlier sessions, and risky sequences such as credential reads followed by external HTTP calls.

| Flag | Description |
|---|---|
| `--session ID` | Scan one session by ID or prefix |
| `--since DURATION_OR_DATE` | Scan recent sessions since a duration or ISO date (default: `7d`) |
| `--watch` | Tail the selected/latest session and alert as new events arrive |
| `--patterns FILE` | Add regex patterns from a plain text file |
| `--project-root DIR` | Root used to detect writes outside the project (default: `.`) |
| `--format text\|json` | Output format (default: `text`) |

Custom patterns are read from `~/.agent-strace/mcp-patterns.txt` by default. Add one case-insensitive regex per line; blank lines and `#` comments are ignored.

### `audit`
```
agent-strace audit [session-id] [--policy FILE]
```
Check tool calls against a policy file. Flags sensitive file access even without a policy. Exits 1 on violations.

**Policy file** (`.agent-scope.json`):
```json
{
  "files": {
    "read":  { "allow": ["src/**", "tests/**"], "deny": [".env"] },
    "write": { "allow": ["src/**"], "deny": [".github/**"] }
  },
  "commands": {
    "allow": ["pytest", "uv run", "cat"],
    "deny":  ["curl", "wget", "rm -rf"]
  },
  "network": { "deny_all": true, "allow": ["localhost"] }
}
```

### `verify`
```
agent-strace verify [session-id] [--format text|json]
agent-strace verify --from-export FILE [--format text|json]
```
Verify a session hash chain, or verify the chain links embedded in an EU AI Act
export package.

### `validate`
```
agent-strace validate [TARGET] [--format text|json] [--allow-active]
                       [--replay-baseline FILE]
agent-strace validate --corpus [--format text|json]
```
Validate a session ID, session prefix, or session directory against the public
[`meta.json` and event schemas](trace-schema.md), relationship rules, ordering
policy, summary consistency, and hash-chain states. The default target is the
latest session. Validation is read-only and exits `0` only for a conformant
`valid` report. The path to the session may pass through root-owned operating
system links such as `/tmp` on macOS. Other symlinks on it are refused; see
[Symlinked parent directories](#symlinked-parent-directories).

| Flag | Description |
|---|---|
| `--format text\|json` | Human-readable result or the versioned validation report |
| `--allow-active` | Treat absent terminal outcomes as in-flight evidence instead of finalized evidence |
| `--replay-baseline FILE` | Compare tool-call behavior with an existing regression fixture; symbolic links are rejected, and a symlink in a parent directory must be [root-owned](#symlinked-parent-directories) |
| `--corpus` | Validate all packaged [Agent Evidence Corpus](agent-evidence-corpus.md) cases and exact expected labels |

The validator distinguishes missing usage from explicit zero usage. It accepts
legacy records that omit event `session_id` and hash links, but it does not
silently repair them. See [Public trace schema](trace-schema.md) for all rules,
status meanings, and compatibility decisions.

### `policy`
```
agent-strace policy [generate] [SESSION_ID ...] [--output FILE] [--dry-run]
agent-strace policy backtest [--policy FILE] [--days N] [--show-sessions] [--format text|json]
agent-strace policy diff OLD_POLICY NEW_POLICY [--days N] [--format text|json]
agent-strace policy coverage [--policy FILE] [--days N] [--show-uncovered] [--format text|json]
```
Generate `.agent-scope.json` from observed traces, or evaluate a proposed policy
offline against stored sessions before enforcement. The existing bare `policy`
form remains an alias for `policy generate`.

| Command / flag | Description |
|---|---|
| `generate [SESSION_ID ...]` | Suggest a policy from selected sessions, or all sessions by default |
| `backtest` | Report per-rule matches and calls/sessions that would be blocked |
| `diff OLD NEW` | Compare both policies over the identical history window |
| `coverage` | Measure the fraction of tool calls matched by an explicit rule |
| `--policy FILE` | Policy to evaluate (default: `.agent-scope.json`) |
| `--days N` | Evaluate sessions started in the last N days (default: `30`) |
| `--show-sessions` | List session IDs containing calls that would be blocked |
| `--show-uncovered` | List calls that did not match an explicit rule |
| `--format text\|json` | Human-readable or machine-readable output |

Backtesting is read-only and uses the same matching semantics as `audit`; it
never enforces a policy or contacts a remote service.

### `audit-tools`
```
agent-strace audit-tools [--repo PATH] [--since DATE] [--approved TOOLS]
```
Detect which AI coding tools are used in a repository: Claude Code, Cursor,
GitHub Copilot, Codex / ChatGPT, Windsurf, and Aider. Detection uses config
file signals and, for some tools, commit message patterns. The report also
flags unrecognized LLM endpoints in committed `.env` files and PII-like
patterns in recent commits. Reads local files and git history only: no network
calls, no API keys. `--since` takes a git date (default: `90 days ago`).
`--approved` is a comma-separated list of the tool names above
(case-insensitive); any other detected tool is reported as unapproved.

### `postmortem`
```
agent-strace postmortem [session-id] [--list] [--stale-after SECONDS]
```
Generate a structured postmortem for failed or crashed sessions. `watch`
writes a lightweight heartbeat while monitoring; if the heartbeat becomes
stale and the session has no clean `SESSION_END`, `postmortem` classifies the
crash and writes `.agent-traces/<session-id>/postmortem.md` with recovery
context.

| Flag | Description |
|---|---|
| `--list` | List crashed sessions and write missing `postmortem.md` files |
| `--stale-after SECONDS` | Heartbeat age before a session is treated as crashed (default: 30) |
| `--agents-md FILE` | AGENTS.md file used for instruction-violation checks |

### `approval`
```
agent-strace approval list [--state pending|approved|denied]
agent-strace approval show <request-id>
agent-strace approval approve <request-id> [--by NAME] [--no-resume]
agent-strace approval deny <request-id> [--reason TEXT] [--by NAME] [--no-kill]
```
Human-in-the-loop approval queue. A policy rule with
`action=require_approval` pauses the agent while it waits for a decision.
`watch` can only pause the agent when it was started with `--agent-pid`.
Approving resumes the agent with `SIGCONT` by default; denying terminates it
with `SIGTERM` by default. A request created by a `watch` without
`--agent-pid` has no agent PID: `approve` and `deny` still record the decision
and exit 0, but signal nothing and say so.

A request is decided once. If two reviewers approve or deny the same request
at the same time, exactly one succeeds and the other exits non-zero with
`Request already approved` or `Request already denied`. Records with an
unknown `state` are skipped by `list` and rejected by `approve` and `deny`.

| Command/flag | Description |
|---|---|
| `list --state STATE` | List requests, optionally filtered to `pending`, `approved`, or `denied` |
| `show REQUEST_ID` | Show a request by ID or ID prefix |
| `approve REQUEST_ID` | Approve a pending request |
| `approve --by NAME` | Record the approver name |
| `approve --no-resume` | Approve without sending `SIGCONT` to the agent |
| `deny REQUEST_ID` | Deny a pending request |
| `deny --reason TEXT` | Record the reason for denial |
| `deny --by NAME` | Record the reviewer name |
| `deny --no-kill` | Deny without sending `SIGTERM` to the agent |

### `rbac`
```
agent-strace rbac assign (--user EMAIL | --group GROUP) --role ROLE [--workspace ID] [--by NAME]
agent-strace rbac revoke (--user EMAIL | --group GROUP) [--workspace ID]
agent-strace rbac list [--workspace ID]
agent-strace rbac check --user EMAIL --action ACTION [--workspace ID]
```
Manage org-level and workspace-scoped role assignments. Org roles are `owner`,
`admin`, `member`, `viewer`, and `machine`. Workspace roles are
`workspace:admin`, `workspace:member`, and `workspace:viewer`.

| Flag | Description |
|---|---|
| `--user EMAIL` | Select a user principal for `assign` or `revoke`; required unless `--group` is used |
| `--group GROUP` | Select a group principal for `assign` or `revoke`; required unless `--user` is used |
| `--role ROLE` | Role to assign; required for `assign` |
| `--workspace ID` | Assign, revoke, list, or check in a workspace context; omit for org-level operations |
| `--by NAME` | Record the assigner name in the audit trail |
| `--action ACTION` | Authorization action to check; required for `check` |

Supported actions are `read_sessions`, `run_agent`, `annotate`,
`manage_policies`, `manage_identities`, `manage_workspaces`,
`export_compliance`, `manage_rbac`, `manage_billing`, and `manage_sso`.
Unknown actions require the effective `owner` role.

Assignments are stored in `rbac.json` in the global trace store (`--trace-dir`, else `$AGENT_TRACE_DIR`, else `.agent-traces`).

### `auth`
```
agent-strace auth login --server URL [--force]
agent-strace auth logout --server URL
agent-strace auth status [--server URL]
```
Authenticate with a hosted collector that has SSO enabled. `login` asks you to open `<URL>/auth/login`, complete the IdP login, and paste the token from the success page. The token is stored per server in `~/.agent-strace/auth.json` (mode `0600`, written atomically under a file lock) and expires after one hour. `--force` re-authenticates when a valid token already exists. `status` without `--server` lists every stored server. Other commands do not read the stored token; `apply --server` authenticates with `--auth-key`.

### `apply`
```
agent-strace apply [--config FILE] [--server URL] [--auth-key KEY] [--dry-run] [--dir DIR]
```
Apply `.agent-strace.yaml` to the local store, or to a hosted collector with `--server URL` (authenticated with `--auth-key KEY`). Use `--dry-run` to preview changes without writing. The local store is `--dir DIR` when given, else the global trace store (`--trace-dir`, else `$AGENT_TRACE_DIR`, else `.agent-traces`).

### `config-diff`
```
agent-strace config-diff [--config FILE] [--dir DIR]
```
Show drift between `.agent-strace.yaml` and the local store: `--dir DIR` when given, else the global trace store. It does not contact a hosted collector.

### `workspace`
```
agent-strace workspace list
agent-strace workspace new <name>
agent-strace workspace use <name>
agent-strace workspace rm <name> [--force]
```
Isolated workspaces — each workspace has its own session store under `<trace-dir>/workspaces/<name>/`. `use` prints `export AGENT_STRACE_WORKSPACE=<name>`; run `eval $(agent-strace workspace use <name>)` so later commands in that shell read and write that workspace. `rm` refuses to delete a workspace that has sessions unless `--force` is given.

### `compliance`
```
agent-strace compliance export [session-id] --framework eu-ai-act|soc2|hipaa|all
                                [--since Nd] [--output FILE]
```
Export a compliance report for the specified framework. Covers session retention, data handling, access logs, and policy enforcement evidence.

This is the legacy compatibility command. For a privacy-minimized, explicitly
heuristic evidence crosswalk with current versioned framework manifests, use
[`compliance-report`](#compliance-report). The two output schemas and claims
are intentionally separate.

| Framework | Coverage |
|---|---|
| `eu-ai-act` | Transparency, human oversight, data governance |
| `soc2` | Access control, availability, confidentiality |
| `hipaa` | PHI handling, audit trail, access logs |

For an auditor-facing EU AI Act Article 12/13 package, use the session export path:

```
agent-strace export <session-id> --format eu-ai-act --output compliance-report.json
agent-strace export --all --since 2026-01-01 --until 2026-03-31 \
  --format eu-ai-act --output Q1-audit.json
agent-strace verify --from-export compliance-report.json
agent-strace audit-readiness [--format text|json]
```

### `compliance-report`

```
agent-strace compliance-report --framework aicpa-tsc|owasp-agentic|eu-ai-act
    [--since UTC_OR_DURATION] [--until UTC]
    [--format json|sarif|pdf] [--output FILE]
    [--policy FILE] [--rescan [installed|FILE]]
```

Build a privacy-minimized evidence crosswalk from the flat trace store and all
workspace stores. It reports only `evidence_observed`, `risk_signal`, `gap`, or
`not_assessed`. These are evidence labels, not control conclusions. JSON is the
authoritative output; SARIF and PDF are deterministic limited projections.
See the [compliance evidence guide](compliance-report.md) for interpretation,
framework sources, privacy behavior, and limitations.

| Flag | Description |
|---|---|
| `--framework ID` | Required crosswalk: `aicpa-tsc`, `owasp-agentic`, or `eu-ai-act` |
| `--since VALUE` | Inclusive UTC timestamp/date or duration such as `90d`; default is `90d` |
| `--until UTC` | Exclusive UTC timestamp/date; default is the fixed snapshot time |
| `--format FORMAT` | `json` (authoritative), `sarif`, or `pdf`; default is `json` |
| `--output FILE`, `-o FILE` | Atomically write a nonsymlinked private file with mode `0600`; required for PDF. A symlink in a parent directory must be [root-owned](#symlinked-parent-directories) |
| `--policy FILE` | Evaluate a bounded local policy at report time; outcomes are `would_allow`/`would_deny`, never historical authorization |
| `--rescan [installed\|FILE]` | With no value or `installed`, explicitly reuse the installed manifest; a file selects a local versioned JSON manifest. Performs no network, CVE, or SBOM scan |

PDF support is optional: `pip install 'agent-strace[pdf]'`.

### `audit-readiness`
```
agent-strace audit-readiness [--retention-days N] [--format text|json]
```
Check whether the local trace store has hash-chain integrity, retention
coverage, timestamp continuity, and hash-chain presence before generating an
EU AI Act audit package.

---

## Analysis across sessions

### `dashboard`
```
agent-strace dashboard [--limit N] [--since Nd] [--html FILE] [--trend]
```
Multi-session aggregate view. `--trend` shows eval quality and behavioral metrics over time.

```bash
# Add a timeline annotation (appears as a vertical marker on trend charts)
agent-strace dashboard annotate --date 2026-05-10 --note "Added retry policy"
```

### `drift`
```
agent-strace drift (--since Nd | --current YYYY-MM-DD:YYYY-MM-DD)
                   [--baseline FILE] [--baseline-range YYYY-MM-DD:YYYY-MM-DD]
                   [--save-baseline FILE] [--threshold N] [--format table|json]
```
Detect behavioral drift across sessions. Pass `--since` or `--current`;
without either, `drift` exits 1. With `--since Nd`, the newer half of the
sessions in that window is the current set, and the older half is the baseline
unless `--baseline FILE` or `--baseline-range` supplies one. `--current`
always needs `--baseline FILE` or `--baseline-range`. Exits 1 when the drift
score exceeds `--threshold` (default: 0.20).

### `baseline`
```
agent-strace baseline update [--since DAYS] [--output FILE]
agent-strace baseline check [session-id] [--baseline FILE] [--sigma N]
agent-strace baseline show [--baseline FILE]
```
Build and inspect a statistical profile of completed sessions, then compare a
session with that profile. The default profile path is
`.agent-traces/baseline.json`.

| Command or flag | Default | Description |
|---|---|---|
| `update` | — | Build a profile from completed sessions in the selected time window |
| `update --since DAYS` | `30` | Include sessions started within the last number of days |
| `update --output FILE` | `.agent-traces/baseline.json` | Write the profile to a custom path |
| `check [session-id]` | latest | Check a session ID or prefix against the profile |
| `check --baseline FILE` | `.agent-traces/baseline.json` | Read a custom profile |
| `check --sigma N` | `2.0` | Flag metrics more than N standard deviations from the mean |
| `show --baseline FILE` | `.agent-traces/baseline.json` | Display profile statistics from a custom path |

The profile tracks cost, tool calls, duration, error rate, and LLM requests.
Cost matches `agent-strace cost` with its default model; a session whose
events cannot be priced adds no cost sample. Profiles saved by older versions
recorded every cost as 0, so rebuild yours with `baseline update` after
upgrading.
`check` exits with status 1 for an anomalous session, a missing session, or a
missing or unreadable baseline, so it can be used as a CI gate. See
[production.md](production.md#baseline-anomaly-detection) for an example.

### `fingerprint`
```
agent-strace fingerprint [--sessions N] [--output FILE] [--format text|json]
agent-strace fingerprint --compare A.json B.json [--threshold N] [--format text|json]
```
Characterize an agent's recent behavior: tool mix, error rate, retry rate, file touch radius, duration, and decision depth. Saved JSON fingerprints can be used as drift baselines or compared directly.

### `freeze`
```
agent-strace freeze [session-id] [--output FILE] [--task TEXT] [--format text|json]
agent-strace regression <fixture-file> [session-id] [--threshold N] [--format text|json]
```
Freeze a session's tool-call sequence as a JSON fixture containing tool names and stable input hashes, not raw tool inputs. `regression` compares a later session against the fixture and exits non-zero when structural divergence exceeds `--threshold` (default: `0.0`).

### `lint`
```
agent-strace lint [session-id] [--all] [--since DURATION] [--strict] [--format text|json]
```
Flag bad behavior patterns: tool loops, reasoning spirals, budget proximity,
context saturation, redundant reads, error-retry loops, no-output sessions, and
post-compaction regressions. The `post-compaction-regression` rule warns when
re-reading, exploration, context saturation, or other lint findings increase
after a detected compaction.

`--strict` exits 1 on any WARN or ERROR. Configure rules via `.agent-strace-lint.json`.

### `eval`
```
agent-strace eval [session-id] [--config FILE] [--baseline FILE]
                  [--save-baseline FILE] [--tolerance N] [--format table|json]
agent-strace eval run [session-id]
agent-strace eval compare <session-a> <session-b>
agent-strace eval ci
agent-strace eval dataset auto [--name NAME] [--filter FILTER] [--since Nd] [--label TEXT]
```
Gate the latest or selected session against named, raw-value criteria. The direct
command exits `1` when a criterion fails or a score regresses beyond the allowed
tolerance, making it suitable for CI. The `run`, `compare`, and `ci` subcommands
remain available for legacy normalized scorers.

`eval dataset auto` adds sessions from the last `--since` days (default: `7d`)
that match `--filter` (default: `has-errors`) to
`<trace-dir>/datasets/NAME.jsonl`, or to
`<trace-dir>/workspaces/<id>/datasets/NAME.jsonl` when `AGENT_STRACE_WORKSPACE`
is set. `--dataset FILE` writes to a different path.
`agent-strace optimize --dataset NAME` reads the same file.

Configure named criteria in `.agent-evals.yaml`:
```yaml
evals:
  - name: cost-ceiling
    scorer: cost_usd
    threshold: 0.50
    fail_on: above
  - name: no-errors
    scorer: error_count
    threshold: 0
    fail_on: above
  - name: task-completed
    scorer: session_status
    expected: completed
    fail_on: not_equal
```

Built-in scorers are `cost_usd`, `error_count`, `session_status`,
`redundant_read_ratio`, `tool_call_count`, `context_fill_ratio`,
`duration_seconds`, and `lint_violations`. A custom Python callable can be
referenced as `package.module:function` and receives session events plus any
criterion-specific parameters it declares.

| Flag | Default | Description |
|---|---|---|
| `--config FILE` | `.agent-evals.yaml` | Named criteria configuration |
| `--baseline FILE` | — | Compare current raw scores with a saved baseline |
| `--save-baseline FILE` | — | Save current scores as versioned JSON |
| `--tolerance N` | `0` | Allowed fractional regression from the baseline |
| `--format table\|json` | `table` | Human-readable or machine-readable output |

### `budget-report`
```
agent-strace budget-report [--since DATE] [--until DATE] [--format text|markdown|json]
```
Weekly spend digest: total cost, top sessions, cost by tool, watchdog savings.

### `team-report`
```
agent-strace team-report [--since DATE] [--until DATE] [--by author|branch|pr] [--export text|csv|json] [--outlier-threshold N]
```
Team cost attribution across recorded sessions. By default it groups spend by git author, using the last author of modified files when git is available. If git is unavailable or a file has no history, it falls back to session attribution and the local user.

| Flag | Description |
|---|---|
| `--since DATE` | Start of reporting window. Accepts ISO dates or durations like `7d`; default is 7 days ago |
| `--until DATE` | End of reporting window. Accepts ISO dates or durations like `7d`; default is now |
| `--by author|branch|pr` | Group by git author, active branch, or PR inferred from branch names like `pr-123` |
| `--export text|csv|json` | Output format. `csv` is intended for spreadsheets and finance workflows |
| `--outlier-threshold N` | Flag sessions whose cost is above `N` times the report average; default is `2.0` |

### `org-report`
```
agent-strace org-report [--month YYYY-MM] [--team SELECTOR]
                         [--format text|json|html] [--anonymize]
                         [--model MODEL] [-o FILE]
                         [--endpoint URL] [--auth-key-file FILE]
                         [--ca-file FILE] [--allow-insecure-http]
```
Build a monthly organization digest from the flat trace directory plus all
workspace stores, or from one explicitly selected authenticated collector.
Spend is an offline estimate for sessions with usable event data; the report
shows its coverage and labels averages as per cost-estimated session. Task
types and anomaly callouts are heuristics, not completion or efficiency
measurements. See the [organization reporting guide](org-report.md) for the
security boundary and interpretation details.

| Flag | Description |
|---|---|
| `--month YYYY-MM` | Half-open UTC calendar month; default is the current UTC month |
| `--team SELECTOR` | Exact team tag or workspace. Use `tag:NAME` or `workspace:NAME` to resolve a name collision |
| `--format text\|json\|html` | Terminal text, strict versioned JSON, or self-contained static HTML |
| `--anonymize` | Replace team/workspace and attributed identity labels with deterministic aliases local to this report |
| `--model MODEL` | Bundled offline pricing model; default is `sonnet` |
| `--output FILE`, `-o FILE` | Atomically write the complete report; symlinked destinations are refused |
| `--endpoint URL` | Explicit collector base URL. This command never infers it from `AGENT_STRACE_ENDPOINT` |
| `--auth-key-file FILE` | Read the collector bearer key from a file; otherwise use `AGENT_STRACE_AUTH_KEY` |
| `--ca-file FILE` | Trust a custom CA bundle for the HTTPS collector |
| `--allow-insecure-http` | Opt in to plain HTTP for a non-loopback collector |

### `cognitive-debt`
```
agent-strace cognitive-debt [--session ID] [--since DATE] [--until DATE]
                            [--by repository|branch] [--threshold N]
                            [--format text|json|badge] [--output FILE]
                            [--github-token TOKEN]
```
Measure review coverage for agent-written code from trace file-write events and local git history. The metric is reported at repository and workflow scope. It does not measure code quality, claim causation, or rank individuals. The report works without a GitHub token. When git history is unavailable, affected review state is unknown rather than zero review. See the [cognitive debt badge guide](cognitive-debt.md) for the versioned JSON contract, Shields output, CI upload, and release examples.

| Flag | Description |
|---|---|
| `--session ID` | Score one session by ID or prefix |
| `--since DATE` | Start of reporting window. Accepts ISO dates or durations like `30d`; default is `30d` |
| `--until DATE` | End of reporting window. Accepts ISO dates or durations like `7d`; default is now |
| `--by repository|branch` | Group summary rows by repository or branch. The legacy `author` value is accepted as an alias for repository and never emits individual rankings |
| `--threshold N` | Label known unreviewed coverage above this ratio; default is `0.7` |
| `--format text|json|badge` | Terminal text, versioned machine JSON, or Shields static JSON |
| `--output FILE`, `-o FILE` | Atomically write the complete output to a file |
| `--github-token TOKEN` | Optional GitHub token for merged PR review/comment enrichment; local git works without it |

`agent-strace watch --rules cognitive-debt:0.8` enables a live rule that alerts when a session has modified files that have not yet had human review.

### `context-score`
```
agent-strace context-score [--file AGENTS.md] [--history N] [--compare]
                           [--min-sessions N] [--format text|json]
```
Compare efficiency stats across AGENTS.md, CLAUDE.md, or GEMINI.md versions. Sessions are grouped by context-file hashes from `config-watch`. Each dimension (cost, tool calls, lint findings, scope violations per session) scores `50 + 100 × (baseline − current) / baseline`, clipped to 0–100; the overall score is their mean. With no version history it reports current-version stats only.

Read with care:
- Efficiency proxies, not outcomes. Tests, task success, merges, and reverts are not measured.
- 50 means no change vs baseline, which grades F (A ≥ 90 … F < 60).
- Lower cost or fewer tool calls scores higher, even if the agent skipped work.
- Before/after only: no task-mix control, no confidence intervals, default minimum 5 sessions per version.
- Baseline is the oldest version, or the previous version with `--compare`.
- Sessions that start before the first `config-watch` snapshot count as the current version.
- Cost is estimated from event size (chars/4) at default model pricing, not recorded token usage.
- Ignores skills and whether the agent read the context file. For adoption, see [skills.md](skills.md). For controlled instruction evidence, see [instruction-compliance.md](instruction-compliance.md).

| Flag | Description |
|---|---|
| `--file PATH` | Context file to score; defaults to the first existing AGENTS.md, CLAUDE.md, or GEMINI.md |
| `--history N` | Days of session history to analyse; default is `30` |
| `--compare` | Compare current context version against the previous version |
| `--min-sessions N` | Minimum sessions per version before scoring; default is `5` |
| `--format text|json` | Output format |

### `skills`
```
agent-strace skills [SESSION_ID] [--since DATE] [--workspace PATH]
                    [--skills-dir DIR ...] [--min-sessions N]
                    [--from-claude [--claude-dir DIR]] [--format text|json]
```
Skill adoption from recorded behavior: available → loaded → followed → outcome. Counts and signals only, no score. Omit `SESSION_ID` to aggregate per skill. Skills with `disable-model-invocation: true` are listed as user-only instead of never loaded, and alias or forwarder skills get hints. To test whether a skill loads when it should, use [`skills eval`](#skills-eval). See [skills.md](skills.md).

| Flag | Description |
|---|---|
| `SESSION_ID` | Session ID or prefix. Omit for aggregate mode |
| `--since DATE` | Aggregate window. Accepts ISO dates or durations like `30d`; default is `30d` |
| `--workspace PATH` | Repository root for skill discovery when a session has no `session_start` cwd; default is the current directory |
| `--skills-dir DIR` | Extra skill directory to scan (plugin or app-bundled skills). Repeatable |
| `--min-sessions N` | Minimum sessions for the never-loaded and user-only lists and for each group in the loaded vs not-loaded comparison; default is `5` |
| `--from-claude` | Read existing Claude Code transcripts (`DIR/projects/*/*.jsonl`) instead of the trace store. Writes no persistent trace store; uses a temporary directory deleted on exit. Imported events are redacted as in `import` |
| `--claude-dir DIR` | Claude Code config directory for `--from-claude`; default is `$CLAUDE_CONFIG_DIR`, else `~/.claude` |
| `--format text\|json` | Output format |

With no sessions in the trace store, the command prints how to read Claude Code history or set up capture instead of an empty report. In JSON mode the report is still printed and the hint goes to stderr.

### `skills eval`
```
agent-strace skills eval CASES_FILE (--agent claude|copilot | --agent-cmd TEMPLATE | --from-traces DIR)
                         [--runs N] [--timeout SECONDS] [--case ID ...]
                         [--workspace PATH] [--skills-dir DIR ...]
                         [--save-traces DIR] [--keep-workspaces] [--format text|json|markdown]
                         [--fail-on-miss] [--fail-on-false-trigger]
```
Experimental skill trigger eval. Runs each case's prompt through a headless agent N times in a temp workspace, captures each run with agent-strace hooks, and counts the runs in which the model loaded the skill. Optional instruction and expected-action fields add compliance evidence without adding a score. Cases whose skill sets `disable-model-invocation: true` are marked user-only. See [skills-eval.md](skills-eval.md) and [instruction-compliance.md](instruction-compliance.md).

| Flag | Description |
|---|---|
| `CASES_FILE` | JSON list of `{id, skill, prompt, should_trigger}` with optional `fixture_dir`, `runs`, `note`, `instructions`, and `expected_actions` |
| `--agent claude\|copilot` | Run a preset headless command. Presets never approve every tool |
| `--agent-cmd TEMPLATE` | Custom agent command. Placeholders: `{prompt}` (required), `{workspace}`, `{trace_dir}`, `{hooks_plugin_dir}`, `{claude_settings}` |
| `--from-traces DIR` | Score recorded runs in `DIR/<case_id>/<run>/` instead of running an agent |
| `--runs N` | Runs per case without its own `runs`; default is `3` |
| `--timeout SECONDS` | Per-run timeout; default is `300` |
| `--case ID` | Only this case. Repeatable |
| `--workspace PATH` | With `--from-traces`: repository root for skill discovery when a run has no recorded cwd |
| `--skills-dir DIR` | Extra skill directory to scan. Repeatable |
| `--save-traces DIR` | Copy each run's trace store to `DIR/<case_id>/run-<n>/` with `run.json`. Must be new or empty |
| `--keep-workspaces` | Keep each run's temp workspace, traces, and agent logs |
| `--format text\|json\|markdown` | Output format |
| `--fail-on-miss` | Exit 1 if a should-trigger run missed or was unscored |
| `--fail-on-false-trigger` | Exit 1 if a should-not-trigger run loaded the skill or was unscored |

### `standup`
```
agent-strace standup [session-id] [--no-llm]
```
Structured standup from a session trace. No LLM required. Covers files touched, approaches tried, dependencies added, TODOs written.

### `freshness`
```
agent-strace freshness [--since DATE] [--scope GLOB]
```
Check how stale the agent's last view of the codebase is. Reports files changed since last session and a freshness score 0–100.

### `oncall`
```
agent-strace oncall --rotation-start DATE [--scope GLOB]
```
Cross-reference agent-modified files against git history to find gaps before a rotation.

### `curve`
```
agent-strace curve [--min-sessions N] [--export csv]
```
Personal agent cost-efficiency curve by task type. Verdict per type: efficient / over sweet spot / do this yourself.

### `inflation`
```
agent-strace inflation [--compare MODELS] [--sessions N]
```
Measure tokenizer cost impact of switching model versions. No API calls required.

### `optimize`
```
agent-strace optimize [--target FILE] [--dataset NAME] [--apply]
                      [--base-url URL] [--model MODEL] [--api-key KEY]
```
Cluster failures by root cause and propose additions to `AGENTS.md` or any instruction file. Three built-in heuristic patterns require no LLM.

### `config-watch`
```
agent-strace config-watch snapshot [--label TEXT] [--watch PATH]
agent-strace config-watch check [--format text|json] [--watch PATH]
agent-strace config-watch history [--format text|json]
agent-strace config-watch affected [--since DURATION] [--format text|json]
```
Track changes to AGENTS.md and other config files. `check` exits 1 when config has changed (CI gate).

---

## Export and integration

### `export`
```
agent-strace export <session-id> [--format json|csv|ndjson|otlp|otlp-genai|temporal|eu-ai-act]
                    [--endpoint URL] [--header KEY:VALUE] [--service-name NAME]
                    [--anonymize] [--scores] [--metrics] [--backend otlp|langfuse]
                    [--all] [--since DURATION_OR_DATE] [--until DATE] [--output FILE]
```
Export a session. See [production.md](production.md) for per-backend OTLP setup.

`--format otlp-genai` writes OTLP/HTTP JSON with the project's
`gen_ai.*` convention and additive OpenInference attributes. With
`--endpoint`, the command appends `/v1/traces`. Use the endpoint base, such as
`http://localhost:6006` for Phoenix or
`https://cloud.langfuse.com/api/public/otel` for Langfuse. Repeated `--header`
flags add backend authentication headers.

`--format otlp` and `otlp-genai` emit `gen_ai.conversation.id` with the
agent's own session ID when the session has one (hook payloads that carry
`session_id`, Claude JSONL `import`), so exports join with native Copilot CLI
and Claude Code telemetry. See
[production.md](production.md#joining-with-native-agent-telemetry).

The GenAI payload preserves provider, model, tools, span parenting, exceptions,
token counts, recorded cost, redaction state, persisted provenance, and
observed structural evidence gaps. It does not reconstruct redacted values,
estimate missing cost, include provider capture-matrix blind spots, or replace
source NDJSON hash-chain verification. See
[OpenInference compatibility](production.md#openinference-compatibility).

`--format temporal` emits OTLP/HTTP JSON using the upstream W3C trace ID and
Temporal activity parent span previously captured by `watch`. It can be written
to a file or sent directly with `--endpoint`. See
[Temporal workflows](integrations.md#temporal-workflows).

`--format eu-ai-act` writes a structured JSON package with Article 12 logging
evidence, Article 13 transparency documentation, hash-chain integrity metadata,
and event-level line hashes for `agent-strace verify --from-export`.

### `share`
```
agent-strace share [session-id] [-o FILE | --stdout] [--open]
agent-strace share [session-id] --include-content [--postmortem] [-o FILE | --stdout] [--open]
agent-strace share [session-id] [--include-content] --dry-run [--format text|json]
agent-strace share --assignment [session-id] --output submission.zip
```
Generate a self-contained HTML evidence packet. No server is needed. The default
packet is privacy-minimized. It preserves event order and timing, tool names and
effect categories, provider and capture provenance, evidence health and known
gaps, failures, retries, recovery, redaction state, and annotations explicitly
marked shareable. It omits prompts, responses, commands, paths, tool input and
result values, and other content-rich values.

The packet's `Est. cost` and its estimated cost breakdown appear only when the
trace records a model name or token usage. Otherwise `Est. cost` says `not
captured` and no breakdown is shown.

| Flag | Description |
|---|---|
| `session-id` | Session ID or prefix (default: latest session) |
| `-o`, `--output FILE` | Output path (default: `session-<id>.html`) |
| `--stdout` | Write the HTML to stdout instead of a file |
| `--open` | Open the written HTML in the default browser |
| `--include-content` | Include full prompts, responses, commands, paths, tool inputs, tool results, and other content-rich values |
| `--postmortem` | Include postmortem analysis; requires `--include-content` |
| `--dry-run` | Print a value-free disclosure preview instead of writing HTML |
| `--format text\|json` | Preview format (default: `text`); requires `--dry-run` |
| `--assignment` | Write a privacy-minimized assignment ZIP instead of HTML |

Use `--include-content` only when a full replay is required. This is the
migration path for users who previously relied on ordinary `share` output
containing full event payloads and every annotation. The generated file displays
a full-content warning. `--postmortem` now returns an actionable error unless
`--include-content` is also present.

Every packet embeds a versioned disclosure manifest with the packet schema,
agent-strace version, selected mode, evidence-health result, provider and
capture provenance, redaction and anonymization state, limitations, behavior
counts, and deterministic SHA-256 integrity references. Source traces and
annotation sidecars are read-only inputs and are not changed.

`--dry-run` matches the selected output mode. Without `--include-content`, it
reports `minimized`. With `--include-content`, it reports `include_content`.
The preview includes category counts, included and omitted annotation counts,
redaction and anonymization actions, sensitive-field warnings, evidence-health
status, known-gap count, provider and capture provenance, included categories,
omitted categories, and limitations. It never prints prompt, response, command,
path, result, or annotation text. No HTML is written, printed, or opened.
`--output`, `--stdout`, and `--open` are ignored with a note on stderr, so the
flag can be appended to the command being reviewed. `--format json` prints one
object with `session_id` plus the
[disclosure preview](share-disclosure-preview.md) fields.

`--assignment` instead creates a deterministic, privacy-minimized ZIP with a
sanitized trace, offline replay, derived stats/cost/lint reports, and a versioned
manifest. It cannot be combined with `--stdout`, `--open`, `--postmortem`, or
`--dry-run`. It also cannot be combined with `--include-content`. Assignment
bundles keep their separate privacy format and are not evidence packets. The
output name must have a lowercase `.zip` suffix and is atomically written with
owner-only permissions. The trace store and the output path may pass through
root-owned operating system links such as `/tmp` on macOS. Other symlinks in
their parent directories are refused; see
[Symlinked parent directories](#symlinked-parent-directories). See the
[assignment review guide](hiring.md) for the privacy boundary and
interpretation limits.

Every event's `data` must be a JSON object. If any event's `data` is an array,
string, number, boolean, or null (for example after a hand edit or from another
writer), share exits 1 before it writes, prints, or opens anything. This applies
to ordinary share, `--dry-run`, and `--assignment`, so the preview never succeeds
for a session that share refuses. Ordinary share and `--dry-run` name the first
such event by position and event type and count the rest, for example
`Error: session event data must be an object: event 4 of 9 (tool_result) has
string data; 1 more event has non-object data`. The message never includes data
values. The preview refuses these sessions instead of counting them, because key
names cannot be read from a non-object payload. Other commands that read event
data refuse the same way; see
[Sessions with non-object event data](#sessions-with-non-object-event-data).

### `score`
```
agent-strace score SUBMISSION.zip --rubric FILE [--format text|json]
agent-strace score DIRECTORY --rubric FILE --compare [--format text|json]
```
Validate an assignment bundle without extracting it, then apply a strict,
deterministic process-telemetry rubric. Comparison mode considers only direct
children with a lowercase `.zip` suffix, anonymizes input filenames, rejects
duplicate bundles, and gives equal scores the same rank with digest-stable tie
ordering.

| Flag | Description |
|---|---|
| `SUBMISSION.zip` | One canonical assignment bundle |
| `DIRECTORY` | Directory of direct lowercase `.zip` files; requires `--compare` |
| `--rubric FILE` | Required strict assignment rubric; symbolic links are rejected, and a symlink in a parent directory must be [root-owned](#symlinked-parent-directories) |
| `--compare` | Score and rank a bounded directory batch |
| `--format text\|json` | Human-readable table/detail or versioned JSON (default: `text`) |

Invalid archives and rubrics exit non-zero. A low rubric score is a report for
human review, not a pass/fail hiring gate. See [hiring.md](hiring.md). The
bundle, the comparison directory, and the rubric follow the same
[symlink rule](#symlinked-parent-directories) as `share --assignment`.

### `pr-comment`
```
agent-strace pr-comment [SESSION_ID] [--dry-run] [--platform github|gitlab]
                           [--repo OWNER/REPO] [--pr NUMBER]
                           [--api-url URL] [--share-url URL]
                           [--review] [--fail-on STATUSES]
                           [--min-confidence low|medium|high]
                           [--findings NAMES] [--config FILE]
                           [--evidence-packet FILE] [--cognitive-debt FILE]
                           [--cognitive-debt-mode report|check]
                           [--artifact-url URL]
                           [--commits RANGE] [--otel-logs FILE]
```
Post a structured session summary to the current branch's open pull request or
merge request. The comment includes duration, estimated cost, model, status,
tool calls, files read and changed, lint findings, errors, redundant reads, and
replay commands. An embedded marker makes repeated runs update the existing
comment.

Cost is shown only when the trace records a model name or token usage, and
then as an estimate (see [`cost`](#cost)). Otherwise the comment says `not
captured`; GitHub Copilot CLI hooks, for example, carry neither. The model row
says `not captured` when no event names a model. Files read and modified use
the same evidence as the reviewability Scope finding: file and edit tool events
plus the files shell commands read or wrote. Paths are shown relative to the
session's working directory (`/tmp` and `/private/tmp` are the same directory
on macOS); paths outside it keep only their last three components.

| Flag | Description |
|---|---|
| `SESSION_ID` | Explicit session ID or prefix; otherwise use sessions attributed to the current branch |
| `--dry-run` | Render locally without resolving a PR or making a network request |
| `--platform github\|gitlab` | Review platform (default: `github`) |
| `--repo OWNER/REPO` | Override the repository/project inferred from CI or the git remote |
| `--pr NUMBER` | Override the PR/MR number inferred from CI or the current branch |
| `--api-url URL` | GitHub Enterprise or self-hosted GitLab API base URL |
| `--share-url URL` | Optional HTML replay URL; `{session_id}` is replaced for each session |
| `--review` | Add the [reviewability](review.md) table to the comment and exit 1 when the threshold fails |
| `--fail-on STATUSES` | With `--review`: statuses that fail. See [`review`](#review). |
| `--min-confidence low\|medium\|high` | With `--review`: lowest confidence that can fail |
| `--findings NAMES` | With `--review`: findings the threshold applies to |
| `--config FILE` | With `--review`: config file with a `review` block |
| `--evidence-packet FILE` | With `--review`: list the evidence packet. See [`review`](#review). |
| `--cognitive-debt FILE` | With `--review`: add the cognitive debt result from a schema v1 JSON report |
| `--cognitive-debt-mode report\|check` | With `--review`: `check` fails unless the cognitive debt result is a pass |
| `--artifact-url URL` | With `--review`: https URL of the uploaded artifact |
| `--commits RANGE` | With `--review`: add external evidence from commit trailers. See [`review`](#review). Ignored without `--review`. |
| `--otel-logs FILE` | With `--review`: add external evidence from a Claude Code OpenTelemetry export. Ignored without `--review`. |

With `--review`, the comment is still posted or updated when the check fails.
The exit code then follows the threshold.

GitHub uses `GITHUB_TOKEN` or `GH_TOKEN`. GitLab requires a writable project or
personal access token in `GITLAB_TOKEN`, `GITLAB_ACCESS_TOKEN`, or
`PRIVATE_TOKEN`; GitLab job tokens cannot write merge-request notes. Tokens are
read only from the environment.

### `review`
```
agent-strace review [SESSION_ID ...] [--format text|json|markdown]
                    [--branch NAME] [--fail-on STATUSES]
                    [--min-confidence low|medium|high]
                    [--findings NAMES] [--config FILE]
                    [--evidence-packet FILE] [--cognitive-debt FILE]
                    [--cognitive-debt-mode report|check]
                    [--artifact-url URL]
                    [--commits RANGE] [--otel-logs FILE]
```
Report what the linked sessions can show a PR reviewer: rationale, scope,
validation, stewardship, and evidence health. Each finding has a status
(`present`, `partial`, `missing`, `inconclusive`), a confidence (`high`,
`medium`, `low`), and its evidence source. Prompt text and command strings are
never printed. See [review.md](review.md).

| Flag | Description |
|---|---|
| `SESSION_ID ...` | Session IDs or prefixes; otherwise use sessions attributed to the current branch |
| `--format text\|json\|markdown` | Terminal text, versioned JSON, or a PR comment body (default: `text`) |
| `--branch NAME` | Branch to link sessions by (default: from CI or git) |
| `--fail-on STATUSES` | Comma-separated statuses that fail: `missing`, `partial`, `inconclusive`, or `none` (default: `missing`) |
| `--min-confidence low\|medium\|high` | Lowest confidence that can fail the check (default: `medium`) |
| `--findings NAMES` | Comma-separated findings the threshold applies to: `rationale`, `scope`, `validation`, `stewardship`, `evidence` (default: all) |
| `--config FILE` | Config file with a `review` block (default: `.agent-strace.yaml`, `.yml`, or `.json` if present) |
| `--evidence-packet FILE` | Evidence packet HTML from `share`. Shown as uploaded, local and not uploaded, or not generated. |
| `--cognitive-debt FILE` | JSON from `cognitive-debt --format json`. Reads `schema_version` 1 and `status`. `unknown` is unavailable, never a pass. |
| `--cognitive-debt-mode report\|check` | `report` adds the result. `check` also fails unless the result is a pass (default: `report`). |
| `--artifact-url URL` | https URL of the uploaded artifact. Without it, files are shown as not uploaded. |
| `--commits RANGE` | Git revision range to read `Entire-Checkpoint` and `Agent-Logs-Url` trailers from |
| `--otel-logs FILE` | Claude Code OpenTelemetry logs export to read `safety_event` from |

`--commits` and `--otel-logs` add [external evidence](evidence-health.md#external-evidence)
signals to the evidence health finding: in text and markdown after the
findings, and in JSON as `facts.external` on the `evidence` finding, in the
same shape as `health --format json`. They never change a finding's status,
the threshold or the exit code.

Flags override the config file. Exit codes: `0` pass, `1` threshold or
cognitive debt check failed, `2` bad config, unknown session, or a
`--commits` value that is not a revision range.

### `sample`
```
agent-strace sample [--strategy worst|diverse|recent|random] [--n N]
                    [--deduplicate] [--seed N] [--output FILE]
```
Export sessions as JSONL for eval datasets. Compatible with LangSmith, Braintrust, and custom eval frameworks.

### `server`
```
agent-strace server [--port N] [--host HOST] [--storage DIR] [--auth-key KEY]
                     [--dashboard]
agent-strace server keygen
```
Start a server-side event collector. See [server.md](server.md).

| Flag | Description |
|---|---|
| `--port N` | Port to listen on (default: 4317) |
| `--host HOST` | Host to bind to (default: `127.0.0.1` with `--dashboard`, otherwise `0.0.0.0`) |
| `--storage DIR` | Trace storage directory (default: `$AGENT_STRACE_STORAGE` or `.agent-traces`) |
| `--dashboard` | Serve the session list and evidence review at `/`; non-loopback binding requires `--auth-key` |
| `--auth-key KEY` | Require `Authorization: Bearer KEY` on all requests (also read from `AGENT_STRACE_AUTH_KEY`) |

`keygen` prints a new `ast_`-prefixed API key to stdout. Set `AGENT_STRACE_AUTH_KEY` on the client side to inject the header automatically into all outbound collector requests.

### `auto`
```
agent-strace auto [--framework NAME] [--detect] -- <command>
```
Run a command with every Python interpreter it starts auto-instrumented: `python script.py`, `python -m module`, and console or shebang scripts that run Python. Exits with the command's exit code. How it works and its limits: [integrations.md](integrations.md#how-agent-strace-auto-works).

| Flag | Description |
|---|---|
| `--framework NAME`, `-f NAME` | Framework to instrument, or a comma-separated list: `openai-agents`, `langchain`, `langgraph`, `litellm`, `anthropic`, `openai`, `strands`, `crewai` |
| `--detect` | Instrument every installed framework (the default when `--framework` is not given) |

Events go to `$AGENT_TRACE_DIR` or `.agent-traces`. Use the global `--trace-dir` (`agent-strace --trace-dir DIR auto -- <command>`) to choose the store.

### `mcp`
```
agent-strace mcp [--transport stdio]
```
Start an MCP server that exposes your session store as queryable tools for a debugging agent: `list_sessions`, `get_session`, `search_events`, `get_session_summary`, `diff_sessions`, `health`, `skills`, and `review`. Read-only. Use the global `--trace-dir` (`agent-strace --trace-dir DIR mcp`) or `AGENT_TRACE_DIR` to pick the store. Client setup for Claude Code, Copilot CLI, Codex, and Gemini CLI: [mcp.md](mcp.md).

### `a2a-tree`
```
agent-strace a2a-tree [session-id] [--format text|json]
```
Visualise the A2A agent call graph. Exports as OTLP spans for Jaeger, Tempo, or any OpenTelemetry backend.

### `identity`
```
agent-strace identity show
agent-strace identity sign <session-id>
agent-strace identity verify <session-id>
```
Machine identity for agent sessions. `show` creates a persistent identity (stored in `~/.agent-strace/identity.json`) if one does not exist. `sign` attaches an HMAC signature to a session. `verify` checks the signature.

Use machine identity to prove which machine produced a session — useful for compliance and multi-machine deployments.

---

## Annotations and metadata

### `annotate`
```
agent-strace annotate <session-id> [--event ID] [--at OFFSET] [--note TEXT] [--label LABEL]
                      [--author NAME] [--shareable] [--list] [--delete ANNOTATION_ID]
                      [--filter-label LABEL] [--filter-author AUTHOR] [--since Nd]
                      [--export-format json]
```
Add notes, labels, and bookmarks to session events. Annotations are private to
the local trace by default. Use `--shareable` to allow an annotation to appear
in a privacy-minimized evidence packet. Full-content packets include all
annotations.

| Flag | Description |
|---|---|
| `--event ID` | Event ID to annotate |
| `--at OFFSET` | Time offset to annotate (e.g. `2m14s`, `1:30`) |
| `--note TEXT` | Text note to attach |
| `--label LABEL` | Label chip (`root-cause`, `decision`, `retry`, `fix`, `question`) |
| `--author NAME` | Author name or email |
| `--shareable` | Allow the annotation to appear in privacy-minimized evidence packets |
| `--list` | List all annotations for the session |
| `--delete ID` | Delete an annotation by ID |
| `--filter-label LABEL` | Filter `--list` by label |
| `--filter-author AUTHOR` | Filter `--list` by author |
| `--since Nd` | Filter `--list` to annotations created in the last N days |
| `--export-format json` | Output `--list` as JSON instead of terminal text |

### `retention`
```
agent-strace retention status
agent-strace retention clean [--dry-run] [--max-age-days N] [--max-sessions N] [--max-size-mb N]
```
Enforce data retention policies. Configure via `.agent-strace.yaml`:
```yaml
retention:
  max_age_days: 30
  max_sessions: 1000
  max_size_mb: 500
  on_delete: log
```

### `tenant`
```
agent-strace tenant report [--month YYYY-MM] [--model MODEL] [--format text|json]
agent-strace tenant export ID [--format json] [--output FILE]
agent-strace tenant delete ID --confirm
```
Report monthly cost allocation, produce a tenant-scoped subject access export,
or irreversibly erase all local trace data for one tenant. Deletion requires
`--confirm` and appends a minimal audit record. See the
[multi-tenant deployment guide](tenancy.md) for isolation and GDPR notes.

### `backfill-meta`
```
agent-strace backfill-meta [--apply] [--agent-name NAME] [--format text|json]
```
Write `meta.json` for session directories that have `events.ndjson` but no
metadata. Stores captured before the subagent capture fix contain many of
these: GitHub Copilot CLI subagents (the `task` tool) fired prompt, tool and
stop hooks but never `sessionStart`. `list`, `skills` and the other readers
that list sessions skip those directories, and `tenant report`, `tenant export`,
`tenant delete` and `org-report` stop with "unrecognized or incomplete session
directory" while any of them exist.

`backfill-meta` does not link those sessions to their parent; only the
Copilot `subagentStart`/`subagentStop` hooks do, as sessions are captured.

The command scans the flat store and every `workspaces/<id>/` store. It is a
dry run unless `--apply` is given.

| Flag | Description |
|---|---|
| `--apply` | Write the metadata. Without it nothing is written |
| `--agent-name NAME` | Agent name when the events do not identify one (default: `agent-cli`, the hooks' default) |
| `--format text\|json` | Output format (default: `text`) |

Metadata is derived from the events, with the same field meaning as the
hook-written metadata:

| Field | Value |
|---|---|
| `started_at` / `ended_at` | Earliest and latest event timestamps |
| `total_duration_ms` | `ended_at - started_at` |
| `tool_calls` / `errors` / `llm_requests` | Count of `tool_call`, `error` and `llm_request` events |
| `agent_name` | From a hook `session_start` mode or a hook `stop` response provider; otherwise `--agent-name`; otherwise `agent-cli` |
| `command` | `"<agent_name> (backfilled)"`, distinct from captured metadata and from hook metadata marked `(no session-start)` |
| `tenant_id` | The single tenant ID on the events, or empty. `AGENT_STRACE_TENANT_ID` is ignored |
| `workspace_id` | The workspace store the directory is in |
| `attribution` | Empty: the current user and git state do not describe the original capture |
| `conversation_id` | Empty: the events do not carry the agent's full session ID, so the export omits `gen_ai.conversation.id` |

Safety:

- `events.ndjson` is only read, so the hash chain checked by `verify` is unchanged.
- `meta.json` is created exclusively and never replaces an existing file, even one that appears during the run.
- Writes hold the store write lock used by every other writer.
- Symlinked entries and workspaces are reported and never followed. Invalid
  session IDs, empty or unparseable event streams, and events naming another
  session are reported and skipped.
- Sessions covered by a tenant erasure tombstone, or named by a pending tenant
  delete or tag journal, are skipped so erased data is never resurrected. An
  unreadable erasure audit or journal blocks the whole store.
- Event streams with more than one tenant ID, and parents whose child sessions
  belong to another tenant, are skipped.
- Directories with `identity.json` are skipped: the signature covers metadata
  fields that cannot be rebuilt.

Backfilled sessions keep their original timestamps, so retention age limits
apply to them as soon as they are visible. Run `agent-strace retention status`
after `--apply` and before the next `retention clean`.

Exit status is 1 when a store cannot be scanned or a write fails, otherwise 0.

---

## Symlinked parent directories

`share --assignment`, `score`, `compliance-report --output`, and `validate`
refuse a path when one of its directories is a symbolic link, because a cloned
repository or an unpacked archive can plant links that move where these
commands read or write. They follow a symlink in a parent directory only when
all of these hold:

- root owns the symlink;
- root owns the directory that holds it;
- group and other users cannot write that directory, or it has the sticky bit.

This covers operating system links such as `/tmp`, `/var`, and `/etc` on macOS,
so absolute paths under the default macOS `TMPDIR` work. The components of a
trusted symlink's target are checked by the same rule. A symlink that the
current user owns is always refused. No symlink is trusted on Windows, where
Python reports owner ID 0 for every file, or when the command runs as root.

For `validate`, the rule covers the path to the session directory, whether
`TARGET` names the directory or the session is found by ID, prefix, or as the
latest session under `--trace-dir` or `AGENT_TRACE_DIR`. It also covers the
directories that hold the `--replay-baseline` file. When this rule refuses a
path, `validate` reports the session as `invalid` and exits 1.

These are still refused when they are symbolic links, even root-owned ones:
the output file, the rubric, the submission bundle, the replay baseline file,
and the trace store directory that `share --assignment` reads.

---

## Sessions with non-object event data

Hook capture records every event's `data` as a JSON object, and
`agent-strace server` refuses an event whose `data` is not an object (see
[server.md](server.md#api-reference)). An older collector stored array, string,
number, and boolean `data` as it arrived, and a hand edit can store any JSON
value, so a session can still contain non-object data. A command that reads
keys from `data` checks the session first. If an event it reads has non-object
data, the command writes one line to stderr and exits 1 before it prints,
writes, sends, or opens anything. `optimize` with an LLM configured is the one
exception, described below. The line looks like this:

```
Error: session event data must be an object: event 6 of 25 (tool_call) has string data; 5 more events have non-object data
```

The message names the first such event by position and event type, gives the
JSON type of its data, and counts every other event in the session with
non-object data. It never includes data values or event IDs. Commands that read
more than one session name the session, for example `Error: session
c3d4e5f6a7b80003 event data must be an object: event 3 of 7 (tool_call) has
array data`.

These commands refuse such sessions: `a2a-tree`, `audit`, `budget-report`,
`cognitive-debt`, `compaction`, `compare`, `compliance export`,
`context-score --file`, `cost`, `dashboard --trend`, `diff`, `explain`,
`export` with `--format otlp`, `otlp-genai`, `temporal`, or `eu-ai-act` or with
`--backend`, `inflation`, `mcp-scan --session`, `oncall`, `optimize`,
`policy generate`, `postmortem`, `pr-comment`, `replay`, `sample`, `share`,
`standup`, `stats`, `team-report`, `tenant report`, `timeline`, `token-budget`,
and `why`. A command refuses only when it reads an event whose data is not an
object, so the same session keeps working with commands that never read that
event. For example, `export --format otlp-genai` and `temporal` pair events by
ID and do not read a tool_call that an error event pairs with.
`export --format temporal` checks the session's Temporal trace context first, so
a session without one keeps its `Temporal export failed` message. `optimize`
with `--base-url` and `--api-key` checks the events in its prompt before it sends
the request. When the LLM returns no patterns, `optimize` falls back to its
heuristics, which read more events, so it can refuse after it sent the request.
`freeze`, `regression`, `compliance-report`, `org-report`, and `compare` for
events other than decisions already stopped on these sessions with their own
error message and exit code 1, and they keep it.

`list`, `inspect`, `tree`, `verify`, and `export --format json`, `ndjson`, or `csv`
keep working on these sessions. The export formats include the data exactly as
stored.

`watch` keeps running when a live event that one of its checks reads has
non-object data. It skips the checks for that event, prints one line such as
`[watch] Skipped checks for one tool_call event because its data is a string,
not an object.` to stderr, and still counts the event. `watch
--compaction-checkpoint` refuses at startup when the session already has an
event with non-object data. `mcp-scan --watch` refuses at startup in the same
way and skips live events with non-object data with an `[mcp-scan]` line.

Without `--stream-otlp`, `watch --stream-to` sends a live event with non-object
data as stored. With `--stream-otlp`, watch turns each batch into spans and
leaves out each such event whose data the span converter reads, with one line
per event on stderr such as `[stream] Left one tool_result event out of the
OTLP stream because its data is an array, not an object.` The other events of
the batch are still sent. The converter reads `user_prompt`,
`assistant_response`, `llm_request`, `llm_response`, `tool_call`,
`tool_result`, and `error` data, but not the data of a tool_call that an error
event pairs with, so that tool_call and events such as `session_start`,
`decision`, and `file_read` stay in the batch. Leaving an event out can change
how the others pair, so the check repeats on the events that are left: when an
error event and the tool_call it pairs with both have non-object data, both are
left out. Nothing is sent for a session with no event left in the batch. If the
conversion still fails, watch prints one line such as `[stream] Could not
convert 4 events to OTLP spans (KeyError); they were not sent.`, which names
the exception type but never its message, and keeps sending later batches.

### Fields with the wrong type

When `data` is an object but a field inside it has the wrong type, commands read
that field as missing. Examples are a number where a tool name belongs, a string
where an arguments object belongs, and a list where a prompt belongs. The
command behaves as if the key were absent and keeps going, without a traceback
or an error message. A number field accepts integers and decimals. `true`,
`false`, and numeric strings such as `"12"` are not numbers, so they read as
missing too. `watch`, `mcp`, `compliance-report`, and `audit-readiness` read
fields the same way. `cognitive-debt` is the exception. It stops with an error
when a write event's tool name, path, or arguments have the wrong type, and it
counts the lines of text inside content stored as a list or an object.

Some output still prints the stored value as it is, for example the `share`
page, `export --format json`, the `otlp` and `otlp-genai` exports, `audit`,
`postmortem`, and the `compliance export` access log. Cost and token estimates
still count the stored size of each event.
`record` and `record-http` link a tool result or a JSON-RPC error to its call
only when the JSON-RPC id is a string or a number. With any other id they
forward both messages and record both events without the link.

---

## GitHub Actions

The `agent-trace eval` composite action runs evals in CI, posts a scored table to the GitHub Actions step summary, and exits non-zero on regression. It can also run the [reviewability check](review.md) as a required status check.

```yaml
permissions:
  contents: read
  pull-requests: write

steps:
  - uses: Siddhant-K-code/agent-trace@v0.93.1
    with:
      config: .agent-evals.yaml
      baseline: .agent-evals-baseline.json
      tolerance: "0.05"
      post-pr-comment: "true"
      github-token: ${{ secrets.GITHUB_TOKEN }}
```

| Input | Default | Description |
|---|---|---|
| `config` | `.agent-evals.yaml` | Eval config file |
| `baseline` | none | Baseline scores file for regression gating |
| `save-baseline` | none | Path to write current scores as a new baseline |
| `tolerance` | `0.05` | Max allowed score regression (0.0 to 1.0) |
| `trace-dir` | `.agent-traces` | Session storage directory |
| `python-version` | `3.12` | Python version |
| `install-extras` | none | Optional extras, e.g. `openai,anthropic` |
| `post-pr-comment` | `false` | Post or update the structured session summary on the PR |
| `github-token` | none | Token for PR comments; pass `${{ secrets.GITHUB_TOKEN }}` |
| `run-eval` | `true` | Run `agent-strace eval`. Set `false` for a review-only job. |
| `review` | `false` | Run `agent-strace review` and fail the step per the threshold. With `post-pr-comment`, the table is added to the single PR comment. |
| `review-fail-on` | none | Statuses that fail the check. Empty uses the config file, then `missing`. |
| `review-min-confidence` | none | Lowest confidence that can fail. Empty uses the config file, then `medium`. |
| `cognitive-debt` | `report` | With `review`: `false` skips the report, `report` adds it, `check` also fails unless the result is a pass. `unknown` is never a pass. |

| Output | Description |
|---|---|
| `passed` | `true` if all scorers passed |
| `summary-path` | Path to the written eval summary markdown |
| `review-passed` | `true` or `false` when `review` ran; empty otherwise |
| `review-result` | `pass`, `fail`, or `error` when `review` ran |
| `review-markdown-path` | Path to `review.md` on the runner |
| `review-json-path` | Path to `review.json` on the runner |
| `evidence-packet-path` | Path to the metadata-only evidence packet; empty if not generated |
| `cognitive-debt-json-path` | Path to the schema v1 cognitive debt JSON; empty if not generated |
| `cognitive-debt-badge-path` | Path to the Shields endpoint JSON; empty if not generated |
| `cognitive-debt-status` | `clean`, `partial`, `above_threshold`, `unknown`, or empty |
| `cognitive-debt-result` | `pass`, `fail`, or `unavailable` |
| `artifacts-dir` | Directory with the PR evidence files |
| `evidence-artifact-url` | URL of the `agent-trace-pr-evidence` artifact; empty if not uploaded |

Trace artifacts are uploaded automatically under the `agent-traces` artifact name.
With `review`, the evidence packet and cognitive debt files are uploaded as
`agent-trace-pr-evidence`, and `review.json` and `review.md` as
`agent-trace-pr-review`. See [review.md](review.md#github-action).

[Marketplace listing](https://github.com/marketplace/actions/agent-trace-eval)
