# Command reference

Full flag reference for every `agent-strace` command.

---

## Session capture

### `record`
```
agent-strace record [--name NAME] [--parent SESSION] [--no-redact] [--mask] -- <command>
```
Capture an MCP stdio server session. Wraps `<command>` as a transparent proxy.

| Flag | Description |
|---|---|
| `--name NAME` | Label for the session |
| `--parent SESSION` | Link this session as a child of a parent session |
| `--redact` | Strip secrets before writing to disk; kept for compatibility because this is now the default |
| `--no-redact` | Disable automatic secret redaction |
| `--mask` | Mask PII (email, phone, CC, SSN) |

`AGENT_STRACE_PARENT_SESSION` can also be set by orchestrators that spawn child agents.

### `record-http`
```
agent-strace record-http <url> [--port N] [--parent SESSION] [--no-redact] [--mask]
```
Capture an MCP HTTP/SSE server session. Listens on `--port` (default: 3100) and proxies to `<url>`.

### `setup`
```
agent-strace setup [--cli claude|codex|gemini|all] [--no-redact] [--global]
```
Print or install hooks config for supported agent CLIs. `--cli claude` prints Claude Code settings JSON for `~/.claude/settings.json`; `--cli codex` prints OpenAI Codex hooks JSON for `~/.codex/hooks.json`; `--cli gemini` writes a Gemini CLI extension under `$GEMINI_CONFIG_DIR/extensions/agent-strace` or `~/.gemini/extensions/agent-strace`; `--cli all` configures all supported CLIs. Secret redaction is enabled by default; use `--no-redact` only for trusted local traces.

### `import`
```
agent-strace import <path.jsonl> [--discover]
```
Import a Claude Code JSONL session log. `--discover` lists available sessions in `~/.claude/projects/`.

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
| `--expand-subagents` | Inline subagent sessions under parent tool_call |
| `--tree` | Show session hierarchy without full replay |

### `tree`
```
agent-strace tree [session-id] [--format text|json]
```
Show the parent/child session hierarchy for a root session, including per-node cost, tool calls, status, and duration. Parent links come from `record --parent`, `AGENT_STRACE_PARENT_SESSION`, A2A trace propagation, or imported trace metadata.

### `list`
```
agent-strace list
```
List all captured sessions with ID, timestamp, duration, tool calls, and errors.

### `inspect`
```
agent-strace inspect <session-id>
```
Dump full session as JSON (meta + events).

### `stats`
```
agent-strace stats [session-id] [--include-subagents]
```
Tool call frequency and timing. `--include-subagents` rolls up across the full subagent tree.

---

## Understanding sessions

### `explain`
```
agent-strace explain [session-id]
```
Plain-English phase breakdown: what the agent did, files touched, retries, wasted time. No LLM required.

### `timeline`
```
agent-strace timeline [session-id] [--format text|json] [--model MODEL]
```
Structured phase-by-phase view with tool calls, errors, retries, and cost per phase.

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
```
Token and dollar cost by phase. Flags wasted spend on failed phases.

| Flag | Default | Description |
|---|---|---|
| `--model` | `sonnet` | `sonnet`, `opus`, `haiku`, `gpt4`, `gpt4o` |
| `--input-price` | — | Custom input price per 1M tokens (requires `--output-price`) |
| `--output-price` | — | Custom output price per 1M tokens (requires `--input-price`) |

### `diff`
```
agent-strace diff <session-a> <session-b> [--semantic] [--compare]
```
Compare two sessions structurally.

| Flag | Description |
|---|---|
| `--semantic` | Compare by outcome, not event order |
| `--compare` | Side-by-side table with verdict (cost, duration, tools, errors) |

### `compare`
```
agent-strace compare [session-id-a] [session-id-b] [--tag TAG] [--format text|json]
```
Regression report with verdict. `--tag` compares the last two sessions whose name contains the tag.

### `token-budget`
```
agent-strace token-budget <session-id> [--model MODEL] [--warn-at PCT]
```
Check token usage against model context limit.

---

## Control and protection

### `watch`
```
agent-strace watch [session-id] [--timeout DURATION] [--budget $N] [--on-violation ACTION]
                   [--on-death CMD] [--policy FILE] [--rules FILE_OR_BUILTINS] [--stream-to URL]
                   [--stream-batch-size N] [--stream-flush-interval S]
                   [--loop-threshold N] [--loop-window N] [--max-context-pct N] [--dry-run]
```
Live session monitor with kill-switch rules.

| Flag | Description |
|---|---|
| `--timeout DURATION` | Kill after duration (e.g. `30m`, `2h`) |
| `--budget $N` | Kill when spend exceeds N dollars |
| `--loop-threshold N` | Alert when the same tool call and arguments repeat N times; default is 3 |
| `--loop-window N` | Number of recent events to scan for repeated identical tool calls; default is 10 |
| `--on-violation terminal\|file\|kill` | Action when a rule fires |
| `--on-death CMD` | Command to run after kill (receives `{post_mortem_path}`) |
| `--policy FILE` | Scope policy file to enforce (default: `.agent-scope.json`) |
| `--rules FILE_OR_BUILTINS` | JSON/YAML rules file, or comma-separated built-ins such as `mcp-poisoning,loop:3/10,budget:$5,timeout:30m,cognitive-debt:0.8` |
| `--stream-to URL` | Stream events to HTTP endpoint in real-time |
| `--dry-run` | Evaluate rules without acting |

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

### `policy`
```
agent-strace policy [--last N] [--output FILE]
```
Generate `.agent-scope.json` from observed traces. Review and tighten before committing.

### `audit-tools`
```
agent-strace audit-tools [--repo .] [--since DATE] [--approved TOOLS]
```
Detect Shadow MCP servers and undeclared agent activity. No network calls, no API keys.

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
agent-strace approval list [--status pending|approved|denied]
agent-strace approval show <request-id>
agent-strace approval approve <request-id> [--note TEXT]
agent-strace approval deny <request-id> [--note TEXT]
```
Human-in-the-loop approval queue. Agents pause at a checkpoint and wait for a human to approve or deny before continuing. Integrates with the `watch` kill-switch — a denied request triggers the configured `--on-violation` action.

### `rbac`
```
agent-strace rbac assign --user USER --role ROLE [--scope org|workspace] [--workspace NAME]
agent-strace rbac revoke --user USER --role ROLE [--scope org|workspace]
agent-strace rbac list [--scope org|workspace]
agent-strace rbac check --user USER --action ACTION [--resource RESOURCE]
```
Role-based access control for the hosted collector. Roles: `admin`, `editor`, `viewer`.

| Flag | Description |
|---|---|
| `--scope org` | Org-wide assignment |
| `--scope workspace` | Scoped to a named workspace |
| `--workspace NAME` | Workspace name (required when `--scope workspace`) |

### `auth`
```
agent-strace auth login --host URL [--client-id ID] [--issuer URL]
agent-strace auth logout
agent-strace auth status
```
Authenticate with a hosted collector via OIDC. Stores the token in `~/.agent-strace/token.json`. All subsequent commands that contact the collector use the stored token automatically.

### `apply`
```
agent-strace apply [--config FILE] [--host URL] [--dry-run]
```
Apply `.agent-strace.yaml` to the local store or a hosted collector. Use `--dry-run` to preview changes without writing.

### `config-diff`
```
agent-strace config-diff [--config FILE] [--host URL]
```
Show the diff between the local `.agent-strace.yaml` and the live config on a hosted collector.

### `workspace`
```
agent-strace workspace list
agent-strace workspace new <name>
agent-strace workspace use <name>
agent-strace workspace rm <name>
```
Isolated workspaces — each workspace has its own session store. Use `use` to print the shell export (`AGENT_STRACE_STORAGE`) for a workspace.

### `compliance`
```
agent-strace compliance export [session-id] --framework eu-ai-act|soc2|hipaa|all
                                [--since Nd] [--output FILE]
```
Export a compliance report for the specified framework. Covers session retention, data handling, access logs, and policy enforcement evidence.

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
agent-strace dashboard [--last N] [--since DATE] [--html FILE] [--trend]
```
Multi-session aggregate view. `--trend` shows eval quality and behavioral metrics over time.

```bash
# Add a timeline annotation (appears as a vertical marker on trend charts)
agent-strace dashboard annotate --date 2026-05-10 --note "Added retry policy"
```

### `drift`
```
agent-strace drift [--since DURATION] [--baseline FILE] [--save-baseline FILE]
                   [--threshold N] [--format text|json]
```
Detect behavioral drift across sessions. Exits non-zero when drift score exceeds `--threshold` (default: 0.20).

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
Flag bad behavior patterns: tool loops, reasoning spirals, budget proximity, context saturation, redundant reads, error-retry loops, no-output sessions.

`--strict` exits 1 on any WARN or ERROR. Configure rules via `.agent-strace-lint.json`.

### `eval`
```
agent-strace eval [session-id] [--format text|json]
agent-strace eval compare <session-a> <session-b>
agent-strace eval ci
```
Score a session against configurable criteria. `eval ci` exits non-zero if any scorer fails.

Configure scorers in `.agent-evals.yaml`:
```yaml
scorers:
  - type: no_errors
  - type: cost_under
    max_dollars: 0.10
  - type: files_scoped
    allowed_paths: ["src/", "tests/"]
```

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

### `cognitive-debt`
```
agent-strace cognitive-debt [--session ID] [--since DATE] [--until DATE]
                            [--by author|branch] [--threshold N]
                            [--format text|json] [--github-token TOKEN]
```
Measure unreviewed agent-written code from trace file-write events and local git history. The report works without a GitHub token; when git history is unavailable it still reports agent-written lines and treats review evidence as unknown.

| Flag | Description |
|---|---|
| `--session ID` | Score one session by ID or prefix |
| `--since DATE` | Start of reporting window. Accepts ISO dates or durations like `30d`; default is `30d` |
| `--until DATE` | End of reporting window. Accepts ISO dates or durations like `7d`; default is now |
| `--by author|branch` | Group summary rows by git author or branch |
| `--threshold N` | Flag sessions above this debt score; default is `0.7` |
| `--format text|json` | Output format |
| `--github-token TOKEN` | Optional GitHub token for merged PR review/comment enrichment; local git works without it |

`agent-strace watch --rules cognitive-debt:0.8` enables a live rule that alerts when a session has modified files that have not yet had human review.

### `standup`
```
agent-strace standup [--session SESSION_ID]
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
agent-strace export <session-id> [--format json|csv|ndjson|otlp|otlp-genai|eu-ai-act]
                    [--endpoint URL] [--header KEY:VALUE] [--service-name NAME]
                    [--anonymize] [--scores] [--metrics] [--backend otlp|langfuse]
                    [--all] [--since DURATION_OR_DATE] [--until DATE] [--output FILE]
```
Export a session. See [production.md](production.md) for per-backend OTLP setup.

`--format eu-ai-act` writes a structured JSON package with Article 12 logging
evidence, Article 13 transparency documentation, hash-chain integrity metadata,
and event-level line hashes for `agent-strace verify --from-export`.

### `share`
```
agent-strace share <session-id> [-o FILE]
```
Generate a self-contained HTML report. No server needed.

### `sample`
```
agent-strace sample [--strategy worst|diverse|recent|random] [--n N]
                    [--deduplicate] [--seed N] [--output FILE]
```
Export sessions as JSONL for eval datasets. Compatible with LangSmith, Braintrust, and custom eval frameworks.

### `server`
```
agent-strace server [--port N] [--host HOST] [--storage DIR] [--auth-key KEY]
agent-strace server keygen
```
Start a server-side event collector. See [server.md](server.md).

| Flag | Description |
|---|---|
| `--port N` | Port to listen on (default: 4317) |
| `--host HOST` | Host to bind to (default: 0.0.0.0) |
| `--storage DIR` | Trace storage directory (default: `$AGENT_STRACE_STORAGE` or `.agent-traces`) |
| `--auth-key KEY` | Require `Authorization: Bearer KEY` on all requests (also read from `AGENT_STRACE_AUTH_KEY`) |

`keygen` prints a new `ast_`-prefixed API key to stdout. Set `AGENT_STRACE_AUTH_KEY` on the client side to inject the header automatically into all outbound collector requests.

### `auto`
```
agent-strace auto [--framework NAME] [--detect] -- <command>
```
Run a command with auto-instrumentation. See [integrations.md](integrations.md).

### `mcp`
```
agent-strace mcp [--transport stdio|http] [--port N]
```
Start an MCP server that exposes your session store as queryable tools for a debugging agent.

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
                      [--author NAME] [--list] [--delete ANNOTATION_ID]
                      [--filter-label LABEL] [--filter-author AUTHOR] [--since Nd]
                      [--export-format json]
```
Add notes, labels, and bookmarks to session events. Annotations appear in shared HTML reports.

| Flag | Description |
|---|---|
| `--event ID` | Event ID to annotate |
| `--at OFFSET` | Time offset to annotate (e.g. `2m14s`, `1:30`) |
| `--note TEXT` | Text note to attach |
| `--label LABEL` | Label chip (`root-cause`, `decision`, `retry`, `fix`, `question`) |
| `--author NAME` | Author name or email |
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

---

## GitHub Actions

The `agent-trace eval` composite action runs evals in CI, posts a scored table to the GitHub Actions step summary, and exits non-zero on regression.

```yaml
- uses: Siddhant-K-code/agent-trace@gha-v1
  with:
    config: .agent-evals.yaml
    baseline: .agent-evals-baseline.json
    tolerance: "0.05"
    save-baseline: "false"
```

| Input | Default | Description |
|---|---|---|
| `config` | `.agent-evals.yaml` | Eval config file |
| `baseline` | none | Baseline scores file for regression gating |
| `save-baseline` | `false` | Overwrite baseline with current scores |
| `tolerance` | `0.05` | Max allowed score regression (0.0 to 1.0) |
| `trace-dir` | `.agent-traces` | Session storage directory |
| `python-version` | `3.12` | Python version |
| `install-extras` | none | Optional extras, e.g. `openai,anthropic` |

| Output | Description |
|---|---|
| `passed` | `true` if all scorers passed |
| `summary-path` | Path to the written eval summary markdown |

Trace artifacts are uploaded automatically under the `agent-traces` artifact name.

[Marketplace listing](https://github.com/marketplace/actions/agent-trace-eval)
