# Command reference

Full flag reference for every `agent-strace` command.

---

## Session capture

### `record`
```
agent-strace record [--name NAME] [--redact] [--mask] -- <command>
```
Capture an MCP stdio server session. Wraps `<command>` as a transparent proxy.

| Flag | Description |
|---|---|
| `--name NAME` | Label for the session |
| `--redact` | Strip secrets before writing to disk |
| `--mask` | Mask PII (email, phone, CC, SSN) |

### `record-http`
```
agent-strace record-http <url> [--port N] [--redact] [--mask]
```
Capture an MCP HTTP/SSE server session. Listens on `--port` (default: 3100) and proxies to `<url>`.

### `setup`
```
agent-strace setup [--redact] [--global]
```
Print Claude Code hooks config JSON. Add `--global` to write to `~/.claude/settings.json`.

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
                   [--on-death CMD] [--rules FILE] [--stream-to URL]
                   [--stream-batch-size N] [--stream-flush-interval S]
                   [--max-context-pct N] [--dry-run]
```
Live session monitor with kill-switch rules.

| Flag | Description |
|---|---|
| `--timeout DURATION` | Kill after duration (e.g. `30m`, `2h`) |
| `--budget $N` | Kill when spend exceeds N dollars |
| `--on-violation kill\|pause\|alert` | Action when a rule fires |
| `--on-death CMD` | Command to run after kill (receives `{post_mortem_path}`) |
| `--rules FILE` | JSON rules file |
| `--stream-to URL` | Stream events to HTTP endpoint in real-time |
| `--dry-run` | Evaluate rules without acting |

**Rules file format** (`.watch-rules.json`):
```json
[
  { "condition": "cost_usd", "threshold": 0.50, "action": "kill" },
  { "condition": "file_path", "glob": "**/production.env", "action": "kill" },
  { "condition": "files_modified", "threshold": 30, "action": "pause" }
]
```

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
agent-strace postmortem [session-id]
```
View the watchdog post-mortem for a killed session.

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
agent-strace export <session-id> [--format json|csv|ndjson|otlp|otlp-genai]
                    [--endpoint URL] [--header KEY:VALUE] [--service-name NAME]
                    [--anonymize] [--scores] [--metrics] [--backend otlp|langfuse]
                    [--since DURATION] [--langfuse-host URL]
```
Export a session. See [production.md](production.md) for per-backend OTLP setup.

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

---

## Annotations and metadata

### `annotate`
```
agent-strace annotate <session-id> <event-offset> [--note TEXT] [--label TEXT]
                      [--bookmark] [--list] [--delete ANNOTATION_ID]
```
Add notes, labels, and bookmarks to session events. Annotations appear in shared HTML reports.

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
