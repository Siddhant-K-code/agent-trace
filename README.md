# agent-trace

[![Run in Ona](https://ona.com/run-in-ona.svg)](https://app.ona.com/#https://github.com/Siddhant-K-code/agent-trace)
[![PyPI](https://img.shields.io/pypi/v/agent-strace)](https://pypi.org/project/agent-strace/)
[![Python](https://img.shields.io/pypi/pyversions/agent-strace)](https://pypi.org/project/agent-strace/)
[![License](https://img.shields.io/github/license/Siddhant-K-code/agent-trace)](LICENSE)
[![CI](https://github.com/Siddhant-K-code/agent-trace/actions/workflows/test.yml/badge.svg)](https://github.com/Siddhant-K-code/agent-trace/actions/workflows/test.yml)
[![Open VSX](https://img.shields.io/open-vsx/v/Siddhant-K-code/agent-strace)](https://open-vsx.org/extension/Siddhant-K-code/agent-strace)
[![VS Marketplace](https://img.shields.io/badge/VS%20Marketplace-v0.1.2-blue?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=Siddhant-K-code.agent-strace)

`strace` for AI agents. Capture and replay every tool call, prompt, and response from Claude Code, Cursor, Gemini CLI, or any MCP client. Analyse, diff, audit, and share what happened.

![demo](assets/demo.svg)

## Why

A coding agent rewrites 20 files in a background session. You get a pull request. You do not get the story. Which files did it read first? Why did it call the same tool three times? What failed before it found the fix?

Most tools trace LLM calls. That is one layer. The gap is everything around it: tool calls, file operations, decision points, error recovery, the actual commands the agent ran. `agent-strace` captures the full session and lets you replay it later. Export to Datadog, Honeycomb, New Relic, or Splunk for production observability.

Set rules to stop the agent: cost ceiling, wrong file touched, too many tool calls. The agent stops. No prompt, no retry, no damage.

## Install

```bash
# With uv (recommended)
uv tool install agent-strace

# Or with pip
pip install agent-strace

# Or run without installing
uvx agent-strace replay
```

**Zero dependencies.** Python 3.10+ standard library only.

## VS Code / Cursor extension

Install the **agent-strace** extension to see live session activity without leaving the editor.

**Install:**
- Search `agent-strace` in the Extensions panel (VS Code, Cursor, or any Open VSX-compatible editor)
- Or install from [open-vsx.org/extension/Siddhant-K-code/agent-strace](https://open-vsx.org/extension/Siddhant-K-code/agent-strace)

**What you get:**

| Feature | Description |
|---|---|
| Status bar | Live cost, tool call count, and active tool name. Click to open the event stream. |
| Gutter annotations | Blue border on files the agent read, amber on files it modified. Inline label shows read/write counts. |
| Event stream panel | Live feed in the Explorer sidebar: every tool call, file op, LLM request, and error. |
| Pause button | Stops the agent mid-session via SIGSTOP. Requires `agent-strace watch` running in a terminal. |

**Setup:**

```bash
# 1. Install agent-strace
pip install agent-strace

# 2. Add hooks to Claude Code (one-time)
agent-strace setup

# 3. Open your project in VS Code / Cursor
# The extension activates automatically when .agent-traces/ exists

# 4. Start Claude Code — the status bar item appears immediately
```

The extension activates automatically when a `.agent-traces/` directory exists in the workspace root. No configuration required.

**Pause / resume** (optional, requires watch running):

```bash
# In a separate terminal, start the watcher
agent-strace watch

# Then use the Pause button in the event stream panel,
# or run: agent-trace: Pause Agent from the command palette
```

## Quick start

### Option 1: Claude Code hooks (full session capture)

Captures everything: user prompts, assistant responses, and every tool call (Bash, Edit, Write, Read, Agent, Grep, Glob, WebFetch, WebSearch, all MCP tools).

```bash
agent-strace setup        # prints hooks config JSON
agent-strace setup --global  # for all projects
```

Add the output to `.claude/settings.json`. Or paste it manually:

```json
{
  "hooks": {
    "UserPromptSubmit": [{ "hooks": [{ "type": "command", "command": "agent-strace hook user-prompt" }] }],
    "PreToolUse": [{ "matcher": "", "hooks": [{ "type": "command", "command": "agent-strace hook pre-tool" }] }],
    "PostToolUse": [{ "matcher": "", "hooks": [{ "type": "command", "command": "agent-strace hook post-tool" }] }],
    "PostToolUseFailure": [{ "matcher": "", "hooks": [{ "type": "command", "command": "agent-strace hook post-tool-failure" }] }],
    "Stop": [{ "hooks": [{ "type": "command", "command": "agent-strace hook stop" }] }],
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "agent-strace hook session-start" }] }],
    "SessionEnd": [{ "hooks": [{ "type": "command", "command": "agent-strace hook session-end" }] }]
  }
}
```

Then use Claude Code normally.

```bash
agent-strace list     # list sessions
agent-strace replay   # replay the latest
agent-strace explain  # plain-English summary of what the agent did
agent-strace stats    # tool call frequency and timing
```

### Option 2: MCP proxy (any MCP client)

Wraps any MCP server. Works with Cursor, Windsurf, or any MCP client.

```bash
agent-strace record -- npx -y @modelcontextprotocol/server-filesystem /tmp
agent-strace replay
```

### Option 3: Python decorator

Wraps your tool functions directly. No MCP required.

```python
from agent_trace import trace_tool, trace_llm_call, start_session, end_session, log_decision

start_session(name="my-agent")  # add redact=True to strip secrets

@trace_tool
def search_codebase(query: str) -> str:
    return search(query)

@trace_llm_call
def call_llm(messages: list, model: str = "claude-4") -> str:
    return client.chat(messages=messages, model=model)

# Log decision points explicitly
log_decision(
    choice="read_file_first",
    reason="Need to understand current implementation before making changes",
    alternatives=["read_file_first", "search_codebase", "write_fix_directly"],
)

search_codebase("authenticate")
call_llm([{"role": "user", "content": "Fix the bug"}])

meta = end_session()
print(f"Replay with: agent-strace replay {meta.session_id}")
```

## CLI commands

| Command | What it does |
|---|---|
| `record` | Capture an MCP stdio session |
| `record-http` | Capture an MCP HTTP/SSE session |
| `replay` | Replay a session in the terminal or as HTML |
| `inspect` | Show raw events for a session |
| `stats` | Summary stats for a session |
| `eval` | Score a session against configurable criteria |
| `eval ci` | CI gate, exits non-zero if any scorer fails |
| `eval compare` | Compare two sessions side by side |
| `drift` | Detect behavioral drift across sessions |
| `optimize` | Propose AGENTS.md improvements from trace failures |
| `dashboard` | Aggregate view across sessions |
| `dashboard --trend` | Eval quality and behavioral metrics over time (HTML) |
| `export` | Export a session (JSON, CSV, OTLP, Langfuse) |
| `diff` | Semantic diff between two sessions |
| `why` | Causal chain for a tool call |
| `explain` | Plain-English session summary |
| `cost` | Estimate session cost |
| `standup` | Structured standup report from a session |
| `oncall` | On-call readiness report for agent-modified files |
| `freshness` | Context freshness check vs last session |
| `watch` | Live session monitor with kill-switch rules |
| `annotate` | Add annotations to session events |
| `audit-tools` | Shadow AI / MCP detection |
| `inflation` | Token inflation across model versions |
| `curve` | Personal cost-efficiency curve |
| `a2a-tree` | Cross-agent trace correlation (A2A protocol) |
| `mcp` | MCP server: expose traces as queryable tools for a debugging agent |

```
agent-strace setup [--redact] [--global]        Generate Claude Code hooks config
agent-strace hook <event>                       Handle a Claude Code hook event (internal)
agent-strace record -- <command>                Record an MCP stdio server session
agent-strace record-http <url> [--port N]       Record an MCP HTTP/SSE server session
agent-strace replay [session-id]                Replay a session (default: latest)
agent-strace replay [session-id] --limit N      Cap output at N events (fast inspection of large sessions)
agent-strace replay --format html [-o file]     Export a self-contained HTML replay viewer
agent-strace replay --expand-subagents          Inline subagent sessions under parent tool_call
agent-strace replay --tree                      Show session hierarchy without full replay
agent-strace list                               List all sessions
agent-strace explain [session-id]               Explain a session in plain English
agent-strace stats [session-id]                 Show tool call frequency and timing
agent-strace stats --include-subagents          Roll up stats across the full subagent tree
agent-strace inspect <session-id>               Dump full session as JSON
agent-strace export <session-id>                Export as JSON, CSV, NDJSON, or OTLP
agent-strace import <session.jsonl>             Import a Claude Code JSONL session log
agent-strace cost [session-id]                  Estimate token cost for a session
agent-strace diff <session-a> <session-b>       Compare two sessions structurally
agent-strace diff --compare <a> <b>             Side-by-side table with verdict
agent-strace diff --semantic <a> <b>            Compare sessions by outcome, not event order
agent-strace why [session-id] <event-number>    Trace the causal chain for an event
agent-strace audit [session-id] [--policy]      Check tool calls against a policy file
agent-strace audit-tools [--repo .] [--approved] Detect Shadow MCP servers and undeclared agent activity in any repo
agent-strace policy [--output file]             Generate .agent-scope.json from observed traces
agent-strace dashboard [--last N] [--html file] Aggregate stats and trends across sessions
agent-strace annotate <session-id> <offset>     Add notes, labels, or bookmarks to events
agent-strace token-budget <session-id>          Check token usage against model context limit
agent-strace replay [session-id] [--limit N]    Replay a session (--limit caps events shown)
agent-strace retention status                   Show session count, size, and what policy would delete
agent-strace retention clean [--dry-run]        Delete sessions that exceed retention limits
agent-strace sample --strategy worst --n 20     Export worst/diverse/random/recent sessions as JSONL
agent-strace export <session> --format otlp-genai  Export with OTel GenAI semantic conventions
agent-strace server [--port 4317] [--storage DIR]  Start a server-side event collector
agent-strace auto [--framework NAME] -- <cmd>      Run a command with auto-instrumentation
agent-strace watch [--timeout DURATION] [--budget $] [--on-death CMD] [--rules file]
                                                Watch a live session; kill/pause on rule breach
agent-strace share <session-id> [-o file]       Export a self-contained HTML report
agent-strace standup [--session id]             Standup report from session trace (no LLM)
agent-strace freshness [--scope glob]           Context freshness check vs last session
agent-strace oncall --rotation-start DATE       On-call readiness for agent-modified files
agent-strace curve [--export csv]               Personal agent cost-efficiency curve
agent-strace inflation [--compare m1,m2]        Token inflation calculator across model versions
agent-strace a2a-tree [session-id]              Visualise A2A agent call graph
```

### Import existing Claude Code sessions

Already ran a session without hooks? Import it directly from Claude Code's native JSONL logs:

```bash
# Discover available sessions
agent-strace import --discover

# Import a specific session
agent-strace import ~/.claude/projects/<project>/<session-id>.jsonl

# Then use it like any captured session
agent-strace replay <session-id>
agent-strace explain <session-id>
agent-strace stats <session-id>
```

Claude Code stores session logs in `~/.claude/projects/`. The import captures tool calls, token usage, subagent invocations, and session metadata.

### Explain a session

Plain-English breakdown of what the agent did, organized by phase, with retry and wasted-time detection:

```bash
agent-strace explain           # latest session
agent-strace explain abc123    # specific session
```

```
Session: abc123 (2m 05s, 47 events)

Phase 1: fix the auth module (0:00–0:05, 5 events)
  Read: AGENTS.md, src/auth.py

Phase 2: run tests — FAILED (0:05–1:20, 12 events)
  Ran: python -m pytest
  Ran: python -m pytest  ← retry

Phase 3: run tests (1:20–2:05, 8 events)
  Ran: uv run pytest

Files touched: 3 read, 0 written
Retries: 1 (wasted 1m 15s, 60% of session)
```

### Estimate cost

Token usage and dollar cost by phase. Flags wasted spend on failed phases.

```bash
agent-strace cost                          # latest session, sonnet pricing
agent-strace cost abc123 --model opus      # specific session and model
agent-strace cost abc123 --input-price 3.0 --output-price 15.0  # custom pricing
```

```
Session: abc123 — Estimated cost: $0.0042
Model: sonnet  |  8,200 input tokens, 3,100 output tokens

  Phase 1: fix the auth module          $0.0008  (19%)  ...
  Phase 2: run tests — FAILED           $0.0021  (50%)  ...  ← wasted
  Phase 3: run tests                    $0.0013  (31%)  ...

Wasted on failed phases: $0.0021 (50%)
```

Supported models: `sonnet` (default), `opus`, `haiku`, `gpt4`, `gpt4o`. Token counts are estimated from payload size (`len / 4`); see [ADR-0008](ADRs/0008-token-cost-estimation-heuristic.md) for details.

See [examples/session_analysis.md](examples/session_analysis.md) for a full walkthrough combining `import`, `explain`, and `cost`.

### Weekly spend digest (budget-report)

Aggregate cost across sessions for a configurable time window. Shows total spend, top sessions, cost by tool, and savings from watchdog budget ceilings.

```bash
# Last 7 days (default)
agent-strace budget-report

# Custom window
agent-strace budget-report --since 2026-05-01 --until 2026-05-23

# Markdown output (paste into Slack or email)
agent-strace budget-report --format markdown

# Machine-readable JSON
agent-strace budget-report --format json
```

Example output:

```
Budget Report — May 16 to May 23, 2026

Total spend:        $47.23  (↑ 12% vs prior period)
Sessions:           34      (↑ 3 vs prior period)
Avg cost/session:   $1.39

Top 5 most expensive sessions:
  1. a84664242afa  $8.43  refactor-auth                   2026-05-21
  2. bf1207728ee6  $6.21  add-test-coverage               2026-05-22
  3. c91ab3312fde  $4.87  fix-login-bug  ⚠ watchdog       2026-05-20

Cost by tool (estimated):
  Bash                  $18.43  (39%)
  Read                  $12.11  (26%)
  Write                  $9.87  (21%)

Sessions terminated by watchdog:  3  ($14.21 saved by budget ceiling)
```

Week-over-week delta is shown when prior-period data exists. The `--format markdown` output is designed to paste directly into Slack without editing.

### Static behaviour analysis (lint)

Analyse a session for known bad patterns — tool loops, reasoning spirals, budget proximity, context saturation, redundant reads, error-retry loops, and sessions that produced no output.

```bash
# Lint the latest session
agent-strace lint

# Lint a specific session
agent-strace lint <session-id>

# Lint all sessions from the last 7 days
agent-strace lint --all --since 7d

# Machine-readable output for CI
agent-strace lint <session-id> --format json

# Exit code 1 on any WARN or ERROR (CI gate)
agent-strace lint <session-id> --strict
```

Example output:

```
WARN   tool-loop              "Bash" called 7 times consecutively (events 34–41). Possible loop.
WARN   reasoning-spiral       4 consecutive LLM calls with no tool call (events 12–15). Agent may be over-reasoning.
ERROR  budget-proximity       Session reached 94% of a $5.00 budget ceiling. Consider raising or splitting the task.
INFO   context-saturation     Input tokens exceeded 80% of model context window at event 28.
INFO   redundant-read         "README.md" read 3 times in this session. Consider caching.

2 error(s), 2 warning(s), 2 info(s). Use --strict for non-zero exit on warnings.
```

Rules are configurable via `.agent-strace-lint.json`:

```json
{
  "tool-loop": { "threshold": 7 },
  "reasoning-spiral": { "enabled": false }
}
```

| Rule | Level | Trigger |
|---|---|---|
| `tool-loop` | WARN | Same tool called 5+ times consecutively |
| `reasoning-spiral` | WARN | 3+ consecutive LLM calls with no tool call |
| `budget-proximity` | ERROR | Session cost exceeded 90% of watchdog budget ceiling |
| `context-saturation` | INFO | Input tokens exceeded 80% of model context window |
| `redundant-read` | INFO | Same file read 3+ times in a session |
| `error-retry-loop` | WARN | Same tool errored and was retried 3+ times |
| `no-output` | WARN | Session completed with no write or file-modifying tool calls |

### Data retention

Enforce configurable retention policies to automatically delete old session data — required for GDPR, SOC 2, and internal data policies.

```bash
# Check current status and what policy would delete
agent-strace retention status

# Preview what would be deleted (no changes made)
agent-strace retention clean --dry-run

# Delete sessions older than 30 days
agent-strace retention clean --max-age-days 30

# Keep only the 1000 most recent sessions
agent-strace retention clean --max-sessions 1000

# Delete oldest sessions when storage exceeds 500 MB
agent-strace retention clean --max-size-mb 500
```

Configure via `.agent-strace.yaml`:

```yaml
retention:
  max_age_days: 30
  max_sessions: 1000
  max_size_mb: 500
  on_delete: log    # log deletions to .agent-traces/retention.log
```

Policies are applied in order: age → count → size. Deletions are logged with session ID and timestamp (not content).

### Trace anonymization

Strip identifying information from traces at export time — original session data is never modified. Complements secret redaction (which strips secrets at capture time).

```bash
# Preview what would be anonymized
agent-strace export SESSION_ID --anonymize --dry-run

# Export with anonymization applied
agent-strace export SESSION_ID --anonymize --output trace-anon.json
```

Anonymized by default:
- Home directory paths → `~/relative/path`
- Hostnames → `<hostname>`
- OS usernames → `<user>`
- Email addresses → `<email>`

Add custom patterns via `.agent-strace/anonymize.yaml`:

```yaml
rules:
  - pattern: "ACME Corp"
    replacement: "<company>"
  - pattern: "192\.168\.\d+\.\d+"
    replacement: "<internal-ip>"
```

### Secret redaction

Strip API keys, tokens, and credentials from traces before they hit disk.

```bash
# Stdio proxy with redaction
agent-strace record --redact -- npx -y @modelcontextprotocol/server-filesystem /tmp

# HTTP proxy with redaction
agent-strace record-http https://mcp.example.com --redact
```

Detected patterns: OpenAI (`sk-*`), GitHub (`ghp_*`, `github_pat_*`), AWS (`AKIA*`), Anthropic (`sk-ant-*`), Slack (`xox*`), JWTs, Bearer tokens, connection strings (`postgres://`, `mysql://`), and any value under keys like `password`, `secret`, `token`, `api_key`, `authorization`.

### HTTP/SSE proxy

For MCP servers that use HTTP transport:

```bash
# Proxy a remote MCP server
agent-strace record-http https://mcp.example.com --port 3100

# Your agent connects to http://127.0.0.1:3100 instead of the remote server
# All JSON-RPC messages are captured, tool call latency is measured
```

The proxy forwards POST `/message` and GET `/sse` to the remote server, capturing every JSON-RPC message in both directions.

### Replay output

A real Claude Code session captured with hooks:

<details><summary>Session Summary</summary>
<p>

```
Session Summary
──────────────────────────────────────────────────
  Session:    201da364-edd6-49
  Command:    claude-code (startup)
  Agent:      claude-code
  Duration:   112.54s
  Tool calls: 8
  Errors:     3
──────────────────────────────────────────────────

+  0.00s ▶ session_start
+  0.07s 👤 user_prompt
              "how many tests does this project have? run them and tell me the results"
+  3.55s → tool_call Glob
              **/*.test.*
+  3.55s → tool_call Glob
              **/test_*.*
+  3.60s ← tool_result Glob (51ms)
+  6.06s → tool_call Bash
              $ python -m pytest tests/ -v 2>&1
+ 27.65s ✗ error Bash
              Command failed with exit code 1
+ 29.89s → tool_call Bash
              $ python3 -m pytest tests/ -v 2>&1
+ 40.56s ✗ error Bash
              No module named pytest
+ 45.96s → tool_call Bash
              $ which pytest || ls /Users/siddhant/Desktop/test-agent-trace/ 2>&1
+ 46.01s ← tool_result Bash (51ms)
+ 48.18s → tool_call Read
              /Users/siddhant/Desktop/test-agent-trace/pyproject.toml
+ 48.23s ← tool_result Read (43ms)
+ 51.43s → tool_call Bash
              $ uv run --with pytest pytest tests/ -v 2>&1
+1m43.67s ← tool_result Bash (5.88s)
              75 tests, all passing in 3.60s
+1m52.54s 🤖 assistant_response
              "75 tests, all passing in 3.60s. Breakdown by file: ..."
```

Tool calls show actual values: commands, file paths, glob patterns. Errors show what failed. Assistant responses are stripped of markdown.

</p>
</details> 

### Filtering

```bash
# Show only tool calls and errors
agent-strace replay --filter tool_call,error

# Replay with timing (watch it unfold)
agent-strace replay --live --speed 2
```

### Export

```bash
# JSON array
agent-strace export a84664 --format json

# CSV (for spreadsheets)
agent-strace export a84664 --format csv

# NDJSON (for streaming pipelines)
agent-strace export a84664 --format ndjson
```

## Trace format

Traces are stored as directories in `.agent-traces/`:

```
.agent-traces/
  a84664242afa4516/
    meta.json        # session metadata
    events.ndjson    # newline-delimited JSON events
```

Each event is a single JSON line:

```json
{
  "event_type": "tool_call",
  "timestamp": 1773562735.09,
  "event_id": "bf1207728ee6",
  "session_id": "a84664242afa4516",
  "data": {
    "tool_name": "read_file",
    "arguments": {"path": "src/auth.py"}
  }
}
```

### Event types

| Type | Description |
|------|-------------|
| `session_start` | Trace session began |
| `session_end` | Trace session ended |
| `user_prompt` | User submitted a prompt to the agent |
| `assistant_response` | Agent produced a text response |
| `tool_call` | Agent invoked a tool |
| `tool_result` | Tool returned a result |
| `llm_request` | Agent sent a prompt to an LLM |
| `llm_response` | LLM returned a completion |
| `file_read` | Agent read a file |
| `file_write` | Agent wrote a file |
| `decision` | Agent chose between alternatives |
| `error` | Something failed |

Events link to each other. A `tool_result` has a `parent_id` pointing to its `tool_call`. This lets you measure latency per tool and trace the full call chain.

## Use with Claude Code, Cursor, Windsurf

### Claude Code (hooks, recommended)

Captures the full session: prompts, responses, and every tool call. See [examples/claude_code_config.md](examples/claude_code_config.md) for the full config.

```bash
agent-strace setup                    # per-project config
agent-strace setup --redact --global  # all projects, with secret redaction
```

### Cursor

Edit `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per-project):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "agent-strace",
      "args": ["record", "--name", "filesystem", "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### Windsurf

Edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "agent-strace",
      "args": ["record", "--name", "filesystem", "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### Any MCP client

The pattern is the same for any tool that uses MCP over stdio:

1. Replace the server `command` with `agent-strace`
2. Prepend `record --name <label> --` to the original args
3. Use the tool normally
4. Run `agent-strace replay` to see what happened

See the [examples/](examples/) directory for full config files.

### Subagent tracing

When an agent spawns subagents (e.g. Claude Code's Agent tool), sessions are linked into a parent-child tree. Replay the full tree inline or view a compact hierarchy:

```bash
# Inline replay: subagent events appear under the parent tool_call that spawned them
agent-strace replay --expand-subagents

# Compact hierarchy: session IDs, durations, tool counts
agent-strace replay --tree

# Aggregated stats across the full tree (tokens, tool calls, errors)
agent-strace stats --include-subagents
```

```
▶ session_start  a84664242afa  agent=claude-code  depth=0
  + 0.00s  👤 "refactor the auth module"
  + 1.23s  → tool_call  Agent  "extract helper functions"
│  ▶ session_start  b12345678901  agent=claude-code  depth=1
│    + 0.00s  → tool_call  Read  src/auth.py
│    + 0.12s  ← tool_result
│    + 0.45s  → tool_call  Write  src/auth_helpers.py
│    + 0.51s  ■ session_end
  + 3.10s  ← tool_result
  + 3.20s  ■ session_end
```

Subagent sessions are linked via `parent_session_id` and `parent_event_id` in session metadata. Existing sessions without these fields are unaffected.

### Session diff

Compare two sessions structurally. Useful for understanding why the same prompt produces different results across runs, or comparing a broken session against a known-good one. Phases are aligned by label using LCS, then per-phase differences in files touched, commands run, and outcomes are reported:

```bash
agent-strace diff abc123 def456
```

```
Comparing: abc123 vs def456

Diverged at phase 2:

  Phase 2: run tests
    abc123 only:  $ python -m pytest
    def456 only:  $ uv run pytest

  abc123: 4m 12s, 47 events, 8 tools, 2 retries
  def456: 2m 05s, 31 events, 5 tools, 0 retries
```

### Causal chain (why)

Trace backwards from any event to find what caused it. Run `agent-strace replay <session-id>` first. The `#N` numbers in the left column are the event numbers:

```bash
agent-strace why abc123 4
```

```
Why did event #4 happen?

  #  4  tool_call: Bash  $ pytest tests/

Causal chain (root → target):

    #  1  user_prompt: "run the test suite"
       (prompt at #1 triggered this)
  ←  #  3  error: exit 1
       (retry after error at #3)
  ←  #  4  tool_call: Bash  $ pytest tests/
```

Causal links are detected via `parent_id` (tool_result → tool_call), error→retry matching (same tool and command), path references (tool_result text containing a path used by a later call), and read→write pairs on the same file.

### Permission audit

Check every tool call against a policy file. Flags sensitive file access (`.env`, `*.pem`, `.ssh/*`, `.github/workflows/*`, etc.) even without a policy:

```bash
agent-strace audit                          # latest session, no policy required
agent-strace audit abc123 --policy .agent-scope.json

# In CI: fail the build if the agent accessed anything outside policy
agent-strace audit --policy .agent-scope.json || exit 1
```

```
AUDIT: Session abc123 (47 events, 23 tool calls)

✅ Allowed (19):
  Read src/auth.py
  Ran: uv run pytest

⚠️  No policy (2):
  Read README.md  (no file read policy for this path)

❌ Violations (2):
  Read .env  ← denied by files.read.deny
  Ran: curl https://example.com  ← denied by commands.deny

🔐 Sensitive files accessed (1):
  Read .env  (event #12)
```

Exits with code 1 when violations are found. Usable in CI.

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

Glob patterns support `**` as a recursive wildcard. File read policy applies to `Read`, `View`, `Grep`, and `Glob` tool calls. Network policy checks URLs embedded in `Bash` commands.

### Auto-generate a policy from your traces

Let agent-trace observe a few sessions and generate `.agent-scope.json` for you:

```bash
# Dry-run: print the suggested policy without writing anything
agent-strace policy

# Write it to disk
agent-strace policy --output .agent-scope.json

# Observe a specific set of sessions
agent-strace policy --last 20 --output .agent-scope.json
```

The generated policy covers every file path and command the agent used, collapsed into glob patterns. Review it, tighten the deny list, and commit it alongside your code.

### Optimize instruction files from trace failures

Cluster failures by root cause and propose concrete additions to `AGENTS.md`, `CLAUDE.md`, or any instruction file. Three built-in heuristic patterns require no LLM.

```bash
# Show proposed additions to AGENTS.md (dry run, no writes)
agent-strace optimize --target AGENTS.md

# Analyze a dataset of failures
agent-strace optimize --dataset auth-failures --target AGENTS.md

# Apply changes
agent-strace optimize --target AGENTS.md --apply

# Use a local Ollama model for LLM-assisted clustering
agent-strace optimize --target AGENTS.md \
  --base-url http://localhost:11434/v1 \
  --model llama3 \
  --api-key ollama \
  --apply
```

Built-in heuristic patterns (no LLM required):

| Pattern | Detection | Proposed fix |
|---|---|---|
| `blind-retry` | Same tool called 3+ times consecutively | Add retry policy to AGENTS.md |
| `error-no-change` | Tool retried after error with no write in between | Add error-handling rule |
| `wide-blast-radius` | More than 8 distinct files written in one session | Add scope discipline rule |

When `OPENAI_API_KEY` and `OPENAI_BASE_URL` are set (or `--api-key` / `--base-url`), the command uses an LLM to cluster failures and generate more targeted proposals. Falls back to heuristics if the LLM call fails.

### PII masking

Sensitive data is masked before it hits disk. Useful when tracing agents that handle user data or credentials.

```bash
# Stdio proxy with masking
agent-strace record --mask -- npx -y @modelcontextprotocol/server-filesystem /tmp

# HTTP proxy with masking
agent-strace record-http https://mcp.example.com --mask
```

Masked by default: email addresses, phone numbers, credit card numbers, US Social Security Numbers, and AWS ARNs. You can also call `mask_event_data()` directly to sanitise events from an existing session before sharing or exporting them.

### Eval scoring

Score a session against configurable criteria. Built-in scorers require no LLM. They run on trace structure alone.

```bash
# Score the latest session (uses .agent-evals.yaml if present)
agent-strace eval

# Score a specific session
agent-strace eval abc123

# JSON output
agent-strace eval abc123 --format json

# Compare two sessions
agent-strace eval compare abc123 def456
```

Built-in scorers:

| Scorer | What it checks |
|---|---|
| `no_errors` | Session had zero ERROR events |
| `cost_under` | Estimated cost stayed below a dollar threshold |
| `files_scoped` | All file operations were within allowed paths |
| `duration_under` | Session completed within a time limit |
| `regex` | A pattern matched in agent responses |

Configure scorers in `.agent-evals.yaml`:

```yaml
scorers:
  - type: no_errors
    threshold: 1.0
  - type: cost_under
    max_dollars: 0.10
    threshold: 1.0
  - type: files_scoped
    allowed_paths: ["src/", "tests/"]
    threshold: 0.90

thresholds:
  pass: 0.85
  warn: 0.70
```

#### CI gate

Block merges when agent quality regresses:

```bash
agent-strace eval ci
```

Exits non-zero if any scorer fails. Add to GitHub Actions:

```yaml
- name: Eval agent session
  run: agent-strace eval ci
  env:
    PYTHONPATH: src
```

### Multi-session dashboard

Get an aggregate view across all your sessions. Useful for spotting trends, outliers, and cost spikes without opening each session individually.

```bash
agent-strace dashboard                    # all sessions
agent-strace dashboard --last 20          # last 20 sessions
agent-strace dashboard --since 2024-06-01 # since a date
agent-strace dashboard --html report.html # self-contained HTML export
```

The terminal view shows total tool calls, errors, tokens, and estimated cost, plus ASCII sparkline charts for each metric over time and a top-tools frequency table. The HTML export is self-contained. No server needed.

### Dataset auto-sampler

Export the sessions most useful for regression suites and eval datasets — without manual inspection.

```bash
# Export the 20 worst-performing sessions (highest error/retry/cost)
agent-strace sample --strategy worst --n 20 --output regression.jsonl

# Export 10 sessions that maximise behavioral variety
agent-strace sample --strategy diverse --n 10 --output diverse.jsonl

# Export the 5 most recent sessions
agent-strace sample --strategy recent --n 5 --output recent.jsonl

# Random sample, reproducible with a seed
agent-strace sample --strategy random --n 15 --seed 42 --output random.jsonl

# Skip sessions with identical tool call sequences
agent-strace sample --strategy worst --n 20 --deduplicate --output regression.jsonl
```

Output is JSONL — one session per line — with full event data and a score breakdown. Compatible with LangSmith, Braintrust, and any custom eval framework.

### Eval trend dashboard

See whether your agent is getting better or worse over time. Reads eval scores and behavioral metrics from session events, then renders a self-contained HTML report with inline SVG charts.

```bash
# Terminal summary
agent-strace dashboard --trend --since 30d

# Self-contained HTML report (no CDN, no JS libraries)
agent-strace dashboard --trend --since 30d --html trend-report.html

# Add a timeline annotation (appears as a vertical marker on all charts)
agent-strace dashboard annotate --date 2026-05-10 --note "Added retry policy to AGENTS.md"
```

The HTML report shows:
- **Eval quality**: pass rate per judge over time, with annotation markers for config changes
- **Behavioral metrics**: error rate, retry rate, cost, session duration as sparklines
- **Recent sessions table**: eval scores inline, click any row to open the full replay

The file is fully self-contained. Attach it to a PR, commit it as a weekly snapshot, or share it with someone who doesn't have agent-strace installed.

### Session attribution

Every session records who and what spawned it: OS user, detected agent provider, git repo and branch, and the chain of parent processes.

```bash
agent-strace show SESSION_ID
# Attribution
#   User:     alice
#   Provider: claude-code
#   Branch:   feat/my-feature
#   Commit:   a1b2c3d
#   CWD:      /home/alice/projects/myapp
```

Detected providers: `claude-code`, `cursor`, `github-copilot`, `cody`, `continue`, and a generic `mcp-client` fallback. Attribution is collected automatically. Nothing to configure.

### Replay annotations

Add notes, labels, and bookmarks to any event. Useful for code review, debugging, and building eval datasets.

```bash
# Add a note to event #12
agent-strace annotate SESSION_ID 12 --note "Why did it call bash here instead of write_file?"

# Tag an event
agent-strace annotate SESSION_ID 12 --label regression

# Bookmark for quick navigation in the HTML viewer
agent-strace annotate SESSION_ID 12 --bookmark

# List all annotations
agent-strace annotate SESSION_ID --list

# Remove one
agent-strace annotate SESSION_ID 12 --delete ANNOTATION_ID
```

Annotations persist alongside the session and appear as a bookmarks sidebar in shared HTML reports. They're also useful for building eval datasets: label sessions as `pass` / `fail` / `interesting` and filter on those labels later.

### Token budget tracking

Long-running agents can burn through a model's context window without warning. The token budget command shows how close you are before you hit the limit.

```bash
agent-strace token-budget SESSION_ID
agent-strace token-budget SESSION_ID --model claude-3-5-sonnet
agent-strace token-budget SESSION_ID --model gpt-4o --warn-at 75
```

In watch mode, the same threshold applies in real time:

```bash
agent-strace watch --max-context-pct 80 SESSION_ID
```

Supported models and their limits:

| Model | Context |
|---|---|
| claude-3-5-sonnet | 200k tokens |
| claude-3-opus | 200k tokens |
| gpt-4o | 128k tokens |
| gpt-4-turbo | 128k tokens |
| gemini-1.5-pro | 1M tokens |

Pass `--limit` to set a custom ceiling for any other model.

### Semantic session diff

Compare two sessions by *outcome*, not raw event order. Useful for regression testing agent behaviour across model versions or prompt changes.

```bash
agent-strace diff SESSION_A SESSION_B --semantic
```

```
Semantic diff: SESSION_A vs SESSION_B

Tools added:    write_file
Tools removed:  bash
Δ tool calls:   +3
Δ errors:       -2
Δ tokens:       +1,200
Outcome:        improved (fewer errors, same task completed)
```

Export a structured JSON report for CI assertions:

```bash
agent-strace diff SESSION_A SESSION_B --semantic --eval-config eval.json
```

### Rich side-by-side comparison

`--compare` produces a structured table across cost, duration, tool calls, redundant reads, context resets, files modified, and errors. The verdict is deterministic and requires no LLM.

```bash
agent-strace diff SESSION_A SESSION_B --compare
```

New metrics: **redundant reads** (files read more than once), **context resets** (LLM requests separated by >120s), **approach divergence** (first phase pairs where behaviour differs). Useful for asserting on in CI.

### Watchdog mode — timeout, budget ceiling, and post-mortem

Enforce a wall-clock timeout and/or token-cost ceiling on any session. When either limit is breached the agent process is terminated and a structured `watchdog-postmortem.json` is written to the session directory. An optional `--on-death` command is invoked with the post-mortem path.

```bash
# Kill after 30 minutes
agent-strace watch --timeout 30m --on-violation kill SESSION_ID

# Kill when spend exceeds $5
agent-strace watch --budget 5.00 --on-violation kill SESSION_ID

# Both limits, with a recovery script
agent-strace watch \
  --timeout 30m \
  --budget 5.00 \
  --on-violation kill \
  --on-death "python recover.py --post-mortem {post_mortem_path}" \
  SESSION_ID
```

`--timeout` accepts human-readable durations: `30s`, `5m`, `2h`, `1h30m`.

The `watchdog-postmortem.json` written on kill contains:

```json
{
  "session_id": "abc123",
  "reason": "DurationWatcher: 1800s elapsed",
  "elapsed_seconds": 1800.0,
  "cost_at_death": 2.34,
  "last_tool_call": { "tool_name": "Bash", "arguments": { "command": "pytest" } },
  "last_llm_response": { "model": "claude-3-5-sonnet", "content": "..." },
  "recovery_context": "Session abc123 was terminated after 1800s ($2.34 spent). ..."
}
```

### Kill switch for runaway sessions

Add a declarative rules file to `agent-strace watch` to pause, kill, or alert when a session crosses a threshold. The agent stops when a rule fires. No prompt, no retry, no damage.

```bash
agent-strace watch --rules .watch-rules.json
agent-strace watch --rules .watch-rules.json --dry-run  # evaluate without acting
```

Example `.watch-rules.json`:

```json
[
  { "condition": "cost_usd", "threshold": 0.50, "action": "kill" },
  { "condition": "file_path", "glob": "**/production.env", "action": "kill" },
  { "condition": "files_modified", "threshold": 30, "action": "pause" }
]
```

**Rule conditions:** `files_modified`, `cost_usd`, `consecutive_test_failures`, `duration_minutes`, `file_path` (glob).

**Actions:**
- `pause`: SIGSTOP the agent process (resume with SIGCONT)
- `kill`: SIGTERM, then SIGKILL after 5s; auto-generates a postmortem
- `alert`: log only, no interruption

### Push-based event streaming

Stream events to an external HTTP endpoint in real-time as they arrive during a watched session. Events are batched and POSTed as [NDJSON](https://ndjsonl.org) (`application/x-ndjson`), so any HTTP server or log aggregator can consume them.

```bash
# Stream all events to a collector
agent-strace watch --stream-to https://collector.example.com/events SESSION_ID

# Tune batch size and flush interval
agent-strace watch \
  --stream-to https://collector.example.com/events \
  --stream-batch-size 20 \
  --stream-flush-interval 5.0 \
  SESSION_ID
```

Each POST body contains one JSON object per line:

```
{"event_type":"tool_call","timestamp":1700000001.0,"session_id":"abc123","data":{...}}
{"event_type":"llm_response","timestamp":1700000002.5,"session_id":"abc123","data":{...}}
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--stream-to URL` | — | HTTP endpoint to POST events to |
| `--stream-batch-size N` | `10` | Max events per POST |
| `--stream-flush-interval S` | `2.0` | Max seconds between flushes |

HTTP failures are logged to stderr but never interrupt the watch loop. The background flush thread is a daemon and is stopped cleanly when the session ends or the watcher exits.

### Behavioral drift detection

Detect when agent behavior has shifted across sessions without an LLM. Computes a behavioral fingerprint across six dimensions (tool mix, error rate, retry pattern, blast radius, session duration, decision depth) and measures divergence from a baseline using Jensen-Shannon divergence.

```bash
# Detect drift over the last 30 days (splits window in half automatically)
agent-strace drift --since 30d

# Compare against a saved baseline
agent-strace drift --baseline .agent-traces/baselines/behavior-main.json

# Save current fingerprint as a baseline
agent-strace drift --since 30d --save-baseline .agent-traces/baselines/behavior-main.json

# JSON output for CI
agent-strace drift --since 30d --format json
```

Exits non-zero when the overall drift score exceeds `--threshold` (default: `0.20`). Commit baseline fingerprints alongside your agent config. They're under 2KB.

Six dimensions tracked:

| Dimension | Drift signal |
|---|---|
| Tool mix | Agent suddenly calling Bash 40% more often |
| Error rate | New class of errors appearing |
| Retry pattern | Agent retrying more after a model update |
| Blast radius | Agent touching more files per task |
| Session duration | Sessions getting longer |
| Decision depth | Agent making fewer explicit decisions |

### Shadow MCP detection

Detect Shadow MCP servers and undeclared agent activity in any repo. No network calls, no API keys. A [CSA survey of 418 security professionals](https://cloudsecurityalliance.org/press-releases/2026/04/21/new-cloud-security-alliance-survey-reveals-82-of-enterprises-have-unknown-ai-agents-in-their-environments) found 82% of enterprises discovered at least one AI agent their security team didn't know about in the past year. `audit-tools` finds yours.

```bash
agent-strace audit-tools
agent-strace audit-tools --repo . --since "90 days ago" --approved cursor,copilot
```

Detected tools: Claude Code, Cursor, GitHub Copilot, Codex/ChatGPT, Windsurf, Aider, Gemini CLI. Identified via file signals (`.cursorrules`, `CLAUDE.md`, `.github/copilot-instructions.md`, etc.) and commit message patterns. Flags unapproved tools, unknown LLM API endpoints in `.env` history, and PII patterns in recently committed files.

### HTML session replay viewer

Generate a single-file HTML viewer for any session. No server, no dependencies. Open in any browser.

```bash
agent-strace replay --format html
agent-strace replay --format html --output review.html SESSION_ID
```

The viewer includes an animated event timeline, scrubber bar, running cost counter, click-to-expand event detail, color-coded event types, and dark theme. All event data is embedded as a JSON constant. Useful for attaching to PR reviews.

### Standup report

Generate a structured standup from a session trace. No LLM required.

```bash
agent-strace standup
agent-strace standup --session SESSION_ID
```

Report covers: files read and modified, approaches tried (including abandoned ones detected from retry patterns), new dependencies added, TODO/FIXME comments written, large changes and auth/migration patterns to review, and session stats (tool calls, retries, errors).

### Context freshness check

Check how stale the agent's last view of the codebase is before handing it a task.

```bash
agent-strace freshness
agent-strace freshness --since 2026-04-01 --scope "src/**"
```

Reports files changed since the last session, per-file change type and line count, a freshness score 0–100, and estimated catch-up reading time. Scope is auto-detected from `CLAUDE.md` / `AGENTS.md`, or overridden with `--scope`.

### On-call readiness

Cross-reference agent-modified files against git history to find gaps before a rotation.

```bash
agent-strace oncall --rotation-start 2026-04-25
agent-strace oncall --rotation-start 2026-04-25 --scope "src/payments/**"
```

For each file the agent has written in the last N days: how long ago it was modified, lines changed, estimated reading time, and total catch-up time before rotation.

### Cost-efficiency curve

See which task types are worth delegating to an agent.

```bash
agent-strace curve
agent-strace curve --min-sessions 10 --export csv
```

Sessions are classified into 10 task types (unit tests, debugging, refactoring, architecture, etc.) and compared against a community sweet-spot benchmark. Verdict per type: **efficient / over sweet spot / do this yourself**. Potential monthly savings are calculated for types running above 1.5× their sweet spot.

### Token inflation calculator

Measure the tokenizer cost impact of switching model versions before committing to an upgrade. No API calls required.

```bash
agent-strace inflation
agent-strace inflation --compare claude-opus-4-6,claude-opus-4-7 --sessions 30
```

Applies per-model inflation factors to stored session content and breaks down the impact by content type (system prompt, tool definitions, user messages, assistant messages). Projects per-session, daily, and monthly cost delta.

| Model | Factor |
|---|---|
| claude-opus-4-7 | 1.38× (community median: 1.3–1.47×, April 2026) |
| gpt-4o | 1.05× (cl100k_base → o200k_base) |

### A2A protocol support

First-class support for agent-to-agent calls following the Google A2A spec. A2A calls are captured as `TOOL_CALL` events with `event_subtype=a2a_call`, backward-compatible with all existing replay and export tooling.

```bash
agent-strace a2a-tree
agent-strace a2a-tree SESSION_ID --format json
```

Builds the full agent call graph by following `sub_session_id` links and `parent_session_id` back-references. Renders as an ASCII tree or exports as OTLP spans for Jaeger, Tempo, or any OpenTelemetry backend.

## Use with security-critical codebases

When AI coding agents work on codebases that handle secrets, attestation logic, or cryptographic material, agent-strace gives you two things: an audit trail of every file touched and every command run, and redaction of secrets before they reach any log.

### Recommended setup for sensitive repos

Add `.claude/settings.json` to the repo root and commit it. Every developer gets the same instrumentation:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": ".*",
      "hooks": [{ "type": "command", "command": "agent-strace hook pre-tool" }]
    }],
    "PostToolUse": [{
      "matcher": ".*",
      "hooks": [{ "type": "command", "command": "agent-strace hook post-tool" }]
    }]
  }
}
```

Or use the setup command:

```bash
cd your-sensitive-repo
agent-strace setup --redact
```

### Secret redaction for TEE and confidential computing stacks

For codebases that handle TEE secrets, the following patterns are redacted automatically:

| Secret type | Pattern matched |
|-------------|----------------|
| EKM shared secrets | 64-char hex strings (e.g. `EKM_SHARED_SECRET`) |
| Bearer tokens | `Bearer [A-Za-z0-9+/=]{20,}` |
| Anthropic API keys | `sk-ant-...` |
| AWS credentials | `AKIA...`, `aws_secret_access_key` |
| Private keys | PEM blocks |

If your codebase uses custom secret formats, add patterns via `--redact-pattern`:

```bash
agent-strace setup --redact --redact-pattern "ATTESTATION_KEY=[A-Fa-f0-9]{64}"
```

### Example: scoping agents away from sensitive components

Combine with [agentic-authz](https://github.com/Siddhant-K-code/agentic-authz) to block agents from security-critical components entirely, and use agent-strace to audit everything they do access:

```
Agent scope:        frontend/ only (enforced by OpenFGA: no tuple = no access)
agent-strace scope: all tool calls logged, secrets redacted, exported to Grafana
```

Any attempt by the agent to read `cvm/attestation-service/` or `cvm/auth-service/` is blocked at the authorization layer before it reaches the filesystem. agent-strace logs the denied attempt with the reason.

---

## Auto-instrumentation

Instrument any supported agent framework without modifying application code.

```bash
# Instrument a specific framework
agent-strace auto --framework langchain -- python my_agent.py

# Auto-detect all installed frameworks
agent-strace auto --detect -- python my_agent.py

# Via environment variable (no CLI wrapper needed)
AGENT_STRACE_AUTO_INSTRUMENT=langchain,litellm python my_agent.py

# Or in code
from agent_trace.integrations import instrument_langchain
instrument_langchain()
```

Supported frameworks:

| Framework | Install | What's traced |
|---|---|---|
| OpenAI Agents SDK | `pip install agent-strace[openai-agents]` | Runner.run, FunctionTool calls |
| LangChain / LangGraph | `pip install agent-strace[langchain]` | BaseTool._run, BaseChatModel._generate |
| LiteLLM | `pip install agent-strace[litellm]` | litellm.completion |
| Anthropic SDK | `pip install anthropic` | messages.create |
| OpenAI SDK | `pip install openai` | chat.completions.create |
| AWS Strands | `pip install agent-strace[strands]` | Agent.__call__, BaseTool.invoke |

Each integration is an optional extra — the core package stays dependency-free (ADR-0003).

## Server-side event collector

Run a central collector so agents in containers, CI, and serverless functions can send traces over the network — no local disk required.

```bash
# Start the collector
agent-strace server --port 4317 --storage ./traces

# Agents point to it via environment variable — no code changes required
AGENT_STRACE_ENDPOINT=http://collector:4317 python my_agent.py
```

The server writes traces in the same `.agent-traces/` format as local mode. All existing CLI commands work against its storage.

### API

| Method | Path | Description |
|---|---|---|
| `POST` | `/events` | Receive a batch of NDJSON events |
| `POST` | `/sessions` | Create or update session metadata |
| `GET` | `/sessions` | List all sessions |
| `GET` | `/sessions/<id>/events` | Stream events for a session |
| `GET` | `/health` | Liveness check |

### Docker

```dockerfile
FROM python:3.12-slim
RUN pip install agent-strace
ENV AGENT_STRACE_STORAGE=/data
VOLUME /data
EXPOSE 4317
CMD ["agent-strace", "server", "--port", "4317"]
```

No authentication in v1 — intended for internal/private network use. Add a reverse proxy (nginx, Caddy) for auth.

## Production tracing (OTLP export)

Export sessions as OpenTelemetry spans to your existing observability stack. Sessions become traces. Tool calls become spans with duration and inputs. Errors get exception events. No new dependencies.

### OTel GenAI semantic conventions

Use `--format otlp-genai` to export with strict [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). This produces AI-native spans that populate token usage charts, cost views, and LLM dashboards in Datadog, Grafana, and Honeycomb automatically.

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:4318
```

Key differences from `--format otlp`:

| Aspect | `--format otlp` | `--format otlp-genai` |
|---|---|---|
| LLM calls | Events on root span | `gen_ai.client.operation` child spans |
| Tool calls | `tool/<name>` spans | `gen_ai.tool.call/<name>` spans |
| Root span | `agent.name` attribute | `gen_ai.agent.id` + `gen_ai.agent.name` |
| Errors | Custom error span | OTel `exception` event format |

`--format otlp` is unchanged for backwards compatibility.

### Datadog

```bash
# Via the Datadog Agent's OTLP receiver (port 4318)
agent-strace export <session-id> --format otlp \
  --endpoint http://localhost:4318

# Or via Datadog's OTLP intake directly
agent-strace export <session-id> --format otlp \
  --endpoint https://http-intake.logs.datadoghq.com:443 \
  --header "DD-API-KEY: $DD_API_KEY"
```

### Honeycomb

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://api.honeycomb.io \
  --header "x-honeycomb-team: $HONEYCOMB_API_KEY" \
  --service-name my-agent
```

### New Relic

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://otlp.nr-data.net \
  --header "api-key: $NEW_RELIC_LICENSE_KEY"
```

### Splunk

```bash
agent-strace export <session-id> --format otlp \
  --endpoint https://ingest.<realm>.signalfx.com \
  --header "X-SF-Token: $SPLUNK_ACCESS_TOKEN"
```

### Grafana Tempo / Jaeger

```bash
# Local collector
agent-strace export <session-id> --format otlp \
  --endpoint http://localhost:4318
```

### Langfuse export

Export sessions and eval scores to [Langfuse](https://langfuse.com). Sessions appear as Traces, tool calls as Spans, LLM calls as Generations, and eval scores as Langfuse Scores.

```bash
# Set credentials
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

# Export latest session with eval scores
agent-strace export --scores --backend langfuse

# Export last 7 days
agent-strace export --since 7d --scores --backend langfuse

# Self-hosted Langfuse
agent-strace export --scores --backend langfuse \
  --langfuse-host https://langfuse.your-domain.com
```

### Export behavioral metrics to any OTLP backend

Export per-session behavioral metrics as OTLP gauge metrics. Compatible with Datadog, Honeycomb, Grafana, New Relic, and any OpenTelemetry backend.

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io

agent-strace export --metrics --backend otlp --since 30d
```

Metrics exported:

| Metric | Description |
|---|---|
| `agent_strace.session.cost_usd` | Estimated cost per session |
| `agent_strace.session.error_rate` | Errors / tool calls |
| `agent_strace.session.retry_rate` | Consecutive same-tool retries / tool calls |
| `agent_strace.session.blast_radius` | Distinct files written |
| `agent_strace.session.duration_s` | Wall-clock session duration |
| `agent_strace.eval.score` | Judge score per session (one per judge, with `judge=` attribute) |

### Dump OTLP JSON without sending

```bash
# Inspect the OTLP payload
agent-strace export <session-id> --format otlp > trace.json
```

### How it maps

| agent-trace | OpenTelemetry |
|---|---|
| session | trace |
| tool_call + tool_result | span (with duration) |
| error | span with error status + exception event |
| user_prompt | event on root span |
| assistant_response | event on root span |
| session_id | trace ID |
| event_id | span ID |
| parent_id | parent span ID |

## Debug with MCP

`agent-strace mcp` starts an MCP server that exposes your session store as queryable tools. Any MCP-compatible client (Claude Code, Cursor, VS Code Copilot) can query traces conversationally. The debugging agent reads its own execution history and surfaces what went wrong.

```bash
agent-strace mcp
```

**Claude Code config** (`.claude/settings.json`):

```json
{
  "mcpServers": {
    "agent-trace": {
      "command": "agent-strace",
      "args": ["mcp"]
    }
  }
}
```

**Cursor config** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "agent-trace": {
      "command": "agent-strace",
      "args": ["mcp"]
    }
  }
}
```

Once connected, you can ask the debugging agent questions like:

> "Look at the most recent session and tell me why it called bash three times in a row."
> "Which files did the agent write in session abc123 that it didn't write in def456?"
> "Find all sessions where the agent hit an error after calling npm test."

### MCP tools

| Tool | Description |
|---|---|
| `list_sessions` | List captured sessions with metadata (timestamp, tool calls, cost, tokens) |
| `get_session` | Full event stream for a session, with optional event type filter |
| `search_events` | Filter events by tool name, file path, exit code, or error flag across sessions |
| `get_session_summary` | Plain-English phase breakdown: what the agent did, files touched, retries |
| `diff_sessions` | Compare two sessions: tool call delta, file overlap, cost delta, error delta |

### Example interactions

```
# List recent sessions
list_sessions(limit=5)

# Get all errors from a session
search_events(session_id="abc123", has_error=true)

# Find all sessions where the agent wrote to package-lock.json
search_events(file_path="package-lock.json")

# Compare two sessions after changing AGENTS.md
diff_sessions(session_a="before_change", session_b="after_change")

# Get a plain-English summary of what went wrong
get_session_summary(session_id="abc123")
```

## How it works

### Claude Code hooks

```
Claude Code agentic loop
  ├── UserPromptSubmit   → agent-strace hook user-prompt
  ├── PreToolUse         → agent-strace hook pre-tool
  ├── PostToolUse        → agent-strace hook post-tool
  ├── PostToolUseFailure → agent-strace hook post-tool-failure
  ├── Stop               → agent-strace hook stop
  ├── SessionStart       → agent-strace hook session-start
  └── SessionEnd         → agent-strace hook session-end
                               ↓
                         .agent-traces/
```

Claude Code fires hook events at every stage of its agentic loop. agent-strace registers as a handler, reads JSON from stdin, and writes trace events. Each hook runs as a separate process. Session state lives in `.agent-traces/.active-session` so PreToolUse and PostToolUse can be correlated for latency measurement.

### MCP stdio proxy

```
Agent ←→ agent-strace proxy ←→ MCP Server (stdio)
              ↓
         .agent-traces/
```

The proxy reads JSON-RPC messages (Content-Length framed or newline-delimited), classifies each one, and writes a trace event. Messages are forwarded unchanged. The agent and server do not know the proxy exists.

### MCP HTTP/SSE proxy

```
Agent ←→ agent-strace proxy (localhost:3100) ←→ Remote MCP Server (HTTPS)
              ↓
         .agent-traces/
```

Same idea, different transport. Listens on a local port, forwards POST and SSE requests to the remote server, captures every JSON-RPC message in both directions.

### Decorator mode

```python
@trace_tool
def my_function(x):
    return x * 2
```

The decorator logs a `tool_call` event before execution and a `tool_result` after. Errors and timing are captured automatically.

### Secret redaction

When `--redact` is enabled (or `redact=True` in the decorator API), trace events pass through a redaction filter before hitting disk. The filter checks key names (`password`, `api_key`) and value patterns (`sk-*`, `ghp_*`, JWTs). Redacted values become `[REDACTED]`. The original data is never stored.

## Project structure

```
src/agent_trace/
  __init__.py       # version
  models.py         # TraceEvent, SessionMeta, EventType
  store.py          # NDJSON file storage
  hooks.py          # Claude Code hooks integration
  proxy.py          # MCP stdio proxy
  http_proxy.py     # MCP HTTP/SSE proxy
  redact.py         # secret redaction (key/value pattern matching)
  masking.py        # PII masking (email, phone, CC, SSN, ARN)
  otlp.py           # OTLP/HTTP JSON exporter with GenAI semantic conventions
  replay.py         # terminal replay, HTML viewer export
  decorator.py      # @trace_tool, @trace_llm_call, log_decision
  jsonl_import.py   # Claude Code JSONL session import
  explain.py        # session phase detection and plain-English summary
  cost.py           # token and cost estimation
  subagent.py       # parent-child session tree, tree replay, stats rollup
  diff.py           # structural, semantic, and side-by-side session comparison
  why.py            # causal chain tracing (backwards event walk)
  audit.py          # policy-based tool call checking, sensitive file detection
  audit_tools.py    # shadow AI detection (file signals + commit patterns)
  policy.py         # generate .agent-scope.json from observed traces
  attribution.py    # session attribution (user, process ancestry, git context)
  dashboard.py      # multi-session aggregate view and trend charts
  annotate.py       # replay annotations (notes, labels, bookmarks)
  token_budget.py   # token budget tracking and context window early warning
  watch.py          # live session watcher with rule-based kill switch
  share.py          # self-contained HTML report export
  standup.py        # standup report from session trace (no LLM)
  freshness.py      # context freshness check vs last session
  oncall.py         # on-call readiness for agent-modified files
  curve.py          # personal agent cost-efficiency curve
  inflation.py      # token inflation calculator across model versions
  a2a.py            # A2A protocol support and cross-agent trace correlation
  cli.py            # CLI entry point
ADRs/               # Architecture Decision Records
```

## Running tests

```bash
pytest
```

## Development

```bash
git clone https://github.com/Siddhant-K-code/agent-trace.git
cd agent-trace

# Run tests
pytest

# Run the example
PYTHONPATH=src python examples/basic_agent.py

# Replay the example
PYTHONPATH=src python -m agent_trace.cli replay

# Build the package
uv build

# Install locally for testing
uv tool install -e .
```

## Related

- [AGENTS.md integration guide](docs/agents-md-integration.md) - how to use agent-strace with AGENTS.md for drift detection and CI gating
- [Architecture Decision Records](ADRs/) - design decisions and their rationale
- [The agent observability gap (blog)](https://siddhantkhare.com/writing/agent-observability-gap) - the problem this tool addresses
- [The agent observability gap (thread)](https://x.com/Siddhant_K_code/status/2032834557628788940) - discussion on X
- [The Agentic Engineering Guide](https://agents.siddhantkhare.com) - chapters 7, 9, 10 cover agent security; chapters 14, 15, 16 cover observability
- [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - semantic conventions for LLM tracing (complementary)

## Sponsor

If agent-trace saves you time, consider [sponsoring the project](https://github.com/sponsors/Siddhant-K-code). It helps keep the work going.

## License

MIT. Use it however you want.
