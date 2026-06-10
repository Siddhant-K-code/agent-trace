# agent-trace

[![Run in Ona](https://ona.com/run-in-ona.svg)](https://app.ona.com/#https://github.com/Siddhant-K-code/agent-trace)
[![PyPI](https://img.shields.io/pypi/v/agent-strace)](https://pypi.org/project/agent-strace/)
[![Python](https://img.shields.io/pypi/pyversions/agent-strace)](https://pypi.org/project/agent-strace/)
[![CI](https://github.com/Siddhant-K-code/agent-trace/actions/workflows/test.yml/badge.svg)](https://github.com/Siddhant-K-code/agent-trace/actions/workflows/test.yml)
[![GitHub Marketplace](https://img.shields.io/badge/Marketplace-agent--trace%20eval-blue?logo=github)](https://github.com/marketplace/actions/agent-trace-eval)
[![Open VSX](https://img.shields.io/open-vsx/v/Siddhant-K-code/agent-strace)](https://open-vsx.org/extension/Siddhant-K-code/agent-strace)
[![VS Marketplace](https://img.shields.io/badge/VS%20Marketplace-v0.2.1-blue?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=Siddhant-K-code.agent-strace)
[![License](https://img.shields.io/github/license/Siddhant-K-code/agent-trace)](LICENSE)

`strace` for AI agents.

![demo](assets/demo.svg)

## Why

A coding agent rewrites 20 files in a background session. You get a pull request. You do not get the story. Which files did it read first? Why did it call the same tool three times? What failed before it found the fix?

Most tools trace LLM calls. That is one layer. The gap is everything around it: tool calls, file operations, decision points, error recovery, the actual commands the agent ran. `agent-strace` captures the full session and lets you replay it later. Export to Datadog, Honeycomb, New Relic, or Splunk for production observability. Set rules to stop the agent: cost ceiling, wrong file touched, too many tool calls. The agent stops. No prompt, no retry, no damage.

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

## Quick start

**Option 1: CLI hooks** — captures prompts, responses, and hook-visible tool calls

```bash
agent-strace setup             # Claude Code hooks for ~/.claude/settings.json
agent-strace setup --cli codex # OpenAI Codex hooks for ~/.codex/hooks.json
agent-strace setup --cli gemini # Gemini CLI extension under ~/.gemini/extensions
agent-strace list              # list sessions
agent-strace replay            # replay the latest
```

Full config and JSON: [docs/setup.md](docs/setup.md)

**Option 2: MCP proxy** — wraps any MCP server, works with Cursor and Windsurf

```bash
agent-strace record -- npx -y @modelcontextprotocol/server-filesystem /tmp
agent-strace replay
```

**Option 3: Python decorator** — no MCP required

```python
from agent_trace import trace_tool, start_session, end_session

start_session(name="my-agent")

@trace_tool
def search_codebase(query: str) -> str:
    return search(query)

end_session()
```

Full setup guide: [docs/setup.md](docs/setup.md)

## What you can do

### Understand a session

| Command | What it does |
|---|---|
| [`agent-strace replay <id>`](docs/commands.md#replay) | Replay a session in the terminal or as HTML |
| [`agent-strace replay <id-a> --diff <id-b>`](docs/commands.md#replay) | Side-by-side session comparison with tool args and output delta |
| [`agent-strace explain <id>`](docs/commands.md#explain) | Plain-English phase summary, no LLM required |
| [`agent-strace timeline <id>`](docs/commands.md#timeline) | Phase-by-phase view with costs and retries |
| [`agent-strace why <id> <event>`](docs/commands.md#why) | Causal chain for a specific decision |
| [`agent-strace diff <id-a> <id-b>`](docs/commands.md#diff) | Structural or semantic session comparison |
| [`agent-strace compare <id-a> <id-b>`](docs/commands.md#compare) | Regression report with verdict |

### Control and protect

| Command | What it does |
|---|---|
| [`agent-strace watch`](docs/commands.md#watch) | Live monitor with kill-switch rules |
| [`agent-strace watch --timeout 30m --budget $5`](docs/commands.md#watch) | Watchdog mode — kills on limit and heartbeats sessions for postmortems |
| [`agent-strace mcp-scan`](docs/commands.md#mcp-scan) | Scan runtime MCP poisoning indicators |
| [`agent-strace audit <id>`](docs/commands.md#audit) | Audit tool calls against a policy file |
| [`agent-strace approval list`](docs/commands.md#approval) | Human-in-the-loop approval queue |
| [`agent-strace rbac assign`](docs/commands.md#rbac) | Org and workspace-scoped role assignments |
| [`agent-strace auth login`](docs/commands.md#auth) | SSO/OIDC login to a hosted collector |
| [`agent-strace apply`](docs/commands.md#apply) | Apply `.agent-strace.yaml` config to local store or collector |
| [`agent-strace workspace new`](docs/commands.md#workspace) | Create an isolated workspace |
| [`agent-strace compliance export`](docs/commands.md#compliance) | Export compliance reports (EU AI Act, SOC 2, HIPAA) |
| [`agent-strace record`](docs/commands.md#record) | Strip secrets from traces before storage by default |
| [`agent-strace export --anonymize`](docs/commands.md#export) | Remove PII at export time |

### Analyse across sessions

| Command | What it does |
|---|---|
| [`agent-strace dashboard`](docs/commands.md#dashboard) | Multi-session overview |
| [`agent-strace budget-report`](docs/commands.md#budget-report) | Weekly spend digest |
| [`agent-strace team-report`](docs/commands.md#team-report) | Team spend by author, branch, or PR |
| [`agent-strace lint <id>`](docs/commands.md#lint) | Flag bad behaviour patterns (loops, spirals, waste) |
| [`agent-strace drift`](docs/commands.md#drift) | Detect behavioural drift over time |
| [`agent-strace fingerprint`](docs/commands.md#fingerprint) | Baseline an agent's behavioural profile |
| [`agent-strace tree`](docs/commands.md#tree) | Show parent/child session hierarchy |
| [`agent-strace freeze`](docs/commands.md#freeze) | Freeze a tool-call sequence for regression checks |
| [`agent-strace standup`](docs/commands.md#standup) | Plain-English summary of yesterday's sessions |
| [`agent-strace eval <id>`](docs/commands.md#eval) | Score a session against behavioural baselines |
| [`agent-strace eval ci`](docs/commands.md#eval) | Fail CI on behavioural regression |

### Export and integrate

| Command | What it does |
|---|---|
| [`agent-strace export --format otlp-genai`](docs/production.md) | Export to Datadog, Honeycomb, Grafana, Jaeger |
| [`agent-strace export --metrics`](docs/production.md#behavioral-metrics) | Export per-session behavioral metrics as OTLP gauges |
| [`agent-strace identity show`](docs/commands.md#identity) | Machine identity — sign and verify sessions |
| [`agent-strace server`](docs/server.md) | Server-side collector for multi-agent, multi-machine |
| [`agent-strace share <id>`](docs/commands.md#share) | Generate a shareable HTML replay |
| [`agent-strace sample`](docs/commands.md#sample) | Export worst sessions as JSONL for eval datasets |

Full flag reference: [docs/commands.md](docs/commands.md)

## VS Code extension

Install **agent-strace** from the [Extensions panel](https://open-vsx.org/extension/Siddhant-K-code/agent-strace) to see live session activity without leaving the editor.

| Feature | Description |
|---|---|
| Status bar | Live cost, tool call count, and active tool name. Click to open the event stream. |
| Gutter annotations | Blue border on files the agent read, amber on files it modified. |
| Event stream panel | Live feed: every tool call, file op, LLM request, and error. |
| Pause button | Stops the agent mid-session via SIGSTOP. |

```bash
pip install agent-strace   # 1. install
agent-strace setup         # 2. add hooks to Claude Code; use --cli codex or --cli gemini for other CLIs
# 3. open project in VS Code — extension activates when .agent-traces/ exists
# 4. start Claude Code — status bar appears immediately
```

Full docs: [docs/vscode.md](docs/vscode.md)

## Production

**OTLP export** — sessions become traces, tool calls become spans:

```bash
agent-strace export <session-id> --format otlp-genai \
  --endpoint http://localhost:4318
```

Per-backend setup (Datadog, Honeycomb, Grafana, New Relic, Splunk, Langfuse): [docs/production.md](docs/production.md)

**Server-side collector** — for containers, CI, and multi-machine setups:

```bash
agent-strace server --port 4317 --storage ./traces
AGENT_STRACE_ENDPOINT=http://collector:4317 python my_agent.py
```

Full guide: [docs/server.md](docs/server.md)

**Auto-instrumentation** — no code changes required:

```python
from agent_trace.integrations import instrument_langchain
instrument_langchain()
```

Supported: OpenAI Agents SDK, LangChain, LangGraph, CrewAI, LiteLLM, Anthropic SDK, OpenAI SDK, AWS Strands. Guide: [docs/integrations.md](docs/integrations.md)

**GitHub Actions** — run evals in CI, post results to the step summary, fail on regression:

```yaml
- uses: Siddhant-K-code/agent-trace@gha-v1
  with:
    config: .agent-evals.yaml
    baseline: .agent-evals-baseline.json
    tolerance: "0.05"
```

[Marketplace listing](https://github.com/marketplace/actions/agent-trace-eval) · [Action reference](action.yml)

## How it works

**Claude Code hooks** — Claude Code fires hook events at every stage of its agentic loop. agent-strace registers as a handler, reads JSON from stdin, and writes trace events. Each hook runs as a separate process; session state in `.agent-traces/.active-session` correlates PreToolUse and PostToolUse for latency measurement.

**MCP stdio proxy** — sits between the agent and the MCP server, reads JSON-RPC messages (Content-Length framed or newline-delimited), classifies each one, and writes a trace event. Messages are forwarded unchanged. The agent and server do not know the proxy exists.

**MCP HTTP/SSE proxy** — same idea, different transport. Listens on a local port, forwards POST and SSE requests to the remote server, captures every JSON-RPC message in both directions.

**Python decorator** — `@trace_tool` logs a `tool_call` event before execution and a `tool_result` after. Errors and timing are captured automatically. `@trace_llm_call` does the same for LLM calls.

## Running tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT. Use it however you want.

---

[Sponsor](https://github.com/sponsors/Siddhant-K-code) · [ADRs](ADRs/) · [Security](docs/security.md) · [PyPI](https://pypi.org/project/agent-strace/)
