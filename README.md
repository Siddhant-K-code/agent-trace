# agent-trace

`strace` for AI agents.

Capture every tool call, LLM request, and decision point. Replay the session later. See what the agent did, in what order, and how long each step took.

We have `strace` for syscalls. We have `tcpdump` for packets. We have nothing for agent tool calls. This fills that gap.

## Why

When a coding agent rewrites 20 files in a background session, you get a pull request. You don't get the story of how it got there. Which files did it read first? What context was in the window when it decided to change the approach? Why did it call the same tool three times?

Existing tools trace LLM calls. That's one layer. The gap is everything around it: tool calls, file operations, decision points, error recovery. `agent-strace` captures the full picture.

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

### Option 1: Claude Code hooks (captures everything)

Trace every tool call Claude Code makes — Bash, Edit, Write, Read, Agent, Grep, Glob, WebFetch, WebSearch, and all MCP tools.

```bash
# Generate the hooks config
agent-strace setup

# Prints JSON to add to .claude/settings.json (or ~/.claude/settings.json with --global)
```

Or add the hooks manually to `.claude/settings.json`:

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

Then use Claude Code normally. Every tool call is traced.

```bash
agent-strace list     # list sessions
agent-strace replay   # replay the latest
agent-strace stats    # tool call frequency and timing
```

### Option 2: MCP proxy (any MCP client)

Wrap any MCP server. Every JSON-RPC message between agent and server is captured.

```bash
# Record a session
agent-strace record -- npx -y @modelcontextprotocol/server-filesystem /tmp

# Replay
agent-strace replay a84664
```

### Option 3: Python decorator

Wrap your tool functions. No MCP required.

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

```
agent-strace setup [--redact] [--global]   Generate Claude Code hooks config
agent-strace hook <event>                  Handle a Claude Code hook event (internal)
agent-strace record -- <command>           Record an MCP stdio server session
agent-strace record-http <url> [--port N]  Record an MCP HTTP/SSE server session
agent-strace replay [session-id]           Replay a session (default: latest)
agent-strace list                          List all sessions
agent-strace stats [session-id]            Show tool call frequency and timing
agent-strace inspect <session-id>          Dump full session as JSON
agent-strace export <session-id>           Export as JSON, CSV, or NDJSON
```

### Secret redaction

Pass `--redact` to strip API keys, tokens, and credentials from traces before they hit disk.

```bash
# Stdio proxy with redaction
agent-strace record --redact -- npx -y @modelcontextprotocol/server-filesystem /tmp

# HTTP proxy with redaction
agent-strace record-http https://mcp.example.com --redact
```

Detected patterns: OpenAI (`sk-*`), GitHub (`ghp_*`, `github_pat_*`), AWS (`AKIA*`), Anthropic (`sk-ant-*`), Slack (`xox*`), JWTs, Bearer tokens, connection strings (`postgres://`, `mysql://`), and any value under keys like `password`, `secret`, `token`, `api_key`, `authorization`.

### HTTP/SSE proxy

For MCP servers that use HTTP transport instead of stdio:

```bash
# Proxy a remote MCP server
agent-strace record-http https://mcp.example.com --port 3100

# Your agent connects to http://127.0.0.1:3100 instead of the remote server
# All JSON-RPC messages are captured, tool call latency is measured
```

The proxy forwards POST `/message` and GET `/sse` to the remote server, capturing every JSON-RPC message in both directions.

### Replay output

```
Session Summary
──────────────────────────────────────────────────
  Session:    a84664242afa4516
  Agent:      coding-agent
  Duration:   0.85s
  Tool calls: 6
  LLM reqs:   2
  Errors:     1
──────────────────────────────────────────────────

+  0.00s ▶ session_start
+  0.00s ⬆ llm_request claude-4 (1 messages)
+  0.13s ⬇ llm_response (132ms)
+  0.13s ◆ decision read_file_first
              reason: Need to understand current implementation before making changes
+  0.13s → tool_call read_file (path)
+  0.16s ← tool_result [text] (22ms)
              "contents of src/auth.py: def hello(): print('world')"
+  0.16s → tool_call search_codebase (query)
+  0.25s ← tool_result [text] (96ms)
+  0.25s ⬆ llm_request claude-4 (3 messages)
+  0.36s ⬇ llm_response (109ms)
+  0.36s ◆ decision apply_fix
              reason: LLM provided a clear fix, confidence is high
+  0.36s → tool_call write_file (path, content)
+  0.41s ← tool_result [text] (45ms)
+  0.41s → tool_call run_tests (test_path)
+  0.61s ✗ error Test failed: tests/test_auth.py
+  0.61s ◆ decision retry_fix
              reason: Tests failed, need to adjust the implementation
+  0.61s → tool_call write_file (path, content)
+  0.63s ← tool_result [text] (27ms)
+  0.64s → tool_call run_tests (test_path)
+  0.85s ← tool_result [text] (216ms)
+  0.85s ■ session_end
```

### Stats output

```
  Tool Call Frequency:
    write_file                        2x  avg: 36ms
    run_tests                         2x  avg: 216ms
    read_file                         1x  avg: 22ms
    search_codebase                   1x  avg: 96ms

  Errors (1):
    Test failed: tests/test_auth.py
```

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

### Claude Code (hooks — captures all tool calls)

Claude Code's [hooks system](https://code.claude.com/docs/en/hooks) fires events for every tool call, not just MCP. This is the recommended integration.

```bash
agent-strace setup        # prints the hooks config JSON
agent-strace setup --redact --global  # with redaction, for all projects
```

Add the output to `.claude/settings.json` (per-project) or `~/.claude/settings.json` (global). See [examples/claude_code_config.md](examples/claude_code_config.md) for the full config and a table of what gets captured.

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

Claude Code fires hook events at every stage of its agentic loop. agent-strace registers as a hook handler, receives JSON on stdin, and writes trace events. This captures the full conversation: user prompts, assistant text responses, and all tool calls (Bash, Edit, Write, Read, Agent, Grep, Glob, WebFetch, WebSearch, and all MCP tools). Session state is tracked via `.agent-traces/.active-session` so separate hook processes can correlate PreToolUse with PostToolUse for latency measurement.

### MCP stdio proxy

```
Agent ←→ agent-strace proxy ←→ MCP Server (stdio)
              ↓
         .agent-traces/
```

The proxy reads JSON-RPC messages (Content-Length framed or newline-delimited), classifies each message as a tool call, result, error, or notification, and writes a trace event. The message is forwarded unchanged. The agent and server don't know the proxy exists.

### MCP HTTP/SSE proxy

```
Agent ←→ agent-strace proxy (localhost:3100) ←→ Remote MCP Server (HTTPS)
              ↓
         .agent-traces/
```

Same idea, different transport. The proxy listens on a local port, forwards POST and SSE requests to the remote server, and captures every JSON-RPC message in both directions. Tool call latency is measured from request to response.

### Decorator mode

```python
@trace_tool
def my_function(x):
    return x * 2
```

The decorator wraps the function call. It logs a `tool_call` event before execution and a `tool_result` event after. If the function raises, it logs an `error` event. Timing is captured automatically.

### Secret redaction

When `--redact` is enabled (or `redact=True` in the decorator API), every trace event is passed through a redaction filter before being written to disk. The filter checks both key names (e.g., `password`, `api_key`) and value patterns (e.g., `sk-*`, `ghp_*`, JWTs). Redacted values are replaced with `[REDACTED]`. The original data is never stored.

## Project structure

```
src/agent_trace/
  __init__.py       # version
  models.py         # TraceEvent, SessionMeta, EventType
  store.py          # NDJSON file storage
  hooks.py          # Claude Code hooks integration
  proxy.py          # MCP stdio proxy
  http_proxy.py     # MCP HTTP/SSE proxy
  redact.py         # secret redaction
  replay.py         # terminal replay and display
  decorator.py      # @trace_tool, @trace_llm_call, log_decision
  cli.py            # CLI entry point
```

## Running tests

```bash
python -m unittest discover -s tests -v
```

## Development

```bash
git clone https://github.com/Siddhant-K-code/agent-trace.git
cd agent-trace

# Run tests
python -m unittest discover -s tests -v

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

- [The agent observability gap (blog)](https://siddhantkhare.com/writing/agent-observability-gap) - the problem this tool addresses
- [The agent observability gap (thread)](https://x.com/Siddhant_K_code/status/2032834557628788940) - discussion on X
- [The Agentic Engineering Guide](https://agents.siddhantkhare.com) - chapters 7, 9, 10 cover agent security and observability
- [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - semantic conventions for LLM tracing (complementary)

## License

AGPL-3.0. You can use, modify, and distribute this freely. If you modify it and distribute or run it as a service, you must open-source your changes under the same license.
