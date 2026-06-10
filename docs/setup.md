# Setup

Three ways to capture agent sessions. Pick the one that matches your agent.

---

## Option 1: Claude Code hooks (recommended)

Captures everything: user prompts, assistant responses, and every tool call (Bash, Edit, Write, Read, Agent, Grep, Glob, WebFetch, WebSearch, all MCP tools).

```bash
# Generate and apply hooks config
agent-strace setup

# For all projects (global config)
agent-strace setup --global

# With secret redaction enabled
agent-strace setup --redact
```

`agent-strace setup` prints the hooks JSON. Add it to `~/.claude/settings.json` (user-level, applies to all projects):

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

Then use Claude Code normally. Sessions appear in `.agent-traces/`.

```bash
agent-strace list     # list sessions
agent-strace replay   # replay the latest
agent-strace explain  # plain-English summary
```

### Import existing sessions

Already ran sessions without hooks? Import from Claude Code's native JSONL logs:

```bash
# Discover available sessions
agent-strace import --discover

# Import a specific session
agent-strace import ~/.claude/projects/<project>/<session-id>.jsonl
```

---

## Option 2: MCP proxy (any MCP client)

Wraps any MCP server. Works with Cursor, Windsurf, or any MCP client that uses stdio transport.

```bash
# Wrap any MCP server
agent-strace record -- npx -y @modelcontextprotocol/server-filesystem /tmp
agent-strace replay
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

### Any MCP client (general pattern)

1. Replace the server `command` with `agent-strace`
2. Prepend `record --name <label> --` to the original args
3. Use the tool normally
4. Run `agent-strace replay` to see what happened

### HTTP/SSE proxy

For MCP servers that use HTTP transport:

```bash
agent-strace record-http https://mcp.example.com --port 3100
# Your agent connects to http://127.0.0.1:3100
```

---

## Option 3: Python decorator

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

---

## Security-sensitive repos

For repos that handle secrets, attestation logic, or cryptographic material:

```bash
cd your-sensitive-repo
agent-strace setup --redact
```

This enables automatic redaction of API keys, tokens, and credentials before they hit disk. Detected patterns: OpenAI (`sk-*`), GitHub (`ghp_*`, `github_pat_*`), AWS (`AKIA*`), Anthropic (`sk-ant-*`), Slack (`xox*`), JWTs, Bearer tokens, connection strings, and any value under keys like `password`, `secret`, `token`, `api_key`, `authorization`.

Add custom patterns:

```bash
agent-strace setup --redact --redact-pattern "ATTESTATION_KEY=[A-Fa-f0-9]{64}"
```

See [security.md](security.md) for the full security guide.
