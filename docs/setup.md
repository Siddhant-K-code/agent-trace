# Setup

Three ways to capture agent sessions. Pick the one that matches your agent.

---

## Option 1: CLI hooks (recommended)

Captures the lifecycle events exposed by each CLI: user prompts, assistant responses, and hook-visible tool calls or edits. Claude Code exposes broad tool coverage; Cursor coverage depends on the native events Cursor emits.

```bash
# Generate Claude Code hooks config
agent-strace setup

# Install OpenAI Codex user-level hooks
agent-strace setup --cli codex

# Install Gemini CLI extension hooks
agent-strace setup --cli gemini

# Install Cursor project hooks
agent-strace setup --cli cursor

# Install GitHub Copilot CLI user-level hooks
agent-strace setup --cli copilot

# Configure all supported hook integrations
agent-strace setup --cli all

# For all projects (global config)
agent-strace setup --global

# Trusted local traces only: disable secret redaction
agent-strace setup --no-redact
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

### OpenAI Codex hooks

`agent-strace setup --cli codex` writes user-level hooks to `$CODEX_CONFIG_DIR/hooks.json` or `~/.codex/hooks.json`, and also prints the JSON:

```json
{
  "hooks": {
    "SessionStart": [{ "matcher": "startup|resume|clear|compact", "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex session-start" }] }],
    "UserPromptSubmit": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex user-prompt" }] }],
    "PreToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex pre-tool" }] }],
    "PostToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex post-tool" }] }],
    "Stop": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex stop" }] }]
  }
}
```

Codex sends one JSON object to each command hook on stdin. agent-strace records the common Codex fields (`session_id`, `turn_id`, `tool_use_id`, `tool_name`, `tool_input`, `tool_response`, `prompt`, and `last_assistant_message`) into the same `.agent-traces/` session store used by Claude Code.

If Codex does not list the hooks after you create the file:

- Use the root `~/.codex/hooks.json` file for user-level hooks, or `<repo>/.codex/hooks.json` for project-level hooks. `~/.codex/hooks/hooks.json` is for plugin-bundled hooks and is not the normal user config path.
- Check `~/.codex/config.toml` and remove `[features].hooks = false` if present. Hooks are enabled by default unless a user, system, or admin config layer disables them.
- Reload Codex or press refresh in the Hooks view. Non-managed command hooks must be reviewed and trusted before they run.

### Gemini CLI hooks

`agent-strace setup --cli gemini` writes a Gemini CLI extension:

```
~/.gemini/extensions/agent-strace/
├── gemini-extension.json
└── hooks/
    └── hooks.json
```

Set `GEMINI_CONFIG_DIR` to install into a different Gemini config directory. The generated `hooks.json` registers `SessionStart`, `BeforeAgent`, `BeforeTool`, `AfterTool`, `AfterAgent`, and `SessionEnd` command hooks:

```json
{
  "hooks": {
    "BeforeTool": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini pre-tool" }] }],
    "AfterTool": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini post-tool" }] }],
    "BeforeAgent": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini user-prompt" }] }],
    "AfterAgent": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini stop" }] }],
    "SessionStart": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini session-start" }] }],
    "SessionEnd": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini session-end" }] }]
  }
}
```

Gemini sends one JSON object to each command hook on stdin. agent-strace records `session_id`, `tool_name`, `tool_input`, `tool_response`, `prompt`, and `prompt_response` into the same `.agent-traces/` session store used by Claude Code and Codex.

### Cursor hooks

`agent-strace setup --cli cursor` writes a project-local Cursor hooks file:

```
.cursor/
└── hooks.json
```

Set `CURSOR_CONFIG_DIR` to write the file somewhere else. The generated config registers Cursor-native prompt, shell, file-edit, assistant-response, session-start, and session-end command hooks:

```json
{
  "version": 1,
  "hooks": {
    "sessionStart": [{ "type": "command", "command": "agent-strace hook --provider cursor session-start" }],
    "beforeSubmitPrompt": [{ "type": "command", "command": "agent-strace hook --provider cursor before-submit-prompt" }],
    "beforeShellExecution": [{ "type": "command", "command": "agent-strace hook --provider cursor before-shell-execution" }],
    "afterShellExecution": [{ "type": "command", "command": "agent-strace hook --provider cursor after-shell-execution" }],
    "afterFileEdit": [{ "type": "command", "command": "agent-strace hook --provider cursor after-file-edit" }],
    "afterAgentResponse": [{ "type": "command", "command": "agent-strace hook --provider cursor after-agent-response" }],
    "stop": [{ "type": "command", "command": "agent-strace hook --provider cursor stop" }],
    "sessionEnd": [{ "type": "command", "command": "agent-strace hook --provider cursor session-end" }]
  }
}
```

Cursor hook coverage depends on the events Cursor emits. Native hooks capture prompts, shell commands, file edits, stop markers, and agent responses when available. MCP server tool calls are still captured most reliably through the MCP proxy configuration below.

### GitHub Copilot CLI hooks

`agent-strace setup --cli copilot` writes user-level Copilot hooks:

```
~/.copilot/
└── hooks/
    └── agent-strace.json
```

Set `COPILOT_HOME` to install into a different Copilot config directory. The generated config registers Copilot lifecycle hooks using the VS Code-compatible event names:

```json
{
  "version": 1,
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider copilot session-start" }] }],
    "UserPromptSubmit": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider copilot user-prompt" }] }],
    "PreToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider copilot pre-tool" }] }],
    "PostToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider copilot post-tool" }] }],
    "PostToolUseFailure": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider copilot post-tool-failure" }] }],
    "Stop": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider copilot stop" }] }],
    "SessionEnd": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider copilot session-end" }] }]
  }
}
```

Copilot sends hook payloads on stdin. agent-strace records session starts, user prompts, hook-visible tool calls/results, stop markers, and session ends. Assistant text capture depends on the fields Copilot includes; stop hooks often provide `transcript_path` and `stop_reason` rather than response text.

Stop and session-end coverage is provider-defined. agent-strace records these hooks when the agent CLI emits them, including empty stop payloads, but it cannot force an agent to emit a stop hook for every UI action such as deleting a conversation or interrupting a run.

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

start_session(name="my-agent")  # use redact=False only for trusted local traces

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
agent-strace setup
```

Secret redaction is enabled by default for all capture paths. It redacts API keys, tokens, credentials, private keys, basic-auth URLs, and connection strings before they hit disk. Detected patterns include OpenAI (`sk-*`), GitHub (`ghp_*`, `github_pat_*`), AWS (`AKIA*` and AWS credential key names), Anthropic (`sk-ant-*`), Slack (`xox*`), JWTs, Bearer tokens, and any value under keys like `password`, `secret`, `token`, `api_key`, or `authorization`.

See [security.md](security.md) for the full security guide.
