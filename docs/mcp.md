# MCP server

`agent-strace mcp` runs a local MCP server over stdio. It lets an agent query the sessions agent-strace has already recorded. It reads the trace store only: no network calls, no writes.

Registry name: `io.github.Siddhant-K-code/agent-trace` (PyPI package `agent-strace`).

## Tools

| Tool | Returns |
|---|---|
| `list_sessions` | Sessions with tool calls, LLM requests, errors, tokens, estimated cost |
| `get_session` | Full event stream for one session, optionally filtered by event type |
| `search_events` | Events matching a tool name, file path, exit code, or error, across one or all sessions |
| `get_session_summary` | Plain-English phase summary (same as `agent-strace explain`) |
| `diff_sessions` | Tool, file, token, cost, and error deltas between two sessions |
| `health` | Evidence health for a session (same JSON as `agent-strace health --format json`). Defaults to the latest session |
| `skills` | Skill adoption report (same JSON as `agent-strace skills --format json`). One session, or an aggregate over `since` (default `30d`) |

`initialize` reports the installed package version.

## Add it to your agent

Install the CLI first (`uv tool install agent-strace` or `pip install agent-strace`), or replace `agent-strace` with `uvx agent-strace` in the commands below.

The server reads `.agent-traces/` in the directory the client starts it from. Clients do not always start servers in your project, so point it at a fixed store with `--trace-dir` or `AGENT_TRACE_DIR` when in doubt. See [Global options](commands.md#global-options).

**Claude Code**

```bash
claude mcp add agent-trace -- agent-strace mcp
# fixed store:
claude mcp add agent-trace -e AGENT_TRACE_DIR=/abs/path/.agent-traces -- agent-strace mcp
```

**GitHub Copilot CLI**

```bash
copilot mcp add agent-trace -- agent-strace mcp
```

Or add it to `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "agent-trace": { "type": "local", "command": "agent-strace", "args": ["mcp"], "tools": ["*"] }
  }
}
```

**Codex**

```bash
codex mcp add agent-trace -- agent-strace mcp
```

Or in `~/.codex/config.toml`:

```toml
[mcp_servers.agent-trace]
command = "agent-strace"
args = ["mcp"]
```

**Gemini CLI**

```bash
gemini mcp add -s user agent-trace agent-strace mcp
```

The Gemini CLI extension already includes the server. Install it with `agent-strace setup --cli gemini` (adds capture hooks too) or `gemini extensions install https://github.com/Siddhant-K-code/agent-trace` (MCP server only; run `agent-strace setup --cli gemini` afterwards to record sessions).

## Example prompts

- "List my last 5 agent-trace sessions and summarize the one with the most errors."
- "Which of my skills never loaded in the last 30 days?"
- "Run health on the latest session and tell me what the trace cannot show."

## Check it by hand

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"health","arguments":{}}}' \
  | agent-strace mcp
```
