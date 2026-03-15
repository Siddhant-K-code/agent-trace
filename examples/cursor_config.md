# Using agent-trace with Cursor

## How it works

Cursor reads MCP server config from `~/.cursor/mcp.json` (global) or
`.cursor/mcp.json` (per-project). agent-trace wraps the server command
so every tool call is captured.

## Setup

### 1. Install agent-trace

```bash
# With uv (recommended)
uv tool install agent-strace

# Or with pip
pip install agent-strace
```

### 2. Edit your MCP config

Open `~/.cursor/mcp.json` (or `.cursor/mcp.json` in your project):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "agent-strace",
      "args": [
        "record",
        "--name", "filesystem",
        "--",
        "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"
      ]
    },
    "github": {
      "command": "agent-strace",
      "args": [
        "record",
        "--name", "github",
        "--",
        "npx", "-y", "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    }
  }
}
```

The pattern: replace the original `command` with `agent-trace` and prepend
`record --name <label> --` to the original args.

### 3. Use Cursor normally

All MCP tool calls are captured in `.agent-traces/`.

### 4. Replay

```bash
agent-strace list
agent-strace replay
agent-strace stats
```
