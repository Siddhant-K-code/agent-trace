# Using agent-trace with Windsurf

## How it works

Windsurf reads MCP server config from `~/.codeium/windsurf/mcp_config.json`.
agent-trace wraps the server command to capture every tool call.

## Setup

### 1. Install agent-trace

```bash
# With uv (recommended)
uv tool install agent-strace

# Or with pip
pip install agent-strace
```

### 2. Edit your MCP config

Open `~/.codeium/windsurf/mcp_config.json`:

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
    }
  }
}
```

### 3. Use Windsurf normally

All MCP tool calls are captured in `.agent-traces/`.

### 4. Replay

```bash
agent-strace list
agent-strace replay
agent-strace stats
```
