# agent-trace for VS Code

Live session overlay for [agent-trace](https://github.com/Siddhant-K-code/agent-trace). Shows what your agent is doing without leaving the editor.

## Features

**Status bar** — cost, tool call count, and active tool name, updated on every event. Click to open the event stream panel.

```
$(pulse) agent  $0.0042  47 calls  [Read]
```

**Gutter annotations** — files the agent has read or modified get a colored left border and inline label:

```
src/auth/middleware.ts  ← agent read 3×, modified 1× this session
src/db/schema.ts        ← agent read 1× this session
```

**Event stream panel** — live feed of every tool call, file op, LLM request, and error in the Explorer sidebar. Same information as `agent-strace watch` but in the editor.

**Pause button** — stop the agent mid-session without killing it. Writes a signal file that `agent-strace watch` picks up and sends SIGSTOP to the agent process. Resume resumes it.

## Requirements

- [agent-trace](https://pypi.org/project/agent-strace/) installed (`pip install agent-strace` or `uv tool install agent-strace`)
- A session started via `agent-strace setup` (Claude Code hooks) or `agent-strace record` (MCP proxy)

The extension activates automatically when a `.agent-traces/` directory exists in the workspace root.

## Usage

1. Install agent-trace and set up hooks:
   ```bash
   agent-strace setup   # adds hooks to .claude/settings.json
   ```
2. Open your project in VS Code / Cursor.
3. Start Claude Code — the status bar item appears as soon as the session starts.
4. Open the **Agent Trace** panel in the Explorer sidebar for the full event stream.

The **Pause** button in the panel (or `agent-trace: Pause Agent` command) sends SIGSTOP to the agent. This requires `agent-strace watch` to be running in a terminal alongside the session.

## Configuration

| Setting | Default | Description |
|---|---|---|
| `agentTrace.traceDir` | `.agent-traces` | Path to trace store, relative to workspace root |
| `agentTrace.showGutterAnnotations` | `true` | Gutter icons on agent-touched files |
| `agentTrace.showInlineText` | `true` | Inline read/write counts at top of file |

## How it works

The extension watches `.agent-traces/.active-session` for the current session ID, then tails `events.ndjson` for new events using `fs.watch`. No polling when idle. No network calls. No new processes.

Pause works by writing `.agent-traces/.pause-request` — `agent-strace watch` checks for this file on every poll cycle and sends SIGSTOP / SIGCONT to the agent PID.
