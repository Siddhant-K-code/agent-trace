# agent-trace for VS Code

Live session overlay for [agent-trace](https://github.com/Siddhant-K-code/agent-trace). See what your agent is doing without leaving the editor.

Works in VS Code, Cursor, and any Open VSX-compatible editor.

## Features

**Status bar** — live cost, tool call count, and active tool name, updated on every event. Click to open the event stream panel.

```
$(pulse) agent  $0.0042  47 calls  [Read]
```

**Gutter annotations** — files the agent has read or modified get a colored left border and inline label.

```
src/auth/middleware.ts  ← agent read 3×, modified 1× this session
src/db/schema.ts        ← agent read 1× this session
```

**Event stream panel** — live feed of every tool call, file op, LLM request, and error in the Explorer sidebar.

**Session browser** — Explorer sidebar tree listing all sessions with timestamp, duration, tool calls, and error count. Click any session to open a summary.

**Post-mortem viewer** — when `agent-strace watch` kills a session (timeout, budget, or rule), the viewer opens automatically. Shows kill reason, cost at death, last tool call, and a copyable recovery context for pasting into a new session.

**Pause button** — stop the agent mid-session without killing it. Sends SIGSTOP to the agent process via `agent-strace watch`. Resume resumes it.

## Requirements

- [agent-strace](https://pypi.org/project/agent-strace/) installed: `pip install agent-strace` or `uv tool install agent-strace`
- A session started via `agent-strace setup` (Claude Code hooks) or `agent-strace record` (MCP proxy)

The extension activates automatically when a `.agent-traces/` directory exists in the workspace root.

## Setup

```bash
# 1. Install agent-strace
pip install agent-strace

# 2. Add hooks to Claude Code (one-time)
agent-strace setup

# 3. Open your project in VS Code / Cursor
# Extension activates automatically when .agent-traces/ exists

# 4. Start Claude Code — status bar appears immediately
```

## Commands

All commands are available from the Command Palette (`Cmd/Ctrl+Shift+P`):

| Command | Description |
|---|---|
| `agent-trace: Open Live Stream` | Open the event stream panel |
| `agent-trace: Open Event Stream` | Open the main agent-strace panel |
| `agent-trace: View Post-Mortem` | View the watchdog post-mortem for the latest killed session |
| `agent-trace: Refresh Session Browser` | Reload the session list |
| `agent-trace: Reveal Session in Browser` | Jump to a session by ID |
| `agent-trace: Pause Agent` | Send SIGSTOP to the agent (requires `watch` running) |
| `agent-trace: Resume Agent` | Send SIGCONT to resume a paused agent |
| `agent-trace: Clear File Decorations` | Remove all gutter annotations from the editor |

## Watchdog integration

Run `agent-strace watch` alongside your session to enable the pause button and post-mortem viewer:

```bash
# Kill session after 30 minutes or $5 spend
agent-strace watch --timeout 30m --budget 5.00 --on-violation kill
```

When the watchdog kills a session, the post-mortem viewer opens automatically. The "Copy recovery context" button copies a summary you can paste into a new session to resume where the agent left off.

## Configuration

| Setting | Default | Description |
|---|---|---|
| `agentTrace.traceDir` | `.agent-traces` | Path to trace store, relative to workspace root |
| `agentTrace.collectorEndpoint` | `""` | Remote collector URL (leave empty for local mode) |
| `agentTrace.watchdogPollIntervalSeconds` | `5` | How often (seconds) the status bar polls for updates |
| `agentTrace.sessionBrowserRefreshInterval` | `5` | How often (seconds) the session browser refreshes |
| `agentTrace.showGutterAnnotations` | `true` | Gutter icons on agent-touched files |
| `agentTrace.showInlineText` | `true` | Inline read/write counts at top of file |

## How it works

The extension watches `.agent-traces/.active-session` for the current session ID, then tails `events.ndjson` for new events using `fs.watch`. No polling when idle. No network calls. No new processes.

Pause works by writing `.agent-traces/.pause-request` — `agent-strace watch` checks for this file on every poll cycle and sends SIGSTOP / SIGCONT to the agent PID.

The post-mortem viewer reads `watchdog-postmortem.json` from the session directory, written by `agent-strace watch` on kill.
