# VS Code extension

The **agent-strace** extension shows live session activity without leaving the editor. Works in VS Code, Cursor, and any Open VSX-compatible editor.

- [Open VSX](https://open-vsx.org/extension/Siddhant-K-code/agent-strace)
- [VS Marketplace](https://marketplace.visualstudio.com/items?itemName=Siddhant-K-code.agent-strace)

---

## Features

| Feature | Description |
|---|---|
| Status bar | Live cost, tool call count, and active tool name. Click to open the event stream. |
| Gutter annotations | Blue border on files the agent read, amber on files it modified. Inline label shows read/write counts. |
| Event stream panel | Live feed in the Explorer sidebar: every tool call, file op, LLM request, and error. |
| Pause button | Stops the agent mid-session via SIGSTOP. Requires `agent-strace watch` running in a terminal. |
| Watchdog status bar | Polls the active session for cost and tool count; updates every 5 seconds (configurable). |
| Post-mortem viewer | Auto-opens when a session is killed by the watchdog. Shows kill reason, cost at death, and a copyable recovery context. |
| Session browser | Explorer sidebar tree listing all sessions with timestamp, duration, tool calls, and error count. |

---

## Setup

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

---

## Commands

All commands are available from the Command Palette (`Cmd/Ctrl+Shift+P`):

| Command | Description |
|---|---|
| `agent-trace: Open Live Stream` | Open the event stream panel |
| `agent-trace: Open Post-Mortem` | View the watchdog post-mortem for the latest killed session |
| `agent-trace: Refresh Session Browser` | Reload the session list in the Explorer sidebar |
| `agent-trace: Reveal Session` | Jump to a session in the browser |
| `agent-trace: Pause Agent` | Send SIGSTOP to the agent process (requires `watch` running) |
| `agent-trace: Resume Agent` | Send SIGCONT to resume a paused agent |
| `agent-trace: Open Panel` | Open the main agent-strace panel |
| `agent-trace: Clear Decorations` | Remove all gutter annotations from the editor |

---

## Pause / resume

The pause button requires `agent-strace watch` running in a separate terminal:

```bash
# In a separate terminal, start the watcher
agent-strace watch

# Then use the Pause button in the event stream panel,
# or run: agent-trace: Pause Agent from the command palette
```

When paused, the agent process receives SIGSTOP and freezes. Resume with the Resume command or SIGCONT.

---

## Settings

All settings are under `agentTrace.*` in VS Code settings:

| Setting | Default | Description |
|---|---|---|
| `agentTrace.traceDir` | `.agent-traces` | Path to trace directory, relative to workspace root |
| `agentTrace.collectorEndpoint` | `""` | Remote collector URL (leave empty for local mode) |
| `agentTrace.watchdogPollIntervalSeconds` | `5` | How often (in seconds) the status bar polls for cost/tool updates |
| `agentTrace.sessionBrowserRefreshInterval` | `5` | How often (in seconds) the session browser refreshes |
| `agentTrace.showGutterAnnotations` | `true` | Show gutter icons on files the agent read or modified |
| `agentTrace.showInlineText` | `true` | Show inline read/write counts at the top of agent-touched files |

---

## Post-mortem viewer

When `agent-strace watch` kills a session, a `watchdog-postmortem.json` is written to the session directory. The extension detects this file and offers to open the post-mortem viewer automatically.

The viewer shows:
- Kill reason (timeout, budget, rule)
- Elapsed time and cost at death
- Last tool call and LLM response
- Recovery context (copyable, for pasting into a new session)

---

## Session browser

The session browser appears in the Explorer sidebar when `.agent-traces/` exists. It lists all sessions with timestamp, duration, tool calls, and error count. Click any session to open a summary. Use `Reveal Session` to jump to a specific session by ID prefix.
