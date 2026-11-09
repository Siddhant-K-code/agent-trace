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
| Evidence review | Ordered local evidence with capture identity, evidence health, gaps, raw fields, annotations, relationships, and reviewer filters. |
| Pause button | Stops the agent mid-session via SIGSTOP. Requires `agent-strace watch --agent-pid PID` running in a terminal. |
| Watchdog status bar | Polls the active session for cost and tool count; updates every 5 seconds (configurable). |
| Post-mortem viewer | Auto-opens when a session is killed by the watchdog. Shows kill reason, cost at death, and a copyable recovery context. |
| Session browser | Explorer sidebar tree listing all sessions. Select one to open its evidence review. |

---

## Setup

```bash
# 1. Install agent-strace
pip install agent-strace

# 2. Add hooks to Claude Code (one-time)
agent-strace setup

# 3. Open your project in VS Code / Cursor
# The extension activates automatically when .agent-traces/ exists

# 4. Start Claude Code. The status bar item appears immediately.
```

The extension activates automatically when a `.agent-traces/` directory exists in the workspace root. No configuration required.

---

## Commands

All commands are available from the Command Palette (`Cmd/Ctrl+Shift+P`):

| Command | Description |
|---|---|
| `agent-trace: Open Live Stream` | Open the live session window |
| `agent-trace: Open Post-Mortem` | View the watchdog post-mortem for the latest killed session |
| `agent-trace: Refresh Session Browser` | Reload the session list in the Explorer sidebar |
| `agent-trace: Reveal Session` | Jump to a session in the browser |
| `agent-trace: Pause Agent` | Send SIGSTOP to the agent process (requires `watch --agent-pid` running) |
| `agent-trace: Resume Agent` | Send SIGCONT to resume a paused agent |
| `agent-trace: Open Evidence Review` | Open the review panel for the active or selected session |
| `agent-trace: Clear Decorations` | Remove all gutter annotations from the editor |
| `agent-trace: Enable Product Telemetry` | Enable anonymous extension usage telemetry |
| `agent-trace: Disable Product Telemetry` | Opt out and delete the extension's anonymous ID |

---

## Pause / resume

The pause button requires `agent-strace watch` running in a separate terminal, with the agent's process ID:

```bash
# In a separate terminal, start the watcher with the agent's PID
agent-strace watch --agent-pid "$(pgrep -n claude)"

# Then use the Pause button in the event stream panel,
# or run: agent-trace: Pause Agent from the command palette
```

When paused, the agent process receives SIGSTOP and freezes. Resume with the Resume command or SIGCONT.

Pause shows a warning and does nothing if `agent-strace watch` is not running for the session (no heartbeat in the last 15 seconds), or if it was started without `--agent-pid` and so cannot signal the agent. The pause request file holds the session ID, so Resume and session end never clear another session's pause.

`watch` tracks who paused the agent. Resume only releases a pause that the editor started. A pause from a `pause` rule is handed to the editor when you run Pause Agent, so Pause then Resume releases it. A pause from a `require_approval` rule stays until the request is decided with `agent-strace approval`; Resume shows the pending request ID instead. After a denial the agent stays stopped and Pause then Resume releases it.

---

## Evidence review

The review panel works for the active session and for stored sessions selected in **Agent Trace Sessions**. It shows:

- session, provider, capture method, agent identity, start, end, and duration
- evidence-health status, observed reasons, declared provider limits, and unresolved links
- redaction mode and whether recorded content is full, mixed, redacted, or metadata-only
- ordered source event IDs, raw provider fields, parent and child links, annotations, and corrections
- filters for files, commands, tools, tests, failures, retries, recovery, privacy changes, and gaps

Missing, ambiguous, and wrong-type relationships stay unresolved. The extension does not invent a parent or terminal result.

The extension asks the installed `agent-strace` CLI for exact evidence health. If the CLI is unavailable, the panel still shows local structural relationship checks and reports health as unknown or partial.

For Claude Code and Codex sessions, **Verify live capture** runs the complete local `setup --verify --live` check against the selected session. The result stays in the panel and includes static hook setup plus recorded evidence.

### Keyboard navigation

| Key | Action |
|---|---|
| `J`, `K`, arrow keys | Move through visible evidence |
| `Enter` | Open or close raw provider fields |
| `/` | Focus evidence search |
| `Escape` | Clear search |

### Local actions

**Compare** runs the existing local `compare` command. **Save minimized share** writes a local HTML file without session identity, absolute timestamps, raw fields, prompts, commands, paths, results, or annotation text. **Export OTLP JSON** calls the existing export path without an endpoint, so nothing is uploaded.

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
| `agentTrace.telemetry.enabled` | `true` | Send privacy-preserving anonymous extension usage events |

---

## Product telemetry

Anonymous extension telemetry is enabled by default. On first activation, the
extension displays a non-blocking disclosure and an immediate disable action.
Disable it at any time with the `agentTrace.telemetry.enabled` application
setting or **agent-trace: Disable Product Telemetry**. The editor-wide telemetry
switch, `DO_NOT_TRACK=1`, and `AGENT_STRACE_TELEMETRY=0` are also respected.

The extension measures activation, fixed command names and outcomes, and
session success/duration with aggregate tool and error counts. It never sends
prompts, responses, event or trace contents, paths, repository data, session
IDs, command arguments, endpoints, usernames, hostnames, or exception messages.
PostHog person profiles and GeoIP enrichment are disabled. See
[Product telemetry](telemetry.md) for the exact schema.

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

The session browser appears in the Explorer sidebar when `.agent-traces/` exists. It lists all sessions with timestamp, cost estimate, status, and agent name. Select a session to open the review panel. The panel remains available when no session is active.

---

## Release checks

The extension release workflow installs locked dependencies, runs lint, compiles TypeScript, runs extension tests, and then packages the VSIX. The extension version remains independent from the Python package version. Release preparation must bump it in a separate release change when a new marketplace build is intended.
