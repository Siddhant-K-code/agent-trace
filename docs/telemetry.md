# Product telemetry

agent-strace can send anonymous product usage events to a PostHog project owned
by the project maintainer. This is separate from agent tracing. The CLI client
never reads `.agent-traces/`; the editor extension sends only session success,
duration, and aggregate tool/error counts from the state it already maintains.
Neither client sends trace contents.

Telemetry is enabled by default on configured CLI and editor-extension builds.
Each surface shows a one-time, non-blocking disclosure, and sends events only
after that notice has been shown or telemetry was explicitly enabled. The CLI
notice appears either on stderr during the first interactive command or, for
hooks-only installs, inside the agent's own UI at session start where the
agent can show it; see [When sending starts](#when-sending-starts). CI remains
disabled by default to avoid ephemeral installation identifiers and test noise.

## When sending starts

The CLI shows its notice once per installation, in the first of these places
that reaches a person. From then on, both CLI commands and hooks send events.

| Where | When the notice appears |
|---|---|
| Interactive `agent-strace` command | On stderr, before the first interactive command. Piped or scripted runs never show it. |
| Claude Code hooks (plugin or `setup --cli claude`) | As a hook message at the start of an interactive `claude` terminal session. Session events are sent from that session's end onward. Other entry points, such as `claude -p` and the Agent SDK, never show it. |
| Gemini CLI hooks | As a hook message at session start: in the terminal UI, or on stderr for `gemini -p`. Session events are sent from that session's end onward. |
| Codex hooks | As a `warning:` line at the start of an interactive session. `codex exec` never shows it. On Codex versions that run the `SessionEnd` hook, session events are sent from that session's end onward. Codex 0.143.0 ignores `SessionEnd`. Installs made before setup registered `SessionEnd` send none until you run `agent-strace setup --cli codex` again and trust the new hook in `/hooks`. The notice still enables later `agent-strace` commands. |
| Cursor and GitHub Copilot CLI hooks | Never: their session-start hooks have no output the user sees. These installs send nothing until one of the places above shows the notice, `agent-strace telemetry enable` is run, or `AGENT_STRACE_TELEMETRY=1` is set. |

The hook message reads:

```text
agent-strace: Anonymous product telemetry is enabled by default. No prompts, arguments, paths, repository data, or trace contents are sent. Disable at any time with: agent-strace telemetry disable
```

It uses each agent's display-only `systemMessage` field, so it is not added to
the model's context. A hook marks the notice as shown only after writing it for
a session that displays it, and only one of several hooks starting at once
shows it. No notice is shown, and nothing is sent, when `DO_NOT_TRACK=1` or
`AGENT_STRACE_TELEMETRY=0` is set, after `agent-strace telemetry disable`, or
in CI.

## User controls

```bash
agent-strace telemetry status
agent-strace telemetry enable
agent-strace telemetry disable

# Environment-level overrides
DO_NOT_TRACK=1 agent-strace list
AGENT_STRACE_TELEMETRY=0 agent-strace list
AGENT_STRACE_TELEMETRY=1 agent-strace list
```

An unset preference means telemetry is enabled, waiting for the one-time notice;
`agent-strace telemetry status` says when it is still waiting. A random
installation ID is created on the first event that is sent and stored in
`~/.agent-strace/telemetry.json`
(mode `0600`; updates are locked and atomic, see [Security](security.md#local-config-and-credential-files)).
`agent-strace telemetry disable` persists the opt-out and deletes the
identifier and the current [analytics session](#analytics-sessions). `DO_NOT_TRACK=1` and `AGENT_STRACE_TELEMETRY=0` disable telemetry
without changing the stored preference. CI is disabled by default;
`AGENT_STRACE_TELEMETRY=1` is required to enable it there.

The VS Code/Open VSX extension has its own application-scoped preference:

- Set `agentTrace.telemetry.enabled` to `false`, or run
  **agent-trace: Disable Product Telemetry** from the Command Palette.
- Run **agent-trace: Enable Product Telemetry** to re-enable it.
- VS Code's editor-wide telemetry switch, `DO_NOT_TRACK=1`, and
  `AGENT_STRACE_TELEMETRY=0` also disable extension events.

Like the CLI, the extension sends nothing until its own notice has been shown,
or until telemetry was explicitly enabled with the setting, the enable command,
or `AGENT_STRACE_TELEMETRY=1`.

The extension stores a separate random ID in VS Code global state and deletes
it, along with its in-memory analytics session, when disabled. CLI and extension preferences are separate; environment-level
opt-outs apply to both when inherited by the editor extension host.

## Collected fields

The CLI emits three events:

| Event | Event-specific fields |
|---|---|
| `agent_strace_cli_command_completed` | command, subcommand, success, exit code, duration, error type, integration, export format/backend |
| `agent_strace_session_completed` | provider, capture method, success, duration when known |
| `agent_strace_telemetry_enabled` | explicit enablement source |

Every event also includes the anonymous installation ID, a random
[analytics session ID](#analytics-sessions), telemetry schema version,
agent-strace version, Python major/minor version, OS family, and CI boolean.

The editor extension emits five events:

| Event | Event-specific fields |
|---|---|
| `agent_strace_vscode_extension_activated` | none |
| `agent_strace_vscode_command_completed` | fixed command name, success, duration, error type |
| `agent_strace_vscode_session_started` | none |
| `agent_strace_vscode_session_completed` | success, duration, aggregate tool-call count, aggregate error count |
| `agent_strace_vscode_telemetry_enabled` | enablement source |

Extension events also include the random extension installation ID, a random
analytics session ID, schema and extension versions, editor family/version,
desktop/web UI kind, remote boolean, and OS family.

The schema does **not** accept prompts, responses, command arguments, file paths,
repository names or hashes, usernames, hostnames, endpoints, trace contents,
trace or agent session IDs, model input/output, exception messages, or stack
traces. PostHog person-profile processing and GeoIP enrichment are disabled for
every event. Like any direct HTTPS destination, PostHog still receives the
network request under its own infrastructure and privacy terms.

## Analytics sessions

PostHog's server-side SDKs send no `$session_id`, so without one every event
stands alone and PostHog's session aggregations are empty. Each client
therefore adds its own random
[`$session_id`](https://posthog.com/docs/data/sessions#server-sdks-and-sessions)
that groups one stretch of use:

- It is a fresh UUIDv7, as PostHog requires for custom session IDs. Its
  embedded time is the session's start, which is never later than the
  session's first event.
- A new session starts after 30 minutes without an event, or 24 hours after
  the session started, PostHog's own session rules. A session is never reused
  after it ends.
- The CLI keeps the current session in `~/.agent-strace/telemetry.json`
  (`analytics_session_id` and `analytics_session_last_event_ms`), so CLI
  commands and agent hooks running in separate processes share it.
- The extension keeps its session in memory, so each editor window has its
  own, and a restarted window starts a new one.
- It is random. It is never derived from, and never equals, a trace session ID
  or an agent's session ID, so it cannot link an event to a recorded trace.
  The CLI drops any `$session_id` or `session_id` passed as an event property.
- Disabling telemetry deletes it with the installation ID.

## Maintainer setup

No database or deployed service is required. Use a managed PostHog project:

1. Create a PostHog Cloud project and choose the US or EU region.
2. Open **Project settings** and copy the **Project token**. This is the public
   event-ingestion token, not a personal API key.
3. Set the public project token and regional host in
   `src/agent_trace/telemetry.py`. Set the same values in
   `vscode-extension/src/telemetry.ts` for extension releases.

   ```python
   DEFAULT_POSTHOG_PROJECT_TOKEN = "phc_your_project_token"
   DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"  # or https://eu.i.posthog.com
   ```

   ```typescript
   const POSTHOG_PROJECT_TOKEN = "phc_your_project_token";
   const POSTHOG_HOST = "https://us.i.posthog.com";
   ```

4. Verify locally without editing the constants first. Run these in an
   interactive terminal: `list` shows the one-time notice and then sends its
   event, while a piped or scripted run sends nothing until the notice has been
   shown.

   ```bash
   export AGENT_STRACE_TELEMETRY_TOKEN="phc_your_project_token"
   export AGENT_STRACE_TELEMETRY_HOST="https://us.i.posthog.com"
   export AGENT_STRACE_TELEMETRY_CONFIG="$(mktemp)"

   agent-strace list
   agent-strace telemetry status
   agent-strace telemetry disable
   ```

5. In PostHog's live events view, confirm that
   `agent_strace_cli_command_completed` arrived and contains only the documented
   properties. Running `agent-strace telemetry enable` explicitly also sends
   `agent_strace_telemetry_enabled`.
6. Install a development VSIX, activate a workspace containing `.agent-traces/`,
   and confirm `agent_strace_vscode_extension_activated` arrives. Exercise one
   extension command and verify only allow-listed properties are present.
7. Commit the public project token and publish the release. End users do not set
   a token; it is part of the distributed package.

`AGENT_STRACE_TELEMETRY_TOKEN`, `AGENT_STRACE_TELEMETRY_HOST`, and
`AGENT_STRACE_TELEMETRY_CONFIG` are intended for development, downstream
distributions, and tests. Network delivery uses a 0.5-second timeout and is
best-effort; failures never change CLI behavior or exit status. The extension
also uses a 0.5-second timeout, keeps no on-disk queue, and never changes editor
behavior when delivery fails.

## Suggested PostHog dashboard

Create these insights:

1. **Weekly active installations:** unique users performing
   `agent_strace_cli_command_completed`.
2. **Command adoption:** event count grouped by `command`, filtered to successful
   events, excluding `share` events with `subcommand = dry_run` (`share --dry-run`
   previews).
3. **Integration adoption:** successful `setup` events grouped by `integration`, excluding `subcommand = verify`.
4. **Reliability:** failed command events divided by all command events, grouped
   by command and agent-strace version.
5. **Activation funnel:** successful `setup` command (excluding `subcommand = verify`) →
   `agent_strace_session_completed` → successful `replay`, `inspect`, `explain`,
   or `export` command.
6. **Retention:** users whose first `agent_strace_session_completed` event is
   followed by another successful command in later weeks.
7. **Extension adoption:** unique installations performing
   `agent_strace_vscode_extension_activated`, grouped by editor and extension
   version.
8. **Extension feature usage:** `agent_strace_vscode_command_completed` grouped
   by command and success.
9. **Sessions per installation:** a trends formula `A / B` over
   `agent_strace_cli_command_completed`, where A aggregates by **Unique
   sessions** and B by **Unique users**. Repeat with the extension events.
10. **Commands per session:** the same event's total count divided by its
    **Unique sessions**, grouped by agent-strace version.
11. **Session length:** **Property value median** of the **Session duration**
    session property on `agent_strace_cli_command_completed`.

Do not use this telemetry for billing, access control, compliance, or security
evidence. Open-source clients and public ingestion endpoints can be modified or
spoofed.
