# Product telemetry

agent-strace can send anonymous product usage events to a PostHog project owned
by the project maintainer. This is separate from agent tracing: no `.agent-traces/`
files or event data are read by the telemetry client.

Telemetry is disabled until the user explicitly opts in. On a configured build,
the first interactive CLI invocation asks once, with a default of **No**.
Non-interactive use and CI do not prompt.

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

Consent and a random installation ID are stored in
`~/.agent-strace/telemetry.json`. Disabling telemetry deletes the identifier.
CI is disabled by default; `AGENT_STRACE_TELEMETRY=1` is required to enable it.

## Collected fields

There are three events:

| Event | Event-specific fields |
|---|---|
| `agent_strace_cli_command_completed` | command, subcommand, success, exit code, duration, error type, integration, export format/backend |
| `agent_strace_session_completed` | provider, capture method, success, duration |
| `agent_strace_telemetry_enabled` | consent source |

Every event also includes the anonymous installation ID, telemetry schema
version, agent-strace version, Python major/minor version, OS family, and CI
boolean.

The schema does **not** accept prompts, responses, command arguments, file paths,
repository names or hashes, usernames, hostnames, endpoints, trace contents,
session IDs, model input/output, exception messages, or stack traces. PostHog
person-profile processing and GeoIP enrichment are disabled for every event.
Like any direct HTTPS destination, PostHog still receives the network request
under its own infrastructure and privacy terms.

## Maintainer setup

No database or deployed service is required. Use a managed PostHog project:

1. Create a PostHog Cloud project and choose the US or EU region.
2. Open **Project settings** and copy the **Project token**. This is the public
   event-ingestion token, not a personal API key.
3. In `src/agent_trace/telemetry.py`, set:

   ```python
   DEFAULT_POSTHOG_PROJECT_TOKEN = "phc_your_project_token"
   DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"  # or https://eu.i.posthog.com
   ```

4. Verify locally without editing the constants first:

   ```bash
   export AGENT_STRACE_TELEMETRY_TOKEN="phc_your_project_token"
   export AGENT_STRACE_TELEMETRY_HOST="https://us.i.posthog.com"
   export AGENT_STRACE_TELEMETRY_CONFIG="$(mktemp)"

   agent-strace telemetry enable
   agent-strace list
   agent-strace telemetry status
   ```

5. In PostHog's live events view, confirm that
   `agent_strace_telemetry_enabled` and
   `agent_strace_cli_command_completed` arrived and contain only the documented
   properties.
6. Commit the public project token and publish the release. End users do not set
   a token; it is part of the distributed package.

`AGENT_STRACE_TELEMETRY_TOKEN`, `AGENT_STRACE_TELEMETRY_HOST`, and
`AGENT_STRACE_TELEMETRY_CONFIG` are intended for development, downstream
distributions, and tests. Network delivery uses a 0.5-second timeout and is
best-effort; failures never change CLI behavior or exit status.

## Suggested PostHog dashboard

Create these insights:

1. **Weekly active installations:** unique users performing
   `agent_strace_cli_command_completed`.
2. **Command adoption:** event count grouped by `command`, filtered to successful
   events.
3. **Integration adoption:** successful `setup` events grouped by `integration`.
4. **Reliability:** failed command events divided by all command events, grouped
   by command and agent-strace version.
5. **Activation funnel:** successful `setup` command →
   `agent_strace_session_completed` → successful `replay`, `inspect`, `explain`,
   or `export` command.
6. **Retention:** users whose first `agent_strace_session_completed` event is
   followed by another successful command in later weeks.

Do not use this telemetry for billing, access control, compliance, or security
evidence. Open-source clients and public ingestion endpoints can be modified or
spoofed.
