# Server-side event collector

Run a central collector so agents in containers, CI, and serverless functions can send traces over the network — no local disk required.

See [ADR-0012](../ADRs/0012-server-side-event-collector.md) for design rationale.

---

## Hosted collector

> **Status: planned.** A managed `collector.agent-strace.dev` endpoint is on the roadmap. When available, it will implement the same API as `agent-strace server` — no client changes required.
>
> Track design and implementation progress in [agent-trace issue #129](https://github.com/Siddhant-K-code/agent-trace/issues/129).

Once live, the only change needed is the endpoint:

```bash
# Self-hosted (today)
export AGENT_STRACE_ENDPOINT=http://localhost:4317

# Hosted (when available, no other changes)
export AGENT_STRACE_ENDPOINT=https://collector.agent-strace.dev
export AGENT_STRACE_AUTH_KEY=ast_<your-key>
```

The hosted endpoint will implement the same API documented below, with API key auth (see [Authentication](#authentication)) and a free tier for individual developers.

---

## Quick start

```bash
# Start the collector
agent-strace server --port 4317 --storage ./traces

# On each agent machine, before the agent starts
export AGENT_STRACE_ENDPOINT=http://collector:4317
```

Only two producers read `AGENT_STRACE_ENDPOINT`:

- The `agent-strace hook` commands that agent CLI hooks run (Claude Code, Codex, Gemini CLI, Cursor, and GitHub Copilot CLI; see [CLI hooks setup](setup.md#option-1-cli-hooks-recommended)). They send events and session metadata to the collector.
- The Python framework integrations in `agent_trace.integrations` (see [integrations.md](integrations.md)). They send events only.

Any other process ignores it, including a plain `python my_agent.py`, `agent-strace record`, and `agent-strace watch`. To stream from `watch`, use `--stream-to` (see [Live streaming from watch](#live-streaming-from-watch)).

The server writes traces in the same `.agent-traces/` format as local mode. All existing CLI commands work against its storage directory.

---

## Local evidence review

Start the collector with its local dashboard:

```bash
agent-strace server --storage ./traces --dashboard
```

Open `http://127.0.0.1:4317/`, select a session, and review ordered evidence, health reasons, provider limits, annotations, raw fields, and unresolved relationships.

The page can compare two sessions, save a privacy-minimized HTML review, and download OTLP JSON. These actions read the configured local store and return local files. They do not upload trace data.

With `--dashboard`, the server defaults to `127.0.0.1`. An unauthenticated dashboard cannot bind to `0.0.0.0`, `::`, a LAN address, or a hostname. To expose it beyond loopback, set an API key explicitly:

```bash
agent-strace server --dashboard --host 0.0.0.0 --auth-key ast_<key>
```

The collector without `--dashboard` keeps its existing `0.0.0.0` default.

Use the event query parameter to target a recorded event:

```text
http://127.0.0.1:4317/session/<session-id>?event=<event-id>
```

---

## Docker

```dockerfile
FROM python:3.12-slim
RUN pip install agent-strace
ENV AGENT_STRACE_STORAGE=/data
VOLUME /data
EXPOSE 4317
CMD ["agent-strace", "server", "--port", "4317"]
```

```bash
docker build -t agent-strace-server .
docker run -p 4317:4317 -v $(pwd)/traces:/data agent-strace-server
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/events` | Receive a batch of NDJSON events |
| `POST` | `/sessions` | Create or update session metadata |
| `GET` | `/sessions` | List all sessions |
| `GET` | `/sessions/<id>` | Read metadata for a session |
| `GET` | `/sessions/<id>/events` | Stream events for a session |
| `GET` | `/health` | Liveness check |

Events are accepted as NDJSON (`application/x-ndjson`), one event per line.
Each line is stored or refused on its own. The collector refuses a line that is
not a valid event, has an invalid session ID, or has a `data` value that is not
a JSON object: an array, a string (including `""`), a number, a boolean, or
`null`. An event without `data` is stored with an empty object. The response is
`{"accepted": N, "errors": M}`, where `errors` counts the lines that were not
stored, with status 200 when every line was stored and 202 otherwise. An event refused for
its `data` is not stored and does not create a session. The collector prints
one line to stderr for it, naming the event type and the JSON type of `data`
but never data values or event IDs:

```
[server] Refused one tool_call event because its data is an array, not an object.
```

GET session IDs are URL-decoded as one containment-safe path component, so
legacy IDs containing spaces, Unicode, or more than 128 characters remain
readable. POST routes retain the strict 1–128 character ASCII policy for new
session IDs.

Upgrade the collector before its clients. A v0.106.0 or older collector
returns 400 for session metadata that contains `conversation_id`. See
[production.md](production.md#version-skew).

---

## Multi-agent correlation

When multiple agents send to the same collector, sessions are linked via `parent_session_id` and `parent_event_id` in session metadata. Use `agent-strace replay --tree` or `agent-strace a2a-tree` to visualise the full call graph.

---

## Authentication

By default the server runs unauthenticated (local use). For any network-accessible deployment, enable API key auth:

```bash
# Generate a key
agent-strace server keygen
# → ast_a3f9b2c1d4e5f6a7b8c9d0e1f2a3b4c5

# Start server with key enforcement
agent-strace server --auth-key ast_a3f9b2c1d4e5f6a7b8c9d0e1f2a3b4c5

# Or via environment variable
AGENT_STRACE_AUTH_KEY=ast_a3f9b2c1d4e5f6a7b8c9d0e1f2a3b4c5 agent-strace server
```

Requests without a matching `Authorization: Bearer <key>` header receive `401 Unauthorized`.

The key authorizes the entire collector instance; it is not scoped to a
`tenant_id`. Treat one collector and storage root as one organization boundary,
and run separate authenticated instances for organizations that must not access
one another. Tenant filters are customer-level scoping inside that boundary,
not an authorization mechanism. Managed multi-organization hosting is tracked
separately in [issue #129](https://github.com/Siddhant-K-code/agent-trace/issues/129).

### Organization reports from a collector

`agent-strace org-report --endpoint https://collector.example.com` reads the
existing `GET /sessions` and `GET /sessions/<id>/events` endpoints. It refuses
collectors that expose the session list without authentication. Supply the key
through `AGENT_STRACE_AUTH_KEY` or `--auth-key-file`; there is no literal key
flag, and `AGENT_STRACE_ENDPOINT` is deliberately ignored for this reporting
command.

The read client requires HTTPS except for loopback HTTP, rejects redirects and
credential-bearing URLs, disables ambient HTTP proxies, and applies bounded
response, session, event, line, nesting, and total-byte limits. `--ca-file`
adds a private CA. `--allow-insecure-http` is an explicit opt-in for a
non-loopback test collector and should not be used on untrusted networks.

Collector reports use one metadata request followed by one event request for
each session selected after the month/team metadata filters. These N+1 reads
are bounded but are not a transactionally consistent snapshot; the generated
report records that limitation. See [Organization reporting](org-report.md).

**Client side:** set `AGENT_STRACE_AUTH_KEY` alongside `AGENT_STRACE_ENDPOINT` in the environment the agent starts from. The `agent-strace hook` commands and the Python integrations then add the `Authorization: Bearer` header to everything they send to the collector:

```bash
export AGENT_STRACE_ENDPOINT=https://collector.example.com
export AGENT_STRACE_AUTH_KEY=ast_a3f9b2c1d4e5f6a7b8c9d0e1f2a3b4c5
```

`agent-strace watch --stream-to` does not send an `Authorization` header and has no flag to add one, so a collector started with `--auth-key` rejects its requests with 401. Stream from `watch` only to a collector that runs without `--auth-key`.

Key format: `ast_` prefix + 32 hex characters. Generated with `secrets.token_hex(16)` — no new dependencies.

---

## Live streaming from watch

Stream events to the collector in real-time during a watched session:

```bash
agent-strace watch \
  --stream-to http://collector:4317/events \
  --stream-batch-size 20 \
  --stream-flush-interval 5.0 \
  SESSION_ID
```

HTTP failures are logged to stderr but never interrupt the watch loop. Each failure line, like the `[watch] Streaming events to` line at startup, names only the collector's `scheme://host[:port]`. Watch does not follow redirects, so pass the final URL, and it refuses a `--stream-to` URL that carries a user name or password. When watch stops, it waits up to 15 seconds for the last POST. See [Live OTLP streaming](production.md#live-otlp-streaming) for the startup checks, the stop wait, and `--stream-otlp`.
