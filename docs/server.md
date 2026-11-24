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

Their POSTs do not follow redirects, so set the final URL.

Any other process ignores it, including a plain `python my_agent.py`, `agent-strace record`, and `agent-strace watch`. To stream from `watch`, use `--stream-to` (see [Live streaming from watch](#live-streaming-from-watch)).

The server writes traces in the same `.agent-traces/` format as local mode. All existing CLI commands work against its storage directory.

---

## Local evidence review

Start the collector with its local dashboard:

```bash
agent-strace server --storage ./traces --dashboard
```

Open `http://127.0.0.1:4317/`, select a session, and review ordered evidence, health reasons, provider limits, annotations, raw fields, and unresolved relationships. The dashboard shows session fields and event text as plain text.

The page can compare two sessions, save a privacy-minimized HTML review, and download OTLP JSON. These actions read the configured local store and return local files. They do not upload trace data. If you start a comparison before the previous one finishes, the panel shows only the newest one and drops the result or error of the earlier one. When a session has an event timestamp that is not a Unix time in seconds, the review, compare, and share actions answer 400 with the message the CLI prints, and so does the OTLP action when that event is one it exports. The OTLP action also answers 400 that way when an event it exports, or the session's `started_at` or `ended_at`, is past the latest time OTLP can carry (2554-07-21T23:34:33.709551615Z). When a session's `meta.json` is [corrupt](commands.md#list), the session list leaves it out, and every action on it answers 400 with the message the CLI prints. When a session's events name another session or tenant, or its `events.ndjson` holds a line that is not an event, its events and every action on it answer 400 the same way.

With `--dashboard`, the server defaults to `127.0.0.1`. An unauthenticated dashboard cannot bind to `0.0.0.0`, `::`, a LAN address, or a hostname. To expose it beyond loopback, set an API key explicitly:

```bash
agent-strace server --dashboard --host 0.0.0.0 --auth-key ast_<key>
```

The collector without `--dashboard` keeps its existing `0.0.0.0` default.

The server refuses two kinds of request that web pages can make:

- A POST with an `Origin` header, on any address and with or without `--auth-key`. Browsers add `Origin` to every POST. The `agent-strace` clients never send it.
- On a loopback address without `--auth-key`, a GET whose `Host` is not `localhost`, an IP address or a name given with `--allow-host`. The port is not checked.

Both get 403. This stops other sites from writing sessions into the collector, and stops DNS rebinding pages from reading them. If a GET gets 403 because a tunnel, a forwarded port or a reverse proxy passes its own name as the `Host`, start the server with `--allow-host NAME` for that name, or with `--auth-key`. A collector on `0.0.0.0` or another non-loopback address does not check the `Host`, so DNS rebinding can still read it. Set `--auth-key` there.

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

Any POST with an `Origin` header gets `403`. On a loopback address without
`--auth-key`, a GET whose `Host` is not `localhost`, an IP address or an
`--allow-host` name gets `403` too. See
[Local evidence review](#local-evidence-review).

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

The collector also refuses an event whose `timestamp` is not a Unix time in
seconds from 0 to 9999-12-30T00:00:00Z: text such as an ISO date, `null`, a
boolean, an object, NaN, Infinity, a negative number, or a time after
9999-12-30T00:00:00Z, such as a time in milliseconds. The limit is the same in
every time zone. On Windows it can be lower, near the end of the year 3000.
Such an event is not stored, counts in `errors` and does not create a session.
An event without `timestamp` is stored with the current time. The stderr line
names the event type but never the timestamp:

```
[server] Refused one tool_call event because its timestamp is not a Unix time in seconds.
```

The collector stores and serves events as strict JSON. A NaN or infinite
number in any other field of a `POST /events` line, such as `NaN` or `1e400`
in `data`, is stored as `null`, and a NaN or infinite `duration_ms` is left
out. `GET /sessions/<id>/events` and the dashboard's
`GET /api/sessions/<id>/events` serve each stored event the same way. That
includes a session that an older collector stored with the bare `NaN`,
`Infinity` or `-Infinity` tokens, so `org-report --endpoint` and the dashboard
can read it. A collector older than v0.118.0 could also store such a
`timestamp`. The routes serve it as `null`, and `org-report --endpoint` still
refuses that session. The stored file is not rewritten. Older `watch --stream-to`
clients have one limit, described in
[Live streaming from watch](#live-streaming-from-watch).

`POST /sessions` checks the metadata before it writes anything. It returns
400 with `{"error": "<message>"}` and stores nothing when
`org-report --endpoint` could not read that metadata back from
`GET /sessions`:

- a field with the wrong type, such as text in `tool_calls` or `ended_at`, or
  `true` in `depth`
- a count, such as `tool_calls` or `depth`, that is negative or not a whole
  number
- a time or duration that is negative, NaN, Infinity or too large for a float
- `started_at` of 0, which would be stored as no `started_at`
- NaN or Infinity in a key the collector does not know
- a key the collector does not know that holds 64 or more levels of nested
  lists or objects

It also returns 400 when `started_at`, or `ended_at` when set, is out of
range, such as a time sent in milliseconds. `org-report --endpoint` can read
such a time, but commands run on the collector's store directory cannot.
`list` and `stats` would show the start time or duration as missing, and one
such `started_at` would stop the `tenant` commands and `org-report` for every
tenant in the store. A store written by an older collector, or edited by hand,
can still hold one. See [`tenant`](commands.md#tenant) for the fix.

A request that creates a session also gets 400 when its body has the reserved
key `_unknown_fields`. An update sets only metadata fields, such as `ended_at`
or `tool_calls`, and ignores other keys. The collector checks the metadata as
it would be after the update, so an update to a session whose stored metadata
is already bad gets 400 unless it fixes every bad field. An update cannot fix
a bad value in a key the collector does not know. A refused update changes
nothing, the tenant tag included.

GET session IDs are URL-decoded as one containment-safe path component, so
legacy IDs containing spaces, Unicode, or more than 128 characters remain
readable. POST routes retain the strict 1–128 character ASCII policy for new
session IDs.

`GET /sessions` leaves out a session whose `meta.json` is corrupt: not valid
JSON, not a JSON object, or nested more than 128 levels. `GET /sessions/<id>`
and `GET /sessions/<id>/events` answer 400 for it with
`{"error": "<message>"}`, the message the CLI prints. They answer the same 400
for an ID prefix whose only match is a corrupt session, not 404. See
[`list`](commands.md#list) for the fix.

They also answer 400 with `{"error": "<message>"}` for an ID prefix that
matches two or more sessions, as in
`{"error": "session ID prefix aaaa matches 2 sessions: aaaa111100000001, aaaa222200000002"}`.
An exact ID wins over a longer ID that starts with it. With `--dashboard`, the
session's events and its review, compare, share, and OTLP actions answer the
same way. See [Session IDs](commands.md#session-ids).

`GET /sessions/<id>` and `GET /sessions/<id>/events` answer 404 with
`{"error": "session not found: <id>"}`, as for a missing session, when the
session's directory or `meta.json` is a symlink, or its `meta.json` is not a
regular file. The collector does not follow the link, and `GET /sessions`
leaves the session out. The POST routes still refuse to write to it.

`GET /sessions/<id>/events` also answers 400 with `{"error": "<message>"}` when
an event names another session or tenant, or a line of `events.ndjson` is not
an event. `GET /sessions` still lists that session, so `org-report --endpoint`
stops with `collector request failed with HTTP 400` instead of reporting
without it. A `meta.json` whose `session_id` names another session counts as
corrupt. Before you rebuild it, see
[Sessions with mismatched or unreadable events](commands.md#sessions-with-mismatched-or-unreadable-events).

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

Requests without a matching `Authorization: Bearer <key>` header receive `401 Unauthorized`. `--auth-key` is the server's only authentication; it has no SSO.

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
trusts a private CA bundle, which replaces the system CA store, so include any
public CAs the collector also needs. `--allow-insecure-http` is an explicit opt-in for a
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

Upgrade `watch --stream-to` clients before or with the collector. A client
older than this release writes a NaN or infinite number in an event as a bare
token, and `watch --stream-to` sends that event with the `prev_hash` from the
client's session. The collector stores the number as `null`, so `verify` on the
collector's copy fails at that client's next event. The client's own copy
still verifies. What counts is the release that wrote the event, so a current
`watch` that streams events an older hook wrote has the same limit. The
`agent-strace hook` commands and the Python integrations do not have this
limit when they send to the collector themselves. They send each event with an
empty `prev_hash`, so the collector chains those events itself and they verify.
