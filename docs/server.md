# Server-side event collector

Run a central collector so agents in containers, CI, and serverless functions can send traces over the network — no local disk required.

See [ADR-0012](../ADRs/0012-server-side-event-collector.md) for design rationale.

---

## Quick start

```bash
# Start the collector
agent-strace server --port 4317 --storage ./traces

# Agents point to it via environment variable — no code changes required
AGENT_STRACE_ENDPOINT=http://collector:4317 python my_agent.py
```

The server writes traces in the same `.agent-traces/` format as local mode. All existing CLI commands work against its storage directory.

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
| `GET` | `/sessions/<id>/events` | Stream events for a session |
| `GET` | `/health` | Liveness check |

Events are accepted as NDJSON (`application/x-ndjson`), one event per line.

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

**Client side** — set `AGENT_STRACE_AUTH_KEY` alongside `AGENT_STRACE_ENDPOINT` and all outbound requests include the header automatically:

```bash
export AGENT_STRACE_ENDPOINT=https://collector.example.com
export AGENT_STRACE_AUTH_KEY=ast_a3f9b2c1d4e5f6a7b8c9d0e1f2a3b4c5
python my_agent.py
```

The `--stream-headers` flag on `agent-strace watch` also works for one-off overrides:

```bash
agent-strace watch --stream-to https://collector.example.com \
  --stream-headers "Authorization=Bearer ast_..."
```

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

HTTP failures are logged to stderr but never interrupt the watch loop.
