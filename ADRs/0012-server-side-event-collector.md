# ADR-0012: Server-Side Event Collector

**Status:** Accepted  
**Date:** 2026-05  
**Deciders:** Siddhant Khare

## Context

agent-strace writes traces to the local filesystem. This works for a single
developer on a single machine but breaks for:

- Agents running in containers or serverless functions (no persistent local disk)
- CI pipelines where each run is ephemeral
- Multi-agent systems where traces from parallel agents need to be correlated
- Teams where the person running the agent is not the person reading the traces

## Decision

Add `agent-strace server` — a lightweight HTTP collector that receives events
from remote agents and stores them in the same `.agent-traces/` format as local
mode. Agents opt in by setting `AGENT_STRACE_ENDPOINT`.

### Transport

HTTP/JSON only (per ADR-0006). No gRPC, no WebSockets, no message queues.

### API

| Method | Path | Description |
|---|---|---|
| POST | /events | Receive a batch of NDJSON events |
| POST | /sessions | Create or update session metadata |
| GET | /sessions | List all sessions |
| GET | /sessions/\<id\>/events | Stream events for a session (NDJSON) |
| GET | /health | Liveness check |

### Storage

The server writes `.agent-traces/<session-id>/` directories identical to local
traces. All existing CLI commands work against server storage without modification.

### No authentication in v1

Intended for internal/private network use. Authentication can be added via a
reverse proxy (nginx, Caddy). This is documented explicitly.

### Implementation

Uses Python stdlib `http.server` — no new runtime dependencies (ADR-0003).
Thread safety via a single `threading.Lock` around all store writes.

## Consequences

- Agents in containers, CI, and serverless can now send traces to a central
  collector without any code changes — just set `AGENT_STRACE_ENDPOINT`.
- The server is single-process and not horizontally scalable in v1. This is
  acceptable for team-scale use (< 100 concurrent agents).
- No authentication means the server must not be exposed to the public internet.
  This is documented in the README.
