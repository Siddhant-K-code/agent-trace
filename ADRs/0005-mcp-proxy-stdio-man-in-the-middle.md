# ADR-0005: MCP Proxy as a Transparent stdio Man-in-the-Middle

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

MCP (Model Context Protocol) servers communicate with agents over stdio using JSON-RPC. To capture tool calls without modifying the agent or server, a proxy must sit between them and forward messages in both directions while recording events.

## Decision

The MCP proxy (`proxy.py`) spawns the target server as a subprocess, then uses two threads to forward messages bidirectionally:

- **agent → server thread**: reads from stdin, classifies and records the message, writes to the server's stdin.
- **server → agent thread**: reads from the server's stdout, classifies and records the message, writes to stdout.

A `threading.Event` (`stop`) coordinates shutdown when either direction closes.

**Message framing auto-detection:** `_read_message()` detects two framing modes:
- LSP-style `Content-Length` headers (if the first non-empty line starts with `Content-Length:`)
- Newline-delimited JSON (if the first non-empty line starts with `{`)

**Message classification** maps JSON-RPC methods to `TraceEvent` types:

| JSON-RPC method | Direction | EventType |
|---|---|---|
| `tools/call` | agent→server | `TOOL_CALL` |
| response with `result.content` | server→agent | `TOOL_RESULT` |
| `resources/read` | agent→server | `FILE_READ` |
| `sampling/createMessage` | agent→server | `LLM_REQUEST` |

The JSON-RPC `id` field links `tool_call` to `tool_result` events via `parent_id`, enabling accurate latency measurement.

## Consequences

- **Transparent to both sides** — neither the agent nor the server needs modification.
- **Dual framing support** broadens compatibility with MCP server implementations.
- **JSON-RPC `id` linking** is more reliable than the hooks approach (which uses `tool_name`) because request IDs are unique per request.
- **Server stderr is forwarded unchanged** — server diagnostic output is not traced or inspected.
- **Thread-based forwarding** adds minimal latency (microseconds) compared to direct stdio.
- **HTTP/SSE variant** (`http_proxy.py`) uses `http.server` and `http.client` from stdlib, listening only on `127.0.0.1` (loopback) to avoid network exposure. SSE responses are forwarded line-by-line with `wfile.flush()` after each line for real-time delivery.
