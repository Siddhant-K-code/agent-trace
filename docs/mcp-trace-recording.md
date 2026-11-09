# MCP trace recording and trace-reader server

AgentTrace can proxy an MCP server to record observed JSON-RPC traffic. It can
also expose the resulting local trace store as a read-only MCP tool server for
an analysis client.

## Record an MCP stdio server

Place `agent-strace record` where the client would normally launch the server:

```bash
agent-strace record --name local-mcp -- /path/to/mcp-server
```

The proxy forwards stdin and stdout while recording the MCP messages it sees.
The [record command reference](commands.md#record) covers parent-session links
and redaction controls.

## Record an MCP HTTP or SSE server

```bash
agent-strace record-http --url http://127.0.0.1:8000/mcp --port 5100 --name local-mcp
```

Point the MCP client at `http://127.0.0.1:5100`. The proxy forwards traffic to
the upstream URL.

## Query traces through MCP

```bash
agent-strace mcp
```

The trace-reader server uses MCP over stdio and exposes tools for listing
sessions, reading stored events, searching events, summarizing a session, and
diffing two sessions. Configure an MCP client to launch `agent-strace mcp` as a
local command.

## Boundaries and risks

- The proxy records only traffic that passes through it.
- It does not observe provider reasoning, UI actions, direct filesystem access,
  direct shell activity, or network calls outside the proxied MCP connection.
- Transport failures can leave a partial trace.
- Redaction is heuristic and does not certify that a trace is free of secrets
  or personal data.
- The trace-reader can return raw stored event data. Give it access only to
  clients that are allowed to read the trace directory.
- `agent-strace mcp` is a local stdio server. It is not the authenticated HTTP
  collector documented in [server.md](server.md).

Use `agent-strace health SESSION_ID` before relying on a recorded session for
review.

## Related documentation

- [MCP recording setup](setup.md#option-2-mcp-proxy)
- [Command reference](commands.md#session-capture)
- [Trace-reader command](commands.md#mcp)
- [Evidence health](evidence-health.md)
- [Collector server](server.md)
