# Architecture Decision Records

This directory documents significant architectural and design decisions made in agent-strace. Each ADR captures the context, the decision, and its consequences.

## Format

Each ADR follows this structure:

```
# ADR-NNNN: Title

**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-XXXX
**Date:** YYYY-MM
**Deciders:** Name(s)

## Context
What situation or problem prompted this decision?

## Decision
What was decided and how does it work?

## Consequences
What are the trade-offs, limitations, and follow-on effects?
```

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-flat-event-stream-data-model.md) | Flat Event Stream as the Core Data Model | Accepted |
| [0002](0002-ndjson-file-storage-no-database.md) | NDJSON File Storage — No Database | Accepted |
| [0003](0003-zero-runtime-dependencies.md) | Zero Runtime Dependencies | Accepted |
| [0004](0004-hook-integration-via-separate-processes.md) | Hook Integration via Separate OS Processes | Accepted |
| [0005](0005-mcp-proxy-stdio-man-in-the-middle.md) | MCP Proxy as a Transparent stdio Man-in-the-Middle | Accepted |
| [0006](0006-otlp-http-json-no-grpc.md) | OTLP Export via HTTP/JSON — No gRPC, No SDK | Accepted |
| [0007](0007-heuristic-redaction.md) | Heuristic Secret Redaction | Accepted |
| [0008](0008-token-cost-estimation-heuristic.md) | Token and Cost Estimation via Character-Count Heuristic | Accepted |
| [0009](0009-claude-code-jsonl-import.md) | Claude Code JSONL Session Import | Accepted |
| [0010](0010-session-explain-phase-detection.md) | Session Explanation via Prompt-Boundary Phase Detection | Accepted |

## Adding a new ADR

1. Copy the format above into a new file: `NNNN-short-title.md`
2. Use the next available number.
3. Add a row to the index table above.
4. Set status to `Proposed` until the decision is merged.
