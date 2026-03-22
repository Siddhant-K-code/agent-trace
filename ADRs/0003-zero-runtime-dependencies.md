# ADR-0003: Zero Runtime Dependencies

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

agent-strace is a developer tool that must be installable in any Python environment without conflict. Many AI agent frameworks already carry large dependency trees (LangChain, LlamaIndex, etc.). A tracing tool that adds its own transitive dependencies risks version conflicts and increases install friction.

## Decision

The `[project]` section of `pyproject.toml` declares no `dependencies`. Every feature is implemented using the Python standard library only:

| Feature | stdlib module used |
|---|---|
| HTTP proxy | `http.server`, `http.client` |
| OTLP export | `urllib.request` |
| JSON serialization | `json` |
| File storage | `pathlib`, `os` |
| CLI | `argparse` |
| Redaction | `re` |
| Session IDs | `uuid` |
| Timestamps | `time`, `datetime` |
| Threading | `threading` |

The constraint applies to runtime dependencies only. Development/test tooling (pytest, hatchling) is not subject to this restriction.

## Consequences

- **`pip install agent-strace` always succeeds** regardless of the target environment's existing packages.
- **No HTTP client library** — OTLP export uses `urllib.request`, which lacks connection pooling, retry logic, and async support. This is acceptable for a fire-and-forget export at session end.
- **No tokenizer** — cost estimation uses the `len(text) // 4` heuristic instead of a real tokenizer. See ADR-0008.
- **No async I/O** — the HTTP proxy uses threads. This is sufficient for the expected request rates (one agent session at a time).
- **Requires Python ≥ 3.10** — `str | None` union syntax and `match` statements are used in several places. This is the minimum version that supports these features without `from __future__ import annotations` workarounds.
