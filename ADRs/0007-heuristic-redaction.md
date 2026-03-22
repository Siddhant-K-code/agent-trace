# ADR-0007: Heuristic Secret Redaction

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

Agent tool calls frequently contain secrets: API keys in environment variables, tokens in HTTP headers, passwords in connection strings. Traces stored on disk or exported to observability platforms must not leak these secrets. Redaction must work without knowing the specific secrets in advance.

## Decision

Redaction (`redact.py`) uses two independent heuristic layers applied to every string value in event data:

**Layer 1 — Key-name matching:** If a dict key matches any name in `SENSITIVE_KEYS` (18 case-insensitive names: `password`, `token`, `secret`, `api_key`, `authorization`, etc.), the entire value is replaced with `[REDACTED]`.

**Layer 2 — Value pattern matching:** 11 compiled regexes detect common secret formats by their structure:
- API key prefixes (`sk-`, `AKIA`, `xoxb-`, `ghp_`, etc.)
- JWT structure (three base64 segments separated by dots)
- Bearer tokens in Authorization headers
- Connection strings with embedded credentials
- High-entropy hex strings (40+ chars)

Pattern matching uses `re.sub()` to replace matched substrings inline, preserving surrounding context (e.g., `curl -H 'Authorization: [REDACTED]' https://api.example.com`).

Redaction is opt-in, controlled by `AGENT_TRACE_REDACT=1`. It is applied recursively to all dict and list values in event data.

## Consequences

- **False positives** — a 40+ char hex string that is a file hash or commit SHA will be redacted. This is accepted as a trade-off for a heuristic approach.
- **False negatives** — novel secret formats not matching any pattern will not be redacted. The pattern set is not exhaustive.
- **Inline replacement** preserves non-secret context, making redacted traces more useful for debugging than full-value replacement.
- **Opt-in by default** — redaction is disabled unless explicitly enabled, avoiding performance overhead and false positives in development environments.
- **No secret scanning at rest** — redaction happens at capture time only. Secrets already written to `events.ndjson` before redaction was enabled are not retroactively redacted.
