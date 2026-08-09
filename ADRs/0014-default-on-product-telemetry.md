# ADR-0014: Default-on Anonymous Product Telemetry

**Status:** Accepted
**Date:** 2026-08
**Deciders:** Siddhant Khare

## Context

ADR-0013 introduced privacy-preserving product telemetry as explicit opt-in.
Opt-in collection cannot reliably show how the broader CLI population adopts
commands and integrations. The strict event allowlist and separation from trace
storage make default collection possible without uploading agent content.

## Decision

Supersede ADR-0013's consent default while retaining its data boundaries:

- Telemetry is enabled when no preference has been stored. The first
  interactive CLI invocation displays a non-blocking, one-time disclosure.
- `agent-strace telemetry disable` stores an opt-out and deletes the anonymous
  installation ID. `DO_NOT_TRACK=1` and `AGENT_STRACE_TELEMETRY=0` also disable
  collection, while `agent-strace telemetry enable` re-enables it explicitly.
- CI and test processes remain disabled by default. CI can opt in with
  `AGENT_STRACE_TELEMETRY=1`.
- The fixed event allowlist, excluded sensitive fields, disabled PostHog person
  profiles and GeoIP enrichment, silent failures, and direct HTTPS/JSON delivery
  from ADR-0013 remain unchanged.

## Consequences

- Maintainers receive more representative aggregate CLI adoption and
  reliability signals.
- Users can opt out before any event by running the disable command or setting a
  supported environment variable.
- Documentation and the first interactive disclosure must make the default and
  controls prominent.
- Direct PostHog requests remain observable to PostHog infrastructure even with
  GeoIP enrichment disabled.
