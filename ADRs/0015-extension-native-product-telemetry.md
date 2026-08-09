# ADR-0015: Extension-native Product Telemetry

**Status:** Accepted
**Date:** 2026-08
**Deciders:** Siddhant Khare

## Context

The Python CLI telemetry introduced in ADR-0013 and made default-on in ADR-0014
cannot observe VS Code/Open VSX extension activation or feature usage. The
extension reads trace storage directly and does not invoke the CLI, so relying
on CLI events leaves the editor experience invisible.

## Decision

Add an extension-native telemetry client with the same privacy boundaries:

- Send HTTPS/JSON directly to the maintainer-controlled PostHog project using
  Node.js built-ins only; add no runtime package dependency.
- Enable telemetry by default after a one-time, non-blocking disclosure. Provide
  application-scoped enable/disable commands and the
  `agentTrace.telemetry.enabled` setting.
- Honor VS Code's editor-wide telemetry switch, `DO_NOT_TRACK=1`, and
  `AGENT_STRACE_TELEMETRY=0` as opt-outs.
- Store a separate random extension installation ID in VS Code global state and
  delete it when extension telemetry is disabled.
- Allow only extension activation, fixed command names, session lifecycle,
  success, duration, and aggregate tool/error counts. Never accept workspace
  identifiers, paths, repositories, command arguments, trace/session IDs,
  prompts, responses, event contents, or exception messages.
- Disable PostHog person profiles and GeoIP enrichment on every event. Keep
  network delivery best-effort with a short timeout.

## Consequences

- Maintainers can measure extension adoption and feature usage across VS Code,
  Cursor, Windsurf, VSCodium, and other Open VSX-compatible editors.
- CLI and extension installation IDs and persisted preferences remain separate;
  shared environment opt-outs disable both when inherited by the editor host.
- Session completion telemetry reads only aggregate counters already maintained
  by the extension; no trace payload fields enter the analytics schema.
- Extension schema tests and compilation run in pull-request CI under Node 20.
