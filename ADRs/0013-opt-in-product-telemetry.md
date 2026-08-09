# ADR-0013: Opt-in Anonymous Product Telemetry

**Status:** Superseded by ADR-0014
**Date:** 2026-08
**Deciders:** Siddhant Khare

## Context

PyPI download counts show distribution but not whether people successfully set
up agent-strace, capture sessions, return to the product, or encounter failing
commands. The project needs basic product usage signals without weakening its
local-first privacy model or adding runtime dependencies.

Agent traces can contain prompts, tool arguments, file paths, repository data,
and model responses. Product analytics must never reuse or upload that data.

## Decision

Add anonymous, explicit opt-in product telemetry with these boundaries:

- Telemetry is disabled until the user consents through the interactive prompt,
  `agent-strace telemetry enable`, or `AGENT_STRACE_TELEMETRY=1`.
- `DO_NOT_TRACK=1` and `AGENT_STRACE_TELEMETRY=0` disable collection. CI is off
  by default unless explicitly enabled.
- A random installation ID is stored separately in
  `~/.agent-strace/telemetry.json`. Disabling telemetry deletes the ID.
- Events use a fixed property allowlist. Prompts, responses, arguments, paths,
  repository metadata, endpoints, trace/session IDs, exception messages, and
  trace contents cannot enter the payload.
- The client sends HTTPS/JSON directly to a maintainer-controlled PostHog
  project using `urllib.request`. Person profiles and GeoIP enrichment are
  disabled on every event.
- Network failures are silent and never affect command output, behavior, or exit
  status. No on-disk event queue is maintained.
- Internal hook invocations are not tracked individually. Hook integrations emit
  one product event only when an agent session ends.

## Consequences

- Maintainers can measure activation, command adoption, integrations, versions,
  retention, and aggregate failure rates without operating a telemetry database.
- Opt-in data will be incomplete and may contain selection bias; it must not be
  treated as billing, compliance, or security evidence.
- The public PostHog project token must be configured before publishing a release.
- Direct ingestion means PostHog receives the network request even though GeoIP
  enrichment and person-profile processing are disabled. A first-party relay can
  be added later without changing the event schema.
- The implementation preserves the zero-runtime-dependency constraint.
