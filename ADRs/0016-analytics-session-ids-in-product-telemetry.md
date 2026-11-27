# ADR-0016: Analytics Session IDs in Product Telemetry

**Status:** Accepted
**Date:** 2026-11
**Deciders:** Siddhant Khare

## Context

The CLI (ADR-0014) and the editor extension (ADR-0015) send events to PostHog
with direct HTTP calls, the way PostHog's server-side SDKs do. Server-side
SDKs send no `$session_id`, so every event stands alone: PostHog's **Unique
sessions** aggregation, session properties such as session duration, and
per-session ratios are empty. Maintainers can count installations and events,
but not how many separate stretches of use an installation has or how many
commands one of them involves.

Server-side events may carry a client-generated `$session_id`. PostHog's
[custom session ID requirements](https://posthog.com/docs/data/sessions#custom-session-ids)
ask for an ID that is consistent within a session, never reused, and a
UUIDv7 whose embedded time is no later than the session's first event and
less than 24 hours before its last. Events that do not meet them are left
out of session aggregations. PostHog's own clients start a new session after
30 minutes without an event or 24 hours after the session started.

ADR-0013 and ADR-0015 list "trace/session IDs" among the identifiers telemetry
must never accept. That rule protects the link between an analytics event and
a recorded trace or an agent's session.

## Decision

Add a random analytics `$session_id` to every product-telemetry event:

- Use a fresh UUIDv7 whose embedded time is the session's start. Use the
  same clock reading, clamped to the range a UUIDv7 time can hold, for the
  rotation check and the event timestamp, so the event never precedes its
  session.
- Start a new session after 30 idle minutes, 24 hours after the start, or when
  the last event time is not between the start and now: the clock moved back,
  or the stored value is malformed. Never reuse an ended session.
- In the CLI, keep the session in `telemetry.json` beside the installation ID,
  updated under the existing file lock, so CLI commands and hooks in separate
  processes share it. Create it only after the disclosure gate and only while
  a valid installation ID exists. Build the UUIDv7 with the standard library,
  since `uuid.uuid7` needs Python 3.14.
- In the extension, keep the session in memory: one per editor window, with
  no new persisted state.
- Delete the session whenever telemetry is disabled and the installation ID
  is deleted.
- Never derive the ID from a trace session ID or an agent session ID, and
  never accept a session ID from event properties. The exclusion of
  "trace/session IDs" in ADR-0013 and ADR-0015 refers to these trace and
  agent identifiers, and it still applies.
- Keep the telemetry schema version unchanged: the property is additive.

## Consequences

- Maintainers can use **Unique sessions**, session duration, sessions per
  installation, and events per session for both clients.
- A session ID adds no information about the user beyond the installation ID
  it is sent with: it is random and groups events that share that ID.
- The privacy wording in the notice, setting, README, and docs now says
  "trace session IDs", so it stays true while a random analytics session ID is
  sent.
- `telemetry.json` gains two fields. Older clients ignore them, and a client
  replaces a malformed session instead of failing.
- CLI and extension sessions never merge, because they also have separate
  installation IDs.
