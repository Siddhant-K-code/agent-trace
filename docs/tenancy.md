# Multi-tenant agent deployments

`agent-strace` can tag sessions and events with an application customer ID,
scope queries and costs to that ID, and perform customer export and erasure
workflows. Tenant tags are additive: sessions recorded before v0.90.0 remain
readable and appear as `(untagged)`.

## Tag every customer session

Assign the tenant when monitoring an active session:

```bash
agent-strace watch --tenant-id customer-acme
```

For hooks, CI, auto-instrumentation, or other non-interactive capture paths,
set the environment variable before the agent starts:

```bash
export AGENT_STRACE_TENANT_ID=customer-acme
python my_agent.py
```

`tenant_id` is stored at the top level of `meta.json` and every event in
`events.ndjson`. The session value is authoritative: agent-strace rejects an
event with a different tenant and refuses to move an already-tagged session to
another tenant. When `watch` tags events already written to the active session,
it atomically rewrites the file and rebuilds the SHA-256 event chain. The
pre-change and post-change terminal hashes are recorded in
`.agent-traces/tenant-audit.ndjson`. A small durable intent journal is used so
an interrupted metadata/event transition can roll forward on restart.

New session and workspace IDs accepted by capture and collector entry points
use a strict 1–128 character ASCII format (`A-Z`, `a-z`, digits, `.`, `_`, and
`-`). Existing safe directory basenames remain readable, reportable,
exportable, and deletable even when they contain spaces, Unicode, or exceed
128 characters. Legacy names must still be a single path component: path
separators, dot segments, NUL/control characters, and symlink escapes are
rejected.

Parent and child sessions must resolve inside the same flat/workspace store
and have the same tenant. A newly created untagged child inherits a tagged
parent's tenant. Legacy untagged parent/child pairs remain readable, but
tagging either side is rejected if it would leave the relationship split
between tagged and untagged (or differently tagged) sessions. For a legacy
untagged tree, unlink the relationship, tag both sessions, and then relink it
as one coordinated migration.

Use stable opaque customer identifiers rather than names, emails, or other
personal data. Tenant IDs must be 256 characters or fewer and cannot contain
control characters.

## Scoped queries and cost attribution

```bash
# Only this customer's sessions
agent-strace list --tenant customer-acme

# Estimated cost over the last 30 days
agent-strace cost --tenant customer-acme --since 30d

# Actual recorded provider/model counters for the same tenant
agent-strace cost --tenant customer-acme --breakdown provider --since 30d

# Cost allocation across every tenant for one UTC month
agent-strace tenant report --month 2026-06
agent-strace tenant report --month 2026-06 --format json
```

The standard estimate uses the selected offline pricing model. Provider
breakdowns use recorded model/token counters and the dated bundled pricing
snapshot. Both perform exact tenant matching before event data is loaded.

Tenant administration uses fail-closed store discovery. Unlike the ordinary
interactive session list, report, export, and deletion abort when a plausible
session directory has malformed metadata/events, missing core files, unsafe
symlinks, or a directory/metadata ID mismatch. Workspace discovery likewise
aborts on symlinked or unsafe workspace roots and children, so an omitted store
cannot produce a false zero report or false successful erasure.

## Subject access export

Export all metadata and events for one customer as a single JSON document:

```bash
agent-strace tenant export customer-acme > customer-acme.json
# or
agent-strace tenant export customer-acme --output customer-acme.json
```

The versioned export contains `tenant_id`, `exported_at`, `session_count`, and
a `sessions` array with complete metadata, event history, session sidecars, and
compaction checkpoints. Its versioned `external_records` section also contains
matching human-approval requests (including tool input), eval-dataset entries,
and retention-log records. It searches the flat store and every workspace
store, regardless of `AGENT_STRACE_WORKSPACE`, and includes relevant hash-only
tenant audit evidence. Review the output's storage location and access controls
before sending it to a data subject.

## Right to erasure

Preview the customer ID carefully, create an access export if required by your
policy, and then run the explicitly confirmed deletion:

```bash
agent-strace tenant delete customer-acme --confirm
```

This is irreversible. It removes every matching session directory, including
annotations, evals, identities, postmortems, and other files stored within it,
matching compaction checkpoints, approval requests, dataset entries, and
retention-log records. Shared JSONL files are atomically filtered so malformed
and unrelated records remain unchanged. Other tenants and untagged sessions
are untouched.

Before removing data, the command fsyncs a hash-only `pending` record to
`.agent-traces/tenant-deletions.ndjson`; it appends `completed` or `partial`
afterward. Tenant IDs and session IDs are never stored there in plaintext.
Hashed session IDs act as tombstones so late hook events cannot recreate erased
trace data. A recoverable intent journal finishes interrupted deletions during
ordinary `TraceStore` startup as well as tenant administration commands. The
operation covers the flat store and every workspace store and clears matching
workspace hook state. Hook markers and pending-call files live beneath their
workspace store. Provider IDs are represented by a path-safe local-session
component and SHA-256 digest rather than being copied into filenames. Legacy
root markers are removed only when no surviving flat or workspace session
makes their ownership ambiguous; markerless pending state is removed only when
its safe suffix identifies an erased session.

Export and deletion reject symlinked session metadata, event streams,
checkpoint directories/files, audit files, and external-record sources. This
fail-closed behavior prevents a trace store from reading or deleting a target
outside its configured root.

Tenant audit files and completed journal directories are outside normal session
retention. Set an explicit operational retention/rotation schedule for
`tenant-audit.ndjson` and `tenant-deletions.ndjson` based on legal advice and
backup policy. Incomplete files under `.tenant-journal/` must not be deleted
until recovery succeeds or an operator has reconciled the recorded intent.

## Isolation and hosted collectors

Tenant tags scope agent-strace queries; they do not replace operating-system
permissions or collector authorization. Anyone who can read the trace directory
can read its NDJSON files, and anyone who can run local commands against it can
select another tenant ID.

The built-in collector's bearer key protects a collector instance, which must
be treated as the organization boundary; it is not an individual-tenant key.
For a hard hosted multi-organization boundary, deploy a separate storage
directory/collector per organization or place each collector behind an
authenticated gateway that derives organization scope from server-side key
metadata. Never accept an organization ID supplied by the client as the
authorization decision. Within an authorized organization, `tenant_id` can be
used as the customer-level tag.

OTLP and live `--stream-otlp` exports attach `tenant_id` to every generated
span and the OTLP resource. Configure equivalent access controls and retention
at the observability backend.
