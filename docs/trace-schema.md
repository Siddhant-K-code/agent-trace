# Public trace schema

AgentTrace publishes JSON Schema draft 2020-12 contracts for the two canonical session records:

- [`meta.json`](../src/agent_trace/schemas/trace/v1/meta.schema.json)
- One [`events.ndjson`](../src/agent_trace/schemas/trace/v1/event.schema.json) record

The schemas are language-neutral release assets. They also ship inside the Python wheel under `agent_trace/schemas/trace/v1/`.

## Version and compatibility policy

Schema v1 describes the existing on-disk session layout. It does not add a version field to stored records and does not rewrite a session.

Compatibility rules:

1. Existing session directories remain `meta.json` plus append-only `events.ndjson`.
2. Fields omitted by the existing serializers remain optional unless they identify the record. `meta.json` requires `session_id` and `started_at`. Each event requires `event_type`, `timestamp`, `event_id`, and object-valued `data`.
3. Legacy events may omit `session_id` and every `prev_hash`. The containing metadata identity is authoritative.
4. The published v1 accepted-instance set is immutable. A new top-level field or event type publishes a new schema version instead of silently widening v1.
5. A backward-compatible schema version must continue to accept v1 records. It may add optional fields while keeping the same storage filenames and append-only event layout.
6. Removing a field, renaming a field, changing a field type, changing an event type meaning, or changing parent semantics is breaking and requires maintainer approval.
7. Storage migrations are separate from schema publication. A new schema version does not require rewriting historical sessions.

The schema version is carried by the schema URI and validation report. It is not injected into historical records.

## Independent validator

```bash
agent-strace validate [TARGET] [--format text|json]
agent-strace validate --corpus [--format text|json]
```

`TARGET` can be a session ID, a session prefix, or a session directory. The default is the latest session in the selected trace store. The command exits `0` only for a conformant `valid` report. It exits `1` for `partial` or `invalid`.

The validator uses the packaged JSON Schemas for record shape, then reuses the bounded assignment reader, compliance record checks, evidence-health relationships, hash-chain verifier, and regression fixture projection. It never writes metadata, events, hashes, or derived files.

### Rules

| Area | Rule |
|---|---|
| JSON | UTF-8 JSON, no duplicate object keys, finite numbers, bounded record sizes, no blank NDJSON records |
| Metadata identity | `meta.json.session_id` equals the containing directory name |
| Event identity | A present event `session_id` equals metadata; an omitted legacy value inherits metadata for validation |
| Event type | `event_type` is one of the published v1 enum values |
| Event data | `data` is a JSON object |
| Event IDs | Every ID is non-empty and unique |
| Parent links | Tool and LLM outcomes use the existing evidence-health request relationships; other parent links must name an earlier event. LLM results with no `parent_id` are a declared gap, not orphans, when the hook provider's [capture matrix](capture-matrix.md) entry lists them in `results_without_requests` |
| Ordering | Timestamps do not decrease, and outcomes or children do not precede their parent |
| Terminal outcomes | Finalized tool and LLM requests have one recorded result or terminal error |
| Session boundaries | A finalized trace has valid start and end boundaries. Events before the first start are invalid unless the hook provider's [capture matrix](capture-matrix.md) entry lists every one of their types in `events_before_start` |
| Hash chain | The existing byte-level `prev_hash` rule is checked without changing legacy records |
| Metadata summaries | Explicit tool-call, LLM-request, and error counts must equal observed events |
| Usage presence | Missing `total_tokens` means unreported; explicit `0` means reported zero |
| Replay baseline | `--replay-baseline` uses the existing regression fixture tool-call projection |

`--allow-active` treats absent terminal outcomes as in-flight evidence. It does not claim that an active trace is complete.

### Hash-chain states

| State | Meaning |
|---|---|
| `empty` | No event records |
| `legacy` | Multiple records and no stored hash links |
| `partial` | Linked and legacy records are mixed |
| `unanchored` | Every available link is valid, but the terminal digest is not externally signed or anchored |
| `broken` | A stored link does not match the previous exact NDJSON line |

JSON key order and whitespace affect line hashes. A producer that intentionally reserializes events must rebuild the chain as one explicit storage operation. The validator never does this.

A line hash covers the line's bytes without its line ending. agent-strace writes `\n` line endings on every OS. Older versions wrote `\r\n` on Windows; those files still verify, and new events appended to them end in `\n`.

## Status and labels

`valid` means the trace conforms to the independent rules. `partial` means the records are parseable but evidence is incomplete or has a non-fatal relationship gap. `invalid` means shape, identity, uniqueness, ordering, integrity, summary, or replay checks failed.

Findings have severity `error`, `warning` or `info`. Any `error` makes the report `invalid`, and any `warning` makes it `partial`. An `info` finding records a gap the provider's capture metadata declares, such as `declared_events_before_session_start` or `declared_results_without_requests`, and does not change the status.

Labels are stable machine-readable facts used by the conformance corpus. Examples include `legacy`, `current`, `missing_outcome`, `terminal_absent`, `tampered`, `redacted`, `recovered`, `summary_contradiction`, `usage_missing`, `usage_zero`, `declared_events_before_session_start`, `declared_results_without_requests`, and `replay_drift`.

The validator reports evidence consistency. It does not prove that the agent completed the user task, that a provider exposed every action, or that an unsigned trace is authentic.
