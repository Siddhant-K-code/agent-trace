# Assignment submissions and review

Assignment mode creates a deliberately minimized process-telemetry bundle for
human review. It is not a hiring decision, a correctness test, or a measure of
candidate or task quality.

## Create a submission

```bash
agent-strace share --assignment SESSION_ID --output submission.zip
```

When `SESSION_ID` is omitted, the latest session is used. The output must end
in lowercase `.zip`. It is written atomically with owner-only permissions.
Do not use ordinary `agent-strace share` for this workflow: ordinary replay
HTML can contain prompts, results, commands, and paths.

Bundle schema v1 has exactly these members:

| Member | Contents |
|---|---|
| `manifest.json` | Version, privacy/limitations, summary, and SHA-256 consistency digests |
| `trace.ndjson` | Allowlisted events with relative time and anonymous references |
| `replay.html` | Offline viewer regenerated only from the sanitized trace and aggregates |
| `stats.json` | Process status, duration, counts, and documented metric semantics |
| `cost.json` | Aggregate token estimate and frozen offline pricing profile |
| `lint.json` | Frozen deterministic signals over minimized telemetry |

The bundle omits raw identity, prompts, results, commands, paths, file contents,
environment data, URLs, and custom model identifiers. Event, tool, resource,
and model references are first-use aliases such as `Tool-001`. Timestamps are
relative to the earliest recorded event. `completed` means only that a
successful process termination was recorded; it does not establish that the
assignment was completed or correct.

The manifest hashes provide internal consistency, not a signature, provenance,
or issuer authenticity. Cost is an offline estimate: assignment v1 serializes
selected input/output event payloads canonically, estimates one token per four
characters (minimum one per input/output event), and applies the named frozen
pricing profile. It is not a tokenizer result or provider bill.

## Treat submissions as hostile

Do not extract a received ZIP or open its HTML first. Validate and score it:

```bash
agent-strace score submission.zip --rubric examples/hiring/rubric.yaml
agent-strace score submissions/ --rubric examples/hiring/rubric.yaml --compare
```

The scorer reads members in memory without extraction and rejects noncanonical
members, paths, metadata, compression streams, sizes, duplicate keys, digest
mismatches, derived-report changes, and regenerated-viewer differences.
Comparison reads only direct lowercase `.zip` children and reports anonymous
`Submission-NNN` references rather than source filenames.

## Rubric format

Rubrics use a deliberately small YAML subset parsed with the standard library.
Keys, indentation, scalar types, duplicates, Unicode controls, and bounds are
strictly checked. Weights must be positive whole numbers. Every criterion is
binary: it receives its full weight when met and zero otherwise. The displayed
percentage is `100 × awarded weight / total weight`; comparison ranking uses
the exact fraction, then the bundle digest for stable ordering. Displayed
percentages use decimal half-up rounding to two places; ranking never uses the
rounded display value. Equal exact fractions share a rank.

```yaml
task: "Implement rate limiting middleware"
max_cost_usd: 0.50
max_duration_minutes: 30
criteria:
  - name: task-completed
    scorer: session_status
    expected: completed
    weight: 3
  - name: cost-efficiency
    scorer: cost_usd
    threshold: 0.50
    fail_on: above
    weight: 2
  - name: no-tool-loops
    scorer: lint_violations
    rule: tool-loop
    threshold: 0
    weight: 2
```

`max_cost_usd` and `max_duration_minutes` are report context only; criteria must
declare thresholds explicitly. `session_status` requires `expected` and defaults
to `fail_on: not_equal`. Numeric scorers require `threshold` and default to
`fail_on: above`; `below` reverses the boundary. Boundaries are inclusive: an
`above` criterion is met at or below the threshold, and a `below` criterion is
met at or above it.

| Scorer | Evidence | Extra fields |
|---|---|---|
| `session_status` | Recorded `completed`, `timeout`, `terminated`, or `not_recorded` | `expected` |
| `cost_usd` | Frozen-profile offline estimate | `threshold`, optional `fail_on` |
| `duration_seconds` | Span between earliest and latest validated events | `threshold`, optional `fail_on` |
| `error_count` | Explicit error-event count | `threshold`, optional `fail_on` |
| `tool_call_count` | Recorded tool-call count | `threshold`, optional `fail_on` |
| `redundant_read_ratio` | Repeated anonymous reads after the first / all reads | `threshold`, optional `fail_on` |
| `lint_violations` | All available findings or one `rule` | `threshold`, optional `fail_on`, optional `rule` |

Assignment-v1 rule scoring supports `tool-loop`, `reasoning-spiral`,
`context-saturation`, `redundant-read`, `error-retry-loop`, and `no-output`.
`budget-proximity` and `post-compaction-regression` are unavailable because
their required raw context is intentionally omitted; absence is not evidence
that no such behavior occurred, and rubrics naming them are rejected.

## Responsible interpretation

Review the declared arithmetic and its evidence alongside an independent review
of the actual work product. Do not infer identity, protected attributes, skill,
intent, task correctness, or outcome quality from anonymous process telemetry.
Establish a documented human-review process and apply it consistently. JSON
output is versioned for audit-friendly downstream handling, but it should not
be wired directly to an automated employment decision.
