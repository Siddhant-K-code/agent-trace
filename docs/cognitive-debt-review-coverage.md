# Cognitive debt and review coverage

`agent-strace cognitive-debt` reports review coverage for observed
agent-written code. It is a repository and workflow metric. It does not measure
code quality, claim causation, or rank individuals.

## Generate a report and badge

```bash
agent-strace cognitive-debt --since 30d --by repository
agent-strace cognitive-debt --since 30d --by branch --format json --output cognitive-debt.json
agent-strace cognitive-debt --since 30d --by repository --format badge --output cognitive-debt-badge.json
```

`--format json` writes the versioned schema v1 report. `--format badge` writes
Shields-compatible static JSON. `--output` writes the complete result
atomically. The command writes no partial report when validation or output
handling fails.

## Reviewed, unreviewed, and unknown

The report keeps three coverage states separate:

- `reviewed`: observed agent-written lines with estimated review evidence.
- `unreviewed`: observed lines remaining after estimated review coverage when
  review state is known.
- `unknown`: authorship or review state could not be established.

Unknown is not zero review and is not a passing result. If git history is
unavailable, affected lines remain unknown instead of becoming unreviewed. A
missing path on a recorded write also produces unknown authorship and review
coverage.

Schema v1 reports `schema_version: 1`, `agent_written_lines`, reviewed,
unreviewed, and unknown line counts and percentages, plus authorship and review
states. The unreviewed ratio is `null` when review state is unknown.

## Status and grouping

| Status | Meaning |
|---|---|
| `clean` | All observed agent-written lines have review evidence |
| `partial` | Review is known and unreviewed coverage is within the configured threshold |
| `above_threshold` | Review is known and unreviewed coverage exceeds `--threshold` |
| `unknown` | No agent-written lines were observed, or authorship or review coverage is incomplete |

Use `--by repository` or `--by branch`. JSON `groups` contain those aggregates,
and `workflows` contain per-session details. The legacy `--by author` value is
only an alias for repository grouping and never emits an individual ranking.

## Interpretation limits

- Written-line counts are estimates derived from recorded payloads, not a git
  patch attribution model.
- Review coverage is estimated from local git and optional GitHub pull request,
  review, and comment signals.
- A review signal does not prove that every changed line was examined.
- No review signal does not prove that no review happened.
- Provider capture gaps can omit file activity before the report runs.
- The badge is presentation output. Use the schema v1 JSON report for
  automation.

Use this report to choose repositories, branches, and workflows for review. Do
not use it as an individual performance score, security certification, or
compliance result.

## Related documentation

- [Cognitive debt badge guide](cognitive-debt.md)
- [Cognitive debt command](commands.md#cognitive-debt)
- [Capture matrix](capture-matrix.md)
- [Evidence health](evidence-health.md)
- [AI agent pull request evidence](ai-agent-pr-evidence.md)
