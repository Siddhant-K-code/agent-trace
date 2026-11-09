# Cognitive debt review coverage

`agent-strace cognitive-debt` reports how much observed agent-written code has review evidence. It is a workflow and repository metric. It does not measure code quality, claim that agent use caused an outcome, or rank individuals.

## Local generation

Generate both the versioned report and a Shields-compatible static badge:

```bash
mkdir -p artifacts
agent-strace cognitive-debt \
  --since 30d \
  --by repository \
  --format json \
  --output artifacts/cognitive-debt.json
agent-strace cognitive-debt \
  --since 30d \
  --by repository \
  --format badge \
  --output artifacts/cognitive-debt-badge.json
```

Both files are complete static JSON. CI can upload them without an AgentTrace service.

## Status and coverage

The report keeps reviewed, unreviewed, and unknown lines separate. Unknown authorship or review state never becomes zero debt.

| Status | Meaning |
|---|---|
| `clean` | All observed agent-written lines have review evidence |
| `partial` | Review is known and some lines lack review evidence, within the configured threshold |
| `above_threshold` | Review is known and the unreviewed ratio is above `--threshold` |
| `unknown` | No agent-written lines were observed, or authorship or review state is incomplete |

The JSON report has `schema_version: 1`. Its root `coverage` object includes:

- `agent_written_lines`
- `reviewed_lines` and `reviewed_percent`
- `unreviewed_lines` and `unreviewed_percent`
- `unknown_review_lines` and `unknown_review_percent`
- `unknown_authorship_lines`
- `authorship_state` and `review_state`
- `unreviewed_ratio`, which is `null` when review state is unknown

`workflows` contains per-session detail. `groups` contains repository or branch aggregates. Neither contains author rankings. The `interpretation` object records that this is review coverage, not code quality or causal evidence.

Malformed trace data, malformed write-tool arguments, malformed local review records, and malformed GitHub review responses fail closed. The command writes no report or badge when input validation fails.

## Shields badge

The badge output follows the Shields endpoint schema:

```json
{
  "schemaVersion": 1,
  "label": "cognitive debt",
  "message": "60% reviewed, 40% unreviewed",
  "color": "yellow"
}
```

Commit or publish `cognitive-debt-badge.json`, then point Shields at its raw URL:

```text
https://img.shields.io/endpoint?url=URL_ENCODED_RAW_JSON_URL
```

Use the versioned report for automation. Use the badge only for presentation.

## CI and release assets

A release job can generate immutable evidence and upload both files:

```bash
mkdir -p artifacts
agent-strace cognitive-debt \
  --since 30d \
  --by repository \
  --format json \
  --output artifacts/cognitive-debt.json
agent-strace cognitive-debt \
  --since 30d \
  --by repository \
  --format badge \
  --output artifacts/cognitive-debt-badge.json
gh release upload "$GITHUB_REF_NAME" \
  artifacts/cognitive-debt.json \
  artifacts/cognitive-debt-badge.json \
  --clobber
```

For an always-current badge, publish the badge JSON to a stable branch path. For release evidence, keep both JSON files as release assets. A later PR Evidence Check can read `artifacts/cognitive-debt.json`, require `schema_version == 1`, and treat `status == "unknown"` as unavailable evidence rather than a passing result.
