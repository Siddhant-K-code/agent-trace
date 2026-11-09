# AI agent pull request evidence

AgentTrace can generate a privacy-minimized Evidence Packet, check pull request
reviewability, attach schema v1 cognitive debt coverage, and publish the files
as GitHub Action artifacts. The local dashboard and VS Code extension provide
the same conservative per-session evidence review surface.

## Local workflow

```bash
agent-strace health SESSION_ID
agent-strace share SESSION_ID --dry-run
agent-strace share SESSION_ID --output evidence-packet.html
agent-strace cognitive-debt --session SESSION_ID --format json --output cognitive-debt.json
agent-strace review SESSION_ID --format markdown \
  --evidence-packet evidence-packet.html \
  --cognitive-debt cognitive-debt.json
```

The default Evidence Packet omits prompts, responses, commands, paths, tool
inputs, tool results, and other content-rich values. It keeps event order,
timing, tool names, effect categories, evidence health, known gaps, redaction,
and shareable annotations. `share --include-content` is the explicit
full-content mode.

`review` reports rationale, scope, validation, stewardship, and evidence
health. Each finding includes a status, confidence, and source. Prompt text and
command strings are not printed.

## Pull request comment

```bash
agent-strace pr-comment SESSION_ID --review \
  --evidence-packet evidence-packet.html \
  --cognitive-debt cognitive-debt.json \
  --dry-run
```

Remove `--dry-run` to post or update the comment. The exit code follows the
configured reviewability threshold. Cognitive debt in `check` mode also fails
unless its result is a pass. An `unknown` cognitive debt status is unavailable,
not passing.

## GitHub Action artifacts

With `review: "true"`, the Action can generate and upload:

- a metadata-only `evidence-packet.html`;
- schema v1 `cognitive-debt.json`;
- Shields-compatible `cognitive-debt-badge.json`;
- `review.json` and `review.md`.

The Action writes the review table to the step summary and can post the same
section to the pull request. Fork pull requests skip comments when the token is
read-only, but the local check still runs. See the
[reviewability guide](review.md#github-action) for the workflow and output
contract.

## Local evidence review

The [session evidence review](session-review-projection.md) is shipped in the
local dashboard and the [VS Code extension](vscode.md#evidence-review). It
shows ordered evidence, raw provider fields, relationship gaps, evidence
health, privacy state, annotations, filters, and local compare, minimized
share, and OTLP export actions.

## Evidence limits

- A trace contains only events exposed by the provider and capture path.
- A recorded test command does not prove that all relevant tests ran.
- A successful tool result does not prove that the code change caused a later
  outcome.
- Missing events can mean absent capture, provider limitations, or unfinished
  work.
- The default Evidence Packet is minimized, but `--include-content`, source
  traces, and the `agent-traces` workflow artifact can contain sensitive data.
- Evidence health reports structural usability, not code quality, security
  certification, legal compliance, or reviewer approval.

Treat the trace as review input. Keep the code diff, CI results, and human
review as separate evidence sources.

## Related documentation

- [PR reviewability check](review.md)
- [Evidence Packet command](commands.md#share)
- [PR comment command](commands.md#pr-comment)
- [GitHub Action inputs and outputs](commands.md#github-actions)
- [Cognitive debt schema and badge](cognitive-debt.md)
- [VS Code evidence review](vscode.md#evidence-review)
- [Evidence health](evidence-health.md)
- [Share disclosure preview](share-disclosure-preview.md)
- [Session review projection](session-review-projection.md)
- [Capture matrix](capture-matrix.md)
