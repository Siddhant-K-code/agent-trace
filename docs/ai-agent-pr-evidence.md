# AI agent pull request evidence

AgentTrace can attach a recorded authoring-session summary to a pull request.
Current releases provide local health checks, replay, disclosure preview, and
the `pr-comment` summary. A minimized Evidence Packet and a dedicated PR Check
are active feature work and are not available on current `main`.

## Current workflow

```bash
agent-strace health SESSION_ID
agent-strace share SESSION_ID --dry-run
agent-strace pr-comment SESSION_ID --dry-run
agent-strace replay SESSION_ID --format html -o session-review.html
```

After reviewing the dry-run output, `agent-strace pr-comment SESSION_ID` can
post or update the structured summary on the pull request associated with the
current branch. The summary reports recorded facts such as files, tool counts,
errors, lint findings, and replay commands.

## Active dependencies

| Capability | Current status |
|---|---|
| Evidence Packet | Active feature work. The minimized, manifest-backed PR evidence artifact is not in the current release. |
| PR Check | Active feature work. The planned reviewability findings and CI threshold are not in the current release. |
| Review projection | Core projection exists, but it does not yet provide the complete PR evidence UI or check. See [session review projection](session-review-projection.md). |

Do not document planned packet or check commands as released until those
changes land and their canonical command documentation is updated.

## Evidence limits

- A trace contains only events exposed by the provider and capture path.
- A recorded test command does not prove that all relevant tests ran.
- A successful tool result does not prove that the code change caused a later
  outcome.
- Missing events can mean absent capture, provider limitations, or unfinished
  work.
- Replay HTML contains event payloads and can contain sensitive data.
- Evidence health reports structural usability, not code quality, security
  certification, legal compliance, or reviewer approval.

Treat the trace as review input. Keep the code diff, CI results, and human
review as separate evidence sources.

## Related documentation

- [PR comment command](commands.md#pr-comment)
- [Evidence health](evidence-health.md)
- [Share disclosure preview](share-disclosure-preview.md)
- [Session review projection](session-review-projection.md)
- [Capture matrix](capture-matrix.md)
