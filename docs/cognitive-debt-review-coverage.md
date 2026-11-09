# Cognitive debt and review coverage

`agent-strace cognitive-debt` estimates how much recorded agent-written code
lacks review signals. It is a prioritization report, not a measurement of what
a human actually read or understood.

## Run the report

```bash
agent-strace cognitive-debt --since 30d --by branch
agent-strace cognitive-debt --session SESSION_ID --format json
agent-strace cognitive-debt --since 30d --github-token "$GH_TOKEN"
```

The command derives changed paths and estimated written lines from recorded
file-write and write-like tool events. It then looks for local git signals
after the session. With a GitHub token, it can also look for a merged pull
request, reviews, review comments, and comments on the same path.

## What the score means

For one session:

```text
debt score = max(0, estimated agent-written lines - estimated reviewed lines)
             / estimated agent-written lines
```

The report groups scores by author or branch and flags sessions above the
selected threshold.

## Limits

- Written-line counts are estimates derived from recorded payloads, not a git
  patch attribution model.
- Reviewed-line counts are weighted heuristics from commit, pull request, and
  comment signals.
- A review or comment does not prove that every changed line was examined.
- No review signal does not prove that no review happened.
- When git is unavailable, reviewed lines can be reported as zero. Treat that
  as missing review evidence, not confirmed zero review.
- GitHub enrichment depends on repository, branch, token, and merged pull
  request discovery.
- Provider capture gaps can omit file activity before the report runs.

Use the report to choose where to inspect code and traces. Do not use it as an
individual performance score, security certification, or compliance result.

## Related documentation

- [Cognitive debt command](commands.md#cognitive-debt)
- [Capture matrix](capture-matrix.md)
- [Evidence health](evidence-health.md)
- [AI agent pull request evidence](ai-agent-pr-evidence.md)
