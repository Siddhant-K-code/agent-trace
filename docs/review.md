# PR reviewability check

`agent-strace review` reports what the authoring session can show a reviewer
about an AI-written pull request. It reads the captured sessions linked to the
branch and reports five findings. It can run as a required status check and
post one compact PR comment.

```bash
agent-strace review                      # terminal text for this branch
agent-strace review --format markdown    # PR comment body
agent-strace review --format json        # versioned JSON
agent-strace pr-comment --review         # post or update the PR comment, exit per threshold
```

## Why

Rahman et al. surveyed 239 practitioners about reviewing AI-authored pull
requests ([arXiv 2610.11179](https://arxiv.org/abs/2610.11179), October 2026).
They name the gap "reviewability debt". Reviewers decide how much effort to
spend based on whether a PR arrives with bounded scope, project-grounded
rationale, validation beyond CI, a responsive contributor, and identifiable
ownership after merge.

Most of that evidence exists in the authoring session and is lost by the time
the PR opens. This check puts it on the PR.

## Findings

Each finding has a status, a confidence, and the source of its evidence.

| Finding | What it checks | Source |
|---|---|---|
| Rationale | Whether the original request was captured | `user_prompt` events. Counts only. Prompt text is never printed. |
| Scope | Files written vs files a prompt named or the session read | File and edit tool events, prompt path mentions, file reads |
| Validation | Tests and checks run inside the session, with exit codes | Shell tool calls and their linked results. CI is not read. |
| Stewardship | Human turns, interventions after the agent started, approval decisions | `user_prompt` events, approval requests, recorded permission modes |
| Evidence health | Capture gaps, redaction state, provider limits, and how the session was linked | [`health`](evidence-health.md), [capture matrix](capture-matrix.md), session link |

Statuses:

| Status | Meaning |
|---|---|
| `present` | The evidence was captured |
| `partial` | Some evidence was captured, or it is stale or failed |
| `missing` | The capture path records this evidence and none was found |
| `inconclusive` | The trace cannot say either way |

Confidence is `high`, `medium`, or `low`. It says how much the status depends on
heuristics. Prompt capture is high. Path matching for scope is medium. A weak
session link lowers evidence confidence to low.

Validation uses the last run of each check. A check that passed before later
file edits is reported as `partial`. Command strings are never printed. They
are mapped to fixed labels such as `pytest`, `npm test`, `ruff`, or `tsc`.

Stewardship counts turns and decisions. It does not rank or score people.

## How sessions are linked

| Link | Meaning |
|---|---|
| `explicit` | Session IDs were named on the command line |
| `branch` | The session's recorded git branch matches the PR branch |
| `commit_time` | No branch match. The session closest to the head commit time was used. |
| `latest` | No branch or commit to match. The latest session was used. |

`commit_time` and `latest` are weak links. Evidence confidence drops to `low`
and a `present` evidence result becomes `partial`.

With no linked session, every finding is `inconclusive` at `low` confidence.
The default threshold passes in that case.

## Threshold

The threshold decides which findings fail the check. The default fails on
`missing` at `medium` or higher confidence. It never fails on `partial` or
`inconclusive` unless you configure it to.

Add a `review` block to `.agent-strace.yaml`:

```yaml
review:
  fail_on: missing, partial     # or none
  min_confidence: medium        # low, medium, high
  findings: rationale, validation
```

`.agent-strace.yml` and `.agent-strace.json` also work. `--fail-on`,
`--min-confidence`, `--findings`, and `--config` override the file.

Exit codes: `0` pass, `1` threshold failed, `2` bad config or unknown session.

## GitHub Action

```yaml
on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  reviewability:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Siddhant-K-code/agent-trace@main   # pin a release tag
        with:
          run-eval: "false"
          review: "true"
          post-pr-comment: "true"
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

The step posts or updates one PR comment with the session summary and the
reviewability table. It always writes the table to the step summary. It exits
non-zero per the threshold, so it works as a required status check. Fork PRs
skip the comment because their tokens are read-only, but still run the check.

The Action reads sessions from `trace-dir` (default `.agent-traces`). Commit
the sessions, or restore them from an artifact, before this step.

## Sample comment section

```markdown
### Reviewability

| Finding | Status | Confidence | Evidence | Source |
|---|---|---|---|---|
| Rationale | present | high | 3 prompts captured in 2 sessions. | user\_prompt events \(content not shown\) |
| Scope | partial | medium | 1 of 2 written files were named in a prompt or read in the session. Not named or read: `src/extra.py` | file and edit tool events, prompt path mentions, file reads |
| Validation | present | high | Last run: pytest passed, exit 0; ruff passed, exit 0. | shell tool calls and linked results inside the session \(not CI\) |
| Stewardship | present | medium | 3 human turns, 1 after the first tool call, 0 approval decisions. Permission modes: default. | user\_prompt events, approval requests, recorded permission modes |
| Evidence health | partial | high | Health: 2 partial \(claude\). Redaction: on. Linked by: branch. | agent-strace health, capture matrix, session link |

Check: **pass**. Threshold fails on `missing` at `medium` or higher confidence (default).
```

## Limits

- The paper is survey evidence about what reviewers say they want. It does not
  show that attaching this evidence changes review outcomes.
- The check reports captured evidence only. It does not claim complete
  capture. Providers differ; see the [capture matrix](capture-matrix.md).
- Validation is in-session only. It does not read CI results.
- Scope uses path matching against prompts and reads. A file can be in scope
  without being named, and the check will still report it as not named.
- Output is deterministic. No LLM writes or summarizes any part of it.
