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

Validation uses the last run of each check across all linked sessions. Sessions
are ordered by start time, then ID. A check that passed before a later file
edit in any linked session is reported as `partial`. Command strings are never
printed. They are mapped to fixed labels such as `pytest`, `npm test`, `ruff`,
or `tsc`.

A run counts only when it has exactly one result recorded after the call, with
an explicit status: an exit code, a success flag, or an error event. A missing,
duplicate, earlier, status-less, or conflicting result makes the outcome
unknown, and validation is `partial`.

The exit code belongs to the whole shell invocation. It is assigned to a check
only when the check is the whole command, such as `pytest -k 'a && b'` or
`CI=1 uv run pytest`. Pipes, `||`, `;`, redirects, background jobs, command
substitution, and multi-line scripts make the outcome unknown. One exception:
in a chain joined only by `&&`, exit 0 means every part ran and passed, so a
pass is kept. A failing `&&` chain is unknown because the failing part is not
known.

Stewardship counts turns and decisions. It does not rank or score people.

## How sessions are linked

| Link | Meaning |
|---|---|
| `explicit` | Session IDs were named on the command line |
| `branch` | The session's recorded git branch matches the PR branch |
| `commit_time` | No branch match. The session closest to the head commit time was used. |
| `latest` | No branch or commit to match. The latest session was used. |

In a pull request job, `commit_time` uses the PR head commit from the event
payload (`pull_request.head.sha` on GitHub, `CI_MERGE_REQUEST_SOURCE_BRANCH_SHA`
on GitLab), not the synthetic merge commit that `actions/checkout` creates.
That commit must be in the clone. Check out the head SHA as in the example
below. If the head SHA is unavailable, the `commit_time` link is skipped and
`latest` is used.

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
  fail_on:                      # a list, a comma list, or none
    - missing
    - partial
  min_confidence: medium        # low, medium, high
  findings: [rationale, validation]
```

`.agent-strace.yml` and `.agent-strace.json` also work. `--fail-on`,
`--min-confidence`, `--findings`, and `--config` override the file.

The config fails closed. An empty or null value, an empty list, an unknown key,
a nested mapping, or any YAML the parser does not support is an error (exit
`2`). Only the literal value `none`, on its own, turns failures off.

Exit codes: `0` pass, `1` threshold or cognitive debt check failed, `2` bad
config or unknown session.

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
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - uses: Siddhant-K-code/agent-trace@main   # pin a release tag
        with:
          run-eval: "false"
          review: "true"
          cognitive-debt: report                 # false, report, or check
          post-pr-comment: "true"
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

The step posts or updates one PR comment with the session summary and the
reviewability table. It always writes the table to the step summary. It exits
non-zero per the threshold, so it works as a required status check. Fork PRs
skip the comment because their tokens are read-only, but still run the check.

The Action reads sessions from `trace-dir` (default `.agent-traces`). Commit
the sessions, or restore them from an artifact, before this step.

### Artifacts

With `review: "true"`, the Action takes the first linked session and writes:

| File | Artifact | Made by |
|---|---|---|
| `evidence-packet.html` | `agent-trace-pr-evidence` | `share SESSION --output FILE`. Metadata only. Prompt and tool content are never included. |
| `cognitive-debt.json` | `agent-trace-pr-evidence` | `cognitive-debt --session SESSION --format json` |
| `cognitive-debt-badge.json` | `agent-trace-pr-evidence` | `cognitive-debt --session SESSION --format badge` |
| `review.json`, `review.md` | `agent-trace-pr-review` | `review --format json` and `--format markdown` |

The review table links the evidence packet only when the upload returned a URL.
Otherwise it says the file is local and not uploaded, or not generated. A
failed packet or upload does not fail the step. The step outputs give each
path, the review result, and the cognitive debt status and result. See
[commands.md](commands.md#github-actions).

The existing `agent-traces` artifact still uploads the whole trace directory.
It can hold prompt and tool content if your traces do.

### Cognitive debt

`cognitive-debt` reads `schema_version` and `status` from the JSON report. It
never reads the badge color.

| Status | Result | With `check` |
|---|---|---|
| `clean`, `partial` | pass | passes |
| `above_threshold` | fail | fails |
| `unknown`, missing, or unsupported report | unavailable | fails |

`unknown` means review coverage could not be measured. It is never a pass. With
`report`, the result is shown and does not change the exit code.

## Sample comment section

Two linked sessions. The second edits a file after the first ran its checks.

```markdown
### Reviewability

| Finding | Status | Confidence | Evidence | Source |
|---|---|---|---|---|
| Rationale | present | high | 3 prompts captured in 2 sessions. | user\_prompt events \(content not shown\) |
| Scope | partial | medium | 1 of 2 written files were named in a prompt or read in the session. Not named or read: `src/extra.py` | file and edit tool events, prompt path mentions, file reads |
| Validation | partial | medium | Last run: pytest passed, exit 0; ruff passed, exit 0. Files changed after the last run of: pytest \(1 file\), ruff \(1 file\). | shell tool calls and linked results inside the session \(not CI\) |
| Stewardship | present | medium | 3 human turns, 1 after the first tool call, 0 approval decisions. Permission modes: default. | user\_prompt events, approval requests, recorded permission modes |
| Evidence health | partial | high | Health: 2 partial \(claude\). Redaction: on. Linked by: branch. | agent-strace health, capture matrix, session link |

**Artifacts**

- Evidence packet: `evidence-packet.html` in the [workflow artifact](https://github.com/OWNER/REPO/actions/runs/123/artifacts/456). Metadata only.
- Cognitive debt: `unknown`, review coverage unknown. Unavailable, not a pass. (report only)

Check: **pass**. Threshold fails on `missing` at `medium` or higher confidence (default).
```

## Limits

- The paper is survey evidence about what reviewers say they want. It does not
  show that attaching this evidence changes review outcomes.
- The check reports captured evidence only. It does not claim complete
  capture. Providers differ; see the [capture matrix](capture-matrix.md).
- Validation is in-session only. It does not read CI results. Many hooks do
  not record exit codes, so a run without one is reported as unknown.
- Scope uses path matching against prompts and reads. A file can be in scope
  without being named, and the check will still report it as not named.
- Cognitive debt measures review coverage, not code quality.
- Output is deterministic. No LLM writes or summarizes any part of it.
