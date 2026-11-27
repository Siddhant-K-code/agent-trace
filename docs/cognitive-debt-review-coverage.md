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

The PR Evidence Action can generate both files for the linked review session,
upload them with the minimized Evidence Packet, and treat `unknown` as
unavailable rather than passing. See [PR reviewability](review.md#cognitive-debt).

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

## Agent-written lines

A write is a `file_write` event or a tool call that `watch` treats as an edit
(see [Paths and edits](commands.md#watch)). An MCP tool whose name ends in
`apply_patch`, `edit_file`, `file_edit`, `search_replace`,
`str_replace_editor`, `str_replace_based_edit_tool`, `write_file` or
`write_to_file`, such as `mcp__filesystem__write_file`, is also a write. These
are not counted:

- a todo tool such as Claude Code `TodoWrite` or Gemini CLI `write_todos`;
- a `str_replace_editor` or `str_replace_based_edit_tool` call with `command`
  `view`;
- a call with `dryRun: true`;
- a call whose linked result is an `error` event, or has `is_error: true` or
  `success: false`;
- a shell command, so a file written by `echo x > f` or `sed -i` is missed.

Each write counts the lines it adds:

| Recorded change | Lines counted | Used by |
|---|---|---|
| Whole new text in `content`, `file_text`, `new_source`, `code_edit` or `text` | Every line | Claude Code `Write` and `NotebookEdit`, Copilot CLI `create`, Gemini CLI `write_file` |
| Old and new text in `old_string`/`new_string`, `old_str`/`new_str`, `old_text`/`new_text` or `oldText`/`newText` | Lines a line diff adds or changes; every new line when the old text is empty | Claude Code `Edit`, Copilot CLI `edit`, Gemini CLI `replace`, `str_replace_editor` |
| An `edits` list of old and new pairs | The sum over the pairs | Claude Code `MultiEdit`, Cursor `afterFileEdit` |
| `expected_replacements` above 1 | The pair's lines times that number | Gemini CLI `replace` |
| `replacement` with no old and new pair | Every line | |
| `apply_patch` text, as the arguments or in `command`, `input` or `patch` | `+` lines per file; a `*** Delete File:` section counts nothing; `*** Move to:` counts under the new name | Codex, Copilot CLI `apply_patch` |
| A unified diff in `diff` or `patch` | `+` lines, without `+++` headers | |
| `delete_file`, or `NotebookEdit` with `edit_mode: delete` | Nothing | |
| No content recorded | 1 line | Imported Codex `apply_patch`, older Cursor traces |

Every write that changes a file counts at least one line, so a deletion-only
edit counts 1. Writes to the same file are added together. A write that names
no file counts as unknown authorship, which makes the report `unknown`.

## Review evidence

Each file gets the weight of its strongest signal. Its reviewed lines are its
agent-written lines times that weight, rounded. When both local git and GitHub
have evidence, the larger count is used.

Local git reads `git log --since=<session start> -- <file>` on the session's
branch: `refs/heads/<branch>`, then `refs/remotes/origin/<branch>`, then
`HEAD`. Commits on other branches are not read.

| Local signal | Weight | Meaning |
|---|---|---|
| `not_committed` | 0 | No commit on the branch touched the file after the session started |
| `committed_after_session` | 0 | A commit touched the file. Committing is not review |
| `reviewed_by_trailer` | 0.75 | A commit body has a `Reviewed-by:` line naming another person |
| `git_unavailable` | unknown | The directory is not a git work tree |
| `not_in_checkout` | unknown | The file is not in the checkout read instead of the recorded directory |

A `Reviewed-by:` value does not count when it has the commit author's email,
or, without an email, the author's name. It also does not count when it
contains `[bot]`, `copilot`, `claude`, `anthropic`, `openai`, `codex`,
`cursor`, `gemini` or `devin`. Words such as "review", "comment" or "line",
`Co-authored-by:` trailers, several commit authors, "pull request" and `(#N)`
in a message are not review evidence.

With `--github-token`, the first merged pull request whose head is the
session's branch is read:

| GitHub signal | Weight | Meaning |
|---|---|---|
| `github_merged_pr` | 0 | A merged pull request exists. Merging is not review |
| `github_reviews` | 0.75 | Another person submitted a review |
| `github_review_comments` | 0.85 | Another person left a review comment on the pull request |
| `github_line_comments` | 1.0 | Another person left a review comment on this file |

Reviews and comments by the pull request author, by bots (`type: Bot` or a
login ending in `[bot]`) and pending reviews do not count. Open pull requests
are not read. A GitHub signal makes the file's review state known, so a merged
pull request with no other person's review counts as unreviewed.

## Reading a trace from another checkout

Git runs in the session's recorded `working_dir`. When that directory no
longer exists, as in CI or another clone, the command uses the git work tree
of the current directory if one of its remotes names the same repository as
the session's recorded `git_repo`. Remotes match on host, owner and name,
ignoring case, the protocol and `.git`. Recorded paths under the old directory
are read at the same place under the work tree's top level, so the recorded
directory must have been the repository root. A file outside the recorded
directory, or missing from the checkout, is `not_in_checkout` and unknown. If
no remote matches, or the session has no `git_repo`, review state stays
unknown. The report still shows the recorded paths.

In GitHub Actions, check out with `fetch-depth: 0` so `git log` can see the
branch history.

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
  patch attribution model. They are exact only when the payload holds the new
  text. Claude Code `replace_all` edits count once, a `NotebookEdit` replace
  counts the whole new cell, and lines that a later write removes still count.
- A write the user denied still counts when the provider records no result for
  it.
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
