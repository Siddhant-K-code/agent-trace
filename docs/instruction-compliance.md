# Instruction compliance benchmark

The instruction compliance workflow extends `agent-strace skills eval`. It
combines existing skill load and follow analysis with repository instruction
file evidence. It does not add a top-level command.

The report answers four bounded questions:

1. Were the expected repository instruction and skill files available?
2. Is there observed or inferred evidence that they loaded?
3. Did the trace contain the expected observable actions after a load?
4. What validation facts were recorded after the load?

This is an evidence inventory. It is not an outcome score, a developer score,
or a causal claim.

## Run the public corpus

The synthetic corpus runs offline. It does not call a model provider.

```bash
agent-strace skills eval examples/instruction-compliance/cases.json \
  --from-traces examples/instruction-compliance/traces \
  --workspace examples/instruction-compliance/fixture \
  --format json > instruction-compliance.json

agent-strace skills eval examples/instruction-compliance/cases.json \
  --from-traces examples/instruction-compliance/traces \
  --workspace examples/instruction-compliance/fixture \
  --format markdown > instruction-compliance.md
```

The corpus includes:

- a should-trigger release task with observed instruction and skill loads;
- a should-not-trigger near miss with no skill load;
- a should-trigger task with an availability-only inference because no
  instruction file read was recorded.

The fixture uses fixed events and timestamps so JSON and Markdown artifacts
are deterministic for the same command arguments.

## Case fields

Instruction compliance cases use the normal `skills eval` fields plus two
optional fields.

```json
{
  "id": "draft-release-notes",
  "skill": "release-notes",
  "prompt": "Draft the release notes and validate the repository.",
  "should_trigger": true,
  "fixture_dir": "fixture",
  "instructions": ["AGENTS.md"],
  "expected_actions": [
    {"id": "write-notes", "kind": "file_write", "path": "RELEASE_NOTES.md"},
    {"id": "run-tests", "kind": "test"}
  ]
}
```

`instructions` accepts repository instruction paths already recognized by the
`skills` report:

- `AGENTS.md`
- `CLAUDE.md`
- `GEMINI.md`
- `.github/copilot-instructions.md`

`expected_actions` supports:

| Kind | Required matcher | Meaning |
|---|---|---|
| `file_read` | `path` glob | An event that reads a matching file, after a load. Shell reads count |
| `file_write` | `path` glob | An event that writes or edits a matching file, after a load. Shell commands never count |
| `tool_call` | `tool` | A call to that tool after a load. Names compare case-insensitively |
| `test` | none | A recorded test command after a load |

The normal `skill` and `should_trigger` fields remain the skill load
expectation. This avoids a second skill detector or duplicate expectation
schema.

### Reads and writes

An event reads a file when it is one of these:

- A `file_read` event. It names its file in `path`, `file_path` or `uri`.
- A call to `read`, `read_file`, `open_file`, `view` or `get_file_contents`,
  in any case. Each path argument counts.
- A `str_replace_editor` or `str_replace_based_edit_tool` call whose
  `command` is `view`. Each path argument counts.
- A tool call whose shell command runs `awk`, `bat`, `cat`, `head`, `less`,
  `more`, `nl`, `sed` or `tail`. Each word after the program that is not an
  option counts as a file. `sed -i` edits, so it is not a read. A `cd`
  earlier in the command changes the directory that later files resolve
  against. Only the command itself is parsed: the call's `command` or `cmd`
  argument, else the event's `command` field.

An event writes a file when it is one of these:

- A `file_write` event. It names its file in `path`, `file_path` or `uri`.
- A call to `apply_patch`, `create`, `delete_file`, `edit`, `edit_file`,
  `file_edit`, `insert`, `multiedit`, `notebookedit`, `patch`, `replace`,
  `search_replace`, `str_replace`, `str_replace_based_edit_tool`,
  `str_replace_editor`, `write`, `write_file` or `write_to_file`, in any
  case. Each path argument counts. An `apply_patch` call whose arguments are
  the patch text, as GitHub Copilot CLI stores them, writes each file the
  patch adds, updates, deletes or moves to. A `str_replace_editor` or
  `str_replace_based_edit_tool` call whose `command` is `view` only reads.

Shell commands count as reads but never as writes. A file written by
`echo x > RELEASE_NOTES.md` or `sed -i` does not meet a `file_write`.

The path arguments are `file_path`, `path`, `absolute_path`, `filePath`,
`target_file`, `notebook_path` and `uri`. They are read from `arguments` or
`tool_input`, also when stored as a JSON string.

An instruction counts as loaded when an event reads it by the same rule. The
read must resolve to the instruction path under the session root described
below. A write is not a load.

### Paths

Each session resolves its paths against the first `cwd` its `session_start`
events record, as skill detection does. A session with no recorded cwd uses
the workspace. That is the run's temp workspace in a live run. With
`--from-traces` it is the case `fixture_dir`, else `--workspace`, else the
current directory. Each session in a run directory uses its own cwd. So a
run saved with `--save-traces` and scored later with `--from-traces`
resolves against the temp workspace it ran in, even after that directory is
gone.

A relative path joins that root. A `path` glob is tried on the path
relative to the root, or on the full path for a file outside the root. It
is then tried on the path as stored. `./.github/x.md` matches `.github/*`.
`../src/x.py` is outside the root, so it does not match `src/*`.

Instruction availability still comes from the workspace on disk.

## Evidence states

Every load and expected action uses one evidence state.

| State | Meaning |
|---|---|
| `observed` | A matching trace event or bounded absence was directly recorded |
| `inferred` | The file was available, but no read event established whether it loaded |
| `missing` | Positive evidence was expected and the bounded trace did not contain it |
| `unavailable` | The file, load anchor, or trace evidence needed to evaluate the claim was unavailable |

An inferred instruction entry has `actual: available_load_unobserved` and
`expectation_met: null`. It does not claim that a client loaded or injected
the file.

## Report shape

JSON keeps the existing trigger counts and adds compliance data only for cases
that declare `instructions` or `expected_actions`.

Each run contains:

- `compliance.loads` for the expected skill and instruction files;
- `compliance.follow.skill` for existing skill follow signals;
- `compliance.follow.instruction_actions` for expected observable actions;
- `compliance.validation` for tests, the last linked test result, errors, and
  session completion after the earliest observed load. It also reports
  observed and unavailable result-correlation counts. An observed last test
  result carries `runner_summary`, the runner's own final summary outcome
  read as in [review Validation](review.md#findings);
- `compliance.analysis` and `compliance.warnings` when malformed or unreadable
  session evidence prevents a bounded conclusion;
- `compliance.evidence_counts` using observed, inferred, missing, and
  unavailable buckets.

Each case and the top-level report contain aggregated load, follow, and
validation evidence counts. No composite percentage, grade, or verdict is
produced.

Unreadable or malformed session evidence is never converted into a missing
skill load or follow signal. Affected evidence is `unavailable`, and the
warning names only the session and error type or skipped event count.

Text and Markdown use compact counts in this order:

```text
observed/inferred/missing/unavailable
```

## Reused analysis

The workflow composes existing analysis instead of defining parallel metrics.

- Skill availability, loads, follow signals, and post-load facts come from
  `agent-strace skills`.
- Trigger expectations and offline trace replay come from
  `agent-strace skills eval`.
- Repository context file path matching and instruction violation facts reuse
  `context-score` and its instruction parsing helpers.

Use the surfaces together:

```bash
agent-strace context-score --compare --format json
agent-strace skills --since 30d --format json
agent-strace skills eval cases.json --from-traces runs --format markdown
```

`context-score` compares efficiency proxies across instruction versions.
`skills` reports recorded adoption evidence. The benchmark tests explicit
trigger and follow expectations against controlled fixtures.

## Metric limitations

- The workflow measures recorded evidence, not task quality or correctness.
- Availability-only instruction inferences are not load observations.
- Observable actions can support an instruction-follow hypothesis, but prose
  style, judgement, and reasoning are not visible in tool traces.
- Post-load validation facts are associated by event order only. They are not
  causal. Duplicate, multiple, or result-before-call correlations are reported
  as unavailable, even when a later test has one clean result.
- A should-not-trigger case is only evidence for its exact prompt and fixture.
- Availability is read from the benchmark workspace at analysis time.
- Live runs remain non-deterministic. Public fixtures should use
  `--from-traces`.
