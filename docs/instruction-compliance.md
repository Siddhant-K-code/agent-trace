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
- a should-trigger task whose instruction load is inferred because no file
  read was recorded.

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
| `file_read` | `path` glob | A matching file read after a load |
| `file_write` | `path` glob | A matching file write or edit after a load |
| `tool_call` | `tool` | An exact tool name after a load |
| `test` | none | A recorded test command after a load |

The normal `skill` and `should_trigger` fields remain the skill load
expectation. This avoids a second skill detector or duplicate expectation
schema.

## Evidence states

Every load and expected action uses one evidence state.

| State | Meaning |
|---|---|
| `observed` | A matching trace event or bounded absence was directly recorded |
| `inferred` | The file was available and the client commonly injects repository instructions, but no read event was recorded |
| `missing` | Positive evidence was expected and the bounded trace did not contain it |
| `unavailable` | The file, load anchor, or trace evidence needed to evaluate the claim was unavailable |

An inferred instruction load has `expectation_met: null`. The report does not
turn a client convention into an observed fact.

## Report shape

JSON keeps the existing trigger counts and adds compliance data only for cases
that declare `instructions` or `expected_actions`.

Each run contains:

- `compliance.loads` for the expected skill and instruction files;
- `compliance.follow.skill` for existing skill follow signals;
- `compliance.follow.instruction_actions` for expected observable actions;
- `compliance.validation` for tests, the last linked test result, errors, and
  session completion after the earliest observed load;
- `compliance.evidence_counts` using observed, inferred, missing, and
  unavailable buckets.

Each case and the top-level report contain aggregated load, follow, and
validation evidence counts. No composite percentage, grade, or verdict is
produced.

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
- Inferred instruction loads are not direct observations.
- Observable actions can support an instruction-follow hypothesis, but prose
  style, judgement, and reasoning are not visible in tool traces.
- Post-load validation facts are associated by event order only. They are not
  causal.
- A should-not-trigger case is only evidence for its exact prompt and fixture.
- Availability is read from the benchmark workspace at analysis time.
- Live runs remain non-deterministic. Public fixtures should use
  `--from-traces`.
