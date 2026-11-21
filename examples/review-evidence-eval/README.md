# Reviewable Agent Work: offline regression example

Score the delivered result separately from the workflow steps a reviewer can verify.
This example uses hand-authored synthetic JSON, Python 3.10+ and no dependencies.
It does not run agents, execute checks mentioned in the fixtures, or call a model or a network service.
The base scorer reads normalized JSON only. A separate adapter demonstrates one
synthetic Phase A packet, without Inspect.

From the repository root:

```bash
python3 examples/review-evidence-eval/score.py
```

The report gives one outcome and a result for each expected workflow step, including
`unknown` results. There is no combined grade. Exit code `0` means all labels
match the hand-labelled regression fixtures, **not** that all outcomes or steps
passed. Exit code `1` means at least one label changed; `regression_mismatches`
lists every change. Exit code `2` indicates invalid inputs or arguments.

## Fixtures and scoring

The synthetic changelog, in order, is:

1. `FIX-12`: Keep display names on import.
2. `FIX-28`: Preserve blank rows on export.
3. `DOC-04`: Refresh the table examples.

For the summary task, the workflow calls for the agent to load the `release-notes`
skill and review the delivered summary against the changelog. The count-only task
is a near miss: it uses the same changelog, but does not call for the skill.
`cases.json` holds the tasks, exact expected artifacts, workflow requirements,
and independent, hand-labelled regression results for the bundled observations.

`observations.json` is keyed by the same case IDs. Each record supplies:

| Field | Meaning |
|---|---|
| `artifact` | The delivered result projected to the task's JSON object, or `null` when it could not be verified. Object keys and values must exactly match the expected artifact; object key order does not matter, but list order does. A present incorrect artifact fails; `null` is unknown. |
| `observed_steps` | Step IDs supported by positive, task-scoped evidence. `skill:release-notes` denotes agent-initiated use, not just availability or a user request. `review:fix-summary` means the specific review took place, not that the artifact passed review. Unrelated steps do not count. |
| `sufficient_capture_for` | Step IDs for which the relevant scope and time window were captured well enough to infer absence. For any other step, lack of a positive observation is unknown, not absence. An explicit observation still counts in partial capture. |

An observed required step passes, and an observed forbidden step fails. With
sufficient capture and no observation, a required step fails and a forbidden step
passes. With partial capture and no observation, either is unknown. No agent
acceptance claim, generic `PASS` string, tool count, or cost enters the score.

| Synthetic case | Outcome | Skill step | Summary review step |
|---|---|---|---|
| `reviewed-summary` | pass | pass (observed) | pass (observed) |
| `wrong-summary-green-lint` | fail (wrong fix ID) | pass (observed) | fail (absent; unrelated lint passed) |
| `summary-missing-skill` | pass | fail (absent) | pass (observed) |
| `summary-partial-capture` | pass | unknown | pass (observed) |
| `count-wrong-activation` | pass | fail (observed) | not scored |
| `count-clean-negative` | pass | pass (absent) | not scored |
| `count-partial-negative` | unknown | unknown | not scored |

## Comparing observations

To inspect another hand-authored observation set, pass its path as the only
argument to `score.py`. Use the same case IDs and criteria. Compare the reports
by case, then by outcome and step label; retain explicit transitions to/from
`unknown`. An improvement can also be a mismatch against a deliberately failing
fixture. The exit code is a local snapshot check, not a quality gate.

## One Phase A packet, scored offline

`packet-fixture/packet.json` is a byte-for-byte synthetic output of the Phase A
`review-evidence` CLI, for `demo/greetings`, PR 7, task `friendlier-label`. Its
`unit` receipt is current and reports `PASS`; the `inspect` step is observed
with a locator in a partial session summary. The local adapter checks the raw
SHA-256, complete target, `inspect` row identity, and both row locators against
the separately supplied `binding.json`. It treats locators as identities and
does not open packet-referenced paths. The binding is a trusted local input,
not a signature or a substitute for the producer's validation.

```bash
example=examples/review-evidence-eval
fixture=$example/packet-fixture
python3 -B "$example/score_packet.py" \
  "$fixture/packet.json" "$fixture/binding.json" "$fixture/evidence.json"
```

The JSON report labels the outcome and inspection `pass` and release-notes
activation `unknown`, with no regression mismatches. Pass `missing-evidence.json`
instead of `evidence.json` to see all three labels stay `unknown` (exit `1`,
because two expected labels changed). A bad binding exits `2` without a score.

- `packet-cases.json` holds the independently authored expected artifact,
  workflow requirements and labels. Neither the adapter nor the candidate
  evidence receives them.
- `evidence.json` is a separately admitted synthetic record bound to the packet
  digest. The candidate greeting came from replaying the synthetic change on a
  supplied `greet.py` baseline and inspecting the resulting AST without
  executing it. This does not verify a real Git head. The candidate is not
  derived from the receipt, criterion coverage, or packet completeness.
- The positive inspection sidecar is a hand-authored synthetic witness. It
  supplies a typed `greeting-inspection`, a source ID, and the exact packet
  observation it corroborates. This exercises the adapter; it does not prove
  an actual inspection or any skill activation. The adapter never maps `inspect`
  to a skill.
- An inspection can be scored absent only if its bound row is `UNOBSERVED`
  and the independent, step-specific record explicitly says `absent` with
  `capture: "sufficient"`. Missing bindings, gaps, global capture declarations,
  and `UNOBSERVED` on its own cannot establish absence. The bundled packet
  uses the positive path; a separate in-memory test exercises the absence path.

## Two-case reviewer exercise

Build the plain and packet-assisted views with Python 3.10+ and local Git:

```bash
python3 -B examples/review-evidence-eval/build_reviewer_exercise.py \
  --producer /path/to/phase-a-checkout \
  --output /path/to/new-exercise-directory
```

The builder extracts the Phase A implementation at
`423601ceb6e290b31e8bb290aac67dc5c041313e` from the supplied checkout's local
Git objects and runs its actual CLI in a temporary directory. It does not
switch branches, write to the producer checkout, or run the proposed changes
or checks. The output directory must be new and its parent must exist.

For each case, `baseline/` and `packet-assisted/` have the same `worksheet.md`
and byte-identical `evidence/`. Only the second view adds the producer's
`packet/` (start at `packet/review.md`). The script checks both arms and the
packet's retained sources for byte parity. It prints an assembly report and
keeps a copy at `assembly.json` outside the review arms. Exit `0` from the
builder means both views were assembled, not that either change was accepted.

When trying a case, open only its assigned arm and record a decision and
rationale in the worksheet. If manually measuring effort, keep assembly and
review effort separate. Check the
[reference answers](reviewer-exercise/reference.md) afterward; they are not
included in either arm. This is a synthetic exercise, not a reviewer study
or a measurement of usefulness or time savings.

To include the pinned producer smoke in the example's tests, set
`REVIEW_EVIDENCE_PRODUCER` to the same checkout and run
`PYTHONPATH=src python3 -B -m unittest tests.test_review_evidence_eval_example`.
