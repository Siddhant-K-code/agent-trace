# Reviewable Agent Work: offline regression example

Score the delivered result separately from the workflow steps a reviewer can verify.
This example uses hand-authored synthetic JSON, Python 3.10+ and no dependencies.
It does not run agents, execute checks mentioned in the fixtures, or call a model or a network service.
It does not read review packets or integrate with Inspect.

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

An eventual Phase A adapter needs only a case ID, the independently checkable
artifact (or a way to retrieve it safely, otherwise `null`), evidence identifying
which named steps occurred in scope, and capture sufficiency for each step where
absence can be established. These are example-local inputs, not assumptions
about packet fields:

- Packet criterion coverage and reported check outcomes do not independently
  establish what the task delivered. Verify the actual artifact separately.
- A packet step marked observed is usable only if its evidence identifies the
  named step in this case. The observed status alone does not identify which
  skill was activated or whether the agent initiated the activation.
- An unobserved step or a gap is not proof of nonactivation. Leave the step out
  of `sufficient_capture_for` unless separate evidence establishes the relevant
  scope was captured despite any redaction, truncation, or missing events.

Expected artifacts and labels still come from the case definitions, not the packet.
