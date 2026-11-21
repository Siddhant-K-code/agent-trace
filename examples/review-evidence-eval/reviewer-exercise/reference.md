# Reviewer exercise reference

Check this after making a decision. These are independently authored reference
concerns, not producer verdicts or the only defensible decisions.

| Case | Example decision | Key evidence and remaining uncertainty |
|---|---|---|
| Greeting | Accept for this synthetic exercise | `change.input` changes `Hi` to `Hello`; `unit.input` reports a relevant pass at the target head; `summary.input` reports the required inspection. Capture is partial, and the report does not independently reproduce the check or authenticate the inspection. |
| Footer | Request evidence | The diff proposes `Draft` to `Ready`, but `format.input` reports only a formatting pass and explicitly does not support the footer criterion. `summary.input` has partial capture and no observation for the required preview. Ask for a current revision-bound footer-content check and evidence of the preview; do not infer that the preview was skipped or that the change failed. |

For a real assessment, define the decision rubric and effort measurement in
advance, use representative cases rather than only these examples, and
counterbalance case/view order so one reviewer does not learn the same answer
from the other view. Obtain authorization and consent before collecting any
human data. Assembly success, coverage, and reported validation are not human
decisions or evidence of time savings, product usefulness, or ROI.
