# Assignment example

Create a synthetic submission and score it with the example rubric:

```bash
python examples/hiring/create_example_submission.py /tmp/example-submission.zip
agent-strace score /tmp/example-submission.zip --rubric examples/hiring/rubric.yaml
```

The generator uses only synthetic events. Real submissions should be created
from the intended recorded session with `agent-strace share --assignment`.

`example-submission/` is the checked-in, extracted golden equivalent of the
synthetic ZIP, included so its six text members are reviewable in source
control. Do not score that directory directly: regenerate the canonical ZIP
with the command above. The test suite repacks and validates the golden members
to preserve assignment-v1 compatibility.
