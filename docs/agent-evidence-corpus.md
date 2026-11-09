# Agent Evidence Corpus

The Agent Evidence Corpus is a versioned set of synthetic AgentTrace sessions under [`src/agent_trace/corpus/v1/`](../src/agent_trace/corpus/v1/). The corpus ships in Python wheels and in the interoperability archive attached to GitHub releases.

Run every case with:

```bash
agent-strace validate --corpus
```

The manifest records the exact expected status and labels for each case. Validation fails if any result changes without a matching manifest update.

## Cases

| Case | Purpose |
|---|---|
| `complete-current` | Complete current trace with valid byte-level hash links |
| `incomplete-missing-outcome` | Finalized tool call with no terminal result |
| `terminal-absent` | Parseable trace with no final session boundary |
| `fixture-review-001` | Representative redacted failure, retry, recovery, human correction, and deliberate orphan gap |
| `tampered` | Valid JSON with a broken stored hash link |
| `summary-contradiction` | Metadata count contradicts observed events |
| `usage-missing` | Usage summary is absent and therefore unknown |
| `usage-zero` | Usage summary is explicitly zero |
| `duplicate-event-id` | Two records share one event ID |
| `out-of-order` | Event timestamps regress |
| `invalid-schema` | Event `data` is not an object |
| `replay-drift` | Tool-call behavior differs from the existing regression baseline |

## Perturbation policy

Equivalence-preserving perturbations do not change the interpreted event graph:

- JSON indentation or object key order, after rebuilding byte-level hash links
- Omission of optional empty fields already omitted by the canonical serializers
- Legacy omission of event `session_id` when metadata supplies the same identity
- Missing usage summary versus another missing usage summary

Meaning-changing perturbations change evidence or its interpretation:

- Event type, timestamp order, event ID, parent ID, or session identity changes
- Tool name or tool input changes used by the regression projection
- Added, removed, duplicated, or reordered terminal outcomes
- Added or removed session boundaries
- A metadata summary that conflicts with observed events
- Explicit zero usage changed to missing usage, or missing usage changed to explicit zero
- Payload changes without rebuilding stored hash links

Serialization equivalence and hash equivalence are different. Byte-level hash links intentionally detect reserialization. Replay equivalence uses the existing tool-call regression projection and ignores unrelated JSON formatting.

## Privacy and license

All cases are hand-authored synthetic data. They contain no private traces, customer data, repository secrets, credentials, personal identifiers, or copied model conversations. Strings that resemble paths, commands, prompts, and results are minimal synthetic examples.

The corpus is distributed under the repository MIT license. It is suitable for public test suites, interoperability checks, research prototypes, and release verification. It is not a benchmark of model quality or evidence of provider coverage.

## Refresh procedure

1. Decide whether the change is equivalence-preserving or meaning-changing.
2. Keep case directory names, session IDs, event IDs, and timestamps stable unless the rule under test requires changing them.
3. Edit `meta.json` and `events.ndjson` using the published v1 schemas.
4. Rebuild every affected `prev_hash` from the exact previous NDJSON line. Do not normalize the file after computing hashes.
5. Update a replay baseline with the existing `agent-strace freeze` fixture shape when tool-call behavior is intentional.
6. Run `agent-strace validate <case-dir> --format json`.
7. Update only that case's expected status and sorted labels in `manifest.json`.
8. Run `agent-strace validate --corpus`.
9. Run `python -m unittest tests.test_trace_validation tests.test_representative_fixture -v`.
10. Build the wheel and confirm the schema and corpus files are present.

Do not refresh expected labels only to make a failing test pass. A label change must follow a documented rule change, an intentional fixture change, or a schema version change.
