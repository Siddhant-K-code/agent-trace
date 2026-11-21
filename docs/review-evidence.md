# Experimental offline PR evidence assembly

`agent-strace review-evidence BUNDLE --output DIR` assembles one local packet
from files an operator has already retained and reviewed. It does not run Git,
checks, or models; read a trace store; send telemetry; or contact GitHub. This
prototype accepts a minimized session summary, not a raw session log.
The separate [assignment ZIP converter](#convert-a-retained-assignment-zip)
can produce that summary from one native retained format. Neither command
replaces `review`, `share`, or CI.

## Retained input (`review-evidence-input/v0`)

The manifest is a UTF-8 JSON object with `schema`, `target`, `sources`, and
optional `association`. `target` has `repo` (`owner/name`), positive integer
`pr`, `task` (ID), and lowercase full Git `base` and `head` object IDs (40 or
64 hex characters, same length, distinct). Object IDs are compared as strings,
not resolved against a repository.

Each source entry has:

| Field | Meaning |
|---|---|
| `id` | Stable, unique ID in this bundle. IDs use 1–64 lowercase ASCII letters, digits, `.`, `_` or `-`, starting with a letter or digit. |
| `path` | Unique, portable, case-insensitively compared path relative to the manifest directory. Use `/` separators, with no traversal, drive paths, symlinks or device names. |
| `sha256` | Lowercase hex SHA-256 of the exact file bytes. |
| `type`, `version` | One of the source types below, with integer `version: 0`. |
| `origin`, `admission` | Attributed origin text and operator-declared `SYNTHETIC` or `REVIEWED_NONSENSITIVE`. |
| `lines` | `[first, last]`, inclusive and 1-based; validated against actual LF-separated lines. It is a locator, not a filter: the entire file is validated and copied. |

The manifest and JSON sources are each capped at 128 KiB, the diff at 4 MiB,
and up to 64 source files at 8 MiB combined. Structured records reject unknown
fields and duplicate keys. No symlink, multiply linked input file, or nonregular
file is admitted. Use exclusively controlled local directories and supply their
real paths (e.g. not the macOS `/var` alias). Only list synthetic or previously
reviewed nonsensitive files. Admission is an operator declaration, not a
privacy scan or redaction pass.

| Source `type` | Supported retained record |
|---|---|
| `unified-diff` | Exactly one UTF-8 unified text diff. Git file, index, and mode headers with text hunks are supported; binary, combined, rename-metadata and metadata-only diffs are not. The binding to `target` is declared by inclusion, not verified against Git. |
| `intent` | Exactly one `review-evidence-intent/v0` JSON object: `schema`, matching `target`, `by`, `summary`, `criteria` (nonempty `[{id,text}]`), optional `expected_steps` (`[{id,text,required}]`, where `required` is a boolean). `by` attributes intent, criteria, and expected steps. |
| `validation-receipt` | Zero or more `review-evidence-receipt/v0` objects: `schema`, `target`, `check_id`, `result` (`PASS` or `FAIL`), `reported_by`, `assessment: {by,relevance,criterion_ids}`. A `RELEVANT` assessment must map to one or more listed criteria, without duplicates; `IRRELEVANT` and `UNASSESSED` must not map any. At most one receipt per check ID and head. |
| `session-summary` | At most one `review-evidence-session-summary/v0` object: `schema`, matching `target`, `session_id`, `reported_by`, `capture: {completeness,limits}`, `observations: [{step_id,note}]`, optional `attempt_id`. Capture is `COMPLETE`, `PARTIAL`, or `UNKNOWN`; `limits` is an array of text notes. Observations need unique step IDs and may include steps not listed in intent. |
| `attempt-inventory` | At most one `review-evidence-attempt-inventory/v0` object: `schema`, matching `target`, `by`, `completeness` (`COMPLETE`, `PARTIAL`, or `UNKNOWN`), and a nonempty `attempts` array. Each row has a unique `id`, optional `reported_outcome` (`PASS`, `FAIL`, or `UNKNOWN`), and optional `billing`. A `COMPLETE` roster must include the declared attempt, if any. |

The optional manifest `association` has `by`, `session_id`, and optional
`attempt_id`. If a summary is present, the association is required and its
identifiers (including whether an attempt was supplied) must match the summary.
Matching only corroborates the identifiers within this bundle; it does not
establish session identity, authorship, or what caused the change.

Billing and reported attempt outcomes are attributed to the inventory author;
an attempt's outcome is not inferred from revision-bound validation receipts.
`billing: {"completeness":"COMPLETE","currency":"USD","amount_minor":25}`
represents a reported per-attempt total: currency is a three-letter uppercase
code and amount is a nonnegative integer in its minor units. For `PARTIAL` or
`UNKNOWN`, omit both amount and currency. The packet lists every supplied
attempt under `attempt_inventory.attempts` and gives a separate
`attempt_inventory.aggregate_cost`. The aggregate is `UNKNOWN` unless the
inventory is declared complete and every attempt has a complete total in the
same currency within the supported integer range. The assembler does not
estimate, convert, or rank costs, and never substitutes zero for unknown.

## Reading the result

Use `packet.json` (`review-evidence-packet/v0`) as the machine interface. It is
published last: an interrupted output directory without it is not a packet,
even if `review.md` exists. Choose a fresh output directory for a retry.
Neither sources nor existing outputs are overwritten.

* `integrity: VALID` means schema, locators, cross-record consistency, and
  hashes checked out locally. Invalid input emits no packet. It does not
  establish authenticity or correctness.
* `coverage: COMPLETE` means each criterion has a relevant current receipt and
  each required step has an observation. It does not require passing results or
  exhaustive capture. A missing observation means `UNOBSERVED`, never proven
  skipped. A receipt with a matching repo/PR/task/base but another compatible
  head is retained as `STALE`, not used for current coverage.
* `observed_validation` summarizes **all current** reported receipts, including
  irrelevant ones: `PASS`, `FAIL`, `MIXED`, or `UNKNOWN` if no current receipt.
  Check `coverage` and the per-check assessment separately. An irrelevant
  passing check can yield `INCONCLUSIVE / PASS`; a relevant failure can yield
  `COMPLETE / FAIL` and exit `0`.

`criteria` rows link to current and stale receipts by `check_source_ids` and
`stale_check_source_ids`. `checks` rows show the `reported_result`, attributed
`assessment`, `binding`, and `coverage_eligible`. `steps` rows distinguish the
expectation's `source` from its nullable `observation_source`. `gaps` gives a
`code`, `subject_id`, and `sources` per finding. A superseded stale receipt is
still listed as `STALE_CHECK` even if current evidence covers that criterion.
Resolve any locator's `source_id` in `sources`, open its relative
`retained_path`, and check `sha256`. Then use the locator's `lines` or JSON
`pointer` (empty string means the whole JSON record). `review.md` presents the
same facts and locators without active links from source text.

## Convert a retained assignment ZIP

The only supported native format is `agent-strace-assignment-bundle/v1`, the
minimized ZIP produced by [`share --assignment`](hiring.md). The converter
validates it with the same bounded loader as `score`: at most 24 MiB on disk,
18 MiB expanded, and 50,000 minimized events. Bundle and binding JSON are each
limited to 128 KiB. The converter does not scan a trace directory, extract ZIP
members to disk, open the embedded viewer, or send telemetry. Plain
JSON/NDJSON exports, standard `share` HTML, and raw session logs are not accepted.

Before conversion, give the existing review-evidence bundle an `association`.
In a separate local `binding.json`, put exactly `archive_sha256` (lowercase
SHA-256 of the specific retained ZIP), `target`, and `association`. The latter
two objects must match the bundle's declarations, including any attempt ID and
attribution. Record the binding when choosing the archive for the review, not
by accepting an arbitrary replacement archive's new hash. The ZIP deliberately
omits original session IDs and has no PR or revision binding. Checking the hash
and the declarations only confirms which bytes the operator associated with
the change; it does not verify identity, authorship, or causality.

```sh
agent-strace review-evidence-import-assignment assignment.zip input/bundle.json --binding binding.json --output input/converted
```

The output is `input/converted/session-summary.json`, in the existing
`review-evidence-session-summary/v0` format. Review it, then explicitly add it
to `input/bundle.json` as a `session-summary` source (relative `path`
`converted/session-summary.json`, `version: 0`, exact SHA-256 and line range,
and an appropriate admission declaration). The converter does not change the
manifest. Do not list the ZIP itself as a packet source.

The summary reports `PARTIAL` because the minimized export cannot establish
named-step coverage; this does not mean the original recording was incomplete.
It carries no observations, raw payloads, check outcomes, or billing receipts.
An anonymous tool event, exit code, or estimated cost is not evidence of a
named step, validation result, or actual billing. A required step without an
observation stays `UNOBSERVED`, even when an independent current receipt
reports `PASS`. The [offline converter test](../tests/test_review_evidence_assignment.py)
demonstrates the round trip with the real exporter and a synthetic trace.

## Reproducible local walkthrough

From the checkout root, assemble the [synthetic greeting bundle](../examples/review-evidence/greeting/).
The fixture contains only the retained input; no trace store, test helpers,
or installed `agent-strace` is needed.

```sh
scratch="$(mktemp -d)"
scratch="$(cd "$scratch" && pwd -P)"
PYTHONPATH=src python -m agent_trace.entry review-evidence \
  examples/review-evidence/greeting/bundle.json --output "$scratch/packet"
python -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "$scratch/packet/packet.json"
cat "$scratch/packet/review.md"
```

Assembly emits `VALID / COMPLETE / PASS` and exits `0`. The SHA-256 of the
exact `packet.json` bytes is
`92c1e896d9ab6b38831709867bdf87f8c5623fa432b73d8e95e0423039ec3019`.
For incomplete coverage, irrelevant passes, cheaper failed attempts, and
stale or missing evidence, run
`PYTHONPATH=src python -m unittest tests.test_review_evidence -v`.
