# Compliance evidence crosswalk

`agent-strace compliance-report` turns local runtime telemetry into a
privacy-minimized framework evidence crosswalk. It is an evidence-preparation
tool, not legal advice, a compliance determination, an attestation report or
opinion,
or an assessment of control design or effectiveness. Framework applicability,
operator role, system classification, retention sufficiency, and required
deployment documentation remain `not_assessed`.

This top-level command is intentionally separate from the legacy
`agent-strace compliance export` command. It has a different schema,
versioned manifests, conservative vocabulary, stronger snapshot validation,
and no compatibility dependency on the legacy builders.

## Generate a report

```bash
# AICPA Trust Services Criteria evidence crosswalk
agent-strace compliance-report --framework aicpa-tsc --since 90d \
  --output evidence.json

# OWASP Agentic Applications crosswalk as SARIF
agent-strace compliance-report --framework owasp-agentic \
  --since 2026-05-01T00:00:00Z --until 2026-08-01T00:00:00Z \
  --format sarif --output evidence.sarif

# Optional PDF projection
pip install 'agent-strace[pdf]'
agent-strace compliance-report --framework eu-ai-act --since 30d \
  --format pdf --output evidence.pdf
```

Windows are strict, half-open UTC intervals: `[since, until)`. A relative
`--since 90d` is resolved against the exclusive end. Sessions are selected by
event time, so a long-running session contributes only its in-window events.
An eventless session is selected only when its metadata start is in the
window. Correlation never promotes a post-window result into in-window result
coverage; it is labeled as boundary context.

The command snapshots the flat store and every workspace store. Every event
stream is read once with bounded, no-follow file access and strictly parsed
before assessment. Malformed or unreadable trace data fails the complete
report because it cannot safely be classified as outside the window. Active
stores are not transactionally frozen, which is disclosed as a limitation.

## Read the vocabulary

Every mapping has exactly one status:

| Status | Meaning |
|---|---|
| `evidence_observed` | Relevant telemetry was recorded; this does not establish that a control is satisfied |
| `risk_signal` | An explicit recorded signal or labeled structural heuristic warrants review; it is not proof of a violation |
| `gap` | Expected telemetry is missing, ambiguous, outside the window, or only retrospective; it is not a compliance failure |
| `not_assessed` | The trace cannot support the assessment, or the relevant explicit signal was not recorded |

Tool calls are correlated with results and errors using event links. Missing,
orphaned, before-call, duplicate, reused, cross-window, and ambiguous links are
reported in coverage. A recorded result proves only that a result record
exists; it is not called successful without an explicit validated outcome.

Authorization is divided into three evidence types:

- A schema-validated `agent_trace_authorization` v1 record evaluated before
  the call is pre-action evidence.
- A uniquely linked approval sidecar is recorded approval context. Its
  timestamp ordering is reported, but it is not silently treated as a
  pre-action decision.
- `--policy FILE` is a bounded report-time evaluation. Results use
  `would_allow` or `would_deny` and include only the exact policy-byte digest
  and declared schema—not its path, rules, or contents.

No linked approval means authorization is uncovered. Duplicate, conflicting,
or incorrectly ordered records are ambiguous gaps. A later operation is
associated with a recorded denial only when its timestamp is at or after that
denial became effective.

## Privacy and integrity

Session and persisted identity values become deterministic report-local
aliases. Evidence never emits trace paths, resource paths, destinations,
URLs, commands, prompts, raw identifiers, policy rule names, or result/error
contents. Secret and redaction detection contributes counts only. Framework
source and licensing URLs remain visible as public citations.

Integrity is reported without overstating local hash links:

- `empty`: no event records;
- `legacy`: all expected links are absent;
- `partial`: some expected links are absent;
- `broken`: a recorded link does not match, including a nonempty first link;
- `unanchored`: local links are internally consistent but have no independent
  external anchor.

## Framework manifests

Installed manifests are strict JSON with a semantic mapping version and a
SHA-256 digest verified by the packaged index. `--rescan installed` explicitly
reuses that installed manifest; bare `--rescan` does the same offline.
`--rescan FILE` uses one explicit local
manifest after the same strict validation. Neither mode downloads updates or
performs dependency vulnerability, CVE, provenance, or SBOM analysis.

- `aicpa-tsc` is an original evidence crosswalk referencing selected AICPA
  Trust Services Criteria identifiers. Criterion wording is not reproduced.
  It is not an attestation report or opinion. The CLI accepts only
  `aicpa-tsc`; consult the official source and licensing requirements in the
  manifest before using the crosswalk.
- `owasp-agentic` maps ASI01–ASI10 from the OWASP Top 10 for Agentic
  Applications 2026. The adapted manifest is CC-BY-SA-4.0; attribution and
  changes are recorded in `frameworks/THIRD_PARTY_NOTICES.md`.
- `eu-ai-act` cites Regulation (EU) 2024/1689 consolidated as of 2026-07-27
  and Regulation (EU) 2026/1744. Applicability and the dates that apply to a
  particular role or deployment are always unassessed.

## Formats and reproducibility

Authoritative JSON includes two digests. `report_digest` covers the complete
canonical envelope except itself. `evidence_digest` excludes generation time
and covers the manifest digest, resolved window, policy provenance, coverage,
control assessments, and minimal authorization timeline. Re-running identical
evidence with the same window and manifest keeps the evidence digest stable.

SARIF 2.1.0 emits only controlled, location-free messages for risk signals and
gaps. PDF is an optional ReportLab projection and embeds both JSON digests and
the limitations. Characters outside the Base-14 WinAnsi repertoire are
rendered as visible `\uXXXX` or `\UXXXXXXXX` escapes, so text is never silently
dropped or substituted. Always preserve the authoritative JSON alongside a
projection used in another system.
