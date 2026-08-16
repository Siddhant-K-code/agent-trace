# Organization reporting

`agent-strace org-report` creates a monthly engineering digest without sending
trace content to a pricing or analytics service. It reports session volume,
covered-session estimated spend, attributed-identity coverage, team and
heuristic task-type breakdowns, and structural anomaly callouts.

## Local shared directory

```bash
agent-strace --trace-dir /srv/agent-traces org-report --month 2026-08
agent-strace --trace-dir /srv/agent-traces org-report \
  --month 2026-08 --team workspace:payments --format html -o august.html
```

The local command reads the flat store and every workspace under the same
trace root. Identical session IDs in different workspaces remain separate
sessions. This filesystem root is the organization boundary. The command does
not read S3 or another object store directly; mount or synchronize that data
into a trace directory, or use the collector API.

## Authenticated collector

```bash
export AGENT_STRACE_AUTH_KEY='ast_...'
agent-strace org-report --month 2026-08 \
  --endpoint https://collector.example.com \
  --ca-file /etc/agent-strace/private-ca.pem \
  --format json -o august.json
```

`--endpoint` is required for remote collector input. The command never implicitly uses
`AGENT_STRACE_ENDPOINT`, so a reporting job cannot silently switch from its
intended local source. Put the bearer key in `AGENT_STRACE_AUTH_KEY` or a
size-bounded `--auth-key-file`; do not place it in command history.

One collector instance and key is the organization boundary. Tenant, team,
and workspace tags are grouping metadata, not authorization. The command
probes the session-list endpoint without credentials and refuses to report if
the collector does not enforce authentication.

HTTPS is the default. Plain HTTP is accepted automatically only for
`localhost` and loopback IP addresses; non-loopback HTTP requires
`--allow-insecure-http`. Collector URLs cannot contain user information,
queries, or fragments. Redirects and ambient HTTP proxies are disabled so the
bearer key is sent only to the explicitly named collector. Responses and JSON
nesting are bounded, and any malformed, truncated, oversized, mixed-session,
or mixed-tenant selected stream fails the complete report.

## Scope and interpretation

The month is a half-open UTC range: `2026-08` includes timestamps from
`2026-08-01T00:00:00Z` up to, but not including, `2026-09-01T00:00:00Z`.
Metadata is filtered by month and team before remote event streams are fetched.

Team selection is exact:

- `--team tag:platform` selects the persisted team tag.
- `--team workspace:platform` selects the workspace.
- `--team platform` works when the name exists in only one domain and fails as
  ambiguous when it exists as both a team tag and a workspace.

Cost values are offline token-size estimates using the selected bundled model
and the pricing snapshot date recorded in the report. Unknown or invalid costs
are excluded from spend and average calculations, and cost coverage shows the
denominator. They are not provider invoices or actual spend.

Attributed identities come only from persisted actor, engineer, user, git
author, OS-user, and hostname fields. The reporting machine's git config is
never consulted. Coverage and confidence counts show how much identity data
was available; they are not roster coverage.

Task types are keyword heuristics. Anomaly callouts compare group averages only
when the subject and at least three other eligible peers meet the minimum
session and cost-coverage floors. Lint signals are structural heuristics. The
report does not infer task completion, productivity, efficiency, or causes.

Remote collector reports first read session metadata and then one event stream per
selected session. The input can change between these N+1 requests, so the
result is not a transactionally consistent snapshot. Local shared directories
can likewise change while read. Every output records this limitation.

## Output and privacy

```bash
agent-strace org-report --month 2026-08 --format text
agent-strace org-report --month 2026-08 --format json --anonymize -o report.json
agent-strace org-report --month 2026-08 --format html --anonymize -o report.html
```

JSON uses the `agent-strace-org-report/v1` schema and rejects non-finite
numbers. HTML is self-contained, contains no external assets, and escapes all
labels. File output is written atomically with private permissions and refuses
symlinked output paths.

Aggregate formats have no standalone session ID, filesystem path, branch,
tenant, hostname, or raw-event fields. Without `--anonymize`, team labels and
anomaly identity labels can appear, including a persisted `user@hostname`
identity. Use `--anonymize` before sharing outside the organization. With it,
complete team/workspace and identity labels become sorted report-local aliases;
aliases are neither stable cross-report identifiers nor hashes of the original
labels.
