# AgentTrace v1 local review workflow

AgentTrace has a broad command surface. This page defines the **focused local-review
workflow** that the project is stabilizing for v1 so users can tell the core path
from optional or experimental surfaces.

> **v1 is not declared complete by this page.** Some stages below depend on work
> that is still being integrated. Status is explicit so documentation does not
> promise behavior that has not shipped.

## The canonical path

1. **Install locally.**
   ```bash
   uv tool install agent-strace
   # or: pip install agent-strace
   ```

2. **Configure one supported coding-agent capture adapter.**
   ```bash
   agent-strace setup              # Claude Code
   agent-strace setup --cli codex
   agent-strace setup --cli gemini
   agent-strace setup --cli cursor
   agent-strace setup --cli copilot
   ```
   Configure only the adapter you actually use. Secret redaction is enabled by
   default. See [setup](setup.md) for adapter-specific limits.

   **Claude Code requires a manual installation step:** `agent-strace setup`
   prints hook JSON; it does **not** write `~/.claude/settings.json`.
   Copy the generated `hooks` entries into that file, merging them with
   existing hooks instead of replacing unrelated settings. See the
   [Claude Code hook example](setup.md#option-1-cli-hooks-recommended).
   For Codex, back up an existing `~/.codex/hooks.json` before running
   setup because the command overwrites that file.

3. **Verify that capture works before trusting a session.**
   This stage is a **v1 candidate, not yet a stable top-level CLI contract**.
   Provider/capture verification is tracked in #239. Until that lands, verify
   that a real session appears in `agent-strace list` and inspect the recorded
   events instead of treating hook installation alone as proof of coverage.

4. **Record a local session.**
   Hook-based adapters record normal agent activity after setup. For an MCP
   server, use:
   ```bash
   agent-strace record -- <server-command> [args...]
   ```
   The local trace is the evidence source of truth for the rest of this path.

5. **Inspect evidence health.**
   A deterministic evidence-health core is being developed in #240. Until its
   CLI integration is released, do not equate a clean-looking replay with
   complete capture. Provider blind spots and missing boundaries must remain
   visible.

6. **Review the ordered session locally.**
   The stable review entry point today is:
   ```bash
   agent-strace replay <session-id>
   ```
   `timeline`, `explain`, `why`, and `diff` are useful secondary views.
   The dedicated review projection/UI is tracked in #242.

7. **Review disclosure before sharing or exporting.**
   Treat trace data as sensitive. The current safe preview for anonymized export
   is:
   ```bash
   agent-strace export <session-id> --anonymize --dry-run
   ```
   A privacy-minimized share-by-default workflow is tracked in #252. Until that
   is released, do not assume ordinary `share` output is minimized.

8. **Optionally compare or export.**
   Comparison is local:
   ```bash
   agent-strace diff <session-a> <session-b>
   ```
   OTLP export is optional and should only be treated as a supported destination
   after the target backend/path has been verified for the evidence you need.

## Surface status

The labels below describe **v1 product status**, not whether a command exists.

### v1 supported core

| Surface | Job | Privacy / failure boundary |
| --- | --- | --- |
| `setup` | Configure a supported capture adapter | Redaction on by default; installation is not proof of complete capture |
| `record`, `record-http` | Capture MCP traffic locally | Captured prompts/tool data can be sensitive |
| `list` | Find local sessions | Reports stored sessions, not capture completeness |
| `replay` | Review one ordered session | Shows recorded evidence only |
| `export --anonymize --dry-run` | Preview anonymization before export | Preview is the review step; anonymization is not a claim of perfect secret detection |

### v1 candidate review helpers

These are useful local-review surfaces, but they are not the minimum v1
compatibility contract: `inspect`, `explain`, `timeline`, `why`, `diff`,
`compare`, `tree`, `cost`, `compaction`, `annotate`, and `identity`.

### Experimental / advanced surfaces

The following remain available but are outside the focused local-review promise
unless a separate compatibility policy says otherwise:

- live control and policy: `watch`, `policy`, `approval`, `rbac`,
  `audit`, `audit-tools`, `mcp-scan`;
- fleet / organization reporting: `dashboard`, `budget-report`,
  `team-report`, `org-report`, `tenant`, `compliance`,
  `compliance-report`, `audit-readiness`;
- behavioral analytics and optimization: `drift`, `baseline`, `fingerprint`,
  `freeze`, `lint`, `eval`, `cognitive-debt`, `context-score`, `skills`,
  `standup`, `freshness`, `oncall`, `curve`, `inflation`, `optimize`;
- hosted / orchestration surfaces: `server`, `auth`, `workspace`,
  `retention`, `auto`, `a2a-tree`, `pr-comment`, and `score`.

Commands not named above should be treated as **not part of the v1 core
compatibility promise** until they are explicitly classified. This is
deliberately conservative: an existing command is not automatically a stable
v1 contract.

## Compatibility and support policy for the v1 path

- **Python:** the package metadata / PyPI classifiers are authoritative; the
  current documented floor is Python 3.10.
- **Capture adapters:** support means the adapter is named in the setup guide and
  its observable limitations are documented. It does not mean access to hidden
  model reasoning or events a provider never emits.
- **Local-first:** the canonical workflow requires no hosted AgentTrace service.
- **Data contract:** recorded evidence may contain prompts, tool inputs/results,
  commands, paths, and other sensitive content. Redaction and anonymization are
  risk-reduction mechanisms, not a security boundary.
- **Breaking changes:** a v1-supported command should not silently change its
  input/output contract. Breaking contract changes require a documented
  migration/deprecation path and should be exercised in a release candidate.
- **Experimental surfaces:** may change faster and must not be presented as
  production compliance or security guarantees.

## Release gate

Before calling the workflow v1-complete, a clean environment must pass:

`install → configure → verify → capture → health → review → disclosure preview → safe share/export`

against the representative fixture, with provider limitations visible. A v1
release candidate should run that path before v1.0 is cut.
