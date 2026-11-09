# Maintainer release runbook

The core package and VS Code extension have independent release lines.

| Line | Tag | Version sources |
|---|---|---|
| Core | `vX.Y.Z` | Python package, Copilot plugin, and plugin marketplace metadata |
| VS Code extension | `vscode-vX.Y.Z` | Extension package and lock metadata |

Do not make one line wait for, inherit, or publish the other line.

## Required credentials and configuration

| Target | Requirement |
|---|---|
| GitHub core release | The `Release` workflow needs `contents: write` and `actions: write` for `GITHUB_TOKEN`. |
| PyPI | The `pypi` GitHub environment must be a trusted publisher for the `agent-strace` project. No long-lived PyPI token is used. |
| MCP Registry | None. The `mcp-registry` job uses GitHub OIDC (`id-token: write`) for the `io.github.Siddhant-K-code` namespace. |
| VS Code Marketplace | None. A Marketplace Trusted Publishing policy for publisher `Siddhant-K-code` trusts the `vscode-publish` environment of `publish-vscode.yml`. See [extension trusted publishing setup](#extension-trusted-publishing-setup). |
| Open VSX | None. An Open VSX trusted publisher registration for `Siddhant-K-code.agent-strace` trusts the same workflow and environment. See [extension trusted publishing setup](#extension-trusted-publishing-setup). |

Protect production credentials with environment approvals or repository secret
access controls where appropriate.

## Core releases

### Version parity

Set the same `X.Y.Z` value in:

- `src/agent_trace/__init__.py`
- `plugin.json`
- `.github/plugin/marketplace.json` at `metadata.version`
- `.github/plugin/marketplace.json` at `plugins[0].version`
- `plugins/agent-strace/plugin.json`
- `.claude-plugin/plugin.json`
- `.claude-plugin/marketplace.json` at `plugins[0].version`
- `server.json` at the top level and at `packages[0].version`

The core tag and GitHub release use `vX.Y.Z`. PyPI reports `X.Y.Z`.
The generated Gemini extension reads the same Python `__version__` value and is
checked during release validation.

### Release order

1. Update all core version sources and release notes in one pull request.
2. Run the Python suite.
3. Run `python scripts/check_capture_matrix_freshness.py`. A stale capture
   matrix entry does not block the release: re-verify it first, or release
   knowing that the entry describes an older provider release. See
   [capture matrix freshness](capture-matrix.md#freshness).
4. Merge the release pull request to `main`.
5. The `Release` workflow checks core parity, creates `vX.Y.Z`, and creates the
   GitHub release at the triggering commit. Core changelog generation ignores
   extension tags.
6. The same workflow dispatches `Publish to PyPI` at the exact core tag. This
   explicit dispatch is required because a release created by `GITHUB_TOKEN`
   does not start a `release: published` workflow.
7. Confirm the GitHub release includes the interoperability schema and Agent
   Evidence Corpus ZIP.
8. After the PyPI upload, the `mcp-registry` job publishes `server.json` to
   the MCP Registry. See [MCP Registry publishing](mcp.md#mcp-registry-publishing).
9. Confirm the GitHub release, PyPI, and the MCP Registry all report the
   intended version.

`Publish to PyPI` also retains its release trigger for core releases created
outside the automated workflow and a manual dispatch with a required core tag.
Extension release tags are explicitly ignored.

### Core recovery

- If parity validation fails, correct every core version source and merge the
  fix. The release workflow retries because no release exists yet.
- If GitHub release creation fails, rerun the failed `Release` workflow job.
- If PyPI dispatch or publishing fails, fix the environment or trusted
  publisher configuration, then manually run `Publish to PyPI` with the
  existing `vX.Y.Z` tag.
- PyPI recovery is duplicate-safe because the publisher checks for an existing
  distribution.
- If only the `mcp-registry` job fails, rerun that job. It skips a version the
  Registry already lists. Manual fallback from the repo root at the release
  tag: `mcp-publisher login github` then `mcp-publisher publish`.

## VS Code extension releases

### Version parity

Set the same `X.Y.Z` value in:

- `vscode-extension/package.json`
- `vscode-extension/package-lock.json` at the top level
- `vscode-extension/package-lock.json` at `packages[""].version`

The extension tag uses `vscode-vX.Y.Z`. VS Code Marketplace and Open VSX report
`X.Y.Z`. Extension versions do not need to match core versions.

### Release order

1. Update only the extension version sources in one pull request.
2. Run `cd vscode-extension && npm ci && npm test`.
3. Merge the extension release pull request to `main`.
4. Create `vscode-vX.Y.Z` at the merge commit and push the tag.
5. The tag starts `Publish VS Code Extension`. Its `build` job checks out that
   exact tag, verifies extension parity, tests, and packages one VSIX. Its
   `publish` job, the only job with `id-token: write`, runs in the
   `vscode-publish` environment and publishes that VSIX to both registries
   with OIDC trusted publishing.
6. Create or update the GitHub release for the same extension tag.
7. Confirm VS Code Marketplace and Open VSX both report `X.Y.Z`.

The extension publisher also retains manual dispatch with a required
`vscode-vX.Y.Z` tag. Core `vX.Y.Z` tags and core releases never start extension
publishing.

### Extension recovery

- If the tag or parity check fails, do not move an existing published tag.
  Correct the release commit before creating a new tag.
- If publishing fails before either registry accepts the version, fix the
  credential or registry configuration and rerun the workflow with the same
  extension tag.
- If one registry succeeds and the other fails, rerun with the same tag. Both
  publishers skip an already published version and retry the missing one.
- If automatic tag dispatch does not start, manually run
  `Publish VS Code Extension` with the existing `vscode-vX.Y.Z` tag.

- `Marketplace OIDC token exchange failed` or `No trusted publisher matches the
  presented token` means a registry policy does not match the workflow. Check
  the owner, repository, workflow filename, and environment name against the
  values below, then rerun with the same tag.

Never rebuild from a branch after a tag exists. Recovery runs must use the
original tag so each registry receives artifacts from the same commit.

### Extension trusted publishing setup

Both registries publish with short-lived credentials exchanged from the GitHub
Actions OIDC token. No Azure subscription, Microsoft Entra app registration,
managed identity, or repository variable is needed: Marketplace Trusted
Publishing is the Microsoft-documented path for GitHub Actions. The Entra
workload identity path (`vsce publish --azure-credential` with a service
connection) is documented for Azure Pipelines.

Global Azure DevOps personal access tokens, the only PAT type that can publish
to the Marketplace, stop working on January 27, 2027. The VS Code publishing
guide still states December 1, 2026, so finish this setup before December 1,
2026.

One-time setup, done once by the publisher owner before the next
`vscode-vX.Y.Z` tag:

1. GitHub environment. In repository **Settings > Environments**, create
   `vscode-publish`. Under **Deployment branches and tags**, choose
   **Selected branches and tags** and add the tag rule `vscode-v*` and the
   branch rule `main` (manual dispatch runs from `main`). Optionally add
   required reviewers.
2. VS Code Marketplace. Sign in to
   <https://marketplace.visualstudio.com/manage>, open publisher
   `Siddhant-K-code`, select **Trusted Publishing**, expand **Create**, and
   enter:

   | Field | Value |
   |---|---|
   | Policy name | `publish-vscode` |
   | Repository owner | `Siddhant-K-code` |
   | Repository | `agent-trace` |
   | Workflow | `publish-vscode.yml` |
   | Environment | `vscode-publish` |

   The resulting OIDC token has audience `marketplace.visualstudio.com`,
   repository `Siddhant-K-code/agent-trace`, workflow
   `.github/workflows/publish-vscode.yml`, and environment `vscode-publish`.
3. Open VSX. Sign in to <https://open-vsx.org> as the owner of the verified
   `Siddhant-K-code` namespace, open **Settings > Trusted Publishers**, pick
   the namespace and extension `agent-strace`, choose **GitHub Actions**, and
   enter owner `Siddhant-K-code`, repository `agent-trace`, workflow
   `publish-vscode.yml`, and environment `vscode-publish`. The extension must
   keep at least one active version.
4. After the first extension release publishes to both registries through the
   workflow, delete the repository secrets `VSCE_PAT` and `OVSX_PAT`, then
   revoke the matching tokens in Azure DevOps (**User settings > Personal
   access tokens**) and Open VSX (**Settings > Access Tokens**). An `OVSX_PAT`
   value takes precedence over trusted publishing in `ovsx`, so it must not be
   reintroduced in the workflow.

Sources:

- [VS Code: Secure automated publishing to Visual Studio Marketplace](https://code.visualstudio.com/api/working-with-extensions/publishing-extension#secure-automated-publishing-to-visual-studio-marketplace)
- [Azure DevOps: Publish with OIDC Trusted Publishing](https://learn.microsoft.com/azure/devops/extend/publish/overview#publish-with-oidc-trusted-publishing)
- [Azure DevOps Blog: Retirement of Global Personal Access Tokens](https://devblogs.microsoft.com/devops/retirement-of-global-personal-access-tokens-in-azure-devops/)
- [Microsoft Learn: Use personal access tokens](https://learn.microsoft.com/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate)
- [Open VSX wiki: Trusted Publishing](https://github.com/eclipse-openvsx/openvsx/wiki/Trusted-Publishing)
