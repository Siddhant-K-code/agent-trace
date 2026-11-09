# Maintainer release runbook

This runbook covers coordinated Python, Copilot plugin, VS Code Marketplace,
and Open VSX releases.

## Required credentials and configuration

| Target | Requirement |
|---|---|
| GitHub release | The `Release` workflow needs `contents: write` and `actions: write` for `GITHUB_TOKEN`. |
| PyPI | The `pypi` GitHub environment must be configured as a trusted publisher for the `agent-strace` project. No long-lived PyPI token is used. |
| VS Code Marketplace | Add a repository secret named `VSCE_PAT` with permission to publish for publisher `Siddhant-K-code`. |
| Open VSX | Create the `Siddhant-K-code` namespace once, then add a repository secret named `OVSX_PAT` with publish access. |

Protect production credentials with environment approvals or repository secret
access controls where appropriate.

## Version parity

Before merging a release change, set the same `X.Y.Z` value in every source
listed below. Do not include the `v` prefix in files.

| Surface | Version source |
|---|---|
| Python package | `src/agent_trace/__init__.py` |
| Copilot plugin | `plugin.json` |
| Copilot plugin marketplace | `.github/plugin/marketplace.json` in both `metadata.version` and `plugins[0].version` |
| VS Code extension | `vscode-extension/package.json` |
| Extension lock metadata | `vscode-extension/package-lock.json` at the top level and in `packages[""].version` |

The Git tag and GitHub release use `vX.Y.Z`. PyPI must report `X.Y.Z`.
VS Code Marketplace and Open VSX must also report `X.Y.Z`.

The release workflow verifies source parity before creating a new release. Each
publisher also verifies that its checked-out tag matches its package version.

## Release order

1. Update all version sources and release notes in one pull request.
2. Run the Python suite and the VS Code extension suite.
3. Merge the release pull request to `main`.
4. The `Release` workflow reads the version, checks parity, creates tag
   `vX.Y.Z`, and creates the GitHub release at the triggering commit.
5. The same workflow explicitly dispatches PyPI and extension publishing at
   that tag. This dispatch is required because releases created by
   `GITHUB_TOKEN` do not start `release: published` workflows.
6. Confirm the GitHub release, PyPI package, VS Code Marketplace listing, and
   Open VSX listing all show `X.Y.Z`.

The publish workflows retain `release: published` triggers for releases created
outside the automated workflow. They also retain manual dispatch with a required
tag input.

## Failure recovery

Do not delete or move a published tag to retry.

- If parity validation fails, correct every version source and merge the fix.
  The release workflow will retry on the new `main` commit because no release
  exists yet.
- If GitHub release creation fails, rerun the failed `Release` workflow job.
- If a publisher dispatch fails, manually run `Publish to PyPI` or
  `Publish VS Code Extension` with the existing `vX.Y.Z` tag.
- If a registry publish fails, fix the credential, environment approval, or
  registry configuration, then rerun the same publish workflow with the same
  tag.
- Duplicate publishes are safe. PyPI checks for an existing distribution, and
  both extension publishers use their duplicate-skip option.
- If one extension registry succeeds and the other fails, rerun the extension
  workflow. The successful registry is skipped and the failed registry retries.

Never rebuild from a branch after a tag exists. All recovery runs must use the
original release tag so each registry receives artifacts from the same commit.
