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
| VS Code Marketplace | Add a repository secret named `VSCE_PAT` with publish permission for publisher `Siddhant-K-code`. |
| Open VSX | Create the `Siddhant-K-code` namespace once, then add a repository secret named `OVSX_PAT` with publish access. |

Protect production credentials with environment approvals or repository secret
access controls where appropriate.

## Core releases

### Version parity

Set the same `X.Y.Z` value in:

- `src/agent_trace/__init__.py`
- `plugin.json`
- `.github/plugin/marketplace.json` at `metadata.version`
- `.github/plugin/marketplace.json` at `plugins[0].version`

The core tag and GitHub release use `vX.Y.Z`. PyPI reports `X.Y.Z`.
Add other core-coupled metadata to this parity check when its publishing
lifecycle is integrated.

### Release order

1. Update all core version sources and release notes in one pull request.
2. Run the Python suite.
3. Merge the release pull request to `main`.
4. The `Release` workflow checks core parity, creates `vX.Y.Z`, and creates the
   GitHub release at the triggering commit. Core changelog generation ignores
   extension tags.
5. The same workflow dispatches `Publish to PyPI` at the exact core tag. This
   explicit dispatch is required because a release created by `GITHUB_TOKEN`
   does not start a `release: published` workflow.
6. Confirm the GitHub release and PyPI both report the intended version.

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
5. The tag starts `Publish VS Code Extension`, which checks out that exact tag,
   verifies extension parity, packages one VSIX, and publishes it to both
   registries.
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

Never rebuild from a branch after a tag exists. Recovery runs must use the
original tag so each registry receives artifacts from the same commit.
