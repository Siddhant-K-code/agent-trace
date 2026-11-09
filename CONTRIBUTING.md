# Contributing

## Before you start

- Search existing issues and pull requests.
- Open an issue before a large behavior or architecture change.
- Read the relevant ADRs. Runtime code in `src/agent_trace/` must use only the
  Python standard library.
- Keep storage changes backward compatible.

## Development setup

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

VS Code extension changes also require:

```bash
cd vscode-extension
npm ci
npm test
```

## Pull requests

- Keep changes focused and include tests for behavior changes.
- Update user-facing documentation when commands, configuration, or release
  behavior changes.
- Do not include secrets, generated trace data, or unrelated formatting.
- Confirm the full Python suite passes. Run the extension suite when its files
  or publishing workflow change.

Maintainers handle version changes and releases using
[the release runbook](docs/releases.md).
