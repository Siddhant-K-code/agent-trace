# AGENTS.md — agent-strace development guide

This file tells AI coding agents how to work with the agent-strace repository.

## Project layout

```
src/agent_trace/    Core library — one module per feature
tests/              One test file per module (test_<module>.py)
ADRs/               Architecture Decision Records — read before adding dependencies
docs/               User-facing documentation (setup, commands, production, integrations, security)
examples/           Usage examples for each integration
pyproject.toml      Package config and optional extras
```

## Constraints

- **Zero runtime dependencies in `src/agent_trace/`.** Use Python stdlib only. If a feature requires a third-party package, it must be an optional extra in `pyproject.toml` under `[project.optional-dependencies]`, imported lazily inside the function with a clear `ImportError` message. See ADR-0003.
- **Never change the `.agent-traces/` storage format** in a way that breaks existing sessions. Adding new fields is fine. Removing or renaming existing fields requires a major version bump and human approval.
- **Every new CLI command must be registered in `cli.py`** following the existing subparser pattern.
- **Every new feature must have tests** in `tests/test_<module>.py`. Run `python -m pytest tests/ -v` to verify.
- **OTLP export uses HTTP/JSON only** — no gRPC. See ADR-0006.

## Development workflow

```bash
# Install in editable mode
pip install -e .

# Run tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_watch.py -v
```

## Adding a new feature

1. Create `src/agent_trace/<feature>.py`
2. Add a `cmd_<feature>` function following the pattern in existing modules
3. Import and register it in `cli.py`
4. Add new `EventType` values to `models.py` if needed
5. Write tests in `tests/test_<feature>.py`
6. Add the command to the appropriate table in `README.md` (one line, linked to docs/)
7. Add full flag reference to `docs/commands.md`
8. If the feature involves production export, update `docs/production.md`
9. If the feature involves a new integration, update `docs/integrations.md`

## Version bumping

Current version: `0.51.1` in `src/agent_trace/__init__.py`.

- New feature (new command, new flag, new integration): bump minor (`0.51.1` → `0.52.0`)
- Bug fix or small improvement: bump patch (`0.51.1` → `0.51.2`)
- Breaking change to CLI or storage format: bump major — check with maintainer first

## docs/ structure

The `docs/` directory contains user-facing documentation. Keep these files current when adding features:

| File | Contents |
|---|---|
| `docs/setup.md` | Full setup for all 3 integration paths, complete JSON configs |
| `docs/commands.md` | Full flag reference for every command |
| `docs/production.md` | Per-backend OTLP setup (Datadog, Honeycomb, Grafana, etc.) |
| `docs/server.md` | Server-side collector setup, Docker, API reference |
| `docs/integrations.md` | Auto-instrumentation for each framework |
| `docs/vscode.md` | VS Code extension docs (setup, commands, settings) |
| `docs/security.md` | Secret redaction, PII anonymization, policy files |

## README policy

`README.md` is a landing page, not documentation. It must stay under 300 lines. Detailed content goes in `docs/`. Do not add detailed flag descriptions, long examples, or configuration dumps to `README.md`.

## ADRs to read before making architectural decisions

- ADR-0003: Zero runtime dependencies
- ADR-0006: OTLP HTTP/JSON only (no gRPC)
- ADR-0001: Flat event stream data model
- ADR-0002: NDJSON file storage (no database)
