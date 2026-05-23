# AGENTS.md — agent-strace development guide

This file tells AI coding agents how to work with the agent-strace repository.

## Project layout

```
src/agent_trace/    Core library — one module per feature
tests/              One test file per module (test_<module>.py)
ADRs/               Architecture Decision Records — read before adding dependencies
docs/               Integration guides
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
6. Update README.md with the new command and an example

## Version bumping

- New feature (new CLI command, new integration, new flag): bump minor (`0.38.1` → `0.39.0`)
- Bug fix or small improvement: bump patch (`0.38.1` → `0.38.2`)
- Breaking change to CLI or storage format: bump major — check with maintainer first

Version is in `src/agent_trace/__init__.py`.

## ADRs to read before making architectural decisions

- ADR-0003: Zero runtime dependencies
- ADR-0006: OTLP HTTP/JSON only (no gRPC)
- ADR-0001: Flat event stream data model
- ADR-0002: NDJSON file storage (no database)
