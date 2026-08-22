---
name: agent-strace
description: Trace and investigate GitHub Copilot CLI sessions with agent-strace. Use when the user wants to capture a Copilot session, replay agent activity, explain a decision, diagnose repeated tool calls or failures, inspect cost, or audit a recorded session.
---

# agent-strace

Use the local `agent-strace` executable. The plugin contributes lifecycle hooks,
but the Python package must also be installed and available on `PATH`.

## Prerequisite check

Run `agent-strace --version`. If it is unavailable, stop and provide one of:

```bash
uv tool install agent-strace
pip install agent-strace
```

Do not silently substitute another tracing tool.

## Capture

Installed plugin hooks automatically record new Copilot CLI sessions to
`.agent-traces/`. Do not also run `agent-strace setup --cli copilot`; that would
register duplicate hooks.

## Investigate

Choose the smallest command set that answers the request:

```bash
agent-strace list
agent-strace inspect <session-id>
agent-strace replay <session-id>
agent-strace explain <session-id>
agent-strace timeline <session-id>
agent-strace why <session-id> <event-number>
agent-strace lint <session-id>
agent-strace audit <session-id>
agent-strace cost <session-id>
agent-strace diff <session-a> <session-b>
agent-strace compare <session-a> <session-b>
```

Resolve an omitted session ID with `agent-strace list`. Prefer text output for
interactive analysis and JSON only when structured processing is necessary.
Report exact session IDs and event numbers for material findings.

## Safety

- Treat trace storage as read-only unless the user requests retention, import,
  anonymization, or another mutating operation.
- Preserve secret redaction.
- Do not claim causal intent from proximity alone; use `why` or clearly label the
  conclusion as an inference.
