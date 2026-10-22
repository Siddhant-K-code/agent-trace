# GitHub Copilot CLI plugin

The `agent-strace` Copilot CLI plugin records lifecycle events and adds a trace
analyst agent and skill for investigating sessions.

## Install

Install the zero-dependency Python CLI first so the plugin hooks can invoke it:

```bash
uv tool install agent-strace
# or: pip install agent-strace
```

Add this repository as a marketplace, then install its plugin:

```bash
copilot plugin marketplace add Siddhant-K-code/agent-trace
copilot plugin install agent-strace@agent-trace
```

Restart Copilot CLI after installation. Do not also run
`agent-strace setup --cli copilot`; both approaches register the same hooks and
would record duplicate events.

## Verify

```bash
copilot plugin list
agent-strace --version
```

In an interactive Copilot CLI session, use `/agent` to select
`agent-strace:trace-analyst` or run `/skills list` to confirm the `agent-strace`
skill loaded. New sessions are written to `.agent-traces/` in the working
directory.

## Use

Ask Copilot to analyze the latest agent trace, or run commands directly:

```bash
agent-strace list
agent-strace replay
agent-strace explain
agent-strace timeline
agent-strace lint
```

The hooks capture session starts and ends, prompts, successful and failed tool
calls, agent-stop events, and subagent starts and stops exposed by Copilot CLI.
Subagents started with the `task` tool are linked to their parent session, so
`agent-strace tree` shows them. Secret redaction remains enabled by default.

Plugins installed before the subagent hooks were added do not register them.
Upgrade agent-strace, run `copilot plugin marketplace update agent-trace` and
`copilot plugin update agent-strace`, then restart Copilot CLI. An older
agent-strace rejects the new hook names with "Unknown hook event"; in a test
`copilot -p` run, Copilot CLI 1.0.103-0 ignored the failed hooks and finished
normally. Sessions
captured before the update stay unlinked.

## Develop locally

From a clone of this repository:

```bash
pip install -e .
copilot plugin marketplace add "$PWD"
copilot plugin install agent-strace@agent-trace
```

Copilot caches installed plugin components. Run
`copilot plugin marketplace update agent-trace` followed by
`copilot plugin update agent-strace` after changing the manifest, hooks, agent,
or skill.

## Uninstall

```bash
copilot plugin uninstall agent-strace
```

Locally installed plugins run only in Copilot CLI. Copilot cloud agent does not
load plugins installed on a developer machine.
