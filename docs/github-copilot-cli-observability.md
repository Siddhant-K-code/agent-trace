# GitHub Copilot CLI observability

AgentTrace records the lifecycle events that GitHub Copilot CLI exposes through
hooks. You can install the Copilot plugin or install only the AgentTrace hooks.
Do not use both at the same time because duplicate hooks can duplicate events.

## Capture limits

Copilot CLI does not expose assistant response text in the recorded stop hook.
It also does not expose model or token usage in the observed hook payloads.
AgentTrace cannot reconstruct those fields or hidden reasoning.

The [capture matrix](capture-matrix.md#github-copilot-cli) names the Copilot CLI
release last exercised and lists the current provider gaps. Other releases can
behave differently.

## Install

Plugin installation adds hooks, a trace analyst agent, and a trace analysis
skill:

```bash
uv tool install agent-strace
copilot plugin marketplace add Siddhant-K-code/agent-trace
copilot plugin install agent-strace@agent-trace
```

For hooks without the plugin:

```bash
agent-strace setup --cli copilot
```

See the [Copilot plugin guide](copilot-plugin.md) for upgrade and uninstall
commands.

## Review captured sessions

```bash
agent-strace list
agent-strace health SESSION_ID
agent-strace replay SESSION_ID
agent-strace tree SESSION_ID
agent-strace stats SESSION_ID --include-subagents
```

Parent `subagentStart` and `subagentStop` events can link a `task` subagent
session to its parent. Sessions captured before those hooks were installed
remain unlinked.

## Known capture gaps

- The first prompt can arrive before the session-start event.
- A tool denied at the permission prompt can remain without a result or error.
- Tool hooks have no call ID, so results are paired by tool name.
- File and command activity is visible only when tool arguments or results
  contain it.
- Local Copilot plugins do not run in Copilot cloud agent.

Run `health` to see both observed trace defects and declared provider
limitations. A partial trace can still be reviewable, but it is not complete
evidence.

For reviewer-facing inspection, the shipped
[session evidence review](session-review-projection.md) is available in the
local dashboard and [VS Code extension](vscode.md#evidence-review).

## Related documentation

- [Copilot CLI plugin](copilot-plugin.md)
- [Copilot setup](setup.md#github-copilot-cli-hooks)
- [Capture matrix](capture-matrix.md#github-copilot-cli)
- [Evidence health](evidence-health.md)
- [VS Code evidence review](vscode.md#evidence-review)
