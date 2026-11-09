# Claude Code plugin

The native `agent-strace` plugin records Claude Code lifecycle events and adds
the shared AgentTrace skill and trace analyst agent. The Python CLI remains a
separate prerequisite because plugin hooks invoke `agent-strace` from `PATH`.

## Install

Install the zero-dependency CLI:

```bash
uv tool install agent-strace
# or: pip install agent-strace
```

Add this repository as a marketplace, then install the plugin:

```bash
claude plugin marketplace add Siddhant-K-code/agent-trace
claude plugin install agent-strace@agent-trace
```

Start a new Claude Code session or run `/reload-plugins`. Review
`hooks/hooks.json` before installation because plugin hooks run with the current
user's permissions.

For local development without installing from the marketplace:

```bash
pip install -e .
claude --plugin-dir .
```

## Verify

Check the installed components and the CLI:

```bash
claude plugin list
claude plugin details agent-strace
agent-strace --version
```

The component inventory should list the `agent-strace` skill, the shared trace
analyst agent, and seven hook events. Claude Code versions that derive the
display name from the shared Copilot-compatible filename can show the agent as
`trace-analyst.agent`. In a trusted workspace, submit a prompt that uses a tool,
then inspect the local trace:

```bash
agent-strace list
agent-strace inspect <session-id>
```

`agent-strace setup --cli claude --verify` checks hooks stored in Claude Code
settings files. It does not inspect plugin hooks, so it can report missing
manual setup while this plugin is working correctly.

## Update

Update the CLI and the marketplace-installed plugin together:

```bash
uv tool upgrade agent-strace
# or: python -m pip install --upgrade agent-strace
claude plugin marketplace update agent-trace
claude plugin update agent-strace@agent-trace
```

Start a new session or run `/reload-plugins` after the update. Claude Code only
installs a new cached copy when the plugin version changes.

## Uninstall

Uninstall from the same scope used for installation:

```bash
claude plugin uninstall agent-strace@agent-trace --scope user
```

Use `--scope project` or `--scope local` when applicable. Remove the marketplace
only when no other installed plugin needs it:

```bash
claude plugin marketplace remove agent-trace
```

Uninstalling the plugin stops future capture. Existing `.agent-traces/` data is
not deleted.

## Duplicate hook avoidance

`hooks/hooks.json` intentionally matches the default output of
`agent-strace setup --cli claude`. Claude Code runs an identical merged handler
once, so an existing default manual setup does not record the same event twice.

Use one installation path when possible. Manual hooks created with
`--redact` or `--no-redact` have different command strings and can run beside
the plugin hooks. Before enabling the plugin, remove those AgentTrace hook
entries from `~/.claude/settings.json`, `$CLAUDE_CONFIG_DIR/settings.json`, and
project or local settings. The alternative is to uninstall the plugin and keep
the manual hooks.

## Privacy boundary

The hooks receive the prompts, tool inputs, tool outputs, and session metadata
that Claude Code exposes for the configured events. AgentTrace writes them to
`.agent-traces/` in the workspace by default, or to `AGENT_TRACE_DIR` when set.
Secret redaction is enabled by default. Do not set `AGENT_TRACE_NO_REDACT=1` for
untrusted or shared workspaces.

The plugin adds no collector by itself. If `AGENT_STRACE_ENDPOINT` is set, the
CLI sends trace events to that configured collector. Anonymous product
telemetry is a separate, default-on path with an allowlisted aggregate schema
that excludes trace contents, prompts, arguments, paths, and session IDs.
Disable it with:

```bash
agent-strace telemetry disable
```

See [Security and privacy](security.md) and
[Product telemetry](telemetry.md) for the full controls.

## Known provider gaps

- The plugin does not bundle Python or the `agent-strace` executable.
- Claude.ai chat can load the skill but ignores the agent and hooks, so this
  plugin does not capture chat sessions.
- Local plugin installs do not automatically reach Claude Code cloud sessions.
  The remote environment must receive the plugin and have the CLI on `PATH`.
- Capture is limited to hook-visible data. It does not expose provider-side
  model internals.
- The packaged hooks cover `UserPromptSubmit`, tool success and failure,
  `Stop`, `SessionStart`, and `SessionEnd`. They do not register separate
  `SubagentStart` or `SubagentStop` handlers. Tool hooks still run inside
  subagents when Claude Code supplies those events.
- Managed settings can block plugin hooks with `allowManagedHooksOnly` or
  disable hooks through policy.

## Release and publication

Before a release, synchronize the version in:

- `src/agent_trace/__init__.py`
- `plugin.json`
- `.github/plugin/marketplace.json`
- `.claude-plugin/plugin.json`
- `.claude-plugin/marketplace.json`

Then run:

```bash
python -m unittest tests.test_plugin tests.test_claude_plugin -v
claude plugin validate --strict .
```

After the synchronized change reaches the default branch, the repository's
release workflow creates the normal GitHub release and publishes the Python
package. The repository marketplace then serves the new plugin version to
users who run `claude plugin update`.

Publishing to Anthropic's directory remains manual:

1. Push the synchronized release to the public GitHub repository.
2. Run strict local validation against the released commit.
3. Open `https://claude.ai/directory/manage`.
4. Submit or update the repository entry and complete portal validation.
5. Confirm the directory version before announcing the release.

Do not commit portal credentials, API tokens, or marketplace secrets.
