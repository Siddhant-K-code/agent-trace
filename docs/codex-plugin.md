# OpenAI Codex plugin

The `agent-strace` Codex plugin records OpenAI Codex sessions through lifecycle
hooks and adds the shared AgentTrace skill. It lives in
`plugins/agent-strace-codex/`, apart from the Claude Code and Copilot CLI
packages, so each agent loads only hooks written for it. The Python CLI remains
a separate prerequisite because the hooks run `agent-strace` from `PATH`.

## Install

Install the zero-dependency CLI, and check that `agent-strace --version` works
in the shell you start Codex from:

```bash
uv tool install agent-strace
# or: pip install agent-strace
```

Add this repository as a Codex marketplace, then install the plugin:

```bash
codex plugin marketplace add Siddhant-K-code/agent-trace
codex plugin add agent-strace@agent-trace
```

Codex reads `.agents/plugins/marketplace.json`, which lists only
`plugins/agent-strace-codex`. The plugin is installed for your user under
`$CODEX_HOME` (default `~/.codex`) and runs in every project.

The third-party [Codex Marketplace](https://www.codex-marketplace.com/) CLI,
which is not affiliated with OpenAI, can also install it. Run it from a
project directory:

```bash
npx codex-marketplace add Siddhant-K-code/agent-trace/plugins/agent-strace-codex --plugin --project
```

It copies the package to `plugins/agent-strace/` in the project and lists it
in the project's `.agents/plugins/marketplace.json`. It also enables the plugin
in `$CODEX_HOME/config.toml`, so Codex loads it in every directory, not only
in that project. Pass the full `plugins/agent-strace-codex` path: the
repository root is not a Codex plugin. Use one install path, not both: two
installs register the same hooks twice.

### Trust the hooks

Codex runs plugin hooks only after you review and trust them. Start `codex`,
open `/hooks`, and trust the nine AgentTrace commands. In the Codex app, use
Settings > Hooks. Until then the plugin records nothing, and `codex exec` skips
untrusted hooks without a warning. Codex 0.164.0 and later tie this trust to
every file in the plugin. Each agent-strace release changes the plugin's
version, so trust all AgentTrace hooks again after every update.

For vetted automation, `codex exec --dangerously-bypass-hook-trust` runs the
hooks without the review.

Hooks run in the session's working directory, so traces go to `.agent-traces/`
in the directory where you started Codex, or to `AGENT_TRACE_DIR` when set.

### Develop locally

From a clone of this repository:

```bash
pip install -e .
codex plugin marketplace add "$PWD"
codex plugin add agent-strace@agent-trace
```

`codex plugin marketplace upgrade` works only for a Git marketplace. After you
change the checkout, run `codex plugin add agent-strace@agent-trace` again to
refresh the installed copy.

## Package contents

| File | Purpose |
|---|---|
| `.codex-plugin/plugin.json` | Manifest and marketplace listing |
| `hooks/hooks.json` | The hooks that `agent-strace setup --cli codex` writes by default |
| `skills/agent-strace/SKILL.md` | The shared AgentTrace skill |

The hooks are:

| Codex event | Command |
|---|---|
| `SessionStart` | `agent-strace hook --provider codex session-start` |
| `UserPromptSubmit` | `agent-strace hook --provider codex user-prompt` |
| `PreToolUse` | `agent-strace hook --provider codex pre-tool` |
| `PostToolUse` | `agent-strace hook --provider codex post-tool` |
| `PostModelRequest` | `agent-strace hook --provider codex post-model-request` |
| `SubagentStart` | `agent-strace hook --provider codex subagent-start` |
| `SubagentStop` | `agent-strace hook --provider codex subagent-stop` |
| `Stop` | `agent-strace hook --provider codex stop` |
| `SessionEnd` | `agent-strace hook --provider codex session-end` (3 second timeout) |

Codex ignores events it does not know. `PostModelRequest` needs Codex 0.164.0
or later, and Codex 0.143.0 also ignores `SessionEnd`. See the
[capture matrix](capture-matrix.md#openai-codex) for what each event records.

The plugin does not register the AgentTrace MCP server. Plugin MCP servers
start in the plugin directory, not your project, so they would read the wrong
`.agent-traces/`. To add the server yourself, see [MCP server](mcp.md).

## Verify

```bash
codex plugin list
agent-strace --version
```

`codex plugin list` should show `agent-strace@agent-trace` as installed and
enabled, with a source ending in `plugins/agent-strace-codex`. It does not list
a project install made with `npx codex-marketplace`. In `/hooks`, the
AgentTrace commands show the plugin as their source. Submit a prompt that
uses a tool, then inspect the trace:

```bash
agent-strace list
agent-strace inspect <session-id>
```

`agent-strace setup --cli codex --verify` also checks the plugin. It finds the
enabled AgentTrace plugin in `$CODEX_HOME/config.toml`, then checks the hooks
file of the copy Codex loads from `$CODEX_HOME/plugins/cache/`. When several
versions are cached, that is `local` if present, else the newest. The output
names the plugin and the file it checked. The `[features]` hooks flag and the
`agent-strace` executable are checked as for manual hooks.

If the plugin is missing hooks, the check fails and says how to update the
plugin. If `$CODEX_HOME/hooks.json` also has AgentTrace hooks, the check warns
that every event is recorded twice. The warning alone does not fail the check.
See [Duplicate hook avoidance](#duplicate-hook-avoidance).

## Update

Update the CLI and the marketplace together:

```bash
uv tool upgrade agent-strace
# or: python -m pip install --upgrade agent-strace
codex plugin marketplace upgrade agent-trace
```

The marketplace upgrade also refreshes the installed plugin. For a project
install, run the same `npx codex-marketplace add` command again. Then trust
the AgentTrace hooks again in `/hooks` and start a new Codex session. Until you
do, the plugin records nothing.

### Moving from the repository-root install

Before this package existed, `codex plugin marketplace add
Siddhant-K-code/agent-trace` read `.claude-plugin/marketplace.json` and
installed the whole repository as a Codex plugin, with Claude Code hooks that
do not pass `--provider codex`. Run `codex plugin marketplace upgrade
agent-trace` to switch that install to this package, then trust the new hooks
in `/hooks`.

## Uninstall

```bash
codex plugin remove agent-strace@agent-trace
```

Remove the marketplace only when no other installed plugin needs it:

```bash
codex plugin marketplace remove agent-trace
```

Remove a project install made with `npx codex-marketplace` from the project
directory:

```bash
npx codex-marketplace remove agent-strace --plugin --project
```

That deletes `plugins/agent-strace/` and its marketplace entry from the
project, and the plugin from `$CODEX_HOME`.

Uninstalling the plugin stops future capture. Existing `.agent-traces/` data is
not deleted.

## Duplicate hook avoidance

Codex runs every hook it finds, so a command that both this plugin and
`agent-strace setup --cli codex` register runs twice and records every event
twice. Use one installation path.

`agent-strace setup --cli codex` and `--cli all` look for an enabled AgentTrace
plugin in `$CODEX_HOME/config.toml`. When they find one, they do not write
`$CODEX_HOME/hooks.json` (default `~/.codex/hooks.json`), and they print why.
A project install made with `npx codex-marketplace` counts too, because it
enables the plugin in `$CODEX_HOME/config.toml`.

If you ran setup before you installed the plugin, `$CODEX_HOME/hooks.json`
still has AgentTrace hooks. Setup and `--verify` then print what to run to
keep only the plugin:

- When the file has only AgentTrace hooks, and setup saved a
  `hooks.json.agent-strace.bak` without them, they print the `cp` command that
  restores that backup.
- When the file has only AgentTrace hooks and there is no such backup, they
  print an `rm` command.
- When the file has other hooks or a `description` too, they tell you to delete
  the entries that run `agent-strace hook`.

Setup does not check project `.codex/hooks.json` files. Remove the AgentTrace
entries from them yourself. Their commands start with
`agent-strace hook --provider codex`.

To switch to manual hooks, remove the plugin first, then run
`agent-strace setup --cli codex`. Setup prints the exact `codex plugin remove`
command. Manual setup is also the way to use `--redact` or `--no-redact`.

## Privacy boundary

The hooks receive the prompts, tool inputs, tool outputs, model request
metadata, and session metadata that Codex exposes for the configured events.
AgentTrace writes them to `.agent-traces/` in the session's working directory
by default, or to `AGENT_TRACE_DIR` when set. Secret redaction is enabled by
default. Do not set `AGENT_TRACE_NO_REDACT=1` for untrusted or shared
workspaces.

The plugin adds no collector by itself. If `AGENT_STRACE_ENDPOINT` is set, the
CLI sends trace events to that configured collector. Anonymous product
telemetry is a separate, default-on path with an allowlisted aggregate schema
that excludes trace contents, prompts, arguments, paths, and session IDs. It
sends nothing until its one-time notice has been shown or telemetry was
explicitly enabled. The SessionStart hook shows that notice once, as a
`warning:` line at the start of an interactive `codex` session. `codex exec`
never shows it; see [When sending starts](telemetry.md#when-sending-starts).
Disable it with:

```bash
agent-strace telemetry disable
```

See [Security and privacy](security.md) and
[Product telemetry](telemetry.md) for the full controls.

## Known provider gaps

- The plugin does not bundle Python or the `agent-strace` executable.
- The plugin records nothing until its hooks are trusted, and each update
  needs that trust again. See [Trust the hooks](#trust-the-hooks).
- Capture is limited to hook-visible data. See the
  [capture matrix](capture-matrix.md#openai-codex).
- OpenAI's plugin directory does not accept plugins with lifecycle hooks.
  Install from this repository's marketplace or from codex-marketplace.com.

## Release and publication

`.codex-plugin/plugin.json` carries the package version, and the release
workflow checks it against `src/agent_trace/__init__.py` with the other
manifests. See the [release runbook](releases.md#version-parity).

`hooks/hooks.json` must equal the default output of
`agent-strace setup --cli codex`, and `skills/agent-strace/SKILL.md` must equal
the root `skills/agent-strace/SKILL.md`. After changing either, run:

```bash
python -m unittest tests.test_codex_plugin -v
```

To list the plugin on codex-marketplace.com, submit it as a plugin with the
path `Siddhant-K-code/agent-trace/plugins/agent-strace-codex`. The scanner
rejects artifacts with more than 128 files, so keep the package small, and
keep binary files such as images out of it.
