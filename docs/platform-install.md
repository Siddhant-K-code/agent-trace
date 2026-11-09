# Platform-team install

This guide is for a platform or IT team that wants AgentTrace capture on every
developer machine, installed once by an administrator, without per-developer
setup. It covers GitHub Copilot CLI policy hooks and Claude Code managed
settings. The default is metadata-only capture with secret redaction on.

For a single developer, use [Setup](setup.md) instead.

## What the vendors provide

### GitHub Copilot CLI: policy hooks

Source: [GitHub Copilot hooks reference, Policy hooks](https://docs.github.com/en/copilot/reference/hooks-configuration#policy-hooks).

- Policy hooks are machine-wide hooks for Copilot CLI only. They are not
  supported in Copilot in the cloud.
- Locations:
  - Linux and macOS: `/etc/github-copilot/policy.d/*.json`, loaded in
    alphabetical order.
  - Windows: `C:\ProgramData\GitHub\Copilot\policy.d\*.json`.
  - Windows Registry: subkeys under `HKLM\Software\Policies\GitHub\Copilot`,
    each with a `Policy` REG_SZ value that holds a JSON policy document.
- The format is the same as user and repository hook files:
  `{ "version": 1, "hooks": { ... } }`.
- On POSIX systems, policy files must be owned by root and must not be group-
  or world-writable.
- The reference says policy hooks "require elevated privileges to install. End
  users cannot modify them."
- Policy hooks load before all other hooks, "cannot be disabled by
  `disableAllHooks`", and run regardless of folder trust. A repository
  `settings.json` with `disableAllHooks` skips every other hook source, and
  "Policy hooks are not affected".
- Policy hooks always run on the host, even when the session sandbox is on.
- Hooks from all sources (policy, user, project, plugins) are combined, and
  every entry runs. Policy hooks do not stop user-level AgentTrace hooks from
  also running.
- Failure behavior: a `preToolUse` command hook that crashes or exits non-zero
  denies the tool call. A hook that times out is skipped (fail-open), and this
  applies to policy hooks too.
- In Copilot CLI a hook entry can use `exec` and `args` to run an executable
  directly, without a shell.

The reference describes no server-delivered organization or enterprise hook
setting. Policy hooks are delivered by putting files or registry values on each
machine, for example with your MDM or configuration management tool. For a
repository-scoped rollout, see
[Using hooks with GitHub Copilot CLI](https://docs.github.com/en/copilot/tutorials/copilot-cli-hooks).

### Claude Code: managed settings

Sources: [Managed settings](https://code.claude.com/docs/en/managed-settings),
[Settings reference](https://code.claude.com/docs/en/settings-reference),
[Hooks reference](https://code.claude.com/docs/en/hooks).

- Managed settings files:
  - macOS: `/Library/Application Support/ClaudeCode/managed-settings.json`
  - Linux and WSL: `/etc/claude-code/managed-settings.json`
  - Windows: `C:\Program Files\ClaudeCode\managed-settings.json`
- A `managed-settings.d/` directory next to `managed-settings.json` holds
  drop-in files. Claude Code merges `managed-settings.json` first, then each
  `*.json` drop-in in alphabetical order. Lists combine, with duplicates
  removed.
- Other managed sources: server-managed settings, a macOS configuration profile
  (`com.anthropic.claudecode`), and the `Settings` value under
  `HKLM\SOFTWARE\Policies\ClaudeCode` on Windows.
- By default (`managedSourcesBehavior: "first-wins"`), Claude Code uses only the
  highest-ranked managed source that delivers a policy key: remote
  (server-managed), then MDM or HKLM, then the files. If your organization
  already delivers server-managed or MDM policy, Claude Code ignores the
  managed settings files. Put the hooks in the source you already use, or set
  `managedSourcesBehavior` to `"merge"`. See
  [How Claude Code combines managed sources](https://code.claude.com/docs/en/managed-settings#how-claude-code-combines-managed-sources).
- `disableAllHooks` in user, project, or local settings "can't disable those
  managed hooks". The settings reference says "Only managed settings can
  disable managed hooks". See
  [Disable or remove hooks](https://code.claude.com/docs/en/hooks#disable-or-remove-hooks).
- `allowManagedHooksOnly: true` in managed settings blocks user, project, local,
  and plugin hooks. Hooks from plugins that managed settings force-enable still
  run. See
  [allowManagedHooksOnly](https://code.claude.com/docs/en/settings-reference#allowmanagedhooksonly).
- `--safe-mode` disables customizations, but "Managed settings policy still
  applies, including policy-configured hooks". See the
  [CLI reference](https://code.claude.com/docs/en/cli-reference).
- If the OS denies reading a managed file, for example a root-only file, the
  session starts without that file's policy.
- A command hook with `args` runs in exec form: the executable is spawned
  directly with no shell. See
  [Exec form and shell form](https://code.claude.com/docs/en/hooks#exec-form-and-shell-form).
- The managed settings page lists what a developer can still change. A
  developer with local administrator rights can edit the managed source, and
  managed settings bind Claude Code only. See
  [What a developer can change](https://code.claude.com/docs/en/managed-settings#what-a-developer-can-change).

## What AgentTrace installs

`agent-strace setup --managed` prints a hooks file and the path to install it
at. It writes nothing. The printed hooks:

- run an absolute `agent-strace` path in exec form (`exec` and `args` for
  Copilot, `command` and `args` for Claude Code), so the user's shell, aliases,
  and `PATH` are not involved;
- pass `--metadata-only` to every hook;
- cover the same events as the per-user setup: 9 Copilot CLI events and 7
  Claude Code events.

### Metadata-only capture

`agent-strace hook --metadata-only` records:

- event types, timestamps, and event IDs;
- session ID, provider, model, start source, permission mode, and stop or end
  reason;
- tool names, tool call IDs, turn IDs, and durations;
- subagent IDs, types, and names, and parent links;
- token counts when the provider reports them;
- the key names of tool input objects (for example `command`), without their
  values.

Every other string value is replaced with `[OMITTED]` before it is written:
prompts, responses, tool input and output, error text, file paths, diffs, the
working directory, and transcript paths. Lists are written empty. Events with
omitted values are marked `"redacted": true`, and every event carries
`"capture": "metadata-only"`.

Secret redaction is always on in this mode. `AGENT_TRACE_NO_REDACT` and
`AGENT_STRACE_NO_REDACT` in the user's environment are ignored, and
`session_start` records `"redaction": "on"`.

The session's `meta.json` still holds the attribution AgentTrace records for
every session: OS user, hostname, process ancestry, git remote URL, branch,
commit, and working directory. Treat traces as identifying data.

## Install

### 1. Install agent-strace in a root-owned location

Policy hooks run the path in the hooks file. If that file disappears or stops
working, Copilot CLI denies every tool call (see
[What this cannot enforce](#what-this-cannot-enforce)). Install a pinned
version at a fixed path that users cannot change:

```bash
sudo python3 -m venv /opt/agent-strace
sudo /opt/agent-strace/bin/pip install 'agent-strace==<version>'
/opt/agent-strace/bin/agent-strace --version
```

On Windows, install into a directory under `C:\Program Files` and use the
`agent-strace.exe` path.

### 2. Print the hooks file

```bash
/opt/agent-strace/bin/agent-strace setup --managed --cli copilot \
  --executable /opt/agent-strace/bin/agent-strace > agent-strace.json

/opt/agent-strace/bin/agent-strace setup --managed --cli claude \
  --executable /opt/agent-strace/bin/agent-strace > agent-strace-claude.json
```

The target path is printed on standard error. Without `--executable`, the
`agent-strace` on `PATH` is used, with a warning if it is inside your home
directory.

### 3. Install the file as root

GitHub Copilot CLI on Linux or macOS:

```bash
sudo install -d -o 0 -g 0 -m 0755 /etc/github-copilot/policy.d
sudo install -o 0 -g 0 -m 0644 agent-strace.json /etc/github-copilot/policy.d/agent-strace.json
```

Claude Code on macOS (on Linux, use `/etc/claude-code/managed-settings.d`):

```bash
sudo install -d -o 0 -g 0 -m 0755 "/Library/Application Support/ClaudeCode/managed-settings.d"
sudo install -o 0 -g 0 -m 0644 agent-strace-claude.json \
  "/Library/Application Support/ClaudeCode/managed-settings.d/agent-strace.json"
```

Mode `0644` matters for Claude Code: a file developers cannot read is skipped.
If you deliver Claude Code policy through server-managed settings or MDM, add
the printed `hooks` object to that policy instead.

On Windows, write the file to `C:\ProgramData\GitHub\Copilot\policy.d\` or
`C:\Program Files\ClaudeCode\managed-settings.d\` from an elevated session, and
make sure standard users cannot write to it.

To send events to a central [collector](server.md) instead of each project's
`.agent-traces/`, add an `env` object to each Copilot policy entry, for example
`"env": {"AGENT_STRACE_ENDPOINT": "https://collector.example.com"}`. Remote
events get the same metadata-only transform and redaction.

### 4. Remove per-user AgentTrace hooks

Managed hooks run in addition to user hooks. A developer's own AgentTrace hooks
would record the same session again, with full content. Remove:

- `~/.copilot/hooks/agent-strace.json` (or under `$COPILOT_HOME`);
- the `agent-strace@agent-trace` Copilot CLI plugin, which bundles hooks;
- AgentTrace hooks in `~/.claude/settings.json` and project settings.

For Claude Code, `allowManagedHooksOnly: true` in managed settings stops
user, project, and plugin hooks from running. It applies to all hooks, not only
AgentTrace, so treat it as a policy decision. Copilot CLI has no equivalent.

## Verify

Run as a developer user, not as root:

```bash
agent-strace setup --managed --cli copilot --verify
agent-strace setup --managed --cli claude --verify
```

The check reads files only and runs the installed executable with `--version`.
It fails (exit 1) when:

- the policy directory or managed files are missing, unreadable, or not valid
  JSON;
- a hooks file or its directory is not owned by root or is group- or
  world-writable (POSIX);
- an event is missing its AgentTrace hook, has more than one, or the hook is not
  the metadata-only exec-form handler that `setup --managed` prints;
- the executable is missing or reports a different version;
- Claude Code managed settings set `disableAllHooks: true`.

It warns, without failing, when user-level AgentTrace hooks also exist, or when
the executable itself is not root-owned. It does not check the Windows
Registry, server-managed settings, MDM profiles, plugins, or repository hooks.

With `--cli claude`, add `--live` to also inspect the newest recorded session
in the current project. See [live capture verification](setup.md#live-capture-verification).

## Detect when capture was disabled

- Run `agent-strace setup --managed --cli <cli> --verify` from your MDM or
  compliance agent on a schedule, and alert on a non-zero exit.
- Check that sessions keep arriving. Each managed session's first event is a
  `session_start` with `"capture": "metadata-only"` and `"redaction": "on"`:

  ```bash
  agent-strace list
  agent-strace export <session-id> --format ndjson | head -1
  agent-strace health <session-id>    # Redaction: on (recorded by the session start hook)
  ```

  A session without `"capture": "metadata-only"` was recorded by a non-managed
  hook. A machine where developers use the agent but no new sessions appear has
  lost capture.
- In Claude Code, `/status` lists the managed sources in use under
  `Setting sources`.

## What this cannot enforce

- **Administrators.** A user with root or local administrator rights can edit or
  delete the policy files. Both vendors say this. Redeploy the files on a
  schedule with your MDM tool.
- **Removing the executable.** If `agent-strace` is removed or broken, Copilot
  CLI denies every tool call, because `preToolUse` fails closed. Capture stops,
  and so does the agent. Claude Code hook failures follow its
  [`onFailure`](https://code.claude.com/docs/en/hooks) rules. The printed hooks
  leave `onFailure` unset.
- **Timeouts.** A Copilot CLI hook that times out is skipped, including a
  policy hook. That event is not recorded.
- **Where traces go.** Hooks run with the user's environment. A user can set
  `AGENT_TRACE_DIR` or `AGENT_STRACE_ENDPOINT` to move traces, unless the policy
  entry sets them. Traces in `.agent-traces/` belong to the user, who can
  delete them.
- **Extra hooks.** User, project, and plugin hooks still run unless Claude Code
  `allowManagedHooksOnly` is set. They can record full content.
- **Other surfaces.** Policy hooks are a Copilot CLI feature. They are not
  supported in Copilot in the cloud, and they do not cover other Copilot
  surfaces or other tools. For Copilot cloud agent, use repository hooks; see
  [Copilot cloud agent capture](copilot-cloud-agent.md). Claude Code managed
  settings bind Claude Code only.
- **Claude Code `--bare`.** Bare mode skips hook auto-discovery. The vendor docs
  do not say whether managed hooks still run in bare mode. Do not rely on them
  for `--bare` runs.
- **Claude Code managed source precedence.** Under the default `first-wins`
  behavior, a file-based install is ignored when server-managed or MDM policy is
  in force.
- **Content outside hooks.** Hooks only see what the vendor sends, so events the
  vendor does not emit are not recorded. Content the vendor does send, such as a
  subagent's final response, is replaced by `[OMITTED]` in metadata-only mode.
- **Not a security boundary.** AgentTrace records what agents did. It does not
  block or approve actions.
