# Setup

Three ways to capture agent sessions. Pick the one that matches your agent.

---

## Option 1: CLI hooks (recommended)

Captures the lifecycle events exposed by each CLI: user prompts, assistant responses, and hook-visible tool calls or edits. Claude Code exposes broad tool coverage; Cursor coverage depends on the native events Cursor emits.

```bash
# Generate Claude Code hooks config
agent-strace setup
# Check the hooks you added to Claude Code settings (read-only; no trace is created)
agent-strace setup --cli claude --verify

# Install OpenAI Codex user-level hooks
agent-strace setup --cli codex
# Check installed Codex config and executable (read-only; no trace is created)
agent-strace setup --cli codex --verify

# Install Gemini CLI extension hooks
agent-strace setup --cli gemini

# Install Cursor project hooks
agent-strace setup --cli cursor

# Install GitHub Copilot CLI user-level hooks
agent-strace setup --cli copilot

# Configure all supported hook integrations
agent-strace setup --cli all

# For all projects (global config)
agent-strace setup --global

# Trusted local traces only: disable secret redaction
agent-strace setup --no-redact
```

`agent-strace setup` prints the hooks JSON and the file to add it to: `~/.claude/settings.json` (user-level, applies to all projects), or `$CLAUDE_CONFIG_DIR/settings.json` when that variable is set:

```json
{
  "hooks": {
    "UserPromptSubmit": [{ "hooks": [{ "type": "command", "command": "agent-strace hook user-prompt" }] }],
    "PreToolUse": [{ "matcher": "", "hooks": [{ "type": "command", "command": "agent-strace hook pre-tool" }] }],
    "PostToolUse": [{ "matcher": "", "hooks": [{ "type": "command", "command": "agent-strace hook post-tool" }] }],
    "PostToolUseFailure": [{ "matcher": "", "hooks": [{ "type": "command", "command": "agent-strace hook post-tool-failure" }] }],
    "Stop": [{ "hooks": [{ "type": "command", "command": "agent-strace hook stop" }] }],
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "agent-strace hook session-start" }] }],
    "SessionEnd": [{ "hooks": [{ "type": "command", "command": "agent-strace hook session-end" }] }]
  }
}
```

Then use Claude Code normally. Sessions appear in `.agent-traces/`.

To keep traces somewhere else, set `AGENT_TRACE_DIR` (a relative path resolves against the working directory). Hooks write there, and `list`, `replay`, `health` and the other commands read from there unless you pass `--trace-dir`. An empty value is treated as unset. See [global options](commands.md#global-options).

Traces can contain prompts and tool output, so `.agent-traces/` writes its own `.gitignore` and stays out of git by default. To commit a trace deliberately, use `git add -f`.

```bash
agent-strace list     # list sessions
agent-strace replay   # replay the latest
agent-strace explain  # plain-English summary
```

### Claude Code hook verification

`agent-strace setup --cli claude --verify` checks the Claude Code hooks without
changing any file. It reads the user settings file:
`$CLAUDE_CONFIG_DIR/settings.json` when that variable is set in your shell,
otherwise `~/.claude/settings.json`. As in Claude Code, a set value is used as
is: `~` is not expanded, a relative path starts from the current directory, and
an empty value means the current directory. That file must be valid JSON and
contain each of the 7 generated hooks exactly once, as a `command` handler. The
`PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `SessionStart` and
`SessionEnd` hooks need a match-all matcher (`""`, `"*"` or omitted).
Unrelated hooks and extra fields such as `timeout` are allowed. Stale
AgentTrace hooks, hooks under the wrong event, an `if` or `args` field on an
AgentTrace hook, and mixed redaction modes fail. Any consistent redaction mode
passes, whatever flags you pass to `--verify`. The check also runs
`agent-strace --version` from `PATH` and fails if it is missing or reports
another version.

It also reads the other settings layers that can stop these hooks:

- `.claude/settings.json` and `.claude/settings.local.json` in the current
  directory. Run the check from the directory where you start Claude Code.
- Managed settings: `managed-settings.json` and the `managed-settings.d/*.json`
  drop-ins in `/Library/Application Support/ClaudeCode` (macOS),
  `/etc/claude-code` (Linux and WSL) or `C:\Program Files\ClaudeCode`
  (Windows).

The check fails when:

- `disableAllHooks` is `true` in the highest layer that sets it (managed, then
  local, then project, then user). An invalid managed value is ignored, as
  Claude Code does. In other files it must be `true` or `false`.
- `disableAllHooks` is `true` in user settings and only a project or local file
  sets it back to `false`. Other projects then run no hooks.
- Managed settings set `allowManagedHooksOnly` to anything but `false`. Claude
  Code reads an invalid value as `true`, and then runs only managed hooks.
- A settings file it reads is not valid JSON or cannot be read. A missing
  project, local or managed file is fine. Claude Code skips a managed file that
  the OS will not let it read, so the check skips it too.
- A project, local or managed file has AgentTrace hooks that are not exact
  copies of the user settings hooks. Claude Code merges hooks from all layers
  and runs an identical handler once, so exact copies pass. Other copies can
  record events twice or with another redaction mode.

**A passing check does not mean Claude Code trusted this folder, ran the hooks
or captured events.** In an interactive session, Claude Code runs no hooks
until you accept the workspace trust dialog. Run a local session and inspect it
with `agent-strace list` and `agent-strace inspect SESSION_ID`.

Not checked: other settings values (run `claude doctor`), the `--settings`,
`--setting-sources`, `--bare` and `--safe-mode` flags, MDM and server-managed
policy, plugin and skill hooks, and `settings.local.json` at the repository
root when you start Claude Code in a subdirectory or worktree. A
`CLAUDE_CONFIG_DIR` set only in a settings `env` block is not followed.

### OpenAI Codex hooks

`agent-strace setup --cli codex` writes user-level hooks to `$CODEX_HOME/hooks.json`, or to `~/.codex/hooks.json` when `CODEX_HOME` is unset or empty, and also prints the JSON shown below. This is the directory Codex reads.

As in Codex, a set `CODEX_HOME` must name an existing directory, and `~` in its value is not expanded. Otherwise setup fails and writes nothing. Setup creates `~/.codex` when it is missing. Codex does not read `CODEX_CONFIG_DIR`, so setup ignores it and prints a note when it names another directory. Older agent-strace versions wrote `hooks.json` there, where Codex never loads it. You can delete that stale file.

Before replacing an existing file, setup copies its original bytes once to `hooks.json.agent-strace.bak` in the same directory. Repeated setup runs preserve that first backup. To restore the pre-AgentTrace config:

```bash
cp ~/.codex/hooks.json.agent-strace.bak ~/.codex/hooks.json
```

When `CODEX_HOME` is set, use `cp "$CODEX_HOME/hooks.json.agent-strace.bak" "$CODEX_HOME/hooks.json"` instead. If no config existed before setup, no backup is created; remove the generated `hooks.json` from that directory. **Setup replaces the entire hooks file**; if you added other tools after a prior backup, inspect and merge hooks manually instead of blindly re-running setup.


```json
{
  "hooks": {
    "SessionStart": [{ "matcher": "startup|resume|clear|compact", "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex session-start" }] }],
    "UserPromptSubmit": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex user-prompt" }] }],
    "PreToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex pre-tool" }] }],
    "PostToolUse": [{ "matcher": ".*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex post-tool" }] }],
    "Stop": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex stop" }] }],
    "SubagentStart": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex subagent-start" }] }],
    "SubagentStop": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex subagent-stop" }] }],
    "SessionEnd": [{ "hooks": [{ "type": "command", "command": "agent-strace hook --provider codex session-end", "timeout": 3 }] }]
  }
}
```

Codex sends one JSON object to each command hook on stdin. agent-strace records the common Codex fields (`session_id`, `turn_id`, `tool_use_id`, `tool_name`, `tool_input`, `tool_response`, `prompt`, `last_assistant_message`, and for subagent hooks `agent_id` and `agent_type`) into the same `.agent-traces/` session store used by Claude Code. `SessionEnd` records a `session_end` event; its `timeout` of 3 seconds is the most Codex allows for that hook. A Codex subagent is not a separate session: `SubagentStart` and `SubagentStop` are recorded as `subagent_start` and `subagent_stop` events in the parent's trace, and the subagent's tool calls are recorded there too. See the [capture matrix](capture-matrix.md#openai-codex) for details.

**Upgrading from an earlier agent-strace:** installs made before `SessionEnd`, `SubagentStart` and `SubagentStop` were added do not have those hooks, so Codex traces have no `session_end` or subagent events. Re-run `agent-strace setup --cli codex`, then trust the new hooks in `/hooks`; Codex treats them as new hooks, and `codex exec` skips them until they are trusted. `agent-strace setup --cli codex --verify` reports them as missing until setup is re-run.

If Codex does not list the hooks after you create the file:

- Use the root `$CODEX_HOME/hooks.json` file (default `~/.codex/hooks.json`) for user-level hooks, or `<repo>/.codex/hooks.json` for project-level hooks. `$CODEX_HOME/hooks/hooks.json` is for plugin-bundled hooks and is not the normal user config path.
- Run setup with the same `CODEX_HOME` that Codex sees. Setup resolves the directory from its own environment.
- Check `$CODEX_HOME/config.toml` (default `~/.codex/config.toml`) and remove `[features].hooks = false` (or the legacy `[features].codex_hooks = false`) if present. Hooks are enabled by default unless a user, system, or admin config layer disables them.
- Reload Codex or press refresh in the Hooks view. Non-managed command hooks must be reviewed and trusted before they run. `codex exec` skips untrusted hooks without a warning, so trust them in an interactive session's `/hooks` view before relying on headless runs.

To diagnose the installed **Codex user-level** configuration without changing
files, run `agent-strace setup --cli codex --verify`. It checks that
`hooks.json` in the directory Codex reads (`$CODEX_HOME`, or `~/.codex` when
`CODEX_HOME` is unset or empty) is valid and
contains the generated commands expected by this AgentTrace version. Unrelated
user hooks are allowed. It also verifies that `agent-strace` is on `PATH`
and that its `--version` output matches this installation, and returns non-zero
for missing/stale config, hooks disabled in `config.toml`, a set `CODEX_HOME`
that is not an existing directory, missing executable or version mismatch.
Like setup, it ignores `CODEX_CONFIG_DIR` and prints a note when it names
another directory. A Codex project-specific hooks file is **not** checked by
this command.

The check also reads `config.toml` from the same directory and fails when its
top-level `[features]` table sets `hooks = false` (including the dotted form
`features.hooks = false`). The legacy key `codex_hooks` counts too; when both
keys are set, `hooks` wins, as in Codex. A missing `config.toml` or an unset
key passes because Codex enables hooks by default. Invalid TOML, or a
non-boolean value for either key, fails with a diagnostic naming the file,
because Codex rejects such a config. Codex profiles
(`[profiles.<name>.features]`), `-c` overrides, and project, system, or managed
config layers are **not** checked. On Python 3.10, which has no `tomllib`, a
minimal built-in reader looks only for these keys and does not validate the
rest of the file.

**A passing check does not mean Codex has enabled or trusted these hooks, that
a provider process ran, or that events were captured.** Reload Codex, review
the command hooks in its Hooks UI, run a local test session, and use
`agent-strace list` / `agent-strace inspect SESSION_ID` to inspect
actually observed evidence. Verification invokes only the local AgentTrace
executable with `--version`; it does not execute hooks or write config.

A successful verification prints the installed hooks path, matching event count,
actual redaction mode, the `config.toml` path and hooks feature state, and
resolved executable path and version. It accepts
consistent default, `--redact` and `--no-redact` installations regardless
of flags passed to `--verify`. On mismatches, inspect and edit the reported
hook entry; re-running setup can overwrite non-AgentTrace user hooks.

### Gemini CLI hooks

`agent-strace setup --cli gemini` writes a Gemini CLI extension to the user extensions directory that Gemini CLI loads:

```
~/.gemini/extensions/agent-strace/
├── gemini-extension.json
└── hooks/
    └── hooks.json
```

When `GEMINI_CLI_HOME` is set and not empty, setup writes to `$GEMINI_CLI_HOME/.gemini/extensions/agent-strace` instead, as Gemini CLI does. The value is used as is: `~` is not expanded, and a relative value starts from the current directory. Run setup with the same `GEMINI_CLI_HOME` that Gemini CLI sees, and prefer an absolute path: a relative one only matches when both start in the same directory. Setup creates missing directories, as Gemini CLI does, and prints the paths it wrote.

Gemini CLI does not read `GEMINI_CONFIG_DIR`, so setup ignores it and prints a note when it names a different directory. In that case an older agent-strace version may have installed the extension under `$GEMINI_CONFIG_DIR/extensions/agent-strace`, which Gemini CLI does not load. You can delete that copy.

The generated `hooks.json` registers `SessionStart`, `BeforeAgent`, `BeforeTool`, `AfterTool`, `AfterAgent`, and `SessionEnd` command hooks:

```json
{
  "hooks": {
    "BeforeTool": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini pre-tool" }] }],
    "AfterTool": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini post-tool" }] }],
    "BeforeAgent": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini user-prompt" }] }],
    "AfterAgent": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini stop" }] }],
    "SessionStart": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini session-start" }] }],
    "SessionEnd": [{ "matcher": "*", "hooks": [{ "type": "command", "command": "agent-strace hook --provider gemini session-end" }] }]
  }
}
```

Gemini sends one JSON object to each command hook on stdin. agent-strace records `session_id`, `tool_name`, `tool_input`, `tool_response`, `prompt`, and `prompt_response` into the same `.agent-traces/` session store used by Claude Code and Codex.

### Cursor hooks

`agent-strace setup --cli cursor` writes a project-local Cursor hooks file:

```
.cursor/
└── hooks.json
```

Set `CURSOR_CONFIG_DIR` to write the file somewhere else. The generated config registers Cursor-native prompt, shell, file-edit, assistant-response, session-start, and session-end command hooks:

```json
{
  "version": 1,
  "hooks": {
    "sessionStart": [{ "type": "command", "command": "agent-strace hook --provider cursor session-start" }],
    "beforeSubmitPrompt": [{ "type": "command", "command": "agent-strace hook --provider cursor before-submit-prompt" }],
    "beforeShellExecution": [{ "type": "command", "command": "agent-strace hook --provider cursor before-shell-execution" }],
    "afterShellExecution": [{ "type": "command", "command": "agent-strace hook --provider cursor after-shell-execution" }],
    "afterFileEdit": [{ "type": "command", "command": "agent-strace hook --provider cursor after-file-edit" }],
    "afterAgentResponse": [{ "type": "command", "command": "agent-strace hook --provider cursor after-agent-response" }],
    "stop": [{ "type": "command", "command": "agent-strace hook --provider cursor stop" }],
    "sessionEnd": [{ "type": "command", "command": "agent-strace hook --provider cursor session-end" }]
  }
}
```

Cursor hook coverage depends on the events Cursor emits. Native hooks capture prompts, shell commands, file edits, stop markers, and agent responses when available. MCP server tool calls are still captured most reliably through the MCP proxy configuration below.

### GitHub Copilot CLI hooks

The recommended setup is the Copilot CLI plugin, which bundles the hooks plus a trace analyst agent and skill:

```bash
copilot plugin marketplace add Siddhant-K-code/agent-trace
copilot plugin install agent-strace@agent-trace
```

See [Copilot CLI plugin](copilot-plugin.md) for verification and usage. Alternatively, `agent-strace setup --cli copilot` writes only the user-level Copilot hooks:

```
~/.copilot/
└── hooks/
    └── agent-strace.json
```

Set `COPILOT_HOME` to install into a different Copilot config directory. The generated config registers Copilot lifecycle hooks using the native event names:

```json
{
  "version": 1,
  "hooks": {
    "sessionStart": [{ "type": "command", "command": "agent-strace hook --provider copilot session-start", "timeoutSec": 10 }],
    "userPromptSubmitted": [{ "type": "command", "command": "agent-strace hook --provider copilot user-prompt", "timeoutSec": 10 }],
    "preToolUse": [{ "type": "command", "command": "agent-strace hook --provider copilot pre-tool", "timeoutSec": 10 }],
    "postToolUse": [{ "type": "command", "command": "agent-strace hook --provider copilot post-tool", "timeoutSec": 10 }],
    "postToolUseFailure": [{ "type": "command", "command": "agent-strace hook --provider copilot post-tool-failure", "timeoutSec": 10 }],
    "agentStop": [{ "type": "command", "command": "agent-strace hook --provider copilot stop", "timeoutSec": 10 }],
    "subagentStart": [{ "type": "command", "command": "agent-strace hook --provider copilot subagent-start", "timeoutSec": 10 }],
    "subagentStop": [{ "type": "command", "command": "agent-strace hook --provider copilot subagent-stop", "timeoutSec": 10 }],
    "sessionEnd": [{ "type": "command", "command": "agent-strace hook --provider copilot session-end", "timeoutSec": 10 }]
  }
}
```

Copilot sends hook payloads on stdin. agent-strace records session starts, user prompts, hook-visible tool calls/results, failed tools, stop markers, subagent starts and stops, and session ends. Copilot's stop hook provides `transcriptPath` and `stopReason` but no response text, so assistant responses are not captured.

Copilot can fire the first `userPromptSubmitted` before `sessionStart`. A subagent started with the `task` tool is recorded as its own session with no start and no end; the parent's `subagentStart` and `subagentStop` hooks link it to the parent, so `agent-strace tree` shows it. If you set up Copilot hooks before this release, re-run `agent-strace setup --cli copilot` (or update the plugin) to register the subagent hooks; sessions captured earlier stay unlinked. See the [capture matrix](capture-matrix.md#github-copilot-cli) for the verified Copilot CLI version and known gaps.

Stop and session-end coverage is provider-defined. agent-strace records these hooks when the agent CLI emits them, including empty stop payloads, but it cannot force an agent to emit a stop hook for every UI action such as deleting a conversation or interrupting a run.

### Import existing sessions

Already ran sessions without hooks? Import from Claude Code's native JSONL logs:

```bash
# Discover available sessions
agent-strace import --discover

# Import a specific session
agent-strace import ~/.claude/projects/<project>/<session-id>.jsonl
```

---

## Option 2: MCP proxy (any MCP client)

Wraps any MCP server. Works with Cursor, Windsurf, or any MCP client that uses stdio transport.

```bash
# Wrap any MCP server
agent-strace record -- npx -y @modelcontextprotocol/server-filesystem /tmp
agent-strace replay
```

### GitHub Copilot Desktop

GitHub Copilot Desktop can connect to local MCP servers, but it does not expose the same hook system as Copilot CLI. For Copilot Desktop, agent-strace traces the MCP traffic that passes through a wrapped MCP server. It does not capture the full Copilot chat transcript, internal planning, token usage, or non-MCP file edits.

Use one shared trace directory so all wrapped MCP servers write to the same store:

```bash
mkdir -p ~/.agent-strace/traces
which agent-strace
```

Desktop apps often do not inherit the same shell `PATH` as your terminal. If `which agent-strace` prints `/Users/alice/.local/bin/agent-strace`, use that full path in Copilot Desktop.

Add an unwrapped reader server if you want Copilot to query existing traces:

| Field | Value |
|---|---|
| Server name | `agent-trace` |
| Type | `Local` |
| Command | `/Users/alice/.local/bin/agent-strace` |
| Arguments | `--trace-dir /Users/alice/.agent-strace/traces mcp` |

This server only exposes tools such as `list_sessions`, `get_session`, `search_events`, `get_session_summary`, and `diff_sessions`. It does not record Copilot Desktop activity by itself.

Wrap each MCP server whose tool traffic you want to record. For example, to trace the filesystem MCP server:

| Field | Value |
|---|---|
| Server name | `filesystem-traced` |
| Type | `Local` |
| Command | `/Users/alice/.local/bin/agent-strace` |
| Arguments | `--trace-dir /Users/alice/.agent-strace/traces record --name copilot-filesystem -- npx -y @modelcontextprotocol/server-filesystem /Users/alice` |

Disable any unwrapped duplicate filesystem MCP server, otherwise Copilot may call the unwrapped server and bypass the recorder.

Ask Copilot Desktop to use the traced server:

```text
Use the filesystem-traced MCP server to list files in /Users/alice/Desktop.
```

Then inspect the captured session:

```bash
TRACE=/Users/alice/.agent-strace/traces

agent-strace --trace-dir "$TRACE" list
grep -Rl '"event_type":"tool_call"' "$TRACE"/*/events.ndjson
agent-strace --trace-dir "$TRACE" replay <session-id>
agent-strace --trace-dir "$TRACE" inspect <session-id>
```

Long-lived MCP servers may keep `meta.json` counters stale while Copilot Desktop is still running. If `agent-strace list` shows `Tool calls: 0` but `replay` or `inspect` shows `tool_call` and `tool_result` events, the capture is working and the summary metadata has not been finalized yet.

### Cursor

Edit `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per-project):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "agent-strace",
      "args": ["record", "--name", "filesystem", "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### Windsurf

Edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "agent-strace",
      "args": ["record", "--name", "filesystem", "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### Any MCP client (general pattern)

1. Replace the server `command` with `agent-strace`
2. Prepend `record --name <label> --` to the original args
3. Use the tool normally
4. Run `agent-strace replay` to see what happened

### HTTP/SSE proxy

For MCP servers that use HTTP transport:

```bash
agent-strace record-http https://mcp.example.com --port 3100
# Your agent connects to http://127.0.0.1:3100
```

---

## Option 3: Python decorator

Wraps your tool functions directly. No MCP required.

```python
from agent_trace import trace_tool, trace_llm_call, start_session, end_session, log_decision

start_session(name="my-agent")  # use redact=False only for trusted local traces

@trace_tool
def search_codebase(query: str) -> str:
    return search(query)

@trace_llm_call
def call_llm(messages: list, model: str = "claude-4") -> str:
    return client.chat(messages=messages, model=model)

log_decision(
    choice="read_file_first",
    reason="Need to understand current implementation before making changes",
    alternatives=["read_file_first", "search_codebase", "write_fix_directly"],
)

search_codebase("authenticate")
call_llm([{"role": "user", "content": "Fix the bug"}])

meta = end_session()
print(f"Replay with: agent-strace replay {meta.session_id}")
```

`start_session()` writes to `$AGENT_TRACE_DIR` when it is set and non-empty, else `.agent-traces`, so the CLI reads the same store. An explicit `start_session(trace_dir=...)` wins.

---

## Security-sensitive repos

For repos that handle secrets, attestation logic, or cryptographic material:

```bash
cd your-sensitive-repo
agent-strace setup
```

Secret redaction is enabled by default for all capture paths. It redacts API keys, tokens, credentials, private keys, basic-auth URLs, and connection strings before they hit disk. Detected patterns include OpenAI (`sk-*`), GitHub (`ghp_*`, `github_pat_*`), AWS (`AKIA*` and AWS credential key names), Anthropic (`sk-ant-*`), Slack (`xox*`), JWTs, Bearer tokens, and any value under keys like `password`, `secret`, `token`, `api_key`, or `authorization`.

See [security.md](security.md) for the full security guide.
