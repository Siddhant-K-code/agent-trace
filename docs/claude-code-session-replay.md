# Replay Claude Code sessions

AgentTrace can replay Claude Code sessions captured by hooks or imported from
Claude Code JSONL logs. Replay shows recorded prompts, responses, tools, files,
errors, timing, and subagent relationships when those fields exist in the
source data.

## Capture limits

AgentTrace does not capture hidden reasoning. It records only fields exposed by
Claude Code hooks or present in an imported JSONL file. A clean replay does not
prove that every action was captured.

Claude Code hook coverage is currently marked unverified in the
[capture matrix](capture-matrix.md#claude-code). Hook setup verification checks
configuration. After recording one harmless local session, live verification
can inspect the newest matching trace. Neither mode starts Claude Code or makes
a model request.

The shipped [Claude Code plugin](claude-code-plugin.md) packages the capture
hooks, shared AgentTrace skill, and trace analyst agent. The Python CLI remains
a separate prerequisite. Static `setup --verify` checks settings-file hooks,
not plugin hooks.

## Capture new sessions with hooks

Generate the hook configuration, merge it into the reported Claude Code
settings file, then run Claude Code normally:

```bash
agent-strace setup --cli claude
agent-strace setup --cli claude --verify
agent-strace setup --cli claude --verify --live
```

`setup --cli claude` prints JSON. It does not edit the settings file. The
[setup guide](setup.md#claude-code-hook-verification) lists the settings layers
that static verification checks and the layers it cannot check. The live form
first runs that static check, then reports event types, boundaries, redaction,
evidence health, and capture-matrix gaps from a matching recorded session.
Workspace trust and the provider run remain the user's responsibility.

## Import an existing Claude Code log

```bash
agent-strace import --discover
agent-strace import ~/.claude/projects/PROJECT/SESSION.jsonl
```

Import preserves only supported fields found in the JSONL log. It is not proof
that the native log contains every provider action.

## Replay and inspect

```bash
agent-strace list
agent-strace health SESSION_ID
agent-strace replay SESSION_ID
agent-strace replay SESSION_ID --format html -o claude-session.html
agent-strace inspect SESSION_ID
```

Use `health` before sharing a replay. It reports observed structural gaps and
declared provider limitations separately. HTML replay is self-contained and
can include sensitive event payloads.

## Related documentation

- [Claude Code plugin](claude-code-plugin.md)
- [Setup](setup.md)
- [Live capture verification](setup.md#live-capture-verification)
- [Capture matrix](capture-matrix.md)
- [Evidence health](evidence-health.md)
- [Replay command reference](commands.md#replay)
- [Share disclosure preview](share-disclosure-preview.md)
