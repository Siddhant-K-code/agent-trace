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
configuration, not whether Claude Code trusted the workspace, ran the hooks, or
produced a trace.

## Capture new sessions with hooks

Generate the hook configuration, merge it into the reported Claude Code
settings file, then run Claude Code normally:

```bash
agent-strace setup --cli claude
agent-strace setup --cli claude --verify
```

`setup --cli claude` prints JSON. It does not edit the settings file. The
[setup guide](setup.md#claude-code-hook-verification) lists the settings layers
that verification checks and the layers it cannot check.

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

- [Setup](setup.md)
- [Capture matrix](capture-matrix.md)
- [Evidence health](evidence-health.md)
- [Replay command reference](commands.md#replay)
- [Share disclosure preview](share-disclosure-preview.md)
