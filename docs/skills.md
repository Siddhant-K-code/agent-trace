# Skill adoption: `agent-strace skills`

`agent-strace skills` reports, per skill, what the recorded trace shows along
four steps:

```
available -> loaded -> followed -> outcome
```

It exists because the usual way of judging skills and AGENTS.md (reading the
model's "thinking") is not reliable. Provider thinking is often summarized or
omitted, and stated reasoning is not always faithful to what the model did.
Tool calls, file reads, commands and results in the trace are a better
instrument, so this command only uses those.

> **There is no score, grade, or verdict.** The command reports observed facts
> with counts (`count/n`), labels inferences as *signals*, and lists what it
> cannot measure. A skill that was never loaded is a fact. A skill whose
> commands were run after loading is a signal. Neither says whether the skill
> was "good".

```bash
agent-strace skills                       # aggregate across sessions (last 30 days)
agent-strace skills <session-id>          # one session
agent-strace skills --skills-dir ~/my-plugin/skills --format json
```

Flags are listed in [commands.md](commands.md#skills).

## 1. Available

Skills are discovered from the filesystem at the time you run the command:

| Scope | Directories |
|---|---|
| Repository (relative to the session working directory) | `.github/skills`, `.claude/skills`, `.agents/skills`, `.gemini/skills`, `.codex/skills`, `.cursor/skills` |
| User | `~/.copilot/skills`, `~/.claude/skills`, `~/.agents/skills`, `~/.gemini/skills`, `~/.codex/skills`, `~/.cursor/skills` |
| Extra | each `--skills-dir DIR` (plugin, extension, or app-bundled skill directories) |

- A skill is `<dir>/<name>/SKILL.md`. A `SKILL.md` directly in `<dir>` is also
  accepted. Deeper nesting is not scanned.
- The name comes from the frontmatter `name:` field, falling back to the
  directory name.
- When the same name appears in more than one directory, the first one wins
  (repository, then user, then extra).
- The union of all directories is used for every session. Directories are not
  filtered per provider.
- The session working directory comes from `session_start.data.cwd`. If it is
  missing, `--workspace` is used, then the current directory.

**Availability reflects the filesystem now, not the session's own time.** A
skill added last week will appear as "available" in older sessions. Use
`--since` to keep the window close to the current skill set.

### Context files

`AGENTS.md`, `CLAUDE.md`, `GEMINI.md` and `.github/copilot-instructions.md` are
reported as present or absent in the session working directory. Most agents
inject these automatically, so whether they were loaded leaves no event in
hooks. The report says so (`load_observable: false`) instead of guessing.

## 2. Loaded

A load is detected from these recorded shapes. Arguments are read from either
`arguments` or `tool_input`. Serialized argument strings are also parsed.

| Agent | Recorded shape | Method | Invoked by |
|---|---|---|---|
| GitHub Copilot | `tool_call` with `tool_name: "skill"`, argument `skill` | `skill_tool` | model |
| Claude Code | `tool_name: "Skill"`, argument `skill` (newer) or `command` (older); `plugin:name` is accepted | `skill_tool` | model |
| Claude Code | a `user_prompt` that starts with `/<known-skill>` (typed slash commands bypass `PreToolUse`) | `user_prompt` (`slash_command`) | user |
| Gemini CLI | `tool_name: "activate_skill"`, argument `name` | `skill_tool` | model |
| Codex, Cursor, others | a read tool or `file_read` event whose path ends in `SKILL.md`, or a shell read (`cat`, `head`, `sed -n`, ...) of a `SKILL.md` path | `file_read` | model |
| Codex | a `$<known-skill>` mention in a user prompt | `user_prompt` (`dollar_mention`) | user |

Prompt-based loads only match names of skills that are available, so a stray
`/tmp` or `$HOME` is not counted. Editing a `SKILL.md` is not a load.

For each load the report records:

- the event id;
- seconds since session start;
- whether the load happened before the first edit;
- whether the load succeeded. This is only set when a linked `tool_result` or
  `error` exists.

A loaded skill with no definition on disk is still reported as a load, with
`definition_found: false`. This is common for app-bundled or plugin skills.
Add their directory with `--skills-dir` to measure follow signals.

## 3. Followed: signals, not verdicts

After the first load of a skill, the following events are checked against that
skill's directory and its `SKILL.md`:

| Signal | Meaning |
|---|---|
| `resource_read` | A file inside the skill directory (other than `SKILL.md`) was read, either by a read tool or by a reader command such as `cat`, `grep`, `jq` or `diff` |
| `script_run` | A command executed a path inside the skill directory, either directly or through an interpreter (`bash`, `python`, `node`, ...) |
| `command_match` | A command's two-token signature matches a command from the skill's fenced `bash`/`sh`/`shell`/`zsh`/`console`/`text` blocks |

A **signature** is the first two words of each simple command, after:

- removing `#` comments and heredoc bodies;
- splitting on newlines, `;`, `&`, `&&`, `||`, pipes and subshell parentheses;
- stripping environment assignments, `env`, `sudo` and git pager flags;
- skipping low-signal builtins such as `cd`, `echo` and `export`.

Relative paths in a command are resolved against the session working directory
and any earlier `cd` in the same command line. A bare file name with no `/`
(for example `cat reference.md` after `cd <skill dir>`) only counts if it names
a file the skill directory actually contains.

Only signatures and paths relative to the skill directory are reported. Command
text is not.

`follow.status` is one of:

- `signals_observed`: at least one signal was seen.
- `no_signal`: the definition has commands or resources, but none were seen
  after the load.
- `not_measurable`: either `definition_not_found`, or
  `definition_has_no_commands_or_resources` (a prose-only skill).

The report also includes `seconds_to_first_signal` and
`first_signal_before_next_prompt`, which says whether the first signal came
before the user's next prompt.

The command never reports "violated" or "ignored". Things to keep in mind:

- Generic signatures such as `go test`, `gh api` or `curl -s` would often run
  without the skill, so a `command_match` on them is weak evidence.
- `resource_read` and `script_run` on skill-owned files are stronger signals.
- The aggregate `follow_signal_kinds` breakdown shows which kind of signal was
  seen.

## 4. Outcome after load: facts only

From events after the load:

- `tests_run`: the number of test commands, using the same classifier as the
  [session review projection](session-review-projection.md).
- `last_test`: the last test command's linked `tool_result` or `error`, with
  `exit_code` and `success` when the agent recorded them. Result text is not
  parsed.
- `errors`: the count of `error` events.
- `session_end_recorded`: whether a `session_end` event exists.

These facts sit next to the load. They do not mean the skill caused them.

## Aggregate mode

Without a session id, sessions started within `--since` (default `30d`) are
summarized per skill:

| Field | Meaning |
|---|---|
| `sessions_available` / `sessions_loaded` | Session counts |
| `load_rate` | Sessions loaded while available / sessions available |
| `invocations` | Load events by user and by model |
| `median_seconds_to_load` | Median time from session start to the first load |
| `loaded_before_first_edit` | count/n, where n = loaded sessions that had an edit |
| `follow_signals`, `follow_signal_kinds` | count/n, where n = loads with a measurable definition |
| `tests_after_load` | Loaded sessions with a test command after the load |

**Never loaded** lists skills that were available in at least `--min-sessions`
sessions (default 5) but never loaded. It is a starting point for
investigation, not evidence that a skill is broken. Two caveats apply, and are
included in JSON output as `caveats.never_loaded` and summarized in one line in
text output:

- **Availability is not relevance.** A skill that was never loaded may simply
  not match any task in the window. The trace cannot tell whether a task
  should have triggered it.
- **Availability reflects today's filesystem.** Sessions that ran before a
  skill was installed are still counted as "available", so `sessions_available`
  is over-counted for recently added skills. There is no install-time heuristic
  to correct for this. Narrowing `--since` reduces the effect.

To find out whether a never-loaded skill fails to trigger, pair the list with a
trigger eval: run tasks where the skill is relevant and check whether it loads.
If it does not, check its `name` and `description` frontmatter.

**Loaded vs available-but-not-loaded** compares session-level facts (any test
run, any error, `session_end` recorded) between the two groups, with n shown for
each. It is suppressed for a skill unless both groups reach `--min-sessions`.
It is correlational: the agent loads a skill because of the task, so the groups
differ in task mix.

## Malformed traces

Events without a usable timestamp are skipped and counted in
`events_skipped_malformed`. Unexpected field types (for example a non-object
`data`) are treated as empty rather than failing the report. In aggregate mode,
a session that cannot be read is skipped and counted in
`sessions_skipped_unreadable`. Both counts are printed when they are non-zero.

## What this cannot measure

- What the skill set looked like when an old session ran (availability is read
  now).
- Whether any task in the window should have triggered a given skill.
- Whether auto-injected context files were read or used.
- Skills a provider injects without a tool call, file read, or prompt mention.
- Whether non-command guidance (tone, judgement, ordering, "prefer X over Y")
  was followed.
- Model reasoning. It is never used as evidence.
- Whether a skill caused an outcome.

## Why there is no score

Composite scores hide these limits and reward the wrong things. A score that
favours fewer tool calls or lower cost will, for example, rank a session that
skipped the tests above one that ran them. Skill evaluations also often find
that skills do not trigger, or add context cost without measurable gain. The
honest unit is the observed fact, together with its n.

## Privacy

The report never prints `SKILL.md` bodies, prompt text, command text, or
absolute skill paths. It prints:

- skill names;
- display directories (with the home directory shown as `~`);
- two-token command signatures;
- paths relative to the skill directory;
- event ids;
- counts.
