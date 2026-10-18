# Skill adoption: `agent-strace skills`

What the trace shows about each skill, in four steps. There is no score, grade,
or verdict: facts are counts (`count/n`) and inferences are labelled *signals*.

```mermaid
flowchart LR
    A["Available<br/>SKILL.md on disk"]:::fact --> L["Loaded<br/>tool call, SKILL.md read,<br/>/name or $name"]:::fact
    L --> F["Followed<br/>resource_read<br/>script_run<br/>command_match"]:::signal
    F --> O["Outcome after load<br/>tests · errors<br/>session_end"]:::fact
    classDef fact fill:#dbeafe,stroke:#1d4ed8,color:#0b1f44
    classDef signal fill:#fef3c7,stroke:#b45309,color:#451a03
```

Blue is observed. Amber is consistent with the skill, not caused by it. Model
"thinking" is never used: it is often summarized or omitted, and not reliably
faithful to what the model did.

```bash
agent-strace skills                                  # aggregate, last 30 days
agent-strace skills <session-id>                     # one session
agent-strace skills --skills-dir ~/plugin/skills --format json
```

Flags: [commands.md](commands.md#skills).

## 1. Available

```mermaid
flowchart TB
    R["1 · Repository (session cwd)<br/>.github/skills<br/>.claude/skills<br/>.agents/skills<br/>.gemini/skills<br/>.codex/skills<br/>.cursor/skills"]:::src
    U["2 · User<br/>~/.copilot/skills<br/>~/.claude/skills<br/>~/.agents/skills<br/>~/.gemini/skills<br/>~/.codex/skills<br/>~/.cursor/skills"]:::src
    X["3 · --skills-dir DIR<br/>plugin and<br/>app-bundled skills"]:::src
    R & U & X --> S["Available skills<br/>DIR/name/SKILL.md<br/>duplicate names:<br/>first source wins"]:::fact
    classDef src fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef fact fill:#dbeafe,stroke:#1d4ed8,color:#0b1f44
```

- A `SKILL.md` directly in a directory also counts. Deeper nesting is not
  scanned. The name is frontmatter `name:`, else the directory name.
- The session cwd is `session_start.data.cwd`, else `--workspace`, else the
  current directory. Every directory applies to every session and provider.
- **Availability is read from today's filesystem.** Older sessions see skills
  added later. Narrow `--since` to reduce this.

Context files (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`,
`.github/copilot-instructions.md`) are reported as present or absent with
`load_observable: false`. Agents inject them without a trace event.

## 2. Loaded

```mermaid
flowchart LR
    c1["Copilot<br/>tool skill"] --> T["skill_tool"]:::model
    c2["Claude Code<br/>tool Skill"] --> T
    c3["Gemini CLI<br/>tool activate_skill"] --> T
    c4["Codex, Cursor, others<br/>read of .../SKILL.md"] --> R["file_read"]:::model
    c5["Any provider<br/>prompt starts /name"] --> P["user_prompt"]:::user
    c6["Any provider<br/>prompt mentions $name"] --> P
    classDef model fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b
    classDef user fill:#dcfce7,stroke:#15803d,color:#052e16
```

Indigo is invoked by the model. Green is invoked by the user.

- Tool arguments: `skill` (Copilot, Claude Code), `command` (older Claude
  Code), `name` (Gemini). `plugin:name` is accepted. Arguments come from
  `arguments` or `tool_input`, and serialized strings are parsed.
- A `SKILL.md` read is a read tool, a `file_read` event, or a shell reader
  (`cat`, `head`, `sed -n`, …). Editing `SKILL.md` is not a load.
- Prompt loads match only available skill names, so `/tmp` or `$HOME` is not
  a load.
- Each load records its event id, seconds since start, whether it came before
  the first edit, and whether it succeeded (only when a linked `tool_result` or
  `error` exists).
- A load with no definition on disk is still a load, with
  `definition_found: false`. Add its directory with `--skills-dir`.

## 3. Followed

```mermaid
flowchart TD
    L(["Skill loaded"]) --> D{"SKILL.md found?"}
    D -- no --> N1["not_measurable<br/>definition_not_found"]:::nm
    D -- yes --> C{"Has commands or<br/>resource files?"}
    C -- no --> N2["not_measurable<br/>definition_has_no_commands_or_resources"]:::nm
    C -- yes --> S{"Any signal after<br/>the load?"}
    S -- yes --> Y["signals_observed"]:::ok
    S -- no --> Z["no_signal"]:::none
    classDef nm fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef ok fill:#fef3c7,stroke:#b45309,color:#451a03
    classDef none fill:#fee2e2,stroke:#b91c1c,color:#450a0a
```

| Signal | Seen after the load | Strength |
|---|---|---|
| `resource_read` | A file in the skill directory, other than `SKILL.md`, is read by a read tool or a reader command (`cat`, `grep`, `jq`, …) | Strong |
| `script_run` | A file in the skill directory is executed, directly or through `bash`, `python`, `node`, … | Strong |
| `command_match` | A command's signature matches one from the skill's fenced `bash`, `sh`, `shell`, `zsh`, `console`, `shell-session` or `text` blocks | Weak for generic commands such as `gh api` or `go test` |

The report never says "violated" or "ignored".

A **signature** is the first two words of each simple command. For example:

```text
cd repo && GH_PAGER= gh api repos/o/r/stacks   # list
sudo git --no-pager fetch origin | tail -1
```

gives `gh api`, `git fetch` and `tail -1`:

```mermaid
flowchart TD
    s1["strip comments<br/>and heredocs"] --> s2["split on newlines<br/>; & && || | ( )"]
    s2 --> s3["drop env vars,<br/>sudo, pager flags"]
    s3 --> s4["skip builtins<br/>cd, echo, export"]
    s4 --> s5["first two words"]:::fact
    classDef fact fill:#dbeafe,stroke:#1d4ed8,color:#0b1f44
```

Relative paths resolve against the session cwd and any earlier `cd` in the same
command line. A bare file name counts only if the skill directory contains it.

## 4. Timing and outcome

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant T as Trace
    U->>A: prompt
    A->>T: load skill
    Note over T: seconds_since_start<br/>loaded_before_first_edit, load_succeeded
    A->>T: first matching command
    Note over T: seconds_to_first_signal<br/>first_signal_before_next_prompt = true
    A->>T: test command and result
    Note over T: outcome_after_load<br/>tests_run, last_test, errors
    U->>A: next prompt
```

Outcome facts sit next to the load. They do not mean the skill caused them.

- `tests_run` uses the [session review projection](session-review-projection.md)
  test classifier.
- `last_test` shows the linked result's `exit_code` and `success` when they
  were recorded. Result text is not parsed.
- `errors` and `session_end_recorded` are also reported.

## Aggregate mode

```mermaid
flowchart TD
    W["Sessions in window<br/>--since, default 30d"] --> P["Per skill<br/>available · loaded<br/>load rate · user/model<br/>median time to load<br/>pre-edit · follow<br/>tests after load"]:::fact
    P --> N["Never loaded<br/>available: N+ sessions<br/>loaded: 0"]:::none
    P --> C["Loaded vs not loaded<br/>tests, errors,<br/>session_end<br/>N+ in both groups"]:::signal
    N -. "lead, not verdict" .-> E["Trigger eval<br/>relevant tasks:<br/>does it load?<br/>skills eval"]
    classDef fact fill:#dbeafe,stroke:#1d4ed8,color:#0b1f44
    classDef signal fill:#fef3c7,stroke:#b45309,color:#451a03
    classDef none fill:#fee2e2,stroke:#b91c1c,color:#450a0a
```

N is `--min-sessions` (default 5). A never-loaded skill is a lead: test it with
[`agent-strace skills eval`](skills-eval.md), which runs tasks that should and
should not load it.

| Field | Denominator (n) |
|---|---|
| `load_rate` | Sessions where the skill was available |
| `loaded_before_first_edit` | Loaded sessions that had an edit |
| `follow_signals`, `follow_signal_kinds` | Loaded sessions with a measurable definition |
| `tests_after_load` | Loaded sessions |

**Never loaded** has two caveats. They appear in JSON as `caveats.never_loaded`
and as one line in the text output:

- **Availability is not relevance.** No task in the window may have needed the
  skill.
- **Availability reflects today's filesystem.** Sessions from before a skill
  was installed are over-counted.

**Loaded vs not loaded** is correlational: agents load skills because of the
task, so the groups differ in task mix.

## Limits

This command cannot measure:

- what the skill set looked like when an old session ran;
- whether a task should have triggered a skill (see
  [skills eval](skills-eval.md));
- whether auto-injected context files were used;
- skills injected without a tool call, file read, or prompt mention;
- whether non-command guidance (tone, ordering, "prefer X over Y") was followed;
- model reasoning; or
- whether a skill caused an outcome.

Malformed events and unreadable sessions are skipped and counted in
`events_skipped_malformed` and `sessions_skipped_unreadable`.

**Why there is no score:** a composite score hides these limits. A score that
favours fewer tool calls or lower cost ranks a session that skipped its tests
above one that ran them.

**Privacy:** the report never prints `SKILL.md` bodies, prompt text, command
text, or absolute skill paths. It prints skill names, `~`-relative
directories, two-word signatures, skill-relative paths, event ids, and counts.
