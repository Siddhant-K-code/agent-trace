# Session analysis: explain, cost, and import

Three commands for understanding what an agent session did and what it cost.

## `agent-strace explain`

Groups the raw event stream into human-readable phases, detects retries, and
reports wasted time. Useful after any session — live-captured or imported.

```bash
agent-strace explain           # latest session
agent-strace explain abc123    # by session ID or prefix
```

### How phases work

The event stream is split at each `USER_PROMPT` boundary. Each phase is
annotated with:

- Files read and written (from `Read`, `Write`, `Edit` tool calls)
- Bash commands run
- Retry count (same command appearing more than once in the phase)
- Failed flag (any `ERROR` event in the phase)

### Example output

```
Session: abc123 (4m 12s, 47 events)

Phase 1: Setup (0:00–0:05, 5 events)
  Read: AGENTS.md, pyproject.toml, src/auth.py

Phase 2: run tests — FAILED (0:05–1:20, 12 events)
  Ran: python -m pytest
  Ran: python3 -m pytest  ← retry
  Ran: pip install pytest  ← retry

Phase 3: Discovery (1:20–1:35, 3 events)
  Read: pyproject.toml

Phase 4: run tests (1:35–4:12, 8 events)
  Ran: uv run pytest

Files touched: 3 read, 0 written
Retries: 3 (wasted 1m 15s, 30% of session)
```

### What counts as a retry

Only exact duplicate Bash commands within the same phase are counted. Semantically
similar commands (`pytest` vs `python -m pytest`) are not detected as retries —
this avoids false positives. Repeated file reads are not counted as retries.

---

## `agent-strace cost`

Estimates token usage and dollar cost per phase. Flags wasted spend on failed phases.

```bash
agent-strace cost                          # latest session, sonnet pricing
agent-strace cost abc123                   # specific session
agent-strace cost abc123 --model opus      # different model pricing
agent-strace cost abc123 --model haiku
agent-strace cost abc123 --input-price 3.0 --output-price 15.0  # custom pricing
```

### Supported models

| Flag | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| `sonnet` (default) | $3.00 | $15.00 |
| `opus` | $15.00 | $75.00 |
| `haiku` | $0.25 | $1.25 |
| `gpt4` | $30.00 | $60.00 |
| `gpt4o` | $5.00 | $15.00 |

Use `--input-price` and `--output-price` to override with any custom pricing.

### Example output

```
Session: abc123 — Estimated cost: $0.0189
Model: sonnet  |  8,200 input tokens, 3,100 output tokens

  Phase 1: Setup                        $0.0021  (11%)  3,200in  800out
  Phase 2: run tests — FAILED           $0.0094  (50%)  4,100in 1,800out  ← wasted
  Phase 3: Discovery                    $0.0018   (9%)    600in  400out
  Phase 4: run tests                    $0.0056  (30%)  2,100in 1,200out

Wasted on failed phases: $0.0094 (50%)
```

### Accuracy

Token counts are estimated from event payload size using the `len / 4` heuristic
(4 characters ≈ 1 token). This is accurate to within ±30–50% for typical English
text. It is useful for identifying expensive phases and wasted spend, not for
billing reconciliation.

---

## `agent-strace import`

Import an existing Claude Code session without re-running it. Claude Code stores
session logs in `~/.claude/projects/` as JSONL files.

### Discover available sessions

```bash
agent-strace import --discover
```

```
Found 4 Claude Code sessions:

  a1b2c3d4e5f6  1,204 KB  /home/user/projects/my-app
  9f8e7d6c5b4a    312 KB  /home/user/projects/agent-trace
  3c2b1a0f9e8d     88 KB  /home/user/projects/dotfiles
  7a6b5c4d3e2f     24 KB  /home/user/projects/scripts

Import with: agent-strace import <path-to-session.jsonl>
```

### Import a session

```bash
agent-strace import ~/.claude/projects/-home-user-projects-my-app/a1b2c3d4e5f6.jsonl
```

```
Imported session a1b2c3d4e5f6
  38 tool calls, 12 LLM requests, 4,200,000 tokens
  94 events
  Replay with: agent-strace replay a1b2c3d4e5f6
```

### Use the imported session

Once imported, the session works like any live-captured session:

```bash
agent-strace replay a1b2c3d4e5f6
agent-strace explain a1b2c3d4e5f6
agent-strace cost a1b2c3d4e5f6
agent-strace stats a1b2c3d4e5f6
agent-strace export a1b2c3d4e5f6 --format json
agent-strace export a1b2c3d4e5f6 --format otlp --endpoint http://localhost:4318
```

### What gets imported

| Claude Code data | agent-strace event |
|---|---|
| User message text | `user_prompt` |
| Assistant text response | `assistant_response` |
| `tool_use` content block | `tool_call` |
| `tool_result` content block | `tool_result` |
| `toolUseResult` field | `tool_result` (if no content-block result) |
| `turn_duration` system entry | contributes to session duration |
| Token usage (`input_tokens`, `output_tokens`, cache) | `SessionMeta.total_tokens` |
| `isSidechain`, `subagent_type`, `caller.type` | tagged on `tool_call` data |
| `queue-operation` entries | skipped |

---

## Combining the three commands

A typical post-session workflow:

```bash
# 1. Import if you didn't have hooks set up
agent-strace import --discover
agent-strace import ~/.claude/projects/.../session.jsonl

# 2. Get the high-level picture
agent-strace explain

# 3. Find out what it cost and where
agent-strace cost

# 4. Drill into the raw events for a specific phase
agent-strace replay --filter tool_call,error

# 5. Export to your observability platform
agent-strace export <session-id> --format otlp --endpoint http://localhost:4318
```
