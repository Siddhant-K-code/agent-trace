# Using agent-strace with AGENTS.md

## What is AGENTS.md?

`AGENTS.md` is a Markdown file placed at the root of a repository that tells AI coding agents how to work with the project. It specifies tool permissions, constraints, coding conventions, and model hints. Claude Code, Cursor, and other agents read it automatically at session start.

See the [AGENTS.md specification](https://docs.anthropic.com/en/docs/claude-code/memory) for the full format.

---

## How agent-strace reads AGENTS.md

agent-strace uses `AGENTS.md` as a **behavioral baseline anchor**. When a session starts, it records the current `AGENTS.md` hash alongside the session metadata. This lets it detect when agent behavior changes after an `AGENTS.md` edit — distinguishing intentional config changes from unexpected drift.

Specifically, agent-strace reads:

| Field | How it's used |
|---|---|
| Tool permissions (`allowed_tools`, `disallowed_tools`) | Checked against actual tool calls in `agent-strace audit` |
| File path constraints (`read_only_paths`, `write_paths`) | Checked in `agent-strace audit --policy` |
| Model hints (`model`) | Compared against `LLM_REQUEST` events in drift detection |
| Any free-text constraints | Indexed for `agent-strace optimize` suggestions |

---

## Change detection

`agent-strace drift` computes a behavioral fingerprint for each session (tool mix, error rate, retry rate, blast radius) and compares it against a saved baseline. When `AGENTS.md` changes, the baseline should be updated so drift is measured from the new intended behavior, not the old one.

```bash
# Check for behavioral drift since the last baseline
agent-strace drift

# Check drift specifically after an AGENTS.md edit
agent-strace drift --baseline .agent-strace/baseline.json
```

The drift report shows which dimensions changed and by how much:

```
Behavioral drift report
=======================
Baseline: .agent-strace/baseline.json (saved 2026-05-01)
Sessions compared: 12

  error_rate     0.04 → 0.12  ▲ +200%  ⚠ DRIFT
  retry_rate     0.08 → 0.09  ▲  +12%  OK
  blast_radius   3.2  → 3.1   ▼   -3%  OK
  tool_mix       0.91 similarity        OK

Verdict: DRIFT DETECTED — error_rate increased significantly.
Run `agent-strace explain` on recent sessions to investigate.
```

---

## Baseline anchoring

After updating `AGENTS.md`, save a new baseline so future drift is measured from the new intended behavior:

```bash
# Save a baseline after an AGENTS.md update
agent-strace drift --save-baseline .agent-strace/baseline.json

# Tag it for traceability
git add .agent-strace/baseline.json
git commit -m "chore: update behavioral baseline after AGENTS.md change"
```

The baseline file is a small JSON document (< 1 KB) that records the statistical distribution of behavioral metrics across recent sessions. It is safe to commit.

---

## CI integration

Gate on behavioral regression whenever `AGENTS.md` changes. Add this to your CI workflow:

```yaml
# .github/workflows/agent-behavior.yml
name: Agent behavior gate

on:
  push:
    paths:
      - AGENTS.md
      - .agent-strace/baseline.json

jobs:
  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install agent-strace
        run: pip install agent-strace

      - name: Check behavioral drift
        run: |
          agent-strace drift \
            --baseline .agent-strace/baseline.json \
            --save-baseline /tmp/new-baseline.json
          # Fails with exit code 1 if drift exceeds threshold

      - name: Run eval CI gate
        run: |
          agent-strace eval ci \
            --baseline .agent-strace/eval-baseline.json \
            --tolerance 1
```

The `drift` command exits non-zero when any dimension exceeds its threshold, making it suitable as a CI gate.

---

## Recommended AGENTS.md snippet

Add this to your project's `AGENTS.md` to tell agents how to work with agent-strace:

```markdown
## Observability

This project uses [agent-strace](https://github.com/Siddhant-K-code/agent-trace) to
capture and analyse agent sessions.

- Traces are stored in `.agent-traces/` (gitignored)
- Run `agent-strace replay` after a session to review what happened
- Run `agent-strace drift` to check for behavioral regression
- Run `agent-strace retention clean --max-age-days 30` periodically to free disk space

Do not delete `.agent-traces/` manually — use `agent-strace retention clean`.
```

---

## Detecting AGENTS.md drift

`agent-strace drift` also detects when the agent's behavior diverges from what `AGENTS.md` specifies — for example, if the agent starts writing to paths that `AGENTS.md` marks as read-only, or calling tools that are listed as disallowed.

```bash
# Audit the latest session against the current AGENTS.md policy
agent-strace audit --policy AGENTS.md

# Check all sessions from the last 7 days
agent-strace audit --policy AGENTS.md --since 7d
```

---

## Full workflow

```
1. Edit AGENTS.md
2. Run a few test sessions with the agent
3. agent-strace drift --save-baseline .agent-strace/baseline.json
4. git commit -m "chore: update baseline after AGENTS.md change"
5. CI gates on future drift automatically
```

This closes the loop between "I updated the agent's instructions" and "I know the agent is actually following them."
