# Using agent-strace with AGENTS.md

## What is AGENTS.md?

`AGENTS.md` is a Markdown file placed at the root of a repository that tells AI coding agents how to work with the project. It specifies tool permissions, constraints, coding conventions, and model hints. Many coding agents read it automatically at session start.

`AGENTS.md` is an open format with no required fields. See [agents.md](https://agents.md/) for the format and the list of agents that read it.

---

## How agent-strace reads AGENTS.md

agent-strace does not record an `AGENTS.md` hash when a session starts. It reads `AGENTS.md` only when you run one of these commands:

| Command | What it does with `AGENTS.md` |
|---|---|
| `agent-strace config-watch snapshot` | Records SHA-256 hashes of `AGENTS.md` and the other tracked config files. `config-watch check` exits 1 if any of them changed since the last snapshot; `config-watch affected` lists sessions that ran after a change |
| `agent-strace context-score` | Groups sessions by `AGENTS.md` version (from `config-watch` snapshots) and compares cost, tool calls, lint findings and scope violations per version |
| `agent-strace postmortem --agents-md AGENTS.md` | Flags shell commands that contradict a "use X, not Y", "never use Y" or "do not use Y" line |
| `agent-strace optimize --target AGENTS.md` | Proposes additions to `AGENTS.md` from trace failures |
| `agent-strace freshness` | Reads file globs listed under a scope heading (a line ending in `:` that mentions scope, files, include or watch) in `CLAUDE.md` or `AGENTS.md` |

agent-strace does not parse tool permissions, path constraints or model hints from `AGENTS.md`. To enforce tool and path rules, write a JSON scope policy and check sessions with `agent-strace audit` (see [Checking sessions against rules](#checking-sessions-against-rules)).

---

## Change detection

`config-watch` tells you when `AGENTS.md` changed. Nothing takes a snapshot automatically, so take one after each edit:

```bash
# Hash AGENTS.md and the other tracked config files
agent-strace config-watch snapshot

# Exit 1 if a tracked file changed since the last snapshot
agent-strace config-watch check

# List sessions that ran after a config change
agent-strace config-watch affected --since 7d
```

`agent-strace drift` measures whether behavior changed. It computes a behavioral fingerprint for a set of sessions (tool mix, error rate, retry rate, tool calls, cost, blast radius, session duration, decision depth) and compares two sets. It does not read `AGENTS.md`.

Pass `--since Nd` or `--current START:END`; without either, `drift` exits 1. With `--since`, drift takes the sessions from the last N days and splits them in half: the newer half is compared against the older half, or against `--baseline FILE` when given.

```bash
# Compare the newer half of the last 14 days with the older half
agent-strace drift --since 14d

# Compare the newer half of the last 7 days with a saved baseline
agent-strace drift --since 7d --baseline .agent-strace/baseline.json
```

The drift report shows an overall score and a score for each dimension (trimmed):

```
Behavioral drift report
Baseline: 2026-10-30 to 2026-11-01 (6 sessions)
Current:  2026-11-06 to 2026-11-07 (6 sessions)
──────────────────────────────────────────────────────────────────────
Overall drift score: 0.40  ❌  HIGH  (threshold: 0.20)

  Dimension               Score  Status
──────────────────────────────────────────────────────────────────────
  tool_mix                0.462  ❌  high
  error_rate              1.000  ❌  high
  retry_rate              0.650  ❌  high
  tool_calls              0.286  ❌  high
  cost_usd                0.274  ❌  high
  blast_radius            0.000  ✅  stable
  session_duration_s      0.438  ❌  high
  decision_depth          0.000  ✅  stable

High-drift dimensions:

  tool_mix (0.46):
    Baseline: Read=40%, Grep=20%, Edit=20%
    Current:  Bash=86%, Edit=14%

  error_rate (1.00):
    Baseline: mean=0.00 p50=0.00 p95=0.00
    Current:  mean=0.29 p50=0.29 p95=0.29
  ...
```

`drift` exits 1 when the overall score is at or above `--threshold` (default 0.20).

---

## Baseline anchoring

After updating `AGENTS.md` and running a few sessions with it, save a new baseline so future drift is measured from the new intended behavior:

```bash
# Save a baseline after an AGENTS.md update
agent-strace drift --since 7d --save-baseline .agent-strace/baseline.json

# Tag it for traceability
git add .agent-strace/baseline.json
git commit -m "chore: update behavioral baseline after AGENTS.md change"
```

`--save-baseline` writes the fingerprint of the current window and exits 0 without comparing. With `--since`, that is the newer half of the sessions in the window, so pick N so that the newer half covers only sessions run after the edit.

The baseline file is a small JSON document with per-dimension statistics (tool mix, means and percentiles) and no event content. It is safe to commit.

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

      # .agent-traces/ is not committed. Restore the recorded sessions here
      # (for example from an artifact) before the checks below.

      - name: Check behavioral drift
        run: |
          agent-strace drift \
            --since 7d \
            --baseline .agent-strace/baseline.json
          # Fails with exit code 1 if the drift score reaches the threshold

      - name: Run eval CI gate
        run: |
          agent-strace eval ci \
            --baseline .agent-strace/eval-baseline.json \
            --tolerance 0.05
```

`drift` exits 1 when the overall drift score is at or above `--threshold` (default 0.20). Do not add `--save-baseline` to the gate: it saves and exits 0 without comparing.

`eval ci` scores the latest session with the scorers in `.agent-evals.yaml` and exits 1 if any scorer fails or any score drops more than `--tolerance` below the baseline. Scores run from 0 to 1, so `--tolerance 0.05` allows a 0.05 drop. Create the baseline with `agent-strace eval ci --save-baseline .agent-strace/eval-baseline.json`.

Both steps need the recorded sessions. `.agent-traces/` git-ignores itself, so a fresh checkout has none: with a baseline file and no sessions in the window, `drift` reports high drift, and `eval ci` exits 1 with "No sessions found." Restore the store before these steps, or pass the global `--trace-dir` (before the subcommand) to point at it.

---

## Recommended AGENTS.md snippet

Add this to your project's `AGENTS.md` to tell agents how to work with agent-strace:

```markdown
## Observability

This project uses [agent-strace](https://github.com/Siddhant-K-code/agent-trace) to
capture and analyse agent sessions.

- Traces are stored in `.agent-traces/`, which git-ignores itself via `.agent-traces/.gitignore`
- Run `agent-strace replay` after a session to review what happened
- Run `agent-strace drift --since 7d` to check for behavioral regression
- Run `agent-strace retention clean --max-age-days 30` periodically to free disk space

Do not delete `.agent-traces/` manually — use `agent-strace retention clean`.
```

---

## Checking sessions against rules

`drift` does not compare sessions with `AGENTS.md`. Two commands check what an agent did against rules:

- `postmortem --agents-md` reads `AGENTS.md` and flags shell commands that contradict a "use X, not Y", "never use Y" or "do not use Y" line. It is a heuristic for one session and only reports matches, under "AGENTS.md violations".
- `audit` checks one session against a JSON scope policy (file read and write, command and network allow/deny lists), not against `AGENTS.md`. It exits 1 if any tool call is denied.

```bash
# Flag commands in the latest session that contradict AGENTS.md
agent-strace postmortem --agents-md AGENTS.md

# Generate a JSON policy from observed sessions, then edit it
agent-strace policy generate --output .agent-scope.json

# Audit the latest session against the policy (exit 1 on violations)
agent-strace audit --policy .agent-scope.json

# Report what the policy would block across the last 7 days
agent-strace policy backtest --policy .agent-scope.json --days 7
```

---

## Full workflow

```
1. Edit AGENTS.md
2. agent-strace config-watch snapshot
3. Run a few test sessions with the agent
4. agent-strace drift --since 7d --save-baseline .agent-strace/baseline.json
5. git commit -m "chore: update baseline after AGENTS.md change"
6. CI compares later sessions against the baseline (see CI integration)
```

This links an instruction change to measured behavior: `config-watch` shows when `AGENTS.md` changed, `drift` shows whether behavior moved, and `postmortem --agents-md` flags commands that contradict it.
