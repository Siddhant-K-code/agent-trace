"""Session diff: structural and semantic comparison of two sessions.

Two modes:
  - Structural (default): compares phase structure, divergence point, files/commands per phase.
  - Semantic (--semantic): compares outcome-level metrics — files touched, commands run,
    cost, duration, errors, and eval scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import TextIO

from .explain import Phase, build_phases, explain_session
from .models import EventType
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhaseDiff:
    index: int
    label_a: str
    label_b: str
    same_label: bool
    files_only_a: list[str]
    files_only_b: list[str]
    cmds_only_a: list[str]
    cmds_only_b: list[str]
    a_failed: bool
    b_failed: bool


@dataclass
class SessionDiff:
    session_a: str
    session_b: str
    divergence_index: int        # first phase index where behaviour differs (-1 = identical)
    phase_diffs: list[PhaseDiff]
    # Summary metrics
    duration_a: float
    duration_b: float
    events_a: int
    events_b: int
    tool_calls_a: int
    tool_calls_b: int
    retries_a: int
    retries_b: int


# ---------------------------------------------------------------------------
# LCS-based phase alignment
# ---------------------------------------------------------------------------

def _lcs_indices(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Return LCS index pairs (i, j) where a[i] == b[j]."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < m and j < n:
        if a[i] == b[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _phase_key(phase: Phase) -> str:
    """Normalised key for LCS matching — use label text."""
    return phase.name.lower().strip()


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------

def diff_sessions(
    store: TraceStore,
    session_a: str,
    session_b: str,
) -> SessionDiff:
    result_a = explain_session(store, session_a)
    result_b = explain_session(store, session_b)

    phases_a = result_a.phases
    phases_b = result_b.phases

    keys_a = [_phase_key(p) for p in phases_a]
    keys_b = [_phase_key(p) for p in phases_b]

    # Align phases via LCS
    aligned = _lcs_indices(keys_a, keys_b)
    aligned_set_a = {i for i, _ in aligned}
    aligned_set_b = {j for _, j in aligned}

    phase_diffs: list[PhaseDiff] = []
    divergence_index = -1

    # Walk aligned pairs
    for pair_idx, (i, j) in enumerate(aligned):
        pa = phases_a[i]
        pb = phases_b[j]

        files_a = set(pa.files_read + pa.files_written)
        files_b = set(pb.files_read + pb.files_written)
        cmds_a = set(pa.commands)
        cmds_b = set(pb.commands)

        only_a_files = sorted(files_a - files_b)
        only_b_files = sorted(files_b - files_a)
        only_a_cmds = sorted(cmds_a - cmds_b)
        only_b_cmds = sorted(cmds_b - cmds_a)

        differs = (
            only_a_files or only_b_files
            or only_a_cmds or only_b_cmds
            or pa.failed != pb.failed
        )

        if differs and divergence_index == -1:
            divergence_index = pair_idx

        phase_diffs.append(PhaseDiff(
            index=pair_idx,
            label_a=pa.name,
            label_b=pb.name,
            same_label=(keys_a[i] == keys_b[j]),
            files_only_a=only_a_files,
            files_only_b=only_b_files,
            cmds_only_a=only_a_cmds,
            cmds_only_b=only_b_cmds,
            a_failed=pa.failed,
            b_failed=pb.failed,
        ))

    # Phases only in A or only in B count as divergence
    if aligned_set_a != set(range(len(phases_a))) or aligned_set_b != set(range(len(phases_b))):
        if divergence_index == -1:
            divergence_index = len(phase_diffs)

    meta_a = store.load_meta(session_a)
    meta_b = store.load_meta(session_b)

    return SessionDiff(
        session_a=session_a,
        session_b=session_b,
        divergence_index=divergence_index,
        phase_diffs=phase_diffs,
        duration_a=result_a.total_duration,
        duration_b=result_b.total_duration,
        events_a=result_a.total_events,
        events_b=result_b.total_events,
        tool_calls_a=meta_a.tool_calls,
        tool_calls_b=meta_b.tool_calls,
        retries_a=result_a.total_retries,
        retries_b=result_b.total_retries,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_duration(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    return f"{int(s) // 60}m {int(s) % 60:02d}s"


def format_diff(result: SessionDiff, out: TextIO = sys.stdout) -> None:
    w = out.write
    a = result.session_a[:12]
    b = result.session_b[:12]

    w(f"\nComparing: {a} vs {b}\n\n")

    if result.divergence_index == -1:
        w("Sessions are structurally identical.\n\n")
    else:
        w(f"Diverged at phase {result.divergence_index + 1}:\n\n")

    for pd in result.phase_diffs:
        if not (pd.files_only_a or pd.files_only_b
                or pd.cmds_only_a or pd.cmds_only_b
                or pd.a_failed != pd.b_failed):
            continue

        w(f"  Phase {pd.index + 1}: {pd.label_a}\n")

        if pd.a_failed and not pd.b_failed:
            w(f"    {a}: FAILED    {b}: passed\n")
        elif pd.b_failed and not pd.a_failed:
            w(f"    {a}: passed    {b}: FAILED\n")

        for f in pd.cmds_only_a:
            w(f"    {a} only:  $ {f[:70]}\n")
        for f in pd.cmds_only_b:
            w(f"    {b} only:  $ {f[:70]}\n")
        for f in pd.files_only_a:
            w(f"    {a} only:  {f}\n")
        for f in pd.files_only_b:
            w(f"    {b} only:  {f}\n")
        w("\n")

    w(f"  {a}: {_fmt_duration(result.duration_a)}, "
      f"{result.events_a} events, "
      f"{result.tool_calls_a} tools, "
      f"{result.retries_a} retries\n")
    w(f"  {b}: {_fmt_duration(result.duration_b)}, "
      f"{result.events_b} events, "
      f"{result.tool_calls_b} tools, "
      f"{result.retries_b} retries\n\n")


# ---------------------------------------------------------------------------
# Semantic diff
# ---------------------------------------------------------------------------

@dataclass
class SemanticDiffReport:
    session_a: str
    session_b: str
    # Metrics
    duration_a: float
    duration_b: float
    cost_a: float
    cost_b: float
    errors_a: int
    errors_b: int
    tool_calls_a: int
    tool_calls_b: int
    llm_requests_a: int
    llm_requests_b: int
    retries_a: int
    retries_b: int
    # File sets
    files_read_both: list[str] = field(default_factory=list)
    files_read_a_only: list[str] = field(default_factory=list)
    files_read_b_only: list[str] = field(default_factory=list)
    files_written_both: list[str] = field(default_factory=list)
    files_written_a_only: list[str] = field(default_factory=list)
    files_written_b_only: list[str] = field(default_factory=list)
    # Command sets
    cmds_both: list[str] = field(default_factory=list)
    cmds_a_only: list[str] = field(default_factory=list)
    cmds_b_only: list[str] = field(default_factory=list)
    # Eval scores (optional)
    eval_scores_a: dict = field(default_factory=dict)
    eval_scores_b: dict = field(default_factory=dict)
    # Verdict
    verdict: str = ""   # "A is better" | "B is better" | "inconclusive"


def semantic_diff(
    store: TraceStore,
    session_a: str,
    session_b: str,
    eval_config: str = ".agent-evals.yaml",
) -> SemanticDiffReport:
    """Compare two sessions at the outcome level."""
    from .cost import estimate_cost

    result_a = explain_session(store, session_a)
    result_b = explain_session(store, session_b)
    meta_a = store.load_meta(session_a)
    meta_b = store.load_meta(session_b)

    # Cost
    try:
        cost_a = estimate_cost(store, session_a).total_cost
    except Exception:
        cost_a = 0.0
    try:
        cost_b = estimate_cost(store, session_b).total_cost
    except Exception:
        cost_b = 0.0

    # Aggregate files and commands across all phases
    def _collect(result):
        reads: set[str] = set()
        writes: set[str] = set()
        cmds: set[str] = set()
        for p in result.phases:
            reads.update(p.files_read)
            writes.update(p.files_written)
            cmds.update(p.commands)
        return reads, writes, cmds

    reads_a, writes_a, cmds_a = _collect(result_a)
    reads_b, writes_b, cmds_b = _collect(result_b)

    # Eval scores
    eval_a: dict = {}
    eval_b: dict = {}
    try:
        from .eval import run_evals
        import os
        if os.path.exists(eval_config):
            eval_a = {r.scorer_name: r.score for r in run_evals(store, session_a, eval_config)}
            eval_b = {r.scorer_name: r.score for r in run_evals(store, session_b, eval_config)}
    except Exception:
        pass

    # Verdict: B is better if it has fewer errors, lower cost, shorter duration
    # and is not worse on any metric
    def _verdict() -> str:
        a_wins = 0
        b_wins = 0
        metrics = [
            (meta_a.errors, meta_b.errors, True),       # lower is better
            (cost_a, cost_b, True),
            (result_a.total_duration, result_b.total_duration, True),
            (result_a.total_retries, result_b.total_retries, True),
        ]
        for va, vb, lower_better in metrics:
            if lower_better:
                if va > vb:
                    b_wins += 1
                elif vb > va:
                    a_wins += 1
        if b_wins > 0 and a_wins == 0:
            return "B is better"
        if a_wins > 0 and b_wins == 0:
            return "A is better"
        return "inconclusive"

    return SemanticDiffReport(
        session_a=session_a,
        session_b=session_b,
        duration_a=result_a.total_duration,
        duration_b=result_b.total_duration,
        cost_a=cost_a,
        cost_b=cost_b,
        errors_a=meta_a.errors,
        errors_b=meta_b.errors,
        tool_calls_a=meta_a.tool_calls,
        tool_calls_b=meta_b.tool_calls,
        llm_requests_a=meta_a.llm_requests,
        llm_requests_b=meta_b.llm_requests,
        retries_a=result_a.total_retries,
        retries_b=result_b.total_retries,
        files_read_both=sorted(reads_a & reads_b),
        files_read_a_only=sorted(reads_a - reads_b),
        files_read_b_only=sorted(reads_b - reads_a),
        files_written_both=sorted(writes_a & writes_b),
        files_written_a_only=sorted(writes_a - writes_b),
        files_written_b_only=sorted(writes_b - writes_a),
        cmds_both=sorted(cmds_a & cmds_b),
        cmds_a_only=sorted(cmds_a - cmds_b),
        cmds_b_only=sorted(cmds_b - cmds_a),
        eval_scores_a=eval_a,
        eval_scores_b=eval_b,
        verdict=_verdict(),
    )


def _pct_change(a: float, b: float) -> str:
    if a == 0:
        return "n/a"
    pct = (b - a) / a * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.0f}%"


def format_semantic_diff(report: SemanticDiffReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    a = report.session_a[:12]
    b = report.session_b[:12]

    w(f"\nSemantic diff: {a} vs {b}\n")
    w("─" * 69 + "\n")
    w(f"  {'':30}  {'Session A':>12}  {'Session B':>12}  {'Change':>8}\n")
    w("─" * 69 + "\n")

    def _row(label: str, va, vb, fmt=str, lower_better: bool = True) -> None:
        change = _pct_change(float(va), float(vb)) if isinstance(va, (int, float)) else ""
        w(f"  {label:<30}  {fmt(va):>12}  {fmt(vb):>12}  {change:>8}\n")

    _row("Duration", _fmt_duration(report.duration_a), _fmt_duration(report.duration_b), fmt=str)
    _row("Cost", f"${report.cost_a:.4f}", f"${report.cost_b:.4f}", fmt=str)
    _row("Errors", report.errors_a, report.errors_b)
    _row("Tool calls", report.tool_calls_a, report.tool_calls_b)
    _row("LLM requests", report.llm_requests_a, report.llm_requests_b)
    _row("Retries", report.retries_a, report.retries_b)
    w("─" * 69 + "\n")

    def _file_rows(label: str, both: list, a_only: list, b_only: list) -> None:
        if both:
            w(f"  {label} (both)    {', '.join(both[:3])}{'...' if len(both)>3 else ''}\n")
        for f in a_only[:3]:
            w(f"  {label} (A only)  {f}\n")
        for f in b_only[:3]:
            w(f"  {label} (B only)  {f}\n")

    _file_rows("Files read", report.files_read_both, report.files_read_a_only, report.files_read_b_only)
    _file_rows("Files written", report.files_written_both, report.files_written_a_only, report.files_written_b_only)
    _file_rows("Commands", report.cmds_both, report.cmds_a_only, report.cmds_b_only)

    if report.eval_scores_a or report.eval_scores_b:
        w("─" * 69 + "\n")
        all_scorers = sorted(set(report.eval_scores_a) | set(report.eval_scores_b))
        for scorer in all_scorers:
            sa = report.eval_scores_a.get(scorer, "n/a")
            sb = report.eval_scores_b.get(scorer, "n/a")
            w(f"  Eval {scorer:<25}  {str(sa):>12}  {str(sb):>12}\n")

    w("─" * 69 + "\n")
    w(f"  Verdict: {report.verdict}\n\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_diff(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    id_a = store.find_session(args.session_a)
    if not id_a:
        sys.stderr.write(f"Session not found: {args.session_a}\n")
        return 1

    id_b = store.find_session(args.session_b)
    if not id_b:
        sys.stderr.write(f"Session not found: {args.session_b}\n")
        return 1

    if getattr(args, "semantic", False):
        eval_config = getattr(args, "eval_config", ".agent-evals.yaml") or ".agent-evals.yaml"
        report = semantic_diff(store, id_a, id_b, eval_config=eval_config)
        format_semantic_diff(report)
        return 0

    result = diff_sessions(store, id_a, id_b)
    format_diff(result)
    return 0
