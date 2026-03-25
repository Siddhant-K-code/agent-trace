"""Session diff: structural behavioral comparison of two sessions.

Compares two sessions by their phase structure (from explain), finds the
divergence point, and reports differences in files touched, commands run,
outcomes, duration, and cost.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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

    result = diff_sessions(store, id_a, id_b)
    format_diff(result)
    return 0
