"""Session-to-session regression testing: agent-strace compare.

Wraps diff.compare_sessions with a first-class CLI workflow for regression
testing: compare two sessions directly, compare the last N sessions with a
given tag, or re-run a session's original prompt and compare live.

Decision divergence is computed as edit distance on decision event text —
no LLM call required.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import TextIO

from .diff import compare_sessions, format_compare, CompareReport
from .models import EventType, SessionMeta
from .store import TraceStore


# ---------------------------------------------------------------------------
# Decision divergence (edit distance on decision text)
# ---------------------------------------------------------------------------

def _decision_texts(store: TraceStore, session_id: str) -> list[str]:
    """Extract decision event text from a session."""
    try:
        events = store.load_events(session_id)
    except Exception:
        return []
    texts: list[str] = []
    for ev in events:
        if ev.event_type == EventType.DECISION:
            text = (
                ev.data.get("text")
                or ev.data.get("content")
                or ev.data.get("reasoning")
                or ""
            )
            if text:
                texts.append(str(text))
    return texts


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two lists of strings (token-level)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def decision_divergence(store: TraceStore, session_a: str, session_b: str) -> int:
    """Number of decision events where reasoning text differs significantly."""
    texts_a = _decision_texts(store, session_a)
    texts_b = _decision_texts(store, session_b)
    return _edit_distance(texts_a, texts_b)


# ---------------------------------------------------------------------------
# JSON serialisation of CompareReport
# ---------------------------------------------------------------------------

def _report_to_dict(report: CompareReport, divergence: int) -> dict:
    return {
        "session_a": report.session_a,
        "session_b": report.session_b,
        "label_a": report.label_a,
        "label_b": report.label_b,
        "duration_a": report.duration_a,
        "duration_b": report.duration_b,
        "cost_a": report.cost_a,
        "cost_b": report.cost_b,
        "tool_calls_a": report.tool_calls_a,
        "tool_calls_b": report.tool_calls_b,
        "files_modified_a": report.files_modified_a,
        "files_modified_b": report.files_modified_b,
        "errors_a": report.errors_a,
        "errors_b": report.errors_b,
        "redundant_reads_a": report.redundant_reads_a,
        "redundant_reads_b": report.redundant_reads_b,
        "decision_divergence": divergence,
        "divergence_points": [
            {"step": step, "description_a": da, "description_b": db}
            for step, da, db in report.divergence_points
        ],
        "verdict": report.verdict,
    }


# ---------------------------------------------------------------------------
# Tag-based session lookup
# ---------------------------------------------------------------------------

def _sessions_by_tag(store: TraceStore, tag: str, last: int = 2) -> list[str]:
    """Return the last N session IDs whose agent_name or command contains tag."""
    all_sessions = store.list_sessions()
    matched = [
        s for s in all_sessions
        if tag.lower() in (s.agent_name or "").lower()
        or tag.lower() in (s.command or "").lower()
    ]
    # list_sessions returns newest-first
    return [s.session_id for s in matched[:last]]


# ---------------------------------------------------------------------------
# --rerun support
# ---------------------------------------------------------------------------

def _get_user_prompt(store: TraceStore, session_id: str) -> str | None:
    """Extract the original user prompt from a session's events."""
    try:
        events = store.load_events(session_id)
    except Exception:
        return None
    for ev in events:
        if ev.event_type == EventType.USER_PROMPT:
            return (
                ev.data.get("content")
                or ev.data.get("text")
                or ev.data.get("prompt")
                or None
            )
    return None


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    fmt = getattr(args, "format", "text")
    tag = getattr(args, "tag", None)
    last = getattr(args, "last", 2)
    rerun = getattr(args, "rerun", False)
    model = getattr(args, "model", None)

    # Resolve the two session IDs
    session_a: str | None = None
    session_b: str | None = None

    if tag:
        ids = _sessions_by_tag(store, tag, last=last)
        if len(ids) < 2:
            sys.stderr.write(
                f"Need at least 2 sessions tagged {tag!r}, found {len(ids)}.\n"
            )
            return 1
        session_a, session_b = ids[1], ids[0]  # older first, newer second

    else:
        raw_a = getattr(args, "session_id_a", None)
        raw_b = getattr(args, "session_id_b", None)

        if not raw_a:
            sys.stderr.write(
                "Usage: agent-strace compare <session-id-a> <session-id-b>\n"
                "       agent-strace compare <session-id> --rerun [--model MODEL]\n"
                "       agent-strace compare --tag TAG [--last N]\n"
            )
            return 1

        full_a = store.find_session(raw_a)
        if not full_a:
            sys.stderr.write(f"Session not found: {raw_a}\n")
            return 1
        session_a = full_a

        if rerun:
            # --rerun: re-execute the original prompt and compare live
            prompt = _get_user_prompt(store, session_a)
            if not prompt:
                sys.stderr.write(
                    f"Session {session_a[:12]} has no stored user_prompt. "
                    "Cannot --rerun without a recorded prompt.\n"
                )
                return 1
            sys.stderr.write(
                f"[compare] --rerun is not yet automated. "
                f"Original prompt for {session_a[:12]}:\n\n{prompt}\n\n"
                "Run the agent with this prompt, then compare the two sessions manually.\n"
            )
            return 1

        if not raw_b:
            sys.stderr.write("Provide two session IDs or use --tag / --rerun.\n")
            return 1

        full_b = store.find_session(raw_b)
        if not full_b:
            sys.stderr.write(f"Session not found: {raw_b}\n")
            return 1
        session_b = full_b

    # Run comparison
    try:
        report = compare_sessions(store, session_a, session_b)
    except Exception as exc:
        sys.stderr.write(f"[compare] Failed: {exc}\n")
        return 1

    divergence = decision_divergence(store, session_a, session_b)

    if fmt == "json":
        sys.stdout.write(json.dumps(_report_to_dict(report, divergence), indent=2) + "\n")
    else:
        # Text: use existing format_compare, then append decision divergence
        format_compare(report, sys.stdout)
        sys.stdout.write(f"Decision divergence:  {divergence} point(s)\n\n")

    return 0
