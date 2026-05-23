"""Dataset auto-sampler: export worst/diverse/random/recent sessions as JSONL.

Surfaces the sessions most useful for building regression suites and eval
datasets, without manual inspection. Zero new dependencies.

Usage:
    agent-strace sample --strategy worst --n 20 --output regression.jsonl
    agent-strace sample --strategy diverse --n 10 --output diverse.jsonl
    agent-strace sample --strategy recent --n 5 --output recent.jsonl
    agent-strace sample --strategy random --n 15 --output random.jsonl
"""

from __future__ import annotations

import argparse
import json
import random as _random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Per-session scoring
# ---------------------------------------------------------------------------

@dataclass
class SessionScore:
    session_id: str
    started_at: float
    error_rate: float       # errors / max(tool_calls, 1)
    retry_rate: float       # retries / max(tool_calls, 1)
    blast_radius: int       # distinct files written
    cost_estimate: float    # estimated cost in dollars
    duration_s: float
    tool_calls: int
    # Composite "worst" score (higher = worse session)
    worst_score: float = 0.0


def _estimate_cost(events: list[TraceEvent]) -> float:
    """Rough cost estimate: $3/M input tokens, $15/M output tokens (Sonnet-class)."""
    total = 0.0
    for ev in events:
        if ev.event_type == EventType.LLM_RESPONSE:
            tokens = ev.data.get("total_tokens", 0) or 0
            # Assume 80/20 input/output split
            inp = int(tokens * 0.8)
            out = int(tokens * 0.2)
            total += (inp / 1_000_000) * 3.0 + (out / 1_000_000) * 15.0
    return total


def _score_session(
    session_id: str,
    meta: SessionMeta,
    events: list[TraceEvent],
) -> SessionScore:
    tool_calls = 0
    errors = 0
    retries = 0
    files_written: set[str] = set()
    prev_tool: str | None = None
    prev_count = 0

    for ev in events:
        if ev.event_type == EventType.TOOL_CALL:
            tool_calls += 1
            name = ev.data.get("tool_name", "")
            if name == prev_tool:
                prev_count += 1
                if prev_count >= 2:
                    retries += 1
            else:
                prev_tool = name
                prev_count = 1
            # Track writes
            if name.lower() in ("write", "edit", "create", "str_replace"):
                path = str(
                    ev.data.get("arguments", {}).get("file_path")
                    or ev.data.get("arguments", {}).get("path")
                    or ""
                )
                if path:
                    files_written.add(path)
        elif ev.event_type == EventType.FILE_WRITE:
            path = ev.data.get("path") or ev.data.get("file_path") or ""
            if path:
                files_written.add(str(path))
        elif ev.event_type == EventType.ERROR:
            errors += 1

    denom = max(tool_calls, 1)
    error_rate = errors / denom
    retry_rate = retries / denom
    blast_radius = len(files_written)
    cost = _estimate_cost(events)
    duration_s = (events[-1].timestamp - events[0].timestamp) if len(events) >= 2 else 0.0

    # Composite worst score: weighted sum of normalised dimensions
    # Weights chosen so a session with many errors + high retry + high cost
    # ranks clearly above a clean session.
    worst_score = (
        error_rate * 0.35
        + retry_rate * 0.30
        + min(blast_radius / 20.0, 1.0) * 0.15
        + min(cost / 5.0, 1.0) * 0.20
    )

    return SessionScore(
        session_id=session_id,
        started_at=meta.started_at,
        error_rate=error_rate,
        retry_rate=retry_rate,
        blast_radius=blast_radius,
        cost_estimate=cost,
        duration_s=duration_s,
        tool_calls=tool_calls,
        worst_score=worst_score,
    )


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def _sample_worst(scores: list[SessionScore], n: int) -> list[SessionScore]:
    """Return the N sessions with the highest worst_score."""
    return sorted(scores, key=lambda s: s.worst_score, reverse=True)[:n]


def _sample_diverse(scores: list[SessionScore], n: int) -> list[SessionScore]:
    """Return N sessions that maximise variety across dimensions.

    Uses a greedy max-min distance approach: start with the worst session,
    then repeatedly pick the session most different from those already chosen.
    Distance is Euclidean in the (error_rate, retry_rate, blast_radius_norm,
    cost_norm) feature space.
    """
    if not scores or n <= 0:
        return []
    if len(scores) <= n:
        return list(scores)

    def _features(s: SessionScore) -> tuple[float, ...]:
        return (
            s.error_rate,
            s.retry_rate,
            min(s.blast_radius / 20.0, 1.0),
            min(s.cost_estimate / 5.0, 1.0),
        )

    def _dist(a: tuple, b: tuple) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    remaining = list(scores)
    # Seed with the worst session
    chosen = [max(remaining, key=lambda s: s.worst_score)]
    remaining = [s for s in remaining if s.session_id != chosen[0].session_id]

    while len(chosen) < n and remaining:
        chosen_feats = [_features(c) for c in chosen]
        # Pick the session with the maximum minimum distance to any chosen session
        best = max(
            remaining,
            key=lambda s: min(_dist(_features(s), cf) for cf in chosen_feats),
        )
        chosen.append(best)
        remaining = [s for s in remaining if s.session_id != best.session_id]

    return chosen


def _sample_random(scores: list[SessionScore], n: int, seed: int | None = None) -> list[SessionScore]:
    """Return N sessions chosen uniformly at random."""
    pool = list(scores)
    if seed is not None:
        _random.seed(seed)
    _random.shuffle(pool)
    return pool[:n]


def _sample_recent(scores: list[SessionScore], n: int) -> list[SessionScore]:
    """Return the N most recent sessions."""
    return sorted(scores, key=lambda s: s.started_at, reverse=True)[:n]


STRATEGIES = {
    "worst": _sample_worst,
    "diverse": _sample_diverse,
    "random": _sample_random,
    "recent": _sample_recent,
}


# ---------------------------------------------------------------------------
# JSONL export
# ---------------------------------------------------------------------------

def _session_to_jsonl_record(
    meta: SessionMeta,
    events: list[TraceEvent],
    score: SessionScore,
) -> dict:
    """Serialise a session to a single JSONL record."""
    return {
        "session_id": meta.session_id,
        "started_at": meta.started_at,
        "agent_name": meta.agent_name,
        "command": meta.command,
        "score": {
            "error_rate": round(score.error_rate, 4),
            "retry_rate": round(score.retry_rate, 4),
            "blast_radius": score.blast_radius,
            "cost_estimate": round(score.cost_estimate, 6),
            "worst_score": round(score.worst_score, 4),
        },
        "events": [json.loads(ev.to_json()) for ev in events],
    }


def run_sample(
    store: TraceStore,
    strategy: str,
    n: int,
    output_path: str,
    deduplicate: bool = False,
    seed: int | None = None,
    out: TextIO = sys.stdout,
) -> int:
    """Run the sampler and write JSONL output. Returns exit code."""
    sessions = store.list_sessions()
    if not sessions:
        out.write("No sessions found.\n")
        return 1

    out.write(f"Sampling {n} sessions by strategy '{strategy}'...\n")
    out.write(f"Scoring {len(sessions)} session(s)...\n")

    scores: list[SessionScore] = []
    seen_tool_sequences: set[str] = set()

    for meta in sessions:
        try:
            events = store.load_events(meta.session_id)
        except Exception:
            continue

        if deduplicate:
            # Fingerprint: sorted tuple of (tool_name, command) pairs
            seq = tuple(sorted(
                (ev.data.get("tool_name", ""), str(ev.data.get("arguments", {}).get("command", "")))
                for ev in events
                if ev.event_type == EventType.TOOL_CALL
            ))
            key = str(seq)
            if key in seen_tool_sequences:
                continue
            seen_tool_sequences.add(key)

        scores.append(_score_session(meta.session_id, meta, events))

    if not scores:
        out.write("No sessions available after filtering.\n")
        return 1

    # Apply strategy
    sampler = STRATEGIES.get(strategy)
    if sampler is None:
        out.write(f"Unknown strategy: {strategy!r}. Choose from: {', '.join(STRATEGIES)}\n")
        return 1

    if strategy == "random":
        selected = _sample_random(scores, n, seed=seed)
    else:
        selected = sampler(scores, n)

    # Write JSONL
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(p, "w", encoding="utf-8") as f:
        for score in selected:
            try:
                meta = store.load_meta(score.session_id)
                events = store.load_events(score.session_id)
                record = _session_to_jsonl_record(meta, events, score)
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
            except Exception:
                continue

    out.write(f"\nExported {written} session(s) to {output_path}\n")
    if selected:
        out.write("\nCriteria used:\n")
        if strategy == "worst":
            out.write("  - highest error rate\n")
            out.write("  - highest retry rate\n")
            out.write("  - highest blast radius\n")
            out.write("  - highest estimated cost\n")
        elif strategy == "diverse":
            out.write("  - maximum behavioral variety (greedy max-min distance)\n")
        elif strategy == "random":
            out.write("  - uniform random sample\n")
        elif strategy == "recent":
            out.write("  - most recent sessions by start time\n")

    return 0


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    return run_sample(
        store=store,
        strategy=getattr(args, "strategy", "worst"),
        n=getattr(args, "n", 20),
        output_path=getattr(args, "output", "sample.jsonl"),
        deduplicate=getattr(args, "deduplicate", False),
        seed=getattr(args, "seed", None),
    )
