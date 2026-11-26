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

from .cost import session_cost
from .event_data import (
    object_field,
    require_object_event_data,
    require_usable_timestamps,
    str_field,
)
from .failures import count_session_errors
from .fileutil import write_atomic
from .models import EventType, SessionMeta, TraceEvent, usable_text, usable_time
from .store import TraceStore


# ---------------------------------------------------------------------------
# Per-session scoring
# ---------------------------------------------------------------------------

@dataclass
class SessionScore:
    session_id: str
    # The session start, or None when it is not a usable time.
    started_at: float | None
    error_rate: float       # errors / max(tool_calls, 1)
    retry_rate: float       # retries / max(tool_calls, 1)
    blast_radius: int       # distinct files written
    # Same as `agent-strace cost`; None when that command refuses the session.
    cost_estimate: float | None
    duration_s: float
    tool_calls: int
    # Composite "worst" score (higher = worse session)
    worst_score: float = 0.0


@dataclass
class _PreparedSession:
    meta: SessionMeta
    events: list[TraceEvent]


# _score_session reads the data of every event of these types. Its cost is
# checked by session_cost, which reports a session it refuses as unpriced.
SCORE_DATA_EVENT_TYPES = frozenset({EventType.TOOL_CALL, EventType.FILE_WRITE})


def _score_session(
    session_id: str,
    meta: SessionMeta,
    events: list[TraceEvent],
) -> SessionScore:
    tool_calls = 0
    errors = count_session_errors(events)
    retries = 0
    files_written: set[str] = set()
    prev_tool: str | None = None
    prev_count = 0

    for ev in events:
        if ev.event_type == EventType.TOOL_CALL:
            tool_calls += 1
            name = str_field(ev.data, "tool_name", "")
            if name == prev_tool:
                prev_count += 1
                if prev_count >= 2:
                    retries += 1
            else:
                prev_tool = name
                prev_count = 1
            # Track writes
            if name.lower() in ("write", "edit", "create", "str_replace"):
                args = object_field(ev.data, "arguments", {})
                path = str(args.get("file_path") or args.get("path") or "")
                if path:
                    files_written.add(path)
        elif ev.event_type == EventType.FILE_WRITE:
            path = ev.data.get("path") or ev.data.get("file_path") or ""
            if path:
                files_written.add(str(path))

    denom = max(tool_calls, 1)
    error_rate = errors / denom
    retry_rate = retries / denom
    blast_radius = len(files_written)
    try:
        cost: float | None = session_cost(events, session_id=session_id)
    except ValueError:
        cost = None
    duration_s = (events[-1].timestamp - events[0].timestamp) if len(events) >= 2 else 0.0

    # Composite worst score: weighted sum of normalised dimensions
    # Weights chosen so a session with many errors + high retry + high cost
    # ranks clearly above a clean session. A session with no cost has no
    # cost term.
    worst_score = (
        error_rate * 0.35
        + retry_rate * 0.30
        + min(blast_radius / 20.0, 1.0) * 0.15
        + (min(cost / 5.0, 1.0) * 0.20 if cost is not None else 0.0)
    )

    return SessionScore(
        session_id=session_id,
        started_at=usable_time(meta.started_at),
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
    cost_norm) feature space. A session with no cost is compared on the
    other dimensions only.
    """
    if not scores or n <= 0:
        return []
    if len(scores) <= n:
        return list(scores)

    def _features(s: SessionScore) -> tuple[float | None, ...]:
        return (
            s.error_rate,
            s.retry_rate,
            min(s.blast_radius / 20.0, 1.0),
            None if s.cost_estimate is None else min(s.cost_estimate / 5.0, 1.0),
        )

    def _dist(a: tuple, b: tuple) -> float:
        return sum(
            (x - y) ** 2 for x, y in zip(a, b) if x is not None and y is not None
        ) ** 0.5

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
    """Return the N most recent sessions.

    A start that is not a usable time sorts as 0, so it is the oldest.
    """
    return sorted(
        scores,
        key=lambda s: 0.0 if s.started_at is None else s.started_at,
        reverse=True,
    )[:n]


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
    """Serialise a session to a single JSONL record.

    started_at is the stored start, or null when it is not a usable time.
    """
    return {
        "session_id": meta.session_id,
        "started_at": meta.started_at if usable_time(meta.started_at) is not None else None,
        "agent_name": usable_text(meta.agent_name),
        "command": usable_text(meta.command),
        "score": {
            "error_rate": round(score.error_rate, 4),
            "retry_rate": round(score.retry_rate, 4),
            "blast_radius": score.blast_radius,
            "cost_estimate": (
                None if score.cost_estimate is None else round(score.cost_estimate, 6)
            ),
            "worst_score": round(score.worst_score, 4),
        },
        "events": [json.loads(ev.to_json()) for ev in events],
    }


def _tool_sequence_key(events: list[TraceEvent]) -> str:
    """Fingerprint a session by the sorted (tool_name, command) pairs of its tool calls."""
    seq = tuple(sorted(
        (
            str_field(ev.data, "tool_name", ""),
            str(object_field(ev.data, "arguments", {}).get("command", "")),
        )
        for ev in events
        if ev.event_type == EventType.TOOL_CALL
    ))
    return str(seq)


def _prepare_sessions(
    store: TraceStore,
    sessions: list[SessionMeta],
    deduplicate: bool,
) -> tuple[list[_PreparedSession], int]:
    """Load and validate the stable session snapshot used for scoring and export.

    A session that fails to load is counted as unreadable and skipped. With
    deduplicate, a repeated tool sequence is not scored, so only its tool_call
    data is read.

    Raise UnusableTimestampError when a scored session's first or last event
    timestamp is not a Unix time in seconds. Scoring reads only those two,
    for the duration, and only when the session has 2 or more events.
    """
    prepared: list[_PreparedSession] = []
    unreadable = 0
    seen_tool_sequences: set[str] = set()
    for meta in sessions:
        try:
            events = store.load_events(meta.session_id)
        except (OSError, ValueError):
            unreadable += 1
            continue
        if deduplicate:
            require_object_event_data(
                events, event_types={EventType.TOOL_CALL}, session_id=meta.session_id
            )
            key = _tool_sequence_key(events)
            if key in seen_tool_sequences:
                continue
            seen_tool_sequences.add(key)
        require_object_event_data(
            events, event_types=SCORE_DATA_EVENT_TYPES, session_id=meta.session_id
        )
        if len(events) >= 2:
            require_usable_timestamps(
                events, indexes={0, len(events) - 1}, session_id=meta.session_id
            )
        prepared.append(_PreparedSession(meta=meta, events=events))
    return prepared, unreadable


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

    prepared, unreadable = _prepare_sessions(store, sessions, deduplicate)

    out.write(f"Sampling {n} sessions by strategy '{strategy}'...\n")
    if unreadable:
        out.write(f"Skipped {unreadable} unreadable session(s).\n")
    out.write(f"Scoring {len(prepared)} session(s)...\n")

    if not prepared:
        out.write("No sessions available after filtering.\n")
        return 1

    scores = [
        _score_session(item.meta.session_id, item.meta, item.events)
        for item in prepared
    ]

    # Apply strategy
    sampler = STRATEGIES.get(strategy)
    if sampler is None:
        out.write(f"Unknown strategy: {strategy!r}. Choose from: {', '.join(STRATEGIES)}\n")
        return 1

    if strategy == "random":
        selected = _sample_random(scores, n, seed=seed)
    else:
        selected = sampler(scores, n)

    prepared_by_id = {item.meta.session_id: item for item in prepared}
    lines: list[str] = []
    for score in selected:
        item = prepared_by_id[score.session_id]
        try:
            record = _session_to_jsonl_record(item.meta, item.events, score)
            lines.append(json.dumps(record, separators=(",", ":")) + "\n")
        except (TypeError, ValueError, OverflowError, RecursionError):
            out.write(
                f"Error: could not serialize sampled session {score.session_id}.\n"
            )
            return 1

    p = Path(output_path)
    try:
        write_atomic(p, "".join(lines))
    except (OSError, ValueError) as exc:
        detail = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
        out.write(f"Error: cannot write {output_path}: {detail}\n")
        return 1

    out.write(f"\nExported {len(selected)} session(s) to {output_path}\n")
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
