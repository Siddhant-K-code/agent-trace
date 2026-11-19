"""Per-session baseline and anomaly detection.

Builds a statistical baseline (mean, stddev) from historical sessions and
flags individual sessions that deviate beyond a configurable sigma threshold.

Usage:
    # Build / update baseline from recent sessions
    agent-strace baseline update [--since 30d] [--output .agent-traces/baseline.json]

    # Check a single session against the baseline
    agent-strace baseline check [session-id] [--baseline FILE] [--sigma 2.0]

    # Show baseline stats
    agent-strace baseline show [--baseline FILE]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

from .cost import estimate_cost
from .event_data import require_object_event_data, require_usable_timestamps
from .explain import PHASE_DATA_EVENT_TYPES
from .failures import count_session_errors
from .models import EventType, TraceEvent, usable_time
from .store import TraceStore


DEFAULT_BASELINE_PATH = ".agent-traces/baseline.json"
DEFAULT_SIGMA = 2.0


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _z_score(value: float, mean: float, stddev: float) -> float:
    """Return z-score; 0 when stddev is 0 (no variance in baseline)."""
    if stddev == 0:
        return 0.0
    return abs(value - mean) / stddev


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MetricStats:
    mean: float = 0.0
    stddev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0


@dataclass
class BaselineProfile:
    """Statistical profile built from N historical sessions."""
    created_at: float = field(default_factory=time.time)
    session_count: int = 0
    cost_usd: MetricStats = field(default_factory=MetricStats)
    tool_calls: MetricStats = field(default_factory=MetricStats)
    duration_ms: MetricStats = field(default_factory=MetricStats)
    error_rate: MetricStats = field(default_factory=MetricStats)   # errors / tool_calls
    llm_requests: MetricStats = field(default_factory=MetricStats)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "BaselineProfile":
        d = json.loads(text)
        profile = cls()
        profile.created_at = d.get("created_at", 0.0)
        profile.session_count = d.get("session_count", 0)
        for metric in ("cost_usd", "tool_calls", "duration_ms", "error_rate", "llm_requests"):
            raw = d.get(metric, {})
            setattr(profile, metric, MetricStats(**{
                k: raw.get(k, 0.0) if k != "count" else raw.get(k, 0)
                for k in ("mean", "stddev", "min", "max", "count")
            }))
        return profile


@dataclass
class AnomalyResult:
    session_id: str
    anomalous: bool
    deviations: dict[str, float] = field(default_factory=dict)   # metric → z-score
    sigma_threshold: float = DEFAULT_SIGMA
    # The session has no end time (still running, crashed, or killed), so it
    # was scored on the events recorded so far.
    open_session: bool = False
    # False when the session could not be measured at all. Such a result is
    # neither OK nor anomalous, and ``baseline check`` exits 1 for it.
    scored: bool = True
    reason: str = ""

    def format(self, out: TextIO = sys.stdout) -> None:
        if not self.scored:
            out.write(f"\nBaseline check: {self.session_id[:12]}  [UNSCORED]\n\n")
            out.write(f"  Could not score this session: {self.reason}\n")
            out.write("  This is not a pass.\n\n")
            return
        status = "ANOMALOUS" if self.anomalous else "OK"
        suffix = "  open session" if self.open_session else ""
        out.write(f"\nBaseline check: {self.session_id[:12]}  [{status}]{suffix}\n\n")
        if self.open_session:
            out.write(
                "  No end time recorded (still running, crashed, or killed).\n"
                "  Scored on the events so far: tool calls, LLM requests,\n"
                "  duration and cost count only when above the mean.\n\n"
            )
        if not self.deviations:
            out.write("  No metrics to compare.\n")
            return
        out.write(f"  {'Metric':<18} {'z-score':>8}  {'Status'}\n")
        out.write(f"  {'-'*18} {'-'*8}  {'-'*10}\n")
        for metric, z in sorted(self.deviations.items()):
            flag = "⚠ ANOMALY" if z > self.sigma_threshold else "ok"
            out.write(f"  {metric:<18} {z:>8.2f}  {flag}\n")
        out.write("\n")


# ---------------------------------------------------------------------------
# Building a baseline
# ---------------------------------------------------------------------------

def _session_cost(
    store: TraceStore, session_id: str, events: list[TraceEvent]
) -> float | None:
    """Return the session's cost as ``agent-strace cost`` with no flags reports it.

    This is ``estimate_cost`` with its default model, so sessions that record
    token usage (such as Codex model requests) are priced from that usage and
    other sessions from event sizes. Returns None when the session cannot be
    priced: its events cannot be read, a tool call or prompt has data that
    is not a JSON object, or an event timestamp is not a Unix time in
    seconds. ``agent-strace cost`` refuses the last two. Other errors
    propagate so that a bug cannot pass for a $0 session.
    """
    try:
        require_object_event_data(
            events, event_types=PHASE_DATA_EVENT_TYPES, session_id=session_id
        )
        require_usable_timestamps(events, session_id=session_id)
        return estimate_cost(store, session_id).total_cost
    except (OSError, ValueError):
        return None


# For an open session these only grow as it runs, so the values so far are
# lower bounds. error_rate is a ratio and can move either way.
LOWER_BOUND_METRICS = frozenset({"tool_calls", "llm_requests", "duration_ms", "cost_usd"})


def _open_duration_ms(started_at: object, events: list[TraceEvent]) -> float | None:
    """Milliseconds from the session start to its last event, or None if unknown.

    Event stamps that are not a Unix time in seconds are skipped. An unusable
    started_at gives None.
    """
    stamps = [e.timestamp for e in events if usable_time(e.timestamp) is not None]
    if not stamps or usable_time(started_at) is None:
        return None
    duration = (max(stamps) - started_at) * 1000
    return duration if duration >= 0 else None


def _measure_session(
    store: TraceStore, session_id: str, *, allow_open: bool = False
) -> tuple[dict[str, float] | None, bool, str]:
    """Return ``(metrics, open_session, reason)`` for one session.

    ``metrics`` is None when the session cannot be measured, and ``reason``
    says why. A session with no ``ended_at`` is measured only when
    *allow_open* is set: its tool calls and LLM requests are the larger of
    the metadata counter and the matching events, since a recorder killed
    mid-run may never write its counters, and its duration runs to its last
    event.

    ``cost_usd`` is left out when the session cannot be priced, so it adds no
    cost sample to a baseline and ``check`` does not compare its cost, rather
    than counting it as $0. A session with nothing to price still costs $0.
    """
    try:
        meta = store.load_meta(session_id)
    except Exception:
        return None, False, "its metadata cannot be read"
    if not meta:
        return None, False, "it has no metadata"
    open_session = not meta.ended_at
    if open_session and not allow_open:
        return None, True, "it has no end time"
    try:
        events: list[TraceEvent] | None = store.load_events(session_id)
    except Exception:
        events = None
    errors = count_session_errors(events or [])
    try:
        tool_calls = float(meta.tool_calls)
        llm_requests = float(meta.llm_requests)
        duration_ms: float | None = float(meta.total_duration_ms or 0.0)
        error_rate = errors / max(meta.tool_calls, 1)
    except (TypeError, ValueError):
        # hand-edited metadata with non-numeric counters
        return None, open_session, "its metadata has non-numeric counters"
    if open_session:
        duration_ms = None
        if events is not None:
            tool_calls = max(tool_calls, float(sum(
                1 for e in events if e.event_type == EventType.TOOL_CALL
            )))
            llm_requests = max(llm_requests, float(sum(
                1 for e in events if e.event_type == EventType.LLM_REQUEST
            )))
            duration_ms = _open_duration_ms(meta.started_at, events)
        error_rate = errors / max(tool_calls, 1)
    metrics = {
        "tool_calls": tool_calls,
        "error_rate": error_rate,
        "llm_requests": llm_requests,
    }
    if duration_ms is not None:
        metrics["duration_ms"] = duration_ms
    if events is not None:
        cost_usd = _session_cost(store, session_id, events)
        if cost_usd is not None:
            metrics["cost_usd"] = cost_usd
    return metrics, open_session, ""


def _session_metrics(store: TraceStore, session_id: str) -> dict[str, float] | None:
    """Extract scalar metrics from a single ended session. Returns None if unusable."""
    metrics, _open, _reason = _measure_session(store, session_id)
    return metrics


def build_baseline(store: TraceStore, since_days: float = 30.0) -> BaselineProfile:
    """Build a baseline profile from sessions completed in the last *since_days* days."""
    cutoff = time.time() - since_days * 86400
    session_ids = store.list_sessions()

    samples: dict[str, list[float]] = {
        "cost_usd": [], "tool_calls": [], "duration_ms": [],
        "error_rate": [], "llm_requests": [],
    }

    session_count = 0
    for meta in session_ids:
        try:
            in_window = bool(meta.ended_at) and meta.started_at >= cutoff
        except TypeError:
            continue  # hand-edited metadata with a non-numeric start time
        if not in_window:
            continue
        m = _session_metrics(store, meta.session_id)
        if m is None:
            continue
        session_count += 1
        for key, val in m.items():
            samples[key].append(val)

    profile = BaselineProfile(session_count=session_count)
    for metric, values in samples.items():
        if not values:
            continue
        mean = _mean(values)
        sd = _stddev(values, mean)
        stats = MetricStats(
            mean=mean,
            stddev=sd,
            min=min(values),
            max=max(values),
            count=len(values),
        )
        setattr(profile, metric, stats)

    return profile


# ---------------------------------------------------------------------------
# Checking a session against the baseline
# ---------------------------------------------------------------------------

def check_session(
    store: TraceStore,
    session_id: str,
    profile: BaselineProfile,
    sigma: float = DEFAULT_SIGMA,
) -> AnomalyResult:
    """Compare a session's metrics against the baseline profile.

    A session with no end time is scored on what it has recorded so far.
    Its lower-bound metrics count only when above the mean, because being
    below the mean says nothing about a session that has not finished. A
    session that cannot be measured gives a result with ``scored=False``.
    """
    metrics, open_session, reason = _measure_session(
        store, session_id, allow_open=True
    )
    if metrics is None:
        return AnomalyResult(session_id=session_id, anomalous=False,
                             sigma_threshold=sigma, open_session=open_session,
                             scored=False, reason=reason)

    deviations: dict[str, float] = {}
    for metric, value in metrics.items():
        stats: MetricStats = getattr(profile, metric)
        if stats.count < 3:
            continue  # not enough data to be meaningful
        if open_session and metric in LOWER_BOUND_METRICS and value <= stats.mean:
            deviations[metric] = 0.0
            continue
        z = _z_score(value, stats.mean, stats.stddev)
        deviations[metric] = z

    anomalous = any(z > sigma for z in deviations.values())
    return AnomalyResult(
        session_id=session_id,
        anomalous=anomalous,
        deviations=deviations,
        sigma_threshold=sigma,
        open_session=open_session,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_baseline(profile: BaselineProfile, path: str = DEFAULT_BASELINE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(profile.to_json(), encoding="utf-8", newline="\n")


def load_baseline(path: str = DEFAULT_BASELINE_PATH) -> BaselineProfile | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return BaselineProfile.from_json(p.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_baseline(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    sub = getattr(args, "baseline_cmd", None)

    if sub == "update":
        since_days = float(getattr(args, "since_days", 30))
        output = getattr(args, "output", DEFAULT_BASELINE_PATH)
        profile = build_baseline(store, since_days=since_days)
        save_baseline(profile, output)
        sys.stdout.write(
            f"Baseline updated: {profile.session_count} sessions, "
            f"saved to {output}\n"
        )
        return 0

    elif sub == "check":
        session_id = getattr(args, "session_id", None)
        if not session_id:
            session_id = store.get_latest_session_id()
        if not session_id:
            sys.stderr.write("No sessions found.\n")
            return 1
        full_id = store.find_session(session_id)
        if not full_id:
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1

        baseline_path = getattr(args, "baseline_path", DEFAULT_BASELINE_PATH)
        sigma = float(getattr(args, "sigma", DEFAULT_SIGMA))
        profile = load_baseline(baseline_path)
        if not profile:
            sys.stderr.write(f"No baseline found at {baseline_path}. "
                             f"Run: agent-strace baseline update\n")
            return 1

        result = check_session(store, full_id, profile, sigma=sigma)
        result.format()
        return 1 if result.anomalous or not result.scored else 0

    elif sub == "show":
        baseline_path = getattr(args, "baseline_path", DEFAULT_BASELINE_PATH)
        profile = load_baseline(baseline_path)
        if not profile:
            sys.stderr.write(f"No baseline found at {baseline_path}\n")
            return 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(profile.created_at))
        sys.stdout.write(f"\nBaseline ({profile.session_count} sessions, built {ts})\n\n")
        sys.stdout.write(f"  {'Metric':<18} {'Mean':>10} {'Stddev':>10} {'Min':>10} {'Max':>10}\n")
        sys.stdout.write(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")
        for metric in ("cost_usd", "tool_calls", "duration_ms", "error_rate", "llm_requests"):
            s: MetricStats = getattr(profile, metric)
            sys.stdout.write(
                f"  {metric:<18} {s.mean:>10.3f} {s.stddev:>10.3f} "
                f"{s.min:>10.3f} {s.max:>10.3f}\n"
            )
        sys.stdout.write("\n")
        return 0

    else:
        sys.stderr.write("Usage: agent-strace baseline <update|check|show>\n")
        return 1
