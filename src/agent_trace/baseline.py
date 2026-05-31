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

from .models import EventType, TraceEvent
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

    def format(self, out: TextIO = sys.stdout) -> None:
        status = "ANOMALOUS" if self.anomalous else "OK"
        out.write(f"\nBaseline check: {self.session_id[:12]}  [{status}]\n\n")
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

def _session_metrics(store: TraceStore, session_id: str) -> dict[str, float] | None:
    """Extract scalar metrics from a single session. Returns None if unusable."""
    try:
        meta = store.load_meta(session_id)
    except Exception:
        return None
    if not meta or not meta.ended_at:
        return None
    try:
        events = store.load_events(session_id)
    except Exception:
        events = []
    errors = sum(1 for e in events if e.event_type == EventType.ERROR)
    tool_calls = max(meta.tool_calls, 1)

    # Compute cost from token events if available (lazy import to avoid circular deps)
    cost_usd = 0.0
    try:
        from .cost import compute_cost
        cost_usd = compute_cost(events) or 0.0
    except Exception:
        pass

    return {
        "cost_usd": cost_usd,
        "tool_calls": float(meta.tool_calls),
        "duration_ms": meta.total_duration_ms or 0.0,
        "error_rate": errors / tool_calls,
        "llm_requests": float(meta.llm_requests),
    }


def build_baseline(store: TraceStore, since_days: float = 30.0) -> BaselineProfile:
    """Build a baseline profile from sessions completed in the last *since_days* days."""
    cutoff = time.time() - since_days * 86400
    session_ids = store.list_sessions()

    samples: dict[str, list[float]] = {
        "cost_usd": [], "tool_calls": [], "duration_ms": [],
        "error_rate": [], "llm_requests": [],
    }

    for meta in session_ids:
        try:
            if not meta or not meta.ended_at or meta.started_at < cutoff:
                continue
            m = _session_metrics(store, meta.session_id)
            if m is None:
                continue
            for key, val in m.items():
                samples[key].append(val)
        except Exception:
            continue

    profile = BaselineProfile(session_count=len(samples["cost_usd"]))
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
    """Compare a session's metrics against the baseline profile."""
    metrics = _session_metrics(store, session_id)
    if metrics is None:
        return AnomalyResult(session_id=session_id, anomalous=False,
                             sigma_threshold=sigma)

    deviations: dict[str, float] = {}
    for metric, value in metrics.items():
        stats: MetricStats = getattr(profile, metric)
        if stats.count < 3:
            continue  # not enough data to be meaningful
        z = _z_score(value, stats.mean, stats.stddev)
        deviations[metric] = z

    anomalous = any(z > sigma for z in deviations.values())
    return AnomalyResult(
        session_id=session_id,
        anomalous=anomalous,
        deviations=deviations,
        sigma_threshold=sigma,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_baseline(profile: BaselineProfile, path: str = DEFAULT_BASELINE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(profile.to_json())


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
        return 1 if result.anomalous else 0

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
