"""Behavioral drift detection across agent sessions.

Computes a behavioral fingerprint (distribution of tool mix, error rate,
retry pattern, blast radius, session duration, decision depth) for a window
of sessions and measures how much that distribution has shifted compared to
a baseline window.

Distance metric: Jensen-Shannon divergence, normalized to [0, 1].
No LLM required. All analysis is structural.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .cost import session_cost
from .event_data import require_usable_timestamps, str_field
from .failures import count_session_errors
from .inert import inert
from .models import (
    MAX_META_DEPTH,
    SINCE_DAYS_FORM,
    EventType,
    TraceEvent,
    _nests_deeper_than,
    parse_since_days,
    usable_number,
    usable_time,
)
from .store import TraceStore


# ---------------------------------------------------------------------------
# Fingerprint data structures
# ---------------------------------------------------------------------------

@dataclass
class DistStats:
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0

    def to_dict(self) -> dict:
        return {"mean": round(self.mean, 4), "p50": round(self.p50, 4), "p95": round(self.p95, 4)}


# int() refuses a decimal string of more than 4300 digits by default, with a
# message that differs between Python versions. The parser counts the digits
# first and gives its own reason.
_MAX_INT_DIGITS = 4300


def _refuse_constant(name: str) -> float:
    raise ValueError("it has a number that is not finite")


def _finite_float(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("it has a number that is not finite")
    return number


def _bounded_int(text: str) -> int:
    if len(text.removeprefix("-")) > _MAX_INT_DIGITS:
        raise ValueError("it has a number with too many digits")
    return int(text)


def _finite_number(value: object) -> bool:
    """Return True if ``value`` is an int or float, not a bool, that fits a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass
class BehavioralFingerprint:
    fingerprint_id: str = ""
    sessions: int = 0
    period_start: str = ""
    period_end: str = ""
    # Tool mix: fraction of tool calls per tool name
    tool_mix: dict[str, float] = field(default_factory=dict)
    # Per-session distributions
    error_rate: DistStats = field(default_factory=DistStats)
    retry_rate: DistStats = field(default_factory=DistStats)
    tool_calls: DistStats = field(default_factory=DistStats)
    # Over priced sessions only; None when no session could be priced.
    cost_usd: DistStats | None = field(default_factory=DistStats)
    blast_radius: DistStats = field(default_factory=DistStats)
    session_duration_s: DistStats = field(default_factory=DistStats)
    decision_depth: DistStats = field(default_factory=DistStats)
    # Sessions `agent-strace cost` refuses, left out of cost_usd.
    unpriced_sessions: int = 0

    def to_dict(self) -> dict:
        return {
            "fingerprint_id": self.fingerprint_id,
            "sessions": self.sessions,
            "period": {"start": self.period_start, "end": self.period_end},
            "tool_mix": {k: round(v, 4) for k, v in self.tool_mix.items()},
            "error_rate": self.error_rate.to_dict(),
            "retry_rate": self.retry_rate.to_dict(),
            "tool_calls": self.tool_calls.to_dict(),
            "cost_usd": None if self.cost_usd is None else self.cost_usd.to_dict(),
            "unpriced_sessions": self.unpriced_sessions,
            "blast_radius": self.blast_radius.to_dict(),
            "session_duration_s": self.session_duration_s.to_dict(),
            "decision_depth": self.decision_depth.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "BehavioralFingerprint":
        """Build a fingerprint from ``text``, or raise ValueError with the reason.

        drift and ``fingerprint --compare`` do math on ``tool_mix`` and the
        seven statistics, and can print the result as JSON. So the text must
        be a JSON object that nests at most MAX_META_DEPTH levels and holds
        no NaN, infinity, float out of range or int of more than 4300 digits.
        ``period``, ``tool_mix`` and each statistic must be objects, but
        ``cost_usd`` may be null. Each ``tool_mix`` value must be a number of
        0 or more. A statistic's ``mean``, ``p50`` and ``p95`` must be finite
        numbers, not bools. A missing field reads as its default.

        ``fingerprint_id``, ``sessions``, ``unpriced_sessions`` and the two
        ``period`` fields keep any value. They are only shown, never used in
        the math.
        """
        try:
            d = json.loads(text, parse_constant=_refuse_constant,
                           parse_float=_finite_float, parse_int=_bounded_int)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"it is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
            ) from None
        except RecursionError:
            raise ValueError("it is nested too deeply") from None
        # Check the depth before the type. Python 3.10 cannot parse a list
        # nested 2000 levels and 3.14 can, so a deep list must be refused as
        # too deep on both, not as "not a JSON object" on one.
        if isinstance(d, (dict, list)) and _nests_deeper_than(d, MAX_META_DEPTH):
            raise ValueError("it is nested too deeply")
        if not isinstance(d, dict):
            raise ValueError("it is not a JSON object")
        period = d.get("period", {})
        if not isinstance(period, dict):
            raise ValueError("period is not an object")
        tool_mix = d.get("tool_mix", {})
        if not isinstance(tool_mix, dict):
            raise ValueError("tool_mix is not an object")
        if any(usable_number(share) is None for share in tool_mix.values()):
            raise ValueError("a tool_mix value is not a number of 0 or more")
        fp = cls(
            fingerprint_id=d.get("fingerprint_id", ""),
            sessions=d.get("sessions", 0),
            period_start=period.get("start", ""),
            period_end=period.get("end", ""),
            tool_mix=tool_mix,
            unpriced_sessions=d.get("unpriced_sessions", 0),
        )
        for attr in (
            "error_rate",
            "retry_rate",
            "tool_calls",
            "cost_usd",
            "blast_radius",
            "session_duration_s",
            "decision_depth",
        ):
            raw = d.get(attr, {})
            if raw is None and attr == "cost_usd":
                fp.cost_usd = None
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"{attr} is not an object")
            stats = {}
            for key in ("mean", "p50", "p95"):
                value = raw.get(key, 0.0)
                if not _finite_number(value):
                    raise ValueError(f"{attr}.{key} is not a finite number")
                stats[key] = value
            setattr(fp, attr, DistStats(**stats))
        return fp


@dataclass
class DimensionDrift:
    name: str
    score: float | None   # 0.0 = identical, 1.0 = maximally different; None = not comparable
    label: str            # "stable" / "moderate" / "high" / "n/a"
    baseline_summary: str = ""
    current_summary: str = ""


@dataclass
class DriftReport:
    overall_score: float
    threshold: float
    baseline_sessions: int
    current_sessions: int
    baseline_period: str
    current_period: str
    dimensions: list[DimensionDrift]
    baseline_unpriced: int = 0
    current_unpriced: int = 0

    @property
    def label(self) -> str:
        if self.overall_score < self.threshold * 0.5:
            return "stable"
        if self.overall_score < self.threshold:
            return "moderate"
        return "high"

    @property
    def exceeded(self) -> bool:
        return self.overall_score >= self.threshold


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def _dist_stats(values: list[float]) -> DistStats:
    if not values:
        return DistStats()
    return DistStats(
        mean=sum(values) / len(values),
        p50=_percentile(values, 50),
        p95=_percentile(values, 95),
    )


def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two probability distributions.

    Both dicts map category -> probability. Missing keys treated as 0.
    Returns a value in [0, 1] (normalized by log(2)).
    """
    keys = set(p) | set(q)
    if not keys:
        return 0.0

    # Normalize to ensure they sum to 1
    p_sum = sum(p.values()) or 1.0
    q_sum = sum(q.values()) or 1.0
    pn = {k: p.get(k, 0.0) / p_sum for k in keys}
    qn = {k: q.get(k, 0.0) / q_sum for k in keys}

    m = {k: (pn[k] + qn[k]) / 2.0 for k in keys}

    def kl(a: dict, b: dict) -> float:
        total = 0.0
        for k in keys:
            av = a.get(k, 0.0)
            bv = b.get(k, 0.0)
            if av > 0 and bv > 0:
                total += av * math.log(av / bv)
        return total

    jsd = (kl(pn, m) + kl(qn, m)) / 2.0
    # Normalize: max JSD is log(2) for binary distributions
    return min(1.0, jsd / math.log(2)) if jsd > 0 else 0.0


def _stats_divergence(a: DistStats, b: DistStats) -> float:
    """Approximate divergence between two DistStats using mean/p50/p95."""
    if a.mean == 0 and b.mean == 0:
        return 0.0
    diffs = []
    for av, bv in [(a.mean, b.mean), (a.p50, b.p50), (a.p95, b.p95)]:
        denom = max(abs(av), abs(bv), 1e-9)
        diffs.append(abs(av - bv) / denom)
    return min(1.0, sum(diffs) / len(diffs))


def _drift_label(score: float, threshold: float) -> str:
    if score < threshold * 0.5:
        return "stable"
    if score < threshold:
        return "moderate"
    return "high"


# ---------------------------------------------------------------------------
# Per-session metrics extraction
# ---------------------------------------------------------------------------

@dataclass
class SessionMetrics:
    session_id: str
    started_at: float
    duration_s: float
    tool_mix: dict[str, int]   # tool_name -> count
    error_count: int
    total_tool_calls: int
    retry_count: int
    cost_usd: float | None     # None when `agent-strace cost` refuses the session
    blast_radius: int          # distinct files written
    decision_count: int


def _extract_metrics(
    session_id: str,
    events: list[TraceEvent],
    meta_started: float,
    cost_usd: float | None = None,
) -> SessionMetrics:
    """Measure one session.

    Raises UnusableTimestampError when the first or last event's timestamp is
    not a Unix time in seconds, since the duration reads both.
    compute_fingerprint then skips the session.
    """
    if events:
        require_usable_timestamps(events, indexes={0, len(events) - 1})
    tool_mix: dict[str, int] = {}
    error_count = count_session_errors(events)
    total_tool_calls = 0
    retry_count = 0
    files_written: set[str] = set()
    decision_count = 0

    prev_tool: str | None = None
    prev_tool_count = 0

    for ev in events:
        if ev.event_type == EventType.TOOL_CALL:
            name = str_field(ev.data, "tool_name", "unknown")
            tool_mix[name] = tool_mix.get(name, 0) + 1
            total_tool_calls += 1
            # Detect retry: same tool called consecutively
            if name == prev_tool:
                prev_tool_count += 1
                if prev_tool_count >= 2:
                    retry_count += 1
            else:
                prev_tool = name
                prev_tool_count = 1

        elif ev.event_type in (EventType.FILE_WRITE,):
            path = str_field(ev.data, "path", "") or str_field(ev.data, "file_path", "")
            if path:
                files_written.add(path)

        elif ev.event_type == EventType.DECISION:
            decision_count += 1

    duration_s = 0.0
    if events:
        duration_s = events[-1].timestamp - events[0].timestamp

    return SessionMetrics(
        session_id=session_id,
        started_at=meta_started,
        duration_s=duration_s,
        tool_mix=tool_mix,
        error_count=error_count,
        total_tool_calls=total_tool_calls,
        retry_count=retry_count,
        cost_usd=cost_usd,
        blast_radius=len(files_written),
        decision_count=decision_count,
    )


def _session_cost(events: list[TraceEvent], session_id: str) -> float | None:
    """Return the `agent-strace cost` total, or None when it refuses the session."""
    try:
        return session_cost(events, session_id=session_id)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def compute_fingerprint(
    store: TraceStore,
    session_ids: list[str],
    fingerprint_id: str = "",
) -> BehavioralFingerprint:
    """Measure the sessions and aggregate them into one fingerprint.

    The period runs from the earliest to the latest usable start. A start
    that is not a usable time is left out of the period, but the session
    still counts. With no usable start, the period and the default ID are
    empty.
    """
    if not session_ids:
        return BehavioralFingerprint(fingerprint_id=fingerprint_id)

    all_metrics: list[SessionMetrics] = []
    timestamps: list[float] = []

    for sid in session_ids:
        try:
            meta = store.load_meta(sid)
            events = store.load_events(sid)
            m = _extract_metrics(sid, events, meta.started_at)
        except Exception:
            continue
        m.cost_usd = _session_cost(events, sid)
        all_metrics.append(m)
        # The period comes from the usable starts only.
        started = usable_time(meta.started_at)
        if started is not None:
            timestamps.append(started)

    if not all_metrics:
        return BehavioralFingerprint(fingerprint_id=fingerprint_id)

    # Aggregate tool mix across all sessions
    combined_tool_mix: dict[str, int] = {}
    for m in all_metrics:
        for tool, count in m.tool_mix.items():
            combined_tool_mix[tool] = combined_tool_mix.get(tool, 0) + count
    total_calls = sum(combined_tool_mix.values()) or 1
    tool_mix_frac = {k: v / total_calls for k, v in combined_tool_mix.items()}

    # Per-session rate distributions
    error_rates = [
        m.error_count / max(m.total_tool_calls, 1) for m in all_metrics
    ]
    retry_rates = [
        m.retry_count / max(m.total_tool_calls, 1) for m in all_metrics
    ]
    tool_calls = [float(m.total_tool_calls) for m in all_metrics]
    costs = [m.cost_usd for m in all_metrics if m.cost_usd is not None]
    blast_radii = [float(m.blast_radius) for m in all_metrics]
    durations = [m.duration_s for m in all_metrics]
    decisions = [float(m.decision_count) for m in all_metrics]

    period_start = ""
    period_end = ""
    if timestamps:
        period_start = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).strftime("%Y-%m-%d")
        period_end = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).strftime("%Y-%m-%d")

    return BehavioralFingerprint(
        fingerprint_id=fingerprint_id or (f"fp_{period_start}" if period_start else ""),
        sessions=len(all_metrics),
        period_start=period_start,
        period_end=period_end,
        tool_mix=tool_mix_frac,
        error_rate=_dist_stats(error_rates),
        retry_rate=_dist_stats(retry_rates),
        tool_calls=_dist_stats(tool_calls),
        cost_usd=_dist_stats(costs) if costs else None,
        blast_radius=_dist_stats(blast_radii),
        session_duration_s=_dist_stats(durations),
        decision_depth=_dist_stats(decisions),
        unpriced_sessions=len(all_metrics) - len(costs),
    )


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------

def compute_drift(
    baseline: BehavioralFingerprint,
    current: BehavioralFingerprint,
    threshold: float = 0.20,
) -> DriftReport:
    dimensions: list[DimensionDrift] = []

    # Tool mix (JS divergence on distributions)
    tool_score = _js_divergence(baseline.tool_mix, current.tool_mix)
    dimensions.append(DimensionDrift(
        name="tool_mix",
        score=round(tool_score, 3),
        label=_drift_label(tool_score, threshold),
        baseline_summary=_top_tools(baseline.tool_mix),
        current_summary=_top_tools(current.tool_mix),
    ))

    # Scalar distributions (approximate divergence via stats)
    for attr, label in [
        ("error_rate", "error_rate"),
        ("retry_rate", "retry_rate"),
        ("tool_calls", "tool_calls"),
        ("cost_usd", "cost_usd"),
        ("blast_radius", "blast_radius"),
        ("session_duration_s", "session_duration"),
        ("decision_depth", "decision_depth"),
    ]:
        b_stat: DistStats | None = getattr(baseline, attr)
        c_stat: DistStats | None = getattr(current, attr)
        if b_stat is None or c_stat is None:
            # No priced session on one side: cost cannot be compared.
            dimensions.append(DimensionDrift(
                name=attr,
                score=None,
                label="n/a",
                baseline_summary=_stats_summary(b_stat),
                current_summary=_stats_summary(c_stat),
            ))
            continue
        score = _stats_divergence(b_stat, c_stat)
        dimensions.append(DimensionDrift(
            name=attr,
            score=round(score, 3),
            label=_drift_label(score, threshold),
            baseline_summary=_stats_summary(b_stat),
            current_summary=_stats_summary(c_stat),
        ))

    # Weighted average (tool_mix weighted 2x, others 1x) over comparable dimensions
    weights = [2.0] + [1.0] * (len(dimensions) - 1)
    scored = [(d.score, w) for d, w in zip(dimensions, weights) if d.score is not None]
    overall = sum(score * w for score, w in scored) / sum(w for _, w in scored)

    return DriftReport(
        overall_score=round(overall, 3),
        threshold=threshold,
        baseline_sessions=baseline.sessions,
        current_sessions=current.sessions,
        baseline_period=f"{baseline.period_start} to {baseline.period_end}",
        current_period=f"{current.period_start} to {current.period_end}",
        dimensions=dimensions,
        baseline_unpriced=baseline.unpriced_sessions,
        current_unpriced=current.unpriced_sessions,
    )


def _stats_summary(stats: DistStats | None) -> str:
    if stats is None:
        return "n/a"
    return f"mean={stats.mean:.2f} p50={stats.p50:.2f} p95={stats.p95:.2f}"


def _top_tools(tool_mix: dict[str, float], n: int = 3) -> str:
    top = sorted(tool_mix.items(), key=lambda x: x[1], reverse=True)[:n]
    return ", ".join(f"{k}={v:.0%}" for k, v in top)


# ---------------------------------------------------------------------------
# Fingerprint terminal output
# ---------------------------------------------------------------------------

def _fmt_seconds(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def print_fingerprint(fp: BehavioralFingerprint, out: TextIO = sys.stdout) -> None:
    """Write a fingerprint as a table.

    Tool names come from the trace, so each goes through inert() and an
    escape code or a line break in it shows as text.
    """
    w = out.write
    w("\nBehavioral fingerprint\n")
    w(f"ID:       {fp.fingerprint_id or '(none)'}\n")
    w(f"Period:   {fp.period_start or '-'} to {fp.period_end or '-'}\n")
    w(f"Sessions: {fp.sessions}\n")
    w("─" * 70 + "\n")

    if fp.sessions == 0:
        w("No sessions available for fingerprinting.\n\n")
        return

    w("Tool call profile:\n")
    if fp.tool_mix:
        for tool, frac in sorted(fp.tool_mix.items(), key=lambda item: item[1], reverse=True):
            w(f"  {inert(tool):<22} {frac:>6.0%}\n")
    else:
        w("  (no tool calls)\n")

    w("\nSession profile (per session):\n")
    w(f"  {'Metric':<22} {'Median':>10} {'P95':>10} {'Mean':>10}\n")
    w("  " + "-" * 46 + "\n")
    w(
        f"  {'duration':<22} "
        f"{_fmt_seconds(fp.session_duration_s.p50):>10} "
        f"{_fmt_seconds(fp.session_duration_s.p95):>10} "
        f"{_fmt_seconds(fp.session_duration_s.mean):>10}\n"
    )
    for label, stats in [
        ("tool calls", fp.tool_calls),
        ("cost usd", fp.cost_usd),
        ("error rate", fp.error_rate),
        ("retry rate", fp.retry_rate),
        ("file touch radius", fp.blast_radius),
        ("decision depth", fp.decision_depth),
    ]:
        if stats is None:
            w(f"  {label:<22} {'n/a':>10} {'n/a':>10} {'n/a':>10}\n")
            continue
        w(f"  {label:<22} {stats.p50:>10.2f} {stats.p95:>10.2f} {stats.mean:>10.2f}\n")
    if fp.unpriced_sessions:
        w(f"\nSessions not priced: {fp.unpriced_sessions} (left out of cost usd)\n")
    w("\n")


# ---------------------------------------------------------------------------
# Session filtering helpers
# ---------------------------------------------------------------------------

def _sessions_in_window(store: TraceStore, since_days: float) -> list[str]:
    import time
    cutoff = time.time() - since_days * 86400
    result = []
    for meta in store.list_sessions():
        # A start that is not a usable time is in no window.
        started = usable_time(meta.started_at)
        if started is not None and started >= cutoff:
            result.append(meta.session_id)
    return result


def _latest_sessions(store: TraceStore, limit: int) -> list[str]:
    return [meta.session_id for meta in store.list_sessions()[:max(0, limit)]]


def _sessions_in_range(store: TraceStore, start_ts: float, end_ts: float) -> list[str]:
    result = []
    for meta in store.list_sessions():
        # A start that is not a usable time is in no window.
        started = usable_time(meta.started_at)
        if started is not None and start_ts <= started <= end_ts:
            result.append(meta.session_id)
    return result


def _parse_date_range(s: str) -> tuple[float, float]:
    """Parse 'YYYY-MM-DD:YYYY-MM-DD' into (start_ts, end_ts)."""
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected 'YYYY-MM-DD:YYYY-MM-DD', got: {s!r}")
    start = datetime.strptime(parts[0].strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    end = datetime.strptime(parts[1].strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() + 86399
    return start, end


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

ICON = {"stable": "✅", "moderate": "⚠️ ", "high": "❌"}

SUGGESTED_NEXT_STEPS = (
    "agent-strace eval dataset auto --name drift-failures --since 14d --filter has-errors",
    "agent-strace optimize --dataset drift-failures --dry-run",
)


def print_report(report: DriftReport, out: TextIO = sys.stdout) -> None:
    """Write the drift report.

    With --compare or --baseline, the periods, session counts and tool names
    come from a fingerprint file that anyone can write. Each of these values
    and each dimension goes through inert(), so an escape code or a line
    break in it shows as text.
    """
    w = out.write
    w("\nBehavioral drift report\n")
    w(f"Baseline: {inert(report.baseline_period)} ({inert(report.baseline_sessions)} sessions)\n")
    w(f"Current:  {inert(report.current_period)} ({inert(report.current_sessions)} sessions)\n")
    if report.baseline_unpriced or report.current_unpriced:
        w(
            f"Sessions not priced: {inert(report.baseline_unpriced)} baseline, "
            f"{inert(report.current_unpriced)} current (left out of cost_usd)\n"
        )
    w("─" * 70 + "\n")

    icon = ICON.get(report.label, "")
    w(f"Overall drift score: {report.overall_score:.2f}  {icon}  {report.label.upper()}"
      f"  (threshold: {report.threshold:.2f})\n\n")

    w(f"  {'Dimension':<22} {'Score':>6}  Status\n")
    w("─" * 70 + "\n")
    for d in report.dimensions:
        icon_d = ICON.get(d.label, "")
        score = f"{'n/a':>6}" if d.score is None else f"{d.score:>6.3f}"
        w(f"  {inert(d.name):<22} {score}  {icon_d}  {inert(d.label)}\n")

    w("\n")
    high = [d for d in report.dimensions if d.label == "high"]
    if high:
        w("High-drift dimensions:\n\n")
        for d in high:
            w(f"  {inert(d.name)} ({d.score:.2f}):\n")
            w(f"    Baseline: {inert(d.baseline_summary)}\n")
            w(f"    Current:  {inert(d.current_summary)}\n\n")

    if report.exceeded:
        w("Suggested next step:\n")
        for command in SUGGESTED_NEXT_STEPS:
            w(f"  {command}\n")
    w("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def read_fingerprint_file(path: str) -> BehavioralFingerprint:
    """Read the fingerprint file at ``path``, or raise ValueError with the reason.

    The reason does not name the file. The file is opened with no check
    that it exists or is a regular file, so a pipe such as ``--baseline
    <(cmd)`` reads too. It is decoded as UTF-8 whatever the locale, since
    the files this tool writes are ASCII JSON.
    """
    if not path:
        raise ValueError("the path is empty")
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        # str(exc) repeats the path, which the refusal line already shows.
        raise ValueError(exc.strerror or "it cannot be read") from None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("it is not valid UTF-8") from None
    # Skip one byte order mark, which some Windows editors write. json.loads
    # refuses text that starts with one.
    if text.startswith("\ufeff"):
        text = text[1:]
    return BehavioralFingerprint.from_json(text)


def _refuse_fingerprint_file(path: str, exc: ValueError) -> int:
    # Show escapes in the path and the reason as text. An empty path shows
    # as '' so that the line still shows a path.
    shown = inert(path) if path else "''"
    sys.stderr.write(f"Could not read fingerprint file {shown}: {inert(exc)}\n")
    return 1


def cmd_drift(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    threshold = getattr(args, "threshold", 0.20)

    # Resolve current session window
    since_raw = getattr(args, "since", None)
    current_range = getattr(args, "current", None)
    baseline_range = getattr(args, "baseline_range", None)

    # With --current, --since is not read.
    days: float | None = None
    if not current_range and since_raw is not None:
        days = parse_since_days(since_raw)
        if days is None:
            sys.stderr.write(f"invalid --since value: {SINCE_DAYS_FORM}\n")
            return 1

    # Resolve baseline fingerprint. A --baseline that cannot be read stops
    # drift before any session math, so --save-baseline writes nothing.
    baseline_fp: BehavioralFingerprint | None = None
    baseline_path = getattr(args, "baseline", None)
    if baseline_path is not None:
        try:
            baseline_fp = read_fingerprint_file(baseline_path)
        except ValueError as exc:
            return _refuse_fingerprint_file(baseline_path, exc)

    if current_range:
        start, end = _parse_date_range(current_range)
        current_ids = _sessions_in_range(store, start, end)
    elif days is not None:
        all_ids = _sessions_in_window(store, days)
        # Split in half: first half = baseline (if no explicit baseline), second = current
        mid = len(all_ids) // 2
        if baseline_fp is None and not baseline_range:
            baseline_ids = all_ids[mid:]   # older sessions (list is newest-first)
            current_ids = all_ids[:mid]
            baseline_fp = compute_fingerprint(store, baseline_ids, "baseline")
        else:
            current_ids = all_ids[:mid]
    else:
        sys.stderr.write("Specify --since Nd or --current YYYY-MM-DD:YYYY-MM-DD\n")
        return 1

    if baseline_range and baseline_fp is None:
        start, end = _parse_date_range(baseline_range)
        baseline_ids = _sessions_in_range(store, start, end)
        baseline_fp = compute_fingerprint(store, baseline_ids, "baseline")

    if baseline_fp is None:
        sys.stderr.write("No baseline available. Use --baseline <file> or --since Nd.\n")
        return 1

    current_fp = compute_fingerprint(store, current_ids, "current")

    # Save baseline if requested
    save_path = getattr(args, "save_baseline", None)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(current_fp.to_json(), newline="\n")
        sys.stdout.write(f"Behavioral fingerprint saved to {save_path}\n")
        return 0

    report = compute_drift(baseline_fp, current_fp, threshold=threshold)

    fmt = getattr(args, "format", "table")
    if fmt == "json":
        data = {
            "overall_score": report.overall_score,
            "threshold": report.threshold,
            "exceeded": report.exceeded,
            "label": report.label,
            "baseline_sessions": report.baseline_sessions,
            "current_sessions": report.current_sessions,
            "baseline_unpriced_sessions": report.baseline_unpriced,
            "current_unpriced_sessions": report.current_unpriced,
            "dimensions": [
                {"name": d.name, "score": d.score, "label": d.label}
                for d in report.dimensions
            ],
        }
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
    else:
        print_report(report)

    return 1 if report.exceeded else 0


def cmd_fingerprint(args: argparse.Namespace) -> int:
    compare_paths = getattr(args, "compare", None) or []
    fmt = getattr(args, "format", "text") or "text"

    if compare_paths:
        if len(compare_paths) != 2:
            sys.stderr.write("Specify exactly two fingerprint files with --compare.\n")
            return 1
        fingerprints = []
        for path in compare_paths:
            try:
                fingerprints.append(read_fingerprint_file(path))
            except ValueError as exc:
                return _refuse_fingerprint_file(path, exc)
        first, second = fingerprints
        report = compute_drift(first, second, threshold=getattr(args, "threshold", 0.20))
        if fmt == "json":
            data = {
                "overall_score": report.overall_score,
                "threshold": report.threshold,
                "exceeded": report.exceeded,
                "label": report.label,
                "baseline_sessions": report.baseline_sessions,
                "current_sessions": report.current_sessions,
                "baseline_unpriced_sessions": report.baseline_unpriced,
                "current_unpriced_sessions": report.current_unpriced,
                "dimensions": [
                    {
                        "name": d.name,
                        "score": d.score,
                        "label": d.label,
                        "first": d.baseline_summary,
                        "second": d.current_summary,
                    }
                    for d in report.dimensions
                ],
            }
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
        else:
            print_report(report)
        return 1 if report.exceeded else 0

    store = TraceStore(args.trace_dir)
    sessions = int(getattr(args, "sessions", 20) or 20)
    session_ids = _latest_sessions(store, sessions)
    fp = compute_fingerprint(store, session_ids, fingerprint_id=getattr(args, "id", "") or "")

    output = getattr(args, "output", None)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(fp.to_json() + "\n", newline="\n")
        message = f"Behavioral fingerprint saved to {out_path}\n"
        if fmt == "json":
            sys.stderr.write(message)
        else:
            sys.stdout.write(message)

    if fmt == "json":
        sys.stdout.write(fp.to_json() + "\n")
    elif not output:
        print_fingerprint(fp)

    return 0 if fp.sessions > 0 else 1
