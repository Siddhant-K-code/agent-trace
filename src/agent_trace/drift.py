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

from .cost import estimate_cost
from .models import EventType, TraceEvent
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
    cost_usd: DistStats = field(default_factory=DistStats)
    blast_radius: DistStats = field(default_factory=DistStats)
    session_duration_s: DistStats = field(default_factory=DistStats)
    decision_depth: DistStats = field(default_factory=DistStats)

    def to_dict(self) -> dict:
        return {
            "fingerprint_id": self.fingerprint_id,
            "sessions": self.sessions,
            "period": {"start": self.period_start, "end": self.period_end},
            "tool_mix": {k: round(v, 4) for k, v in self.tool_mix.items()},
            "error_rate": self.error_rate.to_dict(),
            "retry_rate": self.retry_rate.to_dict(),
            "tool_calls": self.tool_calls.to_dict(),
            "cost_usd": self.cost_usd.to_dict(),
            "blast_radius": self.blast_radius.to_dict(),
            "session_duration_s": self.session_duration_s.to_dict(),
            "decision_depth": self.decision_depth.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "BehavioralFingerprint":
        d = json.loads(text)
        fp = cls(
            fingerprint_id=d.get("fingerprint_id", ""),
            sessions=d.get("sessions", 0),
            period_start=d.get("period", {}).get("start", ""),
            period_end=d.get("period", {}).get("end", ""),
            tool_mix=d.get("tool_mix", {}),
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
            setattr(fp, attr, DistStats(
                mean=raw.get("mean", 0.0),
                p50=raw.get("p50", 0.0),
                p95=raw.get("p95", 0.0),
            ))
        return fp


@dataclass
class DimensionDrift:
    name: str
    score: float          # 0.0 = identical, 1.0 = maximally different
    label: str            # "stable" / "moderate" / "high"
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
    cost_usd: float
    blast_radius: int          # distinct files written
    decision_count: int


def _extract_metrics(
    session_id: str,
    events: list[TraceEvent],
    meta_started: float,
    cost_usd: float = 0.0,
) -> SessionMetrics:
    tool_mix: dict[str, int] = {}
    error_count = 0
    total_tool_calls = 0
    retry_count = 0
    files_written: set[str] = set()
    decision_count = 0

    prev_tool: str | None = None
    prev_tool_count = 0

    for ev in events:
        if ev.event_type == EventType.TOOL_CALL:
            name = ev.data.get("tool_name", "unknown")
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
            path = ev.data.get("path") or ev.data.get("file_path") or ""
            if path:
                files_written.add(path)

        elif ev.event_type == EventType.ERROR:
            error_count += 1

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


def _session_cost(store: TraceStore, session_id: str) -> float:
    try:
        return estimate_cost(store, session_id).total_cost
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def compute_fingerprint(
    store: TraceStore,
    session_ids: list[str],
    fingerprint_id: str = "",
) -> BehavioralFingerprint:
    if not session_ids:
        return BehavioralFingerprint(fingerprint_id=fingerprint_id)

    all_metrics: list[SessionMetrics] = []
    timestamps: list[float] = []

    for sid in session_ids:
        try:
            meta = store.load_meta(sid)
            events = store.load_events(sid)
            m = _extract_metrics(sid, events, meta.started_at, _session_cost(store, sid))
            all_metrics.append(m)
            timestamps.append(meta.started_at)
        except Exception:
            continue

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
    costs = [m.cost_usd for m in all_metrics]
    blast_radii = [float(m.blast_radius) for m in all_metrics]
    durations = [m.duration_s for m in all_metrics]
    decisions = [float(m.decision_count) for m in all_metrics]

    period_start = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).strftime("%Y-%m-%d")
    period_end = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).strftime("%Y-%m-%d")

    return BehavioralFingerprint(
        fingerprint_id=fingerprint_id or f"fp_{period_start}",
        sessions=len(all_metrics),
        period_start=period_start,
        period_end=period_end,
        tool_mix=tool_mix_frac,
        error_rate=_dist_stats(error_rates),
        retry_rate=_dist_stats(retry_rates),
        tool_calls=_dist_stats(tool_calls),
        cost_usd=_dist_stats(costs),
        blast_radius=_dist_stats(blast_radii),
        session_duration_s=_dist_stats(durations),
        decision_depth=_dist_stats(decisions),
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
        b_stat: DistStats = getattr(baseline, attr)
        c_stat: DistStats = getattr(current, attr)
        score = _stats_divergence(b_stat, c_stat)
        dimensions.append(DimensionDrift(
            name=attr,
            score=round(score, 3),
            label=_drift_label(score, threshold),
            baseline_summary=f"mean={b_stat.mean:.2f} p50={b_stat.p50:.2f} p95={b_stat.p95:.2f}",
            current_summary=f"mean={c_stat.mean:.2f} p50={c_stat.p50:.2f} p95={c_stat.p95:.2f}",
        ))

    # Weighted average (tool_mix weighted 2x, others 1x)
    weights = [2.0] + [1.0] * (len(dimensions) - 1)
    overall = sum(d.score * w for d, w in zip(dimensions, weights)) / sum(weights)

    return DriftReport(
        overall_score=round(overall, 3),
        threshold=threshold,
        baseline_sessions=baseline.sessions,
        current_sessions=current.sessions,
        baseline_period=f"{baseline.period_start} to {baseline.period_end}",
        current_period=f"{current.period_start} to {current.period_end}",
        dimensions=dimensions,
    )


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
            w(f"  {tool:<22} {frac:>6.0%}\n")
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
        w(f"  {label:<22} {stats.p50:>10.2f} {stats.p95:>10.2f} {stats.mean:>10.2f}\n")
    w("\n")


# ---------------------------------------------------------------------------
# Session filtering helpers
# ---------------------------------------------------------------------------

def _sessions_in_window(store: TraceStore, since_days: float) -> list[str]:
    import time
    cutoff = time.time() - since_days * 86400
    result = []
    for meta in store.list_sessions():
        if meta.started_at >= cutoff:
            result.append(meta.session_id)
    return result


def _latest_sessions(store: TraceStore, limit: int) -> list[str]:
    return [meta.session_id for meta in store.list_sessions()[:max(0, limit)]]


def _sessions_in_range(store: TraceStore, start_ts: float, end_ts: float) -> list[str]:
    result = []
    for meta in store.list_sessions():
        if start_ts <= meta.started_at <= end_ts:
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


def print_report(report: DriftReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    w("\nBehavioral drift report\n")
    w(f"Baseline: {report.baseline_period} ({report.baseline_sessions} sessions)\n")
    w(f"Current:  {report.current_period} ({report.current_sessions} sessions)\n")
    w("─" * 70 + "\n")

    icon = ICON.get(report.label, "")
    w(f"Overall drift score: {report.overall_score:.2f}  {icon}  {report.label.upper()}"
      f"  (threshold: {report.threshold:.2f})\n\n")

    w(f"  {'Dimension':<22} {'Score':>6}  Status\n")
    w("─" * 70 + "\n")
    for d in report.dimensions:
        icon_d = ICON.get(d.label, "")
        w(f"  {d.name:<22} {d.score:>6.3f}  {icon_d}  {d.label}\n")

    w("\n")
    high = [d for d in report.dimensions if d.label == "high"]
    if high:
        w("High-drift dimensions:\n\n")
        for d in high:
            w(f"  {d.name} ({d.score:.2f}):\n")
            w(f"    Baseline: {d.baseline_summary}\n")
            w(f"    Current:  {d.current_summary}\n\n")

    if report.exceeded:
        w("Suggested next step:\n")
        w("  agent-strace dataset auto --name drift-failures --since 14d --filter has-errors\n")
        w("  agent-strace eval --dataset drift-failures\n")
    w("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_drift(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    threshold = getattr(args, "threshold", 0.20)

    # Resolve baseline fingerprint
    baseline_fp: BehavioralFingerprint | None = None
    baseline_path = getattr(args, "baseline", None)
    if baseline_path and Path(baseline_path).exists():
        baseline_fp = BehavioralFingerprint.from_json(Path(baseline_path).read_text())

    # Resolve current session window
    since_days = getattr(args, "since", None)
    current_range = getattr(args, "current", None)
    baseline_range = getattr(args, "baseline_range", None)

    if current_range:
        start, end = _parse_date_range(current_range)
        current_ids = _sessions_in_range(store, start, end)
    elif since_days:
        days = float(since_days.rstrip("d"))
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
        Path(save_path).write_text(current_fp.to_json())
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
        try:
            first = BehavioralFingerprint.from_json(Path(compare_paths[0]).read_text())
            second = BehavioralFingerprint.from_json(Path(compare_paths[1]).read_text())
        except Exception as exc:
            sys.stderr.write(f"Could not read fingerprint file: {exc}\n")
            return 1
        report = compute_drift(first, second, threshold=getattr(args, "threshold", 0.20))
        if fmt == "json":
            data = {
                "overall_score": report.overall_score,
                "threshold": report.threshold,
                "exceeded": report.exceeded,
                "label": report.label,
                "baseline_sessions": report.baseline_sessions,
                "current_sessions": report.current_sessions,
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
        out_path.write_text(fp.to_json() + "\n")
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
