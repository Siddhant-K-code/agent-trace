"""Multi-session dashboard: aggregate view across sessions with trend data.

Produces a terminal table and an optional self-contained HTML dashboard
showing cost, duration, tool calls, errors, and trend lines across all
(or a filtered set of) sessions.

The --trend mode extends this with per-session eval scores (read from
eval.json written by agent-strace eval), error/retry/cost sparklines
over time, and timeline annotations for config changes or model upgrades.
All charts are inline SVG — no CDN, no JavaScript libraries.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionSummary:
    session_id: str
    started_at: float
    duration_s: float
    tool_calls: int
    llm_requests: int
    errors: int
    total_tokens: int
    estimated_cost: float
    agent_name: str
    succeeded: bool   # True if no errors recorded


@dataclass
class DashboardReport:
    summaries: list[SessionSummary]
    total_cost: float
    total_tokens: int
    total_tool_calls: int
    total_errors: int
    avg_duration_s: float
    success_rate: float   # 0.0–1.0


# ---------------------------------------------------------------------------
# Trend data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalScorePoint:
    """A single judge's score for one session."""
    judge: str
    score: float
    passed: bool


@dataclass
class TrendPoint:
    """All metrics for a single session, used for trend charts."""
    session_id: str
    started_at: float
    error_rate: float        # errors / tool_calls
    retry_rate: float        # consecutive same-tool calls / tool_calls
    cost: float
    duration_s: float
    eval_scores: list[EvalScorePoint] = field(default_factory=list)


@dataclass
class TrendAnnotation:
    date: str    # YYYY-MM-DD
    note: str


@dataclass
class TrendReport:
    points: list[TrendPoint]
    annotations: list[TrendAnnotation]
    judge_names: list[str]   # all judges seen across sessions


# ---------------------------------------------------------------------------
# Annotation storage
# ---------------------------------------------------------------------------

_ANNOTATIONS_FILE = "annotations.jsonl"


def _annotations_path(store: TraceStore) -> Path:
    return store.base_dir / _ANNOTATIONS_FILE


def load_annotations(store: TraceStore) -> list[TrendAnnotation]:
    p = _annotations_path(store)
    if not p.exists():
        return []
    result = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            result.append(TrendAnnotation(date=d["date"], note=d["note"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return result


def save_annotation(store: TraceStore, annotation: TrendAnnotation) -> None:
    p = _annotations_path(store)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"date": annotation.date, "note": annotation.note}) + "\n")


# ---------------------------------------------------------------------------
# Eval score reading
# ---------------------------------------------------------------------------

def _load_eval_scores(store: TraceStore, session_id: str) -> list[EvalScorePoint]:
    eval_path = store.base_dir / session_id / "eval.json"
    if not eval_path.exists():
        return []
    try:
        data = json.loads(eval_path.read_text())
        results = data.get("results") or data.get("judges") or []
        points = []
        for r in results:
            name = r.get("scorer") or r.get("name") or "unknown"
            score = float(r.get("score", 0.0))
            passed = bool(r.get("passed", score >= r.get("threshold", 1.0)))
            points.append(EvalScorePoint(judge=name, score=score, passed=passed))
        return points
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Trend metrics extraction
# ---------------------------------------------------------------------------

def _extract_trend_point(
    store: TraceStore,
    meta: SessionMeta,
) -> TrendPoint:
    try:
        events: list[TraceEvent] = store.load_events(meta.session_id)
    except Exception:
        events = []

    tool_calls = 0
    errors = 0
    retries = 0
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
        elif ev.event_type == EventType.ERROR:
            errors += 1

    error_rate = errors / max(tool_calls, 1)
    retry_rate = retries / max(tool_calls, 1)
    cost = meta.total_tokens / 1_000_000 * 3.0
    duration_s = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0.0

    eval_scores = _load_eval_scores(store, meta.session_id)

    return TrendPoint(
        session_id=meta.session_id,
        started_at=meta.started_at,
        error_rate=error_rate,
        retry_rate=retry_rate,
        cost=cost,
        duration_s=duration_s,
        eval_scores=eval_scores,
    )


# ---------------------------------------------------------------------------
# Trend report builder
# ---------------------------------------------------------------------------

def build_trend_report(
    store: TraceStore,
    since_days: float | None = None,
    limit: int = 200,
) -> TrendReport:
    all_meta = store.list_sessions()

    if since_days is not None:
        cutoff = time.time() - since_days * 86400
        all_meta = [m for m in all_meta if m.started_at >= cutoff]

    sessions = all_meta[:limit]
    # Oldest first for charts
    sessions = list(reversed(sessions))

    points: list[TrendPoint] = []
    judge_names: set[str] = set()

    for meta in sessions:
        pt = _extract_trend_point(store, meta)
        points.append(pt)
        for es in pt.eval_scores:
            judge_names.add(es.judge)

    annotations = load_annotations(store)

    return TrendReport(
        points=points,
        annotations=annotations,
        judge_names=sorted(judge_names),
    )


# ---------------------------------------------------------------------------
# Terminal trend summary
# ---------------------------------------------------------------------------

def format_trend_terminal(report: TrendReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    n = len(report.points)
    if n == 0:
        w("No sessions found.\n")
        return

    w(f"\nTrend report — {n} session{'s' if n != 1 else ''}\n")
    w("─" * 60 + "\n\n")

    # Eval score trends
    if report.judge_names:
        w("Quality trend (eval pass rate):\n")
        mid = n // 2
        for judge in report.judge_names:
            def _pass_rate(pts: list[TrendPoint]) -> float:
                scored = [p for p in pts if any(e.judge == judge for e in p.eval_scores)]
                if not scored:
                    return float("nan")
                passed = sum(
                    1 for p in scored
                    if any(e.judge == judge and e.passed for e in p.eval_scores)
                )
                return passed / len(scored)

            early = _pass_rate(report.points[:mid]) if mid > 0 else float("nan")
            late = _pass_rate(report.points[mid:]) if mid < n else float("nan")

            if early != early or late != late:  # nan check
                w(f"  {judge:<30} (no eval scores)\n")
            else:
                delta = late - early
                arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
                w(f"  {judge:<30} {early*100:.0f}% → {late*100:.0f}%  {arrow} {delta*100:+.0f}pp\n")
        w("\n")

    # Behavioral trends
    def _trend_line(values: list[float], label: str) -> None:
        if not values:
            return
        mid = len(values) // 2
        early_avg = sum(values[:mid]) / max(mid, 1)
        late_avg = sum(values[mid:]) / max(len(values) - mid, 1)
        delta = late_avg - early_avg
        arrow = "↑" if delta < -0.001 else ("↓" if delta > 0.001 else "→")
        # For error/retry, lower is better so flip arrow meaning
        w(f"  {label:<30} {early_avg:.3f} → {late_avg:.3f}  {arrow}\n")

    w("Behavioral trend:\n")
    _trend_line([p.error_rate for p in report.points], "Error rate")
    _trend_line([p.retry_rate for p in report.points], "Retry rate")
    w("\nCost trend:\n")
    _trend_line([p.cost for p in report.points], "Avg cost/session ($)")
    _trend_line([p.duration_s for p in report.points], "Avg duration (s)")
    w("\n")

    if report.annotations:
        w("Annotations:\n")
        for ann in report.annotations:
            w(f"  {ann.date}  {ann.note}\n")
        w("\n")


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

_CHART_W = 560
_CHART_H = 80
_COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"]


def _svg_sparkline(
    values: list[float],
    width: int = _CHART_W,
    height: int = _CHART_H,
    color: str = "#58a6ff",
    annotations: list[tuple[float, str]] | None = None,  # (x_frac, label)
) -> str:
    """Render a list of floats as an inline SVG polyline."""
    if not values or len(values) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'

    mn = min(values)
    mx = max(values)
    rng = mx - mn or 1.0
    pad = 6

    def _x(i: int) -> float:
        return pad + (i / (len(values) - 1)) * (width - 2 * pad)

    def _y(v: float) -> float:
        return pad + (1 - (v - mn) / rng) * (height - 2 * pad)

    pts = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(values))

    ann_lines = ""
    if annotations:
        for x_frac, label in annotations:
            x = pad + x_frac * (width - 2 * pad)
            ann_lines += (
                f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" '
                f'stroke="#e3b341" stroke-width="1" stroke-dasharray="3,2"/>'
                f'<text x="{x+3:.1f}" y="10" fill="#e3b341" '
                f'font-size="8" font-family="monospace">{html.escape(label[:20])}</text>'
            )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="display:block">'
        f'{ann_lines}'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        f'</svg>'
    )


def _annotation_x_fracs(
    points: list[TrendPoint],
    annotations: list[TrendAnnotation],
) -> list[tuple[float, str]]:
    """Map annotation dates to x-axis fractions based on session timestamps."""
    if not points or not annotations:
        return []
    t_min = points[0].started_at
    t_max = points[-1].started_at
    t_rng = t_max - t_min or 1.0
    result = []
    for ann in annotations:
        try:
            ts = datetime.strptime(ann.date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            frac = max(0.0, min(1.0, (ts - t_min) / t_rng))
            result.append((frac, ann.note))
        except ValueError:
            continue
    return result


# ---------------------------------------------------------------------------
# HTML trend report
# ---------------------------------------------------------------------------

def render_html_trend(report: TrendReport) -> str:
    """Produce a self-contained HTML trend report with inline SVG charts."""
    pts = report.points
    ann_fracs = _annotation_x_fracs(pts, report.annotations)

    def _spark(values: list[float], color: str = "#58a6ff") -> str:
        return _svg_sparkline(values, color=color, annotations=ann_fracs)

    # Eval score charts (one per judge)
    eval_section = ""
    for i, judge in enumerate(report.judge_names):
        color = _COLORS[i % len(_COLORS)]
        scores = []
        for p in pts:
            match = next((e.score for e in p.eval_scores if e.judge == judge), None)
            scores.append(match if match is not None else float("nan"))
        # Replace nan with previous value for continuity
        filled: list[float] = []
        last = 0.5
        for v in scores:
            if v == v:  # not nan
                last = v
            filled.append(last)
        pass_rate = (
            sum(1 for p in pts if any(e.judge == judge and e.passed for e in p.eval_scores))
            / max(sum(1 for p in pts if any(e.judge == judge for e in p.eval_scores)), 1)
        )
        eval_section += f"""
        <div class="chart-block">
          <div class="chart-label">{html.escape(judge)} <span class="badge">{pass_rate*100:.0f}% pass</span></div>
          {_spark(filled, color)}
        </div>"""

    # Behavioral charts
    error_spark = _spark([p.error_rate for p in pts], "#f85149")
    retry_spark = _spark([p.retry_rate for p in pts], "#ffa657")
    cost_spark = _spark([p.cost for p in pts], "#3fb950")
    dur_spark = _spark([p.duration_s for p in pts], "#d2a8ff")

    # Annotation list
    ann_html = ""
    if report.annotations:
        ann_html = "<ul>" + "".join(
            f"<li><span class='ann-date'>{html.escape(a.date)}</span> {html.escape(a.note)}</li>"
            for a in report.annotations
        ) + "</ul>"

    # Session table (last 20)
    table_rows = ""
    for p in reversed(pts[-20:]):
        dt = datetime.fromtimestamp(p.started_at, tz=timezone.utc).strftime("%m-%d %H:%M")
        scores_str = ", ".join(
            f"{e.judge}={e.score:.2f}" for e in p.eval_scores
        ) or "—"
        table_rows += (
            f"<tr>"
            f"<td>{html.escape(p.session_id[:12])}</td>"
            f"<td>{dt}</td>"
            f"<td>{p.error_rate:.2%}</td>"
            f"<td>{p.retry_rate:.2%}</td>"
            f"<td>${p.cost:.4f}</td>"
            f"<td>{scores_str}</td>"
            f"</tr>\n"
        )

    n = len(pts)
    period = ""
    if pts:
        t0 = datetime.fromtimestamp(pts[0].started_at, tz=timezone.utc).strftime("%Y-%m-%d")
        t1 = datetime.fromtimestamp(pts[-1].started_at, tz=timezone.utc).strftime("%Y-%m-%d")
        period = f"{t0} to {t1}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>agent-strace trend dashboard</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:monospace;background:#0d1117;color:#c9d1d9;margin:0;padding:24px;font-size:13px}}
h1{{color:#58a6ff;font-size:1.1em;margin:0 0 4px}}
.subtitle{{color:#8b949e;font-size:.85em;margin-bottom:24px}}
h2{{color:#8b949e;font-size:.85em;text-transform:uppercase;letter-spacing:.08em;margin:24px 0 8px;border-bottom:1px solid #21262d;padding-bottom:4px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.chart-block{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px}}
.chart-label{{color:#8b949e;font-size:.8em;margin-bottom:6px}}
.badge{{background:#1f6feb;color:#e6edf3;border-radius:3px;padding:1px 5px;font-size:.75em;margin-left:6px}}
table{{width:100%;border-collapse:collapse;font-size:.82em;margin-top:8px}}
th{{background:#161b22;color:#8b949e;padding:5px 8px;text-align:left;border-bottom:1px solid #30363d}}
td{{padding:4px 8px;border-bottom:1px solid #21262d}}
tr:hover{{background:#161b22}}
.ann-date{{color:#e3b341;margin-right:8px}}
ul{{margin:0;padding-left:20px;color:#c9d1d9;font-size:.85em;line-height:1.8}}
polyline{{fill:none}}
</style>
</head>
<body>
<h1>agent-strace trend dashboard</h1>
<div class="subtitle">{n} sessions &nbsp;·&nbsp; {html.escape(period)}</div>

<h2>Eval quality</h2>
<div class="charts">
{eval_section if eval_section else '<div class="chart-block" style="grid-column:1/-1;color:#8b949e">No eval scores found. Run <code>agent-strace eval</code> to score sessions.</div>'}
</div>

<h2>Behavioral metrics</h2>
<div class="charts">
  <div class="chart-block">
    <div class="chart-label">Error rate (per session)</div>
    {error_spark}
  </div>
  <div class="chart-block">
    <div class="chart-label">Retry rate (per session)</div>
    {retry_spark}
  </div>
  <div class="chart-block">
    <div class="chart-label">Estimated cost ($)</div>
    {cost_spark}
  </div>
  <div class="chart-block">
    <div class="chart-label">Session duration (s)</div>
    {dur_spark}
  </div>
</div>

{f'<h2>Annotations</h2>{ann_html}' if report.annotations else ''}

<h2>Recent sessions</h2>
<table>
<thead><tr>
  <th>Session</th><th>Started</th><th>Error rate</th>
  <th>Retry rate</th><th>Cost</th><th>Eval scores</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_dashboard(
    store: TraceStore,
    limit: int = 50,
    agent_filter: str = "",
) -> DashboardReport:
    """Build a DashboardReport from the most recent *limit* sessions."""
    all_meta = store.list_sessions()

    if agent_filter:
        all_meta = [m for m in all_meta if agent_filter.lower() in m.agent_name.lower()]

    sessions = all_meta[:limit]

    summaries: list[SessionSummary] = []
    for meta in sessions:
        duration_s = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0.0
        # Estimate cost cheaply from token count stored in meta
        cost = meta.total_tokens / 1_000_000 * 3.0  # rough sonnet input price

        summaries.append(SessionSummary(
            session_id=meta.session_id,
            started_at=meta.started_at,
            duration_s=duration_s,
            tool_calls=meta.tool_calls,
            llm_requests=meta.llm_requests,
            errors=meta.errors,
            total_tokens=meta.total_tokens,
            estimated_cost=cost,
            agent_name=meta.agent_name or "unknown",
            succeeded=meta.errors == 0,
        ))

    total_cost = sum(s.estimated_cost for s in summaries)
    total_tokens = sum(s.total_tokens for s in summaries)
    total_tools = sum(s.tool_calls for s in summaries)
    total_errors = sum(s.errors for s in summaries)
    avg_dur = (
        sum(s.duration_s for s in summaries) / len(summaries)
        if summaries else 0.0
    )
    success_rate = (
        sum(1 for s in summaries if s.succeeded) / len(summaries)
        if summaries else 0.0
    )

    return DashboardReport(
        summaries=summaries,
        total_cost=total_cost,
        total_tokens=total_tokens,
        total_tool_calls=total_tools,
        total_errors=total_errors,
        avg_duration_s=avg_dur,
        success_rate=success_rate,
    )


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

def _fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    return f"{int(s)//60}m{int(s)%60:02d}s"


def _fmt_ts(ts: float) -> str:
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return "?"


def format_dashboard(report: DashboardReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    n = len(report.summaries)

    w(f"\nDashboard — {n} session{'s' if n != 1 else ''}\n\n")

    # Summary row
    w(f"  Total cost:    ~${report.total_cost:.4f}\n")
    w(f"  Total tokens:  {report.total_tokens:,}\n")
    w(f"  Tool calls:    {report.total_tool_calls:,}\n")
    w(f"  Errors:        {report.total_errors}\n")
    w(f"  Avg duration:  {_fmt_dur(report.avg_duration_s)}\n")
    w(f"  Success rate:  {report.success_rate*100:.0f}%\n\n")

    if not report.summaries:
        return

    # Table header
    w(f"  {'ID':<14}  {'Started':<12}  {'Dur':>7}  {'Tools':>5}  "
      f"{'LLM':>4}  {'Err':>3}  {'Tokens':>8}  {'Cost':>8}  Status\n")
    w("  " + "-" * 80 + "\n")

    for s in report.summaries:
        status = "✓" if s.succeeded else "✗"
        w(
            f"  {s.session_id[:12]:<14}  {_fmt_ts(s.started_at):<12}  "
            f"{_fmt_dur(s.duration_s):>7}  {s.tool_calls:>5}  "
            f"{s.llm_requests:>4}  {s.errors:>3}  "
            f"{s.total_tokens:>8,}  ${s.estimated_cost:>7.4f}  {status}\n"
        )

    w("\n")

    # Trend: last 10 sessions cost
    if len(report.summaries) >= 3:
        recent = list(reversed(report.summaries[:10]))
        costs = [s.estimated_cost for s in recent]
        max_cost = max(costs) or 1.0
        w("  Cost trend (oldest → newest):\n  ")
        bars = "▁▂▃▄▅▆▇█"
        for c in costs:
            idx = min(int(c / max_cost * (len(bars) - 1)), len(bars) - 1)
            w(bars[idx])
        w("\n\n")


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

def render_html_dashboard(report: DashboardReport) -> str:
    """Produce a self-contained HTML dashboard page."""
    rows_html = ""
    for s in report.summaries:
        status_cls = "ok" if s.succeeded else "err"
        status_sym = "✓" if s.succeeded else "✗"
        rows_html += (
            f"<tr class='{status_cls}'>"
            f"<td>{html.escape(s.session_id[:12])}</td>"
            f"<td>{html.escape(_fmt_ts(s.started_at))}</td>"
            f"<td>{html.escape(_fmt_dur(s.duration_s))}</td>"
            f"<td>{s.tool_calls}</td>"
            f"<td>{s.llm_requests}</td>"
            f"<td>{s.errors}</td>"
            f"<td>{s.total_tokens:,}</td>"
            f"<td>${s.estimated_cost:.4f}</td>"
            f"<td>{status_sym}</td>"
            f"</tr>\n"
        )

    # Sparkline data for Chart.js-free inline SVG
    costs = [s.estimated_cost for s in reversed(report.summaries[:20])]
    max_c = max(costs) or 1.0
    spark_points = ""
    if costs:
        w = 200
        h = 40
        pts = []
        for i, c in enumerate(costs):
            x = int(i / max(len(costs) - 1, 1) * w)
            y = int(h - (c / max_c) * h)
            pts.append(f"{x},{y}")
        spark_points = " ".join(pts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>agent-strace dashboard</title>
<style>
body{{font-family:monospace;background:#0d1117;color:#c9d1d9;margin:0;padding:20px}}
h1{{color:#58a6ff;font-size:1.2em;margin-bottom:16px}}
.stats{{display:flex;gap:24px;margin-bottom:20px;flex-wrap:wrap}}
.stat{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px 20px}}
.stat .label{{font-size:.75em;color:#8b949e}}
.stat .value{{font-size:1.4em;color:#e6edf3;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:.85em}}
th{{background:#161b22;color:#8b949e;padding:6px 10px;text-align:left;border-bottom:1px solid #30363d}}
td{{padding:5px 10px;border-bottom:1px solid #21262d}}
tr.ok td:last-child{{color:#3fb950}}
tr.err td:last-child{{color:#f85149}}
tr:hover{{background:#161b22}}
.spark{{margin-bottom:20px}}
polyline{{fill:none;stroke:#58a6ff;stroke-width:1.5}}
</style>
</head>
<body>
<h1>agent-strace dashboard</h1>
<div class="stats">
  <div class="stat"><div class="label">Sessions</div><div class="value">{len(report.summaries)}</div></div>
  <div class="stat"><div class="label">Est. cost</div><div class="value">${report.total_cost:.4f}</div></div>
  <div class="stat"><div class="label">Total tokens</div><div class="value">{report.total_tokens:,}</div></div>
  <div class="stat"><div class="label">Tool calls</div><div class="value">{report.total_tool_calls:,}</div></div>
  <div class="stat"><div class="label">Errors</div><div class="value">{report.total_errors}</div></div>
  <div class="stat"><div class="label">Success rate</div><div class="value">{report.success_rate*100:.0f}%</div></div>
  <div class="stat"><div class="label">Avg duration</div><div class="value">{_fmt_dur(report.avg_duration_s)}</div></div>
</div>
<div class="spark">
  <svg width="200" height="40" viewBox="0 0 200 40">
    <polyline points="{spark_points}"/>
  </svg>
  <span style="font-size:.75em;color:#8b949e"> cost trend</span>
</div>
<table>
<thead><tr>
  <th>Session</th><th>Started</th><th>Duration</th>
  <th>Tools</th><th>LLM</th><th>Errors</th>
  <th>Tokens</th><th>Cost</th><th>Status</th>
</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_dashboard(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    # annotate subcommand
    dash_cmd = getattr(args, "dash_command", None)
    if dash_cmd == "annotate":
        date = getattr(args, "date", "")
        note = getattr(args, "note", "")
        if not date or not note:
            sys.stderr.write("--date and --note are required\n")
            return 1
        save_annotation(store, TrendAnnotation(date=date, note=note))
        sys.stdout.write(f"Annotation saved: {date}  {note}\n")
        return 0

    # --trend mode
    trend = getattr(args, "trend", False)
    if trend:
        since_raw = getattr(args, "since", None)
        since_days: float | None = None
        if since_raw:
            since_days = float(since_raw.rstrip("d"))

        trend_report = build_trend_report(store, since_days=since_days)

        html_path = getattr(args, "html", None)
        if html_path:
            Path(html_path).write_text(render_html_trend(trend_report))
            sys.stdout.write(f"Trend report written to {html_path}\n")
            return 0

        format_trend_terminal(trend_report)
        return 0

    # Default aggregate dashboard
    limit = getattr(args, "limit", 50) or 50
    agent_filter = getattr(args, "agent", "") or ""
    report = build_dashboard(store, limit=limit, agent_filter=agent_filter)

    output_path = getattr(args, "output", None)
    if output_path:
        html_content = render_html_dashboard(report)
        Path(output_path).write_text(html_content)
        sys.stdout.write(f"Dashboard written to {output_path}\n")
        return 0

    format_dashboard(report)
    return 0
