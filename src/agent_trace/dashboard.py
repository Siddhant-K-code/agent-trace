"""Multi-session dashboard: aggregate view across sessions with trend data.

Produces a terminal table and an optional self-contained HTML dashboard
showing cost, duration, tool calls, errors, and trend lines across all
(or a filtered set of) sessions.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TextIO

from .models import SessionMeta
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
    limit = getattr(args, "limit", 50) or 50
    agent_filter = getattr(args, "agent", "") or ""

    report = build_dashboard(store, limit=limit, agent_filter=agent_filter)

    output_path = getattr(args, "output", None)
    if output_path:
        from pathlib import Path
        html_content = render_html_dashboard(report)
        Path(output_path).write_text(html_content)
        sys.stdout.write(f"Dashboard written to {output_path}\n")
        return 0

    format_dashboard(report)
    return 0
