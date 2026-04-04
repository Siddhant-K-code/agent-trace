"""Postmortem analysis for failed agent sessions.

Identifies the failure point, traces the causal chain, calculates wasted
time and cost, and generates concrete recommendations.
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .cost import estimate_cost
from .explain import explain_session
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimelineEntry:
    offset: float       # seconds from session start
    description: str
    is_root_cause: bool = False
    is_retry: bool = False
    is_failure: bool = False


@dataclass
class PostmortemReport:
    session_id: str
    failed: bool
    status_summary: str             # e.g. "Failed (build error after 4m 12s)"
    root_cause: str                 # one-line description
    root_cause_offset: float        # seconds from session start
    timeline: list[TimelineEntry]
    wasted_seconds: float
    total_seconds: float
    wasted_cost: float
    total_cost: float
    recommendations: list[str]
    agents_md_violations: list[str]  # instructions contradicted by the agent


# ---------------------------------------------------------------------------
# AGENTS.md parsing
# ---------------------------------------------------------------------------

def _load_agents_md(path: str | Path = "AGENTS.md") -> list[str]:
    """Return lines from AGENTS.md that look like instructions."""
    p = Path(path)
    if not p.exists():
        return []
    lines = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        # Keep non-empty, non-header lines that look like instructions
        if stripped and not stripped.startswith("#") and len(stripped) > 10:
            lines.append(stripped)
    return lines


def _detect_agents_md_violations(
    events: list[TraceEvent],
    agents_md_lines: list[str],
) -> list[str]:
    """Find tool_call commands that contradict AGENTS.md instructions.

    Heuristic: if AGENTS.md says "use X" and the agent ran "Y" (a known
    alternative to X), flag it.
    """
    if not agents_md_lines:
        return []

    # Build a simple map of "forbidden tool" → "required tool" from AGENTS.md
    # by looking for patterns like "use X, not Y" or "always use X"
    import re
    forbidden: dict[str, str] = {}  # forbidden_cmd_prefix → required_cmd

    for line in agents_md_lines:
        line_lower = line.lower()
        # "use X, not Y" / "use X not Y"
        m = re.search(r"use\s+(\w+)[,\s]+not\s+(\w+)", line_lower)
        if m:
            required, forbidden_cmd = m.group(1), m.group(2)
            forbidden[forbidden_cmd] = required
        # "never use Y" / "do not use Y"
        m2 = re.search(r"(?:never|do not|don't)\s+use\s+(\w+)", line_lower)
        if m2:
            forbidden_cmd = m2.group(1)
            forbidden[forbidden_cmd] = "(see AGENTS.md)"
        # "always use X" — not yet implemented; would require knowing
        # which tools are equivalent alternatives to X

    violations = []
    for event in events:
        if event.event_type != EventType.TOOL_CALL:
            continue
        args = event.data.get("arguments", {}) or {}
        cmd = str(args.get("command", "")).strip().lower()
        if not cmd:
            continue
        for forbidden_cmd, required in forbidden.items():
            if cmd.startswith(forbidden_cmd):
                violations.append(
                    f"Ran `{cmd[:60]}` — AGENTS.md says use `{required}` instead"
                )
                break

    return violations


# ---------------------------------------------------------------------------
# Root cause detection
# ---------------------------------------------------------------------------

def _find_root_cause(events: list[TraceEvent], base_ts: float) -> tuple[int, str, float]:
    """Return (event_index, description, offset_seconds) of the root cause.

    Strategy:
    1. First ERROR event is the primary failure signal.
    2. If no ERROR, look for a TOOL_RESULT with a non-zero exit code.
    3. If neither, the session is not failed.
    """
    for i, event in enumerate(events):
        if event.event_type == EventType.ERROR:
            msg = event.data.get("message", event.data.get("error", "unknown error"))
            offset = event.timestamp - base_ts
            return i, f"Error: {str(msg)[:120]}", offset

    # Check tool results for failure signals
    for i, event in enumerate(events):
        if event.event_type == EventType.TOOL_RESULT:
            result = str(event.data.get("result", ""))
            if any(sig in result.lower() for sig in ("exit code", "error:", "failed", "traceback")):
                offset = event.timestamp - base_ts
                return i, f"Tool failure: {result[:80]}", offset

    return -1, "No failure detected", 0.0


def _build_timeline(
    events: list[TraceEvent],
    base_ts: float,
    root_cause_idx: int,
) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = []
    seen_commands: dict[str, int] = {}  # cmd → first occurrence index

    for i, event in enumerate(events):
        offset = event.timestamp - base_ts
        is_root = i == root_cause_idx
        is_failure = event.event_type == EventType.ERROR

        if event.event_type == EventType.SESSION_START:
            entries.append(TimelineEntry(offset, "Session start"))

        elif event.event_type == EventType.SESSION_END:
            entries.append(TimelineEntry(offset, "Session end", is_failure=is_failure))

        elif event.event_type == EventType.USER_PROMPT:
            prompt = str(event.data.get("prompt", ""))[:80]
            entries.append(TimelineEntry(offset, f'User: "{prompt}"'))

        elif event.event_type == EventType.TOOL_CALL:
            name = event.data.get("tool_name", "?")
            args = event.data.get("arguments", {}) or {}
            if name.lower() == "bash":
                cmd = str(args.get("command", "")).strip()
                is_retry = cmd in seen_commands
                if not is_retry:
                    seen_commands[cmd] = i
                desc = f"Ran: {cmd[:80]}"
                entries.append(TimelineEntry(offset, desc, is_root_cause=is_root, is_retry=is_retry))
            elif name.lower() in ("read", "view"):
                path = str(args.get("file_path", ""))
                entries.append(TimelineEntry(offset, f"Read {path}", is_root_cause=is_root))
            elif name.lower() in ("write", "edit"):
                path = str(args.get("file_path", ""))
                entries.append(TimelineEntry(offset, f"Write {path}", is_root_cause=is_root))
            else:
                entries.append(TimelineEntry(offset, f"Tool: {name}", is_root_cause=is_root))

        elif event.event_type == EventType.ERROR:
            msg = str(event.data.get("message", event.data.get("error", "error")))[:100]
            entries.append(TimelineEntry(offset, msg, is_root_cause=is_root, is_failure=True))

        elif event.event_type == EventType.FILE_READ:
            uri = str(event.data.get("uri", ""))
            if uri:
                entries.append(TimelineEntry(offset, f"Read {uri}", is_root_cause=is_root))

    return entries


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

def _generate_recommendations(
    report_data: dict,
) -> list[str]:
    recs = []
    violations = report_data.get("violations", [])
    wasted_pct = report_data.get("wasted_pct", 0)
    root_cause = report_data.get("root_cause", "")
    retry_count = report_data.get("retry_count", 0)

    for v in violations:
        recs.append(f"Strengthen AGENTS.md: the instruction was ignored — {v}")

    if retry_count > 2:
        recs.append(
            f"Agent retried {retry_count} times after failure. "
            "Add a pre-tool hook or AGENTS.md instruction to fail fast instead of retrying."
        )

    if wasted_pct > 50:
        recs.append(
            f"{wasted_pct:.0f}% of session time was wasted after the root cause. "
            "Consider adding a cost or duration circuit breaker with `agent-strace watch`."
        )

    if "permission" in root_cause.lower() or "denied" in root_cause.lower():
        recs.append(
            "Permission error detected. Document required permissions in AGENTS.md "
            "or configure a .agent-scope.json policy."
        )

    if "import" in root_cause.lower() or "module" in root_cause.lower():
        recs.append(
            "Import/module error detected. Ensure dependencies are documented in AGENTS.md "
            "or a requirements file."
        )

    if not recs:
        recs.append("Review the root cause event and add a guard to AGENTS.md to prevent recurrence.")

    return recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_session(
    store: TraceStore,
    session_id: str,
    agents_md_path: str | Path = "AGENTS.md",
) -> PostmortemReport:
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    explain = explain_session(store, session_id)

    base_ts = events[0].timestamp if events else meta.started_at
    total_seconds = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0.0
    failed = any(p.failed for p in explain.phases)

    root_cause_idx, root_cause_desc, root_cause_offset = _find_root_cause(events, base_ts)

    # Wasted time = time after root cause
    wasted_seconds = 0.0
    if root_cause_idx >= 0 and root_cause_idx < len(events):
        last_ts = events[-1].timestamp if events else base_ts
        root_ts = events[root_cause_idx].timestamp
        wasted_seconds = max(0.0, last_ts - root_ts)

    # Cost
    try:
        cost_result = estimate_cost(store, session_id)
        total_cost = cost_result.total_cost
        wasted_cost = cost_result.wasted_cost
    except Exception:
        total_cost = 0.0
        wasted_cost = 0.0

    # AGENTS.md violations
    agents_md_lines = _load_agents_md(agents_md_path)
    violations = _detect_agents_md_violations(events, agents_md_lines)

    # Timeline
    timeline = _build_timeline(events, base_ts, root_cause_idx)

    # Status summary
    if failed:
        mins = int(total_seconds) // 60
        secs = int(total_seconds) % 60
        duration_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        status_summary = f"Failed (after {duration_str})"
    else:
        status_summary = "OK (no failures detected)"

    # Retry count
    retry_count = sum(p.retry_count for p in explain.phases)
    wasted_pct = (wasted_seconds / total_seconds * 100) if total_seconds > 0 else 0

    recommendations = _generate_recommendations({
        "violations": violations,
        "wasted_pct": wasted_pct,
        "root_cause": root_cause_desc,
        "retry_count": retry_count,
    })

    return PostmortemReport(
        session_id=session_id,
        failed=failed,
        status_summary=status_summary,
        root_cause=root_cause_desc,
        root_cause_offset=root_cause_offset,
        timeline=timeline,
        wasted_seconds=wasted_seconds,
        total_seconds=total_seconds,
        wasted_cost=wasted_cost,
        total_cost=total_cost,
        recommendations=recommendations,
        agents_md_violations=violations,
    )


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def _fmt_offset(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def format_postmortem(report: PostmortemReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    w(f"\nPOSTMORTEM: Session {report.session_id}\n")
    w(f"Status:     {report.status_summary}\n")
    w(f"Root cause: {report.root_cause}\n\n")

    w("Timeline:\n")
    for entry in report.timeline:
        ts = _fmt_offset(entry.offset)
        suffix = ""
        if entry.is_root_cause:
            suffix = "  ← ROOT CAUSE"
        elif entry.is_retry:
            suffix = "  ← retry"
        elif entry.is_failure:
            suffix = "  ← failure"
        w(f"  {ts:>6}  {entry.description}{suffix}\n")

    w("\n")

    if report.wasted_seconds > 0 and report.total_seconds > 0:
        pct = report.wasted_seconds / report.total_seconds * 100
        w(f"Wasted: {report.wasted_seconds:.0f}s after root cause ({pct:.0f}% of session)\n")
    if report.wasted_cost > 0:
        w(f"Estimated wasted cost: ${report.wasted_cost:.4f}\n")

    if report.agents_md_violations:
        w("\nAGENTS.md violations:\n")
        for v in report.agents_md_violations:
            w(f"  - {v}\n")

    w("\nRecommendations:\n")
    for i, rec in enumerate(report.recommendations, 1):
        w(f"  {i}. {rec}\n")

    w("\n")


# ---------------------------------------------------------------------------
# HTML rendering (used by share.py)
# ---------------------------------------------------------------------------

def render_postmortem_html(report: PostmortemReport) -> str:
    if not report.failed:
        return ""

    def esc(s: str) -> str:
        return html.escape(str(s))

    rows = ""
    for entry in report.timeline:
        ts = _fmt_offset(entry.offset)
        cls = ""
        suffix = ""
        if entry.is_root_cause:
            cls = ' style="color:#f85149;font-weight:bold"'
            suffix = "  ← ROOT CAUSE"
        elif entry.is_retry:
            cls = ' style="color:#e3b341"'
            suffix = "  ← retry"
        elif entry.is_failure:
            cls = ' style="color:#f85149"'
        rows += f'<tr{cls}><td style="color:#484f58;padding-right:12px">{esc(ts)}</td><td>{esc(entry.description)}{esc(suffix)}</td></tr>\n'

    violations_html = ""
    if report.agents_md_violations:
        items = "".join(f"<li>{esc(v)}</li>" for v in report.agents_md_violations)
        violations_html = f'<p style="color:#e3b341;margin-top:8px">AGENTS.md violations:</p><ul style="margin-left:16px;color:#e3b341">{items}</ul>'

    recs_html = "".join(
        f'<li style="margin-bottom:4px">{esc(r)}</li>'
        for r in report.recommendations
    )

    wasted_html = ""
    if report.wasted_seconds > 0 and report.total_seconds > 0:
        pct = report.wasted_seconds / report.total_seconds * 100
        wasted_html = (
            f'<p style="color:#f85149;margin-top:8px">'
            f'Wasted: {report.wasted_seconds:.0f}s after root cause ({pct:.0f}% of session)'
            f'{f" · Est. wasted cost: ${report.wasted_cost:.4f}" if report.wasted_cost > 0 else ""}'
            f"</p>"
        )

    return f"""
<div style="border:1px solid #6e2020;border-radius:8px;padding:16px;margin-bottom:16px;background:#1a0d0d">
  <h2 style="color:#f85149;margin-bottom:8px">Postmortem</h2>
  <p><strong>Status:</strong> {esc(report.status_summary)}</p>
  <p><strong>Root cause:</strong> <span style="color:#f85149">{esc(report.root_cause)}</span></p>
  {wasted_html}
  {violations_html}
  <details style="margin-top:12px">
    <summary style="cursor:pointer;color:#8b949e">Timeline</summary>
    <table style="margin-top:8px;font-size:12px">
      <tbody>{rows}</tbody>
    </table>
  </details>
  <p style="margin-top:12px"><strong>Recommendations:</strong></p>
  <ol style="margin-left:16px;margin-top:4px">{recs_html}</ol>
</div>"""


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_postmortem(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    full_id = store.find_session(session_id)
    if not full_id:
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    agents_md = getattr(args, "agents_md", "AGENTS.md")
    report = analyze_session(store, full_id, agents_md_path=agents_md)
    format_postmortem(report)
    return 1 if report.failed else 0
