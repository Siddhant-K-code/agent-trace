"""Postmortem analysis for failed agent sessions.

Identifies the failure point, traces the causal chain, calculates wasted
time and cost, and generates concrete recommendations.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
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

HEARTBEAT_FILE = ".heartbeat"
CRASH_POSTMORTEM_FILE = "postmortem.md"
DEFAULT_STALE_SECONDS = 30.0

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
    crash_reason: str = ""
    crash_detail: str = ""
    recovery_context: str = ""


@dataclass
class CrashInfo:
    session_id: str
    reason: str
    detail: str
    stale_seconds: float = 0.0
    exit_code: int | None = None
    last_event_type: str = ""
    last_event_at: float = 0.0


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


def heartbeat_path(store: TraceStore, session_id: str) -> Path:
    """Return the heartbeat sidecar path for a session."""
    return store._session_dir(session_id) / HEARTBEAT_FILE


def write_heartbeat(store: TraceStore, session_id: str) -> None:
    """Touch the session heartbeat file with lightweight JSON metadata."""
    path = heartbeat_path(store, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"session_id": session_id, "updated_at": time.time()}
    path.write_text(json.dumps(payload, separators=(",", ":")))


def clear_heartbeat(store: TraceStore, session_id: str) -> None:
    """Remove the heartbeat file after a clean session end."""
    try:
        heartbeat_path(store, session_id).unlink()
    except FileNotFoundError:
        pass


def _has_clean_session_end(events: list[TraceEvent]) -> bool:
    for event in reversed(events):
        if event.event_type == EventType.SESSION_END:
            exit_code = event.data.get("exit_code")
            return exit_code in (None, 0)
    return False


def _session_exit_code(events: list[TraceEvent]) -> int | None:
    for event in reversed(events):
        if event.event_type == EventType.SESSION_END:
            raw = event.data.get("exit_code")
            if raw is None:
                return None
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
    return None


def _event_text(events: list[TraceEvent], limit: int = 20) -> str:
    chunks = []
    for event in events[-limit:]:
        chunks.append(event.event_type.value)
        chunks.append(json.dumps(event.data, sort_keys=True, default=str))
    return "\n".join(chunks).lower()


def _last_action(events: list[TraceEvent]) -> str:
    for event in reversed(events):
        if event.event_type in (EventType.SESSION_START, EventType.SESSION_END):
            continue
        return _describe_event(event)
    return "No recorded action"


def _describe_event(event: TraceEvent) -> str:
    if event.event_type == EventType.TOOL_CALL:
        name = event.data.get("tool_name", "?")
        args = event.data.get("arguments", {}) or {}
        target = (
            args.get("command")
            or args.get("file_path")
            or args.get("path")
            or args.get("url")
            or ""
        )
        return f"{name}({str(target)[:100]})" if target else f"Tool: {name}"
    if event.event_type in (EventType.LLM_REQUEST, EventType.USER_PROMPT):
        text = event.data.get("prompt") or event.data.get("text") or event.data
        return f"Prompt: {str(text)[:100]}"
    if event.event_type == EventType.ERROR:
        return f"Error: {event.data.get('message', event.data.get('error', 'error'))}"
    return event.event_type.value


def _classify_crash(
    events: list[TraceEvent],
    exit_code: int | None = None,
    stale: bool = False,
) -> tuple[str, str]:
    text = _event_text(events)

    if exit_code == 137:
        return "SIGKILL", "Process exited with code 137"
    if exit_code == 143:
        return "SIGTERM", "Process exited with code 143"
    if exit_code == 124:
        return "timeout", "Process exited with timeout code 124"
    if exit_code is not None and exit_code != 0:
        if "memoryerror" in text or "out of memory" in text or "oom" in text:
            return "OOM", "Memory failure signal found near session end"
        if "traceback" in text:
            return "unhandled_exception", f"Process exited with code {exit_code}"
        return "nonzero_exit", f"Process exited with code {exit_code}"

    if "memoryerror" in text or "out of memory" in text or "oom" in text:
        return "OOM", "Memory failure signal found near session end"
    if "context length" in text or "context window" in text or "maximum context" in text:
        return "context_window_exceeded", "Context window error found near session end"
    if "timeout" in text or "timed out" in text:
        return "timeout", "Timeout signal found near session end"
    if "traceback" in text:
        return "unhandled_exception", "Python traceback found near session end"
    if "connectionerror" in text or "network" in text or "connection refused" in text:
        return "network_failure", "Network failure signal found near session end"
    if stale:
        return "unknown", "Heartbeat is stale and no clean SESSION_END event was recorded"
    return "unknown", "No specific crash signature detected"


def detect_crash(
    store: TraceStore,
    session_id: str,
    stale_after_seconds: float = DEFAULT_STALE_SECONDS,
    now: float | None = None,
) -> CrashInfo | None:
    """Detect a crashed or stale session without mutating the store."""
    current_time = time.time() if now is None else now
    try:
        meta = store.load_meta(session_id)
        events = store.load_events(session_id)
    except Exception:
        return None

    exit_code = _session_exit_code(events)
    if exit_code not in (None, 0):
        reason, detail = _classify_crash(events, exit_code=exit_code)
    else:
        if meta.ended_at or _has_clean_session_end(events):
            return None
        hb_path = heartbeat_path(store, session_id)
        if not hb_path.exists():
            return None
        stale_seconds = max(0.0, current_time - hb_path.stat().st_mtime)
        if stale_seconds < stale_after_seconds:
            return None
        reason, detail = _classify_crash(events, stale=True)
        return CrashInfo(
            session_id=session_id,
            reason=reason,
            detail=detail,
            stale_seconds=stale_seconds,
            last_event_type=events[-1].event_type.value if events else "",
            last_event_at=events[-1].timestamp if events else meta.started_at,
        )

    return CrashInfo(
        session_id=session_id,
        reason=reason,
        detail=detail,
        exit_code=exit_code,
        last_event_type=events[-1].event_type.value if events else "",
        last_event_at=events[-1].timestamp if events else meta.started_at,
    )


def find_crashed_sessions(
    store: TraceStore,
    stale_after_seconds: float = DEFAULT_STALE_SECONDS,
    now: float | None = None,
) -> list[CrashInfo]:
    """Return crashed/stale sessions sorted newest first."""
    crashes = []
    for meta in store.list_sessions():
        crash = detect_crash(
            store,
            meta.session_id,
            stale_after_seconds=stale_after_seconds,
            now=now,
        )
        if crash:
            crashes.append(crash)
    return crashes


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
            entries.append(TimelineEntry(offset, "Session end", is_root_cause=is_root, is_failure=is_failure))

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


def _file_write_summary(events: list[TraceEvent]) -> list[str]:
    paths: list[str] = []
    for event in events:
        if event.event_type == EventType.TOOL_CALL:
            name = str(event.data.get("tool_name", "")).lower()
            if name not in ("write", "edit", "create", "str_replace"):
                continue
            args = event.data.get("arguments", {}) or {}
            path = str(args.get("file_path") or args.get("path") or "")
            if path and path not in paths:
                paths.append(path)
        elif event.event_type == EventType.FILE_WRITE:
            path = str(event.data.get("uri") or event.data.get("path") or "")
            if path and path not in paths:
                paths.append(path)
    return paths


def _last_prompt(events: list[TraceEvent]) -> str:
    for event in reversed(events):
        if event.event_type in (EventType.USER_PROMPT, EventType.LLM_REQUEST):
            text = event.data.get("prompt") or event.data.get("text") or event.data
            return str(text).replace("\n", " ")[:240]
    return ""


def _build_recovery_context(
    session_id: str,
    events: list[TraceEvent],
    crash: CrashInfo | None,
) -> str:
    reason = crash.reason if crash else "failure"
    detail = crash.detail if crash else "See the root cause and timeline above."
    last_action = _last_action(events)
    prompt = _last_prompt(events)
    writes = _file_write_summary(events)

    lines = [
        f"Previous session {session_id} stopped before a clean completion.",
        f"Crash reason: {reason} - {detail}",
        f"Last recorded action: {last_action}",
    ]
    if prompt:
        lines.append(f"Last prompt/request: {prompt}")
    if writes:
        lines.append("Files modified before the stop:")
        for path in writes[:12]:
            lines.append(f"- {path}")
        if len(writes) > 12:
            lines.append(f"- ... {len(writes) - 12} more")
    lines.append(
        "Resume by inspecting the files above, checking for partial writes, "
        "then continue from the last recorded action."
    )
    return "\n".join(lines)


def write_crash_postmortem(
    store: TraceStore,
    report: PostmortemReport,
) -> Path:
    """Persist a Markdown postmortem for a crashed session."""
    path = store._session_dir(report.session_id) / CRASH_POSTMORTEM_FILE
    lines = [
        f"# Postmortem: {report.session_id}",
        "",
        f"Status: {report.status_summary}",
        f"Root cause: {report.root_cause}",
    ]
    if report.crash_reason:
        lines.append(f"Crash reason: {report.crash_reason}")
    if report.crash_detail:
        lines.append(f"Crash detail: {report.crash_detail}")
    lines.extend([
        "",
        "## Last Timeline",
    ])
    for entry in report.timeline[-20:]:
        marker = " [root cause]" if entry.is_root_cause else ""
        lines.append(f"- {_fmt_offset(entry.offset)} {entry.description}{marker}")
    lines.extend([
        "",
        "## Recovery Context",
        "",
        report.recovery_context or "No recovery context available.",
        "",
        "## Recommendations",
    ])
    for rec in report.recommendations:
        lines.append(f"- {rec}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_session(
    store: TraceStore,
    session_id: str,
    agents_md_path: str | Path = "AGENTS.md",
    stale_after_seconds: float = DEFAULT_STALE_SECONDS,
) -> PostmortemReport:
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    explain = explain_session(store, session_id)
    crash = detect_crash(store, session_id, stale_after_seconds=stale_after_seconds)

    base_ts = events[0].timestamp if events else meta.started_at
    total_seconds = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0.0
    if not total_seconds and events:
        total_seconds = max(0.0, events[-1].timestamp - base_ts)
    failed = crash is not None or any(p.failed for p in explain.phases)

    root_cause_idx, root_cause_desc, root_cause_offset = _find_root_cause(events, base_ts)
    if crash and root_cause_idx < 0:
        root_cause_idx = len(events) - 1 if events else -1
        root_cause_desc = f"Crash: {crash.reason} ({crash.detail})"
        root_cause_offset = (
            max(0.0, crash.last_event_at - base_ts)
            if crash.last_event_at else 0.0
        )

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
    if crash:
        if total_seconds:
            mins = int(total_seconds) // 60
            secs = int(total_seconds) % 60
            duration_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            status_summary = f"CRASHED ({crash.reason} after {duration_str})"
        else:
            status_summary = f"CRASHED ({crash.reason})"
    elif failed:
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
    recovery_context = _build_recovery_context(session_id, events, crash) if failed else ""

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
        crash_reason=crash.reason if crash else "",
        crash_detail=crash.detail if crash else "",
        recovery_context=recovery_context,
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
    if report.crash_reason:
        w(f"Crash:      {report.crash_reason} — {report.crash_detail}\n\n")

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

    if report.recovery_context:
        w("\nRecovery context:\n")
        for line in report.recovery_context.splitlines():
            w(f"  {line}\n" if line else "\n")

    w("\nRecommendations:\n")
    for i, rec in enumerate(report.recommendations, 1):
        w(f"  {i}. {rec}\n")

    w("\n")


def format_crash_list(crashes: list[CrashInfo], out: TextIO = sys.stdout) -> None:
    w = out.write
    if not crashes:
        w("No crashed sessions found.\n")
        return

    w("\nCrashed sessions\n")
    w("────────────────────────────────────────────────────────────\n")
    w(f"{'Session':<18} {'Reason':<22} {'Stale':>8}  Detail\n")
    for crash in crashes:
        stale = f"{crash.stale_seconds:.0f}s" if crash.stale_seconds else "-"
        w(
            f"{crash.session_id:<18} {crash.reason:<22} {stale:>8}  "
            f"{crash.detail[:70]}\n"
        )
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
    stale_after = float(getattr(args, "stale_after", DEFAULT_STALE_SECONDS))

    if getattr(args, "list", False):
        crashes = find_crashed_sessions(store, stale_after_seconds=stale_after)
        for crash in crashes:
            try:
                report = analyze_session(
                    store,
                    crash.session_id,
                    agents_md_path=getattr(args, "agents_md", "AGENTS.md"),
                    stale_after_seconds=stale_after,
                )
                write_crash_postmortem(store, report)
            except Exception:
                pass
        format_crash_list(crashes, out=sys.stdout)
        return 1 if crashes else 0

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
    report = analyze_session(
        store,
        full_id,
        agents_md_path=agents_md,
        stale_after_seconds=stale_after,
    )
    if report.crash_reason:
        pm_path = write_crash_postmortem(store, report)
        sys.stderr.write(f"Post-mortem written: {pm_path}\n")
    format_postmortem(report, out=sys.stdout)
    return 1 if report.failed else 0
