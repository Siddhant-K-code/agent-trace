"""Compliance export for EU AI Act, SOC 2, and HIPAA audit requirements.

Generates structured JSON reports from session traces that satisfy the
logging and audit trail requirements of common compliance frameworks.

Usage:
    agent-strace compliance export [session-id] --framework eu-ai-act
    agent-strace compliance export [session-id] --framework soc2
    agent-strace compliance export [session-id] --framework hipaa
    agent-strace compliance export [session-id] --framework all

    # Export all sessions in a time window
    agent-strace compliance export --since 30d --framework soc2 --output report.json

Frameworks:
    eu-ai-act   Article 13 transparency + Article 9 risk management logging
    soc2        CC6 (logical access), CC7 (system operations) evidence
    hipaa       §164.312 audit controls, access logs, integrity controls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TextIO

from . import __version__
from .audit import verify_chain
from .cost import _dollars, _event_tokens
from .models import EventType, TraceEvent
from .store import TraceStore

Framework = Literal["eu-ai-act", "soc2", "hipaa", "all"]

_FRAMEWORKS: list[str] = ["eu-ai-act", "soc2", "hipaa"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time_bound(value: str | None, default: float | None = None) -> float | None:
    if not value:
        return default
    raw = value.strip()
    if not raw:
        return default
    if raw.endswith("d") and raw[:-1].replace(".", "", 1).isdigit():
        return time.time() - float(raw[:-1]) * 86400
    if raw.endswith("h") and raw[:-1].replace(".", "", 1).isdigit():
        return time.time() - float(raw[:-1]) * 3600
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return default


def _load_event_lines(store: TraceStore, session_id: str) -> list[str]:
    path = store._session_dir(session_id) / "events.ndjson"
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line.strip()]


def _root_hash(lines: list[str]) -> str:
    if not lines:
        return ""
    return hashlib.sha256(lines[-1].encode()).hexdigest()


def _line_hashes(lines: list[str]) -> list[str]:
    return [hashlib.sha256(line.encode()).hexdigest() for line in lines]


def _summarise(value: Any, limit: int = 240) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text[:limit] + ("..." if len(text) > limit else "")


def _event_tool_name(event: TraceEvent) -> str:
    return str(event.data.get("tool_name") or event.data.get("tool") or "")


def _event_cost(event: TraceEvent) -> float:
    input_tokens, output_tokens = _event_tokens(event)
    return _dollars(input_tokens, output_tokens, "sonnet")


def _data_categories(events: list[TraceEvent]) -> list[str]:
    categories = set()
    for event in events:
        if event.event_type in (EventType.LLM_REQUEST, EventType.USER_PROMPT):
            categories.add("prompt_content")
        elif event.event_type in (EventType.LLM_RESPONSE, EventType.ASSISTANT_RESPONSE):
            categories.add("model_output")
        elif event.event_type in (EventType.TOOL_CALL, EventType.TOOL_RESULT):
            categories.add("tool_io")
        elif event.event_type in (EventType.FILE_READ, EventType.FILE_WRITE):
            categories.add("file_content")
        elif event.event_type == EventType.ERROR:
            categories.add("error_log")
    return sorted(categories)


def _human_oversight_points(store: TraceStore, session_id: str, events: list[TraceEvent]) -> list[dict]:
    points = []
    for event in events:
        if event.event_type == EventType.ERROR:
            points.append({
                "timestamp": _iso(event.timestamp),
                "type": "ERROR_RECORDED",
                "detail": _summarise(event.data, 180),
            })
        elif event.data.get("blocked") or event.data.get("policy_violation"):
            points.append({
                "timestamp": _iso(event.timestamp),
                "type": "POLICY_INTERVENTION",
                "detail": _summarise(event.data, 180),
            })

    session_dir = store._session_dir(session_id)
    if (session_dir / "watchdog-postmortem.json").exists():
        points.append({
            "timestamp": None,
            "type": "WATCHDOG_POSTMORTEM",
            "detail": "watchdog-postmortem.json present",
        })
    if (session_dir / "postmortem.md").exists():
        points.append({
            "timestamp": None,
            "type": "CRASH_POSTMORTEM",
            "detail": "postmortem.md present",
        })
    return points


def _models_used(events: list[TraceEvent]) -> list[str]:
    models = set()
    for event in events:
        model = event.data.get("model") or event.data.get("model_name")
        if model:
            models.add(str(model))
    return sorted(models)


# ---------------------------------------------------------------------------
# Per-framework report builders
# ---------------------------------------------------------------------------

def _eu_ai_act_report(session_id: str, meta, events: list[TraceEvent]) -> dict:
    """EU AI Act Article 13 (transparency) + Article 9 (risk management) evidence."""
    tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
    errors = [e for e in events if e.event_type == EventType.ERROR]
    decisions = [e for e in events if e.event_type == EventType.DECISION]

    return {
        "framework": "eu-ai-act",
        "generated_at": time.time(),
        "session_id": session_id,
        "article_13_transparency": {
            "agent_name": meta.agent_name,
            "session_start": meta.started_at,
            "session_end": meta.ended_at,
            "total_tool_calls": len(tool_calls),
            "total_decisions": len(decisions),
            "tools_used": sorted({e.data.get("tool_name", "") for e in tool_calls if e.data.get("tool_name")}),
        },
        "article_9_risk_management": {
            "error_count": len(errors),
            "error_types": [e.data.get("error_type", "unknown") for e in errors],
            "anomalous_tool_calls": [
                {"event_id": e.event_id, "tool": e.data.get("tool_name"), "ts": e.timestamp}
                for e in tool_calls
                if e.data.get("blocked") or e.data.get("policy_violation")
            ],
        },
        "chain_integrity": {
            "hash_chain_present": any(e.prev_hash for e in events),
        },
    }


def _soc2_report(session_id: str, meta, events: list[TraceEvent]) -> dict:
    """SOC 2 CC6 (logical access) + CC7 (system operations) evidence."""
    tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
    errors = [e for e in events if e.event_type == EventType.ERROR]
    llm_reqs = [e for e in events if e.event_type == EventType.LLM_REQUEST]

    # CC6: logical access — what resources were accessed
    resources_accessed = sorted({
        e.data.get("tool_name", "") for e in tool_calls
        if e.data.get("tool_name")
    })

    # CC7: system operations — availability and error evidence
    return {
        "framework": "soc2",
        "generated_at": time.time(),
        "session_id": session_id,
        "cc6_logical_access": {
            "agent_identity": meta.agent_name,
            "workspace": getattr(meta, "workspace_id", "") or "",
            "team": getattr(meta, "team", "") or "",
            "session_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(meta.started_at)),
            "session_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(meta.ended_at)) if meta.ended_at else None,
            "resources_accessed": resources_accessed,
            "total_operations": len(tool_calls),
        },
        "cc7_system_operations": {
            "llm_requests": len(llm_reqs),
            "errors": len(errors),
            "error_details": [
                {"event_id": e.event_id, "error": e.data.get("error", ""), "ts": e.timestamp}
                for e in errors
            ],
            "duration_ms": meta.total_duration_ms or 0.0,
        },
    }


def _hipaa_report(session_id: str, meta, events: list[TraceEvent]) -> dict:
    """HIPAA §164.312 audit controls and access log evidence."""
    tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
    errors = [e for e in events if e.event_type == EventType.ERROR]

    # Build access log entries
    access_log = []
    for e in tool_calls:
        access_log.append({
            "event_id": e.event_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e.timestamp)),
            "action": e.data.get("tool_name", "unknown"),
            "outcome": "success",
            "duration_ms": e.duration_ms,
        })
    for e in errors:
        access_log.append({
            "event_id": e.event_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e.timestamp)),
            "action": e.data.get("tool_name", "error"),
            "outcome": "failure",
            "error": e.data.get("error", ""),
        })
    access_log.sort(key=lambda x: x["timestamp_utc"])

    return {
        "framework": "hipaa",
        "generated_at": time.time(),
        "session_id": session_id,
        "section_164_312_audit_controls": {
            "agent_name": meta.agent_name,
            "session_id": session_id,
            "session_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(meta.started_at)),
            "session_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(meta.ended_at)) if meta.ended_at else None,
            "access_log": access_log,
            "total_accesses": len(access_log),
            "failures": len(errors),
        },
        "section_164_312_integrity": {
            "hash_chain_present": any(e.prev_hash for e in events),
            "event_count": len(events),
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_compliance(
    store: TraceStore,
    session_id: str,
    framework: Framework = "all",
) -> dict:
    """Build a compliance report for one session.

    Returns a dict with one key per framework requested.
    """
    try:
        meta = store.load_meta(session_id)
    except Exception:
        return {"error": f"session not found: {session_id}"}
    try:
        events = store.load_events(session_id)
    except Exception:
        events = []

    frameworks = _FRAMEWORKS if framework == "all" else [framework]
    report: dict = {
        "session_id": session_id,
        "exported_at": time.time(),
        "frameworks": {},
    }
    for fw in frameworks:
        if fw == "eu-ai-act":
            report["frameworks"][fw] = _eu_ai_act_report(session_id, meta, events)
        elif fw == "soc2":
            report["frameworks"][fw] = _soc2_report(session_id, meta, events)
        elif fw == "hipaa":
            report["frameworks"][fw] = _hipaa_report(session_id, meta, events)

    return report


def export_compliance_bulk(
    store: TraceStore,
    framework: Framework = "all",
    since_days: float = 30.0,
) -> list[dict]:
    """Export compliance reports for all sessions in the last N days."""
    cutoff = time.time() - since_days * 86400
    reports = []
    for meta in store.list_sessions():
        if not meta.ended_at or meta.started_at < cutoff:
            continue
        reports.append(export_compliance(store, meta.session_id, framework))
    return reports


# ---------------------------------------------------------------------------
# EU AI Act Article 12/13 export
# ---------------------------------------------------------------------------

def _eu_ai_act_session_record(store: TraceStore, session_id: str) -> dict:
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    lines = _load_event_lines(store, session_id)
    hashes = _line_hashes(lines)
    chain = verify_chain(store, session_id)

    event_records = []
    for idx, event in enumerate(events):
        input_summary = ""
        output_summary = ""
        if event.event_type in (EventType.USER_PROMPT, EventType.LLM_REQUEST, EventType.TOOL_CALL):
            input_summary = _summarise(event.data)
        elif event.event_type in (EventType.ASSISTANT_RESPONSE, EventType.LLM_RESPONSE, EventType.TOOL_RESULT):
            output_summary = _summarise(event.data)
        else:
            input_summary = _summarise(event.data)

        event_records.append({
            "seq": idx + 1,
            "timestamp": _iso(event.timestamp),
            "event_type": event.event_type.value,
            "event_id": event.event_id,
            "tool_name": _event_tool_name(event),
            "input_summary": input_summary,
            "output_summary": output_summary,
            "duration_ms": event.duration_ms,
            "cost_usd": round(_event_cost(event), 8),
            "prev_hash": event.prev_hash,
            "line_hash": hashes[idx] if idx < len(hashes) else "",
        })

    tools = sorted({
        _event_tool_name(event)
        for event in events
        if event.event_type == EventType.TOOL_CALL and _event_tool_name(event)
    })
    errors = [event for event in events if event.event_type == EventType.ERROR]
    context_resets = [
        event for event in events
        if "context" in json.dumps(event.data, default=str).lower()
        and "reset" in json.dumps(event.data, default=str).lower()
    ]

    article_12 = {
        "system_id": session_id,
        "generation_timestamp": _iso(time.time()),
        "trace_integrity": {
            "hash_chain_valid": chain.ok,
            "root_hash": _root_hash(lines),
            "event_count": len(events),
            "broken_at": chain.broken_at,
            "broken_event_id": chain.broken_event_id,
            "verification_command": f"agent-strace verify {session_id}",
        },
        "events": event_records,
        "data_categories_processed": _data_categories(events),
        "retention_period_days": None,
    }

    article_13 = {
        "system_description": "AI agent session trace",
        "intended_purpose": meta.command or meta.agent_name or "not specified",
        "capabilities_summary": {
            "tools_available": tools,
            "tools_used": tools,
            "models_used": _models_used(events),
            "frameworks_detected": [],
        },
        "human_oversight_points": _human_oversight_points(store, session_id, events),
        "limitations": {
            "context_resets": len(context_resets),
            "errors_encountered": len(errors),
            "unresolved_failures": 0 if meta.ended_at else 1,
        },
    }

    return {
        "session_id": session_id,
        "session_start": _iso(meta.started_at),
        "session_end": _iso(meta.ended_at),
        "agent_name": meta.agent_name,
        "article_12": article_12,
        "article_13": article_13,
    }


def select_sessions(
    store: TraceStore,
    since: str | None = None,
    until: str | None = None,
) -> list[str]:
    """Select session IDs in a time window, newest first."""
    since_ts = _parse_time_bound(since, None)
    until_ts = _parse_time_bound(until, None)
    selected = []
    for meta in store.list_sessions():
        if since_ts is not None and meta.started_at < since_ts:
            continue
        if until_ts is not None and meta.started_at > until_ts:
            continue
        selected.append(meta.session_id)
    return selected


def export_eu_ai_act(
    store: TraceStore,
    session_ids: list[str],
) -> dict:
    """Build an EU AI Act Article 12/13 JSON export for one or more sessions."""
    sessions = [_eu_ai_act_session_record(store, session_id) for session_id in session_ids]
    article_12_sessions = [
        {
            "session_id": session["session_id"],
            **session["article_12"],
        }
        for session in sessions
    ]
    article_13_sessions = [
        {
            "session_id": session["session_id"],
            **session["article_13"],
        }
        for session in sessions
    ]

    return {
        "compliance_metadata": {
            "regulation": "EU AI Act (Regulation (EU) 2024/1689)",
            "articles_covered": ["Article 12", "Article 13"],
            "export_tool": "agent-strace",
            "export_version": __version__,
            "exported_at": _iso(time.time()),
            "tamper_evident": True,
            "session_count": len(sessions),
        },
        "article_12": {
            "logging_obligations": article_12_sessions,
        },
        "article_13": {
            "transparency_documentation": article_13_sessions,
        },
        "sessions": sessions,
    }


def cmd_export_eu_ai_act(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    if getattr(args, "all", False):
        session_ids = select_sessions(
            store,
            since=getattr(args, "since", None),
            until=getattr(args, "until", None),
        )
    else:
        session_id = getattr(args, "session_id", None) or store.get_latest_session_id()
        if not session_id:
            sys.stderr.write("No sessions found.\n")
            return 1
        full_id = store.find_session(session_id) or session_id
        if not store.session_exists(full_id):
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1
        session_ids = [full_id]

    report = export_eu_ai_act(store, session_ids)
    result = json.dumps(report, indent=2)
    output = getattr(args, "output", "") or ""
    if output:
        Path(output).write_text(result + "\n")
        sys.stderr.write(f"EU AI Act export written to {output}\n")
    else:
        sys.stdout.write(result + "\n")
    return 0


# ---------------------------------------------------------------------------
# Readiness and export verification
# ---------------------------------------------------------------------------

def build_audit_readiness(
    store: TraceStore,
    retention_days: float = 90.0,
) -> dict:
    sessions = store.list_sessions()
    chain_results = [verify_chain(store, meta.session_id) for meta in sessions]
    broken = [result for result in chain_results if not result.ok]
    now = time.time()
    oldest = min((meta.started_at for meta in sessions), default=None)

    sorted_sessions = sorted(sessions, key=lambda meta: meta.started_at)
    gaps = []
    for prev, cur in zip(sorted_sessions, sorted_sessions[1:]):
        gap = cur.started_at - prev.started_at
        if gap > 86400:
            gaps.append({
                "from_session": prev.session_id,
                "to_session": cur.session_id,
                "gap_hours": round(gap / 3600, 2),
            })

    missing_hash = []
    for meta in sessions:
        events = store.load_events(meta.session_id)
        if len(events) > 1 and not any(event.prev_hash for event in events[1:]):
            missing_hash.append(meta.session_id)

    retention_days_present = ((now - oldest) / 86400) if oldest else 0.0
    checks = {
        "hash_chain_integrity": {
            "ok": not broken,
            "checked_sessions": len(chain_results),
            "broken_sessions": [result.session_id for result in broken],
        },
        "retention_coverage": {
            "ok": retention_days_present >= retention_days if sessions else False,
            "required_days": retention_days,
            "available_days": round(retention_days_present, 2),
        },
        "timestamp_continuity": {
            "ok": not gaps,
            "gaps_over_24h": gaps,
        },
        "hash_chain_presence": {
            "ok": not missing_hash,
            "sessions_missing_prev_hashes": missing_hash,
        },
    }
    score = 100
    if broken:
        score -= 35
    if missing_hash:
        score -= 20
    if gaps:
        score -= 10
    if sessions and retention_days_present < retention_days:
        score -= 10
    if not sessions:
        score = 0

    return {
        "framework": "eu-ai-act",
        "articles_checked": ["Article 12", "Article 13"],
        "generated_at": _iso(now),
        "session_count": len(sessions),
        "checks": checks,
        "compliance_score": max(0, score),
        "ready": score >= 80 and not broken,
    }


def format_audit_readiness(report: dict, out: TextIO = sys.stdout) -> None:
    checks = report["checks"]
    out.write("EU AI Act readiness check\n")
    out.write("-------------------------\n")
    out.write(f"Sessions analysed: {report['session_count']}\n")
    out.write(
        f"{'OK' if checks['hash_chain_integrity']['ok'] else 'FAIL'} "
        f"Hash chain integrity: {checks['hash_chain_integrity']['checked_sessions']} sessions checked\n"
    )
    out.write(
        f"{'OK' if checks['retention_coverage']['ok'] else 'WARN'} "
        f"Retention coverage: {checks['retention_coverage']['available_days']} / "
        f"{checks['retention_coverage']['required_days']} days\n"
    )
    out.write(
        f"{'OK' if checks['timestamp_continuity']['ok'] else 'WARN'} "
        f"Timestamp continuity: {len(checks['timestamp_continuity']['gaps_over_24h'])} gaps over 24h\n"
    )
    out.write(
        f"{'OK' if checks['hash_chain_presence']['ok'] else 'WARN'} "
        f"Hash chain presence: {len(checks['hash_chain_presence']['sessions_missing_prev_hashes'])} legacy sessions\n"
    )
    out.write(f"\nCompliance score: {report['compliance_score']}/100\n")


def cmd_audit_readiness(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    report = build_audit_readiness(
        store,
        retention_days=float(getattr(args, "retention_days", 90.0)),
    )
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        format_audit_readiness(report)
    return 0 if report["ready"] else 1


def verify_eu_ai_act_export(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    sessions = data.get("sessions", [])
    failures = []
    checked_events = 0
    for session in sessions:
        events = session.get("article_12", {}).get("events", [])
        checked_events += len(events)
        for idx, event in enumerate(events):
            if idx == 0:
                continue
            prev_hash = event.get("prev_hash", "")
            previous_line_hash = events[idx - 1].get("line_hash", "")
            if prev_hash and previous_line_hash and prev_hash != previous_line_hash:
                failures.append({
                    "session_id": session.get("session_id", ""),
                    "seq": event.get("seq"),
                    "event_id": event.get("event_id", ""),
                })
    return {
        "ok": not failures,
        "checked_sessions": len(sessions),
        "checked_events": checked_events,
        "failures": failures,
    }


def cmd_verify_export(args: argparse.Namespace) -> int:
    result = verify_eu_ai_act_export(getattr(args, "from_export"))
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    elif result["ok"]:
        sys.stdout.write(
            f"Export hash chain intact - {result['checked_events']} events checked\n"
        )
    else:
        sys.stdout.write(
            f"Export hash chain failed - {len(result['failures'])} broken links\n"
        )
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_compliance(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    sub = getattr(args, "compliance_cmd", None)

    if sub == "export":
        framework = getattr(args, "framework", "all") or "all"
        output = getattr(args, "output", "") or ""
        since_str = getattr(args, "since", "") or ""
        session_id = getattr(args, "session_id", None)

        if since_str or not session_id:
            # Bulk export
            since_days = float(since_str.rstrip("d")) if since_str else 30.0
            reports = export_compliance_bulk(store, framework=framework,
                                             since_days=since_days)
            result = json.dumps(reports, indent=2)
        else:
            full_id = store.find_session(session_id)
            if not full_id:
                sys.stderr.write(f"Session not found: {session_id}\n")
                return 1
            report = export_compliance(store, full_id, framework=framework)
            result = json.dumps(report, indent=2)

        if output:
            Path(output).write_text(result + "\n")
            sys.stdout.write(f"Compliance report written to {output}\n")
        else:
            sys.stdout.write(result + "\n")
        return 0

    else:
        sys.stderr.write("Usage: agent-strace compliance export [session-id] "
                         "--framework eu-ai-act|soc2|hipaa|all\n")
        return 1
