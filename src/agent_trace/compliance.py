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
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .models import EventType, TraceEvent
from .store import TraceStore

Framework = Literal["eu-ai-act", "soc2", "hipaa", "all"]

_FRAMEWORKS: list[str] = ["eu-ai-act", "soc2", "hipaa"]


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
