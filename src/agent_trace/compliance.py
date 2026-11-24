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
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TextIO

from . import __version__
from .audit import verify_chain
from .cost import _usage_dollars, session_event_usage
from .event_data import (
    NonObjectEventDataError,
    UnusableTimestampError,
    duration_ms_of,
    event_id_of,
    require_object_event_data,
    require_usable_session_times,
    require_usable_timestamps,
    str_field,
)
from .fileutil import split_lf_lines
from .inert import inert
from .models import (
    EventType,
    SessionMeta,
    TraceEvent,
    usable_number,
    usable_text,
    usable_time,
)
from .store import (
    _TENANT_ADMIN_AUXILIARY_DIRS,
    CorruptSessionMetaError,
    RefusedSessionError,
    TraceStore,
    refusal_reason,
    refuse_lines_not_utf8,
    validate_stored_id,
)

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
    path = store._session_file(session_id, "events.ndjson", "event stream")
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    refuse_lines_not_utf8(session_id, text)
    return [line for line in split_lf_lines(text) if line.strip()]


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


def _event_costs(events: list[TraceEvent]) -> list[float]:
    return [_usage_dollars(usage, "sonnet") for usage in session_event_usage(events)]


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
            "agent_name": usable_text(meta.agent_name),
            "session_start": meta.started_at,
            "session_end": meta.ended_at,
            "total_tool_calls": len(tool_calls),
            "total_decisions": len(decisions),
            "tools_used": sorted({
                str_field(e.data, "tool_name", "") for e in tool_calls if str_field(e.data, "tool_name", "")
            }),
        },
        "article_9_risk_management": {
            "error_count": len(errors),
            "error_types": [e.data.get("error_type", "unknown") for e in errors],
            "anomalous_tool_calls": [
                {"event_id": event_id_of(e), "tool": e.data.get("tool_name"), "ts": e.timestamp}
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
        str_field(e.data, "tool_name", "") for e in tool_calls
        if str_field(e.data, "tool_name", "")
    })

    # CC7: system operations — availability and error evidence
    return {
        "framework": "soc2",
        "generated_at": time.time(),
        "session_id": session_id,
        "cc6_logical_access": {
            "agent_identity": usable_text(meta.agent_name),
            "workspace": getattr(meta, "workspace_id", "") or "",
            "team": usable_text(getattr(meta, "team", "")),
            "session_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(meta.started_at)),
            "session_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(meta.ended_at)) if meta.ended_at else None,
            "resources_accessed": resources_accessed,
            "total_operations": len(tool_calls),
        },
        "cc7_system_operations": {
            "llm_requests": len(llm_reqs),
            "errors": len(errors),
            "error_details": [
                {"event_id": event_id_of(e), "error": e.data.get("error", ""), "ts": e.timestamp}
                for e in errors
            ],
            "duration_ms": meta.total_duration_ms if usable_number(meta.total_duration_ms) else 0.0,
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
            "event_id": event_id_of(e),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e.timestamp)),
            "action": e.data.get("tool_name", "unknown"),
            "outcome": "success",
            "duration_ms": duration_ms_of(e),
        })
    for e in errors:
        access_log.append({
            "event_id": event_id_of(e),
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
            "agent_name": usable_text(meta.agent_name),
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

# Every framework report reads the data of every event of these types.
COMPLIANCE_DATA_EVENT_TYPES = frozenset({EventType.TOOL_CALL, EventType.ERROR})


def export_compliance(
    store: TraceStore,
    session_id: str,
    framework: Framework = "all",
    *,
    name_session: bool = False,
) -> dict:
    """Build a compliance report for one session.

    Returns a dict with one key per framework requested, or a dict with an
    ``error`` key when the session has no meta.json or no events.ndjson.
    Raises CorruptSessionMetaError or RefusedSessionError when the store
    refuses to read the session, so a refused session is never reported as
    empty. A symlinked meta.json or events.ndjson is such a refusal. An
    OSError, such as a permission error, propagates. Raises
    NonObjectEventDataError when a tool_call or error event's
    data is not an object, and UnusableTimestampError when such an event's
    timestamp, the session's started_at, or its ended_at when set, is not a
    Unix time in seconds; ``name_session`` adds the session ID to either.
    """
    try:
        meta = store.load_meta(session_id)
        events = store.load_events(session_id)
    except FileNotFoundError:
        return {"error": f"session not found: {session_id}"}
    except (CorruptSessionMetaError, RefusedSessionError):
        raise
    except ValueError as exc:
        raise RefusedSessionError(session_id, refusal_reason(exc)) from exc
    require_object_event_data(
        events,
        event_types=COMPLIANCE_DATA_EVENT_TYPES,
        session_id=session_id if name_session else None,
    )
    require_usable_timestamps(
        events,
        event_types=COMPLIANCE_DATA_EVENT_TYPES,
        session_id=session_id if name_session else None,
    )
    require_usable_session_times(meta, session_id=session_id if name_session else None)

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


def _read_refusal_reason(exc: BaseException) -> str:
    """Return why a session could not be read, without the session ID."""
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return refusal_reason(exc)


def _export_sessions(
    store: TraceStore, *, validate_events: bool = True
) -> list[SessionMeta]:
    """Return every session in the store for a bulk export, newest first.

    A bulk export fails closed. A session the store refuses stops the export
    with an error that names the session and says why, so the export never
    leaves the session out or reports it as empty. The exports do not filter
    by tenant. So, as in ``load_events``, an event may carry a tenant when
    its session has none.

    With ``validate_events=False`` the events are not read. The scores
    ``--since`` batch in langfuse_export uses it, so ``--metrics`` still
    exports a session with an unreadable line, without ``cost_usd``.
    """
    prefix = "corrupt session in tenant store: "
    try:
        return store.list_sessions_strict(
            strict_event_tenants=False, validate_events=validate_events
        )
    except ValueError as exc:
        cause = exc.__cause__
        if isinstance(cause, (CorruptSessionMetaError, RefusedSessionError)):
            raise cause from cause.__cause__
        message = str(exc)
        if cause is None or not message.startswith(prefix):
            raise
        raise RefusedSessionError(
            message[len(prefix):], _read_refusal_reason(cause)
        ) from cause


def export_compliance_bulk(
    store: TraceStore,
    framework: Framework = "all",
    since_days: float = 30.0,
) -> list[dict]:
    """Export compliance reports for the ended sessions of the last N days.

    Raises a ValueError, naming the session or store entry, when the store
    refuses any session or entry in it, even one outside the window.
    """
    cutoff = time.time() - since_days * 86400
    reports = []
    for meta in _export_sessions(store):
        if not meta.ended_at or meta.started_at < cutoff:
            continue
        reports.append(export_compliance(store, meta.session_id, framework, name_session=True))
    return reports


# ---------------------------------------------------------------------------
# EU AI Act Article 12/13 export
# ---------------------------------------------------------------------------

def _eu_ai_act_session_record(
    store: TraceStore,
    session_id: str,
    *,
    name_session: bool = False,
) -> dict:
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    # The record reads the tool name from the data of every event, the
    # timestamp of every event, and the session's start and end times.
    require_object_event_data(events, session_id=session_id if name_session else None)
    require_usable_timestamps(events, session_id=session_id if name_session else None)
    require_usable_session_times(meta, session_id=session_id if name_session else None)
    lines = _load_event_lines(store, session_id)
    hashes = _line_hashes(lines)
    chain = verify_chain(store, session_id)

    event_records = []
    costs = _event_costs(events)
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
            "event_id": event_id_of(event),
            "tool_name": _event_tool_name(event),
            "input_summary": input_summary,
            "output_summary": output_summary,
            "duration_ms": duration_ms_of(event),
            "cost_usd": round(costs[idx], 8),
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
            "hash_chain_state": chain.state,
            "root_hash": _root_hash(lines),
            "event_count": len(events),
            "broken_at": chain.broken_at,
            "broken_event_id": usable_text(chain.broken_event_id),
            "verification_command": f"agent-strace verify {session_id}",
        },
        "events": event_records,
        "data_categories_processed": _data_categories(events),
        "retention_period_days": None,
    }

    article_13 = {
        "system_description": "AI agent session trace",
        "intended_purpose": usable_text(meta.command) or usable_text(meta.agent_name) or "not specified",
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
        "agent_name": usable_text(meta.agent_name),
        "article_12": article_12,
        "article_13": article_13,
    }


def select_sessions(
    store: TraceStore,
    since: str | None = None,
    until: str | None = None,
) -> list[str]:
    """Select session IDs in a time window, newest first.

    Raises a ValueError, naming the session or store entry, when the store
    refuses any session or entry in it, even one outside the window.
    """
    since_ts = _parse_time_bound(since, None)
    until_ts = _parse_time_bound(until, None)
    selected = []
    for meta in _export_sessions(store):
        if since_ts is not None and meta.started_at < since_ts:
            continue
        if until_ts is not None and meta.started_at > until_ts:
            continue
        selected.append(meta.session_id)
    return selected


def export_eu_ai_act(
    store: TraceStore,
    session_ids: list[str],
    *,
    name_sessions: bool = False,
) -> dict:
    """Build an EU AI Act Article 12/13 JSON export for one or more sessions.

    Raises NonObjectEventDataError when any event's data is not an object,
    and UnusableTimestampError when any event's timestamp, a session's
    started_at, or its ended_at when set, is not a Unix time in seconds;
    ``name_sessions`` adds the session ID to either.
    """
    sessions = [
        _eu_ai_act_session_record(store, session_id, name_session=name_sessions)
        for session_id in session_ids
    ]
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
    tamper_evident = bool(sessions) and all(
        session["article_12"]["trace_integrity"]["hash_chain_valid"] is True
        for session in sessions
    )

    return {
        "compliance_metadata": {
            "regulation": "EU AI Act (Regulation (EU) 2024/1689)",
            "articles_covered": ["Article 12", "Article 13"],
            "export_tool": "agent-strace",
            "export_version": __version__,
            "exported_at": _iso(time.time()),
            "tamper_evident": tamper_evident,
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
        try:
            session_ids = select_sessions(
                store,
                since=getattr(args, "since", None),
                until=getattr(args, "until", None),
            )
        except ValueError as exc:
            # A store entry's name can hold any character.
            sys.stderr.write(f"Error: {inert(str(exc))}\n")
            return 1
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

    report = export_eu_ai_act(
        store, session_ids, name_sessions=bool(getattr(args, "all", False))
    )
    result = json.dumps(report, indent=2)
    output = getattr(args, "output", "") or ""
    if output:
        try:
            Path(output).write_text(result + "\n")
        except OSError as exc:
            sys.stderr.write(f"Error: cannot write {output}: {exc.strerror or exc}\n")
            return 1
        sys.stderr.write(f"EU AI Act export written to {output}\n")
    else:
        sys.stdout.write(result + "\n")
    return 0


# ---------------------------------------------------------------------------
# Readiness and export verification
# ---------------------------------------------------------------------------

def _readiness_sessions(
    store: TraceStore,
) -> tuple[list[SessionMeta], list[dict[str, str]]]:
    """Return the sessions to check and the store entries that fail the scan.

    The scan skips only the entries that ``list_sessions_strict`` skips. It
    reads each session's events as the bulk exports do. It then checks the
    data and timestamp of every event, and the session's ended_at when set,
    as ``export --format eu-ai-act`` does.
    A session with a readable meta.json and a usable started_at is in both
    lists when its events.ndjson is missing or refused, or when it fails a
    check. Each failed entry holds the entry's name and the reason. Unlike
    ``list_sessions_strict``, the scan reads a trace directory that is a
    symlink.
    """
    sessions: list[SessionMeta] = []
    refused: list[dict[str, str]] = []
    if not store.base_dir.is_dir():
        return sessions, refused
    for entry in sorted(store.base_dir.iterdir(), key=lambda item: item.name):
        if not entry.is_symlink() and not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        events_path = entry / "events.ndjson"
        if (
            not entry.is_symlink()
            and (
                entry.name in _TENANT_ADMIN_AUXILIARY_DIRS
                or (not store.workspace_id and entry.name == "workspaces")
            )
            and not (meta_path.exists() or events_path.exists())
        ):
            continue
        try:
            session_id = validate_stored_id(entry.name, "stored session directory")
        except ValueError:
            refused.append(
                {"session_id": entry.name, "reason": "not a valid session directory name"}
            )
            continue
        try:
            meta = store.load_meta(session_id)
        except FileNotFoundError:
            refused.append({"session_id": session_id, "reason": "no meta.json"})
            continue
        except (OSError, ValueError, TypeError) as exc:
            refused.append({"session_id": session_id, "reason": _read_refusal_reason(exc)})
            continue
        if usable_time(meta.started_at) is None:
            refused.append({"session_id": session_id, "reason": "started_at is not a usable time"})
            continue
        sessions.append(meta)
        try:
            # The bulk exports read the events this way, so a tenant ID they
            # refuse fails the check too.
            events = store.load_events(session_id, validate_tenant_ids=True)
        except FileNotFoundError:
            refused.append({"session_id": session_id, "reason": "no events.ndjson"})
            continue
        except (OSError, ValueError, TypeError) as exc:
            refused.append({"session_id": session_id, "reason": _read_refusal_reason(exc)})
            continue
        try:
            # The EU AI Act export reads the data and timestamp of every
            # event, and the session's start and end times.
            require_object_event_data(events, session_id=session_id)
            require_usable_timestamps(events, session_id=session_id)
            require_usable_session_times(meta, session_id=session_id)
        except (NonObjectEventDataError, UnusableTimestampError) as exc:
            reason = str(exc).removeprefix(f"session {session_id} ")
            refused.append({"session_id": session_id, "reason": reason})
    sessions.sort(key=lambda meta: (meta.started_at, meta.session_id), reverse=True)
    return sessions, refused


def build_audit_readiness(
    store: TraceStore,
    retention_days: float = 90.0,
) -> dict:
    """Check whether the store's sessions are ready for an EU AI Act audit.

    A session the store refuses to read fails the hash chain integrity
    check, with its reason in ``refused_sessions``. So does a session with an
    event whose data or timestamp the EU AI Act export refuses, or with an
    ended_at it refuses. So the check never passes while it leaves out a
    session.
    """
    sessions, refused = _readiness_sessions(store)
    refused_ids = [entry["session_id"] for entry in refused]
    chain_results = [
        verify_chain(store, meta.session_id)
        for meta in sessions
        if meta.session_id not in refused_ids
    ]
    broken = [result for result in chain_results if result.broken]
    integrity_failed = bool(broken or refused)
    session_count = len({meta.session_id for meta in sessions} | set(refused_ids))
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

    missing_hash = [
        result.session_id
        for result in chain_results
        if result.state in ("legacy", "partial", "empty")
    ]

    retention_days_present = ((now - oldest) / 86400) if oldest else 0.0
    checks = {
        "hash_chain_integrity": {
            "ok": not integrity_failed,
            "checked_sessions": session_count,
            "broken_sessions": [result.session_id for result in broken] + refused_ids,
            "refused_sessions": refused,
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
    if integrity_failed:
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
        "session_count": session_count,
        "checks": checks,
        "compliance_score": max(0, score),
        "ready": score >= 80 and not integrity_failed and not missing_hash,
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
    for entry in checks["hash_chain_integrity"]["refused_sessions"]:
        out.write(f"  cannot read session {inert(entry['session_id'])}: {entry['reason']}\n")
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
        f"Hash chain presence: {len(checks['hash_chain_presence']['sessions_missing_prev_hashes'])} sessions with missing links\n"
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


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_HEX.fullmatch(value) is not None


def _check_export_events(events: list, session_id: Any, fail) -> None:
    previous_line_hash: str | None = None
    for position, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            fail("event is not an object", session_id, position, "")
            previous_line_hash = None
            continue
        event_id = event.get("event_id", "")
        seq = event.get("seq")
        if type(seq) is not int or seq != position:
            fail("seq is not the event's position", session_id, position, event_id)
        prev_hash = event.get("prev_hash", "")
        if position == 1:
            if prev_hash != "":
                fail("first event has a prev_hash", session_id, position, event_id)
        elif prev_hash == "":
            fail("prev_hash is missing", session_id, position, event_id)
        elif not _is_sha256_hex(prev_hash):
            fail("prev_hash is not a SHA-256 hex digest", session_id, position, event_id)
        elif previous_line_hash is not None and prev_hash != previous_line_hash:
            fail(
                "prev_hash does not match the previous event's line_hash",
                session_id, position, event_id,
            )
        line_hash = event.get("line_hash")
        if _is_sha256_hex(line_hash):
            previous_line_hash = line_hash
        else:
            fail("line_hash is not a SHA-256 hex digest", session_id, position, event_id)
            previous_line_hash = None


def _check_export_trace_integrity(
    trace_integrity: dict, events: list, session_id: Any, fail,
) -> None:
    if trace_integrity.get("hash_chain_valid") is not True:
        fail("hash_chain_valid is not true", session_id)
    # Exports made before hash_chain_state was added don't have it.
    if (
        "hash_chain_state" in trace_integrity
        and trace_integrity["hash_chain_state"] != "unanchored"
    ):
        fail("hash_chain_state is not unanchored", session_id)
    if events and isinstance(events[-1], dict):
        if trace_integrity.get("root_hash") != events[-1].get("line_hash"):
            fail("root_hash does not match the last event's line_hash", session_id)
    event_count = trace_integrity.get("event_count")
    if type(event_count) is not int or event_count != len(events):
        fail("event_count does not match the number of events", session_id)


def verify_eu_ai_act_export(path: str | Path) -> dict:
    """Check that an EU AI Act export holds complete, consistent hash chains.

    Each failure names the ``session_id``, the event's position in the session
    as ``seq`` (counting from 1, so an honest export has the same number in
    the event's own ``seq``) and its ``event_id``, and gives a ``reason``.
    ``session_id`` is None for a problem with the whole package. ``seq`` and
    ``event_id`` are None for a problem that is not about one event. The
    check reads only the package, so it can't compare it with the source
    store.
    """
    failures: list[dict[str, Any]] = []

    def fail(
        reason: str,
        session_id: Any = None,
        seq: int | None = None,
        event_id: Any = None,
    ) -> None:
        failures.append({
            "session_id": session_id,
            "seq": seq,
            "event_id": event_id,
            "reason": reason,
        })

    def result(checked_sessions: int = 0, checked_events: int = 0) -> dict:
        return {
            "ok": not failures,
            "checked_sessions": checked_sessions,
            "checked_events": checked_events,
            "failures": failures,
        }

    try:
        data = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError):
        fail("export file can't be read")
        return result()
    except (ValueError, RecursionError):
        fail("export can't be parsed as JSON")
        return result()
    if not isinstance(data, dict):
        fail("export is not a JSON object")
        return result()

    metadata = data.get("compliance_metadata")
    if not isinstance(metadata, dict):
        fail("compliance_metadata is missing or not an object")

    sessions = data.get("sessions")
    if "sessions" not in data:
        fail("sessions is missing")
    elif not isinstance(sessions, list):
        fail("sessions is not a list")
    elif not sessions:
        fail("export has no sessions")
    session_list = sessions if isinstance(sessions, list) else []
    # logging_obligations can be rebuilt only from well-formed sessions.
    comparable = isinstance(sessions, list)

    checked_events = 0
    for position, session in enumerate(session_list, start=1):
        if not isinstance(session, dict):
            fail(f"session {position} is not an object")
            comparable = False
            continue
        session_id = session.get("session_id", "")
        article_12 = session.get("article_12")
        if not isinstance(article_12, dict):
            fail("article_12 is missing or not an object", session_id)
            comparable = False
            continue
        events = article_12.get("events", [])
        if not isinstance(events, list):
            fail("events is not a list", session_id)
            continue
        checked_events += len(events)
        if not events:
            fail("session has no events", session_id)
        _check_export_events(events, session_id, fail)
        trace_integrity = article_12.get("trace_integrity")
        if not isinstance(trace_integrity, dict):
            fail("trace_integrity is missing or not an object", session_id)
            continue
        _check_export_trace_integrity(trace_integrity, events, session_id, fail)

    if isinstance(metadata, dict):
        session_count = metadata.get("session_count")
        if isinstance(sessions, list) and (
            type(session_count) is not int or session_count != len(sessions)
        ):
            fail("session_count does not match the number of sessions")
        if metadata.get("tamper_evident") is not True:
            fail("tamper_evident is not true")

    # export_eu_ai_act writes each session's Article 12 record twice. Both
    # copies must match, or a reader of logging_obligations sees events this
    # check never read. Compare JSON text, because == treats true as 1 and
    # 1 as 1.0.
    package_article_12 = data.get("article_12")
    if not isinstance(package_article_12, dict):
        fail("article_12 is missing or not an object")
    elif comparable:
        expected = [
            {"session_id": session.get("session_id"), **session["article_12"]}
            for session in session_list
        ]
        try:
            same = json.dumps(
                package_article_12.get("logging_obligations"), sort_keys=True,
            ) == json.dumps(expected, sort_keys=True)
        except RecursionError:
            same = False
        if not same:
            fail("article_12.logging_obligations does not match the sessions")

    return result(len(session_list), checked_events)


def _export_failure_location(failure: dict) -> str:
    if failure["seq"] is not None:
        return (
            f"session {inert(failure['session_id'])}, event {failure['seq']} "
            f"(id={inert(failure['event_id'])})"
        )
    if failure["session_id"] is not None:
        return f"session {inert(failure['session_id'])}"
    return "export"


def cmd_verify_export(args: argparse.Namespace) -> int:
    result = verify_eu_ai_act_export(getattr(args, "from_export"))
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    elif result["ok"]:
        sys.stdout.write(
            f"Export hash chain intact - {result['checked_events']} events checked\n"
        )
    else:
        failures = result["failures"]
        noun = "problem" if len(failures) == 1 else "problems"
        sys.stdout.write(f"Export hash chain failed - {len(failures)} {noun}\n")
        for failure in failures:
            sys.stdout.write(
                f"  {_export_failure_location(failure)}: {failure['reason']}\n"
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
            try:
                reports = export_compliance_bulk(store, framework=framework,
                                                 since_days=since_days)
            except ValueError as exc:
                # A store entry's name can hold any character.
                sys.stderr.write(f"Error: {inert(str(exc))}\n")
                return 1
            result = json.dumps(reports, indent=2)
        else:
            full_id = store.find_session(session_id)
            if not full_id:
                sys.stderr.write(f"Session not found: {session_id}\n")
                return 1
            report = export_compliance(store, full_id, framework=framework)
            if "error" in report:
                sys.stderr.write(f"Error: {report['error']}\n")
                return 1
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
