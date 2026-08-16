"""Heuristic, privacy-minimizing compliance evidence crosswalk reports.

This module is deliberately separate from the legacy ``compliance export``
implementation.  It reports observed evidence, risk signals, evidence gaps,
and areas that were not assessed.  It never determines legal applicability,
control satisfaction, audit readiness, or compliance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import tempfile
import textwrap
import unicodedata
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, TextIO

from .approval import ApprovalRequest
from .audit import Policy, _audit_event, _extract_urls, _is_sensitive
from .models import EventType, SessionMeta, TraceEvent
from .redact import contains_redaction_marker, redact_data
from .store import DEFAULT_TRACE_DIR, TraceStore
from .team_report import session_engineer_attribution
from .tenancy import enumerate_trace_stores


REPORT_SCHEMA = "agent-strace-compliance-report/v1"
MANIFEST_SCHEMA = "agent-strace-compliance-framework/v1"
INDEX_SCHEMA = "agent-strace-compliance-manifest-index/v1"
SARIF_SCHEMA = (
    "https://json.schemastore.org/sarif-2.1.0.json"
)
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_META_BYTES = 1024 * 1024
MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
MAX_EVENT_LINE_BYTES = 4 * 1024 * 1024
MAX_JSON_STRING_CHARS = 1024 * 1024
MAX_SESSIONS = 10_000
MAX_EVENTS = 1_000_000
MAX_APPROVAL_RECORDS = 10_000
MAX_APPROVAL_BYTES = 16 * 1024 * 1024
MAX_APPROVAL_RECORD_BYTES = 64 * 1024
MAX_JSON_DEPTH = 40
ASSESSMENT_STATES = (
    "evidence_observed",
    "risk_signal",
    "gap",
    "not_assessed",
)

_DURATION_RE = re.compile(r"^(?P<amount>[1-9][0-9]*)(?P<unit>[dh])$")
_SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_FRAMEWORK_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

_DETECTORS = {
    "access_activity_evidence",
    "authorization_coverage_signal",
    "authorization_evidence",
    "cascading_failure_signal",
    "change_activity_evidence",
    "external_network_signal",
    "goal_hijack_signal",
    "human_trust_signal",
    "identity_evidence",
    "identity_lifecycle_evidence",
    "identity_privilege_signal",
    "integrity_evidence",
    "inter_agent_signal",
    "memory_context_signal",
    "monitoring_evidence",
    "network_activity_evidence",
    "oversight_evidence",
    "recordkeeping_evidence",
    "risk_event_evidence",
    "rogue_agent_signal",
    "sensitive_access_signal",
    "supply_chain_signal",
    "tool_misuse_signal",
    "transparency_metadata_evidence",
    "unexpected_code_execution_signal",
}

_EXPLICIT_SIGNAL_NAMES = {
    "goal_hijack": {"goal_hijack", "agent_goal_hijack", "prompt_injection"},
    "tool_misuse": {"tool_misuse", "tool_exploitation"},
    "identity_privilege": {"identity_abuse", "privilege_abuse", "privilege_escalation"},
    "unexpected_execution": {"unexpected_code_execution", "unexpected_execution", "rce"},
    "memory_context": {"memory_poisoning", "context_poisoning"},
    "inter_agent": {"insecure_inter_agent_communication", "inter_agent_violation"},
    "human_trust": {"human_trust_exploitation", "deceptive_output"},
    "rogue_agent": {"rogue_agent"},
}

_RECORDED_ALLOWED = {"allow", "allowed", "approve", "approved", "authorized"}
_RECORDED_DENIED = {"deny", "denied", "reject", "rejected", "unauthorized"}
_RECORDED_PENDING = {"pending", "requested", "awaiting_approval"}


class ComplianceReportError(ValueError):
    """Raised for safe, user-correctable report input errors."""


def _reject_constant(value: str) -> None:
    raise ComplianceReportError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComplianceReportError("duplicate JSON object key")
        result[key] = value
    return result


def _check_json_shape(value: Any, *, depth_limit: int = MAX_JSON_DEPTH) -> None:
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if depth > depth_limit:
            raise ComplianceReportError("JSON nesting exceeds the supported limit")
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ComplianceReportError("JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str) and len(item) > MAX_JSON_STRING_CHARS:
            raise ComplianceReportError("JSON string exceeds the supported length limit")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ComplianceReportError("JSON numbers must be finite")
        elif not isinstance(item, (str, int, float, bool, type(None))):
            raise ComplianceReportError("unsupported JSON value")


def _strict_json_bytes(
    data: bytes, *, label: str, max_bytes: int = MAX_MANIFEST_BYTES,
) -> Any:
    if len(data) > max_bytes:
        raise ComplianceReportError(f"{label} exceeds the supported size limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ComplianceReportError(f"{label} must be UTF-8 JSON") from exc
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ComplianceReportError(f"{label} is not valid bounded JSON") from exc
    _check_json_shape(parsed)
    return parsed


def _safe_display_text(value: Any, label: str, *, max_chars: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComplianceReportError(f"{label} must be a non-empty string")
    if value != value.strip() or len(value) > max_chars:
        raise ComplianceReportError(f"{label} has unsafe whitespace or length")
    if any(
        unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for char in value
    ):
        raise ComplianceReportError(f"{label} contains unsafe Unicode controls")
    return value


def _required_string(container: dict[str, Any], key: str) -> str:
    value = container.get(key)
    return _safe_display_text(value, f"manifest {key}")


def _validate_manifest(manifest: Any, requested_framework: str) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ComplianceReportError("unsupported compliance manifest schema")
    allowed_root = {"schema", "mapping_version", "framework", "controls"}
    if set(manifest) != allowed_root:
        raise ComplianceReportError("compliance manifest has unknown or missing fields")
    mapping_version = _required_string(manifest, "mapping_version")
    if not _SEMVER_RE.fullmatch(mapping_version):
        raise ComplianceReportError("manifest mapping_version must use semantic versioning")

    framework = manifest.get("framework")
    if not isinstance(framework, dict):
        raise ComplianceReportError("manifest framework must be an object")
    required_framework = {
        "id", "title", "framework_edition", "source_url", "source_as_of",
        "source_checked_at", "applicability_policy",
    }
    optional_framework = {
        "amendment_url", "application_note", "attribution", "changes_notice",
        "content_use_notice", "license", "license_scope", "license_url",
        "licensing_requirements_url", "source_published",
    }
    if not required_framework.issubset(framework) or not set(framework).issubset(
        required_framework | optional_framework
    ):
        raise ComplianceReportError("manifest framework fields do not match the schema")
    framework_id = _required_string(framework, "id")
    if not _SAFE_FRAMEWORK_RE.fullmatch(framework_id):
        raise ComplianceReportError("manifest framework id is invalid")
    if requested_framework != framework_id:
        raise ComplianceReportError("selected manifest does not match --framework")
    for key in required_framework - {"id"}:
        _required_string(framework, key)
    for key in optional_framework:
        if key in framework:
            _required_string(framework, key)
    for key in (
        "source_url", "amendment_url", "license_url",
        "licensing_requirements_url",
    ):
        if key in framework and not str(framework[key]).startswith("https://"):
            raise ComplianceReportError(f"manifest {key} must use HTTPS")
    for key in ("source_as_of", "source_checked_at", "source_published"):
        if key not in framework:
            continue
        try:
            date.fromisoformat(str(framework[key]))
        except ValueError as exc:
            raise ComplianceReportError(f"manifest {key} must use YYYY-MM-DD") from exc
    if framework["applicability_policy"] != "not_assessed":
        raise ComplianceReportError("manifest applicability must remain not_assessed")

    controls = manifest.get("controls")
    if not isinstance(controls, list) or not controls:
        raise ComplianceReportError("manifest controls must be a non-empty array")
    seen: set[str] = set()
    for control in controls:
        if not isinstance(control, dict) or set(control) != {
            "id", "title", "mapping_note", "detectors"
        }:
            raise ComplianceReportError("manifest control fields do not match the schema")
        control_id = _required_string(control, "id")
        _required_string(control, "title")
        _required_string(control, "mapping_note")
        if control_id in seen:
            raise ComplianceReportError("manifest control identifiers must be unique")
        seen.add(control_id)
        detectors = control.get("detectors")
        if not isinstance(detectors, list) or not detectors or not all(
            isinstance(item, str) and item in _DETECTORS for item in detectors
        ):
            raise ComplianceReportError("manifest references an unknown detector")
    return manifest


def _read_regular_file(
    path: Path, *, label: str, max_bytes: int,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ComplianceReportError(f"{label} could not be opened safely") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ComplianceReportError(f"{label} must be a regular file")
        if info.st_size > max_bytes:
            raise ComplianceReportError(f"{label} exceeds the supported size limit")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise ComplianceReportError(f"{label} exceeds the supported size limit")
        return data
    finally:
        os.close(descriptor)


def _package_bytes(name: str) -> bytes:
    try:
        resource = resources.files("agent_trace").joinpath("frameworks", name)
        return _read_regular_file(
            Path(os.fspath(resource)), label="installed compliance manifest",
            max_bytes=MAX_MANIFEST_BYTES,
        )
    except (FileNotFoundError, OSError, TypeError) as exc:
        raise ComplianceReportError("installed compliance manifests are unavailable") from exc


def _safe_local_manifest(path_value: str) -> bytes:
    return _read_regular_file(
        Path(path_value), label="local compliance manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )


def load_framework_manifest(
    framework: str,
    rescan: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load and validate a bundled or explicitly selected local manifest.

    ``rescan=None`` and ``rescan='installed'`` both use the installed,
    digest-verified package manifest. A different value is treated as an
    explicit local manifest path and is never fetched or updated.
    """
    if framework not in {"aicpa-tsc", "owasp-agentic", "eu-ai-act"}:
        raise ComplianceReportError("unsupported compliance framework")
    selection = "bundled"
    filename = f"{framework}.json"
    if rescan not in (None, "installed"):
        raw = _safe_local_manifest(rescan)
        selection = "local"
    else:
        index_raw = _package_bytes("index.json")
        index = _strict_json_bytes(index_raw, label="compliance manifest index")
        if not isinstance(index, dict) or set(index) != {
            "schema", "index_version", "manifests"
        } or index.get("schema") != INDEX_SCHEMA:
            raise ComplianceReportError("unsupported compliance manifest index")
        _required_string(index, "index_version")
        entries = index.get("manifests")
        if not isinstance(entries, list):
            raise ComplianceReportError("manifest index entries must be an array")
        matched = []
        seen_frameworks: set[str] = set()
        seen_files: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "framework", "file", "mapping_version", "sha256"
            }:
                raise ComplianceReportError("manifest index entry is invalid")
            indexed_framework = _required_string(entry, "framework")
            indexed_file = _required_string(entry, "file")
            indexed_version = _required_string(entry, "mapping_version")
            indexed_digest = _required_string(entry, "sha256")
            if not _SAFE_FRAMEWORK_RE.fullmatch(indexed_framework):
                raise ComplianceReportError("manifest index framework id is invalid")
            if (
                Path(indexed_file).name != indexed_file
                or not indexed_file.endswith(".json")
            ):
                raise ComplianceReportError("manifest index file name is unsafe")
            if not _SEMVER_RE.fullmatch(indexed_version):
                raise ComplianceReportError("manifest index mapping version is invalid")
            if not _SHA256_RE.fullmatch(indexed_digest):
                raise ComplianceReportError("manifest index digest is invalid")
            if indexed_framework in seen_frameworks or indexed_file in seen_files:
                raise ComplianceReportError("manifest index entries must be unique")
            seen_frameworks.add(indexed_framework)
            seen_files.add(indexed_file)
            if indexed_framework == framework:
                matched.append(entry)
        if len(matched) != 1:
            raise ComplianceReportError("manifest index selection is ambiguous or missing")
        selected = matched[0]
        filename = _required_string(selected, "file")
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ComplianceReportError("manifest index file name is unsafe")
        expected_digest = _required_string(selected, "sha256")
        if not _SHA256_RE.fullmatch(expected_digest):
            raise ComplianceReportError("manifest index digest is invalid")
        raw = _package_bytes(filename)
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            raise ComplianceReportError("installed compliance manifest digest mismatch")
    manifest = _validate_manifest(
        _strict_json_bytes(raw, label="compliance manifest"), framework
    )
    if rescan in (None, "installed") and manifest["mapping_version"] != selected["mapping_version"]:
        raise ComplianceReportError("manifest mapping version does not match its index")
    digest = hashlib.sha256(raw).hexdigest()
    return manifest, {
        "mapping_version": manifest["mapping_version"],
        "sha256": f"sha256:{digest}",
        "selection": selection,
    }


def _iso_utc(moment: datetime) -> str:
    value = moment.astimezone(timezone.utc)
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_absolute_utc(value: str, label: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ComplianceReportError(f"{label} cannot be empty")
    try:
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
            return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ComplianceReportError(
            f"{label} must be an ISO-8601 UTC date/time"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ComplianceReportError(f"{label} date/time must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def resolve_report_window(
    since: str | None,
    until: str | None,
    *,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Resolve a strict half-open UTC report window."""
    snapshot = now or datetime.now(timezone.utc)
    if snapshot.tzinfo is None or snapshot.utcoffset() is None:
        raise ComplianceReportError("snapshot time must include a UTC offset")
    end = _parse_absolute_utc(until, "--until") if until else snapshot.astimezone(timezone.utc)
    requested = since or "90d"
    duration = _DURATION_RE.fullmatch(requested)
    if duration:
        amount = int(duration.group("amount"))
        try:
            delta = (
                timedelta(days=amount)
                if duration.group("unit") == "d"
                else timedelta(hours=amount)
            )
            start = end - delta
        except (OverflowError, ValueError) as exc:
            raise ComplianceReportError("--since duration is outside the supported range") from exc
    else:
        start = _parse_absolute_utc(requested, "--since")
    if start >= end:
        raise ComplianceReportError("--since must be earlier than --until")
    return start, end


def _finite_timestamp(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComplianceReportError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ComplianceReportError(f"{field} must be finite and non-negative")
    if result > 253402300799:
        raise ComplianceReportError(f"{field} is outside the supported UTC range")
    return result


def _validate_meta(meta: SessionMeta) -> None:
    if not isinstance(meta.session_id, str) or not meta.session_id:
        raise ComplianceReportError("session metadata identifier is invalid")
    _finite_timestamp(meta.started_at, "session started_at")
    if meta.ended_at is not None:
        _finite_timestamp(meta.ended_at, "session ended_at")
    if not isinstance(meta.attribution, dict):
        raise ComplianceReportError("session attribution must be an object")
    _check_json_shape(meta.attribution)


def _validate_events(meta: SessionMeta, events: list[TraceEvent]) -> None:
    for event in events:
        if not isinstance(event.event_type, EventType):
            raise ComplianceReportError("event type is invalid")
        _finite_timestamp(event.timestamp, "event timestamp")
        if not isinstance(event.event_id, str) or not event.event_id:
            raise ComplianceReportError("event identifier is invalid")
        if event.session_id != meta.session_id:
            raise ComplianceReportError("event session identifier mismatch")
        if not isinstance(event.parent_id, str) or not isinstance(event.prev_hash, str):
            raise ComplianceReportError("event link fields must be strings")
        if not isinstance(event.data, dict):
            raise ComplianceReportError("event data must be an object")
        _check_json_shape(event.data)


def _parse_meta_snapshot(raw: bytes, directory_name: str) -> SessionMeta:
    payload = _strict_json_bytes(raw, label="session metadata", max_bytes=MAX_META_BYTES)
    if not isinstance(payload, dict):
        raise ComplianceReportError("session metadata must be an object")
    payload.setdefault("tenant_id", "")
    try:
        meta = SessionMeta(**payload)
    except TypeError as exc:
        raise ComplianceReportError("session metadata fields do not match the schema") from exc
    if meta.session_id != directory_name:
        raise ComplianceReportError("session metadata identifier does not match its directory")
    _validate_meta(meta)
    return meta


def _parse_event_snapshot(
    raw: bytes, meta: SessionMeta,
) -> tuple[list[TraceEvent], str]:
    lines = raw.splitlines()
    if len(lines) > MAX_EVENTS:
        raise ComplianceReportError("event count exceeds the supported limit")
    if any(not line.strip() for line in lines):
        raise ComplianceReportError("event stream contains a blank NDJSON record")
    events: list[TraceEvent] = []
    objects: list[dict[str, Any]] = []
    for line in lines:
        if len(line) > MAX_EVENT_LINE_BYTES:
            raise ComplianceReportError("event record exceeds the supported size limit")
        payload = _strict_json_bytes(
            line, label="trace event", max_bytes=MAX_EVENT_LINE_BYTES
        )
        if not isinstance(payload, dict):
            raise ComplianceReportError("trace event must be an object")
        payload.setdefault("tenant_id", "")
        try:
            payload["event_type"] = EventType(payload["event_type"])
            event = TraceEvent(**payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ComplianceReportError("trace event fields do not match the schema") from exc
        if event.session_id and event.session_id != meta.session_id:
            raise ComplianceReportError("event session identifier mismatch")
        event.session_id = meta.session_id
        if meta.tenant_id and event.tenant_id and event.tenant_id != meta.tenant_id:
            raise ComplianceReportError("event tenant does not match session metadata")
        if meta.tenant_id:
            event.tenant_id = meta.tenant_id
        events.append(event)
        objects.append(payload)
    _validate_events(meta, events)

    if not lines:
        return events, "empty"
    if objects[0].get("prev_hash", ""):
        return events, "broken"
    if len(lines) == 1:
        return events, "unanchored"
    missing = 0
    for index in range(1, len(lines)):
        stored = objects[index].get("prev_hash", "")
        if not isinstance(stored, str):
            raise ComplianceReportError("event prev_hash must be a string")
        if not stored:
            missing += 1
        elif stored != hashlib.sha256(lines[index - 1]).hexdigest():
            return events, "broken"
    if missing == len(lines) - 1:
        return events, "legacy"
    if missing:
        return events, "partial"
    return events, "unanchored"


def _discover_snapshots(
    stores: list[TraceStore],
) -> list[tuple[int, TraceStore, SessionMeta, list[TraceEvent], str, int]]:
    snapshots: list[tuple[int, TraceStore, SessionMeta, list[TraceEvent], str, int]] = []
    total_bytes = 0
    total_events = 0
    auxiliary = {".approvals", ".tenant-journal", "checkpoints", "datasets"}
    for store_index, store in enumerate(stores):
        base = store.base_dir
        if base.is_symlink():
            raise ComplianceReportError("trace store cannot be symlinked")
        if not base.exists():
            continue
        if not base.is_dir():
            raise ComplianceReportError("trace store must be a directory")
        for entry in sorted(base.iterdir(), key=lambda item: item.name):
            if entry.is_symlink():
                raise ComplianceReportError("trace store contains a symlinked entry")
            if not entry.is_dir():
                continue
            if (not store.workspace_id and entry.name == "workspaces") or entry.name in auxiliary:
                continue
            if len(snapshots) >= MAX_SESSIONS:
                raise ComplianceReportError("session count exceeds the supported limit")
            meta_path = entry / "meta.json"
            events_path = entry / "events.ndjson"
            meta_raw = _read_regular_file(
                meta_path, label="session metadata", max_bytes=MAX_META_BYTES
            )
            event_raw = _read_regular_file(
                events_path, label="event stream", max_bytes=MAX_TRACE_BYTES
            )
            total_bytes += len(meta_raw) + len(event_raw)
            if total_bytes > MAX_SNAPSHOT_BYTES:
                raise ComplianceReportError("trace snapshot exceeds the supported size limit")
            meta = _parse_meta_snapshot(meta_raw, entry.name)
            events, integrity = _parse_event_snapshot(event_raw, meta)
            total_events += len(events)
            if total_events > MAX_EVENTS:
                raise ComplianceReportError("snapshot event count exceeds the supported limit")
            snapshots.append((
                store_index, store, meta, events, integrity, len(event_raw)
            ))
    return snapshots


def _load_policy(
    path_value: str | None,
) -> tuple[Policy | None, dict[str, str] | None]:
    if not path_value:
        return None, None
    raw = _read_regular_file(
        Path(path_value), label="policy file", max_bytes=MAX_MANIFEST_BYTES
    )
    parsed = _strict_json_bytes(raw, label="policy file")
    if not isinstance(parsed, dict):
        raise ComplianceReportError("policy file must contain an object")
    files = parsed.get("files", {})
    commands = parsed.get("commands", {})
    network = parsed.get("network", {})
    if not all(isinstance(item, dict) for item in (files, commands, network)):
        raise ComplianceReportError("policy sections must be objects")
    for section in (files.get("read", {}), files.get("write", {})):
        if not isinstance(section, dict):
            raise ComplianceReportError("policy file access rules must be objects")
        for key in ("allow", "deny"):
            values = section.get(key, [])
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ComplianceReportError("policy rule lists must contain strings")
    for key in ("allow", "deny"):
        values = commands.get(key, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ComplianceReportError("policy command rules must contain strings")
    if not isinstance(network.get("deny_all", False), bool):
        raise ComplianceReportError("policy network.deny_all must be a boolean")
    network_allow = network.get("allow", [])
    if not isinstance(network_allow, list) or not all(isinstance(value, str) for value in network_allow):
        raise ComplianceReportError("policy network.allow must contain strings")
    patterns: list[str] = []
    for section in (files.get("read", {}), files.get("write", {}), commands):
        patterns.extend(section.get("allow", []))
        patterns.extend(section.get("deny", []))
    patterns.extend(network_allow)
    if len(patterns) > 1000:
        raise ComplianceReportError("policy rule count exceeds the supported limit")
    for pattern in patterns:
        if len(pattern) > 512 or len(pattern.replace("\\", "/").split("/")) > 64:
            raise ComplianceReportError("policy pattern exceeds the supported complexity limit")
        if pattern.count("**") > 1:
            raise ComplianceReportError("policy patterns may contain at most one recursive wildcard")
    schema = _safe_display_text(
        parsed.get("schema", "unspecified"), "policy schema", max_chars=128
    )
    return Policy.from_dict(parsed), {
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "schema": schema,
        "digest_scope": "exact validated policy file bytes",
    }


def _load_approvals(
    store: TraceStore,
) -> tuple[dict[tuple[str, str], list[ApprovalRequest]], Counter[str], int]:
    directory = store.base_dir / ".approvals"
    if not directory.exists():
        return {}, Counter(), 0
    if directory.is_symlink() or not directory.is_dir():
        raise ComplianceReportError("approval storage is not a safe directory")
    linked: dict[tuple[str, str], list[ApprovalRequest]] = {}
    unlinked: Counter[str] = Counter()
    malformed = 0
    total_bytes = 0
    record_count = 0
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise ComplianceReportError("approval storage contains a symlink")
        if not path.is_file() or path.suffix != ".json":
            continue
        record_count += 1
        if record_count > MAX_APPROVAL_RECORDS:
            raise ComplianceReportError("approval record count exceeds the supported limit")
        raw = _read_regular_file(
            path, label="approval record", max_bytes=MAX_APPROVAL_RECORD_BYTES
        )
        total_bytes += len(raw)
        if total_bytes > MAX_APPROVAL_BYTES:
            raise ComplianceReportError("approval records exceed the total size limit")
        try:
            payload = _strict_json_bytes(
                raw, label="approval record", max_bytes=MAX_APPROVAL_RECORD_BYTES
            )
            if not isinstance(payload, dict):
                raise ComplianceReportError("approval record must be an object")
            allowed = set(ApprovalRequest.__dataclass_fields__)
            if not set(payload).issubset(allowed):
                raise ComplianceReportError("approval record has unknown fields")
            required = {"request_id", "session_id", "event_id", "state", "created_at"}
            if not required.issubset(payload):
                raise ComplianceReportError("approval record is missing required fields")
            state = payload.get("state")
            if state in {"approved", "denied"} and payload.get("decided_at") is None:
                raise ComplianceReportError("decided approval record is missing decided_at")
            if state == "pending" and payload.get("decided_at") is not None:
                raise ComplianceReportError("pending approval record cannot have decided_at")
            request = ApprovalRequest.from_json(json.dumps(payload))
            if request.state not in {"pending", "approved", "denied"}:
                raise ComplianceReportError("approval record state is invalid")
            for value in (
                request.request_id, request.session_id, request.event_id,
                request.rule_name, request.tool_name,
            ):
                if not isinstance(value, str):
                    raise ComplianceReportError("approval identifiers must be strings")
            if not request.request_id:
                raise ComplianceReportError("approval request_id cannot be empty")
            _finite_timestamp(request.created_at, "approval created_at")
            if request.decided_at is not None:
                _finite_timestamp(request.decided_at, "approval decided_at")
            if not request.session_id or not request.event_id:
                unlinked[request.session_id] += 1
                continue
            linked.setdefault((request.session_id, request.event_id), []).append(request)
        except (ComplianceReportError, OSError, TypeError, json.JSONDecodeError):
            malformed += 1
    return linked, unlinked, malformed


def _tool_data(event: TraceEvent) -> tuple[str, dict[str, Any]]:
    data = event.data
    name = str(data.get("tool_name") or data.get("name") or data.get("tool") or "").strip().lower()
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return name, arguments


def _classify_operation(event: TraceEvent) -> tuple[set[str], str, bool, bool]:
    name, arguments = _tool_data(event)
    file_path = str(arguments.get("file_path") or arguments.get("path") or "")
    command = str(arguments.get("command") or "")
    read_names = {"read", "view", "grep", "glob", "file_read"}
    write_names = {"write", "edit", "create", "apply_patch", "file_write"}
    execution_names = {"bash", "shell", "exec", "execute", "terminal", "python"}
    network_names = {"fetch", "http", "request", "web", "web_fetch", "curl"}
    inter_agent_names = {
        "agent", "task", "spawn_agent", "send_message", "followup_task",
        "delegate", "subagent",
    }
    supply_chain = False
    categories: set[str] = set()
    if name in read_names:
        categories.add("file-read")
        resource = "sensitive-file" if file_path and _is_sensitive(file_path) else "file"
    elif name in write_names:
        categories.add("file-write")
        resource = "file"
    else:
        resource = "generic-resource"
    if name in execution_names:
        categories.add("code-execution")
        resource = "command"
        lowered = command.lower()
        supply_chain = bool(re.search(
            r"(?:^|[;&|]\s*)(?:pip|pip3|npm|pnpm|yarn|cargo|gem|go)\s+(?:install|add|get)\b",
            lowered,
        ))
    if name in network_names or (command and _extract_urls(command)):
        categories.add("network")
        resource = "network-endpoint"
    if name in inter_agent_names or any(token in name for token in ("agent", "delegate")):
        categories.add("inter-agent")
        resource = "agent-channel"
    if not categories:
        categories.add("generic")
    sensitive = bool(
        categories & {"file-write", "code-execution", "network", "inter-agent"}
        or resource == "sensitive-file"
    )
    return categories, resource, sensitive, supply_chain


def _explicit_signal(event: TraceEvent, group: str) -> bool:
    accepted = _EXPLICIT_SIGNAL_NAMES[group]
    data = event.data
    values: list[str] = []
    for key in ("security_signal", "risk_signal", "signal_type", "finding_type"):
        value = data.get(key)
        if isinstance(value, str):
            values.append(value.lower().strip())
    signals = data.get("signals", [])
    if isinstance(signals, list):
        values.extend(str(value).lower().strip() for value in signals if isinstance(value, str))
    for name in accepted:
        if data.get(name) is True:
            return True
    return any(value in accepted for value in values)


def _recorded_authorization(event: TraceEvent) -> tuple[str, str, str | None] | None:
    """Validate the documented namespaced pre-action authorization record."""
    record = event.data.get("agent_trace_authorization")
    if record is None:
        return None
    required = {
        "schema", "decision", "mode", "evaluated_at", "policy_sha256", "rule_id"
    }
    if not isinstance(record, dict) or set(record) != required:
        return "ambiguous_record", "invalid", None
    if record.get("schema") != "agent-strace-authorization/v1":
        return "ambiguous_record", "invalid", None
    decision = record.get("decision")
    if decision not in {"allowed", "denied"} or record.get("mode") != "pre_action":
        return "ambiguous_record", "invalid", None
    evaluated_at = record.get("evaluated_at")
    try:
        evaluated = _finite_timestamp(evaluated_at, "authorization evaluated_at")
    except ComplianceReportError:
        return "ambiguous_record", "invalid", None
    policy_digest = record.get("policy_sha256")
    rule_id = record.get("rule_id")
    if (
        not isinstance(policy_digest, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", policy_digest)
        or not isinstance(rule_id, str)
        or not rule_id
        or len(rule_id) > 128
        or any(
            unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
            for char in rule_id
        )
    ):
        return "ambiguous_record", "invalid", None
    if evaluated > float(event.timestamp):
        return "ambiguous_record", "invalid_order", None
    return "recorded_event", decision, policy_digest


def _authorization(
    event: TraceEvent,
    event_index: int,
    approval_records: list[ApprovalRequest],
    policy: Policy | None,
    *,
    call_id_ambiguous: bool = False,
) -> tuple[str, str, bool, str, str | None]:
    if call_id_ambiguous and (
        approval_records or event.data.get("agent_trace_authorization") is not None
    ):
        return "ambiguous_record", "ambiguous", False, "duplicate_call_identifier", None
    recorded = _recorded_authorization(event)
    if recorded:
        return recorded[0], recorded[1], recorded[0] == "recorded_event", (
            "evaluated_at_or_before_call" if recorded[0] == "recorded_event" else "invalid"
        ), recorded[2]
    if approval_records:
        if len(approval_records) != 1:
            return "ambiguous_approval", "ambiguous", False, "duplicate_or_conflicting", None
        approval = approval_records[0]
        created = float(approval.created_at)
        decided = float(approval.decided_at) if approval.decided_at is not None else None
        if created < float(event.timestamp):
            return "ambiguous_approval", "ambiguous", False, "request_before_call", None
        if approval.state == "pending":
            if decided is not None:
                return "ambiguous_approval", "ambiguous", False, "invalid_pending_order", None
            return "recorded_approval", "pending", False, "call_before_request", None
        if decided is None or decided < created:
            return "ambiguous_approval", "ambiguous", False, "invalid_decision_order", None
        return (
            "recorded_approval", approval.state, False,
            "call_before_request_before_decision", None,
        )
    if policy is not None:
        _, arguments = _tool_data(event)
        observed_values = [
            str(arguments.get(key) or "")
            for key in ("file_path", "path", "pattern", "command")
        ]
        if any(
            len(value) > 8192
            or len(value.replace("\\", "/").split("/")) > 128
            for value in observed_values
        ):
            return (
                "retrospective_policy", "not_covered", False,
                "observed_value_complexity_exceeded", None,
            )
        safe_event = copy.copy(event)
        safe_event.data = {"tool_name": _tool_data(event)[0], "arguments": arguments}
        try:
            entries = _audit_event(safe_event, event_index, policy)
        except (AttributeError, RecursionError, TypeError, ValueError):
            return (
                "retrospective_policy", "not_covered", False,
                "bounded_evaluation_failed", None,
            )
        verdicts = {entry.verdict for entry in entries}
        if "denied" in verdicts:
            return "retrospective_policy", "would_deny", False, "evaluated_at_report_time", None
        if verdicts and verdicts <= {"allowed"}:
            return "retrospective_policy", "would_allow", False, "evaluated_at_report_time", None
        return "retrospective_policy", "not_covered", False, "evaluated_at_report_time", None
    return "uncovered", "not_recorded", False, "not_available", None


def _correlate_results(
    events: list[TraceEvent],
    in_scope_indices: set[int] | None = None,
) -> tuple[dict[int, str], Counter[str]]:
    outcomes: dict[int, str] = {}
    quality: Counter[str] = Counter()
    calls_by_id: dict[str, list[int]] = {}
    calls_by_tool_use: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        if event.event_type == EventType.TOOL_CALL:
            calls_by_id.setdefault(event.event_id, []).append(index)
            tool_use = event.data.get("tool_use_id")
            if isinstance(tool_use, str) and tool_use:
                calls_by_tool_use.setdefault(tool_use, []).append(index)
            outcomes[index] = "result_unobserved"
    matched_results: Counter[int] = Counter()
    scope = in_scope_indices if in_scope_indices is not None else set(range(len(events)))

    def relevant(result_index: int, candidates: Iterable[int] = ()) -> bool:
        return result_index in scope or any(item in scope for item in candidates)

    for index, event in enumerate(events):
        if event.event_type not in {EventType.TOOL_RESULT, EventType.ERROR}:
            continue
        matched: int | None = None
        parent_candidates: list[int] = []
        if isinstance(event.parent_id, str) and event.parent_id:
            parent_candidates = calls_by_id.get(event.parent_id, [])
            prior_parent = [item for item in parent_candidates if item < index]
            later_parent = [item for item in parent_candidates if item >= index]
            if len(prior_parent) == 1 and not later_parent:
                matched = prior_parent[0]
            elif len(prior_parent) > 1 or (prior_parent and later_parent):
                if relevant(index, parent_candidates):
                    quality["ambiguous_results"] += 1
                continue
            elif later_parent:
                if relevant(index, later_parent):
                    quality["before_call_results"] += 1
                continue
        tool_matched: int | None = None
        if matched is None:
            tool_use = event.data.get("tool_use_id")
            if isinstance(tool_use, str) and tool_use:
                candidates = calls_by_tool_use.get(tool_use, [])
                prior = [item for item in candidates if item < index]
                later = [item for item in candidates if item >= index]
                if len(prior) == 1 and not later:
                    tool_matched = prior[0]
                elif len(prior) > 1 or (prior and later):
                    if relevant(index, candidates):
                        quality["ambiguous_tool_use_results"] += 1
                    continue
                elif later:
                    if relevant(index, later):
                        quality["before_call_results"] += 1
                    continue
            if tool_matched is not None:
                matched = tool_matched
        else:
            tool_use = event.data.get("tool_use_id")
            if isinstance(tool_use, str) and tool_use:
                candidates = [item for item in calls_by_tool_use.get(tool_use, []) if item < index]
                if len(candidates) != 1 or candidates[0] != matched:
                    if relevant(index, [matched, *candidates]):
                        quality["ambiguous_results"] += 1
                    continue
        if matched is None:
            if relevant(index):
                quality["orphan_results"] += 1
            continue
        if float(event.timestamp) < float(events[matched].timestamp):
            if relevant(index, [matched]):
                quality["before_call_results"] += 1
            continue
        if matched not in scope:
            if index in scope:
                quality["results_for_out_of_window_calls"] += 1
            continue
        if index not in scope:
            quality["boundary_context_results"] += 1
            if outcomes.get(matched) == "result_unobserved":
                outcomes[matched] = (
                    "boundary_context_error"
                    if event.event_type == EventType.ERROR
                    else "boundary_context_result"
                )
            continue
        matched_results[matched] += 1
        if matched_results[matched] > 1:
            outcomes[matched] = "ambiguous_result"
            if matched in scope:
                quality["multiple_results"] += 1
        elif event.event_type == EventType.ERROR:
            outcomes[matched] = "recorded_error"
        else:
            outcomes[matched] = "recorded_result"
    quality["missing_results"] = sum(
        1 for call_index, outcome in outcomes.items()
        if call_index in scope and outcome == "result_unobserved"
    )
    quality["duplicate_call_ids"] = sum(
        len([item for item in indices if item in scope])
        for indices in calls_by_id.values()
        if len(indices) > 1 and any(item in scope for item in indices)
    )
    return outcomes, quality


def _alias_map(prefix: str, values: Iterable[str]) -> dict[str, str]:
    return {
        value: f"{prefix}-{index:03d}"
        for index, value in enumerate(sorted(set(values)), 1)
        if value
    }


def _new_metrics() -> dict[str, Any]:
    return {
        "events": 0,
        "sessions_selected": 0,
        "sessions_readable": 0,
        "sessions_with_identity": 0,
        "sensitive_sessions": 0,
        "sensitive_sessions_with_identity": 0,
        "tool_calls": 0,
        "sensitive_calls": 0,
        "code_execution_calls": 0,
        "network_calls": 0,
        "inter_agent_calls": 0,
        "supply_chain_activity": 0,
        "recorded_results": 0,
        "recorded_errors": 0,
        "ambiguous_results": 0,
        "boundary_context_results": 0,
        "unobserved_results": 0,
        "event_errors": 0,
        "error_event_keys": set(),
        "secret_redaction_signals": 0,
        "authorization": Counter(),
        "authorization_decisions": Counter(),
        "recorded_denials": 0,
        "authorization_by_category": {},
        "category_evidence_refs": {},
        "correlation_quality": Counter(),
        "approval_records": 0,
        "unlinked_approvals": 0,
        "unmatched_approvals": 0,
        "ambiguous_approval_records": 0,
        "out_of_window_approvals": 0,
        "malformed_approvals": 0,
        "integrity": Counter(),
        "explicit_signals": Counter(),
        "explicit_refs": {key: [] for key in _EXPLICIT_SIGNAL_NAMES},
        "operations_after_denial": 0,
        "after_denial_evidence_refs": [],
        "cascading_failures": 0,
        "cascading_evidence_refs": [],
        "session_evidence_refs": [],
        "tool_evidence_refs": [],
        "sensitive_evidence_refs": [],
        "network_evidence_refs": [],
        "change_evidence_refs": [],
        "approval_evidence_refs": [],
        "error_evidence_refs": [],
    }


def _sample(values: Iterable[str], limit: int = 5) -> list[str]:
    return sorted(set(values))[:limit]


def _timestamp_iso(value: float) -> str:
    try:
        return _iso_utc(datetime.fromtimestamp(value, tz=timezone.utc))
    except (OverflowError, OSError, ValueError) as exc:
        raise ComplianceReportError("event timestamp is outside the supported UTC range") from exc


def _scan_snapshot(
    stores: list[TraceStore],
    start: datetime,
    end: datetime,
    policy: Policy | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = _new_metrics()
    inventory: list[
        tuple[int, TraceStore, SessionMeta, list[TraceEvent], str, set[int], str, str]
    ] = []
    start_ts, end_ts = start.timestamp(), end.timestamp()
    for store_index, store, meta, events, integrity, _ in _discover_snapshots(stores):
        in_scope = {
            index for index, event in enumerate(events)
            if start_ts <= float(event.timestamp) < end_ts
        }
        if not in_scope and (events or not start_ts <= float(meta.started_at) < end_ts):
            continue
        scope_key = f"{store_index}\0{meta.session_id}"
        identity, confidence = session_engineer_attribution(meta)
        identity_key = identity if confidence != "unavailable" else ""
        inventory.append((
            store_index, store, meta, events, integrity, in_scope,
            scope_key, identity_key,
        ))
    inventory.sort(key=lambda item: (item[0], float(item[2].started_at), item[2].session_id))
    session_aliases = _alias_map("Session", (item[6] for item in inventory))
    identity_aliases = _alias_map("Identity", (item[7] for item in inventory if item[7]))
    approvals_by_store: dict[
        int, tuple[dict[tuple[str, str], list[ApprovalRequest]], Counter[str], int]
    ] = {}
    selected_ids_by_store: dict[int, set[str]] = {}
    call_counts_by_store: dict[int, Counter[tuple[str, str]]] = {}
    in_scope_calls_by_store: dict[int, set[tuple[str, str]]] = {}
    stores_by_index: dict[int, TraceStore] = {}
    for store_index, store, meta, events, _integrity, in_scope, *_ in inventory:
        selected_ids_by_store.setdefault(store_index, set()).add(meta.session_id)
        stores_by_index[store_index] = store
        counter = call_counts_by_store.setdefault(store_index, Counter())
        scoped = in_scope_calls_by_store.setdefault(store_index, set())
        for event_index, event in enumerate(events):
            if event.event_type != EventType.TOOL_CALL:
                continue
            key = (meta.session_id, event.event_id)
            counter[key] += 1
            if event_index in in_scope:
                scoped.add(key)
    for store_index, store in stores_by_index.items():
        approvals_by_store[store_index] = _load_approvals(store)
        linked, unlinked, malformed = approvals_by_store[store_index]
        selected_ids = selected_ids_by_store[store_index]
        metrics["unlinked_approvals"] += sum(
            count for session_id, count in unlinked.items()
            if session_id in selected_ids
        )
        metrics["malformed_approvals"] += malformed
        for key, records in linked.items():
            if key[0] not in selected_ids:
                continue
            call_count = call_counts_by_store[store_index][key]
            if call_count == 0:
                metrics["unmatched_approvals"] += len(records)
            elif call_count > 1 or len(records) > 1:
                metrics["ambiguous_approval_records"] += len(records)
            elif key not in in_scope_calls_by_store[store_index]:
                metrics["out_of_window_approvals"] += len(records)

    timeline: list[dict[str, Any]] = []
    evidence_counter = 0
    metrics["sessions_selected"] = len(inventory)
    for (
        store_index, _store, meta, events, integrity, in_scope,
        scope_key, identity_key,
    ) in inventory:
        approvals = approvals_by_store[store_index][0]
        session_ref = session_aliases[scope_key]
        metrics["session_evidence_refs"].append(session_ref)
        metrics["integrity"][integrity] += 1
        metrics["sessions_readable"] += 1
        metrics["events"] += len(in_scope)
        if identity_key:
            metrics["sessions_with_identity"] += 1
        identity_ref = identity_aliases.get(identity_key)
        outcomes, correlation_quality = _correlate_results(events, in_scope)
        metrics["correlation_quality"].update(correlation_quality)
        temporal_operations: list[tuple[float, int, str, str]] = []
        denial_evidence: list[tuple[float, int]] = []
        session_sensitive = False
        call_id_counts = Counter(
            event.event_id for event in events
            if event.event_type == EventType.TOOL_CALL
        )
        for event_index, event in enumerate(events):
            if event_index not in in_scope:
                continue
            if event.event_type == EventType.ERROR:
                metrics["event_errors"] += 1
                metrics["error_event_keys"].add((scope_key, event_index))
                metrics["error_evidence_refs"].append(session_ref)
            if redact_data(event.data) != event.data or contains_redaction_marker(event.data):
                metrics["secret_redaction_signals"] += 1
            for group in _EXPLICIT_SIGNAL_NAMES:
                if _explicit_signal(event, group):
                    metrics["explicit_signals"][group] += 1
            if event.event_type != EventType.TOOL_CALL:
                continue
            evidence_counter += 1
            evidence_ref = f"E-{evidence_counter:06d}"
            metrics["tool_calls"] += 1
            metrics["tool_evidence_refs"].append(evidence_ref)
            categories, resource_class, sensitive, supply_chain = _classify_operation(event)
            if supply_chain:
                metrics["supply_chain_activity"] += 1
            if "code-execution" in categories:
                metrics["code_execution_calls"] += 1
                metrics["change_evidence_refs"].append(evidence_ref)
            if "network" in categories:
                metrics["network_calls"] += 1
                metrics["network_evidence_refs"].append(evidence_ref)
            if "inter-agent" in categories:
                metrics["inter_agent_calls"] += 1
            if "file-write" in categories:
                metrics["change_evidence_refs"].append(evidence_ref)
            outcome = outcomes.get(event_index, "result_unobserved")
            temporal_operations.append((
                float(event.timestamp), event_index, outcome, evidence_ref,
            ))
            if outcome == "recorded_error":
                metrics["recorded_errors"] += 1
                metrics["error_evidence_refs"].append(evidence_ref)
            elif outcome == "recorded_result":
                metrics["recorded_results"] += 1
            elif outcome == "ambiguous_result":
                metrics["ambiguous_results"] += 1
            elif outcome in {"boundary_context_result", "boundary_context_error"}:
                metrics["boundary_context_results"] += 1
            else:
                metrics["unobserved_results"] += 1

            approval_records = approvals.get((meta.session_id, event.event_id), [])
            basis, decision, historical, ordering, authorization_policy_digest = _authorization(
                event, event_index + 1, approval_records, policy,
                call_id_ambiguous=call_id_counts[event.event_id] > 1,
            )
            metrics["authorization"][basis] += 1
            metrics["authorization_decisions"][decision] += 1
            category_keys = set(categories)
            if categories & {"file-write", "code-execution"}:
                category_keys.add("change")
            if sensitive:
                category_keys.add("sensitive")
            for category in category_keys:
                counter = metrics["authorization_by_category"].setdefault(
                    category, Counter()
                )
                counter[basis] += 1
                metrics["category_evidence_refs"].setdefault(category, []).append(
                    evidence_ref
                )
            if basis == "recorded_event" and decision == "denied":
                metrics["recorded_denials"] += 1
                denial_evidence.append((
                    float(event.data["agent_trace_authorization"]["evaluated_at"]),
                    event_index,
                ))
            elif basis == "recorded_approval" and decision == "denied":
                metrics["recorded_denials"] += 1
                denial_evidence.append((
                    float(approval_records[0].decided_at), event_index,
                ))
            for group in _EXPLICIT_SIGNAL_NAMES:
                if _explicit_signal(event, group):
                    metrics["explicit_refs"][group].append(evidence_ref)
            if sensitive:
                session_sensitive = True
                metrics["sensitive_calls"] += 1
                metrics["sensitive_evidence_refs"].append(evidence_ref)
                if basis == "recorded_approval" and len(approval_records) == 1:
                    metrics["approval_records"] += 1
                    metrics["approval_evidence_refs"].append(evidence_ref)
                authorization_summary = {
                    "basis": basis,
                    "decision": decision,
                    "pre_action_authorization_evidence": historical,
                    "ordering": ordering,
                }
                if authorization_policy_digest is not None:
                    authorization_summary["policy_sha256"] = authorization_policy_digest
                timeline.append({
                    "evidence_ref": evidence_ref,
                    "session_ref": session_ref,
                    "timestamp_utc": _timestamp_iso(float(event.timestamp)),
                    "operation_categories": sorted(categories),
                    "resource_class": resource_class,
                    "result_status": outcome,
                    "authorization": authorization_summary,
                    "identity_ref": identity_ref,
                })
        ordered_operations = sorted(temporal_operations, key=lambda item: (item[0], item[1]))
        consecutive_recorded_errors = 0
        for _timestamp, _index, outcome, evidence_ref in ordered_operations:
            if consecutive_recorded_errors >= 2:
                metrics["cascading_failures"] += 1
                metrics["cascading_evidence_refs"].append(evidence_ref)
                consecutive_recorded_errors = 0
            if outcome == "recorded_error":
                consecutive_recorded_errors += 1
            else:
                consecutive_recorded_errors = 0
        for operation_time, operation_index, _outcome, evidence_ref in ordered_operations:
            if any(
                operation_index != denial_index
                and (
                    operation_time > effective_time
                    or (
                        operation_time == effective_time
                        and operation_index > denial_index
                    )
                )
                for effective_time, denial_index in denial_evidence
            ):
                metrics["operations_after_denial"] += 1
                metrics["after_denial_evidence_refs"].append(evidence_ref)
        if session_sensitive:
            metrics["sensitive_sessions"] += 1
            if identity_key:
                metrics["sensitive_sessions_with_identity"] += 1
    timeline.sort(key=lambda item: (
        item["timestamp_utc"], item["session_ref"], item["evidence_ref"]
    ))
    return metrics, timeline


def _detector_result(
    status: str,
    *,
    evidence_count: int = 0,
    signal_count: int = 0,
    gap_count: int = 0,
    evidence_refs: Iterable[str] = (),
    limitation: str = "",
) -> dict[str, Any]:
    if status not in ASSESSMENT_STATES:
        raise AssertionError("invalid assessment state")
    result = {
        "status": status,
        "evidence_count": int(evidence_count),
        "signal_count": int(signal_count),
        "gap_count": int(gap_count),
        "evidence_sample": _sample(evidence_refs),
        "limitations": [limitation] if limitation else [],
    }
    return result


def _explicit_risk(metrics: dict[str, Any], group: str) -> dict[str, Any]:
    count = int(metrics["explicit_signals"][group])
    if count:
        return _detector_result(
            "risk_signal", signal_count=count,
            evidence_refs=metrics["explicit_refs"][group],
            limitation="An explicit recorded signal is evidence, not proof of a framework violation.",
        )
    return _detector_result(
        "not_assessed",
        limitation="No explicit signal was recorded; absence is not evidence that the risk was assessed.",
    )


def _authorization_gap(
    metrics: dict[str, Any], *, category: str = "sensitive",
) -> dict[str, Any]:
    counter = metrics["authorization_by_category"].get(category, Counter())
    total = int(sum(counter.values()))
    if not total:
        return _detector_result("not_assessed", limitation="No in-scope sensitive operation was observed.")
    recorded = int(counter["recorded_event"])
    gaps = max(0, total - recorded)
    if gaps:
        return _detector_result(
            "gap", evidence_count=recorded, gap_count=gaps,
            evidence_refs=metrics["category_evidence_refs"].get(category, []),
            limitation="Only a validated namespaced pre-action record is counted as authorization evidence; approval sidecars and retrospective policy evaluation remain contextual evidence.",
        )
    return _detector_result(
            "evidence_observed", evidence_count=recorded,
            evidence_refs=metrics["category_evidence_refs"].get(category, []),
    )


def _run_detector(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    if name == "goal_hijack_signal":
        return _explicit_risk(metrics, "goal_hijack")
    if name == "tool_misuse_signal":
        return _explicit_risk(metrics, "tool_misuse")
    if name == "authorization_coverage_signal":
        return _authorization_gap(metrics)
    if name == "identity_privilege_signal":
        explicit = _explicit_risk(metrics, "identity_privilege")
        if explicit["status"] == "risk_signal":
            return explicit
        missing = metrics["sensitive_sessions"] - metrics["sensitive_sessions_with_identity"]
        if metrics["sensitive_calls"] and missing:
            return _detector_result(
                "gap", gap_count=missing,
                limitation="Persisted identity was unavailable for one or more sessions with sensitive activity.",
            )
        return explicit
    if name == "identity_evidence":
        readable = int(metrics["sensitive_sessions"])
        observed = int(metrics["sensitive_sessions_with_identity"])
        if not readable:
            return _detector_result("not_assessed", limitation="No readable session telemetry was in scope.")
        if observed < readable:
            return _detector_result(
                "gap", evidence_count=observed, gap_count=readable - observed,
                evidence_refs=metrics["session_evidence_refs"],
                limitation="Identity coverage uses persisted attribution only.",
            )
        return _detector_result(
            "evidence_observed", evidence_count=observed,
            evidence_refs=metrics["session_evidence_refs"],
            limitation="Persisted attribution is evidence context, not identity assurance.",
        )
    if name == "identity_lifecycle_evidence":
        return _detector_result(
            "not_assessed",
            limitation="Agent-strace does not capture identity or access removal and modification lifecycle telemetry.",
        )
    if name == "supply_chain_signal":
        count = int(metrics["supply_chain_activity"])
        if count:
            return _detector_result(
                "evidence_observed", evidence_count=count,
                evidence_refs=metrics["tool_evidence_refs"],
                limitation="Package activity is heuristic only; no CVE, dependency, provenance, or SBOM analysis was performed.",
            )
        return _detector_result(
            "not_assessed",
            limitation="No package activity was recognized; supply-chain risk was not scanned.",
        )
    if name == "unexpected_code_execution_signal":
        explicit = _explicit_risk(metrics, "unexpected_execution")
        if explicit["status"] == "risk_signal":
            return explicit
        if metrics["code_execution_calls"]:
            coverage = _authorization_gap(metrics, category="code-execution")
            coverage["limitations"].append(
                "Authorization gaps do not establish unexpected execution."
            )
            return coverage
        return explicit
    if name == "memory_context_signal":
        return _explicit_risk(metrics, "memory_context")
    if name == "inter_agent_signal":
        explicit = _explicit_risk(metrics, "inter_agent")
        if explicit["status"] == "risk_signal":
            return explicit
        count = int(metrics["inter_agent_calls"])
        if count:
            return _detector_result(
                "evidence_observed", evidence_count=count,
                evidence_refs=metrics["tool_evidence_refs"],
                limitation="Observed inter-agent activity does not assess communication security.",
            )
        return explicit
    if name == "cascading_failure_signal":
        count = int(metrics["cascading_failures"])
        if count:
            return _detector_result(
                "risk_signal", signal_count=count,
                evidence_refs=metrics["cascading_evidence_refs"],
                limitation="Repeated correlated errors followed by continued operation are a structural heuristic, not a causal finding.",
            )
        return _detector_result(
            "not_assessed", limitation="No qualifying repeated-error sequence was observed."
        )
    if name == "oversight_evidence":
        observed = int(metrics["approval_records"])
        sensitive = int(metrics["sensitive_calls"])
        gaps = max(0, sensitive - observed)
        if observed and not gaps:
            return _detector_result(
                "evidence_observed", evidence_count=observed,
                evidence_refs=metrics["approval_evidence_refs"],
                limitation="Approval records do not establish oversight design or effectiveness.",
            )
        if sensitive:
            return _detector_result(
                "gap", evidence_count=observed, gap_count=gaps,
                evidence_refs=metrics["sensitive_evidence_refs"],
                limitation="Missing recorded approval is an evidence gap, not proof that human oversight was required or absent.",
            )
        return _detector_result("not_assessed", limitation="No in-scope sensitive operation was observed.")
    if name == "human_trust_signal":
        explicit = _explicit_risk(metrics, "human_trust")
        if explicit["status"] == "risk_signal":
            return explicit
        if metrics["sensitive_calls"] and not metrics["approval_records"]:
            return _detector_result(
                "gap", gap_count=int(metrics["sensitive_calls"]),
                evidence_refs=metrics["sensitive_evidence_refs"],
                limitation="Missing recorded approval does not establish human-agent trust exploitation.",
            )
        return explicit
    if name == "rogue_agent_signal":
        explicit = _explicit_risk(metrics, "rogue_agent")
        if explicit["status"] == "risk_signal":
            return explicit
        if metrics["operations_after_denial"]:
            return _detector_result(
                "gap", gap_count=int(metrics["operations_after_denial"]),
                evidence_refs=metrics["after_denial_evidence_refs"],
                limitation="A later operation after a recorded denial is a heuristic sequence, not proof of a retry or rogue behavior.",
            )
        return explicit
    if name == "access_activity_evidence":
        count = int(metrics["tool_calls"])
        return _detector_result(
            "evidence_observed" if count else "not_assessed",
            evidence_count=count, evidence_refs=metrics["tool_evidence_refs"],
            limitation="Recorded activity does not establish access-control design or effectiveness.",
        )
    if name == "sensitive_access_signal":
        count = int(metrics["sensitive_calls"])
        return _detector_result(
            "evidence_observed" if count else "not_assessed",
            evidence_count=count, evidence_refs=metrics["sensitive_evidence_refs"],
            limitation="Sensitive classifications are heuristic and omit resource values.",
        )
    if name == "authorization_evidence":
        return _authorization_gap(metrics)
    if name in {"network_activity_evidence", "external_network_signal"}:
        count = int(metrics["network_calls"])
        return _detector_result(
            "evidence_observed" if count else "not_assessed",
            evidence_count=count, evidence_refs=metrics["network_evidence_refs"],
            limitation="Network activity is inferred from tool metadata; destinations are intentionally omitted.",
        )
    if name == "monitoring_evidence":
        total = int(metrics["tool_calls"])
        observed = int(metrics["recorded_results"] + metrics["recorded_errors"])
        gaps = int(
            metrics["unobserved_results"]
            + metrics["ambiguous_results"]
            + metrics["boundary_context_results"]
        )
        if not total:
            return _detector_result("not_assessed", limitation="No tool-call telemetry was in scope.")
        return _detector_result(
            "gap" if gaps else "evidence_observed",
            evidence_count=observed, gap_count=gaps,
            evidence_refs=metrics["tool_evidence_refs"],
            limitation="Only in-window, unambiguous outcomes count toward coverage; missing, ambiguous, and boundary-context outcomes are telemetry gaps, not proof of failure.",
        )
    if name == "integrity_evidence":
        states = metrics["integrity"]
        if states["broken"]:
            return _detector_result(
                "risk_signal", signal_count=int(states["broken"]),
                evidence_refs=metrics["session_evidence_refs"],
                limitation="A local hash-link or parse failure was observed; no independent anchor is available.",
            )
        gaps = int(
            states["empty"] + states["legacy"]
            + states["partial"] + states["unanchored"]
        )
        if gaps:
            return _detector_result(
                "gap", gap_count=gaps,
                evidence_refs=metrics["session_evidence_refs"],
                limitation="Empty, legacy, partial, and valid local chains provide incomplete or unanchored integrity evidence.",
            )
        return _detector_result("not_assessed", limitation="No session integrity evidence was in scope.")
    if name == "risk_event_evidence":
        count = int(len(metrics["error_event_keys"]) + metrics["recorded_denials"])
        return _detector_result(
            "risk_signal" if count else "not_assessed",
            signal_count=count, evidence_refs=metrics["error_evidence_refs"],
            limitation="Errors and recorded denials are operational risk signals, not framework violations.",
        )
    if name == "change_activity_evidence":
        count = len(set(metrics["change_evidence_refs"]))
        if not count:
            return _detector_result("not_assessed", limitation="No recognized change activity was observed.")
        auth = _authorization_gap(metrics, category="change")
        auth["evidence_count"] = count
        auth["evidence_sample"] = _sample(metrics["change_evidence_refs"])
        auth["limitations"].append("Recorded change activity does not establish change-management control effectiveness.")
        return auth
    if name == "recordkeeping_evidence":
        selected = int(metrics["sessions_selected"])
        readable = int(metrics["sessions_readable"])
        if not selected:
            return _detector_result("not_assessed", limitation="No sessions were selected in the report window.")
        return _detector_result(
            "gap" if readable < selected else "evidence_observed",
            evidence_count=readable, gap_count=selected - readable,
            evidence_refs=metrics["session_evidence_refs"],
            limitation="Record presence does not assess required retention periods or legal sufficiency.",
        )
    if name == "transparency_metadata_evidence":
        return _detector_result(
            "not_assessed",
            limitation="Runtime metadata is not treated as evidence that required instructions or transparency information were provided.",
        )
    raise AssertionError(f"unimplemented detector: {name}")


def _control_assessment(control: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    detector_results = [_run_detector(name, metrics) for name in control["detectors"]]
    precedence = {"not_assessed": 0, "evidence_observed": 1, "gap": 2, "risk_signal": 3}
    status = max((item["status"] for item in detector_results), key=precedence.get)
    return {
        "control_id": control["id"],
        "title": control["title"],
        "mapping_note": control["mapping_note"],
        "status": status,
        "signal_count": sum(item["signal_count"] for item in detector_results),
        "evidence_count": sum(item["evidence_count"] for item in detector_results),
        "gap_count": sum(item["gap_count"] for item in detector_results),
        "evidence_sample": _sample(
            ref for item in detector_results for ref in item["evidence_sample"]
        ),
        "limitations": sorted({
            limitation
            for item in detector_results
            for limitation in item["limitations"]
        }),
    }


def _coverage(metrics: dict[str, Any]) -> dict[str, Any]:
    tool_calls = int(metrics["tool_calls"])
    recorded_outcomes = int(metrics["recorded_results"] + metrics["recorded_errors"])
    readable = int(metrics["sessions_readable"])
    return {
        "sessions": {
            "selected": int(metrics["sessions_selected"]),
            "readable": readable,
            "event_stream_gap": int(metrics["sessions_selected"] - readable),
        },
        "events_read": int(metrics["events"]),
        "tool_results": {
            "tool_calls": tool_calls,
            "correlated_outcomes": recorded_outcomes,
            "ambiguous_outcomes": int(metrics["ambiguous_results"]),
            "boundary_context_outcomes": int(metrics["boundary_context_results"]),
            "unobserved_outcomes": int(metrics["unobserved_results"]),
            "coverage_percent": round(recorded_outcomes * 100 / tool_calls, 2) if tool_calls else None,
            "correlation_quality": {
                key: int(metrics["correlation_quality"][key])
                for key in (
                    "orphan_results", "multiple_results", "before_call_results",
                    "ambiguous_results", "ambiguous_tool_use_results",
                    "duplicate_call_ids", "missing_results",
                    "boundary_context_results",
                    "results_for_out_of_window_calls",
                )
            },
        },
        "identity": {
            "sessions_with_persisted_attribution": int(metrics["sessions_with_identity"]),
            "sensitive_sessions": int(metrics["sensitive_sessions"]),
            "sensitive_sessions_with_persisted_attribution": int(metrics["sensitive_sessions_with_identity"]),
            "sensitive_session_coverage_percent": (
                round(metrics["sensitive_sessions_with_identity"] * 100 / metrics["sensitive_sessions"], 2)
                if metrics["sensitive_sessions"] else None
            ),
        },
        "authorization": {
            "recorded_event": int(metrics["authorization"]["recorded_event"]),
            "recorded_approval": int(metrics["authorization"]["recorded_approval"]),
            "retrospective_policy": int(metrics["authorization"]["retrospective_policy"]),
            "uncovered": int(metrics["authorization"]["uncovered"]),
            "ambiguous": int(
                metrics["authorization"]["ambiguous_record"]
                + metrics["authorization"]["ambiguous_approval"]
            ),
            "linked_sensitive_approval_calls": int(metrics["approval_records"]),
            "unlinked_approval_records": int(metrics["unlinked_approvals"]),
            "unmatched_approval_records": int(metrics["unmatched_approvals"]),
            "ambiguous_approval_records": int(metrics["ambiguous_approval_records"]),
            "out_of_window_approval_records": int(metrics["out_of_window_approvals"]),
            "malformed_approval_records": int(metrics["malformed_approvals"]),
            "retrospective_outcomes": {
                "would_allow": int(metrics["authorization_decisions"]["would_allow"]),
                "would_deny": int(metrics["authorization_decisions"]["would_deny"]),
            },
            "by_category": {
                category: {
                    "pre_action_record": int(counter["recorded_event"]),
                    "approval_context": int(counter["recorded_approval"]),
                    "retrospective_policy": int(counter["retrospective_policy"]),
                    "uncovered": int(counter["uncovered"]),
                    "ambiguous": int(counter["ambiguous_record"] + counter["ambiguous_approval"]),
                }
                for category, counter in sorted(metrics["authorization_by_category"].items())
            },
        },
        "integrity": {
            state: int(metrics["integrity"][state])
            for state in ("empty", "legacy", "partial", "broken", "unanchored")
        },
        "privacy": {
            "report_local_session_aliases": True,
            "report_local_identity_aliases": True,
            "secret_or_redaction_signals": int(metrics["secret_redaction_signals"]),
            "trace_paths_urls_commands_emitted": False,
        },
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def report_digest(report: dict[str, Any]) -> str:
    payload = copy.deepcopy(report)
    payload.pop("report_digest", None)
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def evidence_digest(report: dict[str, Any]) -> str:
    """Return the generation-time-independent digest of scan evidence."""
    payload = {
        "schema": report["schema"],
        "framework_id": report["framework"]["id"],
        "manifest_sha256": report["manifest"]["sha256"],
        "retrospective_policy": report["retrospective_policy"],
        "window": report["window"],
        "coverage": report["coverage"],
        "controls": report["controls"],
        "authorization_timeline": report["authorization_timeline"],
    }
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def build_compliance_report(
    stores: TraceStore | list[TraceStore],
    framework: str,
    *,
    since: str | None = None,
    until: str | None = None,
    policy_path: str | None = None,
    rescan: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the authoritative deterministic JSON evidence report."""
    manifest, manifest_meta = load_framework_manifest(framework, rescan=rescan)
    snapshot_time = generated_at or datetime.now(timezone.utc)
    start, end = resolve_report_window(since, until, now=snapshot_time)
    policy, policy_provenance = _load_policy(policy_path)
    selected_stores = [stores] if isinstance(stores, TraceStore) else list(stores)
    metrics, timeline = _scan_snapshot(selected_stores, start, end, policy)
    framework_meta = copy.deepcopy(manifest["framework"])
    framework_meta["cli_id"] = framework
    framework_meta["applicability"] = "not_assessed"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at": _iso_utc(snapshot_time),
        "digest_scope": {
            "report_digest": "canonical JSON with report_digest omitted",
            "evidence_digest": "manifest digest, resolved window, coverage, controls, and authorization timeline; generated_at excluded",
        },
        "framework": framework_meta,
        "manifest": manifest_meta,
        "window": {
            "since_utc": _iso_utc(start),
            "until_utc": _iso_utc(end),
            "semantics": "half-open [since_utc, until_utc)",
        },
        "snapshot": {
            "source": "local flat and workspace trace stores",
            "store_count": len(selected_stores),
            "consistency": "best-effort; active stores are not transactionally frozen",
        },
        "retrospective_policy": {
            "used": policy is not None,
            "provenance": policy_provenance,
            "semantics": "would_allow/would_deny are report-time evaluations, not historical authorization",
        },
        "assessment_vocabulary": {
            "evidence_observed": "Relevant telemetry was observed; this does not establish control satisfaction.",
            "risk_signal": "A recorded or explicitly labeled heuristic risk signal warrants review; it is not proof of a violation.",
            "gap": "Expected telemetry or evidence was missing or retrospective; this is not a compliance failure finding.",
            "not_assessed": "The trace cannot support this assessment, or no explicit signal was recorded.",
        },
        "coverage": _coverage(metrics),
        "controls": [
            _control_assessment(control, metrics) for control in manifest["controls"]
        ],
        "authorization_timeline": timeline,
        "rescan": {
            "enabled": rescan is not None,
            "manifest_selection": manifest_meta["selection"],
            "mapping_version": manifest_meta["mapping_version"],
            "manifest_sha256": manifest_meta["sha256"],
            "resolved_window": {
                "since_utc": _iso_utc(start),
                "until_utc": _iso_utc(end),
            },
            "network_updates_performed": False,
            "cve_or_sbom_scan_performed": False,
        },
        "limitations": [
            "This is a heuristic evidence crosswalk, not legal advice, a compliance determination, an audit opinion, or an audit-readiness assessment.",
            "Control satisfaction, implementation effectiveness, legal role, system classification, and framework applicability are not assessed.",
            "Missing telemetry does not establish absence, and observed telemetry does not establish control satisfaction.",
            "Recorded tool-result correlation observes an occurrence; it does not establish historical authorization or control effectiveness.",
            "Retrospective policy evaluation is clearly labeled and is never treated as historical authorization evidence.",
            "Local hash links have no independent external anchor; legacy and partial streams can provide only limited integrity evidence.",
            "The local snapshot can change during collection because active flat and workspace stores are not transactionally frozen.",
            "Trace resource paths, destinations, URLs, commands, raw identifiers, prompts, and result contents are intentionally omitted.",
            "No dependency vulnerability, CVE, provenance, or SBOM scan is performed by compliance-report or --rescan.",
            "JSON is the authoritative representation; SARIF and PDF are limited projections identified by its digest.",
        ],
    }
    report["evidence_digest"] = evidence_digest(report)
    report["report_digest"] = report_digest(report)
    return report


def format_compliance_json(report: dict[str, Any]) -> bytes:
    """Render the authoritative JSON representation."""
    return (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def compliance_report_to_sarif(report: dict[str, Any]) -> dict[str, Any]:
    """Project risk signals and evidence gaps into deterministic SARIF 2.1.0."""
    framework_id = report["framework"]["id"]
    rules = []
    results = []
    for control in sorted(report["controls"], key=lambda item: item["control_id"]):
        rule_id = f"{framework_id}/{control['control_id']}"
        rules.append({
            "id": rule_id,
            "name": control["title"],
            "shortDescription": {"text": control["title"]},
            "fullDescription": {"text": control["mapping_note"]},
            "properties": {
                "controlId": control["control_id"],
                "assessmentVocabulary": list(ASSESSMENT_STATES),
            },
        })
        if control["status"] not in {"risk_signal", "gap"}:
            continue
        is_risk = control["status"] == "risk_signal"
        fingerprint_payload = {
            "framework": framework_id,
            "manifest": report["manifest"]["sha256"],
            "window": report["window"],
            "control_id": control["control_id"],
            "status": control["status"],
            "signal_count": control["signal_count"],
            "gap_count": control["gap_count"],
            "evidence_count": control["evidence_count"],
        }
        fingerprint = hashlib.sha256(
            _canonical_json_bytes(fingerprint_payload)
        ).hexdigest()
        results.append({
            "ruleId": rule_id,
            "level": "warning" if is_risk else "note",
            "message": {
                "text": (
                    f"{control['status']}: {control['signal_count']} risk signal(s), "
                    f"{control['gap_count']} evidence gap(s). Review the authoritative "
                    "JSON report; this is not a compliance finding."
                )
            },
            "properties": {
                "status": control["status"],
                "evidenceSample": control["evidence_sample"],
                "reportDigest": report["report_digest"],
                "evidenceDigest": report["evidence_digest"],
                "noLocationsEmitted": True,
            },
            "partialFingerprints": {
                "agentStraceComplianceEvidence/v1": fingerprint,
            },
        })
    return {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "agent-strace compliance-report",
                    "informationUri": "https://github.com/Siddhant-K-code/agent-trace",
                    "rules": rules,
                }
            },
            "results": results,
            "properties": {
                "authoritativeFormat": "agent-strace compliance-report JSON",
                "reportDigest": report["report_digest"],
                "evidenceDigest": report["evidence_digest"],
                "manifestSha256": report["manifest"]["sha256"],
                "frameworkSourceAndNotices": {
                    key: report["framework"][key]
                    for key in (
                        "source_url", "source_as_of", "amendment_url",
                        "attribution", "license", "license_url", "license_scope",
                        "changes_notice", "content_use_notice",
                        "licensing_requirements_url",
                    )
                    if key in report["framework"]
                },
                "limitations": report["limitations"],
            },
        }],
    }


def format_compliance_sarif(report: dict[str, Any]) -> bytes:
    payload = compliance_report_to_sarif(report)
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _pdf_safe_text(value: Any) -> str:
    """Keep WinAnsi text and visibly escape characters unsupported by Base-14."""
    output: list[str] = []
    for char in str(value):
        try:
            char.encode("cp1252")
            output.append(char)
        except UnicodeEncodeError:
            codepoint = ord(char)
            output.append(
                f"\\u{codepoint:04X}"
                if codepoint <= 0xFFFF
                else f"\\U{codepoint:08X}"
            )
    return "".join(output)


def format_compliance_pdf(report: dict[str, Any]) -> bytes:
    """Render a limited PDF projection using the optional ReportLab extra."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise ComplianceReportError(
            "PDF output requires the optional dependency; install with "
            "pip install 'agent-strace[pdf]'"
        ) from exc

    buffer = io.BytesIO()
    document = canvas.Canvas(
        buffer, pagesize=letter, invariant=1, pageCompression=1,
    )
    document.setTitle("agent-strace compliance evidence crosswalk")
    document.setAuthor("agent-strace")
    width, height = letter
    margin = 54
    y = height - margin

    def line(value: str, *, font: str = "Helvetica", size: int = 9) -> None:
        nonlocal y
        wrapped = textwrap.wrap(_pdf_safe_text(value), width=92) or [""]
        document.setFont(font, size)
        for item in wrapped:
            if y < margin:
                document.showPage()
                y = height - margin
                document.setFont(font, size)
            document.drawString(margin, y, item)
            y -= size + 4

    line("agent-strace compliance evidence crosswalk", font="Helvetica-Bold", size=15)
    line(report["framework"]["title"], font="Helvetica-Bold", size=11)
    line(f"Window: {report['window']['since_utc']} to {report['window']['until_utc']} (end exclusive)")
    line(f"Evidence digest: {report['evidence_digest']}")
    line(f"Authoritative JSON report digest: {report['report_digest']}")
    line(f"Manifest: {report['manifest']['mapping_version']} / {report['manifest']['sha256']}")
    line("Framework source and notices", font="Helvetica-Bold", size=11)
    notice_labels = {
        "source_url": "Source",
        "source_as_of": "Source as of",
        "amendment_url": "Amendment",
        "attribution": "Attribution",
        "license": "License",
        "license_url": "License URL",
        "license_scope": "License scope",
        "changes_notice": "Changes",
        "content_use_notice": "Content use",
        "licensing_requirements_url": "Licensing requirements",
    }
    for key, label in notice_labels.items():
        if key in report["framework"]:
            line(f"{label}: {report['framework'][key]}")
    line("JSON is authoritative. This PDF is a limited projection.", font="Helvetica-Bold")
    y -= 6
    line("Control crosswalk", font="Helvetica-Bold", size=12)
    for control in report["controls"]:
        line(
            f"{control['control_id']} — {control['title']} — {control['status']} "
            f"(signals {control['signal_count']}, gaps {control['gap_count']})",
            font="Helvetica-Bold",
        )
        line(control["mapping_note"])
    y -= 6
    line("Limitations", font="Helvetica-Bold", size=12)
    for limitation in report["limitations"]:
        line(f"- {limitation}")
    document.save()
    return buffer.getvalue()


def _atomic_write_private(path_value: str | Path, data: bytes) -> None:
    path = Path(path_value)
    if path.name in {"", ".", ".."}:
        raise ComplianceReportError("--output must name a file")
    parent = path.parent
    for candidate in (parent, *parent.parents):
        if candidate.is_symlink():
            raise ComplianceReportError("refusing a symlinked output path")
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ComplianceReportError("output directory could not be created") from exc
    if parent.is_symlink() or not parent.is_dir():
        raise ComplianceReportError("output directory is not a safe directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ComplianceReportError("output must be a non-symlinked regular file")
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=parent, prefix=f".{path.name}.", delete=False,
        ) as temporary:
            temporary_name = temporary.name
            os.chmod(temporary_name, 0o600)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ComplianceReportError("output target changed during the write")
        os.replace(temporary_name, path)
        temporary_name = ""
        try:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass


def render_compliance_report(report: dict[str, Any], output_format: str) -> bytes:
    if output_format == "json":
        return format_compliance_json(report)
    if output_format == "sarif":
        return format_compliance_sarif(report)
    if output_format == "pdf":
        return format_compliance_pdf(report)
    raise ComplianceReportError("unsupported compliance report format")


def cmd_compliance_report(args: argparse.Namespace) -> int:
    """CLI handler for the isolated top-level compliance-report command."""
    try:
        output_format = getattr(args, "format", "json")
        output = getattr(args, "output", None)
        if output_format == "pdf" and not output:
            raise ComplianceReportError("PDF output requires --output FILE")
        stores = enumerate_trace_stores(getattr(args, "trace_dir", DEFAULT_TRACE_DIR))
        report = build_compliance_report(
            stores,
            getattr(args, "framework"),
            since=getattr(args, "since", None),
            until=getattr(args, "until", None),
            policy_path=getattr(args, "policy", None),
            rescan=getattr(args, "rescan", None),
        )
        rendered = render_compliance_report(report, output_format)
        if output:
            _atomic_write_private(output, rendered)
        else:
            stream: TextIO = sys.stdout
            stream.write(rendered.decode("utf-8"))
        return 0
    except (ComplianceReportError, OSError, ValueError) as exc:
        # Report errors intentionally avoid echoing raw trace values or local paths.
        sys.stderr.write(f"Error: {exc}\n")
        return 1
