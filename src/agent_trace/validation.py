"""Versioned, dependency-free validation for public AgentTrace records."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

from .assignment import (
    AssignmentError,
    MAX_EVENTS,
    MAX_JSON_BYTES,
    MAX_SOURCE_EVENT_BYTES,
    MAX_SOURCE_META_BYTES,
    MAX_SOURCE_TRACE_BYTES,
    _open_directory_chain,
    _read_child_file,
    _read_regular_nofollow,
    _strict_json_bytes,
)
from .audit import verify_chain_lines
from .compliance_report import (
    ComplianceReportError,
    _parse_event_snapshot,
    _parse_meta_snapshot,
)
from .evidence_health import EvidenceHealthStatus, assess_evidence_health
from .freeze import RegressionFixture, compare_fixtures, extract_steps
from .models import EventType, SessionMeta, TraceEvent
from .store import validate_workspace_id


TRACE_SCHEMA_VERSION = "1.0.0"
VALIDATION_SCHEMA = "agent-strace-trace-validation/v1"
CORPUS_SCHEMA = "agent-strace-agent-evidence-corpus/v1"

_SCHEMA_FILES = {
    "meta": "schemas/trace/v1/meta.schema.json",
    "event": "schemas/trace/v1/event.schema.json",
}
_CORPUS_MANIFEST = "corpus/v1/manifest.json"
_COUNT_FIELDS = {
    "tool_calls": EventType.TOOL_CALL,
    "llm_requests": EventType.LLM_REQUEST,
    "errors": EventType.ERROR,
}
_ORDER_CODES = {
    "timestamp_regression",
    "out_of_order_tool_outcome",
    "out_of_order_llm_outcome",
    "out_of_order_parent",
}
_INVALID_EVIDENCE_CODES = {
    "duplicate_event_id",
    "duplicate_session_start",
    "duplicate_session_end",
    "events_after_session_end",
    "events_before_session_start",
    "mixed_session_ids",
    *_ORDER_CODES,
}
_OUTCOME_CODES = {
    "unpaired_tool_call",
    "unpaired_llm_request",
}
_ORPHAN_CODES = {
    "orphan_tool_result",
    "orphan_llm_response",
    "orphan_error",
    "unlinked_error",
    "unknown_parent",
}


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    severity: str
    message: str
    event_id: str = ""
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: value
            for key, value in payload.items()
            if value not in ("", None)
        }


@dataclass(frozen=True)
class TraceValidationReport:
    target: str
    session_id: str
    status: str
    labels: tuple[str, ...]
    findings: tuple[ValidationFinding, ...]
    hash_chain: dict[str, Any]
    evidence_health: dict[str, Any]
    summary: dict[str, Any]
    replay: dict[str, Any] | None = None
    schema: str = VALIDATION_SCHEMA
    schema_version: str = TRACE_SCHEMA_VERSION

    @property
    def conformant(self) -> bool:
        return self.status == "valid"

    @property
    def exit_code(self) -> int:
        return 0 if self.conformant else 1

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "target": self.target,
            "session_id": self.session_id,
            "status": self.status,
            "conformant": self.conformant,
            "labels": list(self.labels),
            "findings": [finding.to_dict() for finding in self.findings],
            "hash_chain": self.hash_chain,
            "evidence_health": self.evidence_health,
            "summary": self.summary,
        }
        if self.replay is not None:
            payload["replay"] = self.replay
        return payload


def _package_text(relative_path: str) -> str:
    resource = resources.files("agent_trace")
    for part in relative_path.split("/"):
        resource = resource.joinpath(part)
    return resource.read_text(encoding="utf-8")


def load_trace_schema(kind: str) -> dict[str, Any]:
    """Load one packaged public trace schema."""
    try:
        relative = _SCHEMA_FILES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown trace schema: {kind}") from exc
    value = json.loads(_package_text(relative))
    if not isinstance(value, dict):
        raise ValueError(f"packaged {kind} schema must be an object")
    return value


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value).is_integer()
        )
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported packaged schema type: {expected}")


def _schema_errors(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_json_type_matches(value, item) for item in types):
            return [f"{path} must have type {' or '.join(types)}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path} must equal {schema['const']!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(f"{path} is longer than maxLength")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(str(pattern), value) is None:
            errors.append(f"{path} does not match the required pattern")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} is below the minimum")

    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(
                _schema_errors(item, schema["items"], path=f"{path}[{index}]")
            )

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors.extend(
                    _schema_errors(item, properties[key], path=f"{path}.{key}")
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key} is not allowed")
    return errors


def _read_trace_files(root: Path) -> tuple[bytes, bytes]:
    directory_fd = _open_directory_chain(root)
    try:
        meta_raw = _read_child_file(
            directory_fd,
            "meta.json",
            max_bytes=MAX_SOURCE_META_BYTES,
            label="session metadata",
        )
        events_raw = _read_child_file(
            directory_fd,
            "events.ndjson",
            max_bytes=MAX_SOURCE_TRACE_BYTES,
            label="session event stream",
        )
    finally:
        os.close(directory_fd)
    return meta_raw, events_raw


def _parse_schema_records(
    meta_raw: bytes,
    events_raw: bytes,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[ValidationFinding]]:
    findings: list[ValidationFinding] = []
    try:
        meta_value = _strict_json_bytes(
            meta_raw,
            label="session metadata",
            max_bytes=MAX_SOURCE_META_BYTES,
            source_text=True,
        )
    except AssignmentError as exc:
        return {}, [], [
            ValidationFinding("metadata_schema_invalid", "error", str(exc))
        ]
    if not isinstance(meta_value, dict):
        return {}, [], [
            ValidationFinding(
                "metadata_schema_invalid",
                "error",
                "session metadata must be an object",
            )
        ]
    for message in _schema_errors(meta_value, load_trace_schema("meta")):
        findings.append(
            ValidationFinding("metadata_schema_invalid", "error", message)
        )

    raw_lines = events_raw.splitlines()
    if len(raw_lines) > MAX_EVENTS:
        findings.append(
            ValidationFinding(
                "event_schema_invalid",
                "error",
                "event count exceeds the supported limit",
            )
        )
        return meta_value, [], findings

    event_values: list[dict[str, Any]] = []
    event_schema = load_trace_schema("event")
    for line_number, raw_line in enumerate(raw_lines, 1):
        if not raw_line.strip():
            findings.append(
                ValidationFinding(
                    "event_schema_invalid",
                    "error",
                    "event stream contains a blank NDJSON record",
                    line=line_number,
                )
            )
            continue
        if len(raw_line) > MAX_SOURCE_EVENT_BYTES:
            findings.append(
                ValidationFinding(
                    "event_schema_invalid",
                    "error",
                    "event record exceeds the supported size limit",
                    line=line_number,
                )
            )
            continue
        try:
            value = _strict_json_bytes(
                raw_line,
                label="session event",
                max_bytes=MAX_SOURCE_EVENT_BYTES,
                source_text=True,
            )
        except AssignmentError as exc:
            findings.append(
                ValidationFinding(
                    "event_schema_invalid",
                    "error",
                    str(exc),
                    line=line_number,
                )
            )
            continue
        if not isinstance(value, dict):
            findings.append(
                ValidationFinding(
                    "event_schema_invalid",
                    "error",
                    "session event must be an object",
                    line=line_number,
                )
            )
            continue
        event_values.append(value)
        for message in _schema_errors(value, event_schema):
            findings.append(
                ValidationFinding(
                    "event_schema_invalid",
                    "error",
                    message,
                    event_id=str(value.get("event_id", "")),
                    line=line_number,
                )
            )
    return meta_value, event_values, findings


def _semantic_error_code(message: str) -> str:
    lowered = message.lower()
    if "identifier" in lowered or "identity" in lowered or "directory" in lowered:
        return "session_identity_mismatch"
    if "event data" in lowered:
        return "event_data_not_object"
    if "event type" in lowered:
        return "unknown_event_type"
    return "semantic_validation_failed"


def _finding_for_reason(code: str, message: str, event_id: str) -> ValidationFinding:
    severity = "error" if code in _INVALID_EVIDENCE_CODES else "warning"
    return ValidationFinding(code, severity, message, event_id=event_id)


def _parent_findings(events: Sequence[TraceEvent]) -> list[ValidationFinding]:
    positions = {event.event_id: index for index, event in enumerate(events)}
    findings: list[ValidationFinding] = []
    relationship_types = {
        EventType.TOOL_RESULT,
        EventType.LLM_RESPONSE,
        EventType.ERROR,
    }
    for index, event in enumerate(events):
        if not event.parent_id or event.event_type in relationship_types:
            continue
        parent_position = positions.get(event.parent_id)
        if parent_position is None:
            findings.append(
                ValidationFinding(
                    "unknown_parent",
                    "warning",
                    f"event {event.event_id!r} references unknown parent "
                    f"{event.parent_id!r}",
                    event_id=event.event_id,
                )
            )
        elif parent_position >= index:
            findings.append(
                ValidationFinding(
                    "out_of_order_parent",
                    "error",
                    f"event {event.event_id!r} appears before its parent "
                    f"{event.parent_id!r}",
                    event_id=event.event_id,
                )
            )
    return findings


def _summary_result(
    raw_meta: dict[str, Any],
    meta: SessionMeta,
    events: Sequence[TraceEvent],
) -> tuple[dict[str, Any], list[ValidationFinding]]:
    observed = {
        field: sum(event.event_type == event_type for event in events)
        for field, event_type in _COUNT_FIELDS.items()
    }
    findings: list[ValidationFinding] = []
    compared: dict[str, dict[str, int | float]] = {}
    for field in _COUNT_FIELDS:
        if field not in raw_meta:
            continue
        recorded = raw_meta[field]
        compared[field] = {"recorded": recorded, "observed": observed[field]}
        if recorded != observed[field]:
            findings.append(
                ValidationFinding(
                    "summary_contradiction",
                    "error",
                    f"metadata {field} is {recorded}, but the event stream has "
                    f"{observed[field]}",
                )
            )

    observed_end = any(event.event_type == EventType.SESSION_END for event in events)
    if meta.ended_at is not None and not observed_end:
        findings.append(
            ValidationFinding(
                "summary_contradiction",
                "error",
                "metadata has ended_at, but the event stream has no session_end",
            )
        )

    state = "contradictory" if findings else ("consistent" if compared else "unreported")
    return {
        "state": state,
        "compared": compared,
        "observed": observed,
        "usage_presence": (
            "missing"
            if "total_tokens" not in raw_meta
            else ("zero" if raw_meta["total_tokens"] == 0 else "recorded")
        ),
    }, findings


def _is_recovered(events: Sequence[TraceEvent]) -> bool:
    failed_calls = {
        event.parent_id: index
        for index, event in enumerate(events)
        if event.event_type == EventType.ERROR and event.parent_id
    }
    outcomes: dict[str, list[tuple[int, TraceEvent]]] = {}
    for index, event in enumerate(events):
        if event.event_type == EventType.TOOL_RESULT and event.parent_id:
            outcomes.setdefault(event.parent_id, []).append((index, event))
    for call_index, event in enumerate(events):
        if event.event_type != EventType.TOOL_CALL:
            continue
        retry_of = event.data.get("retry_of")
        failure_index = failed_calls.get(str(retry_of))
        if failure_index is None or failure_index >= call_index:
            continue
        for outcome_index, outcome in outcomes.get(event.event_id, []):
            exit_code = outcome.data.get("exit_code")
            if outcome_index > call_index and exit_code in (None, 0, "0"):
                return True
    return False


def _replay_result(
    replay_baseline: Path | None,
    meta: SessionMeta,
    events: Sequence[TraceEvent],
) -> tuple[dict[str, Any] | None, list[ValidationFinding], set[str]]:
    if replay_baseline is None:
        return None, [], set()
    try:
        raw = _read_regular_nofollow(
            replay_baseline,
            max_bytes=MAX_JSON_BYTES,
            label="replay baseline",
        )
        value = _strict_json_bytes(
            raw,
            label="replay baseline",
            max_bytes=MAX_JSON_BYTES,
        )
        if not isinstance(value, dict):
            raise AssignmentError("replay baseline must be an object")
        steps = value.get("steps", [])
        if not isinstance(steps, list):
            raise AssignmentError("replay baseline steps must be an array")
        if any(not isinstance(step, dict) for step in steps):
            raise AssignmentError("replay baseline steps must contain objects")
        expected = RegressionFixture.from_json(
            json.dumps(value, separators=(",", ":"), allow_nan=False)
        )
    except (
        AssignmentError,
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:
        return (
            {"baseline": str(replay_baseline), "state": "invalid"},
            [ValidationFinding("replay_baseline_invalid", "error", str(exc))],
            {"replay_baseline_invalid"},
        )
    actual = RegressionFixture(
        session=meta.session_id,
        task=expected.task,
        steps=extract_steps(list(events)),
    )
    comparison = compare_fixtures(expected, actual, threshold=0.0)
    state = "drift" if comparison.exceeded else "equivalent"
    findings = (
        [
            ValidationFinding(
                "replay_drift",
                "error",
                f"tool-call replay diverged by {comparison.divergence_score:.3f}",
            )
        ]
        if comparison.exceeded
        else []
    )
    return {
        "baseline": str(replay_baseline),
        "state": state,
        "comparison": comparison.to_dict(),
    }, findings, {f"replay_{state}"}


def _base_invalid_report(
    root: Path,
    findings: Iterable[ValidationFinding],
) -> TraceValidationReport:
    finding_tuple = tuple(findings)
    return TraceValidationReport(
        target=str(root),
        session_id=root.name,
        status="invalid",
        labels=tuple(sorted(("invalid", "nonconformant", "schema_invalid"))),
        findings=finding_tuple,
        hash_chain={"state": "not_checked", "valid": False, "event_count": 0},
        evidence_health={},
        summary={"state": "not_checked"},
    )


def validate_trace_directory(
    path: str | Path,
    *,
    allow_active: bool = False,
    replay_baseline: str | Path | None = None,
) -> TraceValidationReport:
    """Validate one trace directory without rewriting it."""
    root = Path(path).absolute()
    try:
        meta_raw, events_raw = _read_trace_files(root)
    except (AssignmentError, OSError, ValueError) as exc:
        return _base_invalid_report(
            root,
            [ValidationFinding("trace_read_failed", "error", str(exc))],
        )

    raw_meta, _raw_events, schema_findings = _parse_schema_records(
        meta_raw, events_raw
    )
    if schema_findings:
        return _base_invalid_report(root, schema_findings)

    try:
        meta = _parse_meta_snapshot(meta_raw, root.name)
        events, chain_state = _parse_event_snapshot(events_raw, meta)
    except ComplianceReportError as exc:
        return _base_invalid_report(
            root,
            [
                ValidationFinding(
                    _semantic_error_code(str(exc)),
                    "error",
                    str(exc),
                )
            ],
        )

    event_lines = [
        line.decode("utf-8")
        for line in events_raw.splitlines()
        if line.strip()
    ]
    chain = verify_chain_lines(event_lines, session_id=meta.session_id)
    findings: list[ValidationFinding] = []
    if not chain.ok or chain_state == "broken":
        findings.append(
            ValidationFinding(
                "hash_chain_broken",
                "error",
                "event hash chain is broken",
                event_id=chain.broken_event_id,
                line=(chain.broken_at + 1) if chain.broken_at is not None else None,
            )
        )
    elif chain_state == "partial":
        findings.append(
            ValidationFinding(
                "hash_chain_partial",
                "warning",
                "event stream mixes linked and legacy hash-chain records",
            )
        )

    health = assess_evidence_health(
        events,
        session_finalized=False if allow_active else True,
    )
    for reason in health.reasons:
        findings.append(
            _finding_for_reason(reason.code, reason.message, reason.event_id)
        )
    findings.extend(_parent_findings(events))

    summary, summary_findings = _summary_result(raw_meta, meta, events)
    findings.extend(summary_findings)

    replay_path = Path(replay_baseline).absolute() if replay_baseline else None
    replay, replay_findings, replay_labels = _replay_result(
        replay_path, meta, events
    )
    findings.extend(replay_findings)

    labels: set[str] = {"schema_valid"}
    labels.add("legacy" if chain_state == "legacy" else "current")
    labels.add(f"hash_chain_{chain_state}")
    labels.update(replay_labels)

    reason_codes = {reason.code for reason in health.reasons}
    complete = (
        health.observed_start
        and health.observed_end
        and not reason_codes.intersection(_OUTCOME_CODES)
    )
    labels.add("complete" if complete else "incomplete")
    if any(event.redacted for event in events) or meta.redacted:
        labels.add("redacted")
    if _is_recovered(events):
        labels.add("recovered")

    if "total_tokens" not in raw_meta:
        labels.add("usage_missing")
    elif raw_meta["total_tokens"] == 0:
        labels.add("usage_zero")
    else:
        labels.add("usage_recorded")

    if "duplicate_event_id" in reason_codes:
        labels.add("duplicate_ids")
    if reason_codes.intersection(_ORDER_CODES):
        labels.add("out_of_order")
    if reason_codes.intersection(_OUTCOME_CODES):
        labels.add("missing_outcome")
    if "missing_session_end" in reason_codes:
        labels.add("terminal_absent")
    if reason_codes.intersection(_ORPHAN_CODES):
        labels.add("orphan_parent")
    if "duplicate_tool_outcome" in reason_codes or "duplicate_llm_outcome" in reason_codes:
        labels.add("duplicate_outcomes")
    if chain_state == "broken" or not chain.ok:
        labels.add("tampered")
    if summary["state"] == "contradictory":
        labels.add("summary_contradiction")

    severities = {finding.severity for finding in findings}
    if "error" in severities or health.status == EvidenceHealthStatus.INVALID:
        status = "invalid"
    elif "warning" in severities:
        status = "partial"
    else:
        status = "valid"
    labels.add(status)
    labels.add("conformant" if status == "valid" else "nonconformant")

    return TraceValidationReport(
        target=str(root),
        session_id=meta.session_id,
        status=status,
        labels=tuple(sorted(labels)),
        findings=tuple(findings),
        hash_chain={
            "state": chain_state,
            "valid": chain.ok and chain_state != "broken",
            "event_count": len(events),
            "broken_at": chain.broken_at,
            "broken_event_id": chain.broken_event_id,
        },
        evidence_health=health.to_dict(),
        summary=summary,
        replay=replay,
    )


def _corpus_root() -> Path:
    return Path(str(resources.files("agent_trace").joinpath("corpus/v1")))


def validate_packaged_corpus() -> dict[str, Any]:
    """Validate every packaged synthetic corpus case against its expectations."""
    root = _corpus_root()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError("unsupported packaged corpus manifest")
    case_results = []
    all_matched = True
    for case in manifest.get("cases", []):
        case_root = root / case["path"]
        baseline = case_root / case["replay_baseline"] if case.get("replay_baseline") else None
        report = validate_trace_directory(
            case_root,
            replay_baseline=baseline,
        )
        expected_labels = sorted(case["expected_labels"])
        matched = (
            report.status == case["expected_status"]
            and list(report.labels) == expected_labels
        )
        all_matched = all_matched and matched
        case_results.append(
            {
                "id": case["id"],
                "matched": matched,
                "expected_status": case["expected_status"],
                "expected_labels": expected_labels,
                "actual": report.to_dict(),
            }
        )
    return {
        "schema": "agent-strace-corpus-validation/v1",
        "corpus_schema": manifest["schema"],
        "corpus_version": manifest["version"],
        "matched": all_matched,
        "case_count": len(case_results),
        "cases": case_results,
    }


def format_validation_text(
    report: TraceValidationReport,
    out: TextIO | None = None,
) -> None:
    out = out or sys.stdout
    out.write(f"Trace validation: {report.status}\n")
    out.write(f"Session: {report.session_id}\n")
    out.write(f"Schema: {report.schema_version}\n")
    out.write(f"Hash chain: {report.hash_chain['state']}\n")
    out.write(f"Labels: {', '.join(report.labels)}\n")
    out.write(f"Findings: {len(report.findings)}\n")
    for finding in report.findings:
        location = f" line {finding.line}" if finding.line is not None else ""
        event = f" event {finding.event_id}" if finding.event_id else ""
        out.write(
            f"  {finding.severity}: {finding.code}:{location}{event} "
            f"{finding.message}\n"
        )


def _resolve_target(args: argparse.Namespace) -> Path | None:
    requested = getattr(args, "target", None)
    if requested:
        direct = Path(requested)
        if direct.is_dir():
            return direct
    root = Path(args.trace_dir)
    workspace = os.environ.get("AGENT_STRACE_WORKSPACE", "")
    if workspace:
        try:
            root = root / "workspaces" / validate_workspace_id(workspace)
        except ValueError:
            return None
    try:
        candidates = [
            item
            for item in root.iterdir()
            if item.is_dir()
            and not item.is_symlink()
            and (item / "meta.json").is_file()
            and (item / "events.ndjson").is_file()
        ]
    except OSError:
        return None

    if requested:
        exact = [item for item in candidates if item.name == requested]
        if exact:
            return exact[0]
        matches = [item for item in candidates if item.name.startswith(requested)]
        return matches[0] if len(matches) == 1 else None
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: max(
            (item / "meta.json").stat().st_mtime_ns,
            (item / "events.ndjson").stat().st_mtime_ns,
        ),
    )


def cmd_validate(args: argparse.Namespace) -> int:
    """CLI handler for trace and packaged corpus validation."""
    if getattr(args, "corpus", False):
        try:
            result = validate_packaged_corpus()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            sys.stderr.write(f"Could not validate packaged corpus: {exc}\n")
            return 1
        if getattr(args, "format", "text") == "json":
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
        else:
            status = "matched" if result["matched"] else "mismatch"
            sys.stdout.write(
                f"Agent Evidence Corpus: {status} "
                f"({result['case_count']} cases)\n"
            )
            for case in result["cases"]:
                marker = "ok" if case["matched"] else "mismatch"
                sys.stdout.write(
                    f"  {marker}: {case['id']} "
                    f"({case['actual']['status']})\n"
                )
        return 0 if result["matched"] else 1

    target = _resolve_target(args)
    if target is None:
        requested = getattr(args, "target", None)
        sys.stderr.write(
            f"Trace not found: {requested}\n" if requested else "No sessions found.\n"
        )
        return 1
    report = validate_trace_directory(
        target,
        allow_active=getattr(args, "allow_active", False),
        replay_baseline=getattr(args, "replay_baseline", None),
    )
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        format_validation_text(report)
    return report.exit_code
