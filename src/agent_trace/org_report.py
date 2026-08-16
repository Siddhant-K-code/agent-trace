"""Organization-wide estimated agent usage and cost reporting."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import statistics
import sys
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .collector_client import (
    CollectorAuthenticationError,
    CollectorClient,
    CollectorClientError,
    CollectorTraceStore,
    MAX_JSON_DEPTH,
)
from .cost import DEFAULT_MODEL, PRICING, PRICING_SNAPSHOT_DATE, estimate_cost
from .curve import classify_session
from .lint import lint_session
from .models import EventType
from .store import validate_stored_id
from .team_report import session_engineer_attribution
from .tenancy import enumerate_trace_stores


MIN_ANOMALY_SESSIONS = 3
MIN_ANOMALY_COVERAGE = 0.80
MIN_ANOMALY_PEERS = 3
TEAM_HIGH_COST_RATIO = 1.5
ENGINEER_HIGH_COST_RATIO = 2.0
MAX_AUTH_KEY_FILE_BYTES = 16 * 1024
MAX_REPORT_LABEL_CHARS = 200


@dataclass(frozen=True)
class TeamBreakdown:
    team: str
    sessions: int
    cost_estimated_sessions: int
    cost_estimate_coverage_percent: float
    estimated_spend_usd: float
    avg_estimated_cost_usd: float
    active_attributed_identities: int
    lint_findings: int
    sessions_with_lint_signals: int
    lint_signals: dict[str, int]


@dataclass(frozen=True)
class TaskTypeBreakdown:
    task_type: str
    sessions: int
    cost_estimated_sessions: int
    cost_estimate_coverage_percent: float
    estimated_spend_usd: float
    avg_estimated_cost_usd: float


@dataclass(frozen=True)
class OrgAnomaly:
    kind: str
    subject: str
    sessions: int
    cost_estimate_coverage_percent: float
    ratio_to_peer_median: float
    eligible_peers: int
    callout: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrgReport:
    generated_at: str
    month: str
    source_mode: str
    organization_boundary: str
    pricing_model: str
    pricing_snapshot_date: str
    cost_estimation_method: str
    team_filter: str
    anonymized: bool
    session_count: int
    cost_estimated_sessions: int
    cost_estimate_coverage_percent: float
    total_estimated_spend_usd: float
    avg_estimated_cost_usd: float
    median_estimated_cost_usd: float
    active_attributed_identities: int
    identity_attributed_sessions: int
    identity_coverage_percent: float
    identity_confidence: dict[str, int]
    teams: tuple[TeamBreakdown, ...]
    task_types: tuple[TaskTypeBreakdown, ...]
    anomalies: tuple[OrgAnomaly, ...]
    snapshot_limitation: str


@dataclass(frozen=True)
class _SessionMetric:
    team: str
    engineer: str
    identity_confidence: str
    task_type: str
    cost: float | None
    lint_rules: tuple[str, ...]


class _SessionSnapshotStore:
    """Expose one already-loaded session snapshot to existing analyzers."""

    def __init__(self, meta, events: list):  # noqa: ANN001
        self._meta = meta
        self._events = tuple(events)

    def load_meta(self, session_id: str):  # noqa: ANN201
        if session_id != self._meta.session_id:
            raise FileNotFoundError(f"session not found in report snapshot: {session_id}")
        return self._meta

    def load_events(self, session_id: str) -> list:
        self.load_meta(session_id)
        return list(self._events)


def month_bounds_utc(month: str) -> tuple[float, float]:
    """Return a half-open UTC timestamp range for ``YYYY-MM``."""
    if not re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", str(month)):
        raise ValueError("month must use YYYY-MM format")
    try:
        start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("month must use YYYY-MM format") from exc
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start.timestamp(), end.timestamp()


def _clean_label(value: str, fallback: str) -> str:
    raw = str(value)
    if len(raw) > MAX_REPORT_LABEL_CHARS:
        raise ValueError(
            f"report label cannot exceed {MAX_REPORT_LABEL_CHARS} characters"
        )
    if any(
        unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"}
        for char in raw
    ):
        raise ValueError("report label cannot contain control or format characters")
    cleaned = raw.strip()
    if raw != cleaned:
        raise ValueError("report label cannot have leading or trailing whitespace")
    return cleaned or fallback


def _finite_nonnegative(value, field: str, *, optional: bool = False) -> float | None:  # noqa: ANN001
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _validate_nested_data(value, field: str) -> None:  # noqa: ANN001
    """Reject hostile nesting before estimates and lint walk event payloads."""
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"{field} nesting exceeds the report limit")
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError(f"{field} object keys must be strings")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{field} numbers must be finite")
        elif not isinstance(item, (str, int, float, bool, type(None))):
            raise ValueError(f"{field} contains a non-JSON value")


def _validate_report_meta(meta) -> None:  # noqa: ANN001
    """Apply one source-neutral schema contract to report metadata."""
    if not isinstance(getattr(meta, "session_id", None), str):
        raise ValueError("session_id must be a string")
    validate_stored_id(meta.session_id, "report session ID")
    _finite_nonnegative(getattr(meta, "started_at", None), "started_at")
    _finite_nonnegative(getattr(meta, "ended_at", None), "ended_at", optional=True)
    _finite_nonnegative(
        getattr(meta, "total_duration_ms", None), "total_duration_ms"
    )
    for field in (
        "agent_name", "command", "parent_session_id", "parent_event_id", "team",
        "workspace_id", "trace_id", "parent_span_id", "trace_flags", "tenant_id",
    ):
        if not isinstance(getattr(meta, field, None), str):
            raise ValueError(f"{field} must be a string")
    _clean_label(meta.team, "")
    _clean_label(meta.workspace_id, "")
    if meta.parent_session_id:
        validate_stored_id(meta.parent_session_id, "report parent session ID")
    for field in ("tool_calls", "llm_requests", "errors", "total_tokens", "depth"):
        value = getattr(meta, field, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if not isinstance(getattr(meta, "redacted", None), bool):
        raise ValueError("redacted must be a boolean")
    attribution = getattr(meta, "attribution", None)
    if not isinstance(attribution, dict):
        raise ValueError("attribution must be an object")
    for field in (
        "actor_id", "engineer_id", "user_id", "git_author",
        "git_author_email", "os_user", "hostname",
    ):
        if field in attribution and attribution[field] is not None and not isinstance(
            attribution[field], str
        ):
            raise ValueError(f"attribution {field} must be a string")
        if attribution.get(field):
            _clean_label(attribution[field], "")
    _validate_nested_data(attribution, "attribution")


def _validate_report_events(meta, events: list) -> None:  # noqa: ANN001
    """Validate selected event objects equally for local and remote stores."""
    stream_tenant = ""
    for event in events:
        if not isinstance(getattr(event, "event_type", None), EventType):
            raise ValueError("event_type must be a recognized event type")
        _finite_nonnegative(getattr(event, "timestamp", None), "event timestamp")
        _finite_nonnegative(
            getattr(event, "duration_ms", None), "event duration_ms", optional=True
        )
        for field in ("event_id", "session_id", "parent_id", "prev_hash", "tenant_id"):
            if not isinstance(getattr(event, field, None), str):
                raise ValueError(f"event {field} must be a string")
        if not event.event_id:
            raise ValueError("event_id must be non-empty")
        if event.session_id != meta.session_id:
            raise ValueError("event session ID does not match its metadata")
        if not isinstance(getattr(event, "data", None), dict):
            raise ValueError("event data must be an object")
        _validate_nested_data(event.data, "event data")
        if not isinstance(getattr(event, "redacted", None), bool):
            raise ValueError("event redacted must be a boolean")
        if meta.tenant_id and event.tenant_id and event.tenant_id != meta.tenant_id:
            raise ValueError("event tenant does not match its metadata")
        if not meta.tenant_id and event.tenant_id:
            if stream_tenant and event.tenant_id != stream_tenant:
                raise ValueError("event stream contains mixed tenants")
            stream_tenant = event.tenant_id


def _alias_map(kind: str, values: set[str]) -> dict[str, str]:
    """Build deterministic report-local aliases without reversible hashing."""
    return {
        value: f"{kind.title()}-{index:03d}"
        for index, value in enumerate(
            sorted(value for value in values if value not in {"(unknown)", "(unassigned)"}),
            1,
        )
    }


def _team_for(meta, workspace_id: str) -> tuple[str, str]:  # noqa: ANN001
    tagged = str(getattr(meta, "team", "") or "").strip()
    workspace = str(workspace_id or getattr(meta, "workspace_id", "") or "").strip()
    return tagged or workspace or "(unassigned)", workspace


def _matches_team(
    meta, workspace_id: str, selector_kind: str, selector_value: str,
) -> bool:  # noqa: ANN001
    if not selector_kind:
        return True
    tagged = str(getattr(meta, "team", "") or "").strip()
    workspace = str(workspace_id or getattr(meta, "workspace_id", "") or "").strip()
    return tagged == selector_value if selector_kind == "tag" else workspace == selector_value


def _resolve_team_selector(inventory: list[tuple], requested: str) -> tuple[str, str]:
    if not requested:
        return "", ""
    if requested.startswith("tag:"):
        value = requested[4:].strip()
        if not value:
            raise ValueError("team tag selector cannot be empty")
        return "tag", value
    if requested.startswith("workspace:"):
        value = requested[10:].strip()
        if not value:
            raise ValueError("workspace selector cannot be empty")
        return "workspace", value
    tags = {
        str(getattr(meta, "team", "") or "").strip()
        for _, meta, _ in inventory
    }
    workspaces = {
        str(workspace or getattr(meta, "workspace_id", "") or "").strip()
        for _, meta, workspace in inventory
    }
    in_tags = requested in tags
    in_workspaces = requested in workspaces
    if in_tags and in_workspaces:
        raise ValueError(
            "ambiguous --team selector; use tag:NAME or workspace:NAME"
        )
    if in_tags:
        return "tag", requested
    if in_workspaces:
        return "workspace", requested
    # Preserve empty-report behavior for an unmatched unprefixed selector.
    return "tag", requested


def _group_metrics(metrics: list[_SessionMetric], field: str) -> dict[str, list[_SessionMetric]]:
    groups: dict[str, list[_SessionMetric]] = {}
    for metric in metrics:
        groups.setdefault(str(getattr(metric, field)), []).append(metric)
    return groups


def _lint_counts(rows: list[_SessionMetric]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for rule in row.lint_rules:
            counts[rule] = counts.get(rule, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _build_anomalies(
    metrics: list[_SessionMetric],
    team_rows: list[TeamBreakdown],
) -> list[OrgAnomaly]:
    anomalies: list[OrgAnomaly] = []
    eligible_teams = [
        team for team in team_rows
        if team.cost_estimated_sessions >= MIN_ANOMALY_SESSIONS
        and team.cost_estimate_coverage_percent >= MIN_ANOMALY_COVERAGE * 100
    ]
    if len(eligible_teams) > MIN_ANOMALY_PEERS:
        for team in eligible_teams:
            peer_averages = [
                peer.avg_estimated_cost_usd
                for peer in eligible_teams
                if peer is not team
            ]
            peer_median = statistics.median(peer_averages)
            if peer_median <= 0:
                continue
            ratio = team.avg_estimated_cost_usd / peer_median
            if ratio < TEAM_HIGH_COST_RATIO:
                continue
            details: tuple[str, ...] = ()
            if team.lint_signals:
                rule, count = next(iter(team.lint_signals.items()))
                details = (
                    f"Most frequent structural lint signal: {rule} ({count} finding(s)); heuristic only.",
                )
            anomalies.append(OrgAnomaly(
                kind="team-estimated-cost",
                subject=team.team,
                sessions=team.cost_estimated_sessions,
                cost_estimate_coverage_percent=team.cost_estimate_coverage_percent,
                ratio_to_peer_median=ratio,
                eligible_peers=len(peer_averages),
                callout=(
                    f"Average estimate per cost-estimated session is {ratio:.2f}x "
                    f"the median of {len(peer_averages)} eligible team peers "
                    f"({team.cost_estimated_sessions} sessions; "
                    f"{team.cost_estimate_coverage_percent:.1f}% coverage)."
                ),
                details=details,
            ))

    eligible_engineers: list[tuple[str, list[_SessionMetric], float, float]] = []
    for engineer, rows in _group_metrics(metrics, "engineer").items():
        priced = [row for row in rows if row.cost is not None]
        coverage = len(priced) / len(rows) if rows else 0.0
        if (
            engineer == "(unknown)"
            or len(priced) < MIN_ANOMALY_SESSIONS
            or coverage < MIN_ANOMALY_COVERAGE
        ):
            continue
        average = sum(row.cost for row in priced if row.cost is not None) / len(priced)
        eligible_engineers.append((engineer, priced, average, coverage))
    if len(eligible_engineers) > MIN_ANOMALY_PEERS:
        for engineer, priced, average, coverage in eligible_engineers:
            peer_averages = [
                item[2] for item in eligible_engineers if item[0] != engineer
            ]
            peer_median = statistics.median(peer_averages)
            if peer_median <= 0:
                continue
            ratio = average / peer_median
            if ratio < ENGINEER_HIGH_COST_RATIO:
                continue
            anomalies.append(OrgAnomaly(
                kind="identity-estimated-cost",
                subject=engineer,
                sessions=len(priced),
                cost_estimate_coverage_percent=coverage * 100,
                ratio_to_peer_median=ratio,
                eligible_peers=len(peer_averages),
                callout=(
                    f"Average estimate per cost-estimated session is {ratio:.2f}x "
                    f"the median of {len(peer_averages)} eligible identity peers "
                    f"({len(priced)} sessions; {coverage * 100:.1f}% coverage)."
                ),
            ))
    return sorted(
        anomalies,
        key=lambda item: (-item.ratio_to_peer_median, item.kind, item.subject),
    )


def build_org_report(
    stores: list,
    month: str,
    *,
    team: str = "",
    anonymize: bool = False,
    model: str = DEFAULT_MODEL,
    source_mode: str = "shared-directory",
    generated_at: str | None = None,
) -> OrgReport:
    """Build a deterministic report from strict local or collector stores."""
    if model not in PRICING:
        raise ValueError(f"unknown pricing model: {model}")
    start, end = month_bounds_utc(month)
    requested_team = str(team or "").strip()
    metrics: list[_SessionMetric] = []
    inventory: list[tuple] = []
    for store in stores:
        workspace_value = getattr(store, "workspace_id", "")
        if not isinstance(workspace_value, str):
            raise ValueError("store workspace_id must be a string")
        workspace_id = workspace_value
        # Remote stores return metadata only here.  The month/team filters run
        # before load_events, estimate_cost, or lint_session fetch any events.
        try:
            discovered = store.list_sessions_strict(validate_events=False)
        except RecursionError as exc:
            raise ValueError("trace metadata nesting exceeds the report limit") from exc
        for meta in discovered:
            _validate_report_meta(meta)
            timestamp = float(meta.started_at)
            if start <= timestamp < end:
                inventory.append((store, meta, workspace_id))

    selector_kind, selector_value = _resolve_team_selector(inventory, requested_team)
    for store, meta, workspace_id in inventory:
        if not _matches_team(meta, workspace_id, selector_kind, selector_value):
            continue
        team_name, _ = _team_for(meta, workspace_id)
        engineer, confidence = session_engineer_attribution(meta)
        engineer = _clean_label(engineer, "(unknown)")
        team_name = _clean_label(team_name, "(unassigned)")
        # Force the selected stream to load before analysis. Collector stores
        # cache it, so cost/lint never trigger duplicate requests.
        try:
            events = store.load_events(meta.session_id)
        except RecursionError as exc:
            raise ValueError("trace event nesting exceeds the report limit") from exc
        try:
            _validate_report_events(meta, events)
            snapshot = _SessionSnapshotStore(meta, events)
            try:
                cost_result = estimate_cost(snapshot, meta.session_id, model=model)
                cost: float | None = float(cost_result.total_cost)
                if not math.isfinite(cost) or cost < 0:
                    cost = None
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                cost = None
            lint = lint_session(snapshot, meta.session_id)
        finally:
            release = getattr(store, "release_events", None)
            if release is not None:
                release(meta.session_id)
        task_type = classify_session(meta.agent_name, meta.command)
        metrics.append(_SessionMetric(
            team=team_name,
            engineer=engineer,
            identity_confidence=confidence,
            task_type=task_type,
            cost=cost,
            lint_rules=tuple(finding.rule for finding in lint.findings),
        ))

    costs = [metric.cost for metric in metrics if metric.cost is not None]
    total_cost = sum(costs)
    average = total_cost / len(costs) if costs else 0.0
    median = statistics.median(costs) if costs else 0.0

    team_rows: list[TeamBreakdown] = []
    for name, rows in _group_metrics(metrics, "team").items():
        priced = [row.cost for row in rows if row.cost is not None]
        team_cost = sum(priced)
        identities = {row.engineer for row in rows if row.engineer != "(unknown)"}
        team_rows.append(TeamBreakdown(
            team=name,
            sessions=len(rows),
            cost_estimated_sessions=len(priced),
            cost_estimate_coverage_percent=len(priced) / len(rows) * 100,
            estimated_spend_usd=team_cost,
            avg_estimated_cost_usd=team_cost / len(priced) if priced else 0.0,
            active_attributed_identities=len(identities),
            lint_findings=sum(len(row.lint_rules) for row in rows),
            sessions_with_lint_signals=sum(bool(row.lint_rules) for row in rows),
            lint_signals=_lint_counts(rows),
        ))
    team_rows.sort(key=lambda row: (-row.estimated_spend_usd, row.team))

    task_rows: list[TaskTypeBreakdown] = []
    for task_type, rows in _group_metrics(metrics, "task_type").items():
        priced = [row.cost for row in rows if row.cost is not None]
        task_cost = sum(priced)
        avg_cost = task_cost / len(priced) if priced else 0.0
        task_rows.append(TaskTypeBreakdown(
            task_type=task_type,
            sessions=len(rows),
            cost_estimated_sessions=len(priced),
            cost_estimate_coverage_percent=len(priced) / len(rows) * 100,
            estimated_spend_usd=task_cost,
            avg_estimated_cost_usd=avg_cost,
        ))
    task_rows.sort(key=lambda row: (-row.sessions, row.task_type))

    anomalies = _build_anomalies(metrics, team_rows)
    if anonymize:
        team_aliases = _alias_map("team", {row.team for row in team_rows})
        identity_aliases = _alias_map(
            "identity", {metric.engineer for metric in metrics}
        )
        team_rows = [
            TeamBreakdown(
                team=team_aliases.get(row.team, row.team),
                sessions=row.sessions,
                cost_estimated_sessions=row.cost_estimated_sessions,
                cost_estimate_coverage_percent=row.cost_estimate_coverage_percent,
                estimated_spend_usd=row.estimated_spend_usd,
                avg_estimated_cost_usd=row.avg_estimated_cost_usd,
                active_attributed_identities=row.active_attributed_identities,
                lint_findings=row.lint_findings,
                sessions_with_lint_signals=row.sessions_with_lint_signals,
                lint_signals=row.lint_signals,
            )
            for row in team_rows
        ]
        anomalies = [
            OrgAnomaly(
                kind=item.kind,
                subject=(
                    team_aliases.get(item.subject, item.subject)
                    if item.kind == "team-estimated-cost"
                    else identity_aliases.get(item.subject, item.subject)
                ),
                sessions=item.sessions,
                cost_estimate_coverage_percent=item.cost_estimate_coverage_percent,
                ratio_to_peer_median=item.ratio_to_peer_median,
                eligible_peers=item.eligible_peers,
                callout=item.callout,
                details=item.details,
            )
            for item in anomalies
        ]

    identities = {metric.engineer for metric in metrics if metric.engineer != "(unknown)"}
    attributed = sum(1 for metric in metrics if metric.engineer != "(unknown)")
    confidence: dict[str, int] = {}
    for metric in metrics:
        confidence[metric.identity_confidence] = (
            confidence.get(metric.identity_confidence, 0) + 1
        )
    filtered_label = _clean_label(requested_team, "") if requested_team else ""
    if anonymize and filtered_label:
        filtered_label = f"anonymized {selector_kind} scope"
    return OrgReport(
        generated_at=(
            generated_at
            if generated_at is not None
            else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        month=month,
        source_mode=source_mode,
        organization_boundary=(
            "one authenticated collector instance and key"
            if source_mode == "collector-instance"
            else "one shared trace directory and its workspace stores"
        ),
        pricing_model=model,
        pricing_snapshot_date=PRICING_SNAPSHOT_DATE,
        cost_estimation_method="offline heuristic token estimate with bundled model pricing",
        team_filter=filtered_label,
        anonymized=anonymize,
        session_count=len(metrics),
        cost_estimated_sessions=len(costs),
        cost_estimate_coverage_percent=(len(costs) / len(metrics) * 100 if metrics else 0.0),
        total_estimated_spend_usd=total_cost,
        avg_estimated_cost_usd=average,
        median_estimated_cost_usd=median,
        active_attributed_identities=len(identities),
        identity_attributed_sessions=attributed,
        identity_coverage_percent=(attributed / len(metrics) * 100 if metrics else 0.0),
        identity_confidence=dict(sorted(confidence.items())),
        teams=tuple(team_rows),
        task_types=tuple(task_rows),
        anomalies=tuple(anomalies),
        snapshot_limitation=(
            "Collector metadata and event streams use bounded N+1 reads and are not a transactionally consistent snapshot."
            if source_mode == "collector-instance"
            else "The shared directory may change while it is read and is not a transactionally consistent snapshot."
        ),
    )


def org_report_payload(report: OrgReport) -> dict:
    payload = asdict(report)
    payload["schema"] = "agent-strace-org-report/v1"
    payload["spend_is_estimate"] = True
    payload["task_classification_is_heuristic"] = True
    payload["anomaly_detection_is_structural"] = True
    payload["spend_scope"] = "cost-estimated sessions only"
    payload["average_cost_denominator"] = "cost_estimated_sessions"
    return payload


def format_org_report_json(report: OrgReport) -> str:
    return json.dumps(
        org_report_payload(report), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def format_org_report_text(report: OrgReport) -> str:
    title = datetime.strptime(report.month, "%Y-%m").strftime("%B %Y")
    lines = [
        f"{title} - Engineering AI Agent Report",
        "=" * 64,
        f"Scope: {report.organization_boundary}",
        f"Covered-session estimated spend ({report.pricing_model}): ${report.total_estimated_spend_usd:.2f}",
        f"Sessions: {report.session_count}",
        f"Average estimate/cost-estimated session: ${report.avg_estimated_cost_usd:.2f}",
        f"Median estimate/cost-estimated session: ${report.median_estimated_cost_usd:.2f}",
        f"Cost estimate coverage: {report.cost_estimate_coverage_percent:.1f}% "
        f"({report.cost_estimated_sessions}/{report.session_count} sessions)",
        f"Active attributed identities: {report.active_attributed_identities}",
        f"Identity attribution coverage: {report.identity_coverage_percent:.1f}%",
    ]
    if report.team_filter:
        lines.append(f"Team scope: {report.team_filter}")
    lines.extend(["", "Team breakdown", "-" * 64])
    if report.teams:
        lines.append(f"{'Team':<24} {'Sessions':>8} {'Covered est.':>12} {'Avg covered':>11} {'Coverage':>9}")
        for row in report.teams:
            lines.append(
                f"{row.team[:24]:<24} {row.sessions:>8} "
                f"${row.estimated_spend_usd:>11.2f} ${row.avg_estimated_cost_usd:>10.2f} "
                f"{row.cost_estimate_coverage_percent:>8.1f}%"
            )
    else:
        lines.append("No sessions matched this scope.")

    lines.extend(["", "Heuristic task type breakdown", "-" * 64])
    if report.task_types:
        lines.append(f"{'Task type':<24} {'Sessions':>8} {'Avg covered':>11} {'Coverage':>9}")
        for row in report.task_types:
            lines.append(
                f"{row.task_type[:24]:<24} {row.sessions:>8} "
                f"${row.avg_estimated_cost_usd:>10.2f} {row.cost_estimate_coverage_percent:>8.1f}%"
            )
    else:
        lines.append("No task types to report.")

    lines.extend(["", "Structural anomaly callouts", "-" * 64])
    if report.anomalies:
        for anomaly in report.anomalies:
            lines.append(
                f"- {anomaly.subject}: {anomaly.callout} "
                f"({anomaly.sessions} sessions)"
            )
            lines.extend(f"  {detail}" for detail in anomaly.details)
    else:
        lines.append(
            f"No cost outliers met the {MIN_ANOMALY_SESSIONS}-session, "
            f"{MIN_ANOMALY_PEERS}-peer, and {MIN_ANOMALY_COVERAGE:.0%}-coverage floors."
        )
    lines.extend([
        "",
        f"Method: {report.cost_estimation_method}; pricing snapshot {report.pricing_snapshot_date}.",
        "Cost, task classification, and anomaly callouts are estimates or heuristics.",
        f"Snapshot limitation: {report.snapshot_limitation}",
    ])
    return "\n".join(lines) + "\n"


def format_org_report_html(report: OrgReport) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    title = datetime.strptime(report.month, "%Y-%m").strftime("%B %Y")
    team_rows = "".join(
        "<tr>"
        f"<td>{esc(row.team)}</td><td>{row.sessions}</td>"
        f"<td>${row.estimated_spend_usd:.2f}</td>"
        f"<td>${row.avg_estimated_cost_usd:.2f}</td>"
        f"<td>{row.cost_estimate_coverage_percent:.1f}%</td>"
        f"<td>{row.active_attributed_identities}</td>"
        "</tr>"
        for row in report.teams
    ) or '<tr><td colspan="6">No sessions matched this scope.</td></tr>'
    task_rows = "".join(
        "<tr>"
        f"<td>{esc(row.task_type)}</td><td>{row.sessions}</td>"
        f"<td>${row.avg_estimated_cost_usd:.2f}</td>"
        f"<td>{row.cost_estimate_coverage_percent:.1f}%</td>"
        "</tr>"
        for row in report.task_types
    ) or '<tr><td colspan="4">No task types to report.</td></tr>'
    anomaly_rows = "".join(
        "<li>"
        f"<strong>{esc(row.subject)}</strong>: {esc(row.callout)} "
        f"<span>({row.sessions} sessions)</span>"
        + "".join(f"<div>{esc(detail)}</div>" for detail in row.details)
        + "</li>"
        for row in report.anomalies
    ) or (
        f"<li>No cost outliers met the {MIN_ANOMALY_SESSIONS}-session, "
        f"{MIN_ANOMALY_PEERS}-peer, and {MIN_ANOMALY_COVERAGE:.0%}-coverage floors.</li>"
    )
    team_scope = (
        f"<p><strong>Team scope:</strong> {esc(report.team_filter)}</p>"
        if report.team_filter else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} - Engineering AI Agent Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172033;background:#fff}}
h1,h2{{color:#111827}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}}
.card{{border:1px solid #dbe2ea;border-radius:10px;padding:1rem}}.value{{font-size:1.55rem;font-weight:700}}
table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}th,td{{border-bottom:1px solid #dbe2ea;padding:.65rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}.note{{background:#f3f6fa;border-radius:8px;padding:1rem}}li{{margin:.7rem 0}}
</style></head><body>
<h1>{esc(title)} - Engineering AI Agent Report</h1>
<p><strong>Scope:</strong> {esc(report.organization_boundary)}</p>{team_scope}
<div class="cards">
<div class="card"><div>Covered-session estimated spend</div><div class="value">${report.total_estimated_spend_usd:.2f}</div></div>
<div class="card"><div>Sessions</div><div class="value">{report.session_count}</div></div>
<div class="card"><div>Average per cost-estimated session</div><div class="value">${report.avg_estimated_cost_usd:.2f}</div></div>
<div class="card"><div>Active attributed identities</div><div class="value">{report.active_attributed_identities}</div></div>
</div>
<h2>Team breakdown</h2><table><thead><tr><th>Team</th><th>Sessions</th><th>Covered-session estimate</th><th>Average per covered session</th><th>Coverage</th><th>Attributed identities</th></tr></thead><tbody>{team_rows}</tbody></table>
<h2>Heuristic task type breakdown</h2><table><thead><tr><th>Task type</th><th>Sessions</th><th>Average per covered session</th><th>Coverage</th></tr></thead><tbody>{task_rows}</tbody></table>
<h2>Structural anomaly callouts</h2><ul>{anomaly_rows}</ul>
<p class="note">Cost, task classification, and anomaly callouts are estimates or heuristics. Cost estimate coverage is {report.cost_estimate_coverage_percent:.1f}%; identity attribution coverage is {report.identity_coverage_percent:.1f}%. {esc(report.snapshot_limitation)}</p>
</body></html>"""


def _load_auth_key(auth_key_file: str) -> str:
    path_value = str(auth_key_file or "").strip()
    if path_value:
        path = Path(path_value)
        if not path.is_file():
            raise CollectorAuthenticationError("collector auth key file was not found")
        if path.stat().st_size > MAX_AUTH_KEY_FILE_BYTES:
            raise CollectorAuthenticationError("collector auth key file is too large")
        try:
            value = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise CollectorAuthenticationError(
                "collector auth key file could not be read"
            ) from exc
    else:
        value = os.environ.get("AGENT_STRACE_AUTH_KEY", "").strip()
    if not value or any(char in value for char in "\r\n"):
        raise CollectorAuthenticationError(
            "set AGENT_STRACE_AUTH_KEY or provide --auth-key-file"
        )
    return value


def _write_atomic(path_value: str, content: str) -> None:
    path = Path(path_value)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("refusing to replace a symlinked report output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as output:
            temporary = output.name
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = ""
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The report itself was already atomically replaced; directory
            # fsync is a best-effort durability improvement on supported OSes.
            pass
    finally:
        if temporary:
            try:
                Path(temporary).unlink()
            except OSError:
                pass


def cmd_org_report(args: argparse.Namespace) -> int:
    endpoint = str(getattr(args, "endpoint", "") or "").strip()
    month = str(getattr(args, "month", "") or "").strip()
    if not month:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
    try:
        if endpoint:
            key = _load_auth_key(getattr(args, "auth_key_file", "") or "")
            client = CollectorClient(
                endpoint,
                key,
                allow_insecure_http=bool(
                    getattr(args, "allow_insecure_http", False)
                ),
                ca_file=str(getattr(args, "ca_file", "") or ""),
            )
            stores = [CollectorTraceStore.load(client)]
            source_mode = "collector-instance"
        else:
            if (
                getattr(args, "auth_key_file", "")
                or getattr(args, "ca_file", "")
                or getattr(args, "allow_insecure_http", False)
            ):
                raise ValueError("collector authentication/TLS flags require --endpoint")
            stores = enumerate_trace_stores(args.trace_dir)
            source_mode = "shared-directory"
        report = build_org_report(
            stores,
            month,
            team=str(getattr(args, "team", "") or ""),
            anonymize=bool(getattr(args, "anonymize", False)),
            model=str(getattr(args, "model", DEFAULT_MODEL) or DEFAULT_MODEL),
            source_mode=source_mode,
        )
        output_format = str(getattr(args, "format", "text") or "text")
        if output_format == "json":
            rendered = format_org_report_json(report)
        elif output_format == "html":
            rendered = format_org_report_html(report)
        else:
            rendered = format_org_report_text(report)
        destination = str(getattr(args, "output", "") or "")
        if destination:
            _write_atomic(destination, rendered)
        else:
            sys.stdout.write(rendered)
        return 0
    except (
        CollectorClientError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        sys.stderr.write(f"org-report failed: {exc}\n")
        return 1
