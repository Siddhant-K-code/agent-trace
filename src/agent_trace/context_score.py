"""Score AGENTS.md / CLAUDE.md quality from session outcomes."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .config_watch import ConfigSnapshot, _load_snapshots
from .cost import estimate_cost
from .freshness import _parse_scope_from_agents_md
from .lint import lint_session
from .models import EventType, SessionMeta, TraceEvent
from .postmortem import _detect_agents_md_violations, _load_agents_md
from .store import TraceStore


_CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md", "GEMINI.md")


@dataclass
class SessionContextMetrics:
    session_id: str
    version_hash: str
    version_label: str
    started_at: float
    cost: float
    tool_calls: int
    lint_findings: int
    redundant_reads: int
    context_saturation: int
    scope_violations: int
    instruction_violations: int


@dataclass
class VersionScore:
    version_hash: str
    label: str
    first_seen: float
    last_seen: float
    sessions: list[SessionContextMetrics] = field(default_factory=list)

    @property
    def session_count(self) -> int:
        return len(self.sessions)

    @property
    def avg_cost(self) -> float:
        return _avg([s.cost for s in self.sessions])

    @property
    def avg_tool_calls(self) -> float:
        return _avg([s.tool_calls for s in self.sessions])

    @property
    def lint_per_session(self) -> float:
        return _avg([s.lint_findings for s in self.sessions])

    @property
    def scope_violations_per_session(self) -> float:
        return _avg([s.scope_violations for s in self.sessions])

    @property
    def redundant_reads_per_session(self) -> float:
        return _avg([s.redundant_reads for s in self.sessions])

    @property
    def context_saturation_rate(self) -> float:
        if not self.sessions:
            return 0.0
        return sum(1 for s in self.sessions if s.context_saturation > 0) / len(self.sessions)

    @property
    def instruction_violation_rate(self) -> float:
        if not self.sessions:
            return 0.0
        return sum(1 for s in self.sessions if s.instruction_violations > 0) / len(self.sessions)


@dataclass
class DimensionScore:
    name: str
    current_value: float
    baseline_value: float | None
    change_pct: float | None
    score: float | None
    sufficient_data: bool


@dataclass
class ContextScoreReport:
    context_file: str
    window_start: float
    window_end: float
    min_sessions: int
    current: VersionScore | None
    baseline: VersionScore | None
    versions: list[VersionScore]
    dimensions: list[DimensionScore]
    suggestions: list[str]
    insufficient_data: bool
    no_history: bool

    @property
    def overall_score(self) -> float | None:
        scores = [d.score for d in self.dimensions if d.score is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    @property
    def grade(self) -> str:
        score = self.overall_score
        if score is None:
            return "insufficient data"
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _choose_context_file(root: Path, explicit: str = "") -> str:
    if explicit:
        return explicit
    for name in _CONTEXT_FILES:
        if (root / name).exists():
            return name
    return "AGENTS.md"


def _snapshot_file_hash(snapshot: ConfigSnapshot, rel_path: str) -> str:
    for file_snapshot in snapshot.files:
        if file_snapshot.path == rel_path:
            return file_snapshot.sha256 if file_snapshot.exists else ""
    return ""


def _session_version(
    meta: SessionMeta,
    snapshots: list[ConfigSnapshot],
    context_file: str,
    current_hash: str,
) -> tuple[str, str]:
    latest: ConfigSnapshot | None = None
    for snapshot in snapshots:
        if snapshot.timestamp <= meta.started_at:
            latest = snapshot
        else:
            break
    if latest is None:
        label = f"current:{current_hash[:8]}" if current_hash else "no-context-file"
        return current_hash, label
    version_hash = _snapshot_file_hash(latest, context_file)
    label = latest.label or latest.snapshot_id[:8]
    return version_hash, label


def _event_path(event: TraceEvent) -> str:
    if event.event_type in (EventType.FILE_READ, EventType.FILE_WRITE):
        return str(event.data.get("path") or event.data.get("file_path") or event.data.get("uri") or "")
    if event.event_type != EventType.TOOL_CALL:
        return ""
    args = event.data.get("arguments", {}) or {}
    if not isinstance(args, dict):
        return ""
    return str(args.get("file_path") or args.get("path") or "")


def _tool_calls(events: list[TraceEvent]) -> int:
    return sum(1 for event in events if event.event_type == EventType.TOOL_CALL)


def _scope_violations(events: list[TraceEvent], scope_globs: list[str]) -> int:
    if not scope_globs:
        return 0
    count = 0
    for event in events:
        path = _event_path(event)
        if not path:
            continue
        if not any(fnmatch.fnmatch(path, pattern) for pattern in scope_globs):
            count += 1
    return count


def _session_cost(store: TraceStore, session_id: str) -> float:
    try:
        return estimate_cost(store, session_id).total_cost
    except Exception:
        return 0.0


def _session_lint(store: TraceStore, session_id: str) -> tuple[int, int, int]:
    try:
        report = lint_session(store, session_id)
    except Exception:
        return 0, 0, 0
    redundant = sum(1 for finding in report.findings if finding.rule == "redundant-read")
    saturation = sum(1 for finding in report.findings if finding.rule == "context-saturation")
    return len(report.findings), redundant, saturation


def _session_metrics(
    store: TraceStore,
    meta: SessionMeta,
    version_hash: str,
    version_label: str,
    scope_globs: list[str],
    instruction_lines: list[str],
) -> SessionContextMetrics:
    try:
        events = store.load_events(meta.session_id)
    except Exception:
        events = []
    lint_count, redundant_reads, context_saturation = _session_lint(store, meta.session_id)
    return SessionContextMetrics(
        session_id=meta.session_id,
        version_hash=version_hash,
        version_label=version_label,
        started_at=meta.started_at,
        cost=_session_cost(store, meta.session_id),
        tool_calls=_tool_calls(events) or getattr(meta, "tool_calls", 0) or 0,
        lint_findings=lint_count,
        redundant_reads=redundant_reads,
        context_saturation=context_saturation,
        scope_violations=_scope_violations(events, scope_globs),
        instruction_violations=len(_detect_agents_md_violations(events, instruction_lines)),
    )


def _change_score(current: float, baseline: float | None) -> tuple[float | None, float | None]:
    if baseline is None:
        return None, None
    if baseline <= 0:
        change = 0.0 if current <= 0 else -1.0
    else:
        change = (baseline - current) / baseline
    score = max(0.0, min(100.0, 50.0 + change * 100.0))
    return change, score


def _dimension(
    name: str,
    current_value: float,
    baseline_value: float | None,
    sufficient_data: bool,
) -> DimensionScore:
    if not sufficient_data:
        return DimensionScore(name, current_value, baseline_value, None, None, False)
    change, score = _change_score(current_value, baseline_value)
    return DimensionScore(name, current_value, baseline_value, change, score, True)


def _suggestions(current: VersionScore | None, context_file: Path) -> list[str]:
    if current is None or not current.sessions:
        return []
    suggestions: list[str] = []
    token_estimate = len(context_file.read_text(errors="replace")) // 4 if context_file.exists() else 0
    if current.context_saturation_rate > 0.20 or token_estimate > 800:
        suggestions.append(
            f"{context_file.name} may be too long; consider splitting scope and conventions."
        )
    if current.instruction_violation_rate > 0.30:
        suggestions.append(
            "Some instructions appear to be ignored; rephrase them as explicit prohibitions."
        )
    if current.redundant_reads_per_session >= 1.0:
        suggestions.append(
            "High redundant-read rate; add specific file path exclusions or reading order guidance."
        )
    if current.scope_violations_per_session >= 1.0:
        suggestions.append(
            "Out-of-scope file accesses detected; add explicit path restrictions for the affected directories."
        )
    return suggestions


def build_context_score_report(
    store: TraceStore,
    workspace_root: Path,
    context_file: str = "",
    history_days: int = 30,
    compare: bool = False,
    min_sessions: int = 5,
) -> ContextScoreReport:
    selected_file = _choose_context_file(workspace_root, context_file)
    file_path = workspace_root / selected_file
    current_hash = _hash_file(file_path)
    now = _time.time()
    window_start = now - history_days * 86400
    snapshots = sorted(_load_snapshots(workspace_root), key=lambda snapshot: snapshot.timestamp)

    scope_globs = _parse_scope_from_agents_md(str(file_path))
    instruction_lines = _load_agents_md(file_path)
    versions: dict[str, VersionScore] = {}
    metas = [meta for meta in store.list_sessions() if meta.started_at >= window_start]
    for meta in metas:
        version_hash, label = _session_version(meta, snapshots, selected_file, current_hash)
        if version_hash not in versions:
            versions[version_hash] = VersionScore(
                version_hash=version_hash,
                label=label,
                first_seen=meta.started_at,
                last_seen=meta.started_at,
            )
        version = versions[version_hash]
        version.first_seen = min(version.first_seen, meta.started_at)
        version.last_seen = max(version.last_seen, meta.started_at)
        version.sessions.append(_session_metrics(
            store,
            meta,
            version_hash,
            label,
            scope_globs,
            instruction_lines,
        ))

    ordered = sorted(versions.values(), key=lambda version: version.last_seen)
    current = ordered[-1] if ordered else None
    baseline = ordered[-2] if compare and len(ordered) >= 2 else (ordered[0] if len(ordered) >= 2 else None)
    no_history = len(ordered) <= 1
    sufficient = bool(
        current
        and current.session_count >= min_sessions
        and (baseline is None or baseline.session_count >= min_sessions)
    )

    dimensions: list[DimensionScore] = []
    if current:
        dimensions = [
            _dimension("Cost efficiency", current.avg_cost, baseline.avg_cost if baseline else None, sufficient and baseline is not None),
            _dimension("Tool efficiency", current.avg_tool_calls, baseline.avg_tool_calls if baseline else None, sufficient and baseline is not None),
            _dimension("Lint violations", current.lint_per_session, baseline.lint_per_session if baseline else None, sufficient and baseline is not None),
            _dimension("Scope adherence", current.scope_violations_per_session, baseline.scope_violations_per_session if baseline else None, sufficient and baseline is not None),
        ]

    return ContextScoreReport(
        context_file=selected_file,
        window_start=window_start,
        window_end=now,
        min_sessions=min_sessions,
        current=current,
        baseline=baseline,
        versions=ordered,
        dimensions=dimensions,
        suggestions=_suggestions(current, file_path),
        insufficient_data=not sufficient,
        no_history=no_history,
    )


def _fmt_date(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _fmt_change(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.0f}%"


def format_context_score_text(report: ContextScoreReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    current = report.current
    w(f"Context file: {report.context_file}\n")
    if current is None:
        w("No sessions found in the selected history window.\n")
        return
    w(f"Sessions analysed: {sum(v.session_count for v in report.versions)} (last {_fmt_date(report.window_start)} to {_fmt_date(report.window_end)})\n")
    if report.no_history:
        w("No context version history found; showing current-version session stats only.\n")
    elif report.insufficient_data:
        w(f"Insufficient data: need at least {report.min_sessions} sessions per compared version.\n")
    w("\n")

    if report.baseline:
        w(f"{'Dimension':<20}  {'Score':>6}  {'vs baseline':>12}\n")
        w("-" * 44 + "\n")
        for dimension in report.dimensions:
            score = "n/a" if dimension.score is None else f"{dimension.score:.0f}"
            w(f"{dimension.name:<20}  {score:>6}  {_fmt_change(dimension.change_pct):>12}\n")
        w("-" * 44 + "\n")
        overall = "n/a" if report.overall_score is None else f"{report.overall_score:.0f}"
        w(f"Overall score: {overall} ({report.grade})\n\n")

    w("Current version stats:\n")
    w(f"  sessions: {current.session_count}\n")
    w(f"  cost/session: ${current.avg_cost:.4f}\n")
    w(f"  tool calls/session: {current.avg_tool_calls:.1f}\n")
    w(f"  lint findings/session: {current.lint_per_session:.1f}\n")
    w(f"  scope violations/session: {current.scope_violations_per_session:.1f}\n")

    if report.baseline:
        base = report.baseline
        w("\nBaseline version stats:\n")
        w(f"  sessions: {base.session_count}\n")
        w(f"  cost/session: ${base.avg_cost:.4f}\n")
        w(f"  tool calls/session: {base.avg_tool_calls:.1f}\n")
        w(f"  lint findings/session: {base.lint_per_session:.1f}\n")
        w(f"  scope violations/session: {base.scope_violations_per_session:.1f}\n")

    if report.suggestions:
        w("\nWhat to improve:\n")
        for suggestion in report.suggestions:
            w(f"  - {suggestion}\n")


def format_context_score_json(report: ContextScoreReport) -> str:
    def version_dict(version: VersionScore | None) -> dict | None:
        if version is None:
            return None
        return {
            "version_hash": version.version_hash,
            "label": version.label,
            "first_seen": version.first_seen,
            "last_seen": version.last_seen,
            "session_count": version.session_count,
            "avg_cost": version.avg_cost,
            "avg_tool_calls": version.avg_tool_calls,
            "lint_per_session": version.lint_per_session,
            "scope_violations_per_session": version.scope_violations_per_session,
            "redundant_reads_per_session": version.redundant_reads_per_session,
            "context_saturation_rate": version.context_saturation_rate,
            "instruction_violation_rate": version.instruction_violation_rate,
        }

    return json.dumps({
        "context_file": report.context_file,
        "window_start": report.window_start,
        "window_end": report.window_end,
        "min_sessions": report.min_sessions,
        "insufficient_data": report.insufficient_data,
        "no_history": report.no_history,
        "overall_score": report.overall_score,
        "grade": report.grade,
        "current": version_dict(report.current),
        "baseline": version_dict(report.baseline),
        "versions": [version_dict(version) for version in report.versions],
        "dimensions": [
            {
                "name": dimension.name,
                "current_value": dimension.current_value,
                "baseline_value": dimension.baseline_value,
                "change_pct": dimension.change_pct,
                "score": dimension.score,
                "sufficient_data": dimension.sufficient_data,
            }
            for dimension in report.dimensions
        ],
        "suggestions": report.suggestions,
    }, indent=2)


def cmd_context_score(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    workspace_root = Path(args.trace_dir).parent
    report = build_context_score_report(
        store,
        workspace_root=workspace_root,
        context_file=getattr(args, "file", "") or "",
        history_days=int(getattr(args, "history", 30)),
        compare=bool(getattr(args, "compare", False)),
        min_sessions=int(getattr(args, "min_sessions", 5)),
    )
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(format_context_score_json(report) + "\n")
    else:
        format_context_score_text(report)
    return 0
