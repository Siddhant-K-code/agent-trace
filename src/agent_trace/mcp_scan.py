"""Runtime MCP tool poisoning scanner.

Scans recorded sessions for suspicious MCP tool descriptions, cross-session
description drift, and high-risk tool-call sequences. The scanner is local,
deterministic, and stdlib-only.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .models import EventType, TraceEvent
from .store import TraceStore


DEFAULT_PATTERN_FILE = Path.home() / ".agent-strace" / "mcp-patterns.txt"

DEFAULT_INJECTION_PATTERNS: list[tuple[str, str]] = [
    ("system-prefix", r"\bSYSTEM\s*:"),
    ("ignore-instructions", r"ignore\s+(all\s+)?(previous|prior)\s+instructions"),
    ("hidden-tag", r"</?HIDDEN\b[^>]*>"),
    ("developer-override", r"\b(developer|system)\s+message\s*[:=]"),
    ("exfiltration", r"\b(exfiltrate|send|upload|post)\b.{0,80}\b(secret|token|key|credential|password)\b"),
    ("credential-read", r"(\.ssh/id_rsa|\.env|/etc/passwd|aws_secret_access_key)"),
]

_CREDENTIAL_PATH_PATTERNS = [
    "/etc/passwd",
    "/etc/shadow",
    "*/.ssh/*",
    "*id_rsa*",
    "*id_ed25519*",
    "*.pem",
    "*.key",
    ".env",
    "*.env",
]

_ARCHIVE_WORDS = ("tar ", "zip ", "gzip ", "7z ", "rar ")
_ENV_DUMP_WORDS = ("env", "printenv", "set")
_HTTP_TOOL_HINTS = ("http", "curl", "wget", "fetch", "request", "webfetch")
_WRITE_TOOLS = {"write", "edit", "create", "str_replace", "multiedit", "notebook_edit"}


@dataclass
class ToolDescription:
    session_id: str
    tool_name: str
    description: str
    event_index: int

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.description.encode("utf-8")).hexdigest()


@dataclass
class McpScanFinding:
    kind: str
    severity: str
    session_id: str
    message: str
    tool_name: str = ""
    event_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class McpScanReport:
    session_ids: list[str]
    findings: list[McpScanFinding] = field(default_factory=list)

    @property
    def high(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    @property
    def medium(self) -> int:
        return sum(1 for f in self.findings if f.severity == "medium")

    @property
    def low(self) -> int:
        return sum(1 for f in self.findings if f.severity == "low")


def _normalise_tool_name(value: str) -> str:
    return value.strip().lower()


def _iter_tool_dicts(raw: Any) -> list[dict[str, Any]]:
    """Return tool dicts from common SESSION_START tool-list shapes."""
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        tools = []
        for name, value in raw.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("name", name)
                tools.append(item)
            elif isinstance(value, str):
                tools.append({"name": name, "description": value})
        return tools
    return []


def extract_tool_descriptions(events: list[TraceEvent], session_id: str = "") -> list[ToolDescription]:
    """Extract runtime tool descriptions from session_start events."""
    descriptions: list[ToolDescription] = []
    for idx, event in enumerate(events, start=1):
        if event.event_type != EventType.SESSION_START:
            continue
        data = event.data or {}
        raw_sets = [
            data.get("tools_available"),
            data.get("tools"),
            data.get("mcp_tools"),
        ]
        for raw in raw_sets:
            for tool in _iter_tool_dicts(raw):
                name = str(
                    tool.get("name")
                    or tool.get("tool_name")
                    or tool.get("id")
                    or ""
                ).strip()
                desc = str(
                    tool.get("description")
                    or tool.get("desc")
                    or tool.get("summary")
                    or ""
                ).strip()
                if name and desc:
                    descriptions.append(ToolDescription(
                        session_id=session_id or event.session_id,
                        tool_name=name,
                        description=desc,
                        event_index=idx,
                    ))
    return descriptions


def load_injection_patterns(path: Path = DEFAULT_PATTERN_FILE) -> list[tuple[str, re.Pattern[str]]]:
    """Load built-in and user-supplied regex patterns."""
    patterns = list(DEFAULT_INJECTION_PATTERNS)
    if path.exists():
        for i, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append((f"user-pattern-{i}", line))

    compiled: list[tuple[str, re.Pattern[str]]] = []
    for name, pattern in patterns:
        try:
            compiled.append((name, re.compile(pattern, re.IGNORECASE | re.DOTALL)))
        except re.error:
            continue
    return compiled


def scan_description_patterns(
    descriptions: list[ToolDescription],
    patterns: list[tuple[str, re.Pattern[str]]] | None = None,
) -> list[McpScanFinding]:
    patterns = patterns or load_injection_patterns()
    findings: list[McpScanFinding] = []
    for desc in descriptions:
        for name, pattern in patterns:
            match = pattern.search(desc.description)
            if not match:
                continue
            findings.append(McpScanFinding(
                kind="description-pattern",
                severity="high",
                session_id=desc.session_id,
                tool_name=desc.tool_name,
                event_index=desc.event_index,
                message=f"{desc.tool_name}: suspicious tool description pattern {name}",
                details={
                    "pattern": name,
                    "hash": desc.digest,
                    "match": match.group(0)[:120],
                },
            ))
    return findings


def _previous_descriptions(
    store: TraceStore,
    current_session_id: str,
) -> dict[str, ToolDescription]:
    current_meta = store.load_meta(current_session_id)
    previous: dict[str, ToolDescription] = {}
    sessions = [
        meta for meta in store.list_sessions()
        if meta.session_id != current_session_id and meta.started_at < current_meta.started_at
    ]
    for meta in sorted(sessions, key=lambda m: m.started_at):
        try:
            events = store.load_events(meta.session_id)
        except Exception:
            continue
        for desc in extract_tool_descriptions(events, meta.session_id):
            previous[_normalise_tool_name(desc.tool_name)] = desc
    return previous


def scan_description_drift(
    store: TraceStore,
    session_id: str,
    descriptions: list[ToolDescription],
) -> list[McpScanFinding]:
    previous = _previous_descriptions(store, session_id)
    findings: list[McpScanFinding] = []
    for desc in descriptions:
        key = _normalise_tool_name(desc.tool_name)
        prev = previous.get(key)
        if not prev or prev.digest == desc.digest:
            continue
        findings.append(McpScanFinding(
            kind="description-drift",
            severity="medium",
            session_id=session_id,
            tool_name=desc.tool_name,
            event_index=desc.event_index,
            message=f"{desc.tool_name}: tool description changed since session {prev.session_id}",
            details={
                "previous_session": prev.session_id,
                "previous_hash": prev.digest,
                "current_hash": desc.digest,
            },
        ))
    return findings


def _tool_name(event: TraceEvent) -> str:
    return str(event.data.get("tool_name") or event.data.get("name") or "").lower()


def _arguments(event: TraceEvent) -> dict[str, Any]:
    args = event.data.get("arguments") or event.data.get("args") or {}
    return args if isinstance(args, dict) else {}


def _target_text(event: TraceEvent) -> str:
    args = _arguments(event)
    values = [
        args.get("file_path"),
        args.get("path"),
        args.get("uri"),
        args.get("url"),
        args.get("command"),
        event.data.get("uri"),
        event.data.get("path"),
        event.data.get("url"),
    ]
    return " ".join(str(v) for v in values if v is not None).strip()


def _is_credential_read(event: TraceEvent) -> bool:
    if event.event_type not in {EventType.TOOL_CALL, EventType.FILE_READ}:
        return False
    text = _target_text(event).lower()
    if not text:
        return False
    return any(fnmatch.fnmatch(text, pat.lower()) or pat.lower() in text for pat in _CREDENTIAL_PATH_PATTERNS)


def _is_env_dump(event: TraceEvent) -> bool:
    if event.event_type != EventType.TOOL_CALL:
        return False
    if _tool_name(event) != "bash":
        return False
    command = str(_arguments(event).get("command", "")).strip().lower()
    return (
        command in _ENV_DUMP_WORDS
        or command.startswith(("env ", "env|", "printenv ", "printenv|"))
        or any(part in command for part in (" printenv", " env"))
    )


def _is_external_http(event: TraceEvent) -> bool:
    text = _target_text(event).lower()
    tool = _tool_name(event)
    if not text and not tool:
        return False
    has_http_tool = any(hint in tool for hint in _HTTP_TOOL_HINTS)
    has_external_url = bool(re.search(r"https?://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)", text))
    return has_external_url or (has_http_tool and "localhost" not in text and "127.0.0.1" not in text)


def _is_read_call(event: TraceEvent) -> bool:
    if event.event_type == EventType.FILE_READ:
        return True
    return event.event_type == EventType.TOOL_CALL and "read" in _tool_name(event)


def _is_archive_command(event: TraceEvent) -> bool:
    if event.event_type != EventType.TOOL_CALL or _tool_name(event) != "bash":
        return False
    command = str(_arguments(event).get("command", "")).lower()
    return any(word in command or command.startswith(word.strip()) for word in _ARCHIVE_WORDS)


def _is_shadow_write(event: TraceEvent, project_root: str = "") -> bool:
    if event.event_type not in {EventType.TOOL_CALL, EventType.FILE_WRITE}:
        return False
    if event.event_type == EventType.TOOL_CALL and _tool_name(event) not in _WRITE_TOOLS:
        return False
    text = _target_text(event)
    if not text:
        return False
    path = Path(text.split()[0]).expanduser()
    if not path.is_absolute():
        return False
    root = Path(project_root or ".").resolve()
    try:
        path.resolve().relative_to(root)
        return False
    except ValueError:
        return True


def scan_behavioural_anomalies(
    events: list[TraceEvent],
    session_id: str,
    project_root: str = "",
) -> list[McpScanFinding]:
    findings: list[McpScanFinding] = []
    credential_reads: list[tuple[int, TraceEvent]] = []
    env_dumps: list[tuple[int, TraceEvent]] = []
    read_window: list[tuple[int, TraceEvent]] = []

    for idx, event in enumerate(events, start=1):
        if _is_credential_read(event):
            credential_reads.append((idx, event))
        if _is_env_dump(event):
            env_dumps.append((idx, event))
        if _is_read_call(event):
            read_window.append((idx, event))
            read_window = read_window[-20:]

        if _is_external_http(event):
            recent_cred = [(i, e) for i, e in credential_reads if idx - i <= 5]
            if recent_cred:
                first_idx, first_event = recent_cred[-1]
                findings.append(McpScanFinding(
                    kind="credential-read-then-exfil",
                    severity="high",
                    session_id=session_id,
                    event_index=idx,
                    message="credential path read followed by external HTTP call",
                    details={
                        "read_event": first_idx,
                        "read_target": _target_text(first_event),
                        "http_target": _target_text(event),
                    },
                ))

            recent_env = [(i, e) for i, e in env_dumps if idx - i <= 5]
            if recent_env:
                first_idx, first_event = recent_env[-1]
                findings.append(McpScanFinding(
                    kind="env-dump-then-exfil",
                    severity="high",
                    session_id=session_id,
                    event_index=idx,
                    message="environment dump followed by external HTTP call",
                    details={
                        "env_event": first_idx,
                        "env_command": _target_text(first_event),
                        "http_target": _target_text(event),
                    },
                ))

        if _is_archive_command(event):
            recent_reads = [(i, e) for i, e in read_window if idx - i <= 20]
            if len(recent_reads) >= 10:
                findings.append(McpScanFinding(
                    kind="mass-read-then-compress",
                    severity="medium",
                    session_id=session_id,
                    event_index=idx,
                    message="many file reads followed by archive/compression command",
                    details={"read_count": len(recent_reads), "command": _target_text(event)},
                ))

        if _is_shadow_write(event, project_root=project_root):
            findings.append(McpScanFinding(
                kind="shadow-write",
                severity="medium",
                session_id=session_id,
                event_index=idx,
                message="write operation targets a path outside the project root",
                details={"target": _target_text(event), "project_root": str(Path(project_root or '.').resolve())},
            ))

    return findings


def scan_live_event(
    events: list[TraceEvent],
    event: TraceEvent,
    session_id: str,
    patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    project_root: str = "",
) -> list[McpScanFinding]:
    """Scan a newly observed event using recent in-session history."""
    findings: list[McpScanFinding] = []

    if event.event_type == EventType.SESSION_START:
        descriptions = extract_tool_descriptions([event], session_id)
        findings.extend(scan_description_patterns(descriptions, patterns=patterns))

    behavioural = scan_behavioural_anomalies(events, session_id, project_root=project_root)
    if behavioural:
        current_index = len(events)
        findings.extend([
            finding for finding in behavioural
            if finding.event_index == current_index
        ])
    return findings


def scan_session(
    store: TraceStore,
    session_id: str,
    patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    project_root: str = "",
) -> McpScanReport:
    events = store.load_events(session_id)
    descriptions = extract_tool_descriptions(events, session_id)
    findings: list[McpScanFinding] = []
    findings.extend(scan_description_patterns(descriptions, patterns=patterns))
    findings.extend(scan_description_drift(store, session_id, descriptions))
    findings.extend(scan_behavioural_anomalies(events, session_id, project_root=project_root))
    return McpScanReport(session_ids=[session_id], findings=findings)


def scan_store(
    store: TraceStore,
    since_seconds: float | None = None,
    patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    project_root: str = "",
) -> McpScanReport:
    cutoff = time.time() - since_seconds if since_seconds else None
    findings: list[McpScanFinding] = []
    scanned: list[str] = []
    for meta in reversed(store.list_sessions()):
        if cutoff and meta.started_at < cutoff:
            continue
        scanned.append(meta.session_id)
        try:
            report = scan_session(store, meta.session_id, patterns=patterns, project_root=project_root)
        except Exception:
            continue
        findings.extend(report.findings)
    return McpScanReport(session_ids=scanned, findings=findings)


def finding_to_dict(finding: McpScanFinding) -> dict[str, Any]:
    data = {
        "kind": finding.kind,
        "severity": finding.severity,
        "session_id": finding.session_id,
        "message": finding.message,
        "details": finding.details,
    }
    if finding.tool_name:
        data["tool_name"] = finding.tool_name
    if finding.event_index is not None:
        data["event_index"] = finding.event_index
    return data


def format_report(report: McpScanReport) -> str:
    lines = [
        "MCP tool scan",
        f"Sessions scanned: {len(report.session_ids)}",
        f"Findings: {len(report.findings)} high={report.high} medium={report.medium} low={report.low}",
    ]
    if not report.findings:
        lines.append("No MCP poisoning indicators detected.")
        return "\n".join(lines)

    for finding in report.findings:
        location = f"event {finding.event_index}" if finding.event_index else "session"
        tool = f" {finding.tool_name}" if finding.tool_name else ""
        lines.append(
            f"[{finding.severity.upper()}] {finding.session_id[:12]} {location}{tool}: {finding.message}"
        )
        for key, value in finding.details.items():
            if key.endswith("hash") and isinstance(value, str):
                value = value[:12]
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _parse_since(value: str) -> float | None:
    if not value:
        return 7 * 86400
    value = value.strip().lower()
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, time.time() - dt.timestamp())
    except ValueError:
        pass
    if value.endswith("d"):
        return float(value[:-1]) * 86400
    if value.endswith("h"):
        return float(value[:-1]) * 3600
    try:
        return float(value) * 86400
    except ValueError:
        return 7 * 86400


def _resolve_session(store: TraceStore, session_id: str) -> str | None:
    if store.session_exists(session_id):
        return session_id
    return store.find_session(session_id)


def _watch_session(
    store: TraceStore,
    session_id: str,
    out: TextIO,
    patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    project_root: str = "",
) -> int:
    events_file = store._session_dir(session_id) / "events.ndjson"
    if not events_file.exists():
        out.write(f"events file not found: {events_file}\n")
        return 1
    out.write(f"Watching session {session_id[:12]} for MCP poisoning indicators...\n")
    out.flush()

    seen_findings: set[str] = set()
    history = store.load_events(session_id)
    initial = scan_session(store, session_id, patterns=patterns, project_root=project_root)
    for finding in initial.findings:
        key = _finding_key(finding)
        seen_findings.add(key)
        out.write(format_report(McpScanReport([session_id], [finding])) + "\n")
    out.flush()

    seen = 0
    with open(events_file, "r", encoding="utf-8") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            try:
                event = TraceEvent.from_json(line)
            except Exception:
                continue
            seen += 1
            history.append(event)
            history = history[-80:]
            findings = scan_live_event(
                history,
                event,
                session_id,
                patterns=patterns,
                project_root=project_root,
            )
            for finding in findings:
                key = _finding_key(finding)
                if key in seen_findings:
                    continue
                seen_findings.add(key)
                out.write(format_report(McpScanReport([session_id], [finding])) + "\n")
                out.flush()
            if event.event_type == EventType.SESSION_END:
                out.write(f"Session ended after {seen} watched events.\n")
                return 0


def cmd_mcp_scan(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    patterns = load_injection_patterns(Path(args.patterns).expanduser()) if args.patterns else load_injection_patterns()
    project_root = getattr(args, "project_root", "") or "."

    if getattr(args, "watch", False):
        session_id = args.session or store.get_latest_session_id()
        if not session_id:
            sys.stderr.write("No sessions found.\n")
            return 1
        resolved = _resolve_session(store, session_id)
        if not resolved:
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1
        return _watch_session(
            store,
            resolved,
            sys.stdout,
            patterns=patterns,
            project_root=project_root,
        )

    if args.session:
        session_id = _resolve_session(store, args.session)
        if not session_id:
            sys.stderr.write(f"Session not found: {args.session}\n")
            return 1
        report = scan_session(store, session_id, patterns=patterns, project_root=project_root)
    else:
        report = scan_store(
            store,
            since_seconds=_parse_since(getattr(args, "since", "")),
            patterns=patterns,
            project_root=project_root,
        )

    if getattr(args, "format", "text") == "json":
        payload = {
            "sessions": report.session_ids,
            "summary": {
                "findings": len(report.findings),
                "high": report.high,
                "medium": report.medium,
                "low": report.low,
            },
            "findings": [finding_to_dict(f) for f in report.findings],
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        sys.stdout.write(format_report(report) + "\n")

    return 1 if report.high else 0


def _finding_key(finding: McpScanFinding) -> str:
    details = json.dumps(finding.details, sort_keys=True, default=str)
    return (
        f"{finding.kind}:{finding.severity}:{finding.session_id}:"
        f"{finding.tool_name}:{finding.event_index}:{details}"
    )
