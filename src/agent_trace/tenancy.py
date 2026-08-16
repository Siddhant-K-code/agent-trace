"""Multi-tenant trace reporting, export, and erasure.

Tenant IDs are additive metadata tags.  Local commands scope every operation
through :class:`TraceStore` before loading event data so one tenant's export or
cost report cannot accidentally include another tenant's sessions.
"""

from __future__ import annotations

import argparse
import base64
import calendar
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .cost import DEFAULT_MODEL, estimate_cost
from .store import TraceStore, _write_atomic, validate_stored_id, validate_tenant_id


@dataclass(frozen=True)
class TenantReportRow:
    tenant_id: str
    sessions: int
    cost_usd: float


@dataclass(frozen=True)
class TenantReport:
    month: str
    rows: list[TenantReportRow]

    @property
    def total_sessions(self) -> int:
        return sum(row.sessions for row in self.rows)

    @property
    def total_cost_usd(self) -> float:
        return sum(row.cost_usd for row in self.rows)


@dataclass(frozen=True)
class TenantDeletionResult:
    tenant_id: str
    deleted_sessions: int
    deleted_events: int
    failed_session_ids: tuple[str, ...]
    audit_path: Path
    audit_paths: tuple[Path, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _month_bounds(month: str) -> tuple[float, float]:
    try:
        parsed = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("month must use YYYY-MM format") from exc
    _, days = calendar.monthrange(parsed.year, parsed.month)
    end = parsed.replace(day=days, hour=23, minute=59, second=59, microsecond=999999)
    return parsed.timestamp(), end.timestamp()


def enumerate_trace_stores(base_dir: str | Path) -> list[TraceStore]:
    """Return the flat store plus every safe workspace store, without env bias."""
    root = TraceStore(base_dir, use_workspace_env=False)
    stores = [root]
    workspaces = root.storage_root / "workspaces"
    if workspaces.is_symlink():
        raise ValueError("refusing symlinked workspaces root")
    if workspaces.exists():
        if not workspaces.is_dir():
            raise ValueError("workspaces root is not a directory")
        for directory in sorted(workspaces.iterdir(), key=lambda item: item.name):
            if directory.is_symlink():
                raise ValueError(
                    f"refusing symlinked workspace entry: {directory.name}"
                )
            if not directory.is_dir():
                continue
            workspace_id = validate_stored_id(
                directory.name, "stored workspace directory"
            )
            store = TraceStore(
                base_dir,
                workspace_id=workspace_id,
                use_workspace_env=False,
                allow_legacy_workspace_id=True,
            )
            if store.base_dir.resolve() not in {item.base_dir.resolve() for item in stores}:
                stores.append(store)
    for store in stores:
        _recover_delete_journals(store)
    return stores


def _coerce_stores(value: TraceStore | list[TraceStore]) -> list[TraceStore]:
    stores = [value] if isinstance(value, TraceStore) else list(value)
    unique: list[TraceStore] = []
    seen: set[Path] = set()
    for store in stores:
        resolved = store.base_dir.resolve()
        if resolved not in seen:
            unique.append(store)
            seen.add(resolved)
    return unique


def build_tenant_report(
    store: TraceStore | list[TraceStore],
    month: str,
    model: str = DEFAULT_MODEL,
) -> TenantReport:
    """Aggregate session counts and estimated cost by tenant for one UTC month."""
    start, end = _month_bounds(month)
    grouped: dict[str, list[float | int]] = {}
    for scoped_store in _coerce_stores(store):
        for meta in scoped_store.list_sessions_strict():
            if not start <= meta.started_at <= end:
                continue
            values = grouped.setdefault(meta.tenant_id, [0, 0.0])
            values[0] = int(values[0]) + 1
            try:
                result = estimate_cost(scoped_store, meta.session_id, model=model)
                values[1] = float(values[1]) + result.total_cost
            except (OSError, ValueError, json.JSONDecodeError):
                continue

    rows = [
        TenantReportRow(
            tenant_id=tenant_id,
            sessions=int(values[0]),
            cost_usd=float(values[1]),
        )
        for tenant_id, values in grouped.items()
    ]
    rows.sort(key=lambda row: (-row.cost_usd, row.tenant_id or "~"))
    return TenantReport(month=month, rows=rows)


def format_tenant_report(report: TenantReport, out: TextIO = sys.stdout) -> None:
    """Write a human-readable tenant cost rollup."""
    title = datetime.strptime(report.month, "%Y-%m").strftime("%B %Y")
    labels = [row.tenant_id or "(untagged)" for row in report.rows]
    tenant_width = min(max([len("Tenant"), *(len(label) for label in labels)]), 48)
    divider = "─" * (tenant_width + 38)
    out.write(
        f"Tenant cost report — {title}\n"
        f"{divider}\n"
        f"{'Tenant':<{tenant_width}}  {'Sessions':>8}  {'Cost':>11}  {'% of total':>12}\n"
        f"{divider}\n"
    )
    total = report.total_cost_usd
    for row, label in zip(report.rows, labels):
        display = label if len(label) <= tenant_width else label[:tenant_width - 1] + "…"
        percent = row.cost_usd / total * 100 if total else 0.0
        out.write(
            f"{display:<{tenant_width}}  {row.sessions:>8}  "
            f"${row.cost_usd:>10.2f}  {percent:>11.1f}%\n"
        )
    out.write(f"{divider}\n")
    out.write(
        f"{'Total':<{tenant_width}}  {report.total_sessions:>8}  "
        f"${report.total_cost_usd:>10.2f}\n"
    )


def _encoded_file(path: Path, relative_to: Path) -> dict:
    data = path.read_bytes()
    try:
        content = data.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = base64.b64encode(data).decode("ascii")
        encoding = "base64"
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "encoding": encoding,
        "content": content,
    }


def _relevant_audit_records(store: TraceStore, tenant_hash: str) -> list[dict]:
    records: list[dict] = []
    for name in ("tenant-audit.ndjson", "tenant-deletions.ndjson"):
        path = store._store_file(name, "tenant audit")
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("tenant_id_sha256") == tenant_hash:
                records.append({"source": name, "record": record})
    return records


def tenant_export_payload(
    store: TraceStore | list[TraceStore], tenant_id: str,
) -> dict:
    """Return a complete, versioned subject-access export."""
    tenant_id = validate_tenant_id(tenant_id)
    sessions = []
    audit_records: list[dict] = []
    external_records: list[dict] = []
    tenant_hash = hashlib.sha256(tenant_id.encode()).hexdigest()
    for scoped_store in _coerce_stores(store):
        with scoped_store.write_lock():
            metas = scoped_store.list_sessions_strict(tenant_id=tenant_id)
            session_ids = {meta.session_id for meta in metas}
            for meta in metas:
                events = scoped_store.load_events(meta.session_id)
                session_dir = scoped_store._session_dir(meta.session_id)
                sidecars = []
                for path in sorted(session_dir.rglob("*")):
                    if path.is_symlink():
                        raise ValueError(f"refusing symlinked session sidecar: {path.name}")
                    if path.is_file() and path.name not in {"meta.json", "events.ndjson"}:
                        sidecars.append(_encoded_file(path, session_dir))
                checkpoint_path = scoped_store.checkpoint_path(meta.session_id)
                checkpoint = (
                    _encoded_file(checkpoint_path, scoped_store.base_dir)
                    if checkpoint_path.exists()
                    else None
                )
                sessions.append({
                    "scope": {
                        "workspace_id": scoped_store.workspace_id or None,
                    },
                    "session": json.loads(meta.to_json()),
                    "events": [json.loads(event.to_json()) for event in events],
                    "sidecars": sidecars,
                    "checkpoint": checkpoint,
                })
            for item in scoped_store.external_session_records(session_ids):
                item["scope"] = {
                    "workspace_id": scoped_store.workspace_id or None
                }
                external_records.append(item)
            for item in _relevant_audit_records(scoped_store, tenant_hash):
                item["scope"] = {"workspace_id": scoped_store.workspace_id or None}
                audit_records.append(item)
    return {
        "schema": "agent-strace-tenant-export/v1",
        "tenant_id": tenant_id,
        "exported_at": _utc_now(),
        "session_count": len(sessions),
        "sessions": sessions,
        "external_records": {
            "schema": "agent-strace-tenant-external-records/v1",
            "records": external_records,
        },
        "audit_records": audit_records,
    }


def _execute_delete_journal_locked(
    store: TraceStore, journal_path: Path, journal: dict,
) -> TenantDeletionResult:
    result = store._complete_delete_journal_locked(journal_path, journal)
    audit_path = result["audit_path"]
    return TenantDeletionResult(
        tenant_id=result["tenant_id"],
        deleted_sessions=result["deleted_sessions"],
        deleted_events=result["deleted_events"],
        failed_session_ids=result["failed_session_ids"],
        audit_path=audit_path,
        audit_paths=(audit_path,),
    )


def _recover_delete_journals(store: TraceStore) -> None:
    store.recover_delete_journals()


def _delete_tenant_one(store: TraceStore, tenant_id: str) -> TenantDeletionResult:
    _recover_delete_journals(store)
    with store.write_lock():
        metas = store.list_sessions_strict(tenant_id=tenant_id)
        operation_id = uuid.uuid4().hex
        journal_dir = store._journal_dir()
        journal_dir.mkdir(parents=True, exist_ok=True)
        journal_path = journal_dir / f"delete-{operation_id}.json"
        sessions = []
        for meta in metas:
            try:
                event_count = len(store.load_events(meta.session_id))
            except (OSError, ValueError, json.JSONDecodeError):
                event_count = 0
            sessions.append({"session_id": meta.session_id, "event_count": event_count})
        journal = {
            "operation": "delete_tenant",
            "operation_id": operation_id,
            "tenant_id": tenant_id,
            "workspace_id": store.workspace_id,
            "phase": "intent",
            "sessions": sessions,
        }
        _write_atomic(journal_path, json.dumps(journal, separators=(",", ":")))
        return _execute_delete_journal_locked(store, journal_path, journal)


def delete_tenant_data(
    store: TraceStore | list[TraceStore], tenant_id: str,
) -> TenantDeletionResult:
    """Hard-delete matching sessions across flat and workspace stores."""
    tenant_id = validate_tenant_id(tenant_id)
    stores = _coerce_stores(store)
    targets = [
        scoped
        for scoped in stores
        if scoped.list_sessions_strict(tenant_id=tenant_id)
    ]
    if not targets:
        targets = stores[:1]
    results = [_delete_tenant_one(scoped, tenant_id) for scoped in targets]
    audit_paths = tuple(path for result in results for path in result.audit_paths)
    failures = tuple(sid for result in results for sid in result.failed_session_ids)
    return TenantDeletionResult(
        tenant_id=tenant_id,
        deleted_sessions=sum(result.deleted_sessions for result in results),
        deleted_events=sum(result.deleted_events for result in results),
        failed_session_ids=failures,
        audit_path=audit_paths[0],
        audit_paths=audit_paths,
    )


def cmd_tenant(args: argparse.Namespace) -> int:
    """Handle the ``tenant`` report/export/delete command family."""
    try:
        stores = enumerate_trace_stores(args.trace_dir)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Tenant store discovery failed: {exc}\n")
        return 1
    action = getattr(args, "tenant_command", "")

    if action == "report":
        month = getattr(args, "month", "") or datetime.now(timezone.utc).strftime("%Y-%m")
        try:
            report = build_tenant_report(stores, month, model=getattr(args, "model", DEFAULT_MODEL))
        except ValueError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        if getattr(args, "format", "text") == "json":
            payload = {
                "month": report.month,
                "total_sessions": report.total_sessions,
                "total_cost_usd": report.total_cost_usd,
                "tenants": [
                    {
                        "tenant_id": row.tenant_id or None,
                        "sessions": row.sessions,
                        "cost_usd": row.cost_usd,
                    }
                    for row in report.rows
                ],
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            format_tenant_report(report)
        return 0

    tenant_id = getattr(args, "tenant_id", "")
    try:
        tenant_id = validate_tenant_id(tenant_id)
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if action == "export":
        try:
            payload = tenant_export_payload(stores, tenant_id)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"Export failed: {exc}\n")
            return 1
        text = json.dumps(payload, indent=2) + "\n"
        output = getattr(args, "output", "") or ""
        if output:
            Path(output).write_text(text, encoding="utf-8")
            sys.stdout.write(f"Exported {payload['session_count']} session(s) to {output}\n")
        else:
            sys.stdout.write(text)
        return 0

    if action == "delete":
        if not getattr(args, "confirm", False):
            sys.stderr.write(
                "Refusing tenant deletion without --confirm. This operation is irreversible.\n"
            )
            return 1
        try:
            result = delete_tenant_data(stores, tenant_id)
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"Deletion failed: {exc}\n")
            return 1
        sys.stdout.write(
            f"Deleted {result.deleted_sessions} session(s) and "
            f"{result.deleted_events} event(s) for tenant {tenant_id}.\n"
            f"Audit record(s): {', '.join(str(path) for path in result.audit_paths)}\n"
        )
        if result.failed_session_ids:
            sys.stderr.write(
                "Failed to delete session(s): " + ", ".join(result.failed_session_ids) + "\n"
            )
            return 1
        return 0

    sys.stderr.write("Specify a tenant subcommand: report, export, or delete.\n")
    return 1
