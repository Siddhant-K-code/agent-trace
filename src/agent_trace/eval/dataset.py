"""Dataset management for eval sessions.

Datasets are JSONL files stored in .agent-traces/datasets/.
Each entry records a session ID, label, and scorer configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..store import TraceStore




@dataclass
class DatasetEntry:
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    label: str = ""
    added_at: float = field(default_factory=time.time)
    scorers: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "DatasetEntry":
        return cls(**json.loads(line))


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def add_entry(dataset_path: str | Path, entry: DatasetEntry) -> None:
    p = Path(dataset_path)
    _ensure_dir(p)
    with open(p, "a", encoding="utf-8") as f:
        f.write(entry.to_json() + "\n")


def list_entries(dataset_path: str | Path) -> list[DatasetEntry]:
    p = Path(dataset_path)
    if not p.exists():
        return []
    entries = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(DatasetEntry.from_json(line))
            except (json.JSONDecodeError, TypeError):
                continue
    return entries


def export_entries(dataset_path: str | Path, out=sys.stdout) -> None:
    for entry in list_entries(dataset_path):
        out.write(entry.to_json() + "\n")


# ---------------------------------------------------------------------------
# Auto-sampling: populate a dataset from stored sessions by signal filter
# ---------------------------------------------------------------------------

def _session_passes_filter(
    store: "TraceStore",
    session_id: str,
    filter_spec: str,
    eval_threshold: float = 0.8,
) -> bool:
    """Return True if the session matches the given filter spec.

    Supported filters:
      has-errors          — session has at least one ERROR event
      high-retry          — retry rate > 30%
      cost-above:N        — estimated cost > $N
      wide-blast          — distinct files written > 10
      long-duration:Ns    — session duration > N seconds
      low-eval-score:N    — eval.json overall score < N
    """
    from ..models import EventType

    try:
        events = store.load_events(session_id)
        meta = store.load_meta(session_id)
    except Exception:
        return False

    spec = filter_spec.strip().lower()

    if spec == "has-errors":
        return any(e.event_type == EventType.ERROR for e in events)

    if spec == "high-retry":
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        if not tool_calls:
            return False
        retries = 0
        prev = None
        run = 0
        for ev in tool_calls:
            name = ev.data.get("tool_name", "")
            if name == prev:
                run += 1
                if run >= 2:
                    retries += 1
            else:
                prev = name
                run = 0
        return retries / len(tool_calls) > 0.30

    if spec.startswith("cost-above:"):
        try:
            threshold_dollars = float(spec.split(":", 1)[1])
        except ValueError:
            return False
        cost = meta.total_tokens / 1_000_000 * 3.0
        return cost > threshold_dollars

    if spec == "wide-blast":
        files: set[str] = set()
        for ev in events:
            if ev.event_type == EventType.FILE_WRITE:
                p = ev.data.get("path") or ev.data.get("file_path") or ""
                if p:
                    files.add(p)
        return len(files) > 10

    if spec.startswith("long-duration:"):
        try:
            max_s = float(spec.split(":", 1)[1].rstrip("s"))
        except ValueError:
            return False
        duration = meta.total_duration_ms / 1000 if meta.total_duration_ms else 0.0
        return duration > max_s

    if spec.startswith("low-eval-score:"):
        try:
            score_threshold = float(spec.split(":", 1)[1])
        except ValueError:
            score_threshold = eval_threshold
        eval_path = store.base_dir / session_id / "eval.json"
        if not eval_path.exists():
            return False
        try:
            data = json.loads(eval_path.read_text())
            results = data.get("results") or data.get("judges") or []
            if not results:
                return False
            avg = sum(float(r.get("score", 0)) for r in results) / len(results)
            return avg < score_threshold
        except Exception:
            return False

    return False


def auto_populate(
    store: "TraceStore",
    dataset_path: str | Path,
    filter_spec: str,
    since_days: float = 7.0,
    label: str = "",
    limit: int = 500,
) -> int:
    """Auto-populate a dataset from sessions matching a filter.

    Returns the number of entries added.
    """
    cutoff = time.time() - since_days * 86400
    added = 0

    existing = {e.session_id for e in list_entries(dataset_path)}

    for meta in store.list_sessions():
        if meta.started_at < cutoff:
            continue
        if meta.session_id in existing:
            continue
        if added >= limit:
            break
        if _session_passes_filter(store, meta.session_id, filter_spec):
            entry = DatasetEntry(
                session_id=meta.session_id,
                label=label or filter_spec,
            )
            add_entry(dataset_path, entry)
            added += 1

    return added


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_dataset(args: argparse.Namespace) -> int:
    dataset_command = getattr(args, "dataset_command", None)
    dataset_path = getattr(args, "dataset", ".agent-traces/datasets/default.jsonl")

    if dataset_command == "add":
        session_id = getattr(args, "session", "")
        label = getattr(args, "label", "")
        if not session_id:
            sys.stderr.write("--session is required\n")
            return 1
        entry = DatasetEntry(session_id=session_id, label=label)
        add_entry(dataset_path, entry)
        sys.stderr.write(f"Added session {session_id} to dataset {dataset_path}\n")
        return 0

    if dataset_command == "list":
        entries = list_entries(dataset_path)
        if not entries:
            sys.stdout.write(f"No entries in {dataset_path}\n")
            return 0
        sys.stdout.write(f"\nDataset: {dataset_path} ({len(entries)} entries)\n")
        sys.stdout.write(f"{'─' * 60}\n")
        for e in entries:
            label = f"  {e.label}" if e.label else ""
            sys.stdout.write(f"  {e.entry_id}  {e.session_id}{label}\n")
        sys.stdout.write(f"{'─' * 60}\n\n")
        return 0

    if dataset_command == "export":
        export_entries(dataset_path)
        return 0

    if dataset_command == "auto":
        from ..store import TraceStore
        filter_spec = getattr(args, "filter", "has-errors") or "has-errors"
        since_raw = getattr(args, "since", "7d") or "7d"
        since_days = float(since_raw.rstrip("d"))
        label = getattr(args, "label", "") or filter_spec
        trace_dir = getattr(args, "trace_dir", ".agent-traces")
        store = TraceStore(trace_dir)
        added = auto_populate(store, dataset_path, filter_spec, since_days=since_days, label=label)
        sys.stdout.write(f"Added {added} session(s) to {dataset_path} (filter: {filter_spec})\n")
        return 0

    sys.stderr.write("Usage: agent-strace eval dataset <add|list|export|auto>\n")
    return 1
