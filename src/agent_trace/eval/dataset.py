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

    sys.stderr.write("Usage: agent-strace eval dataset <add|list|export>\n")
    return 1
