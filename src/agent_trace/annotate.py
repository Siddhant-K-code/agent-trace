"""Replay annotations: attach notes, labels, and bookmarks to trace events.

Annotations are stored in a sidecar file alongside the trace:
  .agent-traces/<session-id>/annotations.jsonl

The main events.ndjson is never modified. Annotations are read-only overlays.

Usage:
    agent-strace annotate <session-id> --event ev-00042 --note "root cause"
    agent-strace annotate <session-id> --event ev-00042 --label root-cause
    agent-strace annotate <session-id> --at 2m14s --note "retry loop starts"
    agent-strace annotate <session-id> --list
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

from .store import TraceStore


# ---------------------------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------------------------

# Predefined label colours (used in share HTML)
LABEL_COLOURS: dict[str, str] = {
    "root-cause": "#f85149",
    "decision":   "#58a6ff",
    "retry":      "#d29922",
    "fix":        "#3fb950",
    "question":   "#bc8cff",
}
DEFAULT_LABEL_COLOUR = "#8b949e"


@dataclass
class Annotation:
    annotation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    event_id: str = ""          # event_id from TraceEvent (empty if offset-based)
    offset_seconds: float = 0.0 # seconds from session start
    label: str = ""             # e.g. "root-cause", "decision", "retry"
    note: str = ""
    author: str = ""
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "Annotation":
        d = json.loads(line)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def label_colour(self) -> str:
        return LABEL_COLOURS.get(self.label, DEFAULT_LABEL_COLOUR)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _annotations_path(store: TraceStore, session_id: str) -> Path:
    return store.annotations_path(session_id)


def add_annotation(
    store: TraceStore,
    session_id: str,
    annotation: Annotation,
) -> None:
    """Append an annotation to the sidecar file."""
    annotation.session_id = session_id
    path = _annotations_path(store, session_id)
    with open(path, "a", encoding="utf-8") as f:
        f.write(annotation.to_json() + "\n")


def load_annotations(store: TraceStore, session_id: str) -> list[Annotation]:
    """Load all annotations for a session."""
    path = _annotations_path(store, session_id)
    if not path.exists():
        return []
    annotations: list[Annotation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                annotations.append(Annotation.from_json(line))
            except (json.JSONDecodeError, TypeError):
                continue
    return annotations


def delete_annotation(
    store: TraceStore,
    session_id: str,
    annotation_id: str,
) -> bool:
    """Remove an annotation by ID. Returns True if found and removed."""
    path = _annotations_path(store, session_id)
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    found = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            a = Annotation.from_json(line)
            if a.annotation_id == annotation_id:
                found = True
                continue
        except Exception:
            pass
        new_lines.append(line)
    if found:
        path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    return found


# ---------------------------------------------------------------------------
# Offset parsing
# ---------------------------------------------------------------------------

def _parse_offset(offset_str: str) -> float:
    """Parse a time offset string like '2m14s', '134s', '2:14' into seconds."""
    s = offset_str.strip().lower()
    # Format: Xm Ys
    import re
    m = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s?)?", s)
    if m and (m.group(1) or m.group(2)):
        minutes = float(m.group(1) or 0)
        seconds = float(m.group(2) or 0)
        return minutes * 60 + seconds
    # Format: M:SS
    m2 = re.fullmatch(r"(\d+):(\d{2}(?:\.\d+)?)", s)
    if m2:
        return float(m2.group(1)) * 60 + float(m2.group(2))
    # Plain seconds
    try:
        return float(s.rstrip("s"))
    except ValueError:
        raise ValueError(f"Cannot parse offset: {offset_str!r}")


def _find_event_by_offset(
    store: TraceStore,
    session_id: str,
    offset_seconds: float,
) -> str:
    """Return the event_id of the event nearest to *offset_seconds*."""
    events = store.load_events(session_id)
    if not events:
        return ""
    base_ts = events[0].timestamp
    best = events[0]
    best_diff = abs((events[0].timestamp - base_ts) - offset_seconds)
    for e in events[1:]:
        diff = abs((e.timestamp - base_ts) - offset_seconds)
        if diff < best_diff:
            best_diff = diff
            best = e
    return best.event_id


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def filter_annotations(
    annotations: list[Annotation],
    label: str = "",
    author: str = "",
    since: float = 0.0,
) -> list[Annotation]:
    """Filter annotations by label, author, and/or minimum created_at timestamp."""
    result = annotations
    if label:
        result = [a for a in result if a.label == label]
    if author:
        result = [a for a in result if a.author == author]
    if since:
        result = [a for a in result if a.created_at >= since]
    return result


def format_annotations(
    annotations: list[Annotation],
    out: TextIO = sys.stdout,
) -> None:
    w = out.write
    if not annotations:
        w("No annotations.\n")
        return
    w(f"\n{len(annotations)} annotation(s):\n\n")
    for a in annotations:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a.created_at))
        label_str = f"  [{a.label}]" if a.label else ""
        event_str = f"  event={a.event_id}" if a.event_id else f"  offset={a.offset_seconds:.1f}s"
        w(f"  {a.annotation_id}  {ts}{label_str}{event_str}\n")
        if a.note:
            w(f"    {a.note}\n")
        if a.author:
            w(f"    — {a.author}\n")
        w("\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_annotate(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    full_id = store.find_session(session_id)
    if not full_id:
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    # --list (with optional filters)
    if getattr(args, "list", False):
        annotations = load_annotations(store, full_id)

        # Apply filters
        filter_label = getattr(args, "filter_label", "") or ""
        filter_author = getattr(args, "filter_author", "") or ""
        since_str = getattr(args, "since", "") or ""
        since_ts = 0.0
        if since_str:
            import re as _re
            m = _re.fullmatch(r"(\d+)d", since_str.strip())
            if m:
                since_ts = time.time() - int(m.group(1)) * 86400
            else:
                sys.stderr.write(f"Invalid --since value: {since_str!r} (use e.g. 7d)\n")
                return 1
        annotations = filter_annotations(annotations, label=filter_label,
                                         author=filter_author, since=since_ts)

        export_fmt = getattr(args, "export_format", "") or ""
        if export_fmt == "json":
            import dataclasses as _dc
            sys.stdout.write(json.dumps([_dc.asdict(a) for a in annotations], indent=2) + "\n")
        else:
            format_annotations(annotations)
        return 0

    # --delete
    delete_id = getattr(args, "delete", None)
    if delete_id:
        found = delete_annotation(store, full_id, delete_id)
        if found:
            sys.stdout.write(f"Deleted annotation {delete_id}\n")
            return 0
        else:
            sys.stderr.write(f"Annotation not found: {delete_id}\n")
            return 1

    # Add annotation
    note = getattr(args, "note", "") or ""
    label = getattr(args, "label", "") or ""
    event_id = getattr(args, "event", "") or ""
    at_str = getattr(args, "at", "") or ""
    author = getattr(args, "author", "") or ""

    if not note and not label:
        sys.stderr.write("Provide --note and/or --label.\n")
        return 1

    offset_seconds = 0.0
    if at_str and not event_id:
        try:
            offset_seconds = _parse_offset(at_str)
            event_id = _find_event_by_offset(store, full_id, offset_seconds)
        except ValueError as e:
            sys.stderr.write(f"{e}\n")
            return 1

    annotation = Annotation(
        session_id=full_id,
        event_id=event_id,
        offset_seconds=offset_seconds,
        label=label,
        note=note,
        author=author,
    )
    add_annotation(store, full_id, annotation)
    sys.stdout.write(f"Annotation {annotation.annotation_id} added to {full_id[:12]}\n")
    return 0
