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
import math
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

from .event_data import UnusableTimestampError, require_usable_timestamps
from .inert import inert
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

_TEXT_FIELDS = ("annotation_id", "session_id", "event_id", "label", "note", "author")
_NUMBER_FIELDS = ("offset_seconds", "created_at")


def _is_finite_number(value: object) -> bool:
    """Return True for an int or float, not a bool, that float() keeps finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


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
    shareable: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "Annotation":
        """Parse one line of an annotations file.

        Raises ValueError when the line is not valid JSON or not a JSON
        object, or when a known field has the wrong type: the text fields
        must be strings, and the number fields must be finite ints or
        floats. Raises RecursionError when the JSON nests too deep. A missing
        field keeps its default. Only JSON true makes the annotation
        shareable.
        """
        d = json.loads(line)
        if not isinstance(d, dict):
            raise ValueError("annotation is not a JSON object")
        for name in _TEXT_FIELDS:
            if name in d and not isinstance(d[name], str):
                raise ValueError(f"annotation field {name} is not a string")
        for name in _NUMBER_FIELDS:
            if name in d and not _is_finite_number(d[name]):
                raise ValueError(f"annotation field {name} is not a finite number")
        known = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in d.items() if k in known}
        values["shareable"] = d.get("shareable") is True
        return cls(**values)

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
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(annotation.to_json() + "\n")


def _read_lines(path: Path) -> list[str]:
    """Read an annotations file as lines.

    read_text() turns CR and CRLF into LF, and the text is split on LF only.
    Another writer can leave U+2028 or another Unicode line break raw inside
    a JSON string, and splitlines() would cut that line in two. Each byte
    that is not valid UTF-8 is read as a lone surrogate, so one bad line does
    not stop the read. Writing with errors="surrogateescape" gives back the
    same byte.
    """
    return path.read_text(encoding="utf-8", errors="surrogateescape").split("\n")


def load_annotations(store: TraceStore, session_id: str) -> list[Annotation]:
    """Load all annotations for a session.

    A line that is not valid UTF-8, or that from_json rejects, is skipped.
    """
    path = _annotations_path(store, session_id)
    if not path.exists():
        return []
    annotations: list[Annotation] = []
    for line in _read_lines(path):
        line = line.strip()
        if line:
            try:
                # Raises UnicodeDecodeError, a ValueError, if the line held a
                # byte that is not valid UTF-8.
                line.encode("utf-8", "surrogateescape").decode("utf-8")
                annotations.append(Annotation.from_json(line))
            except (ValueError, RecursionError):
                continue
    return annotations


def _line_annotation_id(line: str) -> object:
    """Return the annotation_id of a JSON object line, or None.

    It checks no other field, so delete_annotation can remove a line that
    load_annotations skips.
    """
    try:
        d = json.loads(line)
    except (ValueError, RecursionError):
        return None
    return d.get("annotation_id") if isinstance(d, dict) else None


def delete_annotation(
    store: TraceStore,
    session_id: str,
    annotation_id: str,
) -> bool:
    """Remove an annotation by ID. Returns True if found and removed.

    A line that is not valid UTF-8 is written back with the same bytes.
    """
    path = _annotations_path(store, session_id)
    if not path.exists():
        return False
    lines = _read_lines(path)
    new_lines = []
    found = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if _line_annotation_id(line) == annotation_id:
            found = True
            continue
        new_lines.append(line)
    if found:
        path.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8",
            errors="surrogateescape",
            newline="\n",
        )
    return found


# ---------------------------------------------------------------------------
# Offset parsing
# ---------------------------------------------------------------------------

def _parse_offset(offset_str: str) -> float:
    """Parse a time offset string like '2m14s', '134s', '2:14' into seconds.

    Raises ValueError when the string does not parse or the result is not
    finite. load_annotations skips a line whose offset_seconds is not finite.
    """
    s = offset_str.strip().lower()
    import re
    m = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s?)?", s)
    m2 = re.fullmatch(r"(\d+):(\d{2}(?:\.\d+)?)", s)
    if m and (m.group(1) or m.group(2)):
        # Format: Xm Ys
        seconds = float(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    elif m2:
        # Format: M:SS
        seconds = float(m2.group(1)) * 60 + float(m2.group(2))
    else:
        # Plain seconds
        try:
            seconds = float(s.rstrip("s"))
        except ValueError:
            raise ValueError(f"Cannot parse offset: {offset_str!r}")
    # float() reads "inf", "nan" and "1e400", and a long run of digits
    # overflows to inf in every format.
    if not math.isfinite(seconds):
        raise ValueError(f"Cannot parse offset: {offset_str!r}")
    return seconds


def _find_event_by_offset(
    store: TraceStore,
    session_id: str,
    offset_seconds: float,
) -> str:
    """Return the event_id of the event nearest to *offset_seconds*.

    Returns "" when that event_id is not a string, so the annotation is
    offset-based. load_annotations skips a line whose event_id is not a
    string. Raises UnusableTimestampError when an event's timestamp is not a
    Unix time in seconds, because the offsets read every one.
    """
    events = store.load_events(session_id)
    require_usable_timestamps(events)
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
    return best.event_id if isinstance(best.event_id, str) else ""


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
    """Write the annotations as terminal text to *out*.

    Another tool can write any text into an annotations file, so each text
    field goes through inert() and an escape code or a line break in it
    shows as text. In the note, each run of whitespace becomes one space
    first. If time.localtime() rejects created_at, it prints as a number.
    """
    w = out.write
    if not annotations:
        w("No annotations.\n")
        return
    w(f"\n{len(annotations)} annotation(s):\n\n")
    for a in annotations:
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a.created_at))
        except (OverflowError, OSError, ValueError):
            ts = str(a.created_at)
        label_str = f"  [{inert(a.label)}]" if a.label else ""
        event_str = f"  event={inert(a.event_id)}" if a.event_id else f"  offset={a.offset_seconds:.1f}s"
        w(f"  {inert(a.annotation_id)}  {ts}{label_str}{event_str}\n")
        if a.note:
            w(f"    {inert(' '.join(str(a.note).split()))}\n")
        if a.author:
            w(f"    {inert(a.author)}\n")
        if a.shareable is True:
            w("    shareable\n")
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
    shareable = bool(getattr(args, "shareable", False))

    if not note and not label:
        sys.stderr.write("Provide --note and/or --label.\n")
        return 1

    offset_seconds = 0.0
    if at_str and not event_id:
        try:
            offset_seconds = _parse_offset(at_str)
            event_id = _find_event_by_offset(store, full_id, offset_seconds)
        except UnusableTimestampError:
            raise
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
        shareable=shareable,
    )
    add_annotation(store, full_id, annotation)
    sys.stdout.write(f"Annotation {annotation.annotation_id} added to {full_id[:12]}\n")
    return 0
