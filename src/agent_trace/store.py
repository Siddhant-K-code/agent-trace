"""Trace storage.

Traces are stored as directories:
  .agent-traces/
    <session-id>/
      meta.json       # session metadata
      events.ndjson   # newline-delimited JSON events

NDJSON is append-only. No database. No dependencies. Just files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import EventType, SessionMeta, TraceEvent

DEFAULT_TRACE_DIR = ".agent-traces"


class TraceStore:
    def __init__(self, base_dir: str | Path = DEFAULT_TRACE_DIR):
        self.base_dir = Path(base_dir)

    def _session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def create_session(self, meta: SessionMeta) -> Path:
        d = self._session_dir(meta.session_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(meta.to_json())
        # create empty events file
        (d / "events.ndjson").touch()
        return d

    def append_event(self, session_id: str, event: TraceEvent) -> None:
        f = self._session_dir(session_id) / "events.ndjson"
        with open(f, "a") as fh:
            fh.write(event.to_json() + "\n")

    def update_meta(self, meta: SessionMeta) -> None:
        f = self._session_dir(meta.session_id) / "meta.json"
        f.write_text(meta.to_json())

    def load_meta(self, session_id: str) -> SessionMeta:
        f = self._session_dir(session_id) / "meta.json"
        return SessionMeta.from_json(f.read_text())

    def load_events(self, session_id: str) -> list[TraceEvent]:
        f = self._session_dir(session_id) / "events.ndjson"
        events = []
        for line in f.read_text().strip().splitlines():
            if line:
                events.append(TraceEvent.from_json(line))
        return events

    def list_sessions(self) -> list[SessionMeta]:
        if not self.base_dir.exists():
            return []
        sessions = []
        for d in sorted(self.base_dir.iterdir(), reverse=True):
            meta_file = d / "meta.json"
            if meta_file.exists():
                try:
                    sessions.append(SessionMeta.from_json(meta_file.read_text()))
                except (json.JSONDecodeError, TypeError):
                    continue
        return sessions

    def get_latest_session_id(self) -> str | None:
        sessions = self.list_sessions()
        if not sessions:
            return None
        return sessions[0].session_id

    def session_exists(self, session_id: str) -> bool:
        return (self._session_dir(session_id) / "meta.json").exists()

    def find_session(self, prefix: str) -> str | None:
        """Find a session by prefix match."""
        if not self.base_dir.exists():
            return None
        for d in self.base_dir.iterdir():
            if d.name.startswith(prefix) and (d / "meta.json").exists():
                return d.name
        return None

    def annotations_path(self, session_id: str) -> Path:
        """Return the path to the annotations sidecar file."""
        return self._session_dir(session_id) / "annotations.jsonl"
