"""Trace storage.

Traces are stored as directories:
  .agent-traces/
    <session-id>/
      meta.json       # session metadata
      events.ndjson   # newline-delimited JSON events

  With workspace isolation:
  .agent-traces/
    workspaces/
      <workspace-id>/
        <session-id>/
          meta.json
          events.ndjson

NDJSON is append-only. No database. No dependencies. Just files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .models import EventType, SessionMeta, TraceEvent

DEFAULT_TRACE_DIR = ".agent-traces"

# Env var for workspace isolation — sessions are stored under
# .agent-traces/workspaces/<workspace-id>/ when this is set.
_WORKSPACE_ENV = "AGENT_STRACE_WORKSPACE"


def _workspace_base(base_dir: str | Path, workspace_id: str) -> Path:
    """Return the workspace-scoped subdirectory path."""
    return Path(base_dir) / "workspaces" / workspace_id


class TraceStore:
    def __init__(self, base_dir: str | Path = DEFAULT_TRACE_DIR,
                 workspace_id: str = ""):
        """Create a TraceStore.

        workspace_id scopes all reads/writes to a subdirectory:
          <base_dir>/workspaces/<workspace_id>/

        If workspace_id is empty, the AGENT_STRACE_WORKSPACE env var is
        checked. If that is also empty, the flat layout is used.
        """
        wid = workspace_id or os.environ.get(_WORKSPACE_ENV, "")
        if wid:
            self.base_dir = _workspace_base(base_dir, wid)
            self.workspace_id = wid
        else:
            self.base_dir = Path(base_dir)
            self.workspace_id = ""

    def _session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def create_session(self, meta: SessionMeta) -> Path:
        # Stamp workspace_id onto meta so it's visible in exports/reports
        if self.workspace_id and not getattr(meta, "workspace_id", ""):
            meta.workspace_id = self.workspace_id
        d = self._session_dir(meta.session_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(meta.to_json())
        # create empty events file
        (d / "events.ndjson").touch()
        return d

    def append_event(self, session_id: str, event: TraceEvent) -> None:
        f = self._session_dir(session_id) / "events.ndjson"
        # Compute hash chain: SHA-256 of the last line in the file
        if not event.prev_hash:
            try:
                import hashlib as _hashlib
                text = f.read_text() if f.exists() else ""
                last_line = text.rstrip("\n").rsplit("\n", 1)[-1] if text.strip() else ""
                event.prev_hash = _hashlib.sha256(last_line.encode()).hexdigest() if last_line else ""
            except Exception:
                pass
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
        """Return valid sessions sorted newest first by started_at, then descending session ID."""
        if not self.base_dir.exists():
            return []
        sessions = []
        for d in self.base_dir.iterdir():
            meta_file = d / "meta.json"
            if meta_file.exists():
                try:
                    sessions.append(SessionMeta.from_json(meta_file.read_text()))
                except (json.JSONDecodeError, TypeError):
                    continue
        return sorted(
            sessions,
            key=lambda meta: (meta.started_at, meta.session_id),
            reverse=True,
        )

    def get_latest_session(self) -> SessionMeta | None:
        """Return the newest session metadata, or None when the store is empty."""
        sessions = self.list_sessions()
        if not sessions:
            return None
        return sessions[0]

    def get_latest_session_id(self) -> str | None:
        """Return the newest session ID, or None when the store is empty."""
        latest = self.get_latest_session()
        if not latest:
            return None
        return latest.session_id

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
