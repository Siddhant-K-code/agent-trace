"""Shared utilities for all auto-instrumentation integrations."""

from __future__ import annotations

import os
import time
from typing import Any

from ..models import EventType, SessionMeta, TraceEvent
from ..store import TraceStore, DEFAULT_TRACE_DIR


def _get_store() -> TraceStore:
    return TraceStore(os.environ.get("AGENT_TRACE_DIR", DEFAULT_TRACE_DIR))


def _get_or_create_session(store: TraceStore, name: str) -> str:
    """Return the active session ID, creating one if needed."""
    active_path = store.base_dir / ".active-session"
    if active_path.exists():
        sid = active_path.read_text().strip()
        if sid and store.session_exists(sid):
            return sid
    meta = SessionMeta(agent_name=name)
    store.create_session(meta)
    active_path.write_text(meta.session_id)
    return meta.session_id


def emit(event_type: EventType, session_id: str, store: TraceStore, **data: Any) -> None:
    """Write a single event to the store (or remote endpoint if configured)."""
    ev = TraceEvent(event_type=event_type, session_id=session_id, data=dict(data))
    endpoint = os.environ.get("AGENT_STRACE_ENDPOINT", "").rstrip("/")
    if endpoint:
        from ..server import send_event_to_endpoint
        send_event_to_endpoint(ev, endpoint)
    else:
        store.append_event(session_id, ev)
