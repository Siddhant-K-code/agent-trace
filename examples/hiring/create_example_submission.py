"""Create a synthetic, identity-free assignment submission for local demos."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from agent_trace.assignment import _atomic_write_private, build_assignment_bundle
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "example-submission.zip")
    with tempfile.TemporaryDirectory() as directory:
        store = TraceStore(directory)
        meta = SessionMeta(session_id="synthetic-assignment")
        store.create_session(meta)
        events = (
            TraceEvent(EventType.SESSION_START, timestamp=0.0, event_id="start"),
            TraceEvent(
                EventType.TOOL_CALL,
                timestamp=1.0,
                event_id="write",
                data={"tool_name": "Write", "arguments": {}},
            ),
            TraceEvent(
                EventType.SESSION_END,
                timestamp=2.0,
                event_id="end",
                data={"exit_code": 0},
            ),
        )
        for event in events:
            event.session_id = meta.session_id
            store.append_event(meta.session_id, event)
        bundle = build_assignment_bundle(store, meta.session_id)
    _atomic_write_private(output, bundle)
    print(f"Created {output}")


if __name__ == "__main__":
    main()
