"""Read synthetic reliability cases without capture, writes, or network calls."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agent_trace.models import EventType, usable_number, usable_time
from agent_trace.review import build_review_projection
from agent_trace.store import TraceStore
from agent_trace.validation import _corpus_root, validate_trace_directory


CASES = (
    "complete-current",
    "resumed-current",
    "resumed-open",
    "incomplete-missing-outcome",
    "terminal-absent",
)


def render_case(store: TraceStore, case: str, out: TextIO) -> None:
    report = validate_trace_directory(store.base_dir / case)
    out.write(
        f"{case}: validation={report.status}; conformant={str(report.conformant).lower()}; "
        f"health={report.evidence_health['status']}\n"
    )
    meta = store.load_meta(case)
    events = store.load_events(case)
    review = build_review_projection(events)
    out.write(f"  Session: {meta.session_id}\n")
    started = usable_time(meta.started_at)
    ended = usable_time(meta.ended_at)
    lifetime = (
        f"{ended - started:g}s"
        if started is not None and ended is not None and ended >= started
        else "unknown"
    )
    out.write(f"  Recorded lifetime (ended_at - started_at): {lifetime}\n")
    for event in events:
        if event.event_type not in (EventType.SESSION_START, EventType.SESSION_END):
            continue
        offset = f"+{event.timestamp - started:g}s" if started is not None else "unknown"
        detail = ""
        if event.event_type == EventType.SESSION_START and "source" in event.data:
            detail = f"; source={event.data['source']}"
        if event.event_type == EventType.SESSION_END:
            duration = usable_number(event.data.get("duration_ms"))
            if duration is not None:
                detail = f"; cumulative duration={duration / 1000:g}s (not additive)"
        out.write(f"  {event.event_type.value} {event.event_id} at {offset}{detail}\n")
    for item in review.items:
        if item.event_type == EventType.TOOL_RESULT.value:
            out.write(
                f"  tool_result {item.event_id}: parent={item.parent_id}; "
                f"relationship={item.relationship}"
            )
            if "exit_code" in item.data:
                out.write(f"; recorded exit_code={item.data['exit_code']}")
            out.write("\n")
    for finding in report.findings:
        target = f" ({finding.event_id})" if finding.event_id else ""
        out.write(f"  Gap: {finding.code}{target}\n")
    out.write("\n")


def main(out: TextIO | None = None) -> int:
    out = sys.stdout if out is None else out
    out.write(
        "Synthetic evidence only. Recorded commands are not executed.\n"
        "Active work time: unknown. Lifetimes and cumulative durations do not measure it.\n\n"
    )
    store = TraceStore(_corpus_root(), use_workspace_env=False, read_only=True)
    for case in CASES:
        render_case(store, case, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
