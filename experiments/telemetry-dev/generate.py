#!/usr/bin/env python3
"""Build the synthetic #241 sessions and export them as OTLP JSON.

The parent session starts from the corpus fixture
``src/agent_trace/corpus/v1/fixture-review-001``. The fixture already has a
tool failure and retry, a redacted prompt, and an orphan tool result as a
known capture gap. The parent adds what the fixture lacks:

* a provider and a model
* a failed model request and its retry
* token counts with cache reads and writes
* a recorded cost
* provenance
* a ``Task`` tool call that starts a child subagent session

The fixture is only read, never written.

Each payload gets its own session and event IDs. That way the three payloads
never share a trace ID or a span ID when they go to one Telemetry.dev
project:

* ``payload-otlp.json`` comes from ``agent-strace export --format otlp``
  (CLI). It sends the parent and child resource blocks in one request.
* ``payload-otlp-genai.json`` comes from
  ``agent-strace export --format otlp-genai`` (CLI). It also sends the parent
  and child resource blocks in one request.
* ``payload-otlp-genai-tree.json`` does NOT come from the CLI. It calls the
  library function ``session_to_otlp_genai`` with ``parent_trace_id`` and
  ``parent_span_id``, so the child's root span sits under the parent's
  ``Task`` span in one trace. The CLI exports one session per trace and has
  no option for this.

Only synthetic, relative paths and IDs are used. Timestamps are fixed, so the
output is the same byte for byte on every run; ``send.py`` moves them to the
send time. You can run it from any directory:

    python3 experiments/telemetry-dev/generate.py
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
FIXTURE = SRC / "agent_trace" / "corpus" / "v1" / "fixture-review-001"

sys.path.insert(0, str(SRC))

from agent_trace import __version__  # noqa: E402
from agent_trace.cost import _usage_dollars, usage_detail  # noqa: E402
from agent_trace.models import EventType, SessionMeta, TraceEvent  # noqa: E402
from agent_trace.otlp import session_to_otlp_genai  # noqa: E402
from agent_trace.store import TraceStore  # noqa: E402

MODEL = "claude-sonnet-4-5"
PROVIDER = "anthropic"
# These are the counts from the Telemetry.dev cost worked example. Its
# documented server-side price ($0.03765 in the 2026-09-24 snapshot) can then
# be compared with the agent-strace estimate.
PARENT_USAGE = {
    "input_tokens": 3000,
    "cache_read_input_tokens": 8000,
    "cache_creation_input_tokens": 1000,
    "output_tokens": 1500,
}
CHILD_USAGE = {"input_tokens": 2400, "output_tokens": 300}
# This is a recorded, provider-billed cost on the child's model call. It is
# deliberately different from the token price, (2400*3 + 300*15)/1e6 =
# $0.0117. The UI then shows whether Telemetry.dev keeps the client cost that
# AgentTrace sends or calculates its own.
CHILD_SENTINEL_COST_USD = 0.01234

BASE = 1789300800.0  # fixture start time, 2026-09-13T12:00:00Z
VARIANTS = {
    "otlp": "td241-otlp",
    "otlp-genai": "td241-genai",
    "otlp-genai-tree": "td241-tree",
}
OUTPUTS = {
    "otlp": HERE / "payload-otlp.json",
    "otlp-genai": HERE / "payload-otlp-genai.json",
    "otlp-genai-tree": HERE / "payload-otlp-genai-tree.json",
}
ATTRIBUTION = {
    "agent_provider": "claude-code",
    "agent_version": "0.0.0-synthetic",
    "os_user": "synthetic-user",
    "hostname": "synthetic-host",
    "pid": 4242,
    "ppid": 4241,
    "process_name": "synthetic-shell",
    "process_ancestry": ["synthetic-terminal", "synthetic-shell", "claude"],
    "git_repo": "example/synthetic-project",
    "git_branch": "main",
    "git_commit": "1111111111111111111111111111111111111111",
    "working_dir": "/workspace/project",
}


def _fixture_events() -> dict[str, dict]:
    events = {}
    with open(FIXTURE / "events.ndjson", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                events[record["event_id"]] = record
    return events


def parent_cost_usd() -> float:
    usage, _, _ = usage_detail(PARENT_USAGE)
    return round(_usage_dollars(usage, MODEL), 8)


def build_sessions(prefix: str) -> tuple[SessionMeta, list[TraceEvent], SessionMeta, list[TraceEvent]]:
    """Return the parent and child metas and events for one payload."""
    parent_sid = f"{prefix}-parent"
    child_sid = f"{prefix}-child"
    fx = _fixture_events()

    def eid(name: str) -> str:
        return f"{prefix}-{name}"

    def from_fixture(name: str, ts: float, data: dict | None = None,
                     duration_ms: float | None = None) -> TraceEvent:
        record = copy.deepcopy(fx[name])
        record_data = record.get("data", {})
        record_data.update(data or {})
        parent = record.get("parent_id", "")
        # The fixture's orphan points at "missing-tool-call" on purpose; keep it.
        if parent and parent != "missing-tool-call":
            parent = eid(parent)
        return TraceEvent(
            event_type=EventType(record["event_type"]),
            timestamp=ts,
            event_id=eid(name),
            session_id=parent_sid,
            parent_id=parent,
            duration_ms=duration_ms if duration_ms is not None else record.get("duration_ms"),
            data=record_data,
            redacted=bool(record.get("redacted")),
            tenant_id="",
        )

    def new(event_type: EventType, name: str, ts: float, data: dict, parent: str = "",
            duration_ms: float | None = None, session_id: str = parent_sid,
            redacted: bool = False) -> TraceEvent:
        return TraceEvent(
            event_type=event_type,
            timestamp=ts,
            event_id=eid(name),
            session_id=session_id,
            parent_id=eid(parent) if parent else "",
            duration_ms=duration_ms,
            data=data,
            redacted=redacted,
            tenant_id="",
        )

    t = BASE
    parent_events = [
        from_fixture("ev-start", t, {
            "provider": "claude-code", "source": "telemetry-dev-preflight-241",
        }),
        from_fixture("ev-prompt", t + 1),
        new(EventType.LLM_REQUEST, "llm-req-1", t + 1.2, {
            "model": MODEL, "provider": PROVIDER, "max_tokens": 4096,
        }),
        new(EventType.LLM_RESPONSE, "llm-res-1", t + 1.6, {
            "model": MODEL, "provider": PROVIDER, "status": "failed",
            "error": {"type": "overloaded_error", "message": "synthetic overload, retrying"},
        }, parent="llm-req-1"),
        new(EventType.LLM_REQUEST, "llm-req-2", t + 1.8, {
            "model": MODEL, "provider": PROVIDER, "max_tokens": 4096,
            "retry_of": eid("llm-req-1"),
        }),
        new(EventType.LLM_RESPONSE, "llm-res-2", t + 2.6, {
            "model": MODEL, "provider": PROVIDER, "stop_reason": "tool_use",
            "usage": dict(PARENT_USAGE),
            "estimated_cost_usd": parent_cost_usd(),
        }, parent="llm-req-2"),
        from_fixture("ev-read", t + 2.8),
        from_fixture("tool-test-1", t + 3),
        # A live capture pairs the error with its call (pairing.link_response)
        # and records the time between them. The fixture's error has no
        # duration, so without this the span would get the 1 ms default.
        from_fixture("ev-error", t + 4.1, duration_ms=1100.0),
        from_fixture("ev-write", t + 5),
        from_fixture("tool-test-2", t + 6, {"retry_of": eid("tool-test-1")}),
        from_fixture("result-test-2", t + 7),
        new(EventType.TOOL_CALL, "task-1", t + 7.2, {
            "tool_name": "Task",
            "arguments": {
                "description": "review the fix",
                "subagent_type": "code-reviewer",
                "prompt": "Review src/example.py for regressions.",
            },
        }),
        new(EventType.TOOL_RESULT, "task-1-result", t + 7.9, {
            "tool_name": "Task", "result": "review: no regressions found",
        }, parent="task-1", duration_ms=700.0),
        from_fixture("ev-decision", t + 8),
        from_fixture("ev-orphan", t + 9),
        from_fixture("ev-response", t + 10),
        from_fixture("ev-end", t + 11),
    ]
    parent_meta = SessionMeta(
        session_id=parent_sid,
        started_at=t,
        ended_at=t + 11,
        agent_name="claude-code",
        command="claude",
        tool_calls=3,
        llm_requests=2,
        errors=2,  # the tool error and the failed model request
        attribution=dict(ATTRIBUTION),
        conversation_id=f"synthetic-conversation-{prefix}",
        redacted=True,
        tenant_id="",
    )

    agent = {"agent_id": "synthetic-agent-1", "agent_type": "code-reviewer"}
    child_events = [
        new(EventType.SESSION_START, "c-start", t + 7.25, {
            "provider": "claude-code", "source": "telemetry-dev-preflight-241", **agent,
        }, session_id=child_sid),
        new(EventType.USER_PROMPT, "c-prompt", t + 7.3, {
            "prompt": "Review src/example.py for regressions.", **agent,
        }, session_id=child_sid),
        new(EventType.LLM_REQUEST, "c-llm-req-1", t + 7.35, {
            "model": MODEL, "provider": PROVIDER, "max_tokens": 1024, **agent,
        }, session_id=child_sid),
        new(EventType.LLM_RESPONSE, "c-llm-res-1", t + 7.55, {
            "model": MODEL, "provider": PROVIDER, "stop_reason": "tool_use",
            "usage": dict(CHILD_USAGE), "cost_usd": CHILD_SENTINEL_COST_USD, **agent,
        }, parent="c-llm-req-1", session_id=child_sid),
        new(EventType.TOOL_CALL, "c-tool-1", t + 7.6, {
            "tool_name": "read_file",
            "arguments": {"path": "src/example.py", "auth_header": "[REDACTED:bearer]"},
            **agent,
        }, session_id=child_sid, redacted=True),
        new(EventType.TOOL_RESULT, "c-result-1", t + 7.7, {
            "tool_name": "read_file", "result": "def example():\n    return 42\n", **agent,
        }, parent="c-tool-1", duration_ms=100.0, session_id=child_sid),
        new(EventType.ASSISTANT_RESPONSE, "c-response", t + 7.8, {
            "text": "No regressions found in src/example.py.", **agent,
        }, session_id=child_sid),
        new(EventType.SESSION_END, "c-end", t + 7.85, {"reason": "complete", **agent},
            session_id=child_sid),
    ]
    child_meta = SessionMeta(
        session_id=child_sid,
        started_at=t + 7.25,
        ended_at=t + 7.85,
        agent_name="claude-code",
        command="claude",
        tool_calls=1,
        llm_requests=1,
        errors=0,
        parent_session_id=parent_sid,
        parent_event_id=eid("task-1"),
        depth=1,
        attribution=dict(ATTRIBUTION),
        redacted=True,
        tenant_id="",
    )
    return parent_meta, parent_events, child_meta, child_events


def _clean_env(home: str) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("AGENT_STRACE_", "AGENT_TRACE_", "OTEL_"))
    }
    env.update({
        "PYTHONPATH": str(SRC),
        "HOME": home,
        "DO_NOT_TRACK": "1",
        "AGENT_STRACE_TELEMETRY": "0",
    })
    return env


def write_sessions(store: TraceStore, sessions) -> None:
    parent_meta, parent_events, child_meta, child_events = sessions
    store.create_session(parent_meta)
    store.append_events(parent_meta.session_id, parent_events)
    store.create_session(child_meta)
    store.append_events(child_meta.session_id, child_events)


def cli_export(trace_dir: str, home: str, session_id: str, fmt: str, out: Path) -> dict:
    cmd = [
        sys.executable, "-m", "agent_trace.entry", "--trace-dir", trace_dir,
        "export", session_id, "--format", fmt, "--output", str(out),
    ]
    subprocess.run(cmd, check=True, cwd=home, env=_clean_env(home),
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return json.loads(out.read_text(encoding="utf-8"))


def find_span(payload: dict, event_id: str) -> dict:
    for rs in payload["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            for span in ss["spans"]:
                for attr in span.get("attributes", []):
                    if attr["key"].endswith(".event_id") and attr["value"].get("stringValue") == event_id:
                        return span
    raise LookupError(f"no span for event {event_id}")


def tree_payload(store: TraceStore, prefix: str) -> dict:
    parent_sid, child_sid = f"{prefix}-parent", f"{prefix}-child"
    parent = session_to_otlp_genai(store.load_meta(parent_sid), store.load_events(parent_sid))
    task_span = find_span(parent, f"{prefix}-task-1")
    child = session_to_otlp_genai(
        store.load_meta(child_sid), store.load_events(child_sid),
        parent_span_id=task_span["spanId"],
        parent_trace_id=task_span["traceId"],
    )
    return {"resourceSpans": parent["resourceSpans"] + child["resourceSpans"]}


def dump(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    for key in ("AGENT_STRACE_WORKSPACE", "AGENT_STRACE_TENANT_ID"):
        os.environ.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="td241-") as tmp:
        home = os.path.join(tmp, "home")
        trace_dir = os.path.join(tmp, "traces")
        os.makedirs(home)
        store = TraceStore(trace_dir)
        for prefix in VARIANTS.values():
            write_sessions(store, build_sessions(prefix))
        for fmt in ("otlp", "otlp-genai"):
            prefix = VARIANTS[fmt]
            parts = [
                cli_export(trace_dir, home, f"{prefix}-{role}", fmt,
                           Path(tmp) / f"{prefix}-{role}.json")
                for role in ("parent", "child")
            ]
            merged = {"resourceSpans": parts[0]["resourceSpans"] + parts[1]["resourceSpans"]}
            dump(merged, OUTPUTS[fmt])
        dump(tree_payload(store, VARIANTS["otlp-genai-tree"]), OUTPUTS["otlp-genai-tree"])

    for path in OUTPUTS.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        spans = sum(len(ss["spans"]) for rs in payload["resourceSpans"] for ss in rs["scopeSpans"])
        print(f"{path.name}: {spans} spans, {path.stat().st_size} bytes")
    print(f"agent-strace {__version__}; parent model-call cost estimate ${parent_cost_usd()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
