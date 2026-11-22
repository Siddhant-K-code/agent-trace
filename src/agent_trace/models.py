"""Trace data model.

An agent session produces a sequence of events. Each event has a type,
a timestamp, and a payload. The trace is a flat list of events stored
as newline-delimited JSON (NDJSON).

Event types:
  - tool_call: agent invoked a tool (MCP tools/call)
  - tool_result: tool returned a result
  - llm_request: agent sent a prompt to an LLM
  - llm_response: LLM returned a completion
  - file_read: agent read a file
  - file_write: agent wrote a file
  - decision: agent chose between alternatives (extracted from reasoning)
  - error: something failed
  - session_start: trace session began
  - session_end: trace session ended
  - user_prompt: user submitted a prompt to the agent
  - assistant_response: agent produced a text response
  - subagent_start: the agent started a subagent (recorded in the parent)
  - subagent_stop: a subagent finished (recorded in the parent)
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import Any


_UNKNOWN_FIELDS_ATTR = "_unknown_fields"

# SessionMeta refuses metadata that nests objects and arrays more than this
# many levels deep. Real metadata nests 3 levels: the meta object, attribution
# and its process_ancestry list. The limit is twice collector_client's
# MAX_JSON_DEPTH, so the store reads all metadata the collector accepts and the
# collector's own nesting message comes first. It is far below the depth where
# the redactor and dataclasses.asdict run out of stack: about 328 levels on
# Python 3.10 and 3.11.
MAX_META_DEPTH = 128


def _nests_deeper_than(value: Any, limit: int) -> bool:
    """Return True if ``value`` nests objects and arrays more than ``limit`` levels deep.

    ``value`` is level 1. The walk uses a stack, not recursion: on 3.12 and
    later, json.loads parses text far deeper than Python can recurse. A cycle
    ends the walk too, because each step down adds a level.
    """
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        for child in node.values() if isinstance(node, dict) else node:
            if isinstance(child, (dict, list, tuple)):
                if depth == limit:
                    return True
                stack.append((child, depth + 1))
    return False


def _meta_too_deep() -> ValueError:
    return ValueError(f"metadata nests more than {MAX_META_DEPTH} levels")


def usable_number(value: Any) -> float | None:
    """Return ``value`` as a float if it is a finite number >= 0, else None.

    meta.json can hold any JSON value, so readers use this to treat a bool,
    a string, NaN, an infinity, a negative number or an int too large for a
    float as missing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def usable_time(value: Any) -> float | None:
    """Return ``value`` as a float if it is a usable number that
    ``datetime.fromtimestamp`` accepts, else None."""
    try:
        datetime.fromtimestamp(value)
    except (OverflowError, ValueError, OSError, TypeError):
        return None
    return usable_number(value)


def usable_text(value: Any) -> str:
    """Return ``value`` if it is a string, else "".

    meta.json can hold any JSON value, so readers use this to treat an
    agent_name, command or team that is a list, an object, a number, a bool
    or null as missing.
    """
    return value if isinstance(value, str) else ""


def _from_dict(cls, data: Any):
    """Build ``cls`` from ``data``, ignoring keys that are not its fields.

    Newer writers may add optional fields. Ignoring them lets an older reader
    load the record instead of failing with an unexpected keyword argument.
    The ignored keys are kept on the instance and written back by ``to_json``
    so rewriting the record (``update_meta``, ``tag_session``) does not erase
    them.
    """
    if not isinstance(data, dict):
        return cls(**data)
    names = {f.name for f in fields(cls)}
    known = {key: value for key, value in data.items() if key in names}
    unknown = {key: value for key, value in data.items() if key not in names}
    obj = cls(**known)
    if unknown:
        setattr(obj, _UNKNOWN_FIELDS_ATTR, unknown)
    return obj


def _with_unknown_fields(obj: Any, d: dict[str, Any]) -> dict[str, Any]:
    unknown = getattr(obj, _UNKNOWN_FIELDS_ATTR, None)
    if isinstance(unknown, dict):
        d.update({key: value for key, value in unknown.items() if key not in d})
    return d


class EventType(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    DECISION = "decision"
    ERROR = "error"
    USER_PROMPT = "user_prompt"
    ASSISTANT_RESPONSE = "assistant_response"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"


@dataclass
class TraceEvent:
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    parent_id: str = ""  # links tool_result to tool_call, llm_response to llm_request
    duration_ms: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    # SHA-256 hex digest of the previous event's JSON line (empty on first event)
    prev_hash: str = ""
    # True when secret redaction changed or protected this event before write
    redacted: bool = False
    # Appended after all pre-v0.90 fields to preserve positional construction.
    tenant_id: str = field(
        default_factory=lambda: os.environ.get("AGENT_STRACE_TENANT_ID", "").strip()
    )

    def to_json(self) -> str:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        if not d.get("redacted"):
            d.pop("redacted", None)
        # drop None values
        d = {k: v for k, v in d.items() if v is not None and v != ""}
        return json.dumps(_with_unknown_fields(self, d), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> TraceEvent:
        return cls.from_record(json.loads(line))

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> TraceEvent:
        """Build an event from one decoded JSON line. ``d`` is changed in place."""
        d["event_type"] = EventType(d["event_type"])
        # Deserialising a legacy event must not accidentally assign the
        # current process tenant to historical data.
        d.setdefault("tenant_id", "")
        return _from_dict(cls, d)


@dataclass
class SessionMeta:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    agent_name: str = ""
    command: str = ""
    tool_calls: int = 0
    llm_requests: int = 0
    errors: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0
    # Subagent correlation fields (optional — absent on root sessions)
    parent_session_id: str = ""   # session ID of the spawning agent
    parent_event_id: str = ""     # event_id of the tool_call that spawned this session
    depth: int = 0                # nesting depth (0 = root, 1 = first subagent, etc.)
    # Team grouping (optional — used for team budget reports)
    team: str = ""
    # Workspace isolation — sessions in different workspaces are stored in
    # separate subdirectories and never appear in each other's listings
    workspace_id: str = ""
    # Attribution (who/what started this session)
    attribution: dict = field(default_factory=dict)
    # Upstream W3C trace context (optional).  Integrations such as Temporal
    # persist these fields so an offline OTLP export can retain the original
    # distributed trace and parent span.
    trace_id: str = ""
    parent_span_id: str = ""
    trace_flags: str = ""
    # True when secret redaction changed or protected session metadata
    redacted: bool = False
    # Appended after all pre-v0.90 fields to preserve positional construction.
    tenant_id: str = field(
        default_factory=lambda: os.environ.get("AGENT_STRACE_TENANT_ID", "").strip()
    )
    # The agent's own session ID in full (Claude Code ``session_id``, Copilot
    # CLI ``sessionId``). Empty when the agent did not supply one. Exported as
    # ``gen_ai.conversation.id`` (ADR-0011).
    conversation_id: str = ""

    def to_json(self) -> str:
        # asdict and json.dumps recurse, so the depth is checked before either
        # runs. The store then never writes metadata that from_json refuses.
        document = {f.name: getattr(self, f.name) for f in fields(self)}
        if _nests_deeper_than(_with_unknown_fields(self, document), MAX_META_DEPTH):
            raise _meta_too_deep()
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None and v != "" and v != 0}
        return json.dumps(_with_unknown_fields(self, d), indent=2)

    @classmethod
    def from_json(cls, text: str) -> SessionMeta:
        """Build a SessionMeta from ``text``, or raise ValueError.

        Metadata that is not a JSON object, or that nests more than
        MAX_META_DEPTH levels, raises ValueError like any other corrupt
        metadata. That includes text too deep for json.loads, which raises
        RecursionError.
        """
        try:
            data = json.loads(text)
        except RecursionError as exc:
            raise _meta_too_deep() from exc
        if not isinstance(data, dict):
            raise ValueError("metadata is not a JSON object")
        if _nests_deeper_than(data, MAX_META_DEPTH):
            raise _meta_too_deep()
        # Keep pre-tenancy metadata untagged even when the reader itself is
        # running with AGENT_STRACE_TENANT_ID set.
        data.setdefault("tenant_id", "")
        return _from_dict(cls, data)
