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
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
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


def usable_count(value: Any) -> int:
    """Return ``value`` if it is an int that usable_number() accepts, else 0.

    meta.json can hold any JSON value, so readers use this for the counts
    and the depth. A bool, a string, a float such as 2.5, NaN, a negative
    number or an int too large for a float reads as 0, as a missing count
    does. This is the collector's rule for a count, plus the float limit.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if usable_number(value) is not None else 0


# usable_time() refuses a time after this one. Time zones run from UTC-12 to
# UTC+14, so this time falls on 9999-12-30 or earlier in every zone. A reader
# can still add a day to its local or UTC date.
LAST_USABLE_TIME = 253402128000.0  # 9999-12-30T00:00:00Z


def usable_time(value: Any) -> float | None:
    """Return ``value`` as a float if it is a usable Unix time in seconds, else None.

    A usable time is a number from 0 to LAST_USABLE_TIME, which is
    9999-12-30T00:00:00Z. The limit is the same in every time zone, so a
    session is usable on every host or on none. Windows can have a lower
    limit, near the end of the year 3000. So this also returns None for a
    time that ``datetime.fromtimestamp`` refuses in local time or in UTC.
    """
    number = usable_number(value)
    if number is None or number > LAST_USABLE_TIME:
        return None
    try:
        datetime.fromtimestamp(number)
        datetime.fromtimestamp(number, tz=timezone.utc)
    except (OverflowError, ValueError, OSError):
        return None
    return number


_SINCE_DAYS = re.compile(r"([0-9]+(?:\.[0-9]+)?)d?")

SINCE_DAYS_FORM = "use a number of days as N or Nd, such as 7d"


def parse_since_days(value: str) -> float | None:
    """Return the days in a --since value that takes a number of days.

    The value is N or Nd. N is ASCII digits with at most one dot between
    digits, such as 7, 7d or 0.5d. Any other value gives None, and so does
    an N with so many digits that it is not a finite float.
    """
    match = _SINCE_DAYS.fullmatch(value)
    if match is None:
        return None
    days = float(match.group(1))
    return days if math.isfinite(days) else None


def usable_text(value: Any) -> str:
    """Return ``value`` if it is a string, else "".

    meta.json can hold any JSON value, so readers use this to treat an
    agent_name, command, team, parent_event_id, trace_id, parent_span_id or
    trace_flags that is a list, an object, a number, a bool or null as
    missing.
    """
    return value if isinstance(value, str) else ""


def usable_object(value: Any) -> dict:
    """Return ``value`` if it is a dict, else {}.

    meta.json can hold any JSON value, so readers use this to treat an
    attribution that is a string, a list, a number, a bool or null as
    missing. Do not convert with dict(): it turns a list of pairs into an
    object.
    """
    return value if isinstance(value, dict) else {}


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


class _NonFiniteKey(str):
    """The text json.dumps writes for a NaN or infinite float dict key.

    It hashes and compares by identity. So it never merges with another key
    of the same text, and the copy keeps every entry json.dumps wrote.
    """

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    def __ne__(self, other: object) -> bool:
        return self is not other


_LEAVE = object()


def _finite_copy(value: Any) -> Any:
    """Return a copy of ``value`` with each NaN or infinite float as None.

    A NaN or infinite float dict key becomes the text json.dumps writes for
    it. Dicts, lists and tuples are copied. Other values are kept, so
    json.dumps still refuses what it refused before. ``value`` is not
    changed. The walk uses a stack, not recursion, so it copies anything
    json.dumps can write. A container inside itself raises ValueError, as
    json.dumps does. One container in two places is not a cycle.
    """
    root = [value]
    on_path: set[int] = set()
    stack: list[tuple[Any, Any, Any]] = [(root, 0, value)]
    while stack:
        target, slot, item = stack.pop()
        if target is _LEAVE:
            on_path.discard(slot)
        elif isinstance(item, float):
            if not math.isfinite(item):
                target[slot] = None
        elif isinstance(item, (dict, list, tuple)):
            if id(item) in on_path:
                raise ValueError("Circular reference detected")
            on_path.add(id(item))
            stack.append((_LEAVE, id(item), None))
            if isinstance(item, dict):
                copy: Any = {}
                for key, child in item.items():
                    if isinstance(key, float) and not math.isfinite(key):
                        key = _NonFiniteKey(
                            "NaN" if math.isnan(key) else "Infinity" if key > 0 else "-Infinity"
                        )
                    copy[key] = child
                    stack.append((copy, key, child))
            else:
                copy = list(item)
                stack.extend((copy, index, child) for index, child in enumerate(item))
            target[slot] = copy
    return root[0]


def finite_json_dumps(value: Any) -> str:
    """Return ``value`` as strict JSON, with each NaN or infinite float as null.

    NaN and infinity are not JSON, so a strict parser such as JavaScript's
    JSON.parse refuses the bare tokens json.dumps writes for them. This
    writes null instead, as JSON.stringify and TraceEvent.to_json do.
    ``value`` is not changed.
    """
    try:
        return json.dumps(value, allow_nan=False)
    except ValueError:
        pass
    # Most values hold no NaN or infinity, so only a refused value is copied.
    return json.dumps(_finite_copy(value), allow_nan=False)


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

    def to_json(self, *, allow_nan: bool = False) -> str:
        """Return the event as one line of strict JSON.

        NaN and infinity are not JSON. A NaN or infinite float is written as
        null, as JavaScript's JSON.stringify does. A NaN or infinite
        ``duration_ms`` is left out, as a None one is. ``allow_nan=True``
        writes the bare NaN, Infinity and -Infinity tokens instead, so
        tag_session can keep the tokens older releases stored. The event is
        not changed.
        """
        d = asdict(self)
        d["event_type"] = self.event_type.value
        if not d.get("redacted"):
            d.pop("redacted", None)
        # drop None values
        d = {k: v for k, v in d.items() if v is not None and v != ""}
        d = _with_unknown_fields(self, d)
        try:
            return json.dumps(d, separators=(",", ":"), allow_nan=allow_nan)
        except ValueError:
            if allow_nan:
                raise
        # Most events hold no NaN or infinity, so only a refused event is copied.
        duration = d.get("duration_ms")
        if isinstance(duration, float) and not math.isfinite(duration):
            del d["duration_ms"]
        return json.dumps(_finite_copy(d), separators=(",", ":"), allow_nan=False)

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
