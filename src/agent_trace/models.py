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
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


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


@dataclass
class TraceEvent:
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    parent_id: str = ""  # links tool_result to tool_call, llm_response to llm_request
    duration_ms: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        # drop None values
        d = {k: v for k, v in d.items() if v is not None and v != ""}
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> TraceEvent:
        d = json.loads(line)
        d["event_type"] = EventType(d["event_type"])
        return cls(**d)


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

    def to_json(self) -> str:
        d = asdict(self)
        d = {k: v for k, v in d.items() if v is not None and v != "" and v != 0}
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> SessionMeta:
        return cls(**json.loads(text))
