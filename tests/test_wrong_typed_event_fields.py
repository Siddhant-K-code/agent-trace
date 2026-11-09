"""Commands on sessions where a field inside object event data has the wrong type.

``agent-strace server`` stores whatever a client posts, so an object ``data``
can hold a number where a command expects a tool name, a string where it
expects an arguments object, or a list where it expects a prompt. A command
reads such a field as missing. It behaves as if the key were absent and keeps
going, with no traceback and no new error line. A JSON number is an int or a
float but not a boolean, and a numeric string such as ``"12"`` is a string.

Each row sets one field on every event of one type in one session to a value
of the wrong type. It runs the command once with that value and once with the
key deleted. Both runs must exit with the same code, print the same error
lines on stderr and write the same files. For an ``exact`` field the two runs
must also match byte for byte after the temporary path, the current time and
SHA-256 hashes are normalized. The event hash chain covers the stored data, so
the hashes differ between the two runs. A ``loose`` field belongs to a command
that echoes raw event data or counts its stored size, such as share HTML or the
cost estimates, so only the checks above apply. ``sample`` copies whole sessions into the file it
writes, so for its rows only the path of that file is compared.

The first table holds the 49 variants first reported as printing a traceback
on main. ``share`` now writes a minimized packet unless ``--include-content``
is given, so its three variants there no longer reach the reads that crashed.
The second table holds variants found later that printed a traceback too,
including those ``share`` variants with ``--include-content``. The third holds
variants where main caught the exception and printed a failure line instead.
The last test class covers the readers outside the CLI commands: watch, the
MCP server, the dashboard routes, the compliance report, the eval scorer, the
optimize prompt and the stdio proxy.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import re
import tempfile
import threading
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

from agent_trace.compliance_report import _recorded_authorization, build_compliance_report
from agent_trace.eval.scorers import score_files_scoped
from agent_trace.mcp_server import _handle
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.optimize import _build_llm_prompt
from agent_trace.proxy import MCPProxy, _rpc_id
from agent_trace.server import _make_handler
from agent_trace.store import TraceStore
from agent_trace.timeline import _tool_label
from agent_trace.watch import NannyRule, OperationRule, WatcherConfig, watch_session

try:
    from . import test_non_object_event_data as base
except ImportError:  # unittest discover -s tests imports test modules top-level
    import test_non_object_event_data as base

BAD, GOOD, CHILD = base.BAD, base.GOOD, base.CHILD
POLICY, AGENTS_MD = base.POLICY, base.AGENTS_MD

# Stands for "delete the key" wherever a test takes a value to store.
_DELETE = object()


def _with(data: dict, key: str, value) -> dict:
    """Return a copy of *data* with *key* set to *value*, or deleted for _DELETE."""
    data = dict(data)
    if value is _DELETE:
        data.pop(key, None)
    else:
        data[key] = value
    return data


def _set_field(events: list[dict], selector, value) -> list[dict]:
    """Stand-in for the base fixture's ``_poison`` that changes one field.

    *selector* is ``(event_type, key)``. Every event of that type gets *value*
    under *key*, or loses the key when *value* is _DELETE.
    """
    if selector is None:
        return events
    event_type, key = selector
    return [
        {**event, "data": _with(event["data"], key, value)}
        if event["event_type"] == event_type else event
        for event in events
    ]


@dataclass
class Case:
    argv: tuple[str, ...]
    session: str = BAD
    tenant: bool = False
    exact: dict[str, list] = field(default_factory=dict)
    loose: dict[str, list] = field(default_factory=dict)
    # The command copies whole sessions into the file it writes, so the file
    # holds the stored value and only its path is compared.
    copies_events: bool = False

    def fields(self):
        for name, values in self.exact.items():
            yield name, values, True
        for name, values in self.loose.items():
            yield name, values, False


def _case(
    *argv: str,
    session: str = BAD,
    tenant: bool = False,
    copies_events: bool = False,
    exact=None,
    loose=None,
) -> Case:
    return Case(tuple(argv), session, tenant, exact or {}, loose or {}, copies_events)


USER_REPORTED_CASES = [
    _case("audit", BAD, exact={"tool_call.arguments": ["12"], "tool_call.tool_name": [1.5]}),
    _case(
        "audit", BAD, "--policy", POLICY,
        exact={"tool_call.arguments": ["12"], "tool_call.tool_name": [1.5]},
    ),
    _case(
        "audit", BAD, "--verify-chain",
        exact={"tool_call.arguments": ["12"], "tool_call.tool_name": [1.5]},
    ),
    _case("budget-report", loose={"tool_call.tool_name": [["x", "y"], None]}),
    _case("budget-report", "--format", "json", loose={"tool_call.tool_name": [["x", "y"]]}),
    _case("budget-report", "--format", "markdown", loose={"tool_call.tool_name": [["x", "y"]]}),
    _case("compliance", "export", "--since", "7d", loose={"tool_call.tool_name": [["x", "y"]]}),
    _case("compliance", "export", BAD, loose={"tool_call.tool_name": [["x", "y"]]}),
    _case(
        "compliance", "export", BAD, "--framework", "all", "--output", "cx.json",
        loose={"tool_call.tool_name": [["x", "y"]]},
    ),
    _case(
        "context-score", "--file", AGENTS_MD, "--min-sessions", "1",
        exact={"tool_call.arguments": ["12"]},
    ),
    _case(
        "cost", BAD,
        loose={
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "cost", BAD, "--csv",
        loose={
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "diff", BAD, GOOD,
        exact={
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "diff", GOOD, BAD,
        exact={
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "diff", GOOD, BAD, "--compare",
        loose={
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "diff", GOOD, BAD, "--semantic",
        exact={"user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}]},
        loose={"tool_call.arguments": ["12"], "tool_call.tool_name": [1.5]},
    ),
    _case(
        "explain", BAD,
        exact={
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "export", BAD, "--format", "otlp",
        exact={"assistant_response.text": [1.5], "tool_call.arguments": ["12"]},
    ),
    _case(
        "export", BAD, "--format", "otlp-genai",
        exact={
            "assistant_response.text": [1.5, {"k": "v"}],
            "user_prompt.prompt": [1.5, {"k": "v"}],
        },
        loose={"llm_response.model": [1.5], "tool_call.arguments": ["12"]},
    ),
    _case(
        "oncall", "--rotation-start", "2026-01-01",
        exact={
            "file_write.arguments": ["12"],
            "file_write.tool_name": [1.5],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
        },
    ),
    _case(
        "optimize", BAD, "--dry-run",
        exact={"file_write.path": [["x", "y"]]},
        loose={"tool_call.tool_name": [1.5]},
    ),
    _case(
        "policy", "generate", "--output", "gen.yaml",
        exact={"tool_call.arguments": ["12"], "tool_call.tool_name": [1.5]},
    ),
    _case(
        "policy", "generate", "--output", "gen.yaml", session=CHILD,
        exact={"tool_call.arguments": ["12"]},
    ),
    _case(
        "policy", "generate", BAD, "--dry-run",
        exact={"tool_call.arguments": ["12"], "tool_call.tool_name": [1.5]},
    ),
    _case(
        "postmortem", BAD,
        loose={
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "pr-comment", BAD, "--dry-run",
        exact={"user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}]},
        loose={"tool_call.arguments": ["12"], "tool_call.tool_name": [1.5]},
    ),
    _case(
        "replay", BAD,
        exact={
            "assistant_response.text": [1.5],
            "decision.reason": [1.5, {"k": "v"}],
            "session_start.command": [1.5],
            "tool_call.arguments": ["12", 1.5],
            "tool_call.tool_name": [1.5],
            "tool_result.content_types": [1.5],
            "user_prompt.prompt": [1.5, {"k": "v"}],
        },
    ),
    _case(
        "replay", BAD, "--expand-subagents",
        exact={
            "assistant_response.text": [1.5, {"k": "v"}],
            "error.error": [1.5, {"k": "v"}],
            "error.message": [1.5, {"k": "v"}],
            "subagent_start.agent_type": [1.5],
            "subagent_stop.agent_type": [1.5],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "tool_result.result": [1.5, ["x", "y"], {"k": "v"}],
            "user_prompt.prompt": [1.5, {"k": "v"}],
        },
    ),
    _case(
        "replay", BAD, "--expand-subagents", session=CHILD,
        exact={"tool_call.arguments": ["12"]},
    ),
    _case(
        "replay", BAD, "--filter", "tool_call,error",
        exact={"tool_call.arguments": ["12", 1.5], "tool_call.tool_name": [1.5]},
    ),
    _case(
        "replay", BAD, "--limit", "5",
        exact={"session_start.command": [1.5], "user_prompt.prompt": [1.5, {"k": "v"}]},
    ),
    _case(
        "sample", "--strategy", "diverse", "--n", "2", copies_events=True,
        exact={
            "llm_response.total_tokens": ["12"],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
        },
    ),
    _case(
        "sample", "--strategy", "random", "--n", "2", "--seed", "1", copies_events=True,
        exact={
            "llm_response.total_tokens": ["12"],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
        },
    ),
    _case(
        "sample", "--strategy", "recent", "--n", "2", copies_events=True,
        exact={
            "llm_response.total_tokens": ["12"],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
        },
    ),
    _case(
        "sample", "--strategy", "worst", "--n", "2", copies_events=True,
        exact={
            "llm_response.total_tokens": ["12"],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
        },
    ),
    _case(
        "sample", "--strategy", "worst", "--n", "2", "--output", "sample.jsonl", copies_events=True,
        exact={
            "llm_response.total_tokens": ["12"],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
        },
    ),
    _case(
        "share", BAD, "--output", "s.html",
        loose={
            "session_start.command": [[1, 2]],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "share", BAD, "--stdout",
        loose={
            "session_start.command": [[1, 2]],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "share", BAD, "--stdout", "--postmortem",
        loose={
            "session_start.command": [[1, 2]],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "standup", BAD, "--no-llm",
        loose={"tool_call.arguments": ["12", None], "tool_call.tool_name": [1.5]},
    ),
    _case(
        "stats", BAD,
        exact={"error.message": [1.5, {"k": "v"}], "tool_call.tool_name": [["x", "y"], None]},
    ),
    _case(
        "tenant", "report",
        exact={
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "timeline", BAD,
        exact={
            "decision.reason": [1.5, {"k": "v"}],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
        loose={
            "assistant_response.text": [1.5, ["x", "y"], {"k": "v"}],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
        },
    ),
    _case(
        "timeline", BAD, "--format", "json",
        exact={"decision.reason": [1.5, {"k": "v"}]},
        loose={
            "assistant_response.text": [1.5, ["x", "y"], {"k": "v"}],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "token-budget", BAD,
        exact={"llm_request.model": [1.5], "llm_response.output_tokens": ["12", ["x", "y"]]},
        loose={"llm_request.input_tokens": ["12"]},
    ),
    _case(
        "why", BAD, "6",
        exact={"user_prompt.prompt": [1.5, {"k": "v"}]},
        loose={"tool_call.tool_name": [1.5]},
    ),
    _case(
        "why", BAD, "8",
        exact={"user_prompt.prompt": [1.5, {"k": "v"}]},
        loose={"tool_call.tool_name": [1.5]},
    ),
    _case("why", BAD, "10", exact={"user_prompt.prompt": [1.5, {"k": "v"}]}),
    _case(
        "why", BAD, "14",
        exact={"user_prompt.prompt": [1.5, {"k": "v"}]},
        loose={"tool_call.tool_name": [1.5]},
    ),
    _case(
        "why", BAD, "18",
        exact={"user_prompt.prompt": [1.5, {"k": "v"}]},
        loose={"tool_call.tool_name": [1.5]},
    ),
    _case("why", BAD, "25", exact={"user_prompt.prompt": [1.5, {"k": "v"}]}),
]


OTHER_CASES = [
    _case(
        "audit", CHILD, session=CHILD,
        exact={"tool_call.arguments": ["12"], "tool_call.tool_name": [42]},
    ),
    _case(
        "context-score", "--file", AGENTS_MD, "--min-sessions", "1", "--compare",
        exact={"tool_call.arguments": [42]},
    ),
    _case(
        "context-score", "--file", AGENTS_MD, "--min-sessions", "1", "--format", "json",
        loose={"tool_call.arguments": [42]},
    ),
    _case("cost", "--since", "7d", session=CHILD, loose={"tool_call.tool_name": [42]}),
    _case("cost", "--tenant", "acme", tenant=True, loose={"tool_call.tool_name": [42]}),
    _case("cost", BAD, "--tenant", "acme", tenant=True, loose={"tool_call.tool_name": [42]}),
    _case("diff", BAD, CHILD, exact={"tool_call.tool_name": [42]}),
    _case(
        "eval", "dataset", "auto", "--dataset", "ds.jsonl", "--filter", "wide-blast",
        exact={"file_write.path": [["x", "y"]]},
    ),
    _case(
        "explain", session=CHILD,
        exact={"tool_call.arguments": ["12"], "tool_call.tool_name": [42]},
    ),
    _case("explain", CHILD, session=CHILD, exact={"tool_call.tool_name": [42]}),
    _case(
        "export", BAD, "--format", "otlp", "--service-name", "x",
        exact={"tool_call.arguments": [42]},
    ),
    _case("fingerprint", "--sessions", "5", exact={"tool_call.tool_name": [None]}),
    _case("inflation", loose={"llm_request.messages": [1.5, 42]}),
    _case("inflation", "--sessions", "2", loose={"llm_request.messages": [42]}),
    _case("inflation", "--sessions", "3", loose={"llm_request.messages": [42]}),
    _case("optimize", BAD, "--dry-run", "--target", AGENTS_MD, loose={"tool_call.tool_name": [42]}),
    _case("org-report", exact={"tool_call.tool_name": [42]}),
    _case(
        "org-report", "--format", "html", "--output", "o.html",
        exact={"tool_call.tool_name": [42]},
    ),
    _case("org-report", "--format", "json", loose={"tool_call.tool_name": [42]}),
    _case("policy", "backtest", "--policy", POLICY, exact={"tool_call.tool_name": [42]}),
    _case(
        "policy", "backtest", "--policy", POLICY, "--show-sessions", "--format", "json",
        exact={"tool_call.tool_name": [42]},
    ),
    _case(
        "policy", "coverage", "--policy", POLICY, "--format", "json",
        exact={"tool_call.tool_name": [42]},
    ),
    _case(
        "policy", "coverage", "--policy", POLICY, "--show-uncovered",
        exact={"tool_call.tool_name": [42]},
    ),
    _case("policy", "diff", POLICY, POLICY, exact={"tool_call.tool_name": [42]}),
    _case("postmortem", BAD, "--agents-md", AGENTS_MD, loose={"tool_call.tool_name": [42]}),
    _case(
        "pr-comment", BAD, "--dry-run", "--review",
        exact={"user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}]},
        loose={"tool_call.arguments": ["12"], "tool_call.tool_name": [1.5]},
    ),
    _case(
        "replay", session=CHILD,
        exact={"tool_call.arguments": [1.5], "tool_call.tool_name": [42]},
    ),
    _case(
        "replay", BAD, "--expand-subagents", "--format", "html", "--output", "rx.html",
        exact={"tool_call.tool_name": [42]},
    ),
    _case("replay", CHILD, session=CHILD, exact={"tool_call.tool_name": [42]}),
    _case(
        "share", BAD, "--output", "s.html", "--include-content",
        loose={
            "session_start.command": [[1, 2]],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "share", BAD, "--stdout", "--include-content",
        loose={
            "session_start.command": [[1, 2]],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "share", BAD, "--stdout", "--include-content", "--postmortem",
        loose={
            "session_start.command": [[1, 2]],
            "tool_call.arguments": ["12"],
            "tool_call.tool_name": [1.5],
            "user_prompt.prompt": [1.5, ["x", "y"], {"k": "v"}],
        },
    ),
    _case(
        "share", "--stdout", "--include-content", session=CHILD,
        loose={"tool_call.tool_name": [42]},
    ),
    _case(
        "share", CHILD, "--stdout", "--include-content", session=CHILD,
        loose={"tool_call.tool_name": [42]},
    ),
    _case("standup", session=CHILD, exact={"tool_call.tool_name": [42]}),
    _case("standup", "--no-llm", session=CHILD, exact={"tool_call.tool_name": [42]}),
    _case("standup", BAD, loose={"tool_call.tool_name": [42]}),
    _case("standup", CHILD, "--no-llm", session=CHILD, exact={"tool_call.tool_name": [42]}),
    _case("stats", CHILD, session=CHILD, exact={"tool_call.tool_name": [["x", "y"]]}),
    _case("tenant", "report", tenant=True, exact={"tool_call.tool_name": [42]}),
    _case("tenant", "report", "--format", "json", tenant=True, loose={"tool_call.tool_name": [42]}),
    _case("timeline", CHILD, session=CHILD, exact={"tool_call.tool_name": [42]}),
    _case(
        "timeline", CHILD, "--format", "json", session=CHILD,
        loose={"tool_call.tool_name": [42]},
    ),
    _case("why", BAD, "2", exact={"user_prompt.prompt": [42]}),
    _case("why", CHILD, "2", session=CHILD, exact={"user_prompt.prompt": [42]}),
    _case("why", BAD, "3", exact={"user_prompt.prompt": [42]}),
    _case("why", CHILD, "3", session=CHILD, exact={"user_prompt.prompt": [42]}),
    _case("why", BAD, "4", exact={"user_prompt.prompt": [42]}),
    _case("why", CHILD, "4", session=CHILD, exact={"user_prompt.prompt": [42]}),
    _case("why", BAD, "5", exact={"user_prompt.prompt": [42]}),
    _case("why", CHILD, "5", session=CHILD, exact={"user_prompt.prompt": [42]}),
    _case("why", CHILD, "6", session=CHILD, exact={"user_prompt.prompt": [42]}),
    _case("why", BAD, "7", loose={"tool_call.tool_name": [42]}),
    _case("why", CHILD, "7", session=CHILD, exact={"user_prompt.prompt": [42]}),
    _case(
        "why", BAD, "9",
        exact={"error.error": [1.5, {"k": "v"}]},
        loose={"tool_call.tool_name": [42]},
    ),
    _case("why", BAD, "11", loose={"tool_call.tool_name": [1.5, 42]}),
    _case("why", BAD, "12", loose={"tool_call.tool_name": [42]}),
    _case("why", BAD, "13", loose={"tool_call.tool_name": [42]}),
    _case("why", BAD, "15", loose={"tool_call.tool_name": [42]}),
    _case("why", BAD, "16", loose={"tool_call.tool_name": [42]}),
    _case("why", BAD, "17", loose={"tool_call.tool_name": [42]}),
    _case("why", BAD, "19", loose={"tool_call.tool_name": [42]}),
    _case(
        "why", BAD, "20",
        exact={"tool_call.arguments": ["12"]},
        loose={"tool_call.tool_name": [42]},
    ),
    _case("why", BAD, "21", loose={"tool_call.tool_name": [42]}),
    _case("why", BAD, "22", exact={"user_prompt.prompt": [42]}),
    _case("why", BAD, "23", exact={"user_prompt.prompt": [42]}),
    _case(
        "why", BAD, "24",
        exact={"assistant_response.text": [1.5, {"k": "v"}], "user_prompt.prompt": [42]},
    ),
]


FAILED_LINE_CASES = [
    _case(
        "compare", GOOD, BAD,
        loose={
            "tool_call.tool_name": [42],
            "tool_call.arguments": ["text"],
            "user_prompt.prompt": [{"k": "v"}],
        },
    ),
    _case("compare", GOOD, BAD, "--format", "json", loose={"tool_call.tool_name": [42]}),
    _case("compare", "--tag", "claude", "--last", "2", exact={"user_prompt.prompt": [42]}),
    _case(
        "lint", BAD,
        exact={
            "session_start.budget_dollars": ["12"],
            "session_start.max_cost_dollars": [["x", "y"]],
            "tool_call.tool_name": [42],
            "file_read.arguments": ["text"],
            "file_read.tool_name": [42],
        },
    ),
    _case("lint", "--all", session=CHILD, exact={"tool_call.tool_name": [None]}),
    _case("lint", BAD, "--strict", exact={"tool_call.arguments": [1.5]}),
    _case("lint", BAD, "--format", "json", exact={"file_read.tool_name": [{"k": "v"}]}),
    _case(
        "context-score", "--file", AGENTS_MD, "--min-sessions", "1",
        exact={"session_start.budget_dollars": ["text"]},
    ),
    _case("org-report", exact={"session_start.max_cost_dollars": [{"k": "v"}]}),
    _case("pr-comment", BAD, "--dry-run", exact={"file_read.arguments": [42]}),
]

ALL_CASES = USER_REPORTED_CASES + OTHER_CASES + FAILED_LINE_CASES

_ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\.\d{3,}Z?")
_EPOCH_TIME = re.compile(r"\b1[6-9]\d{8}\.\d{3,}\b")
# The event hash chain covers the stored data, so a run with a wrong-typed value
# and a run with the key deleted print different hashes.
_SHA256 = re.compile(r"\b[0-9a-f]{64}\b")
_ERROR_LINE = re.compile(r"Traceback|Error|Failed|failed")


def _normalize(text: str, root: Path) -> str:
    text = text.replace(str(root), "<ROOT>").replace(root.name, "<ROOT>")
    text = _EPOCH_TIME.sub("<NOW>", _ISO_TIME.sub("<TIME>", text))
    return _SHA256.sub("<SHA256>", text)


def _files(root: Path) -> dict[str, str]:
    """Map every file under *root* to its normalized contents."""
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.replace(str(root).encode(), b"<ROOT>").hex()
            files[str(path.relative_to(root))] = _normalize(text, root)
    return files


@dataclass
class Run:
    code: int
    out: str
    err: str
    written: dict[str, str]
    removed: list[str]
    network: list
    browser: list

    @property
    def error_lines(self) -> list[str]:
        return [line for line in self.err.splitlines() if _ERROR_LINE.search(line)]


class TestWrongTypedFieldFixture(unittest.TestCase):
    def test_every_row_changes_an_event_type_the_session_has(self):
        for case in ALL_CASES:
            layout = base.CHILD_LAYOUT if case.session == CHILD else base.BAD_LAYOUT
            for name, values, _exact in case.fields():
                with self.subTest(argv=" ".join(case.argv), field=name):
                    self.assertIn(name.split(".")[0], layout)
                    self.assertTrue(values)

    def test_user_reported_rows_cover_the_49_crashing_variants(self):
        self.assertEqual(len({case.argv for case in USER_REPORTED_CASES}), 49)

    def test_set_field_changes_only_the_chosen_key_of_the_chosen_type(self):
        events = base._session_events(BAD, base.NOW, "bad", child_id=CHILD)
        changed = _set_field(events, ("tool_call", "tool_name"), 42)
        deleted = _set_field(events, ("tool_call", "tool_name"), _DELETE)
        self.assertEqual(sum(event["event_type"] == "tool_call" for event in events), 6)
        for before, after, gone in zip(events, changed, deleted):
            if before["event_type"] == "tool_call":
                self.assertEqual(after["data"], {**before["data"], "tool_name": 42})
                self.assertEqual(gone["data"], _with(before["data"], "tool_name", _DELETE))
                self.assertIn("tool_name", before["data"])
            else:
                self.assertEqual(after, before)
                self.assertEqual(gone, before)
        self.assertIs(_set_field(events, None, 42), events)


class TestCommandsReadWrongTypedFieldsAsMissing(base.CliTestCase):
    """Each wrong-typed run must match the run with the key deleted."""

    def _workspace(self, case: Case, name: str, value) -> Path:
        selector = tuple(name.split("."))
        target = {"child_type": selector} if case.session == CHILD else {"bad_type": selector}
        with mock.patch.object(base, "_poison", _set_field):
            return self.make_workspace(value=value, tenant=case.tenant, **target)

    def _run(self, case: Case, name: str, value) -> Run:
        root = self._workspace(case, name, value)
        before = _files(root)
        result = self.run_cli(root, *case.argv)
        after = _files(root)
        return Run(
            code=result.code,
            out=_normalize(result.out, root),
            err=_normalize(result.err, root),
            written={path: text for path, text in after.items() if before.get(path) != text},
            removed=sorted(path for path in before if path not in after),
            network=result.network,
            browser=result.browser,
        )

    def _check_cases(self, cases: list[Case]) -> None:
        for case in cases:
            for name, values, exact in case.fields():
                argv = " ".join(case.argv)
                with self.subTest(argv=argv, session=case.session, field=name, value="deleted"):
                    missing = self._run(case, name, _DELETE)
                for value in values:
                    with self.subTest(argv=argv, session=case.session, field=name, value=value):
                        wrong = self._run(case, name, value)
                        self.assertNotIn("Traceback", wrong.out + wrong.err)
                        self.assertEqual(wrong.code, missing.code)
                        self.assertEqual(wrong.error_lines, missing.error_lines)
                        self.assertEqual(sorted(wrong.written), sorted(missing.written))
                        self.assertEqual(wrong.removed, missing.removed)
                        self.assertEqual(wrong.network, missing.network)
                        self.assertEqual(wrong.browser, missing.browser)
                        if exact:
                            self.assertEqual(wrong.out, missing.out)
                            self.assertEqual(wrong.err, missing.err)
                            if not case.copies_events:
                                self.assertEqual(wrong.written, missing.written)

    def test_variants_that_printed_a_traceback_on_main(self):
        self._check_cases(USER_REPORTED_CASES)

    def test_other_variants_that_printed_a_traceback_on_main(self):
        self._check_cases(OTHER_CASES)

    def test_variants_that_printed_a_failure_line_on_main(self):
        self._check_cases(FAILED_LINE_CASES)


_WATCH_EVENTS = [
    (EventType.LLM_REQUEST, {"model": "claude-sonnet-4", "input_tokens": 10}),
    (EventType.TOOL_CALL, {"tool_name": "Bash", "arguments": {"command": "pytest -q"}}),
    (EventType.TOOL_CALL, {"tool_name": "Bash", "arguments": {"command": "pytest -q"}}),
    (EventType.TOOL_CALL, {"tool_name": "Bash", "arguments": {"command": "pytest -q"}}),
    (EventType.TOOL_CALL, {"tool_name": "Write", "arguments": {"file_path": "/etc/hosts"}}),
    (EventType.SESSION_END, {"exit_code": 0}),
]

_AUTHORIZATION = {
    "schema": "agent-strace-authorization/v1",
    "decision": "allowed",
    "mode": "pre_action",
    "evaluated_at": 9.0,
    "policy_sha256": "sha256:" + "a" * 64,
    "rule_id": "allow-shell",
}


class TestOtherReadersTreatWrongTypedFieldsAsMissing(unittest.TestCase):
    def _tempdir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name).resolve()

    def _watch(self, event_type: EventType | None = None, key: str = "", value=_DELETE):
        root = self._tempdir()
        store = TraceStore(str(root / "traces"), redact=False)
        store.create_session(SessionMeta(session_id="live-session", started_at=1.0))
        events = [
            base._live_event(kind, _with(data, key, value) if kind == event_type else data, 2.0 + i)
            for i, (kind, data) in enumerate(_WATCH_EVENTS)
        ]
        config = WatcherConfig(
            max_retries=2,
            loop_threshold=3,
            on_violation="terminal",
            operation_rules=[OperationRule(tool_name="*", pattern="*", action="alert")],
        )
        rules = [NannyRule(name="etc", condition='file_path matches "/etc/*"', action="alert")]
        alerts: list[str] = []
        out = io.StringIO()
        with base._chdir(root), base._isolated_env(root), \
                mock.patch("agent_trace.watch._tail_events", return_value=iter(events)), \
                mock.patch("agent_trace.watch._alert_terminal", alerts.append):
            watch_session(store, "live-session", config, out=out, nanny_rules=rules)
        # The running cost estimate counts the stored size of each event.
        return alerts, re.sub(r"\$\d+\.\d{4}", "$<COST>", out.getvalue())

    def test_watch_alerts_match_the_run_without_the_field(self):
        alerts, _ = self._watch()
        for prefix in ("RetryWatcher:", "LoopWatcher:", "OperationWatcher:", "NannyRule 'etc'"):
            self.assertTrue(any(alert.startswith(prefix) for alert in alerts), (prefix, alerts))
        rows = [
            (EventType.TOOL_CALL, "tool_name", [42, None, ["x"], {"k": "v"}, True]),
            (EventType.TOOL_CALL, "arguments", ["text", 42, ["x"], None]),
            (EventType.LLM_REQUEST, "model", [42, ["x"]]),
            (EventType.LLM_REQUEST, "input_tokens", ["12", True]),
        ]
        for event_type, key, values in rows:
            missing = self._watch(event_type, key)
            for value in values:
                with self.subTest(event_type=event_type.value, field=key, value=value):
                    self.assertEqual(self._watch(event_type, key, value), missing)

    def _mcp_call(self, tool: str, arguments: dict, name: str, value) -> dict:
        traces = self._tempdir() / "traces"
        with mock.patch.object(base, "_poison", _set_field):
            base.build_store(traces, bad_type=tuple(name.split(".")), value=value)
        request = {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        return _handle(TraceStore(str(traces)), request)

    def _check_mcp_tool(self, tool: str, arguments: dict, rows) -> None:
        for name, values in rows:
            missing = self._mcp_call(tool, arguments, name, _DELETE)
            self.assertIs(missing["result"]["isError"], False)
            for value in values:
                with self.subTest(tool=tool, field=name, value=value):
                    self.assertEqual(self._mcp_call(tool, arguments, name, value), missing)

    def test_mcp_diff_sessions_matches_the_run_without_the_field(self):
        self._check_mcp_tool("diff_sessions", {"session_a": GOOD, "session_b": BAD}, [
            ("tool_call.tool_name", [42, ["x"], None]),
            ("file_read.path", [["x"], 42]),
            ("file_write.path", [{"k": "v"}]),
        ])

    def test_mcp_session_summary_matches_the_run_without_the_field(self):
        self._check_mcp_tool("get_session_summary", {"session_id": BAD}, [
            ("tool_call.tool_name", [42, None]),
            ("tool_call.arguments", ["12", ["x"]]),
            ("user_prompt.prompt", [42, ["x", "y"], {"k": "v"}]),
        ])

    def _dashboard_get(self, path: str, name: str, value) -> tuple[int, str]:
        traces = self._tempdir() / "traces"
        with mock.patch.object(base, "_poison", _set_field):
            base.build_store(traces, bad_type=tuple(name.split(".")), value=value)
        handler = _make_handler(TraceStore(str(traces)), threading.Lock(), auth_key="", dashboard=True)
        server = HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)
        try:
            # The handler writes an access log line to stderr for each request.
            with contextlib.redirect_stderr(io.StringIO()):
                connection.request("GET", path)
                response = connection.getresponse()
                return response.status, response.read().decode("utf-8")
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def _dashboard_body(self, path: str, name: str, value):
        status, body = self._dashboard_get(path, name, value)
        self.assertEqual(status, 200, body)
        if "/compare?" not in path:
            return body
        report = json.loads(body)
        # The cost estimate counts the stored size of each event, and the
        # verdict compares the two costs.
        for key in ("cost_a", "cost_b", "verdict"):
            del report[key]
        return report

    def test_dashboard_compare_and_otlp_routes_match_the_run_without_the_field(self):
        compare = f"/api/sessions/{GOOD}/compare?other={BAD}"
        otlp = f"/api/sessions/{BAD}/otlp"
        # On main the first row of each route raised AttributeError, which
        # dropped the connection, and the other rows returned 400.
        rows = [
            (compare, "tool_call.tool_name", [42]),
            (compare, "user_prompt.prompt", [42, {"k": "v"}]),
            (otlp, "tool_call.arguments", [42]),
            (otlp, "assistant_response.text", [42]),
        ]
        for path, name, values in rows:
            missing = self._dashboard_body(path, name, _DELETE)
            for value in values:
                with self.subTest(path=path, field=name, value=value):
                    self.assertEqual(self._dashboard_body(path, name, value), missing)

    def _authorization_timeline(self, decision) -> list:
        directory = self._tempdir()
        store = TraceStore(str(directory), use_workspace_env=False, redact=False)
        store.create_session(SessionMeta(session_id="session", started_at=1))
        store.append_event("session", TraceEvent(
            event_type=EventType.TOOL_CALL, timestamp=10, event_id="call", session_id="session",
            data={
                "tool_name": "bash",
                "arguments": {"command": "true"},
                "agent_trace_authorization": _with(_AUTHORIZATION, "decision", decision),
            },
        ))
        report = build_compliance_report(
            store, "owasp-agentic", since="1970-01-01T00:00:05Z", until="1970-01-01T00:01:00Z",
            generated_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
        )
        return report["authorization_timeline"]

    def test_compliance_report_reads_a_wrong_typed_decision_as_missing(self):
        missing = self._authorization_timeline(_DELETE)
        self.assertNotEqual(self._authorization_timeline("allowed"), missing)
        for value in (["allowed"], {"k": "v"}, 42, None):
            with self.subTest(value=value):
                self.assertEqual(self._authorization_timeline(value), missing)
                event = TraceEvent(EventType.TOOL_CALL, data={
                    "agent_trace_authorization": _with(_AUTHORIZATION, "decision", value),
                })
                self.assertEqual(
                    _recorded_authorization(event), ("ambiguous_record", "invalid", None),
                )

    def test_files_scoped_scorer_reads_wrong_typed_fields_as_missing(self):
        calls = [
            {"tool_name": "Read", "arguments": {"file_path": "src/app.py"}},
            {"tool_name": "Write", "arguments": {"file_path": "/etc/hosts"}},
            {"tool_name": "Edit", "arguments": {"file_path": "docs/notes.md"}},
        ]

        def score(key: str, value):
            events = [
                TraceEvent(EventType.TOOL_CALL, data=_with(data, key, value) if i == 1 else data)
                for i, data in enumerate(calls)
            ]
            return score_files_scoped(events, ["src/"])

        for key, values in (("tool_name", [42, ["x"], None]), ("arguments", ["text", ["x"], 42])):
            missing = score(key, _DELETE)
            for value in values:
                with self.subTest(field=key, value=value):
                    self.assertEqual(score(key, value), missing)

    def test_optimize_llm_prompt_reads_a_wrong_typed_error_message_as_missing(self):
        def prompt(value) -> str:
            events = [
                TraceEvent(EventType.TOOL_CALL, data={"tool_name": "Bash", "arguments": {}}),
                TraceEvent(EventType.ERROR, data=_with({"message": "boom"}, "message", value)),
                TraceEvent(EventType.ERROR, data={"message": "second"}),
            ]
            return _build_llm_prompt({"a1b2c3d4e5f60001": events}, "AGENTS.md", "")

        missing = prompt(_DELETE)
        for value in (42, ["x"], None, {"k": "v"}):
            with self.subTest(value=value):
                self.assertEqual(prompt(value), missing)

    def test_timeline_reads_a_wrong_typed_url_or_query_as_missing(self):
        for tool, key in (("WebFetch", "url"), ("WebSearch", "query")):
            def label(value):
                arguments = _with({key: "x"}, key, value)
                return _tool_label(TraceEvent(
                    EventType.TOOL_CALL, data={"tool_name": tool, "arguments": arguments},
                ))

            missing = label(_DELETE)
            for value in (42, None, ["x"], {"k": "v"}, True):
                with self.subTest(tool=tool, value=value):
                    self.assertEqual(label(value), missing)

    def _proxy_events(self, request_id) -> list[TraceEvent]:
        store = TraceStore(str(self._tempdir()), redact=False)
        meta = SessionMeta()
        store.create_session(meta)
        proxy = MCPProxy(["true"], store, meta)
        call = {
            "jsonrpc": "2.0", "id": request_id, "method": "tools/call",
            "params": {"name": "read_file", "arguments": {}},
        }
        result = {
            "jsonrpc": "2.0", "id": request_id,
            "result": {"content": [{"type": "text", "text": "ok"}]},
        }
        for message, direction in ((call, "agent_to_server"), (result, "server_to_agent")):
            source = io.BytesIO((json.dumps(message) + "\n").encode("utf-8"))
            dest = io.BytesIO()
            proxy._forward_and_trace(source, dest, direction, threading.Event())
            self.assertEqual(json.loads(dest.getvalue()), message)
        return store.load_events(meta.session_id)

    def test_proxy_links_only_string_and_number_request_ids(self):
        for request_id, linked in (
            (7, True), (0, True), ("a", True), (1.5, True),
            ([1], False), ({"k": 1}, False), (True, False),
        ):
            with self.subTest(request_id=request_id):
                call, result = self._proxy_events(request_id)
                self.assertEqual(
                    (call.event_type, result.event_type),
                    (EventType.TOOL_CALL, EventType.TOOL_RESULT),
                )
                self.assertEqual(result.parent_id, call.event_id if linked else "")

    def test_rpc_id_reads_ids_that_are_not_strings_or_numbers_as_missing(self):
        for value, expected in (
            (7, 7), (0, 0), (1.5, 1.5), ("a", "a"), ("", ""),
            (True, None), (False, None), ([1], None), ({"k": 1}, None), (None, None),
        ):
            with self.subTest(value=value):
                self.assertEqual(_rpc_id({"request_id": value}), expected)
        self.assertIsNone(_rpc_id({}))


if __name__ == "__main__":
    unittest.main()
