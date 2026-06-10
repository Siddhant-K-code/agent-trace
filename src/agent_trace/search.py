"""Query language for the local session store (`agent-strace search`).

The `.agent-traces/` store is an append-only archive. Listing and replaying a
session both assume you already know which session you want. `search` closes
that gap with a small predicate language evaluated in-process over the existing
NDJSON store — no index, no database, no new dependencies (ADR-0002, ADR-0003).

Supported predicates:
  tool:<name>    session called a tool whose name contains <name>
  file:<path>    session read or wrote a file whose path contains <path>
  error:<text>   session has an error event; <text> matches its message
                 (empty <text> matches any error)
  cost:>N        estimated session cost is greater than N dollars
  cost:<N        estimated session cost is less than N dollars
  cost:N         estimated session cost is within 1 cent of N dollars
  date:<spec>    session started within a date window. <spec> is one of:
                 today, yesterday, this-week, this-month, Nd (last N days),
                 YYYY-MM-DD (that calendar day), YYYY-MM-DD:YYYY-MM-DD (range)

Predicates combine with AND / OR (case-insensitive). AND binds tighter than OR,
so `a AND b OR c` means `(a AND b) OR c`. Quote values that contain spaces:

  agent-strace search "tool:bash AND error:\"permission denied\""
  agent-strace search "file:src/auth.py OR cost:>2.00"
  agent-strace search "date:this-week AND tool:write_file"
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TextIO

from .cost import estimate_cost
from .models import EventType, TraceEvent
from .replay import C
from .store import TraceStore


# ---------------------------------------------------------------------------
# Query model
# ---------------------------------------------------------------------------

@dataclass
class Predicate:
    field: str   # tool | file | error | cost | date
    op: str      # "", ">", "<", "=" — only meaningful for cost
    value: str

    def needs_events(self) -> bool:
        return self.field in ("tool", "file", "error")


class QueryError(ValueError):
    """Raised when a query string cannot be parsed."""


# AND-group = list of predicates that must all match.
# A query is a list of AND-groups; the session matches if any group matches.
Query = list[list[Predicate]]


_VALID_FIELDS = {"tool", "file", "error", "cost", "date"}


def parse_query(text: str) -> Query:
    """Parse a query string into OR-groups of AND-ed predicates."""
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        raise QueryError(f"unbalanced quotes in query: {exc}") from exc

    if not tokens:
        raise QueryError("empty query")

    groups: Query = []
    current: list[Predicate] = []
    expecting_predicate = True

    for token in tokens:
        upper = token.upper()
        if upper == "AND":
            if expecting_predicate:
                raise QueryError("'AND' must follow a predicate")
            expecting_predicate = True
            continue
        if upper == "OR":
            if expecting_predicate:
                raise QueryError("'OR' must follow a predicate")
            groups.append(current)
            current = []
            expecting_predicate = True
            continue

        current.append(_parse_predicate(token))
        expecting_predicate = False

    if expecting_predicate:
        raise QueryError("query ends with a dangling AND/OR")

    groups.append(current)
    return groups


def _parse_predicate(token: str) -> Predicate:
    if ":" not in token:
        raise QueryError(
            f"'{token}' is not a predicate (expected field:value, "
            f"e.g. tool:bash)"
        )
    field, value = token.split(":", 1)
    field = field.lower()
    if field not in _VALID_FIELDS:
        raise QueryError(
            f"unknown field '{field}'. Valid fields: "
            f"{', '.join(sorted(_VALID_FIELDS))}"
        )

    op = ""
    if field == "cost":
        if value.startswith((">", "<", "=")):
            op, value = value[0], value[1:]
        else:
            op = "="
        try:
            float(value)
        except ValueError as exc:
            raise QueryError(
                f"cost value must be a number, got '{value}'"
            ) from exc

    return Predicate(field=field, op=op, value=value)


# ---------------------------------------------------------------------------
# Date windows
# ---------------------------------------------------------------------------

def _parse_date_window(spec: str, now: float | None = None) -> tuple[float, float]:
    """Return (start_ts, end_ts) for a date spec, in local time."""
    now = time.time() if now is None else now
    today = datetime.fromtimestamp(now).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    def _ts(dt: datetime) -> float:
        return dt.timestamp()

    spec = spec.strip().lower()

    if spec == "today":
        return _ts(today), now
    if spec == "yesterday":
        start = today - timedelta(days=1)
        return _ts(start), _ts(today)
    if spec in ("this-week", "this_week"):
        start = today - timedelta(days=today.weekday())  # Monday
        return _ts(start), now
    if spec in ("this-month", "this_month"):
        start = today.replace(day=1)
        return _ts(start), now
    if spec.endswith("d") and spec[:-1].isdigit():
        days = int(spec[:-1])
        return now - days * 86400, now

    # YYYY-MM-DD or YYYY-MM-DD:YYYY-MM-DD
    if ":" in spec:
        lo, hi = spec.split(":", 1)
        start = _parse_day(lo)
        end = _parse_day(hi) + timedelta(days=1)
        return _ts(start), _ts(end)

    day = _parse_day(spec)
    return _ts(day), _ts(day + timedelta(days=1))


def _parse_day(value: str) -> datetime:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError as exc:
        raise QueryError(
            f"invalid date '{value}' (expected YYYY-MM-DD or a keyword like "
            f"today, this-week, 7d)"
        ) from exc


# ---------------------------------------------------------------------------
# Event field extraction
# ---------------------------------------------------------------------------

def _tool_name(event: TraceEvent) -> str:
    return str(event.data.get("tool_name", "")).lower()


def _file_paths(event: TraceEvent) -> list[str]:
    """Return any file paths referenced by an event."""
    paths: list[str] = []
    args = event.data.get("arguments") or {}
    for key in ("file_path", "path"):
        val = args.get(key)
        if val:
            paths.append(str(val))
    direct = event.data.get("path")
    if direct:
        paths.append(str(direct))
    return paths


def _error_message(event: TraceEvent) -> str:
    return str(event.data.get("message", event.data.get("error", "")))


# ---------------------------------------------------------------------------
# Predicate evaluation
# ---------------------------------------------------------------------------

def _match_tool(events: list[TraceEvent], value: str) -> bool:
    needle = value.lower()
    return any(
        e.event_type == EventType.TOOL_CALL and needle in _tool_name(e)
        for e in events
    )


def _match_file(events: list[TraceEvent], value: str) -> bool:
    needle = value.lower()
    for e in events:
        for path in _file_paths(e):
            if needle in path.lower():
                return True
    return False


def _match_error(events: list[TraceEvent], value: str) -> bool:
    needle = value.lower()
    for e in events:
        if e.event_type != EventType.ERROR:
            continue
        if not needle:
            return True
        if needle in _error_message(e).lower():
            return True
    return False


def _match_cost(cost: float, pred: Predicate) -> bool:
    target = float(pred.value)
    if pred.op == ">":
        return cost > target
    if pred.op == "<":
        return cost < target
    return abs(cost - target) <= 0.01


def _match_date(started_at: float, pred: Predicate) -> bool:
    start, end = _parse_date_window(pred.value)
    return start <= started_at < end


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    session_id: str
    started_at: float
    cost: float
    tool_calls: int
    errors: int


def _query_needs_events(query: Query) -> bool:
    return any(p.needs_events() for group in query for p in group)


def _query_needs_cost(query: Query) -> bool:
    return any(p.field == "cost" for group in query for p in group)


def _evaluate_group(
    group: list[Predicate],
    *,
    started_at: float,
    cost: float | None,
    events: list[TraceEvent] | None,
) -> bool:
    for pred in group:
        if pred.field == "date":
            if not _match_date(started_at, pred):
                return False
        elif pred.field == "cost":
            if cost is None or not _match_cost(cost, pred):
                return False
        elif pred.field == "tool":
            if not _match_tool(events or [], pred.value):
                return False
        elif pred.field == "file":
            if not _match_file(events or [], pred.value):
                return False
        elif pred.field == "error":
            if not _match_error(events or [], pred.value):
                return False
    return True


def search_sessions(store: TraceStore, query: Query) -> list[SearchResult]:
    """Return sessions matching *query*, newest first."""
    need_events = _query_needs_events(query)
    need_cost = _query_needs_cost(query)

    results: list[SearchResult] = []
    for meta in store.list_sessions():
        events: list[TraceEvent] | None = None
        if need_events:
            try:
                events = store.load_events(meta.session_id)
            except (OSError, ValueError):
                events = []

        cost: float | None = None
        if need_cost:
            cost = _session_cost(store, meta.session_id)

        matched = any(
            _evaluate_group(
                group, started_at=meta.started_at, cost=cost, events=events
            )
            for group in query
        )
        if not matched:
            continue

        if cost is None:
            cost = _session_cost(store, meta.session_id)

        results.append(
            SearchResult(
                session_id=meta.session_id,
                started_at=meta.started_at,
                cost=cost,
                tool_calls=meta.tool_calls,
                errors=meta.errors,
            )
        )
    return results


def _session_cost(store: TraceStore, session_id: str) -> float:
    try:
        return estimate_cost(store, session_id).total_cost
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_results_text(
    results: list[SearchResult], out: TextIO = sys.stdout
) -> None:
    if not results:
        out.write("No matching sessions.\n")
        return

    out.write(f"\n{C.BOLD}Matching Sessions{C.RESET}\n")
    out.write(f"{C.GRAY}{'─' * 64}{C.RESET}\n")
    out.write(
        f"  {C.DIM}{'ID':<18} {'Started':<24} {'Cost':>9} {'Tools':>6}{C.RESET}\n"
    )
    out.write(f"{C.GRAY}{'─' * 64}{C.RESET}\n")

    for r in results:
        started = datetime.fromtimestamp(r.started_at).astimezone()
        started_str = started.strftime("%Y-%m-%d %H:%M:%S ") + started.strftime("%Z")
        out.write(
            f"  {r.session_id:<18} "
            f"{started_str:<24} "
            f"${r.cost:>8.4f} "
            f"{r.tool_calls:>6}\n"
        )

    out.write(f"{C.GRAY}{'─' * 64}{C.RESET}\n")
    out.write(f"  {len(results)} session(s)\n")
    out.write(
        f"  {C.DIM}Replay one with: agent-strace replay <id>{C.RESET}\n\n"
    )


def format_results_json(results: list[SearchResult]) -> str:
    return json.dumps(
        [
            {
                "session_id": r.session_id,
                "started_at": r.started_at,
                "cost": round(r.cost, 6),
                "tool_calls": r.tool_calls,
                "errors": r.errors,
            }
            for r in results
        ],
        indent=2,
    )


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> int:
    query_str = " ".join(args.query).strip() if args.query else ""
    if not query_str:
        sys.stderr.write(
            'Usage: agent-strace search "<query>"\n'
            "  e.g. agent-strace search \"tool:bash AND error:permission\"\n"
        )
        return 1

    try:
        query = parse_query(query_str)
    except QueryError as exc:
        sys.stderr.write(f"Invalid query: {exc}\n")
        return 1

    store = TraceStore(args.trace_dir)
    results = search_sessions(store, query)

    if getattr(args, "limit", 0):
        results = results[: args.limit]

    if getattr(args, "format", "text") == "json":
        sys.stdout.write(format_results_json(results) + "\n")
    else:
        format_results_text(results)

    # Exit 1 when nothing matched so the command composes in scripts.
    return 0 if results else 1
