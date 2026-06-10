"""Tests for agent-strace search (Issue #172)."""

from __future__ import annotations

import argparse
import io
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.search import (
    Predicate,
    QueryError,
    _parse_date_window,
    cmd_search,
    format_results_json,
    parse_query,
    search_sessions,
)
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> tuple[TraceStore, str]:
    tmp = tempfile.mkdtemp()
    return TraceStore(Path(tmp)), tmp


def _add_session(
    store: TraceStore,
    *,
    started_at: float | None = None,
    tools: list[str] | None = None,
    files: list[str] | None = None,
    errors: list[str] | None = None,
    big_payload: bool = False,
) -> str:
    started_at = time.time() if started_at is None else started_at
    meta = SessionMeta(agent_name="test", command="test")
    sp = store.create_session(meta)
    sid = sp.name

    meta = store.load_meta(sid)
    meta.started_at = started_at

    events: list[TraceEvent] = [
        TraceEvent(event_type=EventType.SESSION_START, timestamp=started_at,
                   session_id=sid, data={}),
    ]
    for tool in (tools or []):
        data = {"tool_name": tool}
        if big_payload:
            data["arguments"] = {"content": "x" * 4000}
        events.append(TraceEvent(event_type=EventType.TOOL_CALL,
                                 timestamp=started_at + 1, session_id=sid, data=data))
        meta.tool_calls += 1
    for path in (files or []):
        events.append(TraceEvent(
            event_type=EventType.TOOL_CALL, timestamp=started_at + 2, session_id=sid,
            data={"tool_name": "write", "arguments": {"file_path": path}}))
        meta.tool_calls += 1
    for msg in (errors or []):
        events.append(TraceEvent(event_type=EventType.ERROR, timestamp=started_at + 3,
                                 session_id=sid, data={"message": msg}))
        meta.errors += 1

    for e in events:
        store.append_event(sid, e)
    store.update_meta(meta)
    return sid


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

class TestParseQuery(unittest.TestCase):
    def test_single_predicate(self):
        q = parse_query("tool:bash")
        self.assertEqual(q, [[Predicate("tool", "", "bash")]])

    def test_and_groups_into_one_or_group(self):
        q = parse_query("tool:bash AND error:permission")
        self.assertEqual(len(q), 1)
        self.assertEqual(len(q[0]), 2)

    def test_or_splits_groups(self):
        q = parse_query("tool:bash OR cost:>2")
        self.assertEqual(len(q), 2)

    def test_and_binds_tighter_than_or(self):
        q = parse_query("tool:a AND tool:b OR tool:c")
        self.assertEqual([len(g) for g in q], [2, 1])

    def test_case_insensitive_operators(self):
        self.assertEqual(len(parse_query("tool:a and tool:b")[0]), 2)

    def test_quoted_value_with_spaces(self):
        q = parse_query('error:"permission denied"')
        self.assertEqual(q[0][0].value, "permission denied")

    def test_cost_operator_parsing(self):
        self.assertEqual(parse_query("cost:>2")[0][0].op, ">")
        self.assertEqual(parse_query("cost:<2")[0][0].op, "<")
        self.assertEqual(parse_query("cost:2")[0][0].op, "=")

    def test_rejects_empty(self):
        with self.assertRaises(QueryError):
            parse_query("   ")

    def test_rejects_unknown_field(self):
        with self.assertRaises(QueryError):
            parse_query("bogus:x")

    def test_rejects_non_predicate(self):
        with self.assertRaises(QueryError):
            parse_query("justtext")

    def test_rejects_dangling_operator(self):
        with self.assertRaises(QueryError):
            parse_query("tool:a AND")

    def test_rejects_leading_operator(self):
        with self.assertRaises(QueryError):
            parse_query("AND tool:a")

    def test_rejects_non_numeric_cost(self):
        with self.assertRaises(QueryError):
            parse_query("cost:>abc")


# ---------------------------------------------------------------------------
# Date windows
# ---------------------------------------------------------------------------

class TestDateWindow(unittest.TestCase):
    def setUp(self):
        # Fixed "now": Wednesday 2026-06-10 12:00 local
        self.now = datetime(2026, 6, 10, 12, 0, 0).timestamp()

    def test_today(self):
        start, end = _parse_date_window("today", self.now)
        self.assertEqual(datetime.fromtimestamp(start).hour, 0)
        self.assertEqual(end, self.now)

    def test_yesterday(self):
        start, end = _parse_date_window("yesterday", self.now)
        self.assertAlmostEqual(end - start, 86400, delta=3600)  # tolerate DST

    def test_this_week_starts_monday(self):
        start, _ = _parse_date_window("this-week", self.now)
        self.assertEqual(datetime.fromtimestamp(start).weekday(), 0)

    def test_n_days(self):
        start, end = _parse_date_window("7d", self.now)
        self.assertAlmostEqual(end - start, 7 * 86400, delta=1)

    def test_explicit_day(self):
        start, end = _parse_date_window("2026-06-01", self.now)
        self.assertEqual(datetime.fromtimestamp(start).day, 1)
        self.assertAlmostEqual(end - start, 86400, delta=3600)

    def test_range(self):
        start, end = _parse_date_window("2026-06-01:2026-06-03", self.now)
        self.assertAlmostEqual(end - start, 3 * 86400, delta=3600)

    def test_invalid_date_raises(self):
        with self.assertRaises(QueryError):
            _parse_date_window("not-a-date", self.now)


# ---------------------------------------------------------------------------
# Search evaluation
# ---------------------------------------------------------------------------

class TestSearch(unittest.TestCase):
    def setUp(self):
        self.store, self.tmp = _make_store()

    def _ids(self, query: str) -> set[str]:
        return {r.session_id for r in search_sessions(self.store, parse_query(query))}

    def test_tool_predicate(self):
        a = _add_session(self.store, tools=["bash"])
        _add_session(self.store, tools=["read"])
        self.assertEqual(self._ids("tool:bash"), {a})

    def test_tool_substring_match(self):
        a = _add_session(self.store, tools=["write_file"])
        self.assertEqual(self._ids("tool:write"), {a})

    def test_file_predicate(self):
        a = _add_session(self.store, files=["src/auth.py"])
        _add_session(self.store, files=["src/main.py"])
        self.assertEqual(self._ids("file:auth.py"), {a})

    def test_error_with_text(self):
        a = _add_session(self.store, errors=["permission denied"])
        _add_session(self.store, errors=["file not found"])
        self.assertEqual(self._ids("error:permission"), {a})

    def test_error_matches_any(self):
        a = _add_session(self.store, errors=["boom"])
        _add_session(self.store, tools=["read"])
        self.assertEqual(self._ids("error:"), {a})

    def test_and_requires_both(self):
        a = _add_session(self.store, tools=["bash"], errors=["permission denied"])
        _add_session(self.store, tools=["bash"])
        self.assertEqual(self._ids("tool:bash AND error:permission"), {a})

    def test_or_matches_either(self):
        a = _add_session(self.store, tools=["bash"])
        b = _add_session(self.store, errors=["permission denied"])
        self.assertEqual(self._ids("tool:bash OR error:permission"), {a, b})

    def test_cost_greater_than(self):
        cheap = _add_session(self.store, tools=["read"])
        pricey = _add_session(self.store, tools=["bash"], big_payload=True)
        ids = self._ids("cost:>0")
        self.assertIn(pricey, ids)
        # cheap session has a tiny but nonzero cost; assert ordering via direct compare
        results = {r.session_id: r.cost for r in
                   search_sessions(self.store, parse_query("cost:>0"))}
        self.assertGreater(results[pricey], results.get(cheap, 0))

    def test_date_predicate(self):
        old = _add_session(self.store, started_at=time.time() - 40 * 86400,
                           tools=["read"])
        recent = _add_session(self.store, started_at=time.time() - 1 * 86400,
                              tools=["read"])
        ids = self._ids("date:7d")
        self.assertIn(recent, ids)
        self.assertNotIn(old, ids)

    def test_no_match_returns_empty(self):
        _add_session(self.store, tools=["read"])
        self.assertEqual(self._ids("tool:nonexistent"), set())

    def test_results_sorted_newest_first(self):
        old = _add_session(self.store, started_at=time.time() - 10000, tools=["bash"])
        new = _add_session(self.store, started_at=time.time(), tools=["bash"])
        results = search_sessions(self.store, parse_query("tool:bash"))
        self.assertEqual(results[0].session_id, new)
        self.assertEqual(results[1].session_id, old)


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------

class TestOutputAndCli(unittest.TestCase):
    def setUp(self):
        self.store, self.tmp = _make_store()

    def test_json_output_shape(self):
        _add_session(self.store, tools=["bash"])
        results = search_sessions(self.store, parse_query("tool:bash"))
        import json
        parsed = json.loads(format_results_json(results))
        self.assertEqual(len(parsed), 1)
        self.assertIn("session_id", parsed[0])
        self.assertIn("cost", parsed[0])
        self.assertIn("tool_calls", parsed[0])

    def test_cmd_search_match_returns_zero(self):
        _add_session(self.store, tools=["bash"])
        args = argparse.Namespace(trace_dir=self.tmp, query=["tool:bash"],
                                  format="json", limit=0)
        self.assertEqual(cmd_search(args), 0)

    def test_cmd_search_no_match_returns_one(self):
        _add_session(self.store, tools=["read"])
        args = argparse.Namespace(trace_dir=self.tmp, query=["tool:bash"],
                                  format="text", limit=0)
        self.assertEqual(cmd_search(args), 1)

    def test_cmd_search_invalid_query_returns_one(self):
        args = argparse.Namespace(trace_dir=self.tmp, query=["justtext"],
                                  format="text", limit=0)
        self.assertEqual(cmd_search(args), 1)

    def test_cmd_search_limit(self):
        for _ in range(3):
            _add_session(self.store, tools=["bash"])
        import json
        buf = io.StringIO()
        import contextlib
        args = argparse.Namespace(trace_dir=self.tmp, query=["tool:bash"],
                                  format="json", limit=2)
        with contextlib.redirect_stdout(buf):
            cmd_search(args)
        self.assertEqual(len(json.loads(buf.getvalue())), 2)


if __name__ == "__main__":
    unittest.main()
