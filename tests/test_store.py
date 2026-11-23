"""Tests for trace storage."""

import concurrent.futures
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent_trace
from agent_trace.fileutil import lock_descriptor, unlock_descriptor
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import (
    AmbiguousSessionPrefixError,
    CorruptSessionMetaError,
    SessionMismatchError,
    TraceStore,
    UnreadableEventLineError,
    ensure_trace_dir_gitignore,
)

try:
    from _concurrency import run_concurrently
    from _no_telemetry import without_telemetry
    from test_fileutil import RAW_SEPARATORS
except ImportError:
    from tests._concurrency import run_concurrently
    from tests._no_telemetry import without_telemetry
    from tests.test_fileutil import RAW_SEPARATORS

SRC_ROOT = Path(agent_trace.__file__).resolve().parent.parent


def raw_separator_session(store: TraceStore, separator: str) -> tuple[str, list[str]]:
    """Write a session whose middle event holds separator raw inside a string.

    Our writers escape the character, but JS JSON.stringify and serde_json
    leave it raw. Returns the session ID and the lines as written, chained.
    """
    meta = SessionMeta(started_at=1.0)
    store.create_session(meta)
    events = [
        TraceEvent(EventType.USER_PROMPT, timestamp=1.0, data={"prompt": "hi"}),
        TraceEvent(EventType.TOOL_CALL, timestamp=2.0, data={
            "tool_name": "Bash", "arguments": {"command": f"echo a{separator}b"},
        }),
        TraceEvent(EventType.TOOL_RESULT, timestamp=3.0,
                   data={"tool_name": "Bash", "result": "ok"}),
    ]
    lines: list[str] = []
    for event in events:
        event.session_id = meta.session_id
        event.tenant_id = ""
        event.prev_hash = hashlib.sha256(lines[-1].encode("utf-8")).hexdigest() if lines else ""
        lines.append(event.to_json().replace(json.dumps(separator)[1:-1], separator))
    assert separator in lines[1]
    path = store.base_dir / meta.session_id / "events.ndjson"
    path.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))
    return meta.session_id, lines


class TestTraceStore(unittest.TestCase):
    def setUp(self):
        os.environ.pop("AGENT_TRACE_NO_REDACT", None)
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_NO_REDACT", None)

    def test_create_and_load_session(self):
        meta = SessionMeta(agent_name="test-agent")
        self.store.create_session(meta)

        loaded = self.store.load_meta(meta.session_id)
        self.assertEqual(loaded.agent_name, "test-agent")
        self.assertEqual(loaded.session_id, meta.session_id)

    def test_append_and_load_events(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        e1 = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "read_file"},
        )
        e2 = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            session_id=meta.session_id,
            parent_id=e1.event_id,
            duration_ms=42.5,
            data={"content_preview": "hello world"},
        )

        self.store.append_event(meta.session_id, e1)
        self.store.append_event(meta.session_id, e2)

        events = self.store.load_events(meta.session_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, EventType.TOOL_CALL)
        self.assertEqual(events[1].parent_id, e1.event_id)
        self.assertAlmostEqual(events[1].duration_ms, 42.5)

    def test_list_sessions(self):
        m1 = SessionMeta(agent_name="first")
        m2 = SessionMeta(agent_name="second")
        self.store.create_session(m1)
        self.store.create_session(m2)

        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_list_sessions_sorted_newest_first_by_started_at(self):
        old = SessionMeta(session_id="zz-old", started_at=1.0)
        new = SessionMeta(session_id="aa-new", started_at=2.0)
        self.store.create_session(old)
        self.store.create_session(new)

        sessions = self.store.list_sessions()
        self.assertEqual(
            [(m.session_id, m.started_at) for m in sessions],
            [("aa-new", 2.0), ("zz-old", 1.0)],
        )

    def test_list_sessions_uses_session_id_tiebreaker(self):
        lower = SessionMeta(session_id="aa-same", started_at=1.0)
        higher = SessionMeta(session_id="zz-same", started_at=1.0)
        self.store.create_session(lower)
        self.store.create_session(higher)

        sessions = self.store.list_sessions()
        self.assertEqual([m.session_id for m in sessions], ["zz-same", "aa-same"])

    def test_list_sessions_skips_malformed_metadata(self):
        valid = SessionMeta(session_id="valid", started_at=1.0)
        self.store.create_session(valid)

        malformed_dir = os.path.join(self.tmpdir, "malformed")
        os.makedirs(malformed_dir)
        with open(os.path.join(malformed_dir, "meta.json"), "w") as f:
            f.write("{not json")
        with open(os.path.join(self.tmpdir, "loose-file"), "w") as f:
            f.write("ignored")

        sessions = self.store.list_sessions()
        self.assertEqual([m.session_id for m in sessions], ["valid"])

    def _tenant_sessions(self) -> tuple[str, str, str]:
        """Create sessions p, c and b of tenant t1, each with one event."""
        ids = ("p" * 16, "c" * 16, "b" * 16)
        for session_id in ids:
            self.store.create_session(SessionMeta(
                session_id=session_id, started_at=1_780_000_000.0, tenant_id="t1",
            ))
            self.store.append_event(session_id, TraceEvent(
                event_type=EventType.USER_PROMPT, session_id=session_id,
                timestamp=1_780_000_000.0, data={"prompt": "hi"},
            ))
        return ids

    def _set_started_at(self, session_id: str, value: object) -> None:
        """Write a value no current writer stores, as an older collector could."""
        path = self.store._session_dir(session_id) / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["started_at"] = value
        path.write_text(json.dumps(meta), encoding="utf-8")

    def test_unusable_started_at_lists_last_and_fails_strict_listing(self):
        p, c, b = self._tenant_sessions()
        for label, value in (
            ("text", "x"), ("bool", True), ("NaN", float("nan")),
            ("infinity", float("inf")), ("400 digits", 10**400),
            ("past year 9999", 1e12), ("past the platform limit", 1e17),
            ("negative", -5),
        ):
            with self.subTest(label):
                self._set_started_at(b, value)
                self.assertEqual([m.session_id for m in self.store.list_sessions()], [p, c, b])
                with self.assertRaisesRegex(
                    ValueError, f"^corrupt session in tenant store: {b}$"
                ):
                    self.store.list_sessions_strict()

    def test_link_parent_redepths_children_past_an_unusable_started_at(self):
        p, c, b = self._tenant_sessions()
        grandchild = "g" * 16
        self.store.create_session(SessionMeta(
            session_id=grandchild, started_at=1_780_000_000.0, tenant_id="t1",
        ))
        self.store.link_parent(grandchild, c)
        self._set_started_at(b, "x")

        linked = self.store.link_parent(c, p)

        self.assertEqual((linked.parent_session_id, linked.depth), (p, 1))
        self.assertEqual(self.store.load_meta(c).parent_session_id, p)
        self.assertEqual(self.store.load_meta(grandchild).depth, 2)

    def test_get_latest_session_returns_newest_meta(self):
        old = SessionMeta(session_id="zz-old", started_at=1.0)
        new = SessionMeta(session_id="aa-new", started_at=2.0)
        self.store.create_session(old)
        self.store.create_session(new)

        latest = self.store.get_latest_session()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.session_id, "aa-new")
        self.assertEqual(latest.started_at, 2.0)

    def test_get_latest_session_id_uses_started_at_not_session_id(self):
        old = SessionMeta(session_id="zz-old", started_at=1.0)
        new = SessionMeta(session_id="aa-new", started_at=2.0)
        self.store.create_session(old)
        self.store.create_session(new)

        self.assertEqual(self.store.get_latest_session_id(), "aa-new")

    def test_find_session_by_prefix(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        prefix = meta.session_id[:6]
        found = self.store.find_session(prefix)
        self.assertEqual(found, meta.session_id)

    def test_find_session_not_found(self):
        found = self.store.find_session("nonexistent")
        self.assertIsNone(found)

    def _add_session_with_meta(self, session_id, meta_text=None, tenant_id=""):
        """Create a session, then replace its meta.json with ``meta_text`` if given."""
        self.store.create_session(
            SessionMeta(session_id=session_id, tenant_id=tenant_id, started_at=1.0)
        )
        if meta_text is not None:
            meta_file = self.store.base_dir / session_id / "meta.json"
            meta_file.write_text(meta_text, encoding="utf-8")

    def _scan_order(self, reverse):
        """Make the directory scan list entries sorted, or sorted in reverse."""
        real_iterdir = Path.iterdir
        return patch.object(
            type(self.store.base_dir), "iterdir",
            lambda path: iter(sorted(real_iterdir(path), reverse=reverse)),
        )

    def test_find_session_names_a_corrupt_match_by_full_id_or_prefix(self):
        self._add_session_with_meta("aaaa1111")
        self._add_session_with_meta("bbbb2222", "{")
        with self.assertRaises(CorruptSessionMetaError) as loaded:
            self.store.load_meta("bbbb2222")
        for typed in ("bbbb2222", "bbbb"):
            with self.subTest(typed=typed):
                with self.assertRaises(CorruptSessionMetaError) as raised:
                    self.store.find_session(typed)
                self.assertEqual(raised.exception.session_id, "bbbb2222")
                self.assertEqual(str(raised.exception), str(loaded.exception))
        self.assertEqual(self.store.find_session("aaaa"), "aaaa1111")
        self.assertIsNone(self.store.find_session("cccc"))

    def _assert_ambiguous(self, prefix, matches, message, tenant_id=None):
        with self.assertRaises(AmbiguousSessionPrefixError) as raised:
            self.store.find_session(prefix, tenant_id=tenant_id)
        self.assertEqual(raised.exception.prefix, prefix)
        self.assertEqual(raised.exception.matches, matches)
        self.assertEqual(str(raised.exception), message)

    def test_find_session_prefers_an_exact_id_in_either_scan_order(self):
        self._add_session_with_meta("run1")
        self._add_session_with_meta("run10")
        self._add_session_with_meta("bbbb", "{")
        self._add_session_with_meta("bbbb1111")
        for reverse in (False, True):
            with self.subTest(reverse=reverse), self._scan_order(reverse):
                self.assertEqual(self.store.find_session("run1"), "run1")
                self.assertEqual(self.store.find_session("run10"), "run10")
                # An exact ID with a corrupt meta.json raises its own error.
                with self.assertRaises(CorruptSessionMetaError) as raised:
                    self.store.find_session("bbbb")
                self.assertEqual(raised.exception.session_id, "bbbb")
                self.assertEqual(self.store.find_session("bbbb1"), "bbbb1111")

    def test_find_session_refuses_two_readable_matches_in_either_scan_order(self):
        self._add_session_with_meta("aaaa111100000001")
        self._add_session_with_meta("aaaa222200000002")
        for reverse in (False, True):
            with self.subTest(reverse=reverse), self._scan_order(reverse):
                self._assert_ambiguous(
                    "aaaa", ["aaaa111100000001", "aaaa222200000002"],
                    "session ID prefix aaaa matches 2 sessions: "
                    "aaaa111100000001, aaaa222200000002",
                )
                self.assertEqual(self.store.find_session("aaaa2"), "aaaa222200000002")

    def test_find_session_refuses_a_readable_and_a_corrupt_match_in_either_scan_order(self):
        # Before, the readable match won.
        self._add_session_with_meta("bbbb1111")
        self._add_session_with_meta("bbbb2222", "{")
        for reverse in (False, True):
            with self.subTest(reverse=reverse), self._scan_order(reverse):
                self._assert_ambiguous(
                    "bbbb", ["bbbb1111", "bbbb2222"],
                    "session ID prefix bbbb matches 2 sessions: bbbb1111, bbbb2222",
                )
                with self.assertRaises(CorruptSessionMetaError):
                    self.store.find_session("bbbb2")

    def test_find_session_refuses_two_corrupt_matches_in_either_scan_order(self):
        # Before, the error named the corrupt match with the lowest ID.
        self._add_session_with_meta("bbbb1111", "[]")
        self._add_session_with_meta("bbbb2222", "{")
        for reverse in (False, True):
            with self.subTest(reverse=reverse), self._scan_order(reverse):
                self._assert_ambiguous(
                    "bbbb", ["bbbb1111", "bbbb2222"],
                    "session ID prefix bbbb matches 2 sessions: bbbb1111, bbbb2222",
                )

    def test_find_session_names_five_matches_and_counts_the_rest(self):
        for number in range(1, 8):
            self._add_session_with_meta(f"cccc000{number}", "{" if number == 7 else None)
        for reverse in (False, True):
            with self.subTest(reverse=reverse), self._scan_order(reverse):
                self._assert_ambiguous(
                    "cccc", [f"cccc000{number}" for number in range(1, 8)],
                    "session ID prefix cccc matches 7 sessions: cccc0001, cccc0002, "
                    "cccc0003, cccc0004, cccc0005 and 2 more",
                )

    def test_find_session_for_a_tenant_counts_only_its_own_sessions(self):
        self._add_session_with_meta("aaaa1111", tenant_id="acme")
        self._add_session_with_meta("aaaa2222", tenant_id="other")
        self._add_session_with_meta("aaaa3333", "{")
        self._add_session_with_meta("bbbb1111", tenant_id="acme")
        self._add_session_with_meta("bbbb2222", tenant_id="other")
        self._add_session_with_meta("bbbb3333", tenant_id="acme")
        for reverse in (False, True):
            with self.subTest(reverse=reverse), self._scan_order(reverse):
                self.assertEqual(self.store.find_session("aaaa", tenant_id="acme"), "aaaa1111")
                self.assertEqual(self.store.find_session("aaaa", tenant_id="other"), "aaaa2222")
                self._assert_ambiguous(
                    "bbbb", ["bbbb1111", "bbbb3333"],
                    "session ID prefix bbbb matches 2 sessions: bbbb1111, bbbb3333",
                    tenant_id="acme",
                )
                self.assertEqual(self.store.find_session("bbbb", tenant_id="other"), "bbbb2222")

    def test_find_session_does_not_let_a_longer_id_stand_in_for_an_exact_entry(self):
        # run1 is in the trace directory but is not a session. Before, run10 was used.
        states = (
            ("meta.json moved out", "run1/meta.json", False),
            ("meta.json a symlink", "run1/meta.json", True),
            ("run1 a symlinked directory", "run1", True),
        )
        for state, moved, linked in states:
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            store = TraceStore(os.path.join(tmp.name, "traces"))
            for session_id in ("run1", "run10"):
                store.create_session(
                    SessionMeta(session_id=session_id, tenant_id="acme", started_at=1.0)
                )
            source = store.base_dir / moved
            target = Path(tmp.name) / source.name
            source.rename(target)
            if linked:
                try:
                    source.symlink_to(target, target_is_directory=target.is_dir())
                except OSError:
                    continue  # This account cannot create symlinks.
            for reverse in (False, True):
                with self.subTest(state=state, reverse=reverse), self._scan_order(reverse):
                    self.assertIsNone(store.find_session("run1"))
                    self.assertEqual(store.find_session("run10"), "run10")
                    # A lookup with a tenant skips this check, as before.
                    self.assertEqual(store.find_session("run1", tenant_id="acme"), "run10")

    def test_find_session_for_a_tenant_skips_a_corrupt_match(self):
        # The tenant of a corrupt session cannot be read, so the lookup must not reveal it.
        self._add_session_with_meta("bbbb1111", tenant_id="acme")
        self._add_session_with_meta("bbbb2222", "{")
        self._add_session_with_meta("cccc2222", "{")
        for reverse in (False, True):
            with self.subTest(reverse=reverse), self._scan_order(reverse):
                self.assertEqual(self.store.find_session("bbbb", tenant_id="acme"), "bbbb1111")
                self.assertIsNone(self.store.find_session("bbbb", tenant_id="other"))
                self.assertIsNone(self.store.find_session("cccc", tenant_id="acme"))
                with self.assertRaises(CorruptSessionMetaError):
                    self.store.find_session("cccc")

    def test_find_session_still_skips_what_it_skipped_before(self):
        base = self.store.base_dir
        self._add_session_with_meta("bbbb2222", "{")
        (base / "bbbb0000").mkdir()
        (base / "bbbb0001" / "meta.json").mkdir(parents=True)
        self._add_session_with_meta("bbbb0002")
        try:
            (base / "bbbb0003").symlink_to(base / "bbbb2222", target_is_directory=True)
            (base / "bbbb0004").mkdir()
            (base / "bbbb0004" / "meta.json").symlink_to(base / "bbbb2222" / "meta.json")
        except OSError:
            pass  # This account cannot create symlinks.
        try:
            (base / "bbbb\x01").mkdir()
            (base / "bbbb\x01" / "meta.json").write_text("{", encoding="utf-8")
        except OSError:
            pass  # Windows refuses a control character in a file name.
        real_load_meta = TraceStore.load_meta

        def load_meta(store, session_id):
            if session_id == "bbbb0002":
                raise PermissionError(13, "Permission denied")
            return real_load_meta(store, session_id)

        for reverse in (False, True):
            with self.subTest(reverse=reverse), self._scan_order(reverse), patch.object(
                TraceStore, "load_meta", load_meta
            ):
                with self.assertRaises(CorruptSessionMetaError) as raised:
                    self.store.find_session("bbbb")
                self.assertEqual(raised.exception.session_id, "bbbb2222")
                self.assertIsNone(self.store.find_session("bbbb0"))

    def test_find_session_names_a_meta_json_that_names_another_session(self):
        self._add_session_with_meta("bbbb2222")
        name_other_session_in_meta(self.store, "bbbb2222")
        for typed in ("bbbb2222", "bbbb"):
            with self.subTest(typed=typed):
                with self.assertRaises(CorruptSessionMetaError) as raised:
                    self.store.find_session(typed)
                self.assertEqual(
                    str(raised.exception),
                    "session bbbb2222 has a corrupt meta.json: "
                    "metadata session ID does not match its directory",
                )
        self.assertIsNone(self.store.find_session(OTHER_SESSION))

    def test_session_exists(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        self.assertTrue(self.store.session_exists(meta.session_id))
        self.assertFalse(self.store.session_exists("fake"))

    def test_update_meta(self):
        meta = SessionMeta(agent_name="test")
        self.store.create_session(meta)

        meta.tool_calls = 10
        meta.errors = 2
        self.store.update_meta(meta)

        loaded = self.store.load_meta(meta.session_id)
        self.assertEqual(loaded.tool_calls, 10)
        self.assertEqual(loaded.errors, 2)

    def test_mutate_meta_applies_change_and_returns_written_meta(self):
        meta = SessionMeta(agent_name="test")
        self.store.create_session(meta)

        def bump(current):
            current.tool_calls += 2
            current.total_tokens += 5

        written = self.store.mutate_meta(meta.session_id, bump)
        loaded = self.store.load_meta(meta.session_id)
        self.assertEqual((written.tool_calls, written.total_tokens), (2, 5))
        self.assertEqual(loaded.to_json(), written.to_json())

    def test_mutate_meta_missing_session_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.store.mutate_meta("0123456789abcdef", lambda m: None)

    def test_mutate_meta_keeps_update_meta_checks(self):
        meta = SessionMeta(agent_name="test")
        self.store.create_session(meta)

        def retag(current):
            current.tenant_id = "tenant-a"

        with self.assertRaisesRegex(ValueError, "tag_session"):
            self.store.mutate_meta(meta.session_id, retag)
        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "")

    @unittest.skipIf(sys.platform == "win32", "store lock needs fcntl.flock")
    def test_mutate_meta_does_not_lose_concurrent_increments(self):
        meta = SessionMeta(agent_name="test")
        self.store.create_session(meta)
        script = (
            "from agent_trace.store import TraceStore\n"
            "store = TraceStore(sys.argv[2])\n"
            "def bump(meta):\n"
            "    meta.tool_calls += 1\n"
            "wait_for_start()\n"
            "for _ in range(25):\n"
            "    store.mutate_meta(sys.argv[3], bump)\n"
        )
        run_concurrently(script, [[self.tmpdir, meta.session_id]] * 8)
        self.assertEqual(self.store.load_meta(meta.session_id).tool_calls, 200)

    def test_empty_events(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        events = self.store.load_events(meta.session_id)
        self.assertEqual(events, [])

    def test_append_event_redacts_secrets_by_default(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={
                "tool_name": "http_request",
                "arguments": {
                    "headers": {
                        "Authorization": "Bearer sk-abc123def456ghi789jkl012mno345pqr678",
                    },
                },
            },
        )

        self.store.append_event(meta.session_id, event)

        events = self.store.load_events(meta.session_id)
        self.assertTrue(events[0].redacted)
        self.assertNotIn("sk-abc123", str(events[0].data))
        self.assertIn("[REDACTED:", str(events[0].data))

    def test_append_event_allows_redaction_opt_out(self):
        store = TraceStore(self.tmpdir, redact=False)
        meta = SessionMeta()
        store.create_session(meta)

        event = TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=meta.session_id,
            data={"prompt": "use sk-abc123def456ghi789jkl012mno345pqr678"},
        )
        store.append_event(meta.session_id, event)

        events = store.load_events(meta.session_id)
        self.assertFalse(events[0].redacted)
        self.assertIn("sk-abc123", str(events[0].data))

    def test_append_event_respects_no_redact_env(self):
        os.environ["AGENT_TRACE_NO_REDACT"] = "1"
        store = TraceStore(self.tmpdir)
        meta = SessionMeta()
        store.create_session(meta)

        event = TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=meta.session_id,
            data={"prompt": "use sk-abc123def456ghi789jkl012mno345pqr678"},
        )
        store.append_event(meta.session_id, event)

        events = store.load_events(meta.session_id)
        self.assertFalse(events[0].redacted)
        self.assertIn("sk-abc123", str(events[0].data))

    def test_create_session_redacts_metadata_by_default(self):
        meta = SessionMeta(
            agent_name="test-agent",
            command="run --api-key sk-abc123def456ghi789jkl012mno345pqr678",
        )
        self.store.create_session(meta)

        loaded = self.store.load_meta(meta.session_id)
        self.assertTrue(loaded.redacted)
        self.assertNotIn("sk-abc123", loaded.command)

    def test_append_events_matches_append_event(self):
        secret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        streams = []
        for batched in (False, True):
            store_dir = tempfile.mkdtemp()
            store = TraceStore(store_dir)
            self.addCleanup(shutil.rmtree, store_dir, True)
            meta = SessionMeta()
            meta.session_id = "batch0001"
            store.create_session(meta)
            store.append_event(meta.session_id, TraceEvent(event_type=EventType.SESSION_START,
                                                           event_id="e-start", timestamp=0.5))
            events = [
                TraceEvent(event_type=EventType.TOOL_CALL, event_id=f"e{i}", timestamp=1.0 + i,
                           data={"tool_name": "bash", "arguments": {"command": f"echo {secret} {i}"}})
                for i in range(3)
            ]
            if batched:
                store.append_events(meta.session_id, events)
            else:
                for event in events:
                    store.append_event(meta.session_id, event)
            streams.append([json.loads(event.to_json()) for event in store.load_events(meta.session_id)])

        self.assertEqual(streams[0], streams[1])
        self.assertTrue(all(event["prev_hash"] for event in streams[1][1:]))
        self.assertNotIn(secret, json.dumps(streams[1]))

    def test_load_events_keeps_a_raw_line_separator_inside_a_value(self):
        for separator in RAW_SEPARATORS:
            with self.subTest(separator=hex(ord(separator))):
                session_id, _ = raw_separator_session(self.store, separator)
                events = self.store.load_events(session_id)
                self.assertEqual(
                    [event.event_type for event in events],
                    [EventType.USER_PROMPT, EventType.TOOL_CALL, EventType.TOOL_RESULT],
                )
                self.assertEqual(events[1].data["arguments"]["command"], f"echo a{separator}b")

    def test_appended_line_separators_are_escaped_and_load_back(self):
        meta = SessionMeta()
        self.store.create_session(meta)
        commands = [f"echo a{separator}b" for separator in RAW_SEPARATORS]
        for command in commands:
            self.store.append_event(meta.session_id, TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={"tool_name": "Bash", "arguments": {"command": command}},
            ))
        raw = (self.store.base_dir / meta.session_id / "events.ndjson").read_bytes()
        for separator in RAW_SEPARATORS:
            self.assertNotIn(separator.encode("utf-8"), raw)
        self.assertEqual(
            [event.data["arguments"]["command"]
             for event in self.store.load_events(meta.session_id)],
            commands,
        )

    def untagged_session_with_one_event(self) -> str:
        meta = SessionMeta(started_at=1.0, tenant_id="")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.SESSION_START, timestamp=1.0, tenant_id="",
        ))
        return meta.session_id

    def test_append_event_refuses_a_non_finite_timestamp_and_writes_nothing(self):
        class Real(float):
            """A float subclass, as numpy's float64 is."""

        session_id = self.untagged_session_with_one_event()
        before = tree_snapshot(self.store.base_dir)
        for timestamp in (float("nan"), float("inf"), float("-inf"), Real("nan")):
            # With a tenant, append_event would first tag the untagged session.
            for tenant in ("", "acme"):
                with self.subTest(timestamp=timestamp, tenant=tenant):
                    event = TraceEvent(event_type=EventType.DECISION, timestamp=timestamp,
                                       data={"choice": "a"}, tenant_id=tenant)
                    with self.assertRaisesRegex(
                        ValueError, "^event timestamp must be a finite number$"
                    ):
                        self.store.append_event(session_id, event)
                    self.assertEqual(tree_snapshot(self.store.base_dir), before)

    def test_append_events_refuses_the_whole_batch_for_one_non_finite_timestamp(self):
        # With a tenant, append_events hands each event to append_event.
        for tenant in ("", "acme"):
            with self.subTest(tenant=tenant):
                session_id = self.untagged_session_with_one_event()
                before = tree_snapshot(self.store.base_dir)
                batch = [
                    TraceEvent(event_type=EventType.TOOL_CALL, timestamp=2.0, tenant_id=tenant,
                               data={"tool_name": "Bash"}),
                    TraceEvent(event_type=EventType.DECISION, timestamp=float("nan"),
                               tenant_id=tenant),
                    TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=3.0, tenant_id=tenant),
                ]
                with self.assertRaisesRegex(
                    ValueError, "^event timestamp must be a finite number$"
                ):
                    self.store.append_events(session_id, batch)
                self.assertEqual(tree_snapshot(self.store.base_dir), before)

_GITIGNORE_BYTES = (
    b"# Created by agent-strace: traces may contain prompts and tool output.\n"
    b"*\n"
)


class TestTraceDirGitignore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.trace_dir = self.root / "traces"
        self._saved_env = {
            key: os.environ.pop(key, None)
            for key in ("AGENT_STRACE_WORKSPACE", "AGENT_TRACE_DIR",
                        "AGENT_TRACE_CLAUDE_SESSION_ID")
        }

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def _new_session(self, store):
        meta = SessionMeta(agent_name="test-agent")
        store.create_session(meta)
        store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "read_file"},
        ))
        return meta.session_id

    def test_new_directory_gets_gitignore_without_premature_creation(self):
        store = TraceStore(self.trace_dir)
        self.assertFalse(self.trace_dir.exists())

        self._new_session(store)

        gitignore = self.trace_dir / ".gitignore"
        self.assertEqual(gitignore.read_bytes(), _GITIGNORE_BYTES)
        self.assertIn("*", gitignore.read_text().splitlines())

    def test_existing_directory_without_gitignore_gets_one(self):
        self.trace_dir.mkdir()
        TraceStore(self.trace_dir)
        self.assertEqual(
            (self.trace_dir / ".gitignore").read_bytes(), _GITIGNORE_BYTES
        )

    def test_helper_never_creates_missing_directory(self):
        ensure_trace_dir_gitignore(self.trace_dir)
        self.assertFalse(self.trace_dir.exists())

    def test_existing_gitignore_is_left_byte_for_byte_unchanged(self):
        self.trace_dir.mkdir()
        original = b"# mine\r\n!keep-me.ndjson\nno-trailing-newline"
        (self.trace_dir / ".gitignore").write_bytes(original)

        store = TraceStore(self.trace_dir)
        self._new_session(store)

        self.assertEqual((self.trace_dir / ".gitignore").read_bytes(), original)

    def test_symlinked_gitignore_is_not_followed_or_replaced(self):
        self.trace_dir.mkdir()
        outside = self.root / "outside.txt"
        outside.write_text("untouched")
        (self.trace_dir / ".gitignore").symlink_to(outside)

        self._new_session(TraceStore(self.trace_dir))

        self.assertTrue((self.trace_dir / ".gitignore").is_symlink())
        self.assertEqual(outside.read_text(), "untouched")

    def test_open_failure_does_not_raise(self):
        self.trace_dir.mkdir()
        with patch("agent_trace.store.os.open", side_effect=PermissionError("ro")):
            store = TraceStore(self.trace_dir)
        self.assertFalse((self.trace_dir / ".gitignore").exists())
        self.assertEqual(store.list_sessions(), [])

    def test_write_failure_does_not_raise_or_leave_empty_file(self):
        self.trace_dir.mkdir()
        with patch("agent_trace.store.os.fdopen", side_effect=OSError("disk full")):
            TraceStore(self.trace_dir)
        self.assertFalse((self.trace_dir / ".gitignore").exists())

    @unittest.skipIf(
        os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        "directory permissions are not enforced for this user/platform",
    )
    def test_read_only_store_does_not_raise(self):
        self.trace_dir.mkdir()
        os.chmod(self.trace_dir, 0o555)
        try:
            store = TraceStore(self.trace_dir)
            self.assertEqual(store.list_sessions(), [])
            self.assertFalse((self.trace_dir / ".gitignore").exists())
        finally:
            os.chmod(self.trace_dir, 0o755)

    def test_session_listing_ignores_gitignore(self):
        store = TraceStore(self.trace_dir)
        first = self._new_session(store)
        second = self._new_session(store)
        self.assertTrue((self.trace_dir / ".gitignore").is_file())

        listed = {meta.session_id for meta in store.list_sessions()}
        strict = {meta.session_id for meta in store.list_sessions_strict()}
        self.assertEqual(listed, {first, second})
        self.assertEqual(strict, {first, second})
        self.assertIn(store.get_latest_session_id(), {first, second})
        self.assertEqual(store.find_session(first[:8]), first)

    def test_workspace_store_ignores_from_storage_root(self):
        from agent_trace.workspace import create_workspace, list_workspaces

        store = TraceStore(self.trace_dir, workspace_id="team-a")
        session_id = self._new_session(store)

        self.assertTrue((self.trace_dir / ".gitignore").is_file())
        self.assertFalse((store.base_dir / ".gitignore").exists())
        self.assertEqual(
            [meta.session_id for meta in store.list_sessions()], [session_id]
        )

        other_root = self.root / "other-traces"
        create_workspace("team-b", str(other_root))
        self.assertTrue((other_root / ".gitignore").is_file())
        self.assertEqual(list_workspaces(str(other_root)), ["team-b"])

    def test_repository_root_or_working_directory_is_never_ignored(self):
        repo_like = self.root / "repo"
        (repo_like / ".git").mkdir(parents=True)
        TraceStore(repo_like)
        self.assertFalse((repo_like / ".gitignore").exists())

        working = self.root / "work"
        (working / "nested").mkdir(parents=True)
        original_cwd = os.getcwd()
        os.chdir(working / "nested")
        try:
            TraceStore(working)
            TraceStore(working / "nested")
        finally:
            os.chdir(original_cwd)
        self.assertFalse((working / ".gitignore").exists())
        self.assertFalse((working / "nested" / ".gitignore").exists())

    def test_hook_capture_path_writes_gitignore(self):
        from agent_trace.hooks import hook_main

        project = self.root / "project"
        project.mkdir()
        payload = {"session_id": "hook-gitignore-01", "cwd": str(project),
                   "source": "startup"}
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))):
            hook_main(["session-start"])

        trace_dir = project / ".agent-traces"
        self.assertEqual((trace_dir / ".gitignore").read_bytes(), _GITIGNORE_BYTES)
        self.assertEqual(
            [meta.session_id for meta in TraceStore(trace_dir).list_sessions()],
            ["hook-gitignore-0"],
        )

    @unittest.skipUnless(shutil.which("git"), "git binary not available")
    def test_git_check_ignore_confirms_trace_files_are_ignored(self):
        repo = self.root / "repo"
        repo.mkdir()
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                env=env, capture_output=True, text=True,
            )

        self.assertEqual(git("init", "-q").returncode, 0)
        (repo / "notes.txt").write_text("tracked")
        session_id = self._new_session(TraceStore(repo / ".agent-traces"))
        trace_file = f".agent-traces/{session_id}/events.ndjson"

        self.assertEqual(git("check-ignore", "-q", trace_file).returncode, 0)
        self.assertEqual(git("check-ignore", "-q", "notes.txt").returncode, 1)
        status = git("status", "--porcelain", "--untracked-files=all").stdout
        self.assertEqual(status.splitlines(), ["?? notes.txt"])
        self.assertEqual(git("add", "-f", trace_file).returncode, 0)
        staged = git("diff", "--cached", "--name-only").stdout.splitlines()
        self.assertEqual(staged, [trace_file])


_APPEND_SCRIPT = """
from agent_trace.models import EventType, TraceEvent
from agent_trace.store import TraceStore
store = TraceStore(sys.argv[2])
session_id = sys.argv[3]
wait_for_start()
for index in range(int(sys.argv[4])):
    store.append_event(
        session_id,
        TraceEvent(event_type=EventType.TOOL_CALL, data={"n": index}),
    )
"""


class TestStoreWriteLock(unittest.TestCase):
    def setUp(self):
        os.environ.pop("AGENT_TRACE_NO_REDACT", None)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_lock_excludes_other_descriptors(self):
        lock_path = Path(self.tmp.name) / ".store.lock"
        with self.store.write_lock():
            fd = os.open(lock_path, os.O_RDWR)
            try:
                with self.assertRaises(TimeoutError):
                    lock_descriptor(fd, timeout=0.1)
            finally:
                os.close(fd)
        fd = os.open(lock_path, os.O_RDWR)
        try:
            lock_descriptor(fd, timeout=1.0)
            unlock_descriptor(fd)
        finally:
            os.close(fd)
        self.assertEqual(lock_path.stat().st_size, 0)

    def test_concurrent_appends_keep_hash_chain(self):
        meta = SessionMeta(agent_name="race")
        self.store.create_session(meta)
        workers, rounds = 4, 20
        run_concurrently(
            _APPEND_SCRIPT,
            [[self.tmp.name, meta.session_id, str(rounds)] for _ in range(workers)],
        )
        events_file = Path(self.tmp.name) / meta.session_id / "events.ndjson"
        lines = events_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), workers * rounds)
        self.assertEqual(json.loads(lines[0]).get("prev_hash", ""), "")
        for previous, line in zip(lines, lines[1:]):
            expected = hashlib.sha256(previous.encode()).hexdigest()
            self.assertEqual(json.loads(line)["prev_hash"], expected)


class TestRawLineSeparatorCommands(unittest.TestCase):
    """The read commands on a session whose line holds a raw line separator."""

    def test_replay_stats_export_and_inspect_exit_zero(self):
        root = Path(tempfile.mkdtemp(prefix="raw-separator-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "home").mkdir()
        trace_dir = root / "traces"
        store = TraceStore(str(trace_dir), use_workspace_env=False)
        sessions = {
            separator: raw_separator_session(store, separator)[0]
            for separator in RAW_SEPARATORS
        }
        env = without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        env["PYTHONPATH"] = str(SRC_ROOT)
        env["HOME"] = env["USERPROFILE"] = str(root / "home")
        env["PYTHONIOENCODING"] = "utf-8"
        jobs = [
            (separator, command)
            for separator in RAW_SEPARATORS
            for command in ("replay", "stats", "export", "inspect")
        ]

        def run(job):
            separator, command = job
            extra = ["--format", "json"] if command == "export" else []
            return subprocess.run(
                [sys.executable, "-m", "agent_trace.entry", "--trace-dir", str(trace_dir),
                 command, sessions[separator], *extra],
                capture_output=True, text=True, encoding="utf-8",
                env=env, cwd=root, timeout=120,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, jobs))
        for (separator, command), result in zip(jobs, results):
            with self.subTest(separator=hex(ord(separator)), command=command):
                self.assertEqual(result.returncode, 0, result.stderr)
                if command == "export":
                    events = json.loads(result.stdout)
                    self.assertEqual(len(events), 3)
                    self.assertEqual(
                        events[1]["data"]["arguments"]["command"], f"echo a{separator}b"
                    )


# json.loads raises RecursionError on text nested this deep on every CI
# Python. 3.14 limits nesting by stack size, so 100,000 levels still parses.
TOO_DEEP_FOR_JSON = 1_000_000
# json.loads parses this on 3.12 and later, but dataclasses.asdict ran out of
# stack on it, so inspect crashed.
TOO_DEEP_TO_COPY = 3_000


def nested(levels: int) -> str:
    """Return JSON text that nests ``levels`` objects around a string."""
    return '{"a":' * levels + '"x"' + "}" * levels


def deep_meta(levels: int) -> str:
    """Return meta.json text for bbbb2222 whose agent_name nests ``levels`` objects."""
    return '{"session_id":"bbbb2222","agent_name":' + nested(levels) + "}"


def store_with_corrupt_meta(root: Path, meta_text: str) -> TraceStore:
    """Return a store holding good session aaaa1111 and bbbb2222 with ``meta_text``."""
    store = TraceStore(str(root), use_workspace_env=False)
    store.create_session(SessionMeta(session_id="aaaa1111", tenant_id="acme", started_at=1.0))
    store.create_session(SessionMeta(session_id="bbbb2222", started_at=2.0))
    (root / "bbbb2222" / "meta.json").write_text(meta_text, encoding="utf-8")
    return store


class TestCorruptSessionMeta(unittest.TestCase):
    """A meta.json that is too deep or not an object is corrupt, like broken JSON."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="corrupt-meta-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_listing_skips_it_and_session_reads_refuse_it(self):
        cases = [
            ("too deep for json.loads", deep_meta(TOO_DEEP_FOR_JSON),
             "metadata nests more than 128 levels"),
            ("too deep to copy", deep_meta(TOO_DEEP_TO_COPY),
             "metadata nests more than 128 levels"),
            ("array", "[]", "metadata is not a JSON object"),
            ("null", "null", "metadata is not a JSON object"),
        ]
        for index, (label, text, reason) in enumerate(cases):
            with self.subTest(label):
                store = store_with_corrupt_meta(self.root / str(index), text)
                self.assertEqual([m.session_id for m in store.list_sessions()], ["aaaa1111"])
                message = f"^session bbbb2222 has a corrupt meta.json: {reason}$"
                with self.assertRaisesRegex(CorruptSessionMetaError, message):
                    store.find_session("bbbb")
                with self.assertRaisesRegex(CorruptSessionMetaError, message):
                    store.load_meta("bbbb2222")
                with self.assertRaisesRegex(CorruptSessionMetaError, message):
                    store.update_meta(SessionMeta(session_id="bbbb2222", started_at=2.0))
                with self.assertRaisesRegex(
                    ValueError, "^corrupt session in tenant store: bbbb2222$"
                ):
                    store.list_sessions_strict()

    def test_attribution_nested_up_to_the_limit_loads(self):
        store = store_with_corrupt_meta(
            self.root, '{"session_id":"bbbb2222","attribution":' + nested(127) + "}"
        )
        meta = store.load_meta("bbbb2222")
        self.assertEqual(SessionMeta.from_json(meta.to_json()).attribution, meta.attribution)
        self.assertEqual(len(store.list_sessions()), 2)

        (self.root / "bbbb2222" / "meta.json").write_text(
            '{"session_id":"bbbb2222","attribution":' + nested(128) + "}", encoding="utf-8"
        )
        with self.assertRaisesRegex(CorruptSessionMetaError, "nests more than 128 levels$"):
            store.load_meta("bbbb2222")
        self.assertEqual(len(store.list_sessions()), 1)

    def test_create_session_refuses_metadata_it_could_not_read_back(self):
        store = TraceStore(str(self.root), use_workspace_env=False)
        meta = SessionMeta(session_id="cccc3333", attribution=json.loads(nested(128)))
        with self.assertRaisesRegex(ValueError, "^metadata nests more than 128 levels$"):
            store.create_session(meta)
        self.assertFalse((self.root / "cccc3333").exists())


class TestCorruptSessionMetaCommands(unittest.TestCase):
    """The commands on a store where one meta.json is corrupt."""

    def test_commands_skip_the_session_or_refuse_it_in_one_line(self):
        root = Path(tempfile.mkdtemp(prefix="corrupt-meta-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "home").mkdir()
        too_deep = "metadata nests more than 128 levels"
        reasons = {}
        for name, meta_text, reason in (
            ("too-deep-for-json", deep_meta(TOO_DEEP_FOR_JSON), too_deep),
            ("too-deep-to-copy", deep_meta(TOO_DEEP_TO_COPY), too_deep),
            ("bad-parent-id", '{"session_id":"bbbb2222","parent_session_id":"../evil"}',
             "invalid stored parent session ID"),
        ):
            store_with_corrupt_meta(root / name, meta_text)
            reasons[name] = reason
        env = without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        env["PYTHONPATH"] = str(SRC_ROOT)
        env["HOME"] = env["USERPROFILE"] = str(root / "home")
        env["PYTHONIOENCODING"] = "utf-8"

        def run(job):
            name, command = job
            return subprocess.run(
                [sys.executable, "-m", "agent_trace.entry", "--trace-dir",
                 str(root / name), *command],
                capture_output=True, text=True, encoding="utf-8",
                env=env, cwd=root, timeout=120,
            )

        readers = [(cmd, "bbbb2222") for cmd in ("inspect", "replay", "export", "stats", "health")]
        strict = "corrupt session in tenant store: bbbb2222\n"
        expected = {
            ("tenant", "report"): (1, "", "Error: " + strict),
            ("tenant", "export", "acme"): (1, "", "Export failed: " + strict),
            ("cognitive-debt",): (1, "", "Error: trace input could not be read safely\n"),
        }
        listing = [("list",), ("stats",), ("health", "--format", "json")]
        jobs = [(name, command) for name in reasons for command in [*listing, *readers, *expected]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, jobs))
        # Run last: it would delete aaaa1111 if it ignored the corrupt session.
        deletes = [run((name, ("tenant", "delete", "acme", "--confirm"))) for name in reasons]

        for (name, command), result in zip(jobs, results):
            with self.subTest(store=name, command=command):
                self.assertNotIn("Traceback", result.stderr)
                if command in readers:
                    refused = f"Error: session bbbb2222 has a corrupt meta.json: {reasons[name]}\n"
                    self.assertEqual(
                        (result.returncode, result.stdout, result.stderr), (1, "", refused)
                    )
                elif command in expected:
                    self.assertEqual(
                        (result.returncode, result.stdout, result.stderr), expected[command]
                    )
                elif command[0] == "health":
                    self.assertEqual(json.loads(result.stdout)["session_id"], "aaaa1111")
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("aaaa1111", result.stdout)
                    self.assertNotIn("bbbb2222", result.stdout)
        for name, result in zip(reasons, deletes):
            with self.subTest(store=name, command="tenant delete"):
                self.assertEqual(
                    (result.returncode, result.stderr), (1, "Deletion failed: " + strict)
                )
                self.assertTrue((root / name / "aaaa1111" / "meta.json").exists())


class TestCorruptSessionMetaLookupCommands(unittest.TestCase):
    """Commands given the ID or a prefix of a session with a corrupt meta.json name it."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="corrupt-lookup-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "home").mkdir()
        self.env = without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        self.env["PYTHONPATH"] = str(SRC_ROOT)
        self.env["HOME"] = self.env["USERPROFILE"] = str(self.root / "home")
        self.env["PYTHONIOENCODING"] = "utf-8"
        from agent_trace.freeze import freeze_session

        good = TraceStore(str(self.root / "good"), use_workspace_env=False)
        good.create_session(SessionMeta(session_id="aaaa1111", started_at=1.0))
        self.fixture = self.root / "fixture.json"
        self.fixture.write_text(
            freeze_session(good, "aaaa1111").to_json() + "\n", encoding="utf-8"
        )

    def _store(self, name: str, *, readable: bool = True) -> Path:
        """Return a trace dir where bbbb2222 has one event and meta.json "{".

        With ``readable``, it also holds aaaa1111 of tenant acme.
        """
        trace_dir = self.root / name
        store = TraceStore(str(trace_dir), use_workspace_env=False)
        if readable:
            store.create_session(
                SessionMeta(session_id="aaaa1111", tenant_id="acme", started_at=1.0)
            )
        store.create_session(SessionMeta(session_id="bbbb2222", started_at=2.0))
        store.append_event(
            "bbbb2222", TraceEvent(EventType.USER_PROMPT, timestamp=2.0, data={"prompt": "hi"})
        )
        (trace_dir / "bbbb2222" / "meta.json").write_text("{", encoding="utf-8")
        with self.assertRaises(CorruptSessionMetaError) as loaded:
            store.load_meta("bbbb2222")
        self.corrupt = str(loaded.exception)
        return trace_dir

    def _check(self, trace_dir: Path, expected: dict) -> None:
        """Run each command in ``expected`` and compare (exit code, stdout, stderr)."""
        def run(command):
            return subprocess.run(
                [sys.executable, "-m", "agent_trace.entry", "--trace-dir", str(trace_dir),
                 *command],
                capture_output=True, text=True, encoding="utf-8",
                env=self.env, cwd=self.root, timeout=120,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, expected))
        for (command, want), result in zip(expected.items(), results):
            with self.subTest(command=command):
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual((result.returncode, result.stdout, result.stderr), want)

    def test_a_full_id_or_prefix_names_the_corrupt_meta_json(self):
        trace_dir = self._store("traces")
        refused = (1, "", f"Error: {self.corrupt}\n")
        expected = {}
        for typed in ("bbbb2222", "bbbb"):
            for command in (
                # These said "Session not found" even for the full ID.
                ("explain", typed),
                ("timeline", "--format", "json", typed),
                ("tree", "--format", "json", typed),
                ("replay", typed, "--tree"),
                ("stats", typed, "--include-subagents"),
                ("a2a-tree", typed),
                ("cost", typed),
                ("compare", "aaaa1111", typed),
                ("diff", "aaaa1111", typed),
                ("why", typed, "1"),
                ("share", typed, "--stdout"),
                ("share", typed, "--assignment"),
                ("freeze", typed),
                ("regression", str(self.fixture), typed),
                ("optimize", typed),
                ("export", typed, "--backend", "langfuse"),
                # These named it for the full ID only.
                ("inspect", typed),
                ("replay", typed),
                ("stats", typed),
                ("health", typed),
                ("export", typed, "--format", "json"),
                ("mcp-scan", "--session", typed),
            ):
                expected[command] = refused
            expected[("review", typed)] = (2, "", f"review: {self.corrupt}\n")
        # verify reads only events.ndjson, so the full ID still verifies.
        expected[("verify", "bbbb2222")] = (0, "✅ Chain intact — 1 events verified\n", "")
        expected[("verify", "bbbb")] = refused
        # cognitive-debt prints one line for any trace it cannot read.
        expected[("cognitive-debt", "--session", "bbbb")] = (
            1, "", "Error: trace input could not be read safely\n"
        )
        # The tenant of a corrupt session cannot be read, so a tenant lookup skips it.
        expected[("cost", "bbbb", "--tenant", "acme")] = (
            1, "", "Session not found for tenant acme: bbbb\n"
        )
        self._check(trace_dir, expected)

    def test_a_session_id_that_matches_nothing_is_named(self):
        trace_dir = self._store("traces")
        self._check(trace_dir, {
            ("optimize", "cccc"): (1, "", "Session not found: cccc\n"),
            ("export", "cccc", "--backend", "langfuse"): (1, "", "Session not found: cccc\n"),
        })

    def test_no_session_id_and_no_readable_session(self):
        trace_dir = self._store("only-corrupt", readable=False)
        no_session = "Session not found. Provide a session ID or record a session first.\n"
        self._check(trace_dir, {
            ("freeze",): (1, "", no_session),
            ("regression", str(self.fixture)): (1, "", no_session),
            ("optimize",): (1, "", "No sessions found.\n"),
        })


AMBIGUOUS_IDS = ("aaaa111100000001", "aaaa222200000002")
AMBIGUOUS = "session ID prefix aaaa matches 2 sessions: aaaa111100000001, aaaa222200000002"


class TestAmbiguousSessionPrefixCommands(unittest.TestCase):
    """A prefix that matches two sessions is refused, and nothing is written."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ambiguous-lookup-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "home").mkdir()
        self.env = without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        self.env["PYTHONPATH"] = str(SRC_ROOT)
        self.env["HOME"] = self.env["USERPROFILE"] = str(self.root / "home")
        self.env["PYTHONIOENCODING"] = "utf-8"
        from agent_trace.freeze import freeze_session

        self.trace_dir = self.root / "traces"
        store = TraceStore(str(self.trace_dir), use_workspace_env=False)
        for number, session_id in enumerate(AMBIGUOUS_IDS, start=1):
            store.create_session(SessionMeta(session_id=session_id, started_at=float(number)))
            store.append_event(session_id, TraceEvent(
                EventType.USER_PROMPT, timestamp=float(number), data={"prompt": "hi"}
            ))
        self.fixture = self.root / "fixture.json"
        self.fixture.write_text(
            freeze_session(store, AMBIGUOUS_IDS[0]).to_json() + "\n", encoding="utf-8"
        )
        # eval gate reads its config before it looks up the session.
        self.gate_config = self.root / "gate.yaml"
        self.gate_config.write_text(
            "evals:\n"
            "  - name: no-errors\n"
            "    scorer: error_count\n"
            "    threshold: 0\n"
            "    fail_on: above\n",
            encoding="utf-8",
        )

    def test_each_command_refuses_the_prefix_and_writes_nothing(self):
        first = AMBIGUOUS_IDS[0]
        commands = [
            ("inspect", "aaaa"),
            ("replay", "aaaa"),
            ("replay", "aaaa", "--tree"),
            ("export", "aaaa", "--format", "json"),
            ("export", "aaaa", "--format", "eu-ai-act"),
            ("export", "aaaa", "--backend", "langfuse"),
            ("export", "aaaa", "--anonymize"),
            ("stats", "aaaa"),
            ("stats", "aaaa", "--include-subagents"),
            ("cost", "aaaa"),
            ("explain", "aaaa"),
            ("timeline", "--format", "json", "aaaa"),
            ("tree", "--format", "json", "aaaa"),
            ("a2a-tree", "aaaa"),
            ("compare", first, "aaaa"),
            ("diff", first, "aaaa"),
            ("why", "aaaa", "1"),
            ("share", "aaaa", "--stdout"),
            ("share", "aaaa", "--assignment"),
            ("freeze", "aaaa"),
            ("regression", str(self.fixture), "aaaa"),
            ("optimize", "aaaa"),
            ("mcp-scan", "--session", "aaaa"),
            ("verify", "aaaa"),
            ("cognitive-debt", "--session", "aaaa"),
            ("identity", "sign", "aaaa"),
            ("identity", "verify", "aaaa"),
            ("audit", "aaaa"),
            ("postmortem", "aaaa"),
            ("token-budget", "aaaa"),
            ("standup", "--no-llm", "aaaa"),
            ("lint", "aaaa"),
            ("compaction", "aaaa"),
            ("skills", "aaaa"),
            ("baseline", "check", "aaaa"),
            ("compliance", "export", "aaaa"),
            ("policy", "generate", "--dry-run", "aaaa"),
            ("eval", "run", "aaaa"),
            ("eval", "gate", "aaaa", "--config", str(self.gate_config)),
            ("eval", "compare", first, "aaaa"),
        ]

        def run(command):
            return subprocess.run(
                [sys.executable, "-m", "agent_trace.entry", "--trace-dir", str(self.trace_dir),
                 *command],
                capture_output=True, text=True, encoding="utf-8",
                env=self.env, cwd=self.root, timeout=120,
            )

        before = tree_snapshot(self.root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, commands))
        for command, result in zip(commands, results):
            with self.subTest(command=command):
                self.assertEqual(
                    (result.returncode, result.stdout, result.stderr),
                    (1, "", f"Error: {AMBIGUOUS}\n"),
                )
        self.assertEqual(tree_snapshot(self.root), before)


OTHER_SESSION = "other-session-4c1d"
OTHER_TENANT = "other-tenant-4c1d"
# The byte 0xff is not UTF-8. The store writes ASCII JSON, so only another
# writer, an edit or a damaged disk leaves such a byte.
NOT_UTF8_LINE = b'{"event_type": "error", "data": {"m": "\xff"}}'


def event_line(**fields) -> str:
    """Return one readable event line, with ``fields`` set or replaced."""
    record = json.loads(TraceEvent(
        EventType.USER_PROMPT, timestamp=1.0, data={"prompt": "hi"}, tenant_id=""
    ).to_json())
    record.update(fields)
    return json.dumps(record)


def write_session_lines(
    store: TraceStore, lines: list[str | bytes], *, tenant_id: str = "", started_at: float = 1.0
) -> str:
    """Create a session whose events.ndjson holds ``lines`` as written. Returns its ID."""
    meta = SessionMeta(started_at=started_at, tenant_id=tenant_id)
    store.create_session(meta)
    events_file = store._session_dir(meta.session_id) / "events.ndjson"
    events_file.write_bytes(b"".join(
        (line if isinstance(line, bytes) else line.encode("utf-8")) + b"\n" for line in lines
    ))
    return meta.session_id


def name_other_session_in_meta(store: TraceStore, session_id: str) -> None:
    meta_file = store._session_dir(session_id) / "meta.json"
    data = json.loads(meta_file.read_text(encoding="utf-8"))
    data["session_id"] = OTHER_SESSION
    meta_file.write_text(json.dumps(data), encoding="utf-8")


def tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TestRefusedSessions(unittest.TestCase):
    """A session whose files name another session or tenant, or hold a line
    that is not an event, is refused with a message that names only the
    session directory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name, use_workspace_env=False)

    def assert_refused(self, error, session_id, reason):
        with self.assertRaises(error) as raised:
            self.store.load_events(session_id)
        exc = raised.exception
        self.assertEqual(str(exc), f"cannot read session {session_id}: {reason}")
        self.assertEqual((exc.session_id, exc.reason), (session_id, reason))
        self.assertNotIn(OTHER_SESSION, str(exc))
        self.assertNotIn(OTHER_TENANT, str(exc))

    def test_a_meta_json_that_names_another_session_is_refused_first(self):
        # load_events reads meta.json first, so this wins over the bad line.
        session_id = write_session_lines(self.store, ["{"])
        name_other_session_in_meta(self.store, session_id)
        for load in (self.store.load_meta, self.store.load_events):
            with self.subTest(load.__name__):
                with self.assertRaises(CorruptSessionMetaError) as raised:
                    load(session_id)
                self.assertEqual(
                    str(raised.exception),
                    f"session {session_id} has a corrupt meta.json: "
                    "metadata session ID does not match its directory",
                )

    def test_each_event_mismatch_names_the_directory_and_not_the_other_id(self):
        cases = [
            ("event session",
             write_session_lines(self.store, [event_line(), event_line(session_id=OTHER_SESSION)]),
             "event session ID does not match its directory"),
            ("numeric event session",
             write_session_lines(self.store, [event_line(session_id=5)]),
             "event session ID does not match its directory"),
            ("event tenant",
             write_session_lines(
                 self.store, [event_line(tenant_id=OTHER_TENANT)], tenant_id="team-a"
             ),
             "event tenant does not match session metadata"),
        ]
        for name, session_id, reason in cases:
            with self.subTest(name):
                self.assert_refused(SessionMismatchError, session_id, reason)

    def test_each_unreadable_kind_names_the_line_and_the_kind(self):
        cases = [
            ("{not json", "is not valid JSON"),
            ("   ", "is not valid JSON"),
            ("[" * TOO_DEEP_FOR_JSON + "]" * TOO_DEEP_FOR_JSON, "is nested too deeply"),
            ("[1, 2]", "is an array, not an object"),
            ('"made_up"', "is a string, not an object"),
            ("5", "is a number, not an object"),
            ("null", "is null, not an object"),
            ("true", "is a boolean, not an object"),
            ('{"timestamp": 1.0}', "has no event_type"),
        ]
        for event_type in ("made_up", 5, None, ["tool_call"], "TOOL_CALL"):
            cases.append((event_line(event_type=event_type), "has an unknown event_type"))
        digit_limit = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else 0
        if digit_limit:
            cases.append((
                '{"event_type": "user_prompt", "n": ' + "1" * (digit_limit + 1) + "}",
                "has a number with too many digits",
            ))
        for line, problem in cases:
            with self.subTest(line=line[:30], problem=problem):
                session_id = write_session_lines(self.store, [event_line(), line, event_line()])
                self.assert_refused(
                    UnreadableEventLineError, session_id, f"line 2 of events.ndjson {problem}"
                )

    def test_line_numbers_count_blank_lines_and_the_other_unreadable_lines(self):
        good = event_line()
        cases = [
            (["", "", good, "{", good], "line 4 of events.ndjson is not valid JSON"),
            (["\r", "\r", good + "\r", "{\r"], "line 4 of events.ndjson is not valid JSON"),
            ([good, "", "{", good], "line 3 of events.ndjson is not valid JSON"),
            ([good, "{", "[1]", good],
             "line 2 of events.ndjson is not valid JSON; 1 more line is unreadable"),
            ([good, "[1]", "{", '{"a": 1}', good],
             "line 2 of events.ndjson is an array, not an object; "
             "2 more lines are unreadable"),
        ]
        for lines, reason in cases:
            with self.subTest(reason):
                session_id = write_session_lines(self.store, lines)
                self.assert_refused(UnreadableEventLineError, session_id, reason)

    def test_a_mismatch_wins_over_an_unreadable_line(self):
        session_id = write_session_lines(self.store, ["{", event_line(session_id=OTHER_SESSION)])
        self.assert_refused(
            SessionMismatchError, session_id, "event session ID does not match its directory"
        )

    def test_a_line_that_is_not_utf8_is_unreadable(self):
        good = event_line()
        cases = [
            ([good, good, NOT_UTF8_LINE], "line 3 of events.ndjson is not valid UTF-8"),
            (["", good, NOT_UTF8_LINE], "line 3 of events.ndjson is not valid UTF-8"),
            ([good, NOT_UTF8_LINE, "{", good],
             "line 2 of events.ndjson is not valid UTF-8; 1 more line is unreadable"),
            ([good, "{", NOT_UTF8_LINE],
             "line 2 of events.ndjson is not valid JSON; 1 more line is unreadable"),
            # In Latin-1, 0x85 is a line break and 0xa0 is a space.
            ([b"\x85", good, b"\xa0"],
             "line 1 of events.ndjson is not valid UTF-8; 1 more line is unreadable"),
        ]
        for lines, reason in cases:
            with self.subTest(lines=lines):
                session_id = write_session_lines(self.store, lines)
                self.assert_refused(UnreadableEventLineError, session_id, reason)

    def test_a_mismatch_wins_over_a_line_that_is_not_utf8(self):
        session_id = write_session_lines(
            self.store, [NOT_UTF8_LINE, event_line(session_id=OTHER_SESSION)]
        )
        self.assert_refused(
            SessionMismatchError, session_id, "event session ID does not match its directory"
        )

    def test_the_json_escape_of_such_a_byte_still_loads(self):
        # json.dumps writes the code point U+DCFF as the six characters \udcff.
        escaped = event_line(data={"m": "\udcff"})
        self.assertIn("\\udcff", escaped)
        session_id = write_session_lines(self.store, [escaped])
        self.assertEqual(self.store.load_events(session_id)[0].data, {"m": "\udcff"})
        session_id = write_session_lines(self.store, [escaped, NOT_UTF8_LINE])
        self.assert_refused(
            UnreadableEventLineError, session_id, "line 2 of events.ndjson is not valid UTF-8"
        )

    def test_writers_refuse_a_line_that_is_not_utf8_and_write_nothing(self):
        reason = "line 2 of events.ndjson is not valid UTF-8; 1 more line is unreadable"
        good = event_line()
        # load_events reads CRLF and a lone CR as one line break, so each file
        # holds the same three lines. The writers read the bytes as written
        # and must name the same line.
        files = {
            "LF": [good, NOT_UTF8_LINE, "{"],
            "CRLF": [good + "\r", NOT_UTF8_LINE + b"\r", "{\r"],
            "lone CR": [good.encode("utf-8") + b"\r" + NOT_UTF8_LINE, "{"],
        }

        def event():
            return TraceEvent(EventType.USER_PROMPT, timestamp=2.0, data={}, tenant_id="")

        writers = {
            "append_event": lambda session_id: self.store.append_event(session_id, event()),
            "append_events": lambda session_id: self.store.append_events(session_id, [event()]),
            "tag_session": lambda session_id: self.store.tag_session(session_id, "team-a"),
        }
        for ending, lines in files.items():
            for name, write in writers.items():
                with self.subTest(ending, writer=name):
                    session_id = write_session_lines(self.store, lines)
                    self.assert_refused(UnreadableEventLineError, session_id, reason)
                    before = tree_snapshot(self.store.base_dir)
                    with self.assertRaises(UnreadableEventLineError) as raised:
                        write(session_id)
                    self.assertEqual(
                        str(raised.exception), f"cannot read session {session_id}: {reason}"
                    )
                    self.assertEqual(tree_snapshot(self.store.base_dir), before)


class TestRefusedSessionCommands(unittest.TestCase):
    """Read commands print one Error line for a refused session and write nothing."""

    def test_read_commands_print_one_line_that_names_only_the_session(self):
        root = Path(tempfile.mkdtemp(prefix="refused-session-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "home").mkdir()
        good = event_line()
        single = [["replay"], ["inspect"], ["export", "--format", "json"], ["stats"]]
        cases = {}

        def case(name, lines, reason, commands, *, tenant_id="", meta_names_other=False):
            trace_dir = root / name
            store = TraceStore(str(trace_dir), use_workspace_env=False)
            write_session_lines(store, [good], started_at=1.0)
            session_id = write_session_lines(store, lines, tenant_id=tenant_id, started_at=2.0)
            message = f"cannot read session {session_id}: {reason}"
            if meta_names_other:
                name_other_session_in_meta(store, session_id)
                message = f"session {session_id} has a corrupt meta.json: {reason}"
            cases[name] = (trace_dir, session_id, message, commands)

        case("meta-session", [good], "metadata session ID does not match its directory",
             single, meta_names_other=True)
        case("event-session", [good, event_line(session_id=OTHER_SESSION)],
             "event session ID does not match its directory", single + [["standup"]])
        case("event-session-number", [event_line(session_id=5)],
             "event session ID does not match its directory", [["replay"]])
        case("event-tenant", [event_line(tenant_id=OTHER_TENANT)],
             "event tenant does not match session metadata", single, tenant_id="team-a")
        case("not-json", [good, "{", good], "line 2 of events.ndjson is not valid JSON",
             [["replay"], ["standup"]])
        case("not-utf8", [good, NOT_UTF8_LINE, good], "line 2 of events.ndjson is not valid UTF-8",
             single + [["standup"], ["export", "--format", "eu-ai-act"]])
        case("array", [good, "[1]"], "line 2 of events.ndjson is an array, not an object",
             [["replay"]])
        case("no-event-type", ['{"timestamp": 1.0}'], "line 1 of events.ndjson has no event_type",
             [["replay"]])
        case("unknown-event-type", [event_line(event_type="made_up")],
             "line 1 of events.ndjson has an unknown event_type", [["replay"]])

        env = without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        env["PYTHONPATH"] = str(SRC_ROOT)
        env["HOME"] = env["USERPROFILE"] = str(root / "home")
        env["PYTHONIOENCODING"] = "utf-8"
        jobs = [(name, command) for name, spec in cases.items() for command in spec[3]]
        before = tree_snapshot(root)

        def run(job):
            name, command = job
            trace_dir, session_id = cases[name][:2]
            # standup with no session ID reads the newest session.
            argv = command[:1] + ([] if command[0] == "standup" else [session_id]) + command[1:]
            return subprocess.run(
                [sys.executable, "-m", "agent_trace.entry", "--trace-dir", str(trace_dir), *argv],
                capture_output=True, text=True, encoding="utf-8",
                env=env, cwd=root, timeout=120,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, jobs))
        for (name, command), result in zip(jobs, results):
            message = cases[name][2]
            with self.subTest(case=name, command=" ".join(command)):
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, f"Error: {message}\n")
                self.assertNotIn(OTHER_SESSION, result.stderr)
                self.assertNotIn(OTHER_TENANT, result.stderr)
        self.assertEqual(tree_snapshot(root), before)


if __name__ == "__main__":
    unittest.main()
