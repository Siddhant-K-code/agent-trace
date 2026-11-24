"""Tests for multi-tenant session isolation and GDPR workflows."""

import contextlib
import io
import json
import os
import hashlib
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from agent_trace.audit import verify_chain
from agent_trace.cli import build_parser, cmd_list, main
from agent_trace.cost import build_provider_cost_report, build_tenant_cost_report, cmd_cost
from agent_trace.event_data import UnusableTenantError
from agent_trace.hooks import handle_pre_tool, handle_session_start
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import node_tree_to_otlp, session_to_otlp, session_to_otlp_genai
from agent_trace.store import (
    TraceStore,
    validate_optional_tenant_id,
    validate_tenant_id,
)
from agent_trace.subagent import build_tree
from agent_trace.temporal import TEMPORAL_TRACE_PARENT_ENV, session_to_temporal_otlp
from agent_trace.tenancy import (
    build_tenant_report,
    cmd_tenant,
    delete_tenant_data,
    enumerate_trace_stores,
    tenant_export_payload,
)
from agent_trace.watch import cmd_watch
from agent_trace.web_dashboard import api_session_otlp

try:
    import _no_telemetry  # noqa: F401
    from test_non_object_event_data import (
        BAD,
        CHILD,
        GOOD,
        _capture_output,
        _chdir,
        _isolated_env,
        _tree_snapshot,
        build_store,
        expected_error,
    )
    from test_store import raw_separator_session
    from test_unusable_event_timestamps import (
        ENDPOINT,
        PARENT_SPAN_ID,
        TRACE_ID,
        TimestampCliTestCase,
        expected_ts_error,
        restamp,
        set_meta,
    )
except ImportError:
    from tests import _no_telemetry  # noqa: F401
    from tests.test_non_object_event_data import (
        BAD,
        CHILD,
        GOOD,
        _capture_output,
        _chdir,
        _isolated_env,
        _tree_snapshot,
        build_store,
        expected_error,
    )
    from tests.test_store import raw_separator_session
    from tests.test_unusable_event_timestamps import (
        ENDPOINT,
        PARENT_SPAN_ID,
        TRACE_ID,
        TimestampCliTestCase,
        expected_ts_error,
        restamp,
        set_meta,
    )


def _event(session_id: str, timestamp: float = 1.0, **kwargs) -> TraceEvent:
    data = kwargs.pop("data", {"prompt": "hello tenant"})
    return TraceEvent(
        event_type=EventType.USER_PROMPT,
        session_id=session_id,
        timestamp=timestamp,
        data=data,
        **kwargs,
    )


def _refuse_constant(token):
    raise ValueError(f"not JSON: {token}")


def _raw_separator_line(record: dict) -> bytes:
    """Return record as one JSON line with a raw U+2028 in an added "note" field.

    Our writers escape U+2028, so only a hand-written line holds it raw.
    """
    line = json.dumps({**record, "note": "a\u2028b"}, ensure_ascii=False)
    return (line + "\n").encode("utf-8")


def _attr_values(span: dict) -> dict:
    values = {}
    for attr in span.get("attributes", []):
        value = next(iter(attr["value"].values()))
        values[attr["key"]] = value
    return values


class TenantTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.tempdir.name, redact=False)

    def tearDown(self):
        self.tempdir.cleanup()

    def create_session(self, session_id: str, tenant_id: str, timestamp: float = 1.0):
        meta = SessionMeta(
            session_id=session_id,
            tenant_id=tenant_id,
            started_at=timestamp,
        )
        self.store.create_session(meta)
        self.store.append_event(session_id, _event(session_id, timestamp))
        return meta

    def write_legacy_session(
        self,
        base_dir: Path,
        session_id: str,
        tenant_id: str,
        timestamp: float = 1.0,
        parent_session_id: str = "",
        workspace_id: str = "",
    ) -> SessionMeta:
        """Write a pre-validation session without using the creation API."""
        session_dir = base_dir / session_id
        session_dir.mkdir(parents=True)
        meta = SessionMeta(
            session_id=session_id,
            tenant_id=tenant_id,
            started_at=timestamp,
            parent_session_id=parent_session_id,
            workspace_id=workspace_id,
        )
        (session_dir / "meta.json").write_text(meta.to_json(), encoding="utf-8")
        event = _event(session_id, timestamp, tenant_id=tenant_id)
        (session_dir / "events.ndjson").write_text(
            event.to_json() + "\n", encoding="utf-8"
        )
        return meta


class TestTenantModelAndStore(TenantTestCase):
    def test_tenant_validators_reject_every_non_string(self):
        self.assertEqual(validate_tenant_id("  customer-acme  "), "customer-acme")
        self.assertEqual(validate_optional_tenant_id(""), "")
        for value in (None, False, 0, [], {}, ["acme"], {"a": 1}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "^tenant ID must be a string$"
                ):
                    validate_tenant_id(value)
                with self.assertRaisesRegex(
                    ValueError, "^tenant ID must be a string$"
                ):
                    validate_optional_tenant_id(value)

    def test_strict_listing_validates_metadata_when_event_validation_is_disabled(self):
        meta = self.create_session("invalid-meta-strict", "tenant-a")
        path = self.store._session_dir(meta.session_id) / "meta.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["tenant_id"] = False
        path.write_text(json.dumps(record), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            f"^corrupt session in tenant store: {meta.session_id}$",
        ):
            self.store.list_sessions_strict(validate_events=False)

    def test_environment_tags_new_sessions_and_events(self):
        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "customer-acme"}):
            meta = SessionMeta(session_id="env-session")
            event = _event(meta.session_id)
            self.store.create_session(meta)
            self.store.append_event(meta.session_id, event)

        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "customer-acme")
        self.assertEqual(self.store.load_events(meta.session_id)[0].tenant_id, "customer-acme")
        raw = json.loads(
            (self.store._session_dir(meta.session_id) / "events.ndjson").read_text()
        )
        self.assertEqual(raw["tenant_id"], "customer-acme")

    def test_create_session_rejects_explicit_non_string_before_environment_fallback(self):
        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "tenant-env"}):
            for index, value in enumerate(
                (None, False, 0, [], {}, ["acme"], {"a": 1})
            ):
                with self.subTest(value=value):
                    meta = SessionMeta(
                        session_id=f"invalid-create-{index}",
                        tenant_id=value,
                    )
                    with self.assertRaisesRegex(
                        ValueError, "^tenant ID must be a string$"
                    ):
                        self.store.create_session(meta)
                    self.assertFalse(self.store.session_exists(meta.session_id))

    def test_create_session_trims_valid_tenant_string(self):
        meta = SessionMeta(session_id="trimmed-create", tenant_id="  customer-acme  ")

        self.store.create_session(meta)

        self.assertEqual(meta.tenant_id, "customer-acme")
        self.assertEqual(
            self.store.load_meta(meta.session_id).tenant_id, "customer-acme"
        )

    def test_legacy_deserialisation_does_not_inherit_reader_environment(self):
        meta_json = '{"session_id":"legacy","started_at":1}'
        event_json = (
            '{"event_type":"user_prompt","timestamp":1,"event_id":"e1",'
            '"session_id":"legacy","data":{}}'
        )
        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "wrong-tenant"}):
            self.assertEqual(SessionMeta.from_json(meta_json).tenant_id, "")
            self.assertEqual(TraceEvent.from_json(event_json).tenant_id, "")

    def test_append_event_rejects_truthy_and_falsey_non_string_tenants(self):
        for index, value in enumerate(
            (None, False, 0, [], {}, ["acme"], {"a": 1})
        ):
            with self.subTest(value=value):
                session_id = f"invalid-event-{index}"
                self.store.create_session(SessionMeta(session_id=session_id))
                event = _event(session_id, tenant_id=value)
                with self.assertRaisesRegex(
                    ValueError, "^tenant ID must be a string$"
                ):
                    self.store.append_event(session_id, event)
                self.assertEqual(self.store.load_events(session_id), [])
                self.assertEqual(self.store.load_meta(session_id).tenant_id, "")

    def test_append_events_rejects_falsey_non_string_tenant_before_writing(self):
        meta = SessionMeta(session_id="invalid-event-batch")
        self.store.create_session(meta)

        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            self.store.append_events(
                meta.session_id,
                [_event(meta.session_id, tenant_id=[])],
            )

        self.assertEqual(self.store.load_events(meta.session_id), [])

    def test_append_without_metadata_keeps_empty_tenant_but_rejects_non_string(self):
        self.store.append_event(
            "event-first-valid",
            _event("event-first-valid", tenant_id=""),
        )
        self.assertFalse(
            (self.store.base_dir / "event-first-valid" / "meta.json").exists()
        )
        raw = json.loads(
            (self.store.base_dir / "event-first-valid" / "events.ndjson").read_text()
        )
        self.assertNotIn("tenant_id", raw)

        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            self.store.append_event(
                "event-first-invalid",
                _event("event-first-invalid", tenant_id=False),
            )
        self.assertFalse((self.store.base_dir / "event-first-invalid").exists())

    def test_authoritative_tenant_accepts_trimmed_match_and_inherits_when_empty(self):
        meta = SessionMeta(session_id="authoritative", tenant_id="  tenant-a  ")
        self.store.create_session(meta)
        self.store.append_event(
            meta.session_id,
            _event(meta.session_id, tenant_id=" tenant-a "),
        )
        self.store.append_event(
            meta.session_id,
            _event(meta.session_id, timestamp=2.0, tenant_id=""),
        )

        self.assertEqual(
            [event.tenant_id for event in self.store.load_events(meta.session_id)],
            ["tenant-a", "tenant-a"],
        )
        self.assertEqual(
            [item.session_id for item in self.store.list_sessions_strict("tenant-a")],
            [meta.session_id],
        )

    def test_strict_listing_rejects_tagged_event_under_untagged_metadata(self):
        meta = self.create_session("legacy-tagged-event", "")
        path = self.store._session_dir(meta.session_id) / "events.ndjson"
        event = json.loads(path.read_text(encoding="utf-8"))
        event["tenant_id"] = "tenant-a"
        path.write_text(json.dumps(event) + "\n", encoding="utf-8")

        self.assertEqual(
            self.store.load_events(meta.session_id)[0].tenant_id,
            "tenant-a",
        )
        self.assertEqual(
            [item.session_id for item in self.store.list_sessions()],
            [meta.session_id],
        )
        for tenant_id in (None, "tenant-a"):
            with self.subTest(tenant_id=tenant_id), self.assertRaisesRegex(
                ValueError,
                f"^corrupt session in tenant store: {meta.session_id}$",
            ):
                self.store.list_sessions_strict(tenant_id)

    def test_session_metadata_rejects_mixed_tenant_event(self):
        meta = self.create_session("isolated", "tenant-a")
        path = self.store._session_dir(meta.session_id) / "events.ndjson"
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.store.append_event(
                meta.session_id,
                _event(meta.session_id, tenant_id="tenant-b"),
            )
        self.assertEqual(path.read_bytes(), before)

    def test_update_and_mutate_meta_reject_non_string_tenants(self):
        meta = SessionMeta(session_id="invalid-meta-update")
        self.store.create_session(meta)
        update = self.store.load_meta(meta.session_id)
        update.tenant_id = []
        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            self.store.update_meta(update)

        def set_false(current):
            current.tenant_id = False

        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            self.store.mutate_meta(meta.session_id, set_false)
        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "")

    def test_mutate_meta_rejects_stored_falsey_non_string_tenant(self):
        meta = SessionMeta(session_id="stored-invalid-meta")
        self.store.create_session(meta)
        path = self.store._session_dir(meta.session_id) / "meta.json"
        record = json.loads(path.read_text())
        record["tenant_id"] = None
        path.write_text(json.dumps(record))

        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            self.store.mutate_meta(
                meta.session_id,
                lambda current: setattr(current, "tool_calls", 1),
            )
        self.assertIsNone(json.loads(path.read_text())["tenant_id"])

    def test_scoped_store_iteration_and_prefix_lookup(self):
        self.create_session("tenant-a-session", "tenant-a", 2.0)
        self.create_session("tenant-b-session", "tenant-b", 3.0)

        self.assertEqual(
            [meta.session_id for meta in self.store.list_sessions("tenant-a")],
            ["tenant-a-session"],
        )
        self.assertEqual(self.store.get_latest_session_id("tenant-b"), "tenant-b-session")
        self.assertIsNone(self.store.find_session("tenant-b", tenant_id="tenant-a"))

    def test_tag_session_preserves_data_and_rebuilds_valid_hash_chain(self):
        meta = SessionMeta(session_id="watch-session", started_at=1.0, tenant_id="")
        self.store.create_session(meta)
        first = _event(meta.session_id, data={"prompt": "first"}, redacted=True)
        second = TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE,
            session_id=meta.session_id,
            timestamp=2.0,
            data={"text": "second"},
        )
        self.store.append_event(meta.session_id, first)
        self.store.append_event(meta.session_id, second)
        original = [(event.event_id, event.data, event.redacted) for event in self.store.load_events(meta.session_id)]

        count = self.store.tag_session(meta.session_id, "tenant-a")

        tagged = self.store.load_events(meta.session_id)
        self.assertEqual(count, 2)
        self.assertEqual([(e.event_id, e.data, e.redacted) for e in tagged], original)
        self.assertEqual({event.tenant_id for event in tagged}, {"tenant-a"})
        self.assertTrue(verify_chain(self.store, meta.session_id).ok)
        audit = json.loads((self.store.base_dir / "tenant-audit.ndjson").read_text())
        self.assertTrue(audit["previous_terminal_hash"])
        self.assertTrue(audit["new_terminal_hash"])
        self.assertNotEqual(audit["previous_terminal_hash"], audit["new_terminal_hash"])

    def test_tag_session_keeps_nan_and_infinity_tokens_from_older_releases(self):
        session_id = "nan-tokens"
        self.store.create_session(SessionMeta(session_id=session_id, started_at=1.0, tenant_id=""))
        nan, inf = float("nan"), float("inf")
        records = [
            {"event_type": "user_prompt", "timestamp": 1.0, "event_id": "e1",
             "session_id": session_id, "data": {"prompt": "hi", "score": nan}},
            {"event_type": "tool_call", "timestamp": 2.0, "event_id": "e2",
             "session_id": session_id, "duration_ms": inf,
             "data": {"tool_name": "Bash", "arguments": {"y": -inf}}},
            {"event_type": "decision", "timestamp": nan, "event_id": "e3",
             "session_id": session_id, "data": {"choice": "a"}},
        ]
        lines = []
        for record in records:
            if lines:
                record = {**record, "prev_hash": hashlib.sha256(lines[-1].encode()).hexdigest()}
            # Written as releases before strict JSON wrote it.
            lines.append(json.dumps(record, separators=(",", ":")))
        events_path = self.store.base_dir / session_id / "events.ndjson"
        events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        def kept_fields(line):
            # json.dumps writes NaN and infinity back as the same tokens.
            record = json.loads(line)
            record.pop("tenant_id", None)
            record.pop("prev_hash", None)
            return json.dumps(record, sort_keys=True)

        self.assertEqual(self.store.tag_session(session_id, "tenant-a"), 3)

        tagged = events_path.read_bytes()
        tagged_lines = tagged.decode("utf-8").splitlines()
        self.assertEqual([kept_fields(line) for line in tagged_lines],
                         [kept_fields(line) for line in lines])
        for token in (b'"score":NaN', b'"duration_ms":Infinity', b'"y":-Infinity',
                      b'"timestamp":NaN'):
            self.assertIn(token, tagged)
        self.assertEqual({json.loads(line)["tenant_id"] for line in tagged_lines}, {"tenant-a"})
        self.assertTrue(verify_chain(self.store, session_id).ok)

        # A second roll-forward of the same journal gives the same bytes.
        journal_dir = self.store.base_dir / ".tenant-journal"
        journal_dir.mkdir(exist_ok=True)
        with self.store.write_lock():
            count = self.store._complete_tag_journal_locked(journal_dir / "tag-again.json", {
                "operation": "tag_session", "operation_id": "d" * 32,
                "session_id": session_id, "tenant_id": "tenant-a",
                "phase": "intent", "previous_terminal_hash": "",
            })
        self.assertEqual(count, 3)
        self.assertEqual(events_path.read_bytes(), tagged)

        # New events are strict JSON, and the chain still verifies.
        self.store.append_event(
            session_id, _event(session_id, 4.0, data={"v": nan}, tenant_id="tenant-a")
        )
        text = events_path.read_bytes()
        self.assertTrue(text.startswith(tagged))
        appended = json.loads(
            text[len(tagged):].decode("utf-8"), parse_constant=_refuse_constant
        )
        self.assertEqual(appended["data"], {"v": None})
        self.assertTrue(verify_chain(self.store, session_id).ok)

    def test_tag_session_conflict_leaves_trace_unchanged(self):
        meta = self.create_session("already-tagged", "tenant-a")
        meta_path = self.store._session_dir(meta.session_id) / "meta.json"
        events_path = self.store._session_dir(meta.session_id) / "events.ndjson"
        before = (meta_path.read_bytes(), events_path.read_bytes())

        with self.assertRaisesRegex(ValueError, "already assigned"):
            self.store.tag_session(meta.session_id, "tenant-b")

        self.assertEqual((meta_path.read_bytes(), events_path.read_bytes()), before)

    def test_tag_session_rejects_non_string_tenants(self):
        for index, value in enumerate((False, 0, [], {}, ["acme"], {"a": 1})):
            with self.subTest(value=value):
                session_id = f"invalid-tag-{index}"
                self.store.create_session(SessionMeta(session_id=session_id))
                with self.assertRaisesRegex(
                    ValueError, "^tenant ID must be a string$"
                ):
                    self.store.tag_session(session_id, value)
                self.assertEqual(self.store.load_meta(session_id).tenant_id, "")

    def test_tag_session_rejects_stored_falsey_non_string_event_tenant(self):
        meta = SessionMeta(session_id="stored-invalid-event")
        self.store.create_session(meta)
        path = self.store._session_dir(meta.session_id) / "events.ndjson"
        record = json.loads(_event(meta.session_id).to_json())
        record["tenant_id"] = False
        path.write_text(json.dumps(record) + "\n")
        before = path.read_bytes()

        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            self.store.tag_session(meta.session_id, "tenant-a")

        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "")

    def test_failed_metadata_replace_rolls_back_event_rewrite(self):
        from agent_trace import store as store_module

        meta = SessionMeta(session_id="atomic", started_at=1.0, tenant_id="")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, _event(meta.session_id))
        meta_path = self.store._session_dir(meta.session_id) / "meta.json"
        events_path = self.store._session_dir(meta.session_id) / "events.ndjson"
        before = (meta_path.read_bytes(), events_path.read_bytes())
        original_write = store_module._write_atomic

        def fail_meta(path, text):
            if Path(path).name == "meta.json":
                raise OSError("simulated metadata failure")
            return original_write(path, text)

        with patch("agent_trace.store._write_atomic", side_effect=fail_meta):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.store.tag_session(meta.session_id, "tenant-a")

        self.assertEqual((meta_path.read_bytes(), events_path.read_bytes()), before)
        self.assertFalse((self.store.base_dir / "tenant-audit.ndjson").exists())

    def test_update_meta_cannot_assign_clear_or_change_tenant(self):
        meta = SessionMeta(session_id="immutable", tenant_id="")
        self.store.create_session(meta)
        meta.tenant_id = "tenant-a"
        with self.assertRaisesRegex(ValueError, "tag_session"):
            self.store.update_meta(meta)
        self.store.tag_session(meta.session_id, "tenant-a")
        tagged = self.store.load_meta(meta.session_id)
        tagged.tenant_id = ""
        with self.assertRaisesRegex(ValueError, "cleared or changed"):
            self.store.update_meta(tagged)
        tagged.tenant_id = "tenant-b"
        with self.assertRaisesRegex(ValueError, "cleared or changed"):
            self.store.update_meta(tagged)

    def test_session_paths_and_metadata_directory_identity_are_strict(self):
        with self.assertRaises(ValueError):
            self.store.create_session(SessionMeta(session_id="../escape"))
        meta = SessionMeta(session_id="directory-id")
        self.store.create_session(meta)
        path = self.store._session_dir(meta.session_id) / "meta.json"
        data = json.loads(path.read_text())
        data["session_id"] = "different-id"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "directory"):
            self.store.load_meta(meta.session_id)
        self.assertEqual(self.store.list_sessions(), [])

    def test_new_ids_are_strict_while_safe_legacy_ids_remain_readable(self):
        legacy_id = "Legacy café session " + "x" * 130
        self.write_legacy_session(
            self.store.base_dir, legacy_id, "tenant-legacy", 1_781_481_600.0
        )

        self.assertEqual(self.store.load_meta(legacy_id).session_id, legacy_id)
        self.assertEqual(self.store.load_events(legacy_id)[0].session_id, legacy_id)
        self.assertEqual(self.store.find_session("Legacy café"), legacy_id)
        self.assertIn(legacy_id, {meta.session_id for meta in self.store.list_sessions()})
        with self.assertRaises(ValueError):
            self.store.create_session(SessionMeta(session_id="new session"))
        for unsafe in (
            "../escape", "nested/name", "nested\\name", "bad\x00id", "bad\x85id"
        ):
            with self.assertRaises(ValueError):
                self.store.load_meta(unsafe)

    def test_parent_tenant_is_enforced_on_create_and_update(self):
        parent_a = SessionMeta(session_id="parent-a", tenant_id="tenant-a")
        parent_b = SessionMeta(session_id="parent-b", tenant_id="tenant-b")
        self.store.create_session(parent_a)
        self.store.create_session(parent_b)

        with self.assertRaisesRegex(ValueError, "cross-tenant"):
            self.store.create_session(SessionMeta(
                session_id="bad-child",
                tenant_id="tenant-b",
                parent_session_id=parent_a.session_id,
            ))

        inherited = SessionMeta(
            session_id="inherited-child", parent_session_id=parent_a.session_id
        )
        self.store.create_session(inherited)
        self.assertEqual(inherited.tenant_id, "tenant-a")

        inherited.parent_session_id = parent_b.session_id
        with self.assertRaisesRegex(ValueError, "cross-tenant"):
            self.store.update_meta(inherited)

        legacy_parent = SessionMeta(session_id="legacy-parent")
        legacy_child = SessionMeta(
            session_id="legacy-child", parent_session_id=legacy_parent.session_id
        )
        self.store.create_session(legacy_parent)
        self.store.create_session(legacy_child)
        self.assertEqual(self.store.load_meta(legacy_child.session_id).tenant_id, "")

    def test_tag_session_cannot_split_existing_parent_child_links(self):
        parent = SessionMeta(session_id="tag-parent")
        child = SessionMeta(
            session_id="tag-child", parent_session_id=parent.session_id
        )
        self.store.create_session(parent)
        self.store.create_session(child)

        with self.assertRaisesRegex(ValueError, "child link"):
            self.store.tag_session(parent.session_id, "tenant-a")
        with self.assertRaisesRegex(ValueError, "parent link"):
            self.store.tag_session(child.session_id, "tenant-a")

        tagged_parent = SessionMeta(session_id="tagged-parent", tenant_id="tenant-a")
        self.store.create_session(tagged_parent)
        legacy_child = self.write_legacy_session(
            self.store.base_dir,
            "legacy-tag-child",
            "",
            parent_session_id=tagged_parent.session_id,
        )
        self.assertEqual(
            self.store.tag_session(legacy_child.session_id, "tenant-a"), 1
        )

        different_child = self.write_legacy_session(
            self.store.base_dir,
            "different-tag-child",
            "",
            parent_session_id=tagged_parent.session_id,
        )
        with self.assertRaisesRegex(ValueError, "already assigned|parent link"):
            self.store.tag_session(different_child.session_id, "tenant-b")

    def test_positional_dataclass_fields_remain_backward_compatible(self):
        self.assertEqual(fields(TraceEvent)[-1].name, "tenant_id")
        # Fields added after tenant_id are appended, never inserted before it.
        self.assertEqual(
            [f.name for f in fields(SessionMeta)][-2:],
            ["tenant_id", "conversation_id"],
        )
        event = TraceEvent(
            EventType.USER_PROMPT, 1.0, "event1", "session1", "", None,
            {"prompt": "hi"}, "previous", True,
        )
        self.assertTrue(event.redacted)
        self.assertEqual(event.prev_hash, "previous")

    def test_tag_journal_is_recovered_and_audit_is_hash_only(self):
        meta = SessionMeta(session_id="recover-tag", tenant_id="")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, _event(meta.session_id))
        operation_id = "a" * 32
        journal_dir = self.store.base_dir / ".tenant-journal"
        journal_dir.mkdir()
        journal = {
            "operation": "tag_session", "operation_id": operation_id,
            "session_id": meta.session_id, "tenant_id": "secret-tenant",
            "phase": "intent", "previous_terminal_hash": "before",
        }
        (journal_dir / f"tag-{operation_id}.json").write_text(json.dumps(journal))

        recovered = TraceStore(self.tempdir.name, redact=False)

        self.assertEqual(recovered.load_meta(meta.session_id).tenant_id, "secret-tenant")
        self.assertFalse((journal_dir / f"tag-{operation_id}.json").exists())
        audit_text = (recovered.base_dir / "tenant-audit.ndjson").read_text()
        self.assertNotIn("secret-tenant", audit_text)
        self.assertNotIn(meta.session_id, audit_text)

    def test_non_string_tag_journal_is_left_for_operator_recovery(self):
        meta = SessionMeta(session_id="invalid-tag-journal")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, _event(meta.session_id))
        operation_id = "b" * 32
        journal_dir = self.store.base_dir / ".tenant-journal"
        journal_dir.mkdir()
        journal_path = journal_dir / f"tag-{operation_id}.json"
        journal_path.write_text(json.dumps({
            "operation": "tag_session",
            "operation_id": operation_id,
            "session_id": meta.session_id,
            "tenant_id": False,
            "phase": "intent",
            "previous_terminal_hash": "before",
        }))

        recovered = TraceStore(self.tempdir.name, redact=False)

        self.assertEqual(recovered.load_meta(meta.session_id).tenant_id, "")
        self.assertTrue(journal_path.exists())
        self.assertFalse((recovered.base_dir / "tenant-audit.ndjson").exists())

    def test_legacy_empty_event_session_id_is_tagged_and_recovered(self):
        meta = SessionMeta(session_id="legacy-empty-event")
        self.store.create_session(meta)
        events_path = self.store._session_dir(meta.session_id) / "events.ndjson"
        legacy_event = _event("")
        events_path.write_text(legacy_event.to_json() + "\n")

        self.assertEqual(self.store.tag_session(meta.session_id, "tenant-a"), 1)
        raw = json.loads(events_path.read_text())
        self.assertEqual(raw["session_id"], meta.session_id)
        self.assertEqual(raw["tenant_id"], "tenant-a")

        mismatch = SessionMeta(session_id="legacy-mismatched-event")
        self.store.create_session(mismatch)
        mismatch_path = self.store._session_dir(mismatch.session_id) / "events.ndjson"
        mismatch_path.write_text(_event("another-session").to_json() + "\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.store.tag_session(mismatch.session_id, "tenant-a")
        self.assertEqual(self.store.load_meta(mismatch.session_id).tenant_id, "")

        second = SessionMeta(session_id="legacy-empty-recovery")
        self.store.create_session(second)
        second_events = self.store._session_dir(second.session_id) / "events.ndjson"
        second_events.write_text(_event("").to_json() + "\n")
        operation_id = "e" * 32
        journal_dir = self.store.base_dir / ".tenant-journal"
        journal_dir.mkdir(exist_ok=True)
        journal = {
            "operation": "tag_session", "operation_id": operation_id,
            "session_id": second.session_id, "tenant_id": "tenant-b",
            "phase": "intent", "previous_terminal_hash": "before",
        }
        (journal_dir / f"tag-{operation_id}.json").write_text(json.dumps(journal))

        recovered = TraceStore(self.tempdir.name, redact=False)
        self.assertEqual(recovered.load_meta(second.session_id).tenant_id, "tenant-b")
        self.assertEqual(
            recovered.load_events(second.session_id)[0].session_id, second.session_id
        )

    def test_post_audit_tag_failure_is_roll_forward_only(self):
        meta = SessionMeta(session_id="tag-commit-fault")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, _event(meta.session_id))
        original_append = self.store._append_tenant_audit

        def append_then_fail(record):
            original_append(record)
            raise OSError("failure after durable audit append")

        with patch.object(
            self.store, "_append_tenant_audit", side_effect=append_then_fail
        ):
            with self.assertRaisesRegex(OSError, "durable audit"):
                self.store.tag_session(meta.session_id, "tenant-a")

        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "tenant-a")
        self.assertEqual(
            self.store.load_events(meta.session_id)[0].tenant_id, "tenant-a"
        )
        self.assertEqual(
            len(list((self.store.base_dir / ".tenant-journal").glob("tag-*.json"))),
            1,
        )

        recovered = TraceStore(self.tempdir.name, redact=False)
        self.assertEqual(recovered.load_meta(meta.session_id).tenant_id, "tenant-a")
        self.assertEqual(
            len(list((recovered.base_dir / ".tenant-journal").glob("tag-*.json"))),
            0,
        )
        audits = (recovered.base_dir / "tenant-audit.ndjson").read_text().splitlines()
        self.assertEqual(len(audits), 1)

    def test_tag_session_keeps_a_raw_line_separator_inside_a_value(self):
        session_id, _ = raw_separator_session(self.store, "\u2028")
        events_path = self.store.base_dir / session_id / "events.ndjson"

        self.assertEqual(self.store.tag_session(session_id, "tenant-a"), 3)

        raw = events_path.read_bytes()
        self.assertNotIn("\u2028".encode("utf-8"), raw)
        self.assertIn(b"\\u2028", raw)
        self.assertTrue(verify_chain(self.store, session_id).ok)
        events = self.store.load_events(session_id)
        self.assertEqual({event.tenant_id for event in events}, {"tenant-a"})
        self.assertEqual(events[1].data["arguments"]["command"], "echo a\u2028b")

    def test_deletion_audit_line_with_a_raw_line_separator_is_read(self):
        session_id = "erased-session"
        operation_id = "f" * 32
        (self.store.base_dir / "tenant-deletions.ndjson").write_bytes(_raw_separator_line({
            "action": "tenant_data_deletion",
            "operation_id": operation_id,
            "session_id_sha256": [hashlib.sha256(session_id.encode()).hexdigest()],
            "status": "completed",
        }))

        self.assertTrue(self.store._deletion_audit_has(operation_id, "completed"))
        with self.assertRaisesRegex(ValueError, "was erased"):
            self.store.create_session(SessionMeta(session_id=session_id))

    def test_tag_recovery_does_not_repeat_an_audit_line_with_a_raw_line_separator(self):
        meta = SessionMeta(session_id="recover-tag-raw", tenant_id="")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, _event(meta.session_id))
        operation_id = "c" * 32
        audit_path = self.store.base_dir / "tenant-audit.ndjson"
        audit_path.write_bytes(_raw_separator_line({
            "action": "session_tenant_assigned", "operation_id": operation_id,
        }))
        journal_dir = self.store.base_dir / ".tenant-journal"
        journal_dir.mkdir()
        (journal_dir / f"tag-{operation_id}.json").write_text(json.dumps({
            "operation": "tag_session", "operation_id": operation_id,
            "session_id": meta.session_id, "tenant_id": "tenant-a",
            "phase": "commit", "previous_terminal_hash": "before",
        }))

        recovered = TraceStore(self.tempdir.name, redact=False)

        self.assertEqual(recovered.load_meta(meta.session_id).tenant_id, "tenant-a")
        self.assertEqual(audit_path.read_bytes().count(b"\n"), 1)


class TestTenantCostsAndCommands(TenantTestCase):
    def test_cost_reports_are_exactly_tenant_scoped(self):
        self.create_session("a", "tenant-a", 2_000.0)
        self.create_session("b", "tenant-b", 2_000.0)

        report = build_tenant_cost_report(
            self.store, "tenant-a", since="1d", now=2_100.0
        )
        providers = build_provider_cost_report(
            self.store, since="1d", now=2_100.0, tenant_id="tenant-a"
        )

        self.assertEqual(report.sessions, 1)
        self.assertGreater(report.total_cost, 0)
        self.assertNotIn("b", {row.session_id for row in providers.records})

    def test_cost_filters_reject_non_string_tenants(self):
        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            build_tenant_cost_report(self.store, False)
        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            build_provider_cost_report(self.store, tenant_id=[])

    def test_monthly_report_includes_tenants_and_untagged(self):
        june = 1_781_481_600.0  # 2026-06-15 UTC
        self.create_session("a", "tenant-a", june)
        self.create_session("untagged", "", june)
        self.create_session("outside", "tenant-b", 1.0)

        report = build_tenant_report(self.store, "2026-06")

        self.assertEqual(report.total_sessions, 2)
        self.assertEqual({row.tenant_id for row in report.rows}, {"tenant-a", ""})

    def test_parser_registers_all_tenant_flags(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["list", "--tenant", "a"]).tenant, "a")
        self.assertEqual(parser.parse_args(["cost", "--tenant", "a"]).tenant, "a")
        self.assertEqual(
            parser.parse_args(["watch", "session", "--tenant-id", "a"]).tenant_id,
            "a",
        )
        report = parser.parse_args(["tenant", "report", "--month", "2026-06"])
        self.assertEqual((report.command, report.tenant_command), ("tenant", "report"))
        delete = parser.parse_args(["tenant", "delete", "a", "--confirm"])
        self.assertTrue(delete.confirm)

    def test_watch_flag_tags_existing_events_before_monitoring(self):
        meta = SessionMeta(session_id="watch-target", tenant_id="")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, _event(meta.session_id))
        args = build_parser().parse_args([
            "--trace-dir", self.tempdir.name,
            "watch", meta.session_id, "--tenant-id", "tenant-a",
        ])

        with patch("agent_trace.watch.watch_session") as monitor:
            self.assertEqual(cmd_watch(args), 0)

        monitor.assert_called_once()
        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "tenant-a")
        self.assertEqual(self.store.load_events(meta.session_id)[0].tenant_id, "tenant-a")

    def test_watch_shows_a_tenant_error_as_one_inert_line(self):
        # The error quotes an event ID from the events file, which the agent
        # can write.
        meta = SessionMeta(session_id="watch-forged", tenant_id="")
        self.store.create_session(meta)
        record = json.loads(_event(meta.session_id).to_json())
        record.update(event_id="evt\x1b[2K\n2026-01-01T00:00:00Z  FORGED", tenant_id="tenant-b")
        events_path = Path(self.tempdir.name) / meta.session_id / "events.ndjson"
        events_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        args = build_parser().parse_args([
            "--trace-dir", self.tempdir.name,
            "watch", meta.session_id, "--tenant-id", "tenant-a",
        ])

        err = io.StringIO()
        with patch("agent_trace.watch.watch_session") as monitor, redirect_stderr(err):
            self.assertEqual(cmd_watch(args), 1)

        monitor.assert_not_called()
        self.assertEqual(
            err.getvalue(),
            "[watch] could not assign tenant: event evt\\x1b[2K\\n2026-01-01T00:00:00Z  FORGED "
            "belongs to another tenant\n",
        )

    def test_watch_environment_fallback_tags_existing_events(self):
        meta = SessionMeta(session_id="watch-env", tenant_id="")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, _event(meta.session_id))
        args = build_parser().parse_args([
            "--trace-dir", self.tempdir.name, "watch", meta.session_id,
        ])

        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "tenant-env"}), \
             patch("agent_trace.watch.watch_session"):
            self.assertEqual(cmd_watch(args), 0)

        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "tenant-env")

    def test_watch_explicit_non_string_tenant_does_not_use_environment_fallback(self):
        meta = SessionMeta(session_id="watch-invalid")
        self.store.create_session(meta)
        args = build_parser().parse_args([
            "--trace-dir", self.tempdir.name, "watch", meta.session_id,
        ])
        args.tenant_id = False

        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "tenant-env"}), \
                patch("agent_trace.watch.watch_session") as monitor, \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(cmd_watch(args), 1)

        monitor.assert_not_called()
        self.assertEqual(
            stderr.getvalue(),
            "[watch] could not assign tenant: tenant ID must be a string\n",
        )
        self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "")

    def test_list_command_passes_exact_tenant_scope(self):
        args = build_parser().parse_args([
            "--trace-dir", self.tempdir.name, "list", "--tenant", "tenant-a",
        ])
        with patch("agent_trace.cli.list_sessions") as render:
            self.assertEqual(cmd_list(args), 0)
        self.assertEqual(render.call_args.kwargs["tenant_id"], "tenant-a")

    def test_list_and_cost_commands_reject_non_string_tenant_filters(self):
        for command, handler in (("list", cmd_list), ("cost", cmd_cost)):
            with self.subTest(command=command):
                args = build_parser().parse_args([
                    "--trace-dir", self.tempdir.name,
                    command, "--tenant", "tenant-a",
                ])
                args.tenant = []
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    self.assertEqual(handler(args), 1)
                self.assertEqual(
                    stderr.getvalue(), "Error: tenant ID must be a string\n"
                )

    def test_cost_command_uses_aggregate_tenant_path(self):
        now = __import__("time").time()
        self.create_session("current-a", "tenant-a", now)
        self.create_session("current-b", "tenant-b", now)
        args = build_parser().parse_args([
            "--trace-dir", self.tempdir.name,
            "cost", "--tenant", "tenant-a", "--since", "1d",
        ])
        with patch("agent_trace.cost.format_tenant_cost") as render:
            self.assertEqual(cmd_cost(args), 0)
        result = render.call_args.args[0]
        self.assertEqual((result.tenant_id, result.sessions), ("tenant-a", 1))


class TestTenantExportAndDelete(TenantTestCase):
    def test_export_and_delete_reject_non_string_tenants(self):
        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            tenant_export_payload(self.store, ["tenant-a"])
        with self.assertRaisesRegex(ValueError, "^tenant ID must be a string$"):
            delete_tenant_data(self.store, False)

    def test_export_contains_only_requested_tenant(self):
        self.create_session("a", "tenant-a")
        self.create_session("b", "tenant-b")

        payload = tenant_export_payload(self.store, "tenant-a")

        self.assertEqual(payload["session_count"], 1)
        self.assertEqual(payload["sessions"][0]["session"]["session_id"], "a")
        self.assertEqual(payload["sessions"][0]["events"][0]["tenant_id"], "tenant-a")
        self.assertNotIn("tenant-b", json.dumps(payload))

    def test_export_reads_an_audit_line_with_a_raw_line_separator(self):
        (self.store.base_dir / "tenant-audit.ndjson").write_bytes(_raw_separator_line({
            "action": "session_tenant_assigned",
            "operation_id": "d" * 32,
            "tenant_id_sha256": hashlib.sha256(b"tenant-a").hexdigest(),
        }))

        records = tenant_export_payload(self.store, "tenant-a")["audit_records"]

        self.assertEqual([item["record"]["note"] for item in records], ["a\u2028b"])

    def test_admin_operations_fail_closed_on_corrupt_session_metadata(self):
        good = self.create_session("admin-good", "tenant-a")
        corrupt = self.store.base_dir / "corrupt-session"
        corrupt.mkdir()
        (corrupt / "meta.json").write_text("{not-json")
        (corrupt / "events.ndjson").write_text("")

        self.assertEqual(
            [meta.session_id for meta in self.store.list_sessions()],
            [good.session_id],
        )
        with self.assertRaisesRegex(ValueError, "corrupt session"):
            build_tenant_report(self.store, "2026-01")
        with self.assertRaisesRegex(ValueError, "corrupt session"):
            tenant_export_payload(self.store, "tenant-a")
        with self.assertRaisesRegex(ValueError, "corrupt session"):
            delete_tenant_data(self.store, "tenant-a")
        self.assertTrue(self.store.session_exists(good.session_id))

    def test_admin_operations_fail_closed_on_non_string_stored_tenants(self):
        for field, value in (("meta", None), ("event", False)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                store = TraceStore(directory, redact=False)
                meta = SessionMeta(session_id=f"invalid-{field}", tenant_id="tenant-a")
                store.create_session(meta)
                store.append_event(
                    meta.session_id,
                    _event(meta.session_id, tenant_id="tenant-a"),
                )
                path = store._session_dir(meta.session_id) / (
                    "meta.json" if field == "meta" else "events.ndjson"
                )
                record = json.loads(path.read_text())
                record["tenant_id"] = value
                path.write_text(json.dumps(record) + ("\n" if field == "event" else ""))

                with self.assertRaisesRegex(ValueError, "corrupt session"):
                    tenant_export_payload(store, "tenant-a")

    def test_delete_fails_closed_on_tagged_event_under_untagged_metadata(self):
        tagged = self.create_session("delete-tagged", "tenant-a")
        untagged = self.create_session("delete-untagged", "")
        path = self.store._session_dir(untagged.session_id) / "events.ndjson"
        event = json.loads(path.read_text(encoding="utf-8"))
        event["tenant_id"] = "tenant-a"
        path.write_text(json.dumps(event) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            f"^corrupt session in tenant store: {untagged.session_id}$",
        ):
            delete_tenant_data(self.store, "tenant-a")

        self.assertTrue(self.store.session_exists(tagged.session_id))
        self.assertTrue(self.store.session_exists(untagged.session_id))
        self.assertFalse(
            (self.store.base_dir / "tenant-deletions.ndjson").exists()
        )
        self.assertFalse((self.store.base_dir / ".tenant-journal").exists())

    def test_commands_refuse_a_store_with_an_unusable_started_at(self):
        session_ids = ("p" * 16, "c" * 16, "b" * 16)
        for session_id in session_ids:
            self.create_session(session_id, "t1", 1_780_000_000.0)
        meta_file = self.store.base_dir / ("b" * 16) / "meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta["started_at"] = "x"
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
        before = {
            path: path.read_bytes() for path in self.store.base_dir.rglob("*") if path.is_file()
        }
        error = "corrupt session in tenant store: " + "b" * 16
        for args, message in (
            (["report", "--month", "2026-05"], f"Error: {error}\n"),
            (["export", "t1"], f"Export failed: {error}\n"),
            (["delete", "t1", "--confirm"], f"Deletion failed: {error}\n"),
        ):
            with self.subTest(args[0]):
                argv = ["--trace-dir", self.tempdir.name, "tenant", *args]
                with patch("sys.stdout", new_callable=io.StringIO) as stdout, \
                        patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    self.assertEqual(cmd_tenant(build_parser().parse_args(argv)), 1)
                self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", message))
        after = {
            path: path.read_bytes() for path in self.store.base_dir.rglob("*") if path.is_file()
        }
        self.assertEqual(after, before)
        self.assertEqual(
            sorted(meta.session_id for meta in self.store.list_sessions()),
            sorted(session_ids),
        )

    def test_admin_operations_fail_closed_on_omitted_symlinked_metadata(self):
        session_id = "hidden-tenant-session"
        session_dir = self.store.base_dir / session_id
        session_dir.mkdir()
        (session_dir / "events.ndjson").write_text("")
        with tempfile.TemporaryDirectory() as external_dir:
            external_meta = Path(external_dir) / "meta.json"
            external_meta.write_text(SessionMeta(
                session_id=session_id, tenant_id="tenant-a"
            ).to_json())
            os.symlink(external_meta, session_dir / "meta.json")

            self.assertEqual(self.store.list_sessions(), [])
            with self.assertRaisesRegex(ValueError, "symlinked trace store entry|symlinked core"):
                tenant_export_payload(self.store, "tenant-a")
            with self.assertRaisesRegex(ValueError, "symlinked trace store entry|symlinked core"):
                delete_tenant_data(self.store, "tenant-a")
            self.assertEqual(
                json.loads(external_meta.read_text())["tenant_id"], "tenant-a"
            )
            self.assertTrue(session_dir.exists())

    def test_workspace_enumeration_and_commands_fail_closed_on_symlinks(self):
        with tempfile.TemporaryDirectory() as external_dir:
            external_root = Path(external_dir)
            external_workspaces = external_root / "workspaces"
            external_workspace = external_workspaces / "outside"
            self.write_legacy_session(
                external_workspace, "outside-session", "tenant-a"
            )
            workspaces = self.store.storage_root / "workspaces"
            os.symlink(external_workspaces, workspaces)

            with self.assertRaisesRegex(ValueError, "symlinked workspaces root"):
                enumerate_trace_stores(self.tempdir.name)
            for action in ("export", "delete"):
                argv = [
                    "--trace-dir", self.tempdir.name, "tenant", action, "tenant-a",
                ]
                if action == "delete":
                    argv.append("--confirm")
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    self.assertEqual(cmd_tenant(build_parser().parse_args(argv)), 1)
                self.assertIn("discovery failed", stderr.getvalue())
            self.assertTrue((external_workspace / "outside-session").exists())

            workspaces.unlink()
            workspaces.mkdir()
            os.symlink(external_workspace, workspaces / "linked-child")
            with self.assertRaisesRegex(ValueError, "symlinked workspace entry"):
                enumerate_trace_stores(self.tempdir.name)
            delete_args = build_parser().parse_args([
                "--trace-dir", self.tempdir.name,
                "tenant", "delete", "tenant-a", "--confirm",
            ])
            with patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(cmd_tenant(delete_args), 1)
            self.assertTrue((external_workspace / "outside-session").exists())

    def test_export_and_delete_cover_external_session_records(self):
        self.create_session("external-a", "tenant-a")
        self.create_session("external-b", "tenant-b")
        approvals = self.store.base_dir / ".approvals"
        approvals.mkdir()
        approval_a = approvals / "request-a.json"
        approval_b = approvals / "request-b.json"
        approval_a.write_text(json.dumps({
            "request_id": "request-a", "session_id": "external-a",
            "tool_input": {"secret": "customer-a"},
        }) + "\n")
        approval_b.write_text(json.dumps({
            "request_id": "request-b", "session_id": "external-b",
            "tool_input": {"keep": True},
        }) + "\n")
        datasets = self.store.base_dir / "datasets"
        datasets.mkdir()
        dataset = datasets / "default.jsonl"
        target_dataset = json.dumps({
            "entry_id": "entry-a", "session_id": "external-a", "label": "private",
        }) + "\n"
        malformed = "not-json\n"
        kept_dataset = json.dumps({
            "entry_id": "entry-b", "session_id": "external-b", "label": "keep",
        })
        dataset.write_text(target_dataset + malformed + kept_dataset)
        retention = self.store.base_dir / "retention.log"
        target_retention = json.dumps({
            "deleted_at": "2026-01-01T00:00:00Z", "session_id": "external-a",
        }) + "\n"
        kept_retention = json.dumps({
            "deleted_at": "2026-01-02T00:00:00Z", "session_id": "external-b",
        }) + "\n"
        retention.write_text(target_retention + kept_retention)

        payload = tenant_export_payload(self.store, "tenant-a")
        external = payload["external_records"]
        self.assertEqual(
            external["schema"], "agent-strace-tenant-external-records/v1"
        )
        self.assertEqual(
            {record["kind"] for record in external["records"]},
            {"approval", "dataset", "retention"},
        )
        self.assertIn("customer-a", json.dumps(external))
        self.assertNotIn("request-b", json.dumps(external))

        delete_tenant_data(self.store, "tenant-a")

        self.assertFalse(approval_a.exists())
        self.assertTrue(approval_b.exists())
        self.assertEqual(dataset.read_text(), malformed + kept_dataset)
        self.assertEqual(retention.read_text(), kept_retention)

    def test_external_record_symlinks_are_rejected_without_following(self):
        self.create_session("external-link", "tenant-a")
        with tempfile.TemporaryDirectory() as external_dir:
            external_root = Path(external_dir)
            external_approval = external_root / "approval.json"
            external_approval.write_text(json.dumps({
                "request_id": "outside", "session_id": "external-link",
                "tool_input": {"secret": "outside"},
            }))
            approvals = self.store.base_dir / ".approvals"
            approvals.mkdir()
            approval_link = approvals / "outside.json"
            os.symlink(external_approval, approval_link)
            with self.assertRaisesRegex(ValueError, "symlinked approval"):
                tenant_export_payload(self.store, "tenant-a")
            approval_link.unlink()

            external_dataset = external_root / "dataset.jsonl"
            external_dataset.write_text(json.dumps({
                "entry_id": "outside", "session_id": "external-link",
            }) + "\n")
            datasets = self.store.base_dir / "datasets"
            datasets.mkdir()
            dataset_link = datasets / "outside.jsonl"
            os.symlink(external_dataset, dataset_link)
            with self.assertRaisesRegex(ValueError, "symlinked dataset"):
                tenant_export_payload(self.store, "tenant-a")
            dataset_link.unlink()

            external_retention = external_root / "retention.log"
            external_retention.write_text(json.dumps({
                "deleted_at": "2026-01-01T00:00:00Z",
                "session_id": "external-link",
            }) + "\n")
            os.symlink(external_retention, self.store.base_dir / "retention.log")
            with self.assertRaisesRegex(ValueError, "symlinked.*retention"):
                delete_tenant_data(self.store, "tenant-a")

            self.assertIn("outside", external_approval.read_text())
            self.assertIn("outside", external_dataset.read_text())
            self.assertIn("external-link", external_retention.read_text())
            self.assertTrue(self.store.session_exists("external-link"))

    def test_delete_removes_only_tenant_data_and_writes_audit(self):
        self.create_session("a", "tenant-a")
        self.create_session("b", "tenant-b")
        checkpoints = self.store.base_dir / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "a.md").write_text("private recovery context")
        (checkpoints / "b.md").write_text("keep")
        (self.store.base_dir / ".active-session.customer").write_text("a")
        (self.store.base_dir / ".pending-calls.customer.json").write_text(
            '{"tool":{"arguments":"private"}}'
        )
        orphan_pending = self.store.base_dir / ".pending-calls.a.json"
        orphan_pending.write_text('{"tool":{"arguments":"stale"}}')
        other_pending = self.store.base_dir / ".pending-calls.other-session.json"
        other_pending.write_text('{"tool":{"arguments":"keep"}}')

        result = delete_tenant_data(self.store, "tenant-a")

        self.assertEqual(result.deleted_sessions, 1)
        self.assertEqual(result.deleted_events, 1)
        self.assertFalse(self.store._session_dir("a").exists())
        self.assertFalse((checkpoints / "a.md").exists())
        self.assertTrue(self.store._session_dir("b").exists())
        self.assertTrue((checkpoints / "b.md").exists())
        self.assertFalse((self.store.base_dir / ".active-session.customer").exists())
        self.assertFalse((self.store.base_dir / ".pending-calls.customer.json").exists())
        self.assertFalse(orphan_pending.exists())
        self.assertTrue(other_pending.exists())
        audits = [json.loads(line) for line in result.audit_path.read_text().splitlines()]
        audit = audits[-1]
        self.assertEqual([row["status"] for row in audits], ["pending", "completed"])
        self.assertEqual(
            audit["tenant_id_sha256"], hashlib.sha256(b"tenant-a").hexdigest()
        )
        self.assertNotIn("tenant-a", result.audit_path.read_text())
        self.assertNotIn('"a"', result.audit_path.read_text())
        self.assertEqual(len(audit["session_id_sha256"]), 1)
        with self.assertRaisesRegex(ValueError, "erased"):
            self.store.create_session(SessionMeta(session_id="a", tenant_id="tenant-a"))
        with self.assertRaisesRegex(ValueError, "erased"):
            self.store.append_event("a", _event("a", tenant_id="tenant-a"))

    def test_delete_command_requires_explicit_confirmation(self):
        self.create_session("a", "tenant-a")
        args = build_parser().parse_args([
            "--trace-dir", self.tempdir.name, "tenant", "delete", "tenant-a",
        ])
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(cmd_tenant(args), 1)
        self.assertIn("--confirm", stderr.getvalue())
        self.assertTrue(self.store.session_exists("a"))

    def test_checkpoint_parent_symlink_fails_without_touching_external_file(self):
        self.create_session("checkpoint-link", "tenant-a")
        with tempfile.TemporaryDirectory() as external_dir:
            external = Path(external_dir) / "checkpoint-link.md"
            external.write_text("outside-store-private-data")
            os.symlink(external_dir, self.store.base_dir / "checkpoints")

            with self.assertRaisesRegex(ValueError, "symlinked.*checkpoint"):
                tenant_export_payload(self.store, "tenant-a")
            with self.assertRaisesRegex(ValueError, "symlinked.*checkpoint"):
                delete_tenant_data(self.store, "tenant-a")

            self.assertEqual(external.read_text(), "outside-store-private-data")
            self.assertTrue(self.store.session_exists("checkpoint-link"))
            self.assertFalse(
                (self.store.base_dir / "tenant-deletions.ndjson").exists()
            )

    def test_core_file_symlinks_are_rejected_without_following(self):
        event_meta = self.create_session("event-link", "tenant-a")
        with tempfile.TemporaryDirectory() as external_dir:
            external_events = Path(external_dir) / "events.ndjson"
            external_events.write_text("outside-events")
            events_path = self.store._session_dir(event_meta.session_id) / "events.ndjson"
            events_path.unlink()
            os.symlink(external_events, events_path)

            with self.assertRaisesRegex(ValueError, "symlinked event stream"):
                self.store.load_events(event_meta.session_id)
            with self.assertRaisesRegex(ValueError, "symlinked.*(event|core)"):
                delete_tenant_data(self.store, "tenant-a")
            self.assertEqual(external_events.read_text(), "outside-events")
            self.assertTrue(self.store._session_dir(event_meta.session_id).exists())

        meta = SessionMeta(session_id="meta-link")
        self.store.create_session(meta)
        with tempfile.TemporaryDirectory() as external_dir:
            external_meta = Path(external_dir) / "meta.json"
            external_meta.write_text(meta.to_json())
            meta_path = self.store._session_dir(meta.session_id) / "meta.json"
            meta_path.unlink()
            os.symlink(external_meta, meta_path)
            with self.assertRaisesRegex(ValueError, "symlinked session metadata"):
                self.store.load_meta(meta.session_id)
            self.assertEqual(external_meta.read_text(), meta.to_json())

    def test_tenant_audit_symlinks_are_rejected_without_following(self):
        meta = SessionMeta(session_id="tag-audit-link")
        self.store.create_session(meta)
        self.store.append_event(meta.session_id, _event(meta.session_id))
        with tempfile.TemporaryDirectory() as external_dir:
            external_audit = Path(external_dir) / "audit.ndjson"
            external_audit.write_text("outside-audit\n")
            os.symlink(external_audit, self.store.base_dir / "tenant-audit.ndjson")
            with self.assertRaisesRegex(ValueError, "symlinked tenant audit"):
                self.store.tag_session(meta.session_id, "tenant-a")
            self.assertEqual(external_audit.read_text(), "outside-audit\n")
            self.assertEqual(self.store.load_meta(meta.session_id).tenant_id, "")
        (self.store.base_dir / "tenant-audit.ndjson").unlink()

        tagged = self.create_session("delete-audit-link", "tenant-b")
        with tempfile.TemporaryDirectory() as external_dir:
            external_audit = Path(external_dir) / "deletions.ndjson"
            external_audit.write_text("outside-deletions\n")
            os.symlink(
                external_audit, self.store.base_dir / "tenant-deletions.ndjson"
            )
            with self.assertRaisesRegex(ValueError, "symlinked.*tenant-deletions"):
                delete_tenant_data(self.store, "tenant-b")
            self.assertEqual(external_audit.read_text(), "outside-deletions\n")
            self.assertTrue(self.store._session_dir(tagged.session_id).exists())

    def test_report_export_and_confirmed_delete_command_paths(self):
        june = 1_781_481_600.0
        self.create_session("a", "tenant-a", june)
        parser = build_parser()

        report_args = parser.parse_args([
            "--trace-dir", self.tempdir.name,
            "tenant", "report", "--month", "2026-06", "--format", "json",
        ])
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(cmd_tenant(report_args), 0)
            report = json.loads(stdout.getvalue())
        self.assertEqual(report["tenants"][0]["tenant_id"], "tenant-a")

        export_args = parser.parse_args([
            "--trace-dir", self.tempdir.name, "tenant", "export", "tenant-a",
        ])
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(cmd_tenant(export_args), 0)
            exported = json.loads(stdout.getvalue())
        self.assertEqual(exported["session_count"], 1)

        delete_args = parser.parse_args([
            "--trace-dir", self.tempdir.name,
            "tenant", "delete", "tenant-a", "--confirm",
        ])
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(cmd_tenant(delete_args), 0)
        self.assertFalse(self.store.session_exists("a"))

    def test_flat_and_all_workspaces_are_reported_exported_and_deleted(self):
        flat = TraceStore(self.tempdir.name, use_workspace_env=False, redact=False)
        one = TraceStore(
            self.tempdir.name, workspace_id="one", use_workspace_env=False, redact=False,
        )
        two = TraceStore(
            self.tempdir.name, workspace_id="two", use_workspace_env=False, redact=False,
        )
        june = 1_781_481_600.0
        for index, store in enumerate((flat, one, two)):
            meta = SessionMeta(session_id=f"scoped-{index}", tenant_id="tenant-global", started_at=june)
            store.create_session(meta)
            store.append_event(meta.session_id, _event(meta.session_id, june))
        sidecar = one._session_dir("scoped-1") / "annotations.jsonl"
        sidecar.write_text('{"note":"included"}\n')
        checkpoints = two.base_dir / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "scoped-2.md").write_text("checkpoint")
        (flat.storage_root / ".active-session.workspace").write_text("scoped-1")
        (flat.storage_root / ".pending-calls.workspace.json").write_text("private")

        with patch.dict(os.environ, {"AGENT_STRACE_WORKSPACE": "one"}):
            stores = enumerate_trace_stores(self.tempdir.name)
            report = build_tenant_report(stores, "2026-06")
            exported = tenant_export_payload(stores, "tenant-global")
            deleted = delete_tenant_data(stores, "tenant-global")

        self.assertEqual(report.total_sessions, 3)
        self.assertEqual(exported["schema"], "agent-strace-tenant-export/v1")
        self.assertEqual(exported["session_count"], 3)
        self.assertTrue(any(row["sidecars"] for row in exported["sessions"]))
        self.assertTrue(any(row["checkpoint"] for row in exported["sessions"]))
        self.assertEqual(deleted.deleted_sessions, 3)
        self.assertEqual(len(deleted.audit_paths), 3)
        self.assertFalse((flat.storage_root / ".active-session.workspace").exists())
        self.assertFalse((flat.storage_root / ".pending-calls.workspace.json").exists())

    def test_legacy_flat_and_workspace_stores_are_reported_exported_and_deleted(self):
        june = 1_781_481_600.0
        legacy_id = "Legacy customer session ü" + "u" * 130
        workspace_id = "Customer Workspace é" + "w" * 130
        second_id = "Workspace legacy session " + "z" * 130
        self.write_legacy_session(
            self.store.base_dir, legacy_id, "tenant-legacy", june
        )
        workspace_base = Path(self.tempdir.name) / "workspaces" / workspace_id
        self.write_legacy_session(
            workspace_base,
            second_id,
            "tenant-legacy",
            june,
            workspace_id=workspace_id,
        )

        with self.assertRaises(ValueError):
            TraceStore(
                self.tempdir.name,
                workspace_id=workspace_id,
                use_workspace_env=False,
                redact=False,
            )
        stores = enumerate_trace_stores(self.tempdir.name)
        report = build_tenant_report(stores, "2026-06")
        exported = tenant_export_payload(stores, "tenant-legacy")

        self.assertEqual(report.total_sessions, 2)
        self.assertEqual(exported["session_count"], 2)
        self.assertEqual(
            {row["session"]["session_id"] for row in exported["sessions"]},
            {legacy_id, second_id},
        )
        self.assertIn(
            workspace_id,
            {row["scope"]["workspace_id"] for row in exported["sessions"]},
        )

        deleted = delete_tenant_data(stores, "tenant-legacy")
        self.assertEqual(deleted.deleted_sessions, 2)
        self.assertFalse((self.store.base_dir / legacy_id).exists())
        self.assertFalse((workspace_base / second_id).exists())

    def test_deletion_intent_journal_recovers_after_restart(self):
        self.create_session("recover-delete", "tenant-a")
        operation_id = "d" * 32
        journal_dir = self.store.base_dir / ".tenant-journal"
        journal_dir.mkdir()
        journal = {
            "operation": "delete_tenant", "operation_id": operation_id,
            "tenant_id": "tenant-a", "workspace_id": "", "phase": "intent",
            "sessions": [{"session_id": "recover-delete", "event_count": 1}],
        }
        (journal_dir / f"delete-{operation_id}.json").write_text(json.dumps(journal))

        TraceStore(self.tempdir.name, redact=False)

        self.assertFalse(self.store.session_exists("recover-delete"))
        self.assertFalse((journal_dir / f"delete-{operation_id}.json").exists())
        audits = [
            json.loads(line)
            for line in (self.store.base_dir / "tenant-deletions.ndjson").read_text().splitlines()
        ]
        self.assertEqual([record["status"] for record in audits], ["pending", "completed"])

    def test_non_string_delete_journal_is_left_for_operator_recovery(self):
        self.create_session("invalid-delete-journal", "False")
        operation_id = "9" * 32
        journal_dir = self.store.base_dir / ".tenant-journal"
        journal_dir.mkdir()
        journal_path = journal_dir / f"delete-{operation_id}.json"
        journal_path.write_text(json.dumps({
            "operation": "delete_tenant",
            "operation_id": operation_id,
            "tenant_id": False,
            "workspace_id": "",
            "phase": "intent",
            "sessions": [{
                "session_id": "invalid-delete-journal",
                "event_count": 1,
            }],
        }))

        recovered = TraceStore(self.tempdir.name, redact=False)

        self.assertTrue(recovered.session_exists("invalid-delete-journal"))
        self.assertTrue(journal_path.exists())
        self.assertFalse((recovered.base_dir / "tenant-deletions.ndjson").exists())

    def test_cross_tenant_subagent_link_is_rejected(self):
        root = SessionMeta(session_id="tree-root", tenant_id="tenant-a")
        self.store.create_session(root)
        with self.assertRaisesRegex(ValueError, "cross-tenant"):
            self.store.create_session(SessionMeta(
                session_id="tree-child", tenant_id="tenant-b",
                parent_session_id=root.session_id, depth=1,
            ))
        child = self.write_legacy_session(
            self.store.base_dir,
            "legacy-tree-child",
            "tenant-b",
            parent_session_id=root.session_id,
        )
        with self.assertRaisesRegex(ValueError, "cross-tenant"):
            build_tree(self.store, root.session_id)

    def test_workspace_hook_state_and_legacy_root_cleanup_do_not_collide(self):
        raw_session = "shared-provider-session-123456"
        local_session = raw_session[:16]
        root = Path(self.tempdir.name)
        for workspace, tenant in (("one", "tenant-one"), ("two", "tenant-two")):
            with patch.dict(os.environ, {
                "AGENT_TRACE_DIR": self.tempdir.name,
                "AGENT_STRACE_WORKSPACE": workspace,
                "AGENT_STRACE_TENANT_ID": tenant,
            }):
                handle_session_start({"session_id": raw_session, "source": "startup"})
                handle_pre_tool({
                    "session_id": raw_session,
                    "tool_name": "Read",
                    "tool_input": {"path": workspace},
                })

        one = TraceStore(
            self.tempdir.name, workspace_id="one", use_workspace_env=False, redact=False
        )
        two = TraceStore(
            self.tempdir.name, workspace_id="two", use_workspace_env=False, redact=False
        )
        state_digest = hashlib.sha256(f"claude\0{raw_session}".encode()).hexdigest()
        local_hex = raw_session[:16].encode().hex()
        state_suffix = f".v2.claude.{local_hex}.{state_digest}"
        scoped_marker = f".active-session{state_suffix}"
        pending = f".pending-calls{state_suffix}.json"
        self.assertTrue((one.base_dir / scoped_marker).exists())
        self.assertTrue((two.base_dir / scoped_marker).exists())
        self.assertTrue((one.base_dir / pending).exists())
        self.assertTrue((two.base_dir / pending).exists())

        legacy_marker = root / ".active-session.legacy"
        legacy_pending = root / ".pending-calls.legacy.json"
        legacy_marker.write_text(local_session, encoding="utf-8")
        legacy_pending.write_text("{}", encoding="utf-8")

        delete_tenant_data(one, "tenant-one")
        self.assertFalse((one.base_dir / scoped_marker).exists())
        self.assertFalse((one.base_dir / pending).exists())
        self.assertTrue((two.base_dir / scoped_marker).exists())
        self.assertTrue((two.base_dir / pending).exists())
        self.assertTrue(legacy_marker.exists())
        self.assertTrue(legacy_pending.exists())

        delete_tenant_data(two, "tenant-two")
        self.assertFalse(legacy_marker.exists())
        self.assertFalse(legacy_pending.exists())


class TestTenantOTLPAndRemoteHooks(TenantTestCase):
    def test_tenant_attribute_is_on_every_exported_span_and_resource(self):
        meta = self.create_session("otlp", "tenant-a")
        events = self.store.load_events(meta.session_id)
        for converter in (session_to_otlp, session_to_otlp_genai):
            payload = converter(meta, events)
            resource = payload["resourceSpans"][0]
            self.assertEqual(_attr_values(resource["resource"])["tenant_id"], "tenant-a")
            for span in resource["scopeSpans"][0]["spans"]:
                self.assertEqual(_attr_values(span)["tenant_id"], "tenant-a")

    def test_remote_hook_posts_tenant_session_metadata(self):
        with patch.dict(os.environ, {
            "AGENT_TRACE_DIR": self.tempdir.name,
            "AGENT_STRACE_ENDPOINT": "https://collector.invalid",
            "AGENT_STRACE_TENANT_ID": "tenant-a",
        }), patch(
            "agent_trace.server.send_session_meta_to_endpoint", return_value=True
        ) as send_meta, patch(
            "agent_trace.server.send_event_to_endpoint", return_value=True
        ):
            handle_session_start({"session_id": "remote-session", "source": "startup"})

        sent_meta = send_meta.call_args.args[0]
        self.assertEqual(sent_meta.tenant_id, "tenant-a")
        self.assertEqual(sent_meta.session_id, "remote-session"[:16])


PAST_LIMIT = "is past the latest time OTLP can carry (2554-07-21T23:34:33.709551615Z)"
YEAR_3000 = 32503680000.0

# Each meta.json tenant_id the store would refuse, with the reason the line gives.
REFUSED_TENANTS = (
    ("null", None, "must be a string"),
    ("zero", 0, "must be a string"),
    ("false", False, "must be a string"),
    ("empty array", [], "must be a string"),
    ("empty object", {}, "must be a string"),
    ("number", 5, "must be a string"),
    ("true", True, "must be a string"),
    ("float", 1.5, "must be a string"),
    ("NaN", float("nan"), "must be a string"),
    ("Infinity", float("inf"), "must be a string"),
    ("array", ["acme"], "must be a string"),
    ("object", {"a": 1}, "must be a string"),
    ("whitespace", "   ", "cannot be empty"),
    ("control character", "ac\x01me", "cannot contain control characters"),
    ("300 characters", "x" * 300, "cannot exceed 256 characters"),
)
# build_tree refuses a tree whose sessions have different tenants, and NaN
# differs from itself. test_a_tree_names_the_session_whose_tenant_is_refused
# covers NaN in a tree.
TREE_REFUSED_TENANTS = tuple(case for case in REFUSED_TENANTS if case[0] != "NaN")
# Each accepted tenant_id, with the tenant_id values the resource and each span carry.
ACCEPTED_TENANTS = (
    (" acme ", [{"stringValue": "acme"}]),
    ("", []),
)

OTLP_ROUTES = (
    (),
    ("-o", "out.json"),
    ("--endpoint", ENDPOINT, "--protocol", "http/json"),
    ("--endpoint", ENDPOINT, "--protocol", "http/protobuf"),
)
SINGLE_PATHS = tuple(
    ("export", BAD, "--format", fmt, *route)
    for fmt in ("otlp", "otlp-genai", "temporal")
    for route in OTLP_ROUTES
)
TREE_PATHS = tuple(
    ("export", BAD, "--format", fmt, "--include-subagents", *route)
    for fmt in ("otlp", "otlp-genai")
    for route in OTLP_ROUTES
)
WRITTEN_PATHS = (
    ("export", BAD, "--format", "otlp"),
    ("export", BAD, "--format", "otlp-genai"),
    ("export", BAD, "--format", "temporal"),
    ("export", BAD, "--format", "otlp", "--include-subagents"),
    ("export", BAD, "--format", "otlp-genai", "--include-subagents"),
)


def _tenant_values(payload: dict) -> list[list[dict]]:
    """The tenant_id values of each resource and each span, one list for each."""
    holders = []
    for resource_spans in payload["resourceSpans"]:
        holders.append(resource_spans["resource"])
        for scope_spans in resource_spans["scopeSpans"]:
            holders.extend(scope_spans["spans"])
    return [
        [attr["value"] for attr in holder.get("attributes", []) if attr["key"] == "tenant_id"]
        for holder in holders
    ]


def _tenant_key_value(tenant: str) -> bytes:
    """The protobuf KeyValue of a tenant_id string attribute."""
    text = tenant.encode()
    any_value = b"\x0a" + bytes([len(text)]) + text
    return b"\x0a\x09tenant_id\x12" + bytes([len(any_value)]) + any_value


class TestOTLPBuildersRefuseUnusableTenant(unittest.TestCase):
    """Every OTLP builder refuses a meta.json tenant_id the store would refuse."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.traces = Path(tmp.name) / ".agent-traces"
        build_store(self.traces)
        self.store = TraceStore(str(self.traces), use_workspace_env=False, redact=False)

    def builders(self, tenant_id) -> dict:
        set_meta(
            self.traces / GOOD,
            tenant_id=tenant_id, trace_id=TRACE_ID, parent_span_id=PARENT_SPAN_ID,
        )
        meta = self.store.load_meta(GOOD)
        events = self.store.load_events(GOOD)
        return {
            "session_to_otlp": lambda: session_to_otlp(meta, events),
            "session_to_otlp_genai": lambda: session_to_otlp_genai(meta, events),
            "node_tree_to_otlp": lambda: node_tree_to_otlp(build_tree(self.store, GOOD)),
            "node_tree_to_otlp genai": lambda: node_tree_to_otlp(
                build_tree(self.store, GOOD), genai=True,
            ),
            "session_to_temporal_otlp": lambda: session_to_temporal_otlp(meta, events),
            "api_session_otlp": lambda: json.loads(api_session_otlp(self.store, GOOD)),
        }

    def test_every_builder_refuses_a_tenant_the_store_would_refuse(self):
        checked = ("session_to_temporal_otlp", "api_session_otlp")
        for name, value, reason in REFUSED_TENANTS:
            for builder, build in self.builders(value).items():
                with self.subTest(tenant=name, builder=builder):
                    if builder in checked:
                        error, message = UnusableTenantError, f"session tenant_id {reason}"
                    else:
                        error, message = ValueError, f"tenant ID {reason}"
                    with self.assertRaises(error) as context:
                        build()
                    self.assertEqual(str(context.exception), message)

    def test_every_builder_exports_an_accepted_tenant_stripped(self):
        for value, expected in ACCEPTED_TENANTS:
            for builder, build in self.builders(value).items():
                with self.subTest(tenant=value, builder=builder):
                    values = _tenant_values(build())
                    self.assertGreaterEqual(len(values), 3)
                    self.assertEqual(values, [expected] * len(values))


class TenantCliTestCase(TimestampCliTestCase):
    def tenant_workspace(self, tenant_id, **store_options) -> Path:
        """A workspace whose BAD and CHILD sessions store *tenant_id* as given.

        BAD also gets a Temporal trace context.
        """
        root = self.make_workspace(**store_options)
        traces = root / "work" / ".agent-traces"
        set_meta(
            traces / BAD,
            tenant_id=tenant_id, trace_id=TRACE_ID, parent_span_id=PARENT_SPAN_ID,
        )
        set_meta(traces / CHILD, tenant_id=tenant_id)
        return root


class TestOTLPExportsRefuseUnusableTenant(TenantCliTestCase):
    """Every OTLP export path refuses a session whose tenant_id the store would refuse."""

    def test_every_otlp_path_refuses_a_tenant_the_store_would_refuse(self):
        for name, value, reason in REFUSED_TENANTS:
            root = self.tenant_workspace(value)
            for argv in SINGLE_PATHS:
                with self.subTest(tenant=name, argv=argv):
                    self.assert_refused(root, list(argv), f"Error: session tenant_id {reason}\n")
        for name, value, reason in TREE_REFUSED_TENANTS:
            root = self.tenant_workspace(value)
            for argv in TREE_PATHS:
                with self.subTest(tenant=name, argv=argv):
                    self.assert_refused(
                        root, list(argv), f"Error: session {BAD} tenant_id {reason}\n",
                    )

    def test_a_tree_names_the_session_whose_tenant_is_refused(self):
        root = self.make_workspace(tenant=True)
        for name, value, reason in REFUSED_TENANTS:
            # build_tree refuses a child whose tenant differs from the root's,
            # so the child gets its tenant after the build.
            def child_tenant(store, session_id, value=value):
                tree = build_tree(store, session_id)
                for child in tree.children:
                    child.meta.tenant_id = value
                return tree

            with patch("agent_trace.subagent.build_tree", child_tenant):
                for argv in TREE_PATHS:
                    with self.subTest(tenant=name, argv=argv):
                        self.assert_refused(
                            root, list(argv), f"Error: session {CHILD} tenant_id {reason}\n",
                        )

    def test_an_accepted_tenant_is_exported_stripped(self):
        out = Path("out.json")
        for value, expected in ACCEPTED_TENANTS:
            root = self.tenant_workspace(value)
            for argv in WRITTEN_PATHS:
                with self.subTest(tenant=value, argv=argv):
                    result = self.run_cli(root, *argv, "-o", str(out))
                    self.assertEqual(result.code, 0, result.err)
                    written = root / "work" / out
                    values = _tenant_values(json.loads(written.read_text(encoding="utf-8")))
                    written.unlink()
                    self.assertGreaterEqual(len(values), 3)
                    self.assertEqual(values, [expected] * len(values))

    def test_a_session_the_other_checks_refuse_keeps_their_line(self):
        root = self.tenant_workspace(5)
        for argv in SINGLE_PATHS:
            with self.subTest(case="tenant alone", argv=argv):
                self.assert_refused(root, list(argv), "Error: session tenant_id must be a string\n")

        every_format = ("otlp", "otlp-genai", "temporal")
        # Each case: the formats it covers, the build_store options, a change
        # to the stored BAD session, and the line.
        cases = (
            ("non-object data", every_format, {"bad_type": "tool_call", "value": "text"}, None,
             expected_error(BAD, "tool_call", "string", named=False)),
            ("null event timestamp", every_format, {},
             lambda bad: restamp(bad, "tool_call", None),
             expected_ts_error(BAD, "tool_call", "null", named=False)),
            ("null started_at", every_format, {},
             lambda bad: set_meta(bad, started_at=None),
             "Error: session started_at must be a Unix time in seconds\n"),
            ("started_at past the OTLP limit", every_format, {},
             lambda bad: set_meta(bad, started_at=YEAR_3000),
             f"Error: session started_at {PAST_LIMIT}\n"),
            ("event past the OTLP limit", every_format, {},
             lambda bad: restamp(bad, "llm_request", YEAR_3000),
             f"Error: session event timestamp {PAST_LIMIT}: event 3 of 25 (llm_request); "
             "1 more event has a timestamp past that time\n"),
            ("no Temporal trace context", ("temporal",), {},
             lambda bad: set_meta(bad, trace_id="", parent_span_id=""),
             "Temporal export failed: session has no Temporal trace context; "
             f"run watch with {TEMPORAL_TRACE_PARENT_ENV} set\n"),
            ("invalid trace ID", ("temporal",), {},
             lambda bad: set_meta(bad, trace_id="z" * 32),
             "Temporal export failed: traceparent contains an invalid trace ID\n"),
        )
        for case, formats, options, change, expected in cases:
            root = self.tenant_workspace(5, **options)
            if change:
                change(root / "work" / ".agent-traces" / BAD)
            for argv in SINGLE_PATHS:
                if argv[3] in formats:
                    with self.subTest(case=case, argv=argv):
                        self.assert_refused(root, list(argv), expected)

    def test_a_tree_checks_tenants_after_every_session_passes_the_other_checks(self):
        root = self.tenant_workspace(5)
        for argv in TREE_PATHS:
            with self.subTest(case="tenant alone", argv=argv):
                self.assert_refused(
                    root, list(argv), f"Error: session {BAD} tenant_id must be a string\n",
                )
        for case, session_id, fields, expected in (
            ("root started_at", BAD, {"started_at": None},
             f"Error: session {BAD} started_at must be a Unix time in seconds\n"),
            ("child started_at", CHILD, {"started_at": float("nan")},
             f"Error: session {CHILD} started_at must be a Unix time in seconds\n"),
            ("child past the OTLP limit", CHILD, {"started_at": YEAR_3000},
             f"Error: session {CHILD} started_at {PAST_LIMIT}\n"),
        ):
            root = self.tenant_workspace(5)
            set_meta(root / "work" / ".agent-traces" / session_id, **fields)
            for argv in TREE_PATHS:
                with self.subTest(case=case, argv=argv):
                    self.assert_refused(root, list(argv), expected)


class _OTLPSink(BaseHTTPRequestHandler):
    """Record each POST and answer 200."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.requests.append((self.path, self.headers.get("Content-Type"), body))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args):
        pass


class TestOTLPPushesRefuseUnusableTenant(TenantCliTestCase):
    """A local collector gets nothing for a refused tenant and the accepted tenant stripped."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sink = HTTPServer(("127.0.0.1", 0), _OTLPSink)
        cls.sink.requests = []
        cls.sink_thread = threading.Thread(target=cls.sink.serve_forever, daemon=True)
        cls.sink_thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.sink.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.sink.shutdown()
        cls.sink.server_close()
        cls.sink_thread.join()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.sink.requests.clear()

    def push(self, root: Path, *argv: str) -> tuple[int, str, str]:
        """Run agent-strace with the network open, and return its exit code and output."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(_chdir(root / "work"))
            stack.enter_context(_isolated_env(root / "home"))
            stack.enter_context(patch.dict(os.environ, {"no_proxy": "*", "NO_PROXY": "*"}))
            stack.enter_context(patch("agent_trace.cli._product_telemetry.capture"))
            stack.enter_context(
                patch("agent_trace.cli._product_telemetry.maybe_show_telemetry_notice")
            )
            stack.enter_context(patch.object(sys, "argv", ["agent-strace", *argv]))
            captured = stack.enter_context(_capture_output())
            try:
                main()
                code = 0
            except SystemExit as exc:
                code = 0 if exc.code is None else exc.code
        return code, captured.out, captured.err

    def endpoint_argv(self, fmt: str, protocol: str, *extra: str) -> tuple:
        return (
            "export", BAD, "--format", fmt, *extra,
            "--endpoint", self.endpoint, "--protocol", protocol,
        )

    def test_a_refused_tenant_sends_nothing(self):
        protocols = ("http/json", "http/protobuf")
        for name, value, reason in REFUSED_TENANTS:
            root = self.tenant_workspace(value)
            paths = [
                (self.endpoint_argv(fmt, protocol), f"Error: session tenant_id {reason}\n")
                for fmt in ("otlp", "otlp-genai", "temporal")
                for protocol in protocols
            ]
            if name != "NaN":
                paths += [
                    (
                        self.endpoint_argv(fmt, protocol, "--include-subagents"),
                        f"Error: session {BAD} tenant_id {reason}\n",
                    )
                    for fmt in ("otlp", "otlp-genai")
                    for protocol in protocols
                ]
            for argv, expected in paths:
                with self.subTest(tenant=name, argv=argv):
                    self.assertEqual(self.push(root, *argv), (1, "", expected))
                    self.assertEqual(self.sink.requests, [])

    def test_an_accepted_tenant_is_sent_stripped(self):
        for value, expected in ACCEPTED_TENANTS:
            root = self.tenant_workspace(value)
            for fmt, extra in (
                ("otlp", ()),
                ("otlp-genai", ()),
                ("temporal", ()),
                ("otlp", ("--include-subagents",)),
                ("otlp-genai", ("--include-subagents",)),
            ):
                with self.subTest(tenant=value, format=fmt, extra=extra):
                    bodies = {}
                    for protocol in ("http/json", "http/protobuf"):
                        self.sink.requests.clear()
                        code, _, err = self.push(root, *self.endpoint_argv(fmt, protocol, *extra))
                        self.assertEqual(code, 0, err)
                        self.assertEqual(
                            [path for path, _, _ in self.sink.requests], ["/v1/traces"],
                        )
                        bodies[protocol] = self.sink.requests[0][2]
                    values = _tenant_values(json.loads(bodies["http/json"]))
                    self.assertGreaterEqual(len(values), 3)
                    self.assertEqual(values, [expected] * len(values))
                    protobuf = bodies["http/protobuf"]
                    if expected:
                        self.assertEqual(protobuf.count(b"\x0a\x09tenant_id"), len(values))
                        self.assertEqual(protobuf.count(_tenant_key_value("acme")), len(values))
                    else:
                        self.assertNotIn(b"tenant_id", bodies["http/json"])
                        self.assertNotIn(b"tenant_id", protobuf)


if __name__ == "__main__":
    unittest.main()
