"""Tests for workspace isolation (issue #135)."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore, _WORKSPACE_ENV, _workspace_base
from agent_trace.workspace import (
    list_workspaces,
    create_workspace,
    delete_workspace,
    workspace_session_count,
)


def _make_session(store: TraceStore) -> str:
    meta = SessionMeta(agent_name="test")
    meta.started_at = time.time() - 60
    meta.ended_at = time.time()
    store.create_session(meta)
    store.update_meta(meta)
    return meta.session_id


class TestWorkspaceStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.pop(_WORKSPACE_ENV, None)

    def tearDown(self):
        os.environ.pop(_WORKSPACE_ENV, None)

    def test_workspace_scopes_base_dir(self):
        store = TraceStore(self.tmpdir, workspace_id="alpha")
        expected = _workspace_base(self.tmpdir, "alpha")
        self.assertEqual(store.base_dir, expected)

    def test_no_workspace_uses_flat_layout(self):
        store = TraceStore(self.tmpdir)
        from pathlib import Path
        self.assertEqual(store.base_dir, Path(self.tmpdir))

    def test_env_var_sets_workspace(self):
        os.environ[_WORKSPACE_ENV] = "beta"
        store = TraceStore(self.tmpdir)
        self.assertEqual(store.workspace_id, "beta")

    def test_explicit_workspace_overrides_env(self):
        os.environ[_WORKSPACE_ENV] = "env-ws"
        store = TraceStore(self.tmpdir, workspace_id="explicit-ws")
        self.assertEqual(store.workspace_id, "explicit-ws")

    def test_sessions_isolated_between_workspaces(self):
        store_a = TraceStore(self.tmpdir, workspace_id="ws-a")
        store_b = TraceStore(self.tmpdir, workspace_id="ws-b")
        sid_a = _make_session(store_a)
        sid_b = _make_session(store_b)

        sessions_a = [m.session_id for m in store_a.list_sessions()]
        sessions_b = [m.session_id for m in store_b.list_sessions()]

        self.assertIn(sid_a, sessions_a)
        self.assertNotIn(sid_b, sessions_a)
        self.assertIn(sid_b, sessions_b)
        self.assertNotIn(sid_a, sessions_b)

    def test_workspace_id_stamped_on_meta(self):
        store = TraceStore(self.tmpdir, workspace_id="stamped-ws")
        sid = _make_session(store)
        meta = store.load_meta(sid)
        self.assertEqual(meta.workspace_id, "stamped-ws")

    def test_flat_store_does_not_stamp_workspace(self):
        store = TraceStore(self.tmpdir)
        sid = _make_session(store)
        meta = store.load_meta(sid)
        self.assertEqual(meta.workspace_id, "")


class TestWorkspaceCommands(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_list_empty(self):
        result = list_workspaces(self.tmpdir)
        self.assertEqual(result, [])

    def test_create_and_list(self):
        create_workspace("ws-1", self.tmpdir)
        create_workspace("ws-2", self.tmpdir)
        result = list_workspaces(self.tmpdir)
        self.assertIn("ws-1", result)
        self.assertIn("ws-2", result)

    def test_create_idempotent(self):
        create_workspace("ws-x", self.tmpdir)
        create_workspace("ws-x", self.tmpdir)  # should not raise
        self.assertIn("ws-x", list_workspaces(self.tmpdir))

    def test_delete_existing(self):
        create_workspace("ws-del", self.tmpdir)
        existed = delete_workspace("ws-del", self.tmpdir)
        self.assertTrue(existed)
        self.assertNotIn("ws-del", list_workspaces(self.tmpdir))

    def test_delete_nonexistent_returns_false(self):
        existed = delete_workspace("no-such-ws", self.tmpdir)
        self.assertFalse(existed)

    def test_session_count(self):
        store = TraceStore(self.tmpdir, workspace_id="counted")
        _make_session(store)
        _make_session(store)
        count = workspace_session_count("counted", self.tmpdir)
        self.assertEqual(count, 2)

    def test_session_count_empty(self):
        create_workspace("empty-ws", self.tmpdir)
        self.assertEqual(workspace_session_count("empty-ws", self.tmpdir), 0)


if __name__ == "__main__":
    unittest.main()
