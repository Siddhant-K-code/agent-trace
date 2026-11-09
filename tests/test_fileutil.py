"""Tests for the cross-process file lock and atomic write helpers."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.fileutil import (
    file_lock,
    lock_descriptor,
    lock_path_for,
    unlock_descriptor,
    write_atomic,
)

try:
    from _concurrency import run_concurrently
except ImportError:
    from tests._concurrency import run_concurrently

_COUNTER_SCRIPT = """
import json
from pathlib import Path
from agent_trace.fileutil import file_lock, write_atomic
path = Path(sys.argv[2])
wait_for_start()
for _ in range(int(sys.argv[3])):
    with file_lock(path):
        data = json.loads(path.read_text()) if path.exists() else {"count": 0}
        data["count"] += 1
        write_atomic(path, json.dumps(data))
"""


class TestWriteAtomic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_content_and_leaves_no_temp_files(self):
        path = self.dir / "config.json"
        write_atomic(path, "one")
        write_atomic(path, "two")
        self.assertEqual(path.read_text(), "two")
        self.assertEqual([p.name for p in self.dir.iterdir()], ["config.json"])

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_explicit_mode_overrides_existing_mode(self):
        path = self.dir / "auth.json"
        path.write_text("{}")
        path.chmod(0o644)
        write_atomic(path, "{}", mode=0o600)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_existing_mode_is_kept_without_explicit_mode(self):
        path = self.dir / "data.json"
        path.write_text("{}")
        path.chmod(0o640)
        write_atomic(path, "[]")
        self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    @unittest.skipIf(not hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlinks_refused_unless_allowed(self):
        target = self.dir / "target.json"
        target.write_text("{}")
        link = self.dir / "link.json"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("cannot create symlinks")
        with self.assertRaises(ValueError):
            write_atomic(link, "x")
        write_atomic(link, "x", allow_symlinks=True)
        self.assertFalse(link.is_symlink())
        self.assertEqual(link.read_text(), "x")


class TestFileLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "nested" / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_lock_uses_sidecar_file_and_creates_parent(self):
        with file_lock(self.path):
            self.assertTrue(lock_path_for(self.path).exists())
        self.assertFalse(self.path.exists())

    def test_held_lock_times_out_second_holder(self):
        with file_lock(self.path):
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                with file_lock(self.path, timeout=0.2):
                    pass
            self.assertGreaterEqual(time.monotonic() - started, 0.15)

    def test_lock_is_released_after_exception(self):
        with self.assertRaises(RuntimeError):
            with file_lock(self.path):
                raise RuntimeError("boom")
        with file_lock(self.path, timeout=0.5):
            pass

    def test_concurrent_processes_do_not_lose_updates(self):
        workers, rounds = 6, 25
        run_concurrently(
            _COUNTER_SCRIPT,
            [[str(self.path), str(rounds)] for _ in range(workers)],
        )
        self.assertEqual(json.loads(self.path.read_text())["count"], workers * rounds)


class TestLockDescriptor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "store.lock"

    def tearDown(self):
        self.tmp.cleanup()

    def _open(self) -> int:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, fd)
        return fd

    def test_second_descriptor_waits_until_unlock(self):
        first, second = self._open(), self._open()
        lock_descriptor(first)
        with self.assertRaises(TimeoutError):
            lock_descriptor(second, timeout=0.1)
        unlock_descriptor(first)
        lock_descriptor(second, timeout=1.0)
        unlock_descriptor(second)

    def test_lock_leaves_lock_file_empty(self):
        fd = self._open()
        lock_descriptor(fd)
        unlock_descriptor(fd)
        self.assertEqual(self.path.stat().st_size, 0)

    def test_closing_descriptor_releases_lock(self):
        first = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        lock_descriptor(first)
        os.close(first)
        second = self._open()
        lock_descriptor(second, timeout=1.0)
        unlock_descriptor(second)

    def test_unlock_without_lock_is_ignored(self):
        unlock_descriptor(self._open())


if __name__ == "__main__":
    unittest.main()
