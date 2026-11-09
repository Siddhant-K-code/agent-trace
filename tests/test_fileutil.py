"""Tests for the cross-process file lock, atomic write, and symlink trust helpers."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace import fileutil
from agent_trace.fileutil import (
    MAX_TRUSTED_SYMLINKS,
    file_lock,
    has_untrusted_symlink,
    lock_path_for,
    trusted_symlink_target,
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


def _stat_result(kind: int, mode: int, uid: int) -> os.stat_result:
    return os.stat_result((kind | mode, 0, 0, 1, uid, 0, 0, 0, 0, 0))


_ROOT_LINK = _stat_result(stat.S_IFLNK, 0o755, 0)
_ROOT_DIR = _stat_result(stat.S_IFDIR, 0o755, 0)


def _system_root_owned_link() -> str | None:
    """Return an OS symlink in / that root owns, such as /tmp on macOS."""
    if os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() == 0:
        return None
    root = os.stat("/")
    if root.st_uid != 0 or root.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return None
    for candidate in ("/tmp", "/var", "/etc", "/bin", "/sbin", "/lib"):
        try:
            info = os.lstat(candidate)
        except OSError:
            continue
        if stat.S_ISLNK(info.st_mode) and info.st_uid == 0:
            return candidate
    return None


class TestSymlinkTrustPolicy(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.object(fileutil, "_WINDOWS", False),
            patch.object(fileutil.os, "geteuid", create=True, return_value=1000),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_root_owned_link_in_root_owned_directory_is_trusted(self):
        self.assertTrue(fileutil._symlink_is_trusted(_ROOT_LINK, _ROOT_DIR))

    def test_link_owned_by_current_user_is_refused(self):
        link = _stat_result(stat.S_IFLNK, 0o755, 1000)
        self.assertFalse(fileutil._symlink_is_trusted(link, _ROOT_DIR))

    def test_directory_owned_by_current_user_is_refused(self):
        directory = _stat_result(stat.S_IFDIR, 0o755, 1000)
        self.assertFalse(fileutil._symlink_is_trusted(_ROOT_LINK, directory))

    def test_writable_directory_is_refused_unless_sticky(self):
        for mode, expected in (
            (0o775, False), (0o757, False), (0o777, False), (0o1777, True), (0o1775, True),
        ):
            with self.subTest(mode=oct(mode)):
                directory = _stat_result(stat.S_IFDIR, mode, 0)
                self.assertIs(fileutil._symlink_is_trusted(_ROOT_LINK, directory), expected)

    def test_non_link_or_non_directory_is_refused(self):
        self.assertFalse(fileutil._symlink_is_trusted(_ROOT_DIR, _ROOT_DIR))
        regular = _stat_result(stat.S_IFREG, 0o644, 0)
        self.assertFalse(fileutil._symlink_is_trusted(_ROOT_LINK, regular))

    def test_windows_never_trusts_a_link(self):
        with patch.object(fileutil, "_WINDOWS", True):
            self.assertFalse(fileutil._symlink_is_trusted(_ROOT_LINK, _ROOT_DIR))
            self.assertIsNone(trusted_symlink_target("/tmp"))

    def test_root_process_never_trusts_a_link(self):
        with patch.object(fileutil.os, "geteuid", create=True, return_value=0):
            self.assertFalse(fileutil._symlink_is_trusted(_ROOT_LINK, _ROOT_DIR))
            self.assertIsNone(trusted_symlink_target("/tmp"))


@unittest.skipIf(os.name == "nt" or not hasattr(os, "symlink"), "POSIX symlink resolution")
class TestHasUntrustedSymlink(unittest.TestCase):
    """Walk tests that trust only the links each test marks as trusted."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Resolve first so the walk meets only the links each test makes.
        self.root = Path(os.path.realpath(tmp.name))
        (self.root / "real" / "sub").mkdir(parents=True)
        self.trusted: set[tuple[int, int]] = set()
        for patcher in (
            patch.object(fileutil, "_trusts_root_owned_symlinks", return_value=True),
            patch.object(fileutil, "_symlink_is_trusted", side_effect=self._is_trusted),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _is_trusted(self, link_stat, dir_stat):
        return (link_stat.st_dev, link_stat.st_ino) in self.trusted

    def _link(self, name: str, target: str, *, trusted: bool) -> Path:
        path = self.root / name
        path.symlink_to(target, target_is_directory=True)
        if trusted:
            info = os.lstat(path)
            self.trusted.add((info.st_dev, info.st_ino))
        return path

    def test_real_directories_and_missing_components_are_clean(self):
        self.assertFalse(has_untrusted_symlink(self.root / "real" / "sub"))
        self.assertFalse(has_untrusted_symlink(self.root / "missing" / "child"))
        (self.root / "file").write_text("not a directory")
        self.assertFalse(has_untrusted_symlink(self.root / "file" / "child"))

    def test_untrusted_link_is_found_as_last_or_middle_component(self):
        self._link("plain", "real", trusted=False)
        self.assertTrue(has_untrusted_symlink(self.root / "plain"))
        self.assertTrue(has_untrusted_symlink(self.root / "plain" / "sub"))

    def test_trusted_relative_absolute_and_parent_targets_are_followed(self):
        (self.root / "nested").mkdir()
        self._link("relative", "real", trusted=True)
        self._link("absolute", str(self.root / "real"), trusted=True)
        self._link("nested/up", "../real", trusted=True)
        for name in ("relative", "absolute", "nested/up"):
            with self.subTest(name=name):
                self.assertFalse(has_untrusted_symlink(self.root / name / "sub"))

    def test_untrusted_link_inside_trusted_target_is_found(self):
        (self.root / "other").mkdir()
        self._link("real/inner", "../other", trusted=False)
        self._link("outer", "real", trusted=True)
        self._link("direct", "real/inner", trusted=True)
        self.assertTrue(has_untrusted_symlink(self.root / "outer" / "inner"))
        self.assertTrue(has_untrusted_symlink(self.root / "direct"))
        self.assertFalse(has_untrusted_symlink(self.root / "outer" / "sub"))

    def test_trusted_link_loop_and_long_chain_are_refused(self):
        self._link("loop", "loop", trusted=True)
        self.assertTrue(has_untrusted_symlink(self.root / "loop"))
        previous = "real"
        for index in range(MAX_TRUSTED_SYMLINKS):
            previous = self._link(f"hop{index}", previous, trusted=True).name
        self.assertFalse(has_untrusted_symlink(self.root / previous))
        self._link("one-more", previous, trusted=True)
        self.assertTrue(has_untrusted_symlink(self.root / "one-more"))

    def test_dangling_trusted_link_ends_the_walk(self):
        self._link("dangling", "missing", trusted=True)
        self.assertFalse(has_untrusted_symlink(self.root / "dangling" / "child"))

    def test_relative_path_starts_at_the_working_directory(self):
        self._link("plain", "real", trusted=False)
        self._link("relative", "real", trusted=True)
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.assertTrue(has_untrusted_symlink(Path("plain") / "sub"))
        self.assertFalse(has_untrusted_symlink(Path("relative") / ".." / "real" / "sub"))

    def test_dotdot_onto_untrusted_link_is_found(self):
        self._link("plain", "real", trusted=False)
        path = self.root / "real" / "sub" / ".." / ".." / "plain" / "sub"
        self.assertTrue(has_untrusted_symlink(path))

    def test_dotdot_above_working_directory_onto_untrusted_link_is_found(self):
        self._link("plain", "real", trusted=False)
        previous = os.getcwd()
        os.chdir(self.root / "real")
        self.addCleanup(os.chdir, previous)
        # Path(".").parent is ".", so a lexical ".." would stay in "real".
        self.assertTrue(has_untrusted_symlink(Path("..") / "plain" / "sub"))

    def test_windows_refuses_any_symlink_among_lexical_ancestors(self):
        self._link("relative", "real", trusted=True)
        with patch.object(fileutil, "_WINDOWS", True):
            self.assertTrue(has_untrusted_symlink(self.root / "relative" / "sub"))
            self.assertFalse(has_untrusted_symlink(self.root / "real" / "sub"))


class TestSystemSymlinks(unittest.TestCase):
    def test_root_owned_system_link_is_trusted(self):
        link = _system_root_owned_link()
        if link is None:
            self.skipTest("no root-owned symlink in /, or running as root")
        target = os.readlink(link)
        self.assertEqual(trusted_symlink_target(link), target)
        root_fd = os.open("/", os.O_RDONLY)
        try:
            self.assertEqual(trusted_symlink_target(link[1:], dir_fd=root_fd), target)
        finally:
            os.close(root_fd)
        self.assertFalse(has_untrusted_symlink(link))

    @unittest.skipIf(os.name == "nt" or not hasattr(os, "symlink"), "POSIX symlinks")
    def test_link_made_by_current_user_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "link"
            link.symlink_to(directory, target_is_directory=True)
            self.assertIsNone(trusted_symlink_target(link))
            self.assertTrue(has_untrusted_symlink(link))


if __name__ == "__main__":
    unittest.main()
