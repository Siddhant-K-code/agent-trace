"""Small stdlib helpers for safe file updates across processes.

``write_atomic`` writes to a unique sibling temporary file and renames it over
the target, so readers never see a half-written file and concurrent writers
never share a temporary name.

``file_lock`` holds an exclusive cross-process lock for a path.  The lock
lives in a sibling ``<name>.lock`` file, so the target itself can be replaced
atomically while the lock is held.  Wrap every read-modify-write of a shared
file in ``file_lock`` to avoid lost updates.  The lock file is left in place
after release: deleting it would let two processes lock different inodes.

``trusted_symlink_target`` and ``has_untrusted_symlink`` decide which
symlinks in a path's parent directories may be followed.  Only links that
root owns, in a root-owned directory that no other user can change, are
trusted, such as ``/tmp`` and ``/var`` on macOS.  A cloned repository or an
unpacked archive creates links owned by the current user, so those are never
trusted.
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # POSIX
    msvcrt = None  # type: ignore[assignment]


_POLL_INTERVAL = 0.01

_WINDOWS = os.name == "nt"

# Most trusted symlinks one path walk follows, so a link loop ends.
MAX_TRUSTED_SYMLINKS = 32


def _trusts_root_owned_symlinks() -> bool:
    # Windows reports uid 0 for every file. A process running as root owns
    # every root-owned link, so for it they are the current user's links.
    if _WINDOWS:
        return False
    geteuid = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() != 0


def _symlink_is_trusted(link_stat: os.stat_result, dir_stat: os.stat_result) -> bool:
    """Return True when a symlink in a parent directory may be followed.

    *link_stat* is the link's own ``lstat`` result and *dir_stat* is the
    ``stat`` result of the directory that holds it.  Root must own both, and
    group or other users must not be able to write the directory unless it
    has the sticky bit, so no other user can swap the link.
    """
    if not _trusts_root_owned_symlinks():
        return False
    if not stat.S_ISLNK(link_stat.st_mode) or link_stat.st_uid != 0:
        return False
    if not stat.S_ISDIR(dir_stat.st_mode) or dir_stat.st_uid != 0:
        return False
    if dir_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return bool(dir_stat.st_mode & stat.S_ISVTX)
    return True


def trusted_symlink_target(path: str | Path, *, dir_fd: int | None = None) -> str | None:
    """Return the target of the symlink *path* when it is trusted, else None.

    With *dir_fd*, *path* is one name inside that open directory.  None also
    means *path* is not a symlink or could not be read.
    """
    if not _trusts_root_owned_symlinks():
        return None
    try:
        link_stat = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        dir_stat = os.stat(Path(path).parent, dir_fd=dir_fd)
        if not _symlink_is_trusted(link_stat, dir_stat):
            return None
        target = os.readlink(path, dir_fd=dir_fd)
    except OSError:
        return None
    return target or None


def has_untrusted_symlink(path: str | Path) -> bool:
    """Return True when resolving the directory *path* meets an untrusted symlink.

    Every component, including the last, is checked the way the kernel
    resolves it: trusted symlinks are followed and the components of their
    targets are checked too.  A missing component or a non-directory ends the
    walk with False, so callers keep their own errors for those.  On Windows,
    which resolves ".." before symlinks, any symlink among the lexical
    ancestors counts.
    """
    path = Path(path)
    if _WINDOWS:
        return any(candidate.is_symlink() for candidate in (path, *path.parents))
    if path.is_absolute():
        current = Path(path.anchor)
        pending = list(reversed(path.parts[1:]))
    else:
        current = Path(".")
        pending = list(reversed(path.parts))
    hops = 0
    while pending:
        part = pending.pop()
        if part in {"", "."}:
            continue
        candidate = current / part
        try:
            info = os.lstat(candidate)
        except OSError:
            return False
        if stat.S_ISLNK(info.st_mode):
            hops += 1
            target = trusted_symlink_target(candidate)
            if target is None or hops > MAX_TRUSTED_SYMLINKS:
                return True
            if target.startswith("/"):
                current = Path("/")
            pending.extend(reversed(target.split("/")))
            continue
        if not stat.S_ISDIR(info.st_mode):
            return False
        current = candidate
    return False


def lock_path_for(path: str | Path) -> Path:
    """Return the sidecar lock file used by ``file_lock`` for *path*."""
    path = Path(path)
    return path.with_name(path.name + ".lock")


def _try_lock(fd: int) -> bool:
    """Try once to take the lock. Return False only when another holder has it."""
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        except OSError:
            # Filesystems without flock support (some network mounts): carry
            # on unlocked, as the trace store lock does.
            pass
        return True
    if msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    return True


def _lock_blocking(fd: int) -> None:
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            pass
        return
    while not _try_lock(fd):
        time.sleep(_POLL_INTERVAL)


def _unlock(fd: int) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt is not None:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


@contextlib.contextmanager
def file_lock(path: str | Path, *, timeout: float | None = None) -> Iterator[None]:
    """Hold an exclusive cross-process lock for *path*.

    Blocks until the lock is free, or raises ``TimeoutError`` once *timeout*
    seconds pass.  Missing parent directories are created with mode 0700.
    The lock is not re-entrant: do not nest two ``file_lock`` calls for the
    same path in one process.
    """
    lock_path = lock_path_for(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, 0o600)
    try:
        if timeout is None:
            _lock_blocking(fd)
        else:
            deadline = time.monotonic() + max(0.0, timeout)
            while not _try_lock(fd):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for lock on {lock_path}")
                time.sleep(_POLL_INTERVAL)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def write_atomic(
    path: str | Path,
    text: str,
    *,
    mode: int | None = None,
    allow_symlinks: bool = False,
) -> None:
    """Write *text* to a unique sibling temporary file and atomically replace *path*.

    With *mode* set, the new file gets exactly that mode (for example 0600 for
    files holding tokens).  Otherwise an existing file's mode is kept, and a
    new file gets the private mode of ``tempfile.mkstemp``.  Symlinked paths
    are refused unless *allow_symlinks* is set, in which case a symlinked
    target is replaced by a regular file.
    """
    path = Path(path)
    if not allow_symlinks and (path.is_symlink() or path.parent.is_symlink()):
        raise ValueError(f"refusing symlinked atomic write path: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        if mode is not None:
            os.chmod(temporary_name, mode)
        elif path.exists():
            os.chmod(temporary_name, path.stat().st_mode)
        _replace(temporary_name, path)
        temporary_name = ""
        fsync_directory(path.parent)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass


def _replace(source: str, target: Path) -> None:
    # On Windows a reader holding the target open makes the rename fail
    # briefly with PermissionError. Retry for a short time before giving up.
    attempts = 50 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(_POLL_INTERVAL)


def fsync_directory(path: str | Path) -> None:
    """Best-effort directory fsync for durable rename/unlink operations."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
