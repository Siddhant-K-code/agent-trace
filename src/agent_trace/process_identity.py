"""Tell whether a PID still names the process it named earlier.

``watch --agent-pid`` and ``approval approve`` / ``deny`` signal a PID that
was recorded earlier, possibly much earlier. If the agent exits, the OS can
give its PID to an unrelated process. To avoid stopping or killing that
process, callers record the process start time with the PID and check it
again before signalling.

The identity is an opaque string with a source prefix:

- ``proc:<ticks>``: field 22 (``starttime``) of ``/proc/PID/stat`` on Linux.
- ``ps:<lstart>``: ``ps -o lstart= -p PID`` elsewhere on POSIX (macOS, BSD).
  ``lstart`` is read in UTC and has one-second resolution, so a PID reused
  within the same second is not detected.

Windows has no cheap stdlib lookup, so the identity is unknown there. When
either the recorded or the current identity is unknown, or they come from
different sources, :func:`pid_reused` returns False and callers keep their
old behaviour.

:func:`list_processes` reads the whole process table in one pass, with the
same identity for each process, and :func:`descendants` walks it down from
one PID. ``watch`` uses them to kill the processes the agent started.
"""

from __future__ import annotations

import os
import subprocess
from collections import deque
from collections.abc import Mapping
from typing import NamedTuple

PID_REUSED_REASON = "the PID now belongs to another process"

_PS_TIMEOUT_SECONDS = 2.0


class ProcessEntry(NamedTuple):
    """One process in the table that :func:`list_processes` returns."""

    ppid: int
    identity: str


class ProcessListError(Exception):
    """The process table could not be read. The message says why."""


def _proc_stat_fields(raw: str) -> list[str] | None:
    """Return the fields of /proc/PID/stat from field 3 on, or None."""
    # The command name (field 2) is in parentheses and may contain spaces or
    # ")", so split after the last ")". The rest starts at field 3.
    _, sep, rest = raw.rpartition(")")
    fields = rest.split()
    if not sep or len(fields) < 20:
        return None
    return fields


def _proc_identity(fields: list[str]) -> str | None:
    start = fields[19]
    return f"proc:{start}" if start.isdigit() else None


def _proc_start_time(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            raw = f.read().decode("ascii", "replace")
    except OSError:
        return None
    fields = _proc_stat_fields(raw)
    return None if fields is None else _proc_identity(fields)


def _ps_env() -> dict[str, str]:
    # lstart is printed in local time; pin the zone and locale so the same
    # process reads the same after a timezone change or from another shell.
    return dict(os.environ, LC_ALL="C", LANG="C", TZ="UTC0")


def _ps_start_time(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, env=_ps_env(),
            timeout=_PS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    start = " ".join(result.stdout.split())
    if result.returncode != 0 or not start:
        return None
    return f"ps:{start}"


def _list_proc(root: str = "/proc") -> dict[int, ProcessEntry]:
    try:
        names = os.listdir(root)
    except OSError as exc:
        raise ProcessListError(f"cannot read {root}: {exc.strerror or exc}") from None
    table: dict[int, ProcessEntry] = {}
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(root, name, "stat"), "rb") as f:
                raw = f.read().decode("ascii", "replace")
        except OSError:
            continue  # the process ended during the scan
        fields = _proc_stat_fields(raw)
        if fields is None or not fields[1].isdigit():
            continue
        identity = _proc_identity(fields)
        if identity is not None:
            table[int(name)] = ProcessEntry(int(fields[1]), identity)
    if not table:
        raise ProcessListError(f"{root} lists no process")
    return table


def _list_ps() -> dict[int, ProcessEntry]:
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,lstart="],
            capture_output=True, text=True, errors="replace", env=_ps_env(),
            timeout=_PS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise ProcessListError(
            f"ps did not finish within {_PS_TIMEOUT_SECONDS:g}s"
        ) from None
    except (OSError, subprocess.SubprocessError) as exc:
        reason = getattr(exc, "strerror", None) or exc
        raise ProcessListError(f"cannot run ps: {reason}") from None
    if result.returncode != 0:
        raise ProcessListError(f"ps exited with status {result.returncode}")
    table: dict[int, ProcessEntry] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        # The same text that _ps_start_time reads for this PID.
        table[int(parts[0])] = ProcessEntry(int(parts[1]), "ps:" + " ".join(parts[2:]))
    if not table:
        raise ProcessListError("ps listed no process")
    return table


def list_processes() -> dict[int, ProcessEntry]:
    """Return every process as PID → (parent PID, identity). POSIX only.

    Each identity is what :func:`process_identity` returns for that PID at
    the time of the listing. Linux reads ``/proc`` and skips a process that
    ends during the scan. Elsewhere one ``ps`` call lists every process.
    Raise :class:`ProcessListError` when the table cannot be read.
    """
    if os.path.exists("/proc/self/stat"):
        return _list_proc()
    return _list_ps()


def _parent_chain(table: Mapping[int, ProcessEntry], pid: int) -> set[int]:
    """Return *pid* and its ancestors in *table*. Stops at a repeated PID."""
    chain: set[int] = set()
    while pid in table and pid not in chain:
        chain.add(pid)
        pid = table[pid].ppid
    return chain


def descendants(
    table: Mapping[int, ProcessEntry], root: int, own_pid: int,
) -> list[tuple[int, str]]:
    """Return (PID, identity) of each process whose parent chain leads to *root*.

    Parents come before their children. *root* is not included, and
    neither is any PID of 1 or lower. *own_pid* and the processes it
    started are left out. When *own_pid* runs under *root*, the processes
    between them are left out too, because signalling them can end
    *own_pid*. A parent cycle in *table* cannot make this loop.
    """
    if root == own_pid or own_pid in _parent_chain(table, root):
        return []
    above_own = _parent_chain(table, own_pid)
    children: dict[int, list[int]] = {}
    for pid, entry in table.items():
        if pid > 1 and entry.ppid != pid:
            children.setdefault(entry.ppid, []).append(pid)
    found: list[tuple[int, str]] = []
    seen = {root, own_pid}
    queue = deque([root])
    while queue:
        for pid in sorted(children.get(queue.popleft(), ())):
            if pid in seen:
                continue
            seen.add(pid)
            queue.append(pid)
            if pid not in above_own:
                found.append((pid, table[pid].identity))
    return found


def process_identity(pid: object) -> str | None:
    """Return the start-time identity of ``pid``, or None when unknown."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if os.name != "posix":
        return None
    if os.path.exists("/proc/self/stat"):
        return _proc_start_time(pid)
    return _ps_start_time(pid)


def pid_reused(pid: object, recorded: object) -> bool:
    """Return True only when ``pid`` provably names a different process now.

    ``recorded`` is the identity stored when the PID was taken. An empty or
    missing record, an unreadable current identity, or identities from
    different sources all return False.
    """
    if not isinstance(recorded, str) or ":" not in recorded:
        return False
    current = process_identity(pid)
    if current is None:
        return False
    if current.split(":", 1)[0] != recorded.split(":", 1)[0]:
        return False
    return current != recorded
