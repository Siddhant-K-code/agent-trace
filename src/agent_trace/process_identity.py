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
"""

from __future__ import annotations

import os
import subprocess

PID_REUSED_REASON = "the PID now belongs to another process"

_PS_TIMEOUT_SECONDS = 2.0


def _proc_start_time(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            raw = f.read().decode("ascii", "replace")
    except OSError:
        return None
    # The command name (field 2) is in parentheses and may contain spaces or
    # ")", so split after the last ")". The rest starts at field 3.
    _, sep, rest = raw.rpartition(")")
    fields = rest.split()
    if not sep or len(fields) < 20:
        return None
    start = fields[19]
    return f"proc:{start}" if start.isdigit() else None


def _ps_start_time(pid: int) -> str | None:
    # lstart is printed in local time; pin the zone and locale so the same
    # process reads the same after a timezone change or from another shell.
    env = dict(os.environ, LC_ALL="C", LANG="C", TZ="UTC0")
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, env=env,
            timeout=_PS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    start = " ".join(result.stdout.split())
    if result.returncode != 0 or not start:
        return None
    return f"ps:{start}"


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
