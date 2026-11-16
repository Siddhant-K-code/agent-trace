"""Session-relative file paths and shell file targets.

Shared by the PR comment summary and the reviewability check so both count
the same files under the same names: explicit file events plus the files a
shell command read or wrote, as found by :mod:`agent_trace.shell_split`.

``watch`` uses :func:`absolute_cwd` and :func:`resolve_path` to read the
relative paths a tool call names against the session's working directory.
"""

from __future__ import annotations

import ntpath
import posixpath
import re
from typing import Sequence

from .models import EventType, TraceEvent
from .shell_split import read_targets, write_targets

SHELL_TOOLS = frozenset({
    "bash", "shell", "run", "exec", "exec_command", "execute_command",
    "terminal", "run_command", "run_in_terminal", "powershell",
})


def session_cwd(events: Sequence[TraceEvent]) -> str:
    """Return the working directory the session start event recorded."""
    for event in events:
        if event.event_type == EventType.SESSION_START:
            cwd = event.data.get("cwd")
            if isinstance(cwd, str) and cwd.strip():
                return cwd.strip().replace("\\", "/").rstrip("/")
    return ""


_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+://")


def absolute_cwd(value: object) -> str:
    """Return ``value`` if it can resolve relative paths, otherwise ``""``.

    That is a non-empty string holding an absolute path: POSIX, such as
    ``/app``, or Windows, such as ``C:\\repo`` or ``\\\\server\\share``.
    """
    if isinstance(value, str) and (value.startswith("/") or _WINDOWS_ABSOLUTE.match(value)):
        return value
    return ""


def resolve_path(path: str, cwd: str = "") -> str:
    """Return ``path`` joined to ``cwd`` and normalized, as text only.

    ``.`` and ``..`` collapse as in ``normpath``, for absolute paths too.
    Symlinks are not resolved and the file system is not read. A relative
    path stays as given when ``cwd`` is empty. A URI such as
    ``file:///app/x`` and a path that starts with ``~`` stay as given. A
    Windows path, or a relative path under a Windows ``cwd``, comes back with
    ``/`` separators, such as ``C:/repo/x.py``.
    """
    if not path or _URI.match(path) or path.startswith("~"):
        return path
    if path.startswith("/"):
        return posixpath.normpath(path)
    if _WINDOWS_ABSOLUTE.match(path):
        return ntpath.normpath(path).replace("\\", "/")
    if not cwd:
        return path
    if _WINDOWS_ABSOLUTE.match(cwd):
        return ntpath.normpath(ntpath.join(cwd, path)).replace("\\", "/")
    return posixpath.normpath(posixpath.join(cwd, path))


def cwd_forms(cwd: str) -> tuple[str, ...]:
    """Return ``cwd`` and its macOS alias (``/tmp`` is ``/private/tmp``)."""
    if not cwd:
        return ()
    if re.match(r"^/private/(?:tmp|var|etc)(?:/|$)", cwd):
        return cwd, cwd[len("/private"):]
    if re.match(r"^/(?:tmp|var|etc)(?:/|$)", cwd):
        return cwd, "/private" + cwd
    return (cwd,)


def normalise_path(path: str, cwd: str) -> str:
    """Return ``path`` relative to ``cwd`` when it is inside it."""
    value = path.strip().replace("\\", "/")
    for form in cwd_forms(cwd):
        if value.startswith(form + "/"):
            value = value[len(form) + 1:]
            break
    while value.startswith("./"):
        value = value[2:]
    return value


def command_text(event: TraceEvent) -> str:
    """Return the shell command a tool call ran, or an empty string."""
    args = event.data.get("arguments")
    if not isinstance(args, dict):
        args = event.data.get("tool_input")
    if not isinstance(args, dict):
        args = {}
    command = args.get("command") or args.get("cmd") or event.data.get("command") or ""
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    return command if isinstance(command, str) else ""


def is_shell_call(event: TraceEvent) -> bool:
    if event.event_type != EventType.TOOL_CALL:
        return False
    tool = str(event.data.get("tool_name") or "").lower()
    return tool in SHELL_TOOLS or bool(command_text(event))


def shell_writes(event: TraceEvent, cwd: str) -> list[str]:
    """Return paths a shell tool call wrote by redirection or ``tee``."""
    if not is_shell_call(event):
        return []
    return [normalise_path(path, cwd) for path in write_targets(command_text(event))]


def shell_reads(event: TraceEvent, cwd: str) -> list[str]:
    """Return paths a shell tool call read with a plain file viewer."""
    if not is_shell_call(event):
        return []
    return [normalise_path(path, cwd) for path in read_targets(command_text(event))]


__all__ = [
    "SHELL_TOOLS",
    "absolute_cwd",
    "command_text",
    "cwd_forms",
    "is_shell_call",
    "normalise_path",
    "resolve_path",
    "session_cwd",
    "shell_reads",
    "shell_writes",
]
