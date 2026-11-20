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
                value = cwd.strip().replace("\\", "/")
                if value == "/" or re.fullmatch(r"[A-Za-z]:/", value):
                    return value
                return value.rstrip("/")
    return ""


_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{2})")
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


def _relative_to(value: str, cwd: str, *, windows: bool) -> str | None:
    """Return an absolute path relative to ``cwd`` when it is contained in it."""
    path_module = ntpath if windows else posixpath
    value = path_module.normpath(value)
    cwd = path_module.normpath(cwd)
    if windows:
        value_drive, value_tail = ntpath.splitdrive(value)
        cwd_drive, cwd_tail = ntpath.splitdrive(cwd)
        if value_drive.startswith("\\\\") and not value_tail:
            value = value_drive + "\\"
        if cwd_drive.startswith("\\\\") and not cwd_tail:
            cwd = cwd_drive + "\\"
    try:
        common = path_module.commonpath((cwd, value))
    except ValueError:
        return None
    same_root = (
        ntpath.normcase(common) == ntpath.normcase(cwd)
        if windows
        else common == cwd
    )
    if not same_root:
        return None
    relative = path_module.relpath(value, cwd)
    return relative.replace("\\", "/") if windows else relative


def normalise_path(path: str, cwd: str) -> str:
    """Return ``path`` relative to ``cwd`` when it is inside it.

    Absolute Windows paths are compared case-insensitively, and absolute paths
    are normalized lexically before their containment is checked.
    """
    value = path.strip().replace("\\", "/")
    working_directory = cwd.replace("\\", "/")
    for form in cwd_forms(working_directory):
        form_windows = bool(_WINDOWS_ABSOLUTE.match(form))
        value_windows = bool(_WINDOWS_ABSOLUTE.match(value))
        if form_windows and value_windows:
            relative = _relative_to(value, form, windows=True)
        elif not form_windows and form.startswith("/") and value.startswith("/"):
            relative = _relative_to(value, form, windows=False)
        else:
            relative = None
        if relative is not None:
            return relative
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
