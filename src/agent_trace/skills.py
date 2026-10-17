"""Skill adoption evidence: available -> loaded -> followed -> outcome.

This module reports what recorded trace events show about agent skills
(``SKILL.md`` packages).  It deliberately produces **no grades, composite
scores, or verdicts**:

* *available* is a filesystem fact observed **now**, not at session time;
* *loaded* is a recorded tool call, file read, or user prompt;
* *followed* is a set of signals (resource reads, script runs, command
  signature matches), never a judgement that guidance was obeyed or ignored;
* *outcome* lists facts recorded after the load and is never causal.

Prompt text and ``SKILL.md`` bodies are never emitted.  Only skill names,
two-token command signatures, and paths relative to a skill directory are
reported.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
import statistics
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, TextIO

from .cost import parse_since
from .models import EventType, SessionMeta, TraceEvent
from .review import is_test_command
from .store import TraceStore

SCHEMA_VERSION = 1

REPO_SKILL_DIRS = (
    ".github/skills",
    ".claude/skills",
    ".agents/skills",
    ".gemini/skills",
    ".codex/skills",
    ".cursor/skills",
)
USER_SKILL_DIRS = (
    "~/.copilot/skills",
    "~/.claude/skills",
    "~/.agents/skills",
    "~/.gemini/skills",
    "~/.codex/skills",
    "~/.cursor/skills",
)
CONTEXT_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    ".github/copilot-instructions.md",
)

NOT_MEASURABLE = (
    "Availability is read from the filesystem now, not as it was when the session ran.",
    "Context files (AGENTS.md, CLAUDE.md, GEMINI.md, copilot-instructions.md) are "
    "auto-injected by most agents; their load is not observable from hooks.",
    "Skills injected by a provider without a tool call, file read, or prompt "
    "mention leave no event and are invisible here.",
    "Whether non-command guidance (style, judgement, ordering) was followed is "
    "not observable from tool events.",
    "Model reasoning is not used: summarized or omitted thinking is not reliable evidence.",
    "Outcome facts after a load are not causal; task mix is not controlled.",
)

_SKILL_FILE = "SKILL.md"
_MAX_RESOURCE_FILES = 500
_MAX_REPORTED_PATHS = 10
_MAX_COMMAND_CHARS = 100_000
_MAX_ARGUMENT_CHARS = 200_000

_SKILL_TOOLS = {"skill": ("skill", "command", "name"), "activate_skill": ("name", "skill")}
_EDIT_TOOLS = frozenset({
    "apply_patch",
    "create",
    "delete_file",
    "edit",
    "edit_file",
    "file_edit",
    "insert",
    "multiedit",
    "notebookedit",
    "patch",
    "replace",
    "search_replace",
    "str_replace_based_edit_tool",
    "str_replace_editor",
    "write",
    "write_file",
    "write_to_file",
})
_PATH_KEYS = ("file_path", "path", "absolute_path", "filePath", "target_file", "notebook_path", "uri")
_SHELL_FENCE_LANGS = frozenset({"bash", "sh", "shell", "zsh", "console", "shell-session", "text"})
_SHELL_PUNCT = ";&|()<>\n"
_LEADING_KEYWORDS = frozenset({"do", "then", "else", "elif", "if", "while", "until", "!", "{", "}", "time", "done", "fi", "esac"})
_SKIPPED_KEYWORDS = frozenset({"for", "case", "select", "function"})
_LOW_SIGNAL_PROGRAMS = frozenset({
    ".", ":", "[", "[[", "alias", "cd", "echo", "eval", "exit", "export", "false",
    "local", "printf", "read", "return", "set", "shift", "sleep", "source", "test",
    "true", "unset", "wait",
})
_INTERPRETERS = frozenset({
    "bash", "bun", "deno", "node", "npx", "perl", "pwsh", "python", "python3",
    "ruby", "sh", "ts-node", "tsx", "uv", "zsh",
})
_READ_PROGRAMS = frozenset({"awk", "bat", "cat", "head", "less", "more", "nl", "sed", "tail"})
_GIT_PAGER_FLAGS = frozenset({"--no-pager", "-P", "--paginate"})
_RESOURCE_PROGRAMS = _READ_PROGRAMS | frozenset({"cp", "diff", "egrep", "grep", "jq", "rg", "rsync", "wc", "yq"})
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_HEREDOC = re.compile(r"(?<!<)<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_INTERPRETER_VERSION = re.compile(r"^(python|ruby|perl|node)[0-9.]*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)")
_FENCE_CLOSE = re.compile(r"\s*(`{3,}|~{3,})\s*")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SkillDefinition:
    """A ``SKILL.md`` found on disk.  ``skill_dir`` is never emitted."""

    name: str
    skill_dir: str
    skill_file: str
    source: str
    source_dir: str
    commands: tuple[str, ...]
    resources: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "dir": self.source_dir,
            "definition_commands": len(self.commands),
            "definition_resources": len(self.resources),
        }


@dataclass(frozen=True)
class ScannedDir:
    display: str
    source: str
    exists: bool
    skills: int


@dataclass
class SkillCatalog:
    skills: dict[str, SkillDefinition] = field(default_factory=dict)
    also_in: dict[str, list[str]] = field(default_factory=dict)
    scanned: list[ScannedDir] = field(default_factory=list)

    def by_file(self) -> dict[str, SkillDefinition]:
        return {_real(defn.skill_file): defn for defn in self.skills.values()}


@dataclass(frozen=True)
class SkillLoad:
    name: str
    method: str  # skill_tool | file_read | user_prompt
    invoked_by: str  # model | user
    detail: str
    event_id: str
    timestamp: float
    seconds_since_start: float
    loaded_before_first_edit: bool | None
    load_succeeded: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "invoked_by": self.invoked_by,
            "detail": self.detail,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "seconds_since_start": round(self.seconds_since_start, 3),
            "loaded_before_first_edit": self.loaded_before_first_edit,
            "load_succeeded": self.load_succeeded,
        }


@dataclass(frozen=True)
class _Segment:
    """One simple command with paths pre-resolved against its effective cwd.

    Candidate paths are ``(resolved, bare)`` pairs; ``bare`` marks a token
    without a path separator, which only counts when it names a known
    resource of the skill.
    """

    signature: str
    script: tuple[str, bool] | None
    resources: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class _Prepared:
    """Per-event facts computed once and shared by every loaded skill."""

    event: TraceEvent
    paths: tuple[str, ...]
    segments: tuple[_Segment, ...]
    is_test: bool


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _expanduser(path: str) -> str:
    try:
        return os.path.expanduser(path)
    except (OSError, ValueError):
        return path


def _real(path: str | os.PathLike[str]) -> str:
    try:
        return os.path.realpath(os.path.expanduser(str(path)))
    except (OSError, ValueError):
        return ""


def _display_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _frontmatter_name(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = re.match(r"^name\s*:\s*(.*?)\s*$", line)
        if match:
            value = match.group(1)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            return value.strip()
    return ""


def extract_skill_commands(text: str) -> tuple[str, ...]:
    """Return sorted unique command signatures from fenced shell blocks."""
    signatures: set[str] = set()
    fence = ""
    capture = False
    lines: list[str] = []
    for line in text.splitlines():
        if not fence:
            match = _FENCE.match(line)
            if match:
                fence = match.group(1)[0]
                capture = match.group(2).lower() in _SHELL_FENCE_LANGS
                lines = []
            continue
        if _FENCE_CLOSE.fullmatch(line) and line.strip()[0] == fence:
            if capture:
                signatures.update(command_signatures(_strip_prompts("\n".join(lines))))
            fence = ""
            continue
        if capture:
            lines.append(line)
    return tuple(sorted(signatures))


def _strip_prompts(block: str) -> str:
    cleaned = []
    for line in block.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("$ "):
            stripped = stripped[2:]
        elif stripped.startswith("> ") or stripped.startswith("% "):
            stripped = stripped[2:]
        cleaned.append(stripped)
    return "\n".join(cleaned)


def _skill_resources(skill_dir: Path, skill_file: Path) -> tuple[str, ...]:
    resources: list[str] = []
    root = str(skill_dir)
    for current, dirs, files in os.walk(root):
        # A nested SKILL.md belongs to a different skill; never count it here.
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".")
            and d != "node_modules"
            and not os.path.isfile(os.path.join(current, d, _SKILL_FILE))
        )
        for name in sorted(files):
            if name.startswith("."):
                continue
            full = os.path.join(current, name)
            if os.path.abspath(full) == os.path.abspath(skill_file):
                continue
            resources.append(os.path.relpath(full, root))
            if len(resources) >= _MAX_RESOURCE_FILES:
                return tuple(resources)
    return tuple(resources)


def _load_definition(skill_file: Path, source: str, source_dir: str) -> SkillDefinition | None:
    try:
        text = skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    skill_dir = skill_file.parent
    name = _frontmatter_name(text) or skill_dir.name
    return SkillDefinition(
        name=name,
        skill_dir=_real(skill_dir),
        skill_file=_real(skill_file),
        source=source,
        source_dir=source_dir,
        commands=extract_skill_commands(text),
        resources=_skill_resources(skill_dir, skill_file),
    )


def _definitions_in(directory: Path, source: str, display: str) -> list[SkillDefinition]:
    found: list[SkillDefinition] = []
    # os.path.isfile/isdir return False on any OSError (e.g. EACCES); pathlib's
    # methods re-raise some errnos before Python 3.13.
    root_file = directory / _SKILL_FILE
    if os.path.isfile(root_file):
        definition = _load_definition(root_file, source, display)
        if definition:
            found.append(definition)
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return found
    for child in children:
        candidate = child / _SKILL_FILE
        if not child.name.startswith(".") and os.path.isdir(child) and os.path.isfile(candidate):
            definition = _load_definition(candidate, source, display)
            if definition:
                found.append(definition)
    return found


def discover_skills(
    workspace: str | os.PathLike[str] | None,
    extra_dirs: Iterable[str] = (),
    *,
    user_dirs: Iterable[str] = USER_SKILL_DIRS,
) -> SkillCatalog:
    """Discover skills from repo, user, and extra directories.

    Precedence for duplicate names is repo, then user, then extra.  Later
    duplicates are recorded in ``also_in`` instead of replacing the first.
    """
    catalog = SkillCatalog()
    candidates: list[tuple[Path, str, str]] = []
    if workspace:
        root = Path(os.path.expanduser(str(workspace)))
        for rel in REPO_SKILL_DIRS:
            candidates.append((root / rel, "repo", rel))
    for user_dir in user_dirs:
        candidates.append((Path(os.path.expanduser(user_dir)), "user", _display_path(os.path.expanduser(user_dir))))
    for extra in extra_dirs:
        expanded = os.path.abspath(os.path.expanduser(extra))
        candidates.append((Path(expanded), "extra", _display_path(expanded)))

    seen_dirs: set[str] = set()
    for directory, source, display in candidates:
        exists = os.path.isdir(directory)
        key = _real(directory) if exists else str(directory)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        definitions = _definitions_in(directory, source, display) if exists else []
        catalog.scanned.append(ScannedDir(display=display, source=source, exists=exists, skills=len(definitions)))
        for definition in definitions:
            if definition.name in catalog.skills:
                catalog.also_in.setdefault(definition.name, []).append(display)
            else:
                catalog.skills[definition.name] = definition
    return catalog


def context_files(workspace: str | os.PathLike[str] | None) -> list[dict[str, object]]:
    root = Path(os.path.expanduser(str(workspace))) if workspace else None
    return [
        {
            "path": rel,
            "present": bool(root and os.path.isfile(root / rel)),
            "load_observable": False,
        }
        for rel in CONTEXT_FILES
    ]


# --------------------------------------------------------------------------- #
# Shell parsing
# --------------------------------------------------------------------------- #


def _strip_heredocs(command: str) -> str:
    kept: list[str] = []
    pending: list[str] = []
    for line in command.split("\n"):
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue
        kept.append(line)
        pending.extend(match.group(2) for match in _HEREDOC.finditer(line))
    return "\n".join(kept)


_WORD_BOUNDARY = frozenset(" \t\r\n;&|()<>")


def _strip_comments(command: str) -> str:
    """Drop ``# ...`` comments the way a shell does, keeping the newline.

    ``#`` starts a comment only at the start of a word and outside quotes;
    shlex's own comment handling also swallows the newline and fires mid-word.
    """
    out: list[str] = []
    quote = ""
    i = 0
    length = len(command)
    while i < length:
        char = command[i]
        if quote:
            out.append(char)
            if char == "\\" and quote == '"' and i + 1 < length:
                out.append(command[i + 1])
                i += 2
                continue
            if char == quote:
                quote = ""
        elif char == "\\" and i + 1 < length:
            out.append(char)
            out.append(command[i + 1])
            i += 2
            continue
        elif char in "'\"":
            quote = char
            out.append(char)
        elif char == "#" and (i == 0 or command[i - 1] in _WORD_BOUNDARY):
            end = command.find("\n", i)
            if end == -1:
                break
            i = end
            continue
        else:
            out.append(char)
        i += 1
    return "".join(out)


def _tokenize(command: str) -> list[str]:
    command = command[:_MAX_COMMAND_CHARS].replace("\\\n", " ")
    command = _strip_comments(_strip_heredocs(command))
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_SHELL_PUNCT)
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return re.findall(r"[^\s;&|()<>]+|[;&|()<>\n]+", command)


def _is_break(token: str) -> bool:
    # shlex groups adjacent punctuation ("&&\n", ";\n", "\n\n", ");"), so a
    # token is a command separator if it is punctuation-only and contains one.
    if not token or not set(token) <= set(_SHELL_PUNCT):
        return False
    return token in ("&", "&&") or any(char in token for char in ";|()\n")


def _segments(command: str) -> list[list[str]]:
    """Split a shell command into simple-command word lists.

    Leading env assignments, ``env``, ``sudo`` (and its flags), and shell
    keywords are stripped.  Words stop at the first redirection operator.
    """
    segments: list[list[str]] = []
    current: list[str] = []
    for token in _tokenize(command):
        if _is_break(token):
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)

    cleaned: list[list[str]] = []
    for words in segments:
        while words and words[0] in _LEADING_KEYWORDS:
            words = words[1:]
        if not words or words[0] in _SKIPPED_KEYWORDS:
            continue
        while words and (_ENV_ASSIGNMENT.match(words[0]) or words[0] == "env"):
            words = words[1:]
        if words and words[0] == "sudo":
            words = words[1:]
            while words and words[0].startswith("-"):
                words = words[1:]
        if words and _program_name(words[0]) == "git":
            # Display-only pager flags do not change which git command ran.
            while len(words) > 1 and words[1] in _GIT_PAGER_FLAGS:
                words = [words[0]] + words[2:]
        trimmed: list[str] = []
        for word in words:
            if word and set(word) <= set("<>&"):
                break
            trimmed.append(word)
        if trimmed:
            cleaned.append(trimmed)
    return cleaned


def _signature(words: list[str]) -> str:
    program = words[0]
    if (
        not program
        or program in _LOW_SIGNAL_PROGRAMS
        or program.startswith(("$", "#", "<", "{"))
    ):
        return ""
    return " ".join(words[:2])


def command_signatures(command: str) -> list[str]:
    """Return normalized two-token signatures for each simple command."""
    signatures = []
    for words in _segments(command):
        signature = _signature(words)
        if signature:
            signatures.append(signature)
    return signatures


def _program_name(word: str) -> str:
    base = os.path.basename(word)
    match = _INTERPRETER_VERSION.match(base)
    return match.group(1) if match else base


def _resolve(token: str, cwd: str) -> str:
    if "\x00" in token or "\x00" in cwd:
        return ""
    if token.startswith("file://"):
        token = token[len("file://"):]
    expanded = _expanduser(token)
    if not os.path.isabs(expanded):
        if not cwd:
            return ""
        expanded = os.path.join(cwd, expanded)
    return _real(os.path.normpath(expanded))


def _inside(path: str, directory: str) -> bool:
    return bool(path) and bool(directory) and path != directory and path.startswith(directory.rstrip(os.sep) + os.sep)


def _is_skill_file(token: str) -> bool:
    return "\x00" not in token and re.split(r"[\\/]", token)[-1] == _SKILL_FILE


def _looks_like_path(token: str) -> bool:
    return "/" in token or token.startswith(("~", ".")) or _is_skill_file(token)


# --------------------------------------------------------------------------- #
# Event helpers
# --------------------------------------------------------------------------- #


def _parse_mapping(value: str) -> Mapping[str, Any] | None:
    text = value.strip()
    if not text.startswith("{") or len(text) > _MAX_ARGUMENT_CHARS:
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError, MemoryError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
            return None
    return parsed if isinstance(parsed, Mapping) else None


def event_arguments(event: TraceEvent) -> Mapping[str, Any]:
    """Return tool arguments from ``arguments`` or ``tool_input``.

    Some adapters store arguments as a serialized mapping (JSON or a Python
    literal); those are parsed without evaluating code.
    """
    data = event.data if isinstance(event.data, Mapping) else {}
    for key in ("arguments", "tool_input"):
        value = data.get(key)
        if isinstance(value, Mapping):
            return value
        if isinstance(value, str):
            parsed = _parse_mapping(value)
            if parsed is not None:
                return parsed
    return {}


def _coerce_timestamp(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _sanitize_events(events: Iterable[TraceEvent]) -> tuple[list[TraceEvent], int]:
    """Return events with well-typed fields, plus the count of unusable events.

    Trace files are not schema-checked on load, so ``data`` may be a non-dict,
    ``timestamp`` a string, or ids non-strings.  Events whose timestamp cannot
    be read as a finite number are skipped; other fields are normalized.
    """
    clean: list[TraceEvent] = []
    skipped = 0
    for event in events:
        if not isinstance(event, TraceEvent) or not isinstance(event.event_type, EventType):
            skipped += 1
            continue
        timestamp = _coerce_timestamp(event.timestamp)
        if timestamp is None:
            skipped += 1
            continue
        changes: dict[str, Any] = {}
        if type(event.timestamp) not in (int, float):
            changes["timestamp"] = timestamp
        if not isinstance(event.data, dict):
            changes["data"] = {}
        if not isinstance(event.event_id, str):
            changes["event_id"] = ""
        if not isinstance(event.parent_id, str):
            changes["parent_id"] = ""
        clean.append(replace(event, **changes) if changes else event)
    return clean, skipped


def _tool_name(event: TraceEvent) -> str:
    return str(event.data.get("tool_name") or event.data.get("tool") or "")


def _event_command(event: TraceEvent) -> str:
    if event.event_type != EventType.TOOL_CALL:
        return ""
    if _tool_name(event).lower() in _SKILL_TOOLS:
        return ""
    args = event_arguments(event)
    command = args.get("command") or args.get("cmd") or event.data.get("command")
    if isinstance(command, (list, tuple)) and all(isinstance(part, str) for part in command):
        return shlex.join(command)
    return command if isinstance(command, str) else ""


def _event_paths(event: TraceEvent) -> list[str]:
    if event.event_type == EventType.FILE_READ:
        source: Mapping[str, Any] = event.data
    elif event.event_type == EventType.TOOL_CALL:
        name = _tool_name(event).lower()
        if name in _EDIT_TOOLS or name in _SKILL_TOOLS:
            return []
        source = event_arguments(event)
    else:
        return []
    paths = []
    for key in _PATH_KEYS:
        value = source.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


def _is_edit(event: TraceEvent) -> bool:
    if event.event_type == EventType.FILE_WRITE:
        return True
    return event.event_type == EventType.TOOL_CALL and _tool_name(event).lower() in _EDIT_TOOLS


def _session_cwd(events: list[TraceEvent]) -> str:
    for event in events:
        if event.event_type == EventType.SESSION_START:
            cwd = event.data.get("cwd")
            if isinstance(cwd, str) and cwd:
                return cwd
    return ""


def _linked_results(events: list[TraceEvent]) -> dict[str, TraceEvent]:
    linked: dict[str, TraceEvent] = {}
    for event in events:
        if event.parent_id and event.event_type in (EventType.TOOL_RESULT, EventType.ERROR):
            linked.setdefault(event.parent_id, event)
    return linked


def _result_facts(result: TraceEvent | None) -> dict[str, object]:
    if result is None:
        return {"linked_result": None, "exit_code": None, "success": None}
    exit_code = result.data.get("exit_code")
    success = result.data.get("success")
    return {
        "linked_result": "error" if result.event_type == EventType.ERROR else "tool_result",
        "exit_code": exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None,
        "success": success if isinstance(success, bool) else None,
    }


def _load_succeeded(result: TraceEvent | None) -> bool | None:
    facts = _result_facts(result)
    if facts["linked_result"] is None:
        return None
    if facts["linked_result"] == "error" or facts["success"] is False:
        return False
    exit_code = facts["exit_code"]
    if isinstance(exit_code, int) and exit_code != 0:
        return False
    return True


# --------------------------------------------------------------------------- #
# Per-session analysis
# --------------------------------------------------------------------------- #


def _match_skill_name(raw: str, known: Iterable[str]) -> str:
    candidate = raw.strip().lstrip("/$")
    known_set = set(known)
    if candidate in known_set:
        return candidate
    if ":" in candidate:
        tail = candidate.rsplit(":", 1)[1]
        if tail in known_set:
            return tail
    return candidate


def _prompt_loads(prompt: str, names: Iterable[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    stripped = prompt.lstrip()
    first = re.match(r"^/(?:[A-Za-z0-9_.-]+:)?([A-Za-z0-9_.-]+)(?=\s|$)", stripped)
    for name in sorted(names):
        if first and first.group(1) == name:
            found.append((name, "slash_command"))
        elif re.search(r"(?<![\w$])\$" + re.escape(name) + r"(?![\w-])", prompt):
            found.append((name, "dollar_mention"))
    return found


def _find_loads(
    ordered: list[tuple[int, TraceEvent]],
    catalog: SkillCatalog,
    cwd: str,
) -> list[tuple[str, str, str, str, int, TraceEvent]]:
    """Return (name, method, invoked_by, detail, index, event) tuples."""
    loads = []
    by_file = catalog.by_file()
    names = list(catalog.skills)
    for index, event in ordered:
        if event.event_type == EventType.USER_PROMPT:
            prompt = event.data.get("prompt")
            if isinstance(prompt, str) and prompt:
                for name, detail in _prompt_loads(prompt, names):
                    loads.append((name, "user_prompt", "user", detail, index, event))
            continue

        tool = _tool_name(event).lower()
        if event.event_type == EventType.TOOL_CALL and tool in _SKILL_TOOLS:
            args = event_arguments(event)
            for key in _SKILL_TOOLS[tool]:
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    name = _match_skill_name(value, names)
                    loads.append((name, "skill_tool", "model", tool, index, event))
                    break
            continue

        for raw in _event_paths(event):
            if _is_skill_file(raw):
                loads.append((_skill_name_for_path(raw, cwd, by_file), "file_read", "model", "read_tool", index, event))
                break
        else:
            command = _event_command(event)
            if command:
                for name in _shell_skill_reads(command, cwd, by_file):
                    loads.append((name, "file_read", "model", "shell_read", index, event))
    return loads


def _skill_name_for_path(raw: str, cwd: str, by_file: Mapping[str, SkillDefinition]) -> str:
    resolved = _resolve(raw, cwd)
    if resolved in by_file:
        return by_file[resolved].name
    parent = os.path.basename(os.path.dirname(os.path.normpath(_expanduser(raw))))
    return parent or "SKILL.md"


def _shell_skill_reads(command: str, cwd: str, by_file: Mapping[str, SkillDefinition]) -> list[str]:
    names: list[str] = []
    effective = cwd
    for words in _segments(command):
        program = _program_name(words[0])
        if program == "cd" and len(words) > 1:
            effective = _resolve(words[1], effective) or effective
            continue
        if program not in _READ_PROGRAMS:
            continue
        if program == "sed" and any(word.startswith("-i") for word in words[1:]):
            continue
        for word in words[1:]:
            if _is_skill_file(word) and not word.startswith("-"):
                name = _skill_name_for_path(word, effective, by_file)
                if name not in names:
                    names.append(name)
    return names


def _empty_signals() -> dict[str, dict[str, object]]:
    return {
        "resource_read": {"count": 0, "paths": []},
        "script_run": {"count": 0, "paths": []},
        "command_match": {"count": 0, "signatures": [], "by_signature": {}},
    }


def _candidate(token: str, effective: str) -> tuple[str, bool] | None:
    """Return ``(resolved, bare)`` for a command argument, or ``None``."""
    if not token or token.startswith("-"):
        return None
    if _looks_like_path(token):
        resolved = _resolve(token, effective)
        return (resolved, False) if resolved else None
    if not effective or "\x00" in token:
        return None
    # A bare name (e.g. after ``cd <skill dir>``) is joined, not resolved; it
    # only counts later if it names a known resource of the skill.
    return os.path.join(effective, token), True


def _segment_candidates(words: list[str], program: str, effective: str) -> _Segment:
    script = None
    if _looks_like_path(words[0]):
        resolved = _resolve(words[0], effective)
        script = (resolved, False) if resolved else None
    elif program in _INTERPRETERS:
        for word in words[1:]:
            if word.startswith("-") or word == "run":
                continue
            script = _candidate(word, effective)
            break
    resources: tuple[tuple[str, bool], ...] = ()
    if program in _RESOURCE_PROGRAMS and not (
        program == "sed" and any(word.startswith("-i") for word in words[1:])
    ):
        resources = tuple(
            candidate for candidate in (_candidate(word, effective) for word in words[1:]) if candidate
        )
    return _Segment(signature=_signature(words), script=script, resources=resources)


def _prepare_event(event: TraceEvent, cwd: str, *, resolve: bool) -> _Prepared:
    """Parse one event once.  Paths are only resolved when ``resolve`` is set."""
    paths: tuple[str, ...] = ()
    if resolve:
        paths = tuple(path for path in (_resolve(raw, cwd) for raw in _event_paths(event)) if path)
    segments: list[_Segment] = []
    is_test = False
    if event.event_type == EventType.TOOL_CALL:
        is_test = event.data.get("is_test") is True or event.data.get("category") == "test"
        command = _event_command(event)
        effective = cwd
        for words in _segments(command) if command else []:
            if not is_test and is_test_command(" ".join(words)):
                is_test = True
            if not resolve:
                continue
            program = _program_name(words[0])
            if program == "cd" and len(words) > 1:
                effective = _resolve(words[1], effective) or effective
                continue
            segments.append(_segment_candidates(words, program, effective))
    return _Prepared(event=event, paths=paths, segments=tuple(segments), is_test=is_test)


def _follow_signals(
    definition: SkillDefinition | None,
    load_event: TraceEvent,
    after: list[_Prepared],
) -> dict[str, object]:
    """Report follow signals after a load.  Never a verdict."""
    base: dict[str, object] = {
        "status": "not_measurable",
        "reason": None,
        "signals": _empty_signals(),
        "seconds_to_first_signal": None,
        "first_signal_before_next_prompt": None,
    }
    if definition is None:
        base["reason"] = "definition_not_found"
        return base
    if not definition.commands and not definition.resources:
        base["reason"] = "definition_has_no_commands_or_resources"
        return base

    commands = set(definition.commands)
    resource_set = frozenset(definition.resources)
    skill_dir = definition.skill_dir
    skill_file = definition.skill_file
    signals = _empty_signals()
    by_signature: dict[str, int] = {}
    first_ts: float | None = None
    prompt_seen = False
    first_before_prompt: bool | None = None

    def relative(candidate: tuple[str, bool]) -> str:
        resolved, bare = candidate
        if resolved == skill_file or not _inside(resolved, skill_dir):
            return ""
        rel = os.path.relpath(resolved, skill_dir)
        return "" if bare and rel not in resource_set else rel

    def record(kind: str, rel_path: str) -> None:
        bucket = signals[kind]
        bucket["count"] = int(bucket["count"]) + 1
        paths = bucket["paths"]
        if rel_path not in paths and len(paths) < _MAX_REPORTED_PATHS:
            paths.append(rel_path)

    for prepared in after:
        event = prepared.event
        if event.event_type == EventType.USER_PROMPT:
            prompt_seen = True
            continue
        observed = False
        for resolved in prepared.paths:
            rel = relative((resolved, False))
            if rel:
                record("resource_read", rel)
                observed = True
                break

        matched_here: set[str] = set()
        for segment in prepared.segments:
            if segment.signature and segment.signature in commands:
                matched_here.add(segment.signature)
            if segment.script:
                rel = relative(segment.script)
                if rel:
                    record("script_run", rel)
                    observed = True
                    continue
            for candidate in segment.resources:
                rel = relative(candidate)
                if rel:
                    record("resource_read", rel)
                    observed = True
                    break
        if matched_here:
            signals["command_match"]["count"] = int(signals["command_match"]["count"]) + 1
            for signature in matched_here:
                by_signature[signature] = by_signature.get(signature, 0) + 1
            observed = True
        if observed and first_ts is None:
            first_ts = event.timestamp
            first_before_prompt = not prompt_seen

    signals["command_match"]["signatures"] = sorted(by_signature)
    signals["command_match"]["by_signature"] = dict(sorted(by_signature.items()))
    base["signals"] = signals
    base["status"] = "signals_observed" if first_ts is not None else "no_signal"
    if first_ts is not None:
        base["seconds_to_first_signal"] = round(max(0.0, first_ts - load_event.timestamp), 3)
        base["first_signal_before_next_prompt"] = first_before_prompt
    return base


def _outcome(
    after: list[_Prepared],
    linked: Mapping[str, TraceEvent],
) -> dict[str, object]:
    tests = [prepared.event for prepared in after if prepared.is_test]
    last_test = None
    if tests:
        last = tests[-1]
        last_test = {"event_id": last.event_id, **_result_facts(linked.get(last.event_id))}
    return {
        "tests_run": len(tests),
        "last_test": last_test,
        "errors": sum(1 for prepared in after if prepared.event.event_type == EventType.ERROR),
        "session_end_recorded": any(prepared.event.event_type == EventType.SESSION_END for prepared in after),
    }


def _session_facts(prepared: list[_Prepared]) -> dict[str, bool]:
    return {
        "tests_run": any(item.is_test for item in prepared),
        "errors": any(item.event.event_type == EventType.ERROR for item in prepared),
        "session_end_recorded": any(item.event.event_type == EventType.SESSION_END for item in prepared),
    }


def analyze_session(
    events: list[TraceEvent],
    *,
    session_id: str = "",
    started_at: float | None = None,
    workspace: str = "",
    extra_dirs: Iterable[str] = (),
    user_dirs: Iterable[str] = USER_SKILL_DIRS,
    catalog: SkillCatalog | None = None,
) -> dict[str, Any]:
    """Build the available -> loaded -> followed -> outcome report for one session."""
    events, skipped = _sanitize_events(events)
    ordered = sorted(enumerate(events), key=lambda pair: (pair[1].timestamp, pair[0]))
    session_cwd = _session_cwd([event for _, event in ordered])
    cwd = session_cwd or workspace or os.getcwd()
    cwd_source = "session_start" if session_cwd else ("--workspace" if workspace else "current_directory")
    if catalog is None:
        catalog = discover_skills(cwd, extra_dirs, user_dirs=user_dirs)
    base = _real(cwd) or cwd

    candidates = [event.timestamp for _, event in ordered]
    meta_start = _coerce_timestamp(started_at) if started_at else None
    if meta_start:
        candidates.append(meta_start)
    start = min(candidates) if candidates else 0.0
    first_edit = next((pos for pos, (_, event) in enumerate(ordered) if _is_edit(event)), None)
    linked = _linked_results([event for _, event in ordered])
    position = {index: pos for pos, (index, _) in enumerate(ordered)}

    grouped: dict[str, list[SkillLoad]] = {}
    first_pos: dict[str, int] = {}
    first_event: dict[str, TraceEvent] = {}
    for name, method, invoked_by, detail, index, event in _find_loads(ordered, catalog, base):
        pos = position[index]
        load = SkillLoad(
            name=name,
            method=method,
            invoked_by=invoked_by,
            detail=detail,
            event_id=event.event_id,
            timestamp=event.timestamp,
            seconds_since_start=max(0.0, event.timestamp - start),
            loaded_before_first_edit=None if first_edit is None else pos < first_edit,
            load_succeeded=_load_succeeded(linked.get(event.event_id)) if method != "user_prompt" else None,
        )
        grouped.setdefault(name, []).append(load)
        first_pos.setdefault(name, pos)
        first_event.setdefault(name, event)

    measurable_from = min((first_pos[n] for n in grouped if n in catalog.skills), default=None)
    prepared = [
        _prepare_event(event, base, resolve=measurable_from is not None and pos > measurable_from)
        for pos, (_, event) in enumerate(ordered)
    ]

    loaded = []
    for name in sorted(grouped, key=lambda n: first_pos[n]):
        definition = catalog.skills.get(name)
        after = prepared[first_pos[name] + 1:]
        loaded.append({
            "name": name,
            "definition_found": definition is not None,
            "source": definition.source if definition else None,
            "dir": definition.source_dir if definition else None,
            "loads": [load.to_dict() for load in grouped[name]],
            "follow": _follow_signals(definition, first_event[name], after),
            "outcome_after_load": _outcome(after, linked),
        })

    available = []
    for name in sorted(catalog.skills):
        entry = catalog.skills[name].summary()
        entry["also_in"] = list(catalog.also_in.get(name, []))
        available.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "session",
        "session_id": session_id,
        "workspace": _display_path(cwd),
        "workspace_source": cwd_source,
        "availability_basis": "filesystem_now",
        "scanned_dirs": [
            {"dir": scanned.display, "source": scanned.source, "exists": scanned.exists, "skills": scanned.skills}
            for scanned in catalog.scanned
        ],
        "context_files": context_files(cwd),
        "available": available,
        "loaded": loaded,
        "available_not_loaded": [name for name in sorted(catalog.skills) if name not in grouped],
        "first_edit_recorded": first_edit is not None,
        "session_facts": _session_facts(prepared),
        "events_skipped_malformed": skipped,
        "not_measurable": list(NOT_MEASURABLE),
    }


# --------------------------------------------------------------------------- #
# Aggregate analysis
# --------------------------------------------------------------------------- #


def _ratio(numerator: int, denominator: int) -> dict[str, object]:
    return {"count": numerator, "n": denominator}


def aggregate_reports(
    reports: list[dict[str, Any]],
    *,
    min_sessions: int = 5,
) -> dict[str, Any]:
    """Aggregate per-session reports into per-skill counts.  No scores."""
    names: set[str] = set()
    for report in reports:
        names.update(entry["name"] for entry in report["available"])
        names.update(entry["name"] for entry in report["loaded"])

    skills = []
    for name in sorted(names):
        available_sessions = [r for r in reports if any(e["name"] == name for e in r["available"])]
        loaded_entries = []
        for report in reports:
            for entry in report["loaded"]:
                if entry["name"] == name:
                    loaded_entries.append((report, entry))
        loaded_ids = {report["session_id"] for report, _ in loaded_entries}
        loaded_while_available = sum(1 for r in available_sessions if r["session_id"] in loaded_ids)

        user = model = 0
        first_offsets = []
        before_edit = before_edit_n = 0
        follow = follow_n = 0
        kinds = {"resource_read": 0, "script_run": 0, "command_match": 0}
        tests_after = 0
        for _, entry in loaded_entries:
            loads = entry["loads"]
            user += sum(1 for load in loads if load["invoked_by"] == "user")
            model += sum(1 for load in loads if load["invoked_by"] == "model")
            first = loads[0]
            first_offsets.append(first["seconds_since_start"])
            if first["loaded_before_first_edit"] is not None:
                before_edit_n += 1
                before_edit += 1 if first["loaded_before_first_edit"] else 0
            if entry["follow"]["status"] != "not_measurable":
                follow_n += 1
                follow += 1 if entry["follow"]["status"] == "signals_observed" else 0
                for kind in kinds:
                    if entry["follow"]["signals"][kind]["count"]:
                        kinds[kind] += 1
            if entry["outcome_after_load"]["tests_run"]:
                tests_after += 1

        not_loaded = [r for r in available_sessions if r["session_id"] not in loaded_ids]
        loaded_reports = [report for report, _ in loaded_entries]
        comparison: dict[str, Any] = {
            "label": "correlational; session-level facts; task mix is not controlled",
            "loaded_n": len(loaded_reports),
            "available_not_loaded_n": len(not_loaded),
            "suppressed": len(loaded_reports) < min_sessions or len(not_loaded) < min_sessions,
        }
        if not comparison["suppressed"]:
            for fact in ("tests_run", "errors", "session_end_recorded"):
                comparison[fact] = {
                    "loaded": _ratio(sum(1 for r in loaded_reports if r["session_facts"][fact]), len(loaded_reports)),
                    "available_not_loaded": _ratio(sum(1 for r in not_loaded if r["session_facts"][fact]), len(not_loaded)),
                }

        skills.append({
            "name": name,
            "sessions_available": len(available_sessions),
            "sessions_loaded": len(loaded_entries),
            "load_rate": _ratio(loaded_while_available, len(available_sessions)),
            "invocations": {"user": user, "model": model},
            "median_seconds_to_load": round(statistics.median(first_offsets), 3) if first_offsets else None,
            "loaded_before_first_edit": _ratio(before_edit, before_edit_n),
            "follow_signals": _ratio(follow, follow_n),
            "follow_signal_kinds": {kind: _ratio(count, follow_n) for kind, count in kinds.items()},
            "tests_after_load": _ratio(tests_after, len(loaded_entries)),
            "definition_found": any(entry["definition_found"] for _, entry in loaded_entries) or bool(available_sessions),
            "comparison": comparison,
        })

    never_loaded = [
        {"name": skill["name"], "sessions_available": skill["sessions_available"]}
        for skill in skills
        if skill["sessions_available"] >= min_sessions and skill["sessions_loaded"] == 0
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "aggregate",
        "sessions": len(reports),
        "min_sessions": min_sessions,
        "availability_basis": "filesystem_now",
        "skills": skills,
        "never_loaded": sorted(never_loaded, key=lambda item: (-item["sessions_available"], item["name"])),
        "not_measurable": list(NOT_MEASURABLE),
    }


def build_session_report(
    store: TraceStore,
    session_id: str,
    *,
    workspace: str = "",
    extra_dirs: Iterable[str] = (),
    user_dirs: Iterable[str] = USER_SKILL_DIRS,
    catalogs: dict[str, SkillCatalog] | None = None,
) -> dict[str, Any]:
    meta: SessionMeta | None
    try:
        meta = store.load_meta(session_id)
    except Exception:
        meta = None
    events = store.load_events(session_id)
    clean, _ = _sanitize_events(events)
    cwd = _session_cwd(sorted(clean, key=lambda event: event.timestamp)) or workspace or os.getcwd()
    catalog = None
    if catalogs is not None:
        catalog = catalogs.get(cwd)
        if catalog is None:
            catalog = catalogs[cwd] = discover_skills(cwd, extra_dirs, user_dirs=user_dirs)
    return analyze_session(
        clean,
        session_id=session_id,
        started_at=meta.started_at if meta else None,
        workspace=workspace,
        extra_dirs=extra_dirs,
        user_dirs=user_dirs,
        catalog=catalog,
    )


def build_aggregate_report(
    store: TraceStore,
    *,
    since: float,
    workspace: str = "",
    extra_dirs: Iterable[str] = (),
    user_dirs: Iterable[str] = USER_SKILL_DIRS,
    min_sessions: int = 5,
) -> dict[str, Any]:
    catalogs: dict[str, SkillCatalog] = {}
    extra = list(extra_dirs)
    reports = []
    skipped = 0
    for meta in store.list_sessions():
        try:
            started = _coerce_timestamp(meta.started_at)
            if started is not None and started < since:
                continue
            reports.append(build_session_report(
                store, meta.session_id, workspace=workspace, extra_dirs=extra,
                user_dirs=user_dirs, catalogs=catalogs,
            ))
        except Exception:  # one unreadable or malformed session must not abort the report
            skipped += 1
            continue
    report = aggregate_reports(reports, min_sessions=min_sessions)
    report["since"] = since
    report["sessions_skipped_unreadable"] = skipped
    return report


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def _frac(ratio: Mapping[str, object]) -> str:
    return f"{ratio['count']}/{ratio['n']}" if ratio["n"] else "-"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 120:
        return f"{value:.0f}s"
    if value < 7200:
        return f"{value / 60:.0f}m"
    return f"{value / 3600:.1f}h"


def _describe_result(facts: Mapping[str, object] | None) -> str:
    if not facts or facts.get("linked_result") is None:
        return "no linked result"
    if facts["linked_result"] == "error":
        return "error event recorded"
    parts = []
    if facts.get("exit_code") is not None:
        parts.append(f"exit_code={facts['exit_code']}")
    if facts.get("success") is not None:
        parts.append(f"success={str(facts['success']).lower()}")
    return ", ".join(parts) if parts else "result recorded without exit status"


def _yes_no(value: bool | None, unknown: str = "n/a") -> str:
    return unknown if value is None else ("yes" if value else "no")


def format_session_text(report: Mapping[str, Any], out: TextIO | None = None) -> None:
    out = out or sys.stdout
    w = out.write
    w(f"Skills - session {str(report['session_id'])[:16]}\n")
    w(f"Workspace: {report['workspace']} (from {report['workspace_source']})\n")
    if report.get("events_skipped_malformed"):
        w(f"Skipped {report['events_skipped_malformed']} event(s) without a usable timestamp.\n")
    existing = [d for d in report["scanned_dirs"] if d["exists"]]
    w(f"Scanned {len(report['scanned_dirs'])} skill directories ({len(existing)} exist). "
      "Availability reflects the filesystem now, not session time.\n\n")

    w("Context files (auto-injected by most agents; load not observable from hooks):\n")
    for item in report["context_files"]:
        w(f"  {item['path']:<34} {'present' if item['present'] else 'absent'}\n")

    w(f"\nAvailable skills ({len(report['available'])}):\n")
    for entry in report["available"]:
        w(f"  {entry['name']:<28} {entry['dir']}\n")
    if not report["available"]:
        w("  (none found)\n")

    w(f"\nLoaded skills ({len(report['loaded'])}):\n")
    if not report["loaded"]:
        w("  (no load recorded)\n")
    for entry in report["loaded"]:
        w(f"  {entry['name']}\n")
        for load in entry["loads"]:
            w(f"    load: {load['method']} by {load['invoked_by']} ({load['detail']}) at +{_fmt_seconds(load['seconds_since_start'])}"
              f"; before first edit: {_yes_no(load['loaded_before_first_edit'], 'no edit recorded')}"
              f"; load result: {_yes_no(load['load_succeeded'], 'not recorded')}\n")
        if not entry["definition_found"]:
            w("    definition: not found in scanned directories (use --skills-dir)\n")
        follow = entry["follow"]
        reason = f" ({follow['reason']})" if follow.get("reason") else ""
        timing = ""
        if follow.get("seconds_to_first_signal") is not None:
            turn = "same turn" if follow["first_signal_before_next_prompt"] else "after a later prompt"
            timing = f"; first signal +{_fmt_seconds(follow['seconds_to_first_signal'])} after load ({turn})"
        w(f"    follow: {follow['status']}{reason}{timing}\n")
        signals = follow["signals"]
        if signals["resource_read"]["count"]:
            w(f"      resource_read x{signals['resource_read']['count']}: {', '.join(signals['resource_read']['paths'])}\n")
        if signals["script_run"]["count"]:
            w(f"      script_run x{signals['script_run']['count']}: {', '.join(signals['script_run']['paths'])}\n")
        if signals["command_match"]["count"]:
            matched = ", ".join(f"{sig} x{n}" for sig, n in signals["command_match"]["by_signature"].items())
            w(f"      command_match in {signals['command_match']['count']} command(s): {matched}\n")
        outcome = entry["outcome_after_load"]
        last = _describe_result(outcome["last_test"]) if outcome["last_test"] else "n/a"
        w(f"    after load: tests run {outcome['tests_run']} (last: {last}); "
          f"errors {outcome['errors']}; session_end recorded: {_yes_no(outcome['session_end_recorded'])}\n")

    not_loaded = report["available_not_loaded"]
    w(f"\nAvailable but not loaded ({len(not_loaded)}): {', '.join(not_loaded) if not_loaded else '-'}\n")
    _write_not_measurable(report, out)


def format_aggregate_text(report: Mapping[str, Any], out: TextIO | None = None) -> None:
    import datetime

    out = out or sys.stdout
    w = out.write
    since = report.get("since")
    window = ""
    if since:
        window = f" since {datetime.datetime.fromtimestamp(since, datetime.timezone.utc).strftime('%Y-%m-%d')}"
    w(f"Skills across {report['sessions']} sessions{window}. "
      "Availability reflects the filesystem now, not session time.\n")
    if report.get("sessions_skipped_unreadable"):
        w(f"Skipped {report['sessions_skipped_unreadable']} unreadable or malformed session(s).\n")
    w("Counts are shown as count/n. There is no score or grade.\n\n")
    if not report["skills"]:
        w("No skills available or loaded in the selected sessions.\n")
        _write_not_measurable(report, out)
        return

    width = max(5, min(48, max(len(skill["name"]) + 1 for skill in report["skills"])))
    header = (f"{'skill':<{width}} {'avail':>5} {'loaded':>6} {'load rate':>9} {'user/model':>10} "
              f"{'median t':>8} {'pre-edit':>8} {'follow':>7} {'tests':>6}\n")
    w(header)
    w("-" * (len(header) - 1) + "\n")
    missing = False
    for skill in report["skills"]:
        inv = skill["invocations"]
        label = skill["name"] if skill["definition_found"] else skill["name"] + "*"
        missing = missing or not skill["definition_found"]
        w(f"{label[:width]:<{width}} {skill['sessions_available']:>5} {skill['sessions_loaded']:>6} "
          f"{_frac(skill['load_rate']):>9} {inv['user']:>4}/{inv['model']:<5} "
          f"{_fmt_seconds(skill['median_seconds_to_load']):>8} {_frac(skill['loaded_before_first_edit']):>8} "
          f"{_frac(skill['follow_signals']):>7} {_frac(skill['tests_after_load']):>6}\n")
    w("\nColumns: load rate = sessions loaded while available / sessions available; "
      "user/model = load invocations; median t = time from session start to first load; "
      "pre-edit = loaded before first edit (n = loaded sessions with an edit); "
      "follow = sessions with follow signals (n = measurable loads); "
      "tests = sessions with a test command after the load.\n")
    if missing:
        w("* loaded, but no definition found in the scanned directories; follow is not measurable "
          "(add the directory with --skills-dir).\n")

    with_kinds = [skill for skill in report["skills"] if skill["follow_signals"]["count"]]
    if with_kinds:
        w("\nFollow signal kinds (sessions with each kind / measurable loads):\n")
        for skill in with_kinds:
            kinds = skill["follow_signal_kinds"]
            w(f"  {skill['name']:<{width}} resource_read {_frac(kinds['resource_read']):>7}  "
              f"script_run {_frac(kinds['script_run']):>7}  command_match {_frac(kinds['command_match']):>7}\n")

    w(f"\nNever loaded (available in >= {report['min_sessions']} sessions, loaded in 0):\n")
    if report["never_loaded"]:
        for item in report["never_loaded"]:
            w(f"  {item['name']} (available in {item['sessions_available']} sessions)\n")
    else:
        w("  (none)\n")

    compared = [skill for skill in report["skills"] if not skill["comparison"]["suppressed"]]
    w("\nLoaded vs available-but-not-loaded (correlational; session-level facts; task mix not controlled):\n")
    if not compared:
        w(f"  suppressed: needs n >= {report['min_sessions']} in both groups for a skill\n")
    for skill in compared:
        comp = skill["comparison"]
        w(f"  {skill['name']}: loaded n={comp['loaded_n']}, not loaded n={comp['available_not_loaded_n']}\n")
        for fact in ("tests_run", "errors", "session_end_recorded"):
            w(f"    {fact:<22} {_frac(comp[fact]['loaded']):>8} vs {_frac(comp[fact]['available_not_loaded']):>8}\n")
    _write_not_measurable(report, out)


def _write_not_measurable(report: Mapping[str, Any], out: TextIO) -> None:
    out.write("\nNot measurable here:\n")
    for line in report["not_measurable"]:
        out.write(f"  - {line}\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def cmd_skills(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    workspace = getattr(args, "workspace", "") or ""
    extra_dirs = list(getattr(args, "skills_dirs", None) or [])
    min_sessions = int(getattr(args, "min_sessions", 5))
    if min_sessions < 1:
        sys.stderr.write("--min-sessions must be at least 1\n")
        return 2
    as_json = getattr(args, "format", "text") == "json"

    raw_session = getattr(args, "session_id", None)
    if raw_session:
        session_id = store.find_session(raw_session)
        if not session_id:
            sys.stderr.write(f"Session not found: {raw_session}\n")
            return 1
        try:
            report = build_session_report(store, session_id, workspace=workspace, extra_dirs=extra_dirs)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Only the exception type: messages can echo trace content.
            sys.stderr.write(f"Could not read session {session_id}: {type(exc).__name__}\n")
            return 1
        if as_json:
            sys.stdout.write(json.dumps(report, indent=2) + "\n")
        else:
            format_session_text(report)
        return 0

    try:
        since = parse_since(getattr(args, "since", "30d"))
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    report = build_aggregate_report(
        store,
        since=since,
        workspace=workspace,
        extra_dirs=extra_dirs,
        min_sessions=min_sessions,
    )
    if as_json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        format_aggregate_text(report)
    return 0


__all__ = [
    "CONTEXT_FILES",
    "NOT_MEASURABLE",
    "REPO_SKILL_DIRS",
    "USER_SKILL_DIRS",
    "SkillCatalog",
    "SkillDefinition",
    "aggregate_reports",
    "analyze_session",
    "build_aggregate_report",
    "build_session_report",
    "cmd_skills",
    "command_signatures",
    "discover_skills",
    "event_arguments",
    "extract_skill_commands",
]
