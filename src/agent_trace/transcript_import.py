"""Read GitHub Copilot and OpenAI Codex session transcripts for ``skills``.

Both formats are internal to their agents and undocumented, so they can
change without notice.  Parsing is defensive: unknown record types are
ignored, a line that is not a JSON object (for example a truncated last
line) is skipped, and missing fields fall back to empty values.

Only what the skills report needs is kept: the working directory, the
skills each session listed as available, skill loads, tool names, file
paths, shell commands, and exit status.  Prompt text, model output, and
reasoning are not stored.  A user prompt is reduced to its skill mention
tokens (``/name`` at the start, ``$name`` anywhere) so user loads can be
counted.  Copilot ``user.message`` records the user did not type (skill
bodies, runtime continuations, messages from other sessions) are not
prompts and are dropped.

Recorded availability goes into the SESSION_START event as
``available_skills`` (a list of ``{"name", "source", "path"}``) together
with ``availability: "recorded"``.  A session that lists no skills has
neither field, and the report falls back to skill directories on disk.
A Copilot builtin skill path that a CLI upgrade deleted is replaced by the
same skill in the newest installed package version.  Copilot
``session.resume`` records become continuation SESSION_START events, so load
timing is measured from the latest activation instead of the session's
original creation time.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore, validate_session_id

_MAX_LINE_CHARS = 50_000_000
_MAX_TOKENS = 50
_NAME = re.compile(r"^[A-Za-z0-9_.@:/+-]{1,200}$")
_LEADING_SLASH = re.compile(r"^\s*(/[A-Za-z0-9_.:-]+)(?=\s|$)")
_DOLLAR = re.compile(r"(?<![\w$])\$[A-Za-z][A-Za-z0-9_.:-]*")
_PATH_KEYS = ("path", "file_path", "absolute_path", "filePath", "target_file", "notebook_path", "uri")
_COMMAND_KEYS = ("command", "cmd")
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish"})
_SHELL_FLAGS = frozenset({"-c", "-lc", "-cl", "-ic", "-ilc", "-lic"})

_COPILOT_BLOCK = re.compile(r"<available_skills>(.*?)</available_skills>", re.DOTALL)
_COPILOT_SKILL = re.compile(r"<skill>\s*<name>\s*(.*?)\s*</name>(.*?)</skill>", re.DOTALL)
_COPILOT_LOCATION = re.compile(r"<location>\s*(.*?)\s*</location>", re.DOTALL)
_COPILOT_PKG_BUILTIN = re.compile(
    r"^(?P<pkg>.*[\\/]pkg[\\/][^\\/]+)[\\/](?P<version>[^\\/]+)[\\/]builtin[\\/](?P<rest>.+)$"
)
_DIGIT_RUN = re.compile(r"([0-9]+)")
# user.message sources that mark text the user did not type: a skill body
# (skill-<name>), a runtime continuation (system, autopilot), or a message
# from another session (agent-<id>).  A typed prompt has no source.
_COPILOT_NOT_PROMPT = re.compile(r"^(?:skill-|agent-|system$|autopilot$)")

_CODEX_BLOCK = re.compile(r"<skills_instructions>(.*?)</skills_instructions>", re.DOTALL)
_CODEX_ROOT = re.compile(r"^- `([^`]+)` = `([^`]+)`\s*$")
_CODEX_SKILL = re.compile(
    r"^- (.+?): (?:.*? )?\((file|executor package|cloud package): ([^()\n]+)\)\s*$"
)
_CODEX_INJECTED = re.compile(r"^\s*<skill>\s*<name>\s*(.*?)\s*</name>\s*(?:<path>\s*(.*?)\s*</path>)?", re.DOTALL)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object lines; skip blank, oversized, or malformed lines."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or len(line) > _MAX_LINE_CHARS:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _timestamp(value: object) -> float | None:
    """Parse an ISO 8601 timestamp; return ``None`` when it is unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000 if number > 1e11 else number
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.timestamp()


def skill_tokens(text: str) -> str:
    """Reduce a prompt to its skill mention tokens; drop everything else."""
    if not text:
        return ""
    tokens: list[str] = []
    lead = _LEADING_SLASH.match(text)
    if lead:
        tokens.append(lead.group(1))
    for match in _DOLLAR.findall(text):
        if match not in tokens:
            tokens.append(match)
        if len(tokens) >= _MAX_TOKENS:
            break
    return " ".join(tokens)


def _kept_arguments(raw: object) -> dict[str, str]:
    """Keep only path, command, and skill arguments.  File bodies are dropped."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    if not isinstance(raw, Mapping):
        return {}
    kept: dict[str, str] = {}
    for key in (*_PATH_KEYS, *_COMMAND_KEYS, "skill"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            kept[key] = value
        elif key in _COMMAND_KEYS and isinstance(value, list) and all(isinstance(v, str) for v in value):
            kept[key] = _unwrap_shell(value)
    return kept


def _unwrap_shell(argv: list[str]) -> str:
    """Return the script of ``[shell, -lc, script]``; else the joined argv."""
    if len(argv) == 3 and os.path.basename(argv[0]) in _SHELLS and argv[1] in _SHELL_FLAGS:
        return argv[2]
    return shlex.join(argv)


@dataclass
class _Session:
    """Events for one session, written to the store once complete."""

    agent: str
    session_id: str = ""
    cwd: str = ""
    started: float | None = None
    last_ts: float | None = None
    events: list[TraceEvent] = field(default_factory=list)
    available: dict[str, dict[str, Any]] = field(default_factory=dict)
    listed: bool = False
    ended: bool = False

    def ts(self, raw: object) -> float:
        value = _timestamp(raw)
        if value is None:
            value = self.last_ts if self.last_ts is not None else (self.started or 0.0)
        self.last_ts = value
        return value

    def add(self, event_type: EventType, raw_ts: object, data: dict[str, Any], parent_id: str = "") -> TraceEvent:
        event = TraceEvent(event_type=event_type, timestamp=self.ts(raw_ts), data=data, parent_id=parent_id)
        self.events.append(event)
        return event

    def make_available(self, name: str, source: str, path: str, *, listed: bool) -> None:
        name = name.strip()
        if not _NAME.match(name):
            return
        entry = self.available.get(name)
        if entry is None:
            self.available[name] = {"name": name, "source": source, "path": path, "listed": listed}
            self.listed = self.listed or listed
            return
        if path and not entry.get("path"):
            entry["path"] = path
        if listed and not entry.get("listed"):
            entry["listed"] = True
            self.listed = True

    def write(self, store: TraceStore, fallback_start: float) -> str:
        """Write the session.  Returns ``recorded``, ``unrecorded``, or ``duplicate``."""
        start = self.started
        if start is None:
            stamps = [event.timestamp for event in self.events if event.timestamp]
            start = min(stamps) if stamps else fallback_start
        for event in self.events:
            if not event.timestamp:
                event.timestamp = start
        meta = SessionMeta(session_id=self.session_id, started_at=start, agent_name=self.agent)
        if store.create_session(meta, if_missing=True) is None:
            return "duplicate"
        data: dict[str, Any] = {"cwd": self.cwd, "transcript": self.agent}
        if self.listed:
            data["availability"] = "recorded"
            data["available_skills"] = sorted(self.available.values(), key=lambda item: item["name"])
        store.append_events(self.session_id, [
            TraceEvent(event_type=EventType.SESSION_START, timestamp=start, data=data),
            *self.events,
        ])
        return "recorded" if self.listed else "unrecorded"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# --------------------------------------------------------------------------- #
# GitHub Copilot (CLI and app): <COPILOT_HOME>/session-state/<id>/events.jsonl
# --------------------------------------------------------------------------- #


def default_copilot_dir() -> str:
    """Copilot's config directory: ``$COPILOT_HOME`` when set, else ``~/.copilot``."""
    return os.environ.get("COPILOT_HOME", "").strip() or "~/.copilot"


def discover_copilot_sessions(copilot_dir: str) -> list[Path]:
    """Return every ``session-state/<id>/events.jsonl``; directories without one are skipped."""
    root = Path(copilot_dir).expanduser() / "session-state"
    try:
        children = sorted(root.iterdir())
    except OSError:
        return []
    found = []
    for child in children:
        candidate = child / "events.jsonl"
        if os.path.isfile(candidate):
            found.append(candidate)
    return found


def _version_key(name: str) -> list[object]:
    # re.split with a group alternates text and digit runs, so list items compare by type.
    return [int(part) if index % 2 else part for index, part in enumerate(_DIGIT_RUN.split(name))]


def _copilot_newest_builtin(path: str) -> str:
    """Return ``path`` in the newest Copilot package version that has it, else ``""``.

    Copilot CLI records a builtin skill as
    ``<cache>/pkg/<platform>/<version>/builtin/<name>/SKILL.md`` and deletes
    old version directories when it upgrades, so recorded paths go stale.
    """
    match = _COPILOT_PKG_BUILTIN.match(path)
    if not match:
        return ""
    pkg = match.group("pkg")
    try:
        versions = [entry.name for entry in os.scandir(pkg) if "0" <= entry.name[:1] <= "9"]
    except OSError:
        return ""
    for version in sorted(versions, key=_version_key, reverse=True):
        candidate = os.path.join(pkg, version, "builtin", match.group("rest"))
        if os.path.isfile(candidate):
            return candidate
    return ""


def _copilot_current_path(path: str) -> str:
    """Return where a recorded SKILL.md is now: the path itself, a newer package version, or ``""``."""
    if not path:
        return ""
    if os.path.isfile(path):
        return path
    return _copilot_newest_builtin(path)


def copilot_skill_index(paths: Iterable[Path]) -> tuple[dict[str, str], list[str]]:
    """Read the recorded skill loads of every session once.

    Returns ``(known_paths, roots)``.  ``known_paths`` maps a skill name to its
    SKILL.md when every load of it that still resolves to a file resolves to
    the same one.  ``roots`` are the skill directories those files are in (the
    directory holding ``<name>/SKILL.md``); builtin skills use the newest
    Copilot package version.  Project loads are left out because a project
    skill belongs to one workspace.
    """
    seen: dict[str, set[str]] = {}
    roots: dict[str, None] = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if '"skill.invoked' not in line or len(line) > _MAX_LINE_CHARS:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    data = _mapping(record.get("data")) if isinstance(record, dict) else {}
                    name, skill_path = _text(data.get("name")).strip(), _text(data.get("path")).strip()
                    if not name or not skill_path or _text(data.get("source")) == "project":
                        continue
                    current = _copilot_current_path(skill_path)
                    if not current:
                        continue
                    seen.setdefault(name, set()).add(current)
                    newest = _copilot_newest_builtin(current) or current
                    roots.setdefault(os.path.dirname(os.path.dirname(newest)), None)
        except OSError:
            continue
    known = {name: next(iter(found)) for name, found in seen.items() if len(found) == 1}
    return known, list(roots)


def copilot_skill_paths(paths: Iterable[Path]) -> dict[str, str]:
    """Map skill name to the SKILL.md its recorded loads resolve to; see ``copilot_skill_index``."""
    return copilot_skill_index(paths)[0]


def _copilot_available(session: _Session, data: Mapping[str, Any]) -> None:
    texts = [_text(data.get("content"))]
    for block in data.get("contentBlocks") or ():
        texts.append(_text(_mapping(block).get("content")))
    for text in texts:
        if "<available_skills>" not in text:
            continue
        for block in _COPILOT_BLOCK.findall(text):
            for name, rest in _COPILOT_SKILL.findall(block):
                location = _COPILOT_LOCATION.search(rest)
                session.make_available(name, location.group(1).strip() if location else "copilot", "", listed=True)


def _same_skill(a: str, b: str) -> bool:
    return a == b or a.endswith(":" + b) or b.endswith(":" + a)


def import_copilot_session(
    path: str | os.PathLike[str],
    store: TraceStore,
    known_paths: Mapping[str, str] | None = None,
) -> str:
    """Import one Copilot ``events.jsonl``.

    Returns ``recorded`` (the session listed its available skills),
    ``unrecorded``, ``duplicate``, or ``empty`` (no session records).
    """
    path = Path(path)
    session = _Session(agent="copilot", session_id=validate_session_id(path.parent.name))
    calls: dict[str, TraceEvent] = {}
    pending_skill_calls: list[tuple[str, TraceEvent]] = []
    seen = False
    for record in _records(path):
        kind = _text(record.get("type"))
        data = _mapping(record.get("data"))
        raw_ts = record.get("timestamp")
        if kind == "session.start":
            seen = True
            context = _mapping(data.get("context"))
            session.cwd = session.cwd or _text(context.get("cwd")) or _text(data.get("cwd"))
            start = _timestamp(data.get("startTime")) or _timestamp(raw_ts)
            if start is not None and session.started is None:
                session.started = start
        elif kind == "session.resume":
            seen = True
            context = _mapping(data.get("context"))
            cwd = _text(context.get("cwd")) or _text(data.get("cwd"))
            session.cwd = session.cwd or cwd
            resume_data = {"source": "resume"}
            if cwd:
                resume_data["cwd"] = cwd
            resume_time = _timestamp(data.get("resumeTime"))
            session.add(
                EventType.SESSION_START,
                resume_time if resume_time is not None else raw_ts,
                resume_data,
            )
        elif kind == "system.message":
            _copilot_available(session, data)
        elif kind == "user.message":
            seen = True
            if _COPILOT_NOT_PROMPT.match(_text(data.get("source")).strip()):
                continue
            session.add(EventType.USER_PROMPT, raw_ts, {"prompt": skill_tokens(_text(data.get("content")))})
        elif kind == "tool.execution_start":
            seen = True
            tool = _text(data.get("toolName")).strip()
            if not tool:
                continue
            args = _kept_arguments(data.get("arguments"))
            event = session.add(EventType.TOOL_CALL, raw_ts, {"tool_name": tool, "arguments": args})
            call_id = _text(data.get("toolCallId"))
            if call_id:
                calls[call_id] = event
            if tool.lower() == "skill" and args.get("skill"):
                pending_skill_calls.append((args["skill"], event))
        elif kind == "tool.execution_complete":
            call = calls.pop(_text(data.get("toolCallId")), None)
            if call is None:
                continue
            result: dict[str, Any] = {"tool_name": call.data.get("tool_name", "")}
            if isinstance(data.get("success"), bool):
                result["success"] = data["success"]
            exit_code = _mapping(data.get("shellExecution")).get("exitCode")
            if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                result["exit_code"] = exit_code
            session.add(EventType.TOOL_RESULT, raw_ts, result, parent_id=call.event_id)
        elif kind in ("skill.invoked", "skill.invoked_ref"):
            seen = True
            name = _text(data.get("name")).strip()
            if not name:
                continue
            recorded = _text(data.get("path"))
            session.make_available(name, _text(data.get("source")) or "copilot",
                                   _copilot_current_path(recorded) or recorded, listed=False)
            match = next((i for i in range(len(pending_skill_calls) - 1, -1, -1)
                          if _same_skill(pending_skill_calls[i][0], name)), None)
            if match is not None:
                pending_skill_calls.pop(match)
            elif "user" in _text(data.get("trigger")).lower():
                session.add(EventType.USER_PROMPT, raw_ts, {"prompt": "/" + name})
            else:
                call = session.add(EventType.TOOL_CALL, raw_ts, {"tool_name": "skill", "arguments": {"skill": name}})
                session.add(EventType.TOOL_RESULT, raw_ts, {"tool_name": "skill", "success": True},
                            parent_id=call.event_id)
        elif kind == "session.shutdown":
            session.ended = True
            session.add(EventType.SESSION_END, raw_ts, {})
    if not seen:
        return "empty"
    for name, entry in session.available.items():
        if not entry["path"] and entry["source"] != "project" and known_paths and name in known_paths:
            entry["path"] = known_paths[name]
    return session.write(store, _mtime(path))


# --------------------------------------------------------------------------- #
# OpenAI Codex: <CODEX_HOME>/sessions/**/rollout-*.jsonl (+ archived_sessions)
# --------------------------------------------------------------------------- #


def default_codex_dir() -> str:
    """Codex's home: a non-empty ``$CODEX_HOME`` as given, else ``~/.codex`` expanded.

    Mirrors ``cli._codex_config_dir``: Codex does not expand ``~`` in
    ``CODEX_HOME``, so neither does this.
    """
    return os.environ.get("CODEX_HOME") or str(Path.home() / ".codex")


def discover_codex_rollouts(codex_dir: str) -> list[Path]:
    """Return rollout files under ``sessions`` and ``archived_sessions``.

    Archived threads are included: they are past sessions moved out of the
    active list, and the report covers past sessions.
    """
    root = Path(codex_dir)
    found: list[Path] = []
    for sub in ("sessions", "archived_sessions"):
        base = root / sub
        if not os.path.isdir(base):
            continue
        try:
            found.extend(sorted(path for path in base.rglob("rollout-*.jsonl") if os.path.isfile(path)))
        except OSError:
            continue
    return found


def _codex_available(session: _Session, text: str, aliases: dict[str, str]) -> None:
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            if stripped.startswith("### Skill roots"):
                section = "roots"
            elif stripped.startswith(("### Available skills", "### Added or updated skills")):
                section = "skills"
            else:
                section = ""
            continue
        if stripped.startswith("## "):
            section = ""
            continue
        if section == "roots":
            root = _CODEX_ROOT.match(stripped)
            if root:
                aliases[root.group(1)] = root.group(2)
        elif section == "skills":
            skill = _CODEX_SKILL.match(stripped)
            if not skill:
                continue
            name, kind, locator = skill.group(1), skill.group(2), skill.group(3).strip()
            path = ""
            if kind == "file":
                head, _, rest = locator.partition("/")
                if head in aliases:
                    path = os.path.join(aliases[head], rest) if rest else aliases[head]
                elif os.path.isabs(locator):
                    path = locator
            session.make_available(name, "codex", path, listed=True)


def _codex_message_texts(payload: Mapping[str, Any]) -> list[str]:
    return [_text(_mapping(item).get("text")) for item in payload.get("content") or ()]


def import_codex_rollout(path: str | os.PathLike[str], store: TraceStore) -> str:
    """Import one Codex rollout.

    Returns ``recorded``, ``unrecorded``, or ``duplicate`` for an imported
    thread; ``not_rollout`` when the first record is not ``session_meta``
    (other tools also write ``rollout-*.jsonl`` files there); ``review``
    for an approval review thread (``thread_source: guardian_review``),
    which Codex starts to review a tool call and which never loads skills.
    """
    path = Path(path)
    session = _Session(agent="codex")
    aliases: dict[str, str] = {}
    turn_prompt: TraceEvent | None = None
    turn_prompt_from_user = False
    first = True

    def mention(name: str, raw_ts: object) -> None:
        nonlocal turn_prompt
        token = "$" + name
        if turn_prompt is None:
            turn_prompt = session.add(EventType.USER_PROMPT, raw_ts, {"prompt": token})
        elif token not in turn_prompt.data["prompt"].split():
            turn_prompt.data["prompt"] = (turn_prompt.data["prompt"] + " " + token).strip()

    for record in _records(path):
        kind = _text(record.get("type"))
        payload = _mapping(record.get("payload"))
        raw_ts = record.get("timestamp")
        if first:
            first = False
            if kind != "session_meta":
                return "not_rollout"
            if _text(payload.get("thread_source")) == "guardian_review":
                return "review"
            session.session_id = validate_session_id(_text(payload.get("id")))
            session.cwd = _text(payload.get("cwd"))
            session.started = _timestamp(payload.get("timestamp")) or _timestamp(raw_ts)
            continue
        if kind == "turn_context":
            turn_prompt, turn_prompt_from_user = None, False
        elif kind == "world_state":
            for value in _mapping(payload.get("state")).values():
                body = _text(_mapping(value).get("body"))
                if "skills" in body.lower():
                    _codex_available(session, body, aliases)
        elif kind == "response_item":
            item_type = _text(payload.get("type"))
            role = _text(payload.get("role"))
            if item_type == "message" and role == "developer":
                for text in _codex_message_texts(payload):
                    for block in _CODEX_BLOCK.findall(text):
                        _codex_available(session, block, aliases)
            elif item_type == "message" and role == "user":
                for text in _codex_message_texts(payload):
                    injected = _CODEX_INJECTED.match(text)
                    if injected and _NAME.match(injected.group(1)):
                        session.make_available(injected.group(1), "codex", injected.group(2) or "", listed=False)
                        mention(injected.group(1), raw_ts)
            elif item_type == "function_call" and _text(payload.get("namespace")) == "skills" \
                    and _text(payload.get("name")) == "read":
                try:
                    package = _text(json.loads(_text(payload.get("arguments")) or "{}").get("package"))
                except (ValueError, AttributeError):
                    package = ""
                if package:
                    session.add(EventType.TOOL_CALL, raw_ts, {"tool_name": "skill", "arguments": {"skill": package}})
        elif kind == "event_msg" and _text(payload.get("type")) == "item_completed":
            item = _mapping(payload.get("item"))
            item_type = _text(item.get("type"))
            status = _text(item.get("status"))
            if item_type == "UserMessage":
                tokens = skill_tokens(" ".join(_codex_message_texts(item)))
                if turn_prompt is not None and not turn_prompt_from_user:
                    merged = turn_prompt.data["prompt"].split()
                    merged += [token for token in tokens.split() if token not in merged]
                    turn_prompt.data["prompt"] = " ".join(merged)
                else:
                    turn_prompt = session.add(EventType.USER_PROMPT, raw_ts, {"prompt": tokens})
                turn_prompt_from_user = True
                continue
            if item_type == "CommandExecution":
                command = item.get("command")
                if isinstance(command, list) and all(isinstance(part, str) for part in command):
                    command = _unwrap_shell(command)
                if not isinstance(command, str) or not command:
                    continue
                call = session.add(EventType.TOOL_CALL, raw_ts,
                                   {"tool_name": "exec_command", "arguments": {"command": command}})
            elif item_type == "FileChange":
                call = session.add(EventType.TOOL_CALL, raw_ts, {"tool_name": "apply_patch", "arguments": {}})
            elif item_type == "McpToolCall":
                server, tool = _text(item.get("server")), _text(item.get("tool"))
                call = session.add(EventType.TOOL_CALL, raw_ts,
                                   {"tool_name": f"mcp__{server}__{tool}" if server else tool or "mcp", "arguments": {}})
            else:
                continue
            result: dict[str, Any] = {"tool_name": call.data["tool_name"]}
            exit_code = item.get("exit_code")
            if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                result["exit_code"] = exit_code
                result["success"] = exit_code == 0
            elif status:
                result["success"] = status == "completed"
            session.add(EventType.TOOL_RESULT, raw_ts, result, parent_id=call.event_id)
    if first:
        return "not_rollout"
    return session.write(store, _mtime(path))


__all__ = [
    "copilot_skill_index",
    "copilot_skill_paths",
    "default_codex_dir",
    "default_copilot_dir",
    "discover_codex_rollouts",
    "discover_copilot_sessions",
    "import_codex_rollout",
    "import_copilot_session",
    "skill_tokens",
]
