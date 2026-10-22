"""Read-only verification of installed Codex and Claude Code hooks (not runtime capture)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

try:
    import tomllib as _tomllib
except ModuleNotFoundError:  # Python 3.10: _FeatureFlagReader is used instead.
    _tomllib = None

_REDACTION_PREFIXES = {
    "AGENT_TRACE_REDACT=1 ": "redact",
    "AGENT_TRACE_NO_REDACT=1 ": "no-redact",
}
_MATCH_ALL = (None, "", "*", ".*")
# Codex and Claude Code ignore matchers on these events.
_IGNORED_MATCHER_EVENTS = ("Stop", "UserPromptSubmit")
_AGENTTRACE_COMMAND = "agent-strace hook "
_CLAUDE_TOOL_EVENTS = ("PreToolUse", "PostToolUse", "PostToolUseFailure")
_CLAUDE_SETUP_HINT = "Run agent-strace setup --cli claude and add the printed hooks to that file."
# Codex applies [features] keys in sorted order, so "hooks" wins over its
# legacy alias "codex_hooks" when both are set.
_HOOKS_FEATURE_KEYS = ("hooks", "codex_hooks")
_NOT_BOOLEAN = object()
_WS = re.compile(r"[ \t]*")
_BLANK = re.compile(r"(?:[ \t\r\n]|#[^\n]*)*")
_END_OF_LINE = re.compile(r"[ \t]*(?:#[^\n]*)?(?:\r?\n|\Z)")
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_BASIC_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
_LITERAL_STRING = re.compile(r"'([^'\n]*)'")
_SCALAR = re.compile(r"\d{4}-\d{2}-\d{2}[Tt ]\d{2}:[0-9A-Za-z_+.:-]*|[0-9A-Za-z_+.:-]+")


class _FeatureFlagReader:
    """Find top-level [features] hook flags in TOML without tomllib (Python 3.10).

    Other values are skipped rather than validated, so some invalid TOML that
    tomllib rejects is accepted here. Escapes in quoted keys are not decoded.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.flags: dict[str, object] = {}

    def read(self) -> dict[str, object]:
        table: tuple[str, ...] | None = ()
        while True:
            self._skip(_BLANK)
            if self.pos == len(self.text):
                return self.flags
            if self._accept("[["):
                self._record(self._key(), [])
                self._expect("]]")
                table = None  # Array-of-tables entries are never the top-level table.
            elif self._accept("["):
                table = self._key()
                self._expect("]")
                self._record(table, [])
            else:
                key = self._key()
                self._expect("=")
                value = self._value()
                if table is not None:
                    self._record(table + key, value)
            self._end_line()

    def _record(self, path: tuple[str, ...], value: object) -> None:
        if path[:1] != ("features",):
            return
        if len(path) == 1:
            if isinstance(value, list):  # [features] or features = { ... }
                for key, item in value:
                    self._record(path + key, item)
            return
        name = path[1]
        if name not in _HOOKS_FEATURE_KEYS:
            return
        if len(path) == 2 and not isinstance(value, list):
            if name in self.flags:
                raise self._error(f"duplicate key features.{name}")
            self.flags[name] = value
        else:
            self.flags[name] = _NOT_BOOLEAN  # A table where Codex expects a boolean.

    def _value(self) -> object:
        """Return True, False, inline-table (key, value) pairs, or _NOT_BOOLEAN."""
        self._skip(_WS)
        for quote in ('"', "'"):
            if self._accept(quote * 3):
                self._skip_multiline_string(quote)
                return _NOT_BOOLEAN
        if self.text.startswith(('"', "'"), self.pos):
            self._string()
            return _NOT_BOOLEAN
        if self._accept("["):
            self._items("]", self._value)
            return _NOT_BOOLEAN
        if self._accept("{"):
            return self._items("}", self._pair)
        match = _SCALAR.match(self.text, self.pos)
        if match is None:
            raise self._error("expected a value")
        self.pos = match.end()
        return {"true": True, "false": False}.get(match.group(), _NOT_BOOLEAN)

    def _pair(self) -> tuple[tuple[str, ...], object]:
        key = self._key()
        self._expect("=")
        return key, self._value()

    def _items(self, close: str, item) -> list:
        """Read a comma-separated array or inline table; newlines are allowed."""
        results = []
        while True:
            self._skip(_BLANK)
            if self._accept(close):
                return results
            results.append(item())
            self._skip(_BLANK)
            if not self._accept(","):
                self._expect(close)
                return results

    def _key(self) -> tuple[str, ...]:
        parts = []
        while True:
            self._skip(_WS)
            if self.text.startswith(('"', "'"), self.pos):
                parts.append(self._string())
            else:
                match = _BARE_KEY.match(self.text, self.pos)
                if match is None:
                    raise self._error("expected a key")
                parts.append(match.group())
                self.pos = match.end()
            self._skip(_WS)
            if not self._accept("."):
                return tuple(parts)

    def _string(self) -> str:
        pattern = _BASIC_STRING if self.text[self.pos] == '"' else _LITERAL_STRING
        match = pattern.match(self.text, self.pos)
        if match is None:
            raise self._error("unterminated string")
        self.pos = match.end()
        return match.group(1)

    def _skip_multiline_string(self, quote: str) -> None:
        text, start = self.text, self.pos
        end = start
        while True:
            end = text.find(quote * 3, end)
            if end < 0:
                raise self._error("unterminated multi-line string")
            backslashes = 0
            while quote == '"' and end - backslashes > start and text[end - backslashes - 1] == "\\":
                backslashes += 1
            if backslashes % 2 == 0:
                break
            end += 1
        # A closing run of four or five quotes keeps one or two as content.
        self.pos = end + 3
        for _ in range(2):
            self._accept(quote)

    def _end_line(self) -> None:
        match = _END_OF_LINE.match(self.text, self.pos)
        if match is None:
            raise self._error("expected end of line")
        self.pos = match.end()

    def _accept(self, token: str) -> bool:
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._accept(token):
            raise self._error(f"expected {token!r}")

    def _skip(self, pattern: re.Pattern) -> None:
        self.pos = pattern.match(self.text, self.pos).end()

    def _error(self, message: str) -> ValueError:
        line = self.text.count("\n", 0, self.pos) + 1
        return ValueError(f"{message} (line {line})")


def _read_hooks_feature_flags(text: str) -> dict[str, object]:
    """Return hooks flags set in the top-level [features] table, as found."""
    if _tomllib is None:
        return _FeatureFlagReader(text).read()
    features = _tomllib.loads(text).get("features")
    if not isinstance(features, dict):
        return {}
    return {key: features[key] for key in _HOOKS_FEATURE_KEYS if key in features}


def check_codex_hooks_feature(config_path: Path) -> tuple[list[str], str]:
    """Check that a Codex config.toml does not disable hooks, without writes.

    Returns (problems, summary). Codex enables hooks by default, so a missing
    file or unset key passes. Only the top-level [features] table is read;
    profiles, -c overrides, and project, system, or managed layers are not.
    """
    try:
        text = config_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return [], "not found; hooks enabled by default"
    except (OSError, UnicodeError) as exc:
        return [f"Cannot read Codex config {config_path}: {exc}"], ""
    try:
        flags = _read_hooks_feature_flags(text)
    except ValueError as exc:  # Includes tomllib.TOMLDecodeError.
        return [
            f"Cannot parse Codex config {config_path} as TOML: {exc}. "
            "Codex rejects invalid config files; fix the syntax and run this check again."
        ], ""

    invalid = [key for key in _HOOKS_FEATURE_KEYS if key in flags and not isinstance(flags[key], bool)]
    if invalid:
        return [
            f"Invalid Codex config {config_path}: [features].{key} must be true or false."
            for key in invalid
        ], ""
    if not flags.get("hooks", flags.get("codex_hooks", True)):
        disabled = [key for key in _HOOKS_FEATURE_KEYS if flags.get(key) is False]
        found = " and ".join(f"[features].{key} = false" for key in disabled)
        if "codex_hooks" in disabled:
            found += " (codex_hooks is the legacy name of hooks)"
        settings = "that setting" if len(disabled) == 1 else "both settings"
        return [
            f"Codex hooks are disabled in {config_path}: {found}. "
            f"Remove {settings} or set [features].hooks = true, then restart Codex."
        ], ""
    for key in _HOOKS_FEATURE_KEYS:
        if key in flags:
            return [], f"[features].{key} = true"
    return [], "hooks not set; enabled by default"


def _command_and_mode(command: str) -> tuple[str, str]:
    """Normalize installed redaction flags without hiding the event action."""
    for prefix, mode in _REDACTION_PREFIXES.items():
        if command.startswith(prefix):
            return command[len(prefix):], mode
    return command, "default"


def _matcher_matches(event: str, actual: object, expected: object) -> bool:
    if event in _IGNORED_MATCHER_EVENTS or actual in _MATCH_ALL:
        return True
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    # Codex SessionStart accepts additional sources beyond those we require.
    return set(expected.split("|")) <= set(actual.split("|"))


def _expected_commands(expected: dict) -> dict[str, dict[str, dict]]:
    """Map each expected event to its AgentTrace commands, without redaction flags."""
    # Verification flags must not change the expected mode: installation with
    # --redact or --no-redact must verify with default CLI arguments.
    by_event: dict[str, dict[str, dict]] = {}
    for event, groups in expected["hooks"].items():
        commands = {}
        for group in groups:
            for handler in group["hooks"]:
                command, _ = _command_and_mode(handler["command"])
                commands[command] = {"type": handler["type"], "matcher": group.get("matcher")}
        by_event[event] = commands
    return by_event


def _agenttrace_handlers(groups: list) -> Iterator[tuple[object, dict, str]]:
    """Yield (matcher, handler, command) for each AgentTrace handler of one event."""
    for group in groups:
        if not isinstance(group, dict):
            continue
        entries = group.get("hooks")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            command = entry.get("command")
            if not isinstance(command, str):
                continue
            # Other commands are unrelated user hooks: allowed without modification.
            if _AGENTTRACE_COMMAND in _command_and_mode(command)[0]:
                yield group.get("matcher"), entry, command


def _describe_matcher(matcher: object) -> str:
    if matcher in (None, ""):
        return 'a match-all matcher ("", "*" or omitted)'
    return repr(matcher)


def _check_agenttrace_hooks(
    hooks: dict,
    expected: dict,
    provider: str,
    handler_problem: Callable[[str, dict], str | None] | None = None,
) -> tuple[list[str], set[str]]:
    """Check that each expected AgentTrace hook is installed once with the right fields.

    Returns (problems, redaction modes of the valid hooks). handler_problem
    reports provider-specific handler fields that stop a hook from running.
    """
    expected_by_event = _expected_commands(expected)
    all_expected_commands = {
        command for commands in expected_by_event.values() for command in commands
    }
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    modes: set[str] = set()
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            if event in expected_by_event:
                problems.append(f"{event}: hooks must be a list.")
            continue
        for matcher, entry, command in _agenttrace_handlers(groups):
            canonical, mode = _command_and_mode(command)
            key = (event, canonical)
            if key in seen:
                problems.append(f"{event}: duplicate AgentTrace hook command {command!r}.")
                continue
            seen.add(key)
            required = expected_by_event.get(event, {}).get(canonical)
            if required is None:
                if canonical in all_expected_commands:
                    problems.append(
                        f"{event}: AgentTrace hook {command!r} is assigned to the wrong event."
                    )
                else:
                    problems.append(
                        f"{event}: stale or unexpected AgentTrace hook command {command!r}."
                    )
                continue

            valid = True
            if entry.get("type") != required["type"]:
                problems.append(
                    f"{event}: {command!r} has handler type {entry.get('type')!r}, "
                    f"expected {required['type']!r}."
                )
                valid = False
            if not _matcher_matches(event, matcher, required["matcher"]):
                problems.append(
                    f"{event}: matcher is {matcher!r}, "
                    f"expected {_describe_matcher(required['matcher'])} for {command!r}."
                )
                valid = False
            problem = handler_problem(event, entry) if handler_problem else None
            if problem:
                problems.append(f"{event}: {command!r} {problem}")
                valid = False
            if valid:
                modes.add(mode)

    for event, commands in expected_by_event.items():
        for command in commands:
            if (event, command) not in seen:
                problems.append(
                    f"{event}: missing {json.dumps({'type': commands[command]['type'], 'command': command})}."
                )

    if len(modes) > 1:
        problems.append(
            f"{provider} AgentTrace hooks use inconsistent redaction modes: "
            + ", ".join(sorted(modes))
            + ". Install one mode consistently."
        )
    return problems, modes


def _check_executable(version: str, provider: str) -> tuple[list[str], str | None]:
    """Check that the agent-strace on PATH reports this version."""
    executable = shutil.which("agent-strace")
    if executable is None:
        return [
            "The agent-strace executable is not on PATH. Install this "
            f"AgentTrace release into the same environment used by {provider}."
        ], None
    try:
        probe = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"Cannot run {executable} --version: {exc}"], executable
    expected_version = f"agent-strace {version}"
    if probe.returncode != 0 or probe.stdout.strip() != expected_version:
        return [
            f"AgentTrace CLI version mismatch: expected {expected_version!r}; "
            f"got {probe.stdout.strip()!r} (exit {probe.returncode})."
        ], executable
    return [], executable


def verify_codex_setup(
    hooks_path: Path,
    expected: dict,
    version: str,
    *,
    config_path: Path | None = None,
    details: dict | None = None,
) -> tuple[bool, list[str]]:
    """Verify the correct event, command, and type exactly once, without writes.

    Only installed AgentTrace entries are inspected; arbitrary user hooks and
    documented Codex fields on handlers remain valid. When config_path is
    given, its [features] table must not disable Codex hooks.
    """
    problems: list[str] = []
    feature_summary = None
    if config_path is not None:
        problems, feature_summary = check_codex_hooks_feature(config_path)
    if not hooks_path.is_file():
        problems.append(
            f"Missing Codex hooks config: {hooks_path}. "
            "Run agent-strace setup --cli codex to install it."
        )
        return False, problems

    try:
        installed = json.loads(hooks_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        problems.append(f"Cannot read valid Codex hooks JSON at {hooks_path}: {exc}")
        return False, problems

    hooks = installed.get("hooks") if isinstance(installed, dict) else None
    if not isinstance(hooks, dict):
        problems.append(f"Invalid Codex hooks structure in {hooks_path}: missing hooks object.")
        return False, problems

    hook_problems, modes = _check_agenttrace_hooks(hooks, expected, "Codex")
    executable_problems, executable = _check_executable(version, "Codex")
    problems += hook_problems + executable_problems

    if not problems and details is not None:
        details.update(
            event_count=len(expected["hooks"]),
            event_total=len(expected["hooks"]),
            redaction=next(iter(modes), "default"),
            executable=executable,
            version=f"agent-strace {version}",
        )
        if feature_summary is not None:
            details["hooks_feature"] = feature_summary
    return not problems, problems


def _claude_handler_problem(event: str, entry: dict) -> str | None:
    """Report Claude Code handler fields that stop an AgentTrace hook from running."""
    if "if" in entry:
        if event in _CLAUDE_TOOL_EVENTS:
            return 'has an "if" condition, so Claude Code runs it only for matching tool calls. Remove "if".'
        return (
            'has an "if" condition, which Claude Code evaluates only on tool events, '
            'so this hook never runs. Remove "if".'
        )
    if "args" in entry:
        return (
            'has "args", so Claude Code starts the whole command as one executable name, '
            'which fails. Remove "args".'
        )
    return None


def _read_claude_settings(path: Path, scope: str) -> tuple[dict | None, str | None]:
    """Return (settings, problem) for one settings file; settings is None if absent.

    A whitespace-only file reads as {}. For a managed file the OS denies,
    PermissionError propagates, since Claude Code then starts without it.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return None, None
    except (OSError, UnicodeError) as exc:
        if scope == "managed" and isinstance(exc, PermissionError):
            raise
        return None, f"Cannot read Claude Code {scope} settings {path}: {exc}"
    try:
        settings = json.loads(text) if text.strip() else {}
    except ValueError as exc:
        effect = (
            "Claude Code refuses to start until it is fixed."
            if scope == "managed"
            else "Claude Code reports a settings error and does not apply the file until it is fixed."
        )
        return None, f"Cannot read valid Claude Code {scope} settings JSON at {path}: {exc}. {effect}"
    if not isinstance(settings, dict):
        return None, f"Invalid Claude Code {scope} settings {path}: the top level must be a JSON object."
    if "hooks" in settings and not isinstance(settings["hooks"], dict):
        return None, f"Invalid Claude Code {scope} settings {path}: hooks must be an object."
    return settings, None


def _read_claude_managed(managed_dir: Path) -> tuple[list[tuple[Path, dict]], int, list[str]]:
    """Read managed-settings.json, then managed-settings.d/*.json in name order.

    Returns (files read, count of files the OS denied, problems).
    """
    paths = [managed_dir / "managed-settings.json"]
    drop_ins = managed_dir / "managed-settings.d"
    denied = 0
    problems: list[str] = []
    try:
        paths += sorted(
            (p for p in drop_ins.iterdir() if p.suffix == ".json" and not p.name.startswith(".")),
            key=lambda p: p.name,
        )
    except (FileNotFoundError, NotADirectoryError):
        pass
    except PermissionError:
        denied += 1
    except OSError as exc:
        problems.append(f"Cannot read Claude Code managed settings directory {drop_ins}: {exc}")
    read: list[tuple[Path, dict]] = []
    for path in paths:
        try:
            settings, problem = _read_claude_settings(path, "managed")
        except PermissionError:
            denied += 1
            continue
        if problem:
            problems.append(problem)
        elif settings is not None:
            read.append((path, settings))
    return read, denied, problems


def _managed_bool(value: object) -> bool | None:
    """Read a managed Boolean lock as Claude Code does; None when invalid."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in ("true", "false"):
        return value == "true"
    return None


def _agenttrace_handlers_by_event(hooks: dict) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for event, groups in hooks.items():
        if isinstance(groups, list):
            handlers = [entry for _, entry, _ in _agenttrace_handlers(groups)]
            if handlers:
                found[event] = handlers
    return found


def _claude_extra_hooks(
    path: Path, scope: str, settings: dict, user_handlers: dict[str, list[dict]]
) -> tuple[int, str | None]:
    """Return (AgentTrace handler count, problem) for a file other than user settings.

    Claude Code runs a handler defined in several settings files once, so
    only exact copies of the user settings handlers are safe.
    """
    handlers = _agenttrace_handlers_by_event(settings.get("hooks", {}))
    differing = sorted(
        event
        for event, entries in handlers.items()
        if any(entry not in user_handlers.get(event, []) for entry in entries)
    )
    count = sum(len(entries) for entries in handlers.values())
    if not differing:
        return count, None
    fix = (
        "Ask your administrator to remove them, or make your user settings hooks identical."
        if scope == "managed"
        else "Remove them, or make them identical to the user settings hooks."
    )
    return count, (
        f"Claude Code {scope} settings {path} have AgentTrace hooks for "
        f"{', '.join(differing)} that are not exact copies of the user settings hooks. "
        "Claude Code merges hooks from all settings files and runs only identical "
        f"handlers once, so these can record events twice or with another redaction mode. {fix}"
    )


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:
        return False


def verify_claude_setup(
    settings_path: Path,
    expected: dict,
    version: str,
    *,
    project_dir: Path | None = None,
    managed_dir: Path | None = None,
    details: dict | None = None,
) -> tuple[bool, list[str]]:
    """Verify the AgentTrace hooks in Claude Code user settings, without writes.

    settings_path is the user settings file that setup targets, and each
    expected hook must appear there once. Project and local settings in
    project_dir and the managed settings files in managed_dir are read too,
    because they can turn hooks off or add AgentTrace hooks that also run.
    Unrelated user hooks are allowed.
    """
    problems: list[str] = []
    modes: set[str] = set()
    user_handlers: dict[str, list[dict]] = {}
    user, problem = _read_claude_settings(settings_path, "user")
    if problem:
        problems.append(problem)
    elif user is None:
        problems.append(f"Missing Claude Code user settings: {settings_path}. {_CLAUDE_SETUP_HINT}")
    else:
        user_handlers = _agenttrace_handlers_by_event(user.get("hooks", {}))
        if not user_handlers:
            problems.append(
                f"No AgentTrace hooks in Claude Code user settings {settings_path}. {_CLAUDE_SETUP_HINT}"
            )
        else:
            hook_problems, modes = _check_agenttrace_hooks(
                user["hooks"], expected, "Claude Code", _claude_handler_problem
            )
            problems += hook_problems

    # Settings files that can set disableAllHooks, lowest precedence first.
    files: list[tuple[str, Path, dict]] = [("user", settings_path, user)] if user is not None else []
    layers: list[tuple[str, Path, list[str]]] = []
    if project_dir is not None:
        for scope, name in (("project", "settings.json"), ("local", "settings.local.json")):
            path = project_dir / ".claude" / name
            notes: list[str] = []
            layers.append((scope, path, notes))
            if _same_file(path, settings_path):
                notes.append("same file as user settings")
                continue
            settings, problem = _read_claude_settings(path, scope)
            if problem:
                problems.append(problem)
                continue
            if settings is None:
                notes.append("not found")
                continue
            files.append((scope, path, settings))
            notes.append("found")
            count, problem = _claude_extra_hooks(path, scope, settings, user_handlers)
            if problem:
                problems.append(problem)
            elif count:
                notes.append(f"{count} AgentTrace hooks identical to user settings")

    managed: dict[str, tuple[object, Path]] = {}
    managed_notes: list[str] = []
    if managed_dir is not None:
        read, denied, managed_problems = _read_claude_managed(managed_dir)
        problems += managed_problems
        layers.append(("managed", managed_dir, managed_notes))
        if read:
            managed_notes.append(f"{len(read)} file{'s' if len(read) != 1 else ''} read")
        if denied:
            managed_notes.append(
                f"{denied} not readable, so Claude Code skips {'it' if denied == 1 else 'them'}"
            )
        if not read and not denied:
            managed_notes.append("no managed settings files")
        copies = 0
        for path, settings in read:
            count, problem = _claude_extra_hooks(path, "managed", settings, user_handlers)
            copies += count
            if problem:
                problems.append(problem)
            # A later file replaces a single value. A null leaves an earlier value in place.
            for key in ("allowManagedHooksOnly", "disableAllHooks"):
                if settings.get(key) is not None:
                    managed[key] = (settings[key], path)
        if copies:
            managed_notes.append(f"{copies} AgentTrace hooks identical to user settings")

    if "allowManagedHooksOnly" in managed:
        value, source = managed["allowManagedHooksOnly"]
        flag = _managed_bool(value)
        if flag is False:
            managed_notes.append("allowManagedHooksOnly = false")
        else:
            invalid = "" if flag else ", which Claude Code reads as true because it is invalid"
            problems.append(
                f"Claude Code managed settings {source} set allowManagedHooksOnly to "
                f"{json.dumps(value)}{invalid}. Only managed hooks run, so the AgentTrace hooks "
                f"in {settings_path} are blocked. Ask your administrator to remove that setting."
            )

    for scope, path, settings in files:
        if "disableAllHooks" in settings and not isinstance(settings["disableAllHooks"], bool):
            problems.append(
                f"Invalid Claude Code {scope} settings {path}: disableAllHooks must be true or false."
            )
    winner = None
    if "disableAllHooks" in managed:
        value, source = managed["disableAllHooks"]
        if not isinstance(value, bool):
            managed_notes.append(
                f"disableAllHooks = {json.dumps(value)}, ignored because it is not true or false"
            )
        else:
            # Managed settings take precedence over every other settings file.
            winner = "managed"
            if value:
                problems.append(
                    f"Claude Code hooks are disabled by managed settings {source}: "
                    "disableAllHooks = true. Ask your administrator to remove that setting."
                )
            else:
                managed_notes.append("disableAllHooks = false")
    if winner is None:
        setters = [
            (scope, path, settings["disableAllHooks"])
            for scope, path, settings in files
            if isinstance(settings.get("disableAllHooks"), bool)
        ]
        if setters:
            winner, path, value = setters[-1]
            if value:
                fix = "Remove it or set it to false."
                if winner == "project":
                    local = project_dir / ".claude" / "settings.local.json"
                    fix = f"Remove it, or set disableAllHooks to false in {local} for this project only."
                problems.append(
                    f"Claude Code hooks are disabled in {path}: disableAllHooks = true. {fix}"
                )
            elif user is not None and user.get("disableAllHooks") is True:
                problems.append(
                    f"Claude Code hooks are disabled in {settings_path}: disableAllHooks = true. "
                    f"{winner.capitalize()} settings {path} turn them back on only in this project, "
                    "so other projects run no hooks. Remove it or set it to false."
                )
    for scope, path, notes in layers:
        settings = next((s for sc, _, s in files if sc == scope), None)
        if settings is not None and isinstance(settings.get("disableAllHooks"), bool):
            value = settings["disableAllHooks"]
            note = f"disableAllHooks = {json.dumps(value)}"
            if value and winner != scope:
                note += f", overridden by {winner} settings"
            notes.append(note)

    executable_problems, executable = _check_executable(version, "Claude Code")
    problems += executable_problems

    if not problems and details is not None:
        details.update(
            event_count=len(expected["hooks"]),
            event_total=len(expected["hooks"]),
            redaction=next(iter(modes), "default"),
            executable=executable,
            version=f"agent-strace {version}",
            layers=[(scope, path, "; ".join(notes)) for scope, path, notes in layers],
        )
    return not problems, problems
