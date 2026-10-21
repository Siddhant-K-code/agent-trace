"""Read-only verification of installed Codex hooks (not runtime capture)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
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
_IGNORED_MATCHER_EVENTS = ("Stop", "UserPromptSubmit")
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

    # Verification flags must not change the expected mode: installation with
    # --redact or --no-redact must verify with default CLI arguments.
    expected_by_event: dict[str, dict[str, dict]] = {}
    all_expected_commands: set[str] = set()
    for event, groups in expected["hooks"].items():
        commands = {}
        for group in groups:
            for handler in group["hooks"]:
                command, _ = _command_and_mode(handler["command"])
                commands[command] = {"type": handler["type"], "matcher": group.get("matcher")}
                all_expected_commands.add(command)
        expected_by_event[event] = commands

    seen: set[tuple[str, str]] = set()
    modes: set[str] = set()
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            if event in expected_by_event:
                problems.append(f"{event}: hooks must be a list.")
            continue
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
                canonical, mode = _command_and_mode(command)
                if "agent-strace hook " not in canonical:
                    continue  # Unrelated user hook: allowed without modification.
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
                if not _matcher_matches(event, group.get("matcher"), required["matcher"]):
                    problems.append(
                        f"{event}: matcher is {group.get('matcher')!r}, "
                        f"expected {required['matcher']!r} for {command!r}."
                    )
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
            "Codex AgentTrace hooks use inconsistent redaction modes: "
            + ", ".join(sorted(modes))
            + ". Install one mode consistently."
        )

    executable = shutil.which("agent-strace")
    if executable is None:
        problems.append(
            "The agent-strace executable is not on PATH. Install this "
            "AgentTrace release into the same environment used by Codex."
        )
    else:
        try:
            probe = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"Cannot run {executable} --version: {exc}")
        else:
            expected_version = f"agent-strace {version}"
            if probe.returncode != 0 or probe.stdout.strip() != expected_version:
                problems.append(
                    f"AgentTrace CLI version mismatch: expected {expected_version!r}; "
                    f"got {probe.stdout.strip()!r} (exit {probe.returncode})."
                )

    if not problems and details is not None:
        details.update(
            event_count=len(expected_by_event),
            event_total=len(expected_by_event),
            redaction=next(iter(modes), "default"),
            executable=executable,
            version=f"agent-strace {version}",
        )
        if feature_summary is not None:
            details["hooks_feature"] = feature_summary
    return not problems, problems
