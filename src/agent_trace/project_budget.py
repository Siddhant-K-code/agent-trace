"""Per-project rolling budget guardrails.

Reads the budget keys (weekly, warn_at, stop_at and per_session_max, in a
``budget`` mapping or at the top level) from the first of
``.agent-strace.yaml``, ``.agent-strace.yml`` and ``.agent-strace.json`` in
the working directory, and evaluates local session spend over a rolling
seven-day window. A file that sets no budget keys sets no budget. A file
that cannot be read, or whose budget keys cannot be used, raises
ProjectBudgetConfigError, so a budget the user wrote is never dropped. The
module is intentionally stdlib-only so budget enforcement stays available in
the core package.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from .cost import estimate_cost
from .models import usable_time
from .store import TraceStore


DEFAULT_CONFIG_PATHS = (
    ".agent-strace.yaml",
    ".agent-strace.yml",
    ".agent-strace.json",
)
ROLLING_WINDOW_SECONDS = 7 * 24 * 60 * 60
BUDGET_KEYS = ("weekly", "warn_at", "stop_at", "per_session_max")
# Dollar amounts. Like watch --rules budget:$5, they may start with one $.
_DOLLAR_KEYS = ("weekly", "per_session_max")
# warn_at and stop_at are fractions of weekly. A value above 1, such as 1.2,
# may be deliberate; a value above this is almost surely a percent, such as
# 90, which would never fire.
_MAX_FRACTION = 10


@dataclass(frozen=True)
class ProjectBudgetConfig:
    """Config for a per-project rolling cost budget."""

    weekly: float = 0.0
    warn_at: float = 0.80
    stop_at: float = 1.00
    per_session_max: float | None = None

    @property
    def enabled(self) -> bool:
        return self.weekly > 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectBudgetConfig":
        """Return the budget that *data*, the top level of a config file, sets.

        Raises ValueError when its budget keys cannot be used; see
        project_budget_keys.
        """
        return cls.from_keys(project_budget_keys(data))

    @classmethod
    def from_keys(cls, keys: dict[str, float | None]) -> "ProjectBudgetConfig":
        """Return the budget for keys that project_budget_keys returned.

        A key that is null keeps its default. per_session_max 0 means no
        per-session limit.
        """
        values = {key: value for key, value in keys.items() if value is not None}
        if values.get("per_session_max") == 0:
            del values["per_session_max"]
        return cls(**values)


class ProjectBudgetConfigError(ValueError):
    """A project budget config file that cannot be read or used.

    The message names the file and what is wrong with it, and for a key its
    name. watch, record and record-http print it and refuse to start rather
    than run without the budget.
    """


class _Problem(ValueError):
    """What is wrong with a budget config, before the file is named."""


class _Mapping(dict):
    """A parsed mapping that remembers the keys it was given more than once."""

    def __init__(self) -> None:
        super().__init__()
        self.duplicate_keys: list[Any] = []


def mapping_from_pairs(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
    """Build a JSON object; use it as the object_pairs_hook of json.loads.

    json.loads keeps only the last value of a key given twice. The mapping
    this returns remembers such keys, so project_budget_keys can refuse a
    budget key given twice.
    """
    mapping = _Mapping()
    for key, value in pairs:
        if key in mapping and key not in mapping.duplicate_keys:
            mapping.duplicate_keys.append(key)
        mapping[key] = value
    return mapping


def project_budget_keys(
    data: dict[str, Any], *, show_values: bool = True, mapping: str = "mapping",
) -> dict[str, float | None]:
    """Return the budget keys that *data*, the top level of a config file, sets.

    The keys are read from the budget mapping, or from the top level when
    there is no budget key; a file with both is refused. Inside budget only
    the four budget keys are allowed. Other top-level keys are ignored. Each
    key that is present maps to a finite float, or to None when it is null.

    Raises ValueError for the first problem, naming the key. The message
    shows the value only when *show_values* is true. *mapping* is what the
    message calls a mapping, such as "JSON object".
    """
    def not_value(value: Any) -> str:
        return f", not {_shown(value, mapping)}" if show_values else ""

    duplicates = getattr(data, "duplicate_keys", ())
    top_keys = [key for key in BUDGET_KEYS if key in data]
    if "budget" not in data:
        for key in top_keys:
            if key in duplicates:
                raise _Problem(f"{key} is given twice")
        return {key: _checked(key, data[key], key, not_value) for key in top_keys}

    if "budget" in duplicates:
        raise _Problem("budget is given twice")
    block = data["budget"]
    if block is not None and not isinstance(block, dict):
        raise _Problem(f"budget must be a {mapping}{not_value(block)}")
    if top_keys:
        raise _Problem(
            f"it has both budget and a top-level {top_keys[0]}; "
            f"move {top_keys[0]} under budget"
        )
    if block is None:
        return {}
    for key in block:
        if key not in BUDGET_KEYS:
            raise _Problem(_unknown_key(key))
    for key in getattr(block, "duplicate_keys", ()):
        raise _Problem(f"budget.{key} is given twice")
    return {
        key: _checked(key, block[key], f"budget.{key}", not_value)
        for key in BUDGET_KEYS if key in block
    }


def _unknown_key(key: Any) -> str:
    shown = repr(key)
    if len(shown) > 42:
        shown = "that is too long to show"
    return f"budget has an unknown key {shown}; use weekly, warn_at, stop_at or per_session_max"


def _checked(key: str, value: Any, name: str, not_value: Callable[[Any], str]) -> float | None:
    """Return the value of budget key *key*, called *name* in a refusal."""
    if value is None:
        return None
    number = _number(value, dollars=key in _DOLLAR_KEYS)
    if key in _DOLLAR_KEYS:
        if number is None or number < 0:
            raise _Problem(f"{name} must be a finite number, 0 or more{not_value(value)}")
    elif number is None or not 0 <= number <= _MAX_FRACTION:
        raise _Problem(
            f"{name} must be a number from 0 to {_MAX_FRACTION}, a fraction of weekly "
            f"such as 0.9 for 90%{not_value(value)}"
        )
    # Adding 0.0 turns -0.0 into 0.0.
    return number + 0.0


def _number(value: Any, *, dollars: bool) -> float | None:
    """Return *value* as a finite float; None when it is not one.

    Numbers and numeric strings are accepted, and a dollar amount may start
    with one $. true and false are not numbers.
    """
    if isinstance(value, str):
        text = value.strip()
        if dollars and text.startswith("$"):
            text = text[1:]
        try:
            number = float(text)
        except ValueError:
            return None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _shown(value: Any, mapping: str = "mapping") -> str:
    """Return *value* as a refusal shows it: short and on one line."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        try:
            text = repr(value)
        except ValueError:  # an int with more digits than str() allows
            text = ""
        return text if 0 < len(text) <= 40 else "a number too long to show"
    if isinstance(value, str):
        return repr(value) if len(value) <= 40 else f"a string of {len(value)} characters"
    if isinstance(value, dict):
        return f"a {mapping}"
    if isinstance(value, list):
        return "a list"
    return f"a {type(value).__name__}"


@dataclass(frozen=True)
class ProjectBudgetStatus:
    """Current spend against the configured rolling budget."""

    config: ProjectBudgetConfig
    spent: float
    window_start: float
    window_end: float

    @property
    def weekly(self) -> float:
        return self.config.weekly

    @property
    def warn_amount(self) -> float:
        return self.weekly * self.config.warn_at

    @property
    def stop_amount(self) -> float:
        return self.weekly * self.config.stop_at

    @property
    def percent(self) -> float:
        if self.weekly <= 0:
            return 0.0
        return self.spent / self.weekly

    @property
    def should_warn(self) -> bool:
        return self.config.enabled and self.spent >= self.warn_amount

    @property
    def should_stop(self) -> bool:
        return self.config.enabled and self.spent >= self.stop_amount


def load_project_budget_config(
    config_path: str | Path | None = None,
) -> ProjectBudgetConfig:
    """Load the project budget config from JSON or a small YAML subset.

    If ``config_path`` is omitted, the standard project config filenames are
    checked in order, and only the first that exists is read. When no file
    exists, or the file sets no budget keys, the budget is off. Raises
    ProjectBudgetConfigError when the file cannot be read or parsed, or a
    budget key holds a value that cannot be used; it never falls back to a
    disabled budget then.
    """

    for path in _candidate_paths(config_path):
        # A dangling symlink is read, and refused, rather than skipped.
        if path.exists() or path.is_symlink():
            return ProjectBudgetConfig.from_keys(_read_budget_keys(path))
    return ProjectBudgetConfig()


def rolling_project_spend(
    store: TraceStore,
    now: float | None = None,
    window_seconds: float = ROLLING_WINDOW_SECONDS,
    exclude_session_id: str = "",
) -> float:
    """Estimate spend for sessions started in the rolling project window.

    A session whose start time is not usable is in no window.
    """

    window_end = time.time() if now is None else now
    window_start = window_end - window_seconds
    total = 0.0

    for meta in store.list_sessions():
        if meta.session_id == exclude_session_id:
            continue
        started = usable_time(meta.started_at)
        if started is None or not (window_start <= started < window_end):
            continue
        try:
            total += estimate_cost(store, meta.session_id).total_cost
        except Exception:
            continue
    return total


def project_budget_status(
    store: TraceStore,
    config: ProjectBudgetConfig,
    now: float | None = None,
    current_session_spend: float = 0.0,
    exclude_session_id: str = "",
) -> ProjectBudgetStatus:
    """Return current rolling budget status."""

    window_end = time.time() if now is None else now
    window_start = window_end - ROLLING_WINDOW_SECONDS
    spent = rolling_project_spend(
        store,
        now=window_end,
        exclude_session_id=exclude_session_id,
    ) + max(0.0, current_session_spend)
    return ProjectBudgetStatus(
        config=config,
        spent=spent,
        window_start=window_start,
        window_end=window_end,
    )


def format_project_budget_status(status: ProjectBudgetStatus) -> str:
    """Format a concise status line for terminal/watch alerts."""

    pct = status.percent * 100
    return (
        f"ProjectBudget: ${status.spent:.2f} spent in rolling 7 days "
        f"({pct:.0f}% of ${status.weekly:.2f})"
    )


def warn_if_budget_near_limit(
    store: TraceStore,
    config: ProjectBudgetConfig,
    out: TextIO,
    now: float | None = None,
) -> ProjectBudgetStatus:
    """Print a warning when the rolling budget is at or above warn_at."""

    status = project_budget_status(store, config, now=now)
    if status.should_warn:
        out.write(f"{format_project_budget_status(status)}\n")
        out.flush()
    return status


def enforce_new_session_budget(
    store: TraceStore,
    config: ProjectBudgetConfig,
    out: TextIO,
    now: float | None = None,
) -> bool:
    """Return False and print an error when a new session should be refused."""

    status = project_budget_status(store, config, now=now)
    if not status.should_stop:
        return True

    out.write(
        f"agent-strace: weekly project budget exceeded; refusing new session "
        f"(${status.spent:.2f} spent, stop threshold ${status.stop_amount:.2f})\n"
    )
    out.flush()
    return False


def _candidate_paths(config_path: str | Path | None) -> Iterable[Path]:
    if config_path:
        yield Path(config_path)
        return
    for path in DEFAULT_CONFIG_PATHS:
        yield Path(path)


def _config_error(path: Path, problem: str) -> ProjectBudgetConfigError:
    shown = str(path)
    if not (shown and shown.isprintable()):
        shown = repr(shown)
    return ProjectBudgetConfigError(f"invalid project budget in {shown}: {problem}")


def _read_budget_keys(path: Path) -> dict[str, float | None]:
    """Return the checked budget keys of the config file at *path*."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        problem = f"cannot read it ({exc.strerror})" if exc.strerror else "cannot read it"
        raise _config_error(path, problem) from None
    try:
        # Not UTF-8 is refused even with no budget keys in sight: a UTF-16
        # file, as Windows PowerShell writes, would hide them.
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise _config_error(path, "it is not valid UTF-8") from None
    # YAML and JSON readers may skip a byte order mark, and editors such as
    # Notepad write one.
    if text.startswith("\ufeff"):
        text = text[1:]
    try:
        return _budget_keys_in(text, is_json=path.suffix.lower() == ".json")
    except _Problem as exc:
        raise _config_error(path, str(exc)) from None


def _budget_keys_in(text: str, *, is_json: bool) -> dict[str, float | None]:
    """Return the checked budget keys of the text of a config file.

    A .json file must hold a JSON object. Any other file is read as JSON when
    it starts with { or [, and as YAML otherwise. An empty file sets no keys.
    """
    if not text.strip():
        return {}
    if not is_json and text.lstrip()[:1] not in ("{", "["):
        return project_budget_keys(_parse_budget_yaml(text))
    try:
        data = _parse_json(text)
        if not isinstance(data, dict):
            raise _Problem("the top level must be a JSON object")
    except _Problem:
        # A YAML file that only looks like JSON is refused when it might
        # hold a budget, and ignored otherwise.
        if is_json or _BUDGET_WORD.search(text):
            raise
        return {}
    return project_budget_keys(data, mapping="JSON object")


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=mapping_from_pairs)
    except json.JSONDecodeError as exc:
        raise _Problem(
            f"it is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
        ) from None
    except RecursionError:
        raise _Problem("it is nested too deeply") from None
    except ValueError:
        # json.loads raises a plain ValueError for an integer with more
        # digits than int() accepts.
        raise _Problem("it has a number with too many digits") from None


# A budget key name as a word. A line the YAML reader cannot read is refused
# when it holds one.
_BUDGET_WORD = re.compile(
    r"(?<![A-Za-z0-9_-])(?:budget|weekly|warn_at|stop_at|per_session_max)(?![A-Za-z0-9_-])"
)
# key: value, with the key plain or quoted. As before, the space after the
# colon may be left out, so weekly:20 still works.
_YAML_PAIR = re.compile(
    r"""(?P<key>[A-Za-z_][A-Za-z0-9_-]*|"(?:[^"\\]|\\.)*"|'(?:[^']|'')*')"""
    r"[ \t]*:(?P<value>.*)"
)
# Any other plain key, such as "my budget: 5", whose entry is skipped.
_YAML_OTHER_KEY = re.compile(r"""[^\s\-?:,\[\]{}#&*!|>'"%@`].*?:(?:[ \t]|$)""")
_YAML_MARKER = re.compile(r"(?:---|\.\.\.)(?:[ \t]+#.*)?")
_YAML_EMPTY_MAPPING = re.compile(r"\{[ \t]*\}(?:[ \t]+#.*)?")
_YAML_NULLS = ("~", "null", "Null", "NULL")
_YAML_BOOLS = {"true": True, "True": True, "TRUE": True, "false": False, "False": False, "FALSE": False}
_YAML_NUMBER = re.compile(r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?", re.ASCII)
_YAML_INT = re.compile(r"[-+]?[0-9]+", re.ASCII)
# YAML reads an unquoted value that starts with one of these as a flow
# collection, anchor, alias, tag, block scalar or reserved character.
_YAML_INDICATORS = "[]{},&*!|>%@`"


def _parse_budget_yaml(text: str) -> _Mapping:
    """Return the budget keys of a config file written in YAML.

    Reads the budget block, one key: value per line, and the budget keys at
    the top level. Values are read as in watch's rules files. Every other
    top-level entry is skipped with its indented lines, so the file can hold
    review, retention or apply sections. A line that cannot be read is
    refused when it names a budget key, and skipped otherwise.
    """
    top = _Mapping()
    # What the indented lines that follow belong to: "none" before the first
    # key, "skip", "unreadable", "budget" (the block) or "value".
    entry = "none"
    block: _Mapping | None = None
    block_lead = ""
    for number, raw_line in enumerate(re.split(r"\r\n|\r|\n", text), 1):
        line = raw_line.rstrip(" \t")
        body = line.lstrip(" \t")
        if not body or body.startswith("#"):
            continue
        lead = line[:len(line) - len(body)]
        if lead:
            if entry == "skip":
                continue
            if entry == "budget":
                if block is None:
                    block, block_lead = _Mapping(), lead
                    top["budget"] = block
                elif lead != block_lead:
                    raise _Problem(f"line {number}: unexpected indentation")
                _budget_block_line(block, body, number)
            elif entry == "value" or _mentions_budget(body):
                raise _Problem(f"line {number}: unexpected indentation")
            continue

        if _YAML_MARKER.fullmatch(body):
            entry = "none"
            continue
        if body == "-" or body.startswith(("- ", "-\t")):
            if entry == "skip":
                # A list may sit at the indent of its key: "steps:\n- run".
                continue
            if entry == "budget" and block is None:
                raise _Problem(f"line {number}: budget must be a mapping, not a list")
            if _mentions_budget(body):
                raise _Problem(f"line {number}: the top level must be a mapping, not a list")
            entry = "unreadable"
            continue
        pair = _YAML_PAIR.fullmatch(body)
        if pair is None:
            if _YAML_OTHER_KEY.match(body):
                entry = "skip"
            elif not _mentions_budget(body):
                entry = "unreadable"
            elif body[0] in "{[":
                raise _Problem(
                    f"line {number}: flow style is not supported; write one key per line"
                )
            else:
                raise _Problem(f"line {number}: expected key: value")
            continue

        key = _yaml_key(pair.group("key"))
        value = pair.group("value").lstrip(" \t")
        if key != "budget" and key not in BUDGET_KEYS:
            entry = "skip"
            continue
        if key in top:
            raise _Problem(f"line {number}: {key} is given twice")
        entry = "value"
        if key != "budget":
            top[key] = _yaml_value(value, number)
        elif not value or value.startswith("#"):
            top[key], block, entry = None, None, "budget"
        elif _YAML_EMPTY_MAPPING.fullmatch(value):
            top[key] = _Mapping()
        elif value[0] in "{[":
            raise _Problem(
                f"line {number}: budget uses flow style; "
                "write each key on its own line, indented under budget:"
            )
        else:
            top[key] = _yaml_value(value, number)
    return top


def _budget_block_line(block: _Mapping, body: str, number: int) -> None:
    """Add one line of the budget block to *block*."""
    if body == "-" or body.startswith(("- ", "-\t")):
        raise _Problem(f"line {number}: budget must be a mapping, not a list")
    pair = _YAML_PAIR.fullmatch(body)
    if pair is None:
        raise _Problem(f"line {number}: expected key: value")
    key = _yaml_key(pair.group("key"))
    if key not in BUDGET_KEYS:
        raise _Problem(f"line {number}: {_unknown_key(key)}")
    if key in block:
        raise _Problem(f"line {number}: budget.{key} is given twice")
    block[key] = _yaml_value(pair.group("value").lstrip(" \t"), number)


def _mentions_budget(body: str) -> bool:
    """Return True when a line, without its comment, names a budget key."""
    return bool(_BUDGET_WORD.search(re.split(r"[ \t]#", body, maxsplit=1)[0]))


def _yaml_key(text: str) -> str:
    """Return a key the YAML reader matched, without its quotes."""
    if text[0] == '"':
        return re.sub(r"\\(.)", r"\1", text[1:-1])
    if text[0] == "'":
        return text[1:-1].replace("''", "'")
    return text


def _yaml_value(text: str, line: int) -> Any:
    """Return one value of the YAML reader: None, a bool, a number or a string.

    Quoting works as in watch's rules files. An unquoted value ends before a
    comment that starts with whitespace and #.
    """
    if not text or text.startswith("#"):
        return None
    if text[0] in ("'", '"'):
        return _yaml_quoted(text, line)
    comment = re.search(r"[ \t]#", text)
    if comment:
        text = text[:comment.start()].rstrip(" \t")
    if (
        text[0] in _YAML_INDICATORS or text in ("-", "?", ":")
        or text[:2] in ("- ", "? ", ": ", "-\t", "?\t", ":\t")
    ):
        raise _Problem(f"line {line}: a value cannot start with {text[0]!r} unless it is quoted")
    if ": " in text or ":\t" in text or text.endswith(":"):
        raise _Problem(
            f"line {line}: a value cannot contain ': ' or end with ':' unless it is quoted"
        )
    if text in _YAML_NULLS:
        return None
    if text in _YAML_BOOLS:
        return _YAML_BOOLS[text]
    if _YAML_INT.fullmatch(text):
        try:
            return int(text)
        except ValueError:  # more digits than int() accepts
            return text
    if _YAML_NUMBER.fullmatch(text):
        return float(text)
    return text


def _yaml_quoted(text: str, line: int) -> str:
    """Return a quoted value without its quotes.

    In single quotes '' stands for one quote; in double quotes the only
    escapes are \\" and \\\\. Only a comment may follow the closing quote.
    """
    quote = text[0]
    chars: list[str] = []
    index = 1
    while True:
        if index >= len(text):
            raise _Problem(f"line {line}: the closing quote is missing")
        char = text[index]
        if char == quote:
            if quote == "'" and text[index + 1:index + 2] == "'":
                chars.append("'")
                index += 2
                continue
            break
        if char == "\\" and quote == '"':
            escaped = text[index + 1:index + 2]
            if escaped not in ('"', "\\"):
                raise _Problem(
                    f'line {line}: in double quotes a backslash may only escape " or \\; '
                    "use single quotes for other backslashes"
                )
            chars.append(escaped)
            index += 2
            continue
        chars.append(char)
        index += 1
    if not re.fullmatch(r"(?:[ \t]+(?:#.*)?)?", text[index + 1:], re.DOTALL):
        raise _Problem(f"line {line}: only a comment may follow the closing quote")
    return "".join(chars)
