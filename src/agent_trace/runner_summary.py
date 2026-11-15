"""Read a test run's outcome from the runner's own final summary.

A shell call's exit status belongs to the whole invocation, so in
``pytest | tail -5`` it is the exit status of ``tail``. When that status
cannot be pinned on the test runner, the runner's final summary in the linked
output can still say how the run ended.

Parsing is deterministic and strict. Only the closing summary that the runner
itself prints counts. A summary that is absent, repeated, cut off, or that
contradicts itself gives no outcome. Output that was redacted, omitted, or
truncated by the recorder is not read.

Supported runners: unittest, pytest, jest, vitest. Runners without one
run-wide summary line (``go test``, ``cargo test``) print one line per package
or binary, and a ``tail`` can drop a failing one, so they are not read.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable

from .models import EventType, TraceEvent

SUMMARY_RUNNERS = frozenset({"unittest", "pytest", "jest", "vitest"})

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Suffix the hook recorder appends when it cuts a long result.
_TRUNCATION_SUFFIX = "... (truncated)"
_TEXT_KEYS = ("result", "output", "content_preview")
_STREAM_KEYS = ("stdout", "stderr", "output")


def _unwrap_mapping(text: str) -> str | None:
    """Return the streams of a result recorded as a dict, else ``text``.

    Some hooks record a tool response object with ``str(dict)``, which escapes
    newlines. Only a complete literal is read; anything else is left as is.
    """
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text
    value: Any = None
    try:
        value = json.loads(stripped)
    except ValueError:
        try:
            value = ast.literal_eval(stripped)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return text
    if not isinstance(value, dict):
        return text
    streams = [value[key] for key in _STREAM_KEYS if isinstance(value.get(key), str)]
    return "\n".join(streams) if streams else None


def output_text(result: TraceEvent | None) -> str | None:
    """Return the readable output of one linked tool result, or None.

    None when there is no result, it is an error event, it was redacted, its
    content was omitted, or the recorder truncated it.
    """
    if result is None or result.event_type != EventType.TOOL_RESULT or result.redacted:
        return None
    data = result.data if isinstance(result.data, dict) else {}
    if data.get("content_omitted") or data.get("truncated") is True:
        return None
    text = next(
        (data[key] for key in _TEXT_KEYS if isinstance(data.get(key), str) and data[key].strip()),
        None,
    )
    if text is None:
        streams = [data[key] for key in ("stdout", "stderr") if isinstance(data.get(key), str)]
        text = "\n".join(streams) if streams else None
    if text is None or "[REDACTED" in text or text.rstrip().endswith(_TRUNCATION_SUFFIX):
        return None
    text = _unwrap_mapping(text)
    if text is None:
        return None
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n")]


# unittest -------------------------------------------------------------------

# Matched across whitespace, so a result recorded with its newlines collapsed
# reads the same as the original output.
_UNITTEST_RAN = re.compile(r"(?<!\S)Ran (\d+) tests? in \d+(?:\.\d+)?s(?!\S)")
_UNITTEST_SUMMARY = re.compile(
    r"(?<!\S)Ran (\d+) tests? in \d+(?:\.\d+)?s\s+(OK|FAILED|NO TESTS RAN)(?: \(([^()]*)\))?(?!\S)"
)
_UNITTEST_COUNT = re.compile(r"^(failures|errors|skipped|expected failures|unexpected successes)=(\d+)$")


def _unittest(text: str) -> str | None:
    if len(_UNITTEST_RAN.findall(text)) != 1:
        return None
    summary = _UNITTEST_SUMMARY.search(text)
    if summary is None:
        return None
    total, word, detail = int(summary.group(1)), summary.group(2), summary.group(3)
    counts: dict[str, int] = {}
    for part in (detail.split(", ") if detail else []):
        match = _UNITTEST_COUNT.match(part)
        if match is None or match.group(1) in counts:
            return None
        counts[match.group(1)] = int(match.group(2))
    bad = counts.get("failures", 0) + counts.get("errors", 0) + counts.get("unexpected successes", 0)
    ran_ok = total - counts.get("skipped", 0) - counts.get("expected failures", 0)
    if word == "OK" and bad == 0 and ran_ok > 0:
        return "passed"
    if word == "FAILED" and bad > 0:
        return "failed"
    return None


# pytest ---------------------------------------------------------------------

_PYTEST_DURATION = r"in \d+(?:\.\d+)?(?:s|\sseconds)(?: \(\d+:\d\d:\d\d(?:\.\d+)?\))?"
_PYTEST_LINE = re.compile(
    r"^(?:=+ )?(?P<items>\d+ [a-z]+(?:, \d+ [a-z]+)*|no tests ran) " + _PYTEST_DURATION + r"(?: =+)?$"
)
_PYTEST_WORDS = {
    "passed": "passed", "failed": "failed", "error": "error", "errors": "error",
    "skipped": "skipped", "xfailed": "xfailed", "xpassed": "xpassed",
    "warning": "warning", "warnings": "warning", "deselected": "deselected",
    "rerun": "rerun", "reruns": "rerun",
}
# A coverage gate fails the run after the summary says the tests passed.
_PYTEST_GATE = re.compile(
    r"Required test coverage of [\d.]+% not reached|Coverage failure: total of",
)


def _counts(items: list[str], words: dict[str, str]) -> dict[str, int] | None:
    counts: dict[str, int] = {}
    for item in items:
        number, _, word = item.partition(" ")
        key = words.get(word)
        if key is None or key in counts or not number.isdigit():
            return None
        counts[key] = int(number)
    return counts


def _pytest(text: str) -> str | None:
    lines = _lines(text)
    matches = [match for match in map(_PYTEST_LINE.match, lines) if match]
    if len(matches) != 1 or matches[0].group("items") == "no tests ran":
        return None
    counts = _counts(matches[0].group("items").split(", "), _PYTEST_WORDS)
    if counts is None:
        return None
    if counts.get("failed", 0) or counts.get("error", 0):
        return "failed"
    if counts.get("passed", 0) and not any(_PYTEST_GATE.search(line) for line in lines):
        return "passed"
    return None


# jest and vitest ------------------------------------------------------------

# Jest: "Tests:       1 failed, 2 passed, 3 total". When suites were skipped
# the suites line reads "1 passed, 1 of 2 total": the first number counts only
# failed and passed suites.
_JEST_LINE = re.compile(
    r"^(?P<name>Test Suites|Tests):\s+(?P<items>(?:\d+ [a-z]+, )*)"
    r"(?P<count>\d+)(?: of (?P<total>\d+))? total$"
)
_JEST_WORDS = {word: word for word in ("failed", "passed", "skipped", "todo")}
_JEST_GATE = re.compile(r"coverage threshold for .* not met")
# Vitest: "Tests  1 failed | 2 passed (3)".
_VITEST_LINE = re.compile(
    r"^(?P<name>Test Files|Tests)\s+(?P<items>\d+ [a-z]+(?: \| \d+ [a-z]+)*) \((?P<count>\d+)\)$"
)
_VITEST_WORDS = {word: word for word in ("failed", "passed", "skipped", "todo")}
_VITEST_ERRORS = re.compile(r"^Errors\s+\d+ errors?$")
_VITEST_GATE = re.compile(r"Coverage for \w+ \([\d.]+%\) does not meet (global )?threshold", re.IGNORECASE)


def _jest_adds_up(counts: dict[str, int], match: re.Match[str]) -> bool:
    count = int(match.group("count"))
    if match.group("total") is None:
        return sum(counts.values()) == count
    ran = counts.get("failed", 0) + counts.get("passed", 0)
    return ran == count and sum(counts.values()) <= int(match.group("total"))


def _vitest_adds_up(counts: dict[str, int], match: re.Match[str]) -> bool:
    return sum(counts.values()) == int(match.group("count"))


def _suite_lines(
    lines: list[str], pattern: re.Pattern[str], split: Callable[[str], list[str]],
    words: dict[str, str], names: tuple[str, str],
    adds_up: Callable[[dict[str, int], re.Match[str]], bool],
) -> tuple[dict[str, int], dict[str, int]] | None:
    found: dict[str, list[dict[str, int]]] = {name: [] for name in names}
    for line in lines:
        match = pattern.match(line)
        if match is None:
            continue
        counts = _counts(split(match.group("items")), words)
        if counts is None or not adds_up(counts, match):
            return None
        found[match.group("name")].append(counts)
    if any(len(entries) != 1 for entries in found.values()):
        return None
    return found[names[0]][0], found[names[1]][0]


def _suite_outcome(files: dict[str, int], tests: dict[str, int], gated: bool) -> str | None:
    if files.get("failed", 0) or tests.get("failed", 0):
        return "failed"
    if tests.get("passed", 0) and not gated:
        return "passed"
    return None


def _jest(text: str) -> str | None:
    lines = _lines(text)
    parsed = _suite_lines(
        lines, _JEST_LINE, lambda items: [item for item in items.split(", ") if item],
        _JEST_WORDS, ("Test Suites", "Tests"), _jest_adds_up,
    )
    if parsed is None:
        return None
    gated = any(_JEST_GATE.search(line) for line in lines)
    return _suite_outcome(*parsed, gated)


def _vitest(text: str) -> str | None:
    lines = _lines(text)
    parsed = _suite_lines(
        lines, _VITEST_LINE, lambda items: items.split(" | "), _VITEST_WORDS,
        ("Test Files", "Tests"), _vitest_adds_up,
    )
    if parsed is None:
        return None
    if any(_VITEST_ERRORS.match(line) for line in lines):
        return "failed"
    gated = any(_VITEST_GATE.search(line) for line in lines)
    return _suite_outcome(*parsed, gated)


_PARSERS: dict[str, Callable[[str], str | None]] = {
    "unittest": _unittest,
    "pytest": _pytest,
    "jest": _jest,
    "vitest": _vitest,
}


def parse_summary(runner: str, text: str) -> str | None:
    """Return ``"passed"`` or ``"failed"`` from one runner's final summary.

    None when the runner is not supported or the summary is absent,
    repeated, incomplete, or contradicts itself.
    """
    parser = _PARSERS.get(runner)
    if parser is None or not text:
        return None
    return parser(_ANSI_RE.sub("", text))


def summary_outcome(runner: str, result: TraceEvent | None) -> str | None:
    """Return the runner summary outcome in one linked tool result, or None."""
    if runner not in SUMMARY_RUNNERS:
        return None
    text = output_text(result)
    return parse_summary(runner, text) if text is not None else None
