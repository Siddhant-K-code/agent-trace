"""Make text inert before it goes on a terminal line or a log line.

Text that the watched agent controls, such as a tool call's command, and text
that a server sends, such as an HTTP reason phrase, can hold escape codes and
line breaks. Printed raw, an escape code can erase or hide lines on a terminal,
and a line break can forge a log entry. :func:`inert` shows those characters
as escapes instead, so the reader sees that they were there.

Raw output such as CSV keeps the text as it is, except for a lone UTF-16
surrogate. UTF-8 cannot encode one, so :func:`escape_surrogates` shows it as
its escape.

This module imports nothing from agent_trace, so any module can use it without
an import cycle.
"""

from __future__ import annotations


def inert(text: object) -> str:
    """Return *text* as plain text for one terminal line or one log line.

    Each character that ``str.isprintable()`` rejects becomes the escape that
    ``repr()`` shows for it, without the quotes: ``\\x1b``, ``\\n``, ``\\t``,
    ``\\r``, ``\\x9b``, ``\\u202e``, ``\\ud800``. Every printable character
    stays as it is, including non-ASCII letters, emoji and backslashes, so
    printable text comes back unchanged. A value that is not a str is passed
    through ``str()`` first.
    """
    text = str(text)
    if text.isprintable():
        return text
    return "".join(ch if ch.isprintable() else repr(ch)[1:-1] for ch in text)


def escape_surrogates(value: object) -> object:
    """Return *value* with each lone surrogate in a str shown as its escape.

    A stored string can hold a lone UTF-16 surrogate such as ``\\ud83d``.
    Writing it to a strict UTF-8 stream raises UnicodeEncodeError, so it
    becomes the escape that the stored JSON line holds. UTF-8 can encode every
    other character, so the rest of the text stays as it is. A value that is
    not a str comes back unchanged.
    """
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode("utf-8")
    return value
