"""Split recorded shell command text into words and separators.

Agents often run multi-line scripts: several commands on separate lines,
heredocs that write files, and comments. A shell treats a newline as a
command separator and never runs a heredoc body as a command, so neither does
this tokenizer:

- ``\\`` followed by a newline joins the two lines;
- heredoc bodies (``<<EOF``, ``<<'EOF'``, ``<<"EOF"``, ``<<-EOF``) are dropped,
  up to and including the terminator line;
- ``# ...`` comments are dropped, keeping the newline;
- a newline is returned as a separator token, like ``;``.

This is a best-effort reading for classification, not a shell parser.
"""

from __future__ import annotations

import re
import shlex

PUNCTUATION = ";&|()<>\n"

_HEREDOC = re.compile(r"(?<!<)<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_WORD_BOUNDARY = frozenset(" \t\r\n;&|()<>")


def strip_heredocs(command: str) -> str:
    """Drop heredoc bodies and their terminator lines."""
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


def strip_comments(command: str) -> str:
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


def split_tokens(command: str) -> tuple[list[str], bool]:
    """Return ``(tokens, parsed)`` for a shell command.

    Punctuation tokens group adjacent characters (``&&``, ``2>&1`` gives
    ``2``, ``>&``, ``1``; ``;\\n``). ``parsed`` is False when the text did not
    tokenize as shell (for example an unbalanced quote) and a plain split
    on whitespace and punctuation was used instead.
    """
    text = command.replace("\r\n", "\n").replace("\\\n", " ")
    text = strip_comments(strip_heredocs(text))
    lexer = shlex.shlex(text, posix=True, punctuation_chars=PUNCTUATION)
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer), True
    except ValueError:
        return re.findall(r"[^\s;&|()<>]+|[;&|()<>\n]+", text), False


def is_operator(token: str) -> bool:
    """Return True for a punctuation-only token (separator or redirect)."""
    return bool(token) and set(token) <= set(PUNCTUATION)


def is_break(token: str) -> bool:
    """Return True for a token that separates two simple commands.

    shlex groups adjacent punctuation (``&&\\n``, ``;\\n``, ``\\n\\n``, ``);``),
    so any punctuation-only token that contains a separator counts.
    """
    if not is_operator(token):
        return False
    return token in ("&", "&&") or any(char in token for char in ";|()\n")


def command_segments(command: str) -> list[list[str]]:
    """Split a command into simple-command word lists at separators."""
    segments: list[list[str]] = []
    current: list[str] = []
    for token in split_tokens(command)[0]:
        if is_break(token):
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments
