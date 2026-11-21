"""Split recorded shell command text into words and separators.

Agents often run multi-line scripts: several commands on separate lines,
heredocs that write files, and comments. A shell treats a newline as a
command separator and never runs a heredoc body as a command, so neither does
this tokenizer:

- ``\\`` followed by a newline joins the two lines, except inside single
  quotes, ``$'...'``, and comments;
- ``$'...'`` (ANSI-C quoting) is decoded into one word;
- heredoc bodies (``<<EOF``, ``<<'EOF'``, ``<<"EOF"``, ``<<-EOF``) are dropped,
  up to and including the terminator line; ``<<-`` ignores leading tabs only;
- ``# ...`` comments are dropped up to the newline, which is kept; a
  trailing ``\\`` inside a comment does not join the next line;
- a newline is returned as a separator token, like ``;``.

This is a best-effort reading for classification, not a shell parser.
"""

from __future__ import annotations

import posixpath
import re
import shlex

PUNCTUATION = ";&|()<>\n"

_WORD_BOUNDARY = frozenset(" \t\r\n;&|()<>")

# Marks a word that is only an fd number written directly before ``<`` or ``>``
# (``2>``), so it is not read as a file operand. Removed before tokens leave
# this module.
_FD_MARK = "\ue000"

_ANSI_C_ESCAPES = {
    "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n",
    "r": "\r", "t": "\t", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?",
}


def _quoted_step(text: str, i: int, quote: str, end: int) -> tuple[int, str]:
    """Advance past one character inside ``'...'``, ``"..."`` or ``$'...'``."""
    if text[i] == "\\" and quote != "'" and i + 1 < end:
        return i + 2, quote
    if text[i] == quote[-1]:
        return i + 1, ""
    return i + 1, quote


def _ansi_c_end(text: str, start: int) -> int | None:
    """Return the index after the quote closing ``$'...'`` opened at ``start``."""
    i = start + 2
    while i < len(text):
        if text[i] == "\\":
            i += 2
        elif text[i] == "'":
            return i + 1
        else:
            i += 1
    return None


def _ansi_c_decode(body: str) -> str:
    """Decode the backslash escapes of a ``$'...'`` body as bash does."""
    out: list[str] = []
    i = 0
    while i < len(body):
        char = body[i]
        if char != "\\" or i + 1 >= len(body):
            out.append(char)
            i += 1
            continue
        escape = body[i + 1]
        if escape in _ANSI_C_ESCAPES:
            out.append(_ANSI_C_ESCAPES[escape])
            i += 2
            continue
        if escape in "01234567":
            digits = re.match(r"[0-7]{1,3}", body[i + 1:])
            if digits:
                out.append(chr(int(digits.group(), 8) & 0xFF))
                i += 1 + len(digits.group())
                continue
        if escape in "xuU":
            width = {"x": 2, "u": 4, "U": 8}[escape]
            digits = re.match(rf"[0-9A-Fa-f]{{1,{width}}}", body[i + 2:])
            if digits and int(digits.group(), 16) <= 0x10FFFF:
                out.append(chr(int(digits.group(), 16)))
                i += 2 + len(digits.group())
                continue
        if escape == "c" and i + 2 < len(body):
            out.append(chr(ord(body[i + 2]) & 0x1F))
            i += 3
            continue
        out.append(char + escape)
        i += 2
    return "".join(out)


def _heredoc_word(line: str, start: int) -> tuple[str, bool, int] | None:
    """Return a quote-removed heredoc word, whether it was quoted, and its end."""
    word: list[str] = []
    quote = ""
    quoted = False
    consumed = False
    i = start
    while i < len(line):
        char = line[i]
        if quote == "'":
            consumed = True
            if char == "'":
                quote = ""
            else:
                word.append(char)
            i += 1
            continue
        if quote == '"':
            consumed = True
            if char == '"':
                quote = ""
                i += 1
                continue
            if char == "\\" and i + 1 < len(line):
                following = line[i + 1]
                if following in '$`"\\':
                    word.append(following)
                else:
                    word.extend((char, following))
                i += 2
                continue
            word.append(char)
            i += 1
            continue
        if char in " \t\r" or char in PUNCTUATION:
            break
        consumed = True
        if char in "'\"":
            quote = char
            quoted = True
            i += 1
            continue
        if char == "\\":
            if i + 1 >= len(line):
                return None
            quoted = True
            word.append(line[i + 1])
            i += 2
            continue
        word.append(char)
        i += 1
    if quote or not consumed:
        return None
    return "".join(word), quoted, i


def _heredoc_openers(
    line: str, quote: str, arithmetic_depth: int
) -> tuple[list[tuple[str, bool, bool]], str, int]:
    """Find heredocs in one command line and return the ending quote state."""
    openers: list[tuple[str, bool, bool]] = []
    i = 0
    while i < len(line):
        char = line[i]
        if arithmetic_depth:
            if char == "(":
                arithmetic_depth += 1
            elif char == ")":
                arithmetic_depth -= 1
            i += 1
            continue
        if quote:
            i, quote = _quoted_step(line, i, quote, len(line))
            continue
        if char == "\\" and i + 1 < len(line):
            i += 2
            continue
        if line.startswith("$'", i):
            quote = "$'"
            i += 2
            continue
        if char in "'\"":
            quote = char
            i += 1
            continue
        if char == "#" and (i == 0 or line[i - 1] in _WORD_BOUNDARY):
            break
        if line.startswith("$((", i):
            arithmetic_depth = 2
            i += 3
            continue
        if line.startswith("((", i):
            arithmetic_depth = 2
            i += 2
            continue
        if (
            char == "<"
            and i + 1 < len(line)
            and line[i + 1] == "<"
            and (i == 0 or line[i - 1] != "<")
            and (i + 2 == len(line) or line[i + 2] != "<")
        ):
            word_start = i + 2
            strip_tabs = word_start < len(line) and line[word_start] == "-"
            if strip_tabs:
                word_start += 1
            while word_start < len(line) and line[word_start] in " \t\r":
                word_start += 1
            parsed = _heredoc_word(line, word_start)
            if parsed is not None:
                delimiter, quoted, end = parsed
                openers.append((delimiter, strip_tabs, quoted))
                i = end
                continue
        i += 1
    return openers, quote, arithmetic_depth


def _continues_line(line: str) -> bool:
    trailing = len(line) - len(line.rstrip("\\"))
    return trailing % 2 == 1


def _continues_command_line(line: str, quote: str) -> bool:
    """Return whether the physical command line escapes its ending newline."""
    trailing = len(line) - len(line.rstrip("\\"))
    if trailing % 2 == 0:
        return False
    i = 0
    end = len(line) - trailing
    while i < end:
        char = line[i]
        if quote:
            i, quote = _quoted_step(line, i, quote, end)
            continue
        if char == "\\" and i + 1 < end:
            i += 2
            continue
        if line.startswith("$'", i):
            quote = "$'"
            i += 1
        elif char in "'\"":
            quote = char
        elif char == "#" and (i == 0 or line[i - 1] in _WORD_BOUNDARY):
            return False
        i += 1
    return quote not in ("'", "$'")


def strip_heredocs(command: str) -> str:
    """Drop heredoc bodies and their terminator lines."""
    kept: list[str] = []
    pending: list[tuple[str, bool, bool]] = []
    continued = False
    quote = ""
    arithmetic_depth = 0
    logical_line = ""
    logical_quote = ""
    logical_arithmetic_depth = 0
    continuing_command = False
    for line in command.split("\n"):
        if pending:
            delimiter, strip_tabs, quoted = pending[0]
            candidate = line.lstrip("\t") if strip_tabs else line
            if not continued and candidate == delimiter:
                pending.pop(0)
                continued = False
            else:
                continued = not quoted and _continues_line(candidate)
            continue
        kept.append(line)
        if not continuing_command:
            logical_line = ""
            logical_quote = quote
            logical_arithmetic_depth = arithmetic_depth
        continuing_command = _continues_command_line(
            logical_line + line, logical_quote
        )
        logical_line += line[:-1] if continuing_command else line
        openers, ending_quote, ending_arithmetic_depth = _heredoc_openers(
            logical_line, logical_quote, logical_arithmetic_depth
        )
        if continuing_command:
            continue
        pending.extend(openers)
        quote = ending_quote
        arithmetic_depth = ending_arithmetic_depth
    return "\n".join(kept)


def strip_comments(command: str) -> str:
    """Drop ``# ...`` comments the way a shell does, keeping the newline.

    ``#`` starts a comment only at the start of a word and outside quotes;
    shlex's own comment handling also swallows the newline and fires mid-word.
    """
    return _rewrite(command, fold=False)


def _rewrite(command: str, fold: bool) -> str:
    """Drop comments; with ``fold``, also prepare the text for shlex.

    Folding joins ``\\``-newline outside single quotes, rewrites ``$'...'``
    as an equivalent single-quoted word, and marks fd numbers written
    directly before ``<`` or ``>`` with :data:`_FD_MARK`.
    """
    out: list[str] = []
    quote = ""
    word_start = True
    i = 0
    length = len(command)
    while i < length:
        char = command[i]
        if quote:
            start = i
            i, quote = _quoted_step(command, i, quote, length)
            if not (fold and quote == '"' and command[start:i] == "\\\n"):
                out.append(command[start:i])
            continue
        if char == "\\" and i + 1 < length:
            if not (fold and command[i + 1] == "\n"):
                out.append(command[i:i + 2])
                word_start = False
            i += 2
            continue
        if command.startswith("$'", i):
            word_start = False
            if not fold:
                quote = "$'"
                out.append("$'")
                i += 2
                continue
            end = _ansi_c_end(command, i)
            if end is None:
                out.append(command[i:])
                break
            body = _ansi_c_decode(command[i + 2:end - 1])
            out.append("'" + body.replace("'", "'\\''") + "'")
            i = end
            continue
        if char == "#" and word_start:
            end = command.find("\n", i)
            if end == -1:
                break
            i = end
            continue
        if fold and word_start and char in "0123456789":
            end = i
            while end < length and command[end] in "0123456789":
                end += 1
            if end < length and command[end] in "<>":
                out.append(_FD_MARK + command[i:end])
                word_start = False
                i = end
                continue
        if char in "'\"":
            quote = char
        out.append(char)
        word_start = char in _WORD_BOUNDARY
        i += 1
    return "".join(out)


def _lex(command: str) -> tuple[list[str], bool]:
    """Tokenize like :func:`split_tokens`, keeping :data:`_FD_MARK` on fd words."""
    text = _rewrite(strip_heredocs(command.replace("\r\n", "\n")), fold=True)
    lexer = shlex.shlex(text, posix=True, punctuation_chars=PUNCTUATION)
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer), True
    except ValueError:
        return re.findall(r"[^\s;&|()<>]+|[;&|()<>\n]+", text), False


def split_tokens(command: str) -> tuple[list[str], bool]:
    """Return ``(tokens, parsed)`` for a shell command.

    Punctuation tokens group adjacent characters (``&&``, ``2>&1`` gives
    ``2``, ``>&``, ``1``; ``;\\n``). ``parsed`` is False when the text did not
    tokenize as shell (for example an unbalanced quote) and a plain split
    on whitespace and punctuation was used instead.
    """
    tokens, parsed = _lex(command)
    return [token.replace(_FD_MARK, "") for token in tokens], parsed


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


_WRITE_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>"})
_UNRESOLVED = re.compile(r"[$`*?\[\]{}]|^~")
_ABSOLUTE = re.compile(r"^(?:/|[A-Za-z]:/)")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _literal_path(word: str) -> str | None:
    if not word or word == "-" or _UNRESOLVED.search(word):
        return None
    return word


def _resolve(target: str, base: str | None) -> str | None:
    if target.startswith("/dev/"):
        return None
    if _ABSOLUTE.match(target):
        return posixpath.normpath(target)
    if base is None:
        return None
    return posixpath.normpath(posixpath.join(base, target))


def _command_words(words: list[str]) -> list[str]:
    words = list(words)
    while words and _ENV_ASSIGN.match(words[0]):
        words.pop(0)
    if words and words[0] == "sudo":
        words.pop(0)
    return words


# Words that can come before a command name; ``{`` and ``if``/``while``/
# ``until`` also open a compound command.
_PREFIX_WORDS = frozenset(
    {"!", "{", "if", "then", "elif", "else", "do", "while", "until", "time"}
)
_CLOSERS = {"}": "{", "fi": "if", "done": "loop", "esac": "case"}
_OPERATORS = (";;&", "&&", "||", ";;", ";&", "|&", "&", "|", ";", "(", ")")


def _operators(token: str) -> list[str]:
    """Split a punctuation token into operators, skipping newlines and redirects."""
    found: list[str] = []
    i = 0
    while i < len(token):
        for operator in _OPERATORS:
            if token.startswith(operator, i):
                found.append(operator)
                i += len(operator)
                break
        else:
            i += 1
    return found


def _tee_files(words: list[str]) -> list[str]:
    files: list[str] = []
    options = True
    for word in words[1:]:
        if options and word == "--":
            options = False
        elif options and word.startswith("-") and word != "-":
            continue
        else:
            files.append(word)
    return files


def _cd_base(words: list[str], base: str | None) -> str | None:
    args = [word for word in words[1:] if word not in ("-P", "-L", "-e", "--")]
    if len(args) != 1:
        return None
    target = _literal_path(args[0])
    if target is None:
        return None
    if _ABSOLUTE.match(target):
        return posixpath.normpath(target)
    if base is None:
        return None
    return posixpath.normpath(posixpath.join(base, target))


_READ_COMMANDS = frozenset({"cat", "head", "tail", "less", "more", "nl", "wc"})
# Short options that take the next word as their value.
_READ_VALUE_FLAGS = {
    "head": frozenset({"-n", "-c"}),
    "tail": frozenset({"-n", "-c", "-s", "--pid", "--sleep-interval", "--max-unchanged-stats"}),
    "nl": frozenset({"-b", "-d", "-f", "-h", "-i", "-l", "-n", "-s", "-v", "-w"}),
}


def _read_files(name: str, words: list[str]) -> list[str]:
    files: list[str] = []
    value_flags = _READ_VALUE_FLAGS.get(name, frozenset())
    options = True
    skip = False
    for word in words[1:]:
        if skip:
            skip = False
        elif options and word == "--":
            options = False
        elif options and word in value_flags:
            skip = True
        elif options and len(word) > 1 and word[0] in "-+":
            continue
        else:
            files.append(word)
    return files


def _file_targets(command: str) -> tuple[list[str], list[str]]:
    """Return ``(written, read)`` paths for :func:`write_targets` and :func:`read_targets`."""
    tokens, parsed = _lex(command)
    if not parsed:
        return [], []
    written: list[str] = []
    read: list[str] = []
    base: str | None = ""
    # Open ``(``, ``{``, ``if``, loops and ``case``: [kind, base when opened,
    # whether the compound command was piped into, whether a finished branch
    # changed the directory].
    frames: list[list] = []
    words: list[str] = []
    redirects: list[str] = []
    inputs: list[str] = []
    arithmetic = False
    piped = False

    def add(found: list[str], targets: list[str], at: str | None) -> None:
        for word in targets:
            literal = _literal_path(word)
            path = _resolve(literal, at) if literal else None
            if path and path not in found:
                found.append(path)

    def next_branch(kind: str) -> None:
        """Start an ``elif``/``else``/case branch from the directory before it."""
        nonlocal base
        if frames and frames[-1][0] == kind:
            frames[-1][3] = frames[-1][3] or base != frames[-1][1]
            base = frames[-1][1]

    def finish(separator: str) -> None:
        nonlocal base
        operators = _operators(separator)
        first = operators[0] if operators else ""
        detached = first in ("|", "|&", "&")
        in_pipe = piped
        command_words = list(words)
        while command_words and command_words[0] in _PREFIX_WORDS:
            word = command_words.pop(0)
            if word in ("{", "if", "while", "until"):
                kind = "loop" if word in ("while", "until") else word
                frames.append([kind, base, in_pipe, False])
                in_pipe = False
            elif word in ("elif", "else"):
                next_branch("if")
        at = base
        if command_words and command_words[0] in _CLOSERS:
            kind = _CLOSERS[command_words.pop(0)]
            if frames and frames[-1][0] == kind:
                _, saved, frame_piped, changed = frames.pop()
                # Redirects on a compound command open before its body runs.
                at = saved
                if frame_piped or detached:
                    base = saved
                elif kind != "{" and (changed or base != saved):
                    base = None
        command_words = _command_words(command_words)
        name = command_words[0].rsplit("/", 1)[-1] if command_words else ""
        if name in ("for", "select", "case"):
            frames.append(["case" if name == "case" else "loop", base, in_pipe, False])
        elif not arithmetic and name != "[[":
            targets = list(redirects)
            if name == "tee":
                targets.extend(_tee_files(command_words))
            add(written, targets, at)
            sources = list(inputs)
            if name in _READ_COMMANDS:
                sources.extend(_read_files(name, command_words))
            add(read, sources, at)
            if name == "cd" and not (in_pipe or detached):
                base = _cd_base(command_words, base)
        if first in (";;", ";&", ";;&"):
            next_branch("case")

    def close_paren() -> None:
        nonlocal base
        if frames and frames[-1][0] == "case":
            return
        for position in range(len(frames) - 1, -1, -1):
            if frames[position][0] == "(":
                base = frames[position][1]
                del frames[position:]
                return

    index = 0
    while index < len(tokens):
        token = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        operand = bool(following) and not is_operator(following)
        if token in _WRITE_REDIRECTS and operand:
            redirects.append(following)
            index += 2
            continue
        if token == "<" and operand:
            inputs.append(following)
            index += 2
            continue
        if is_break(token):
            finish(token)
            words, redirects, inputs = [], [], []
            arithmetic = token.endswith("((")
            operators = _operators(token)
            for operator in operators:
                if operator == "(":
                    frames.append(["(", base, False, False])
                elif operator == ")":
                    close_paren()
            piped = bool(operators) and operators[-1] in ("|", "|&")
        elif is_operator(token):
            if operand:
                index += 1
        elif not token.startswith(_FD_MARK):
            words.append(token)
        index += 1
    finish("")
    return written, read


def write_targets(command: str) -> list[str]:
    """Return file paths a command writes through redirection or ``tee``.

    Covers ``>``, ``>>``, ``>|``, ``&>``, ``&>>`` (with or without a leading
    fd number) and ``tee [-a] FILE...``. A path is relative to the directory
    the command started in, unless an earlier ``cd DIR`` in the same command
    moved it: an absolute ``DIR`` is applied, a relative literal one is joined,
    and any other ``cd`` makes later relative targets unknown, so they are
    dropped. A ``cd`` inside ``(...)`` or ``$(...)`` ends there, and one in a
    pipeline or run with ``&`` is ignored. A ``cd`` inside ``{ ...; }``
    carries on after the group (unless the group is piped or run with ``&``).
    A ``cd`` inside ``if``, ``while``, ``until``, ``for`` or ``case`` applies
    to the rest of that branch or body; the next ``elif``, ``else`` or case
    pattern starts from the directory before the compound command, and after
    it ends later relative targets are dropped, since the directory then
    depends on which branch ran or how often the loop did. Targets under ``/dev/``, fd
    duplications such as ``2>&1``, and
    words with ``$``, backticks, globs, braces, or a leading ``~`` are skipped.
    Other ways to write files (``sed -i``, ``cp``, interpreters) are not read.
    """
    return _file_targets(command)[0]


def read_targets(command: str) -> list[str]:
    """Return file paths a command reads with a plain file viewer.

    Covers file operands of ``cat``, ``head``, ``tail``, ``less``, ``more``,
    ``nl``, and ``wc`` (option words, the values of options such as
    ``-n 20``, and fd numbers such as the ``2`` in ``2>/dev/null`` are
    skipped) and ``<`` input redirect targets. Paths follow the
    same ``cd`` and literal-word rules as :func:`write_targets`. Operands of
    ``grep``, ``sed``, ``awk`` and other tools are not read.
    """
    return _file_targets(command)[1]
