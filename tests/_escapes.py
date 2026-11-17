"""Payloads and checks for tests of commands that print text from a trace.

A trace value can hold escape codes and line breaks: a tool output, a command
the agent chose, a tool name an MCP server chose. Printed raw, they can move
the cursor, erase lines and write to the clipboard. These helpers check that
a formatter shows each of them as text.
"""

from agent_trace.replay import C

# ESC, an OSC 52 clipboard write ended by BEL, a CSI erase-line, CR, LF, a C1
# CSI and a right-to-left override.
PAYLOAD = "A\x1b]52;c;aWQK\x07B\x1b[2K\rC\nD\x9bE\u202eF"
# How a command, a path or a name shows PAYLOAD.
PAYLOAD_TEXT = r"A\x1b]52;c;aWQK\x07B\x1b[2K\rC\nD\x9bE\u202eF"
# How free text, such as a prompt or a tool result, shows PAYLOAD.
PAYLOAD_FREE = r"A\x1b]52;c;aWQK\x07B\x1b[2K C D\x9bE\u202eF"

# The same kinds of characters, for values cut at 20 and at 8 characters.
MID = "\x1b]52;c;aWQK\x07\x1b[2K\r\n\x9b\u202e"
MID_TEXT = r"\x1b]52;c;aWQK\x07\x1b[2K\r\n\x9b\u202e"
SHORT = "Z\x1b]\x07\r\n\x9b\u202e"
SHORT_TEXT = r"Z\x1b]\x07\r\n\x9b\u202e"

# Plain text, which prints as it is. PLAIN_SHORT fits in 8 characters.
PLAIN = "café ✓ 日本語 😀"
PLAIN_SHORT = "café ✓😀"

# The kind of a value -> (payload, how it shows, plain text that fits).
KINDS = {
    "free": (PAYLOAD, PAYLOAD_FREE, PLAIN),
    "text": (PAYLOAD, PAYLOAD_TEXT, PLAIN),
    "mid": (MID, MID_TEXT, PLAIN),
    "short": (SHORT, SHORT_TEXT, PLAIN_SHORT),
}

_COLOR_CODES = sorted(
    (code for name, code in vars(C).items() if not name.startswith("_")),
    key=len,
    reverse=True,
)


def strip_colors(text):
    """Remove the formatters' own color codes, the values of replay.C."""
    for code in _COLOR_CODES:
        text = text.replace(code, "")
    return text


def assert_shows_as_text(case, output, expected, plain_output):
    """Check that *output* shows a value from the trace as *expected*.

    Without the formatter's own color codes, each line must be printable.
    *plain_output* is the output for the same input with a plain value, and
    *output* must have as many lines, so a line break in the value adds none.
    """
    for line in strip_colors(output).split("\n"):
        case.assertTrue(line.isprintable(), ascii(line))
    case.assertIn(expected, output)
    case.assertEqual(output.count("\n"), plain_output.count("\n"), ascii(output))


def check_each_value(case, cases, render):
    """Run *cases* through *render* with a payload in one value at a time.

    Each case is ``(label, kind, build)``. ``build(value)`` returns the input
    for ``render()``, with *value* in the place the label names and plain
    text everywhere else.
    """
    for label, kind, build in cases:
        payload, expected, _ = KINDS[kind]
        with case.subTest(label):
            assert_shows_as_text(case, render(build(payload)), expected, render(build("plain")))


def check_plain_text_unchanged(case, cases, render):
    """Check that plain ASCII and Unicode text in each value prints as it is."""
    for label, kind, build in cases:
        plain = KINDS[kind][2]
        with case.subTest(label):
            case.assertIn(plain, render(build(plain)))
