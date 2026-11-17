"""Tests for agent_trace.inert."""

import unittest

from agent_trace.inert import inert


class TestInert(unittest.TestCase):
    def test_unprintable_characters_become_escapes(self):
        cases = {
            "\x1b": "\\x1b",
            "\n": "\\n",
            "\t": "\\t",
            "\r": "\\r",
            "\x00": "\\x00",
            "\x7f": "\\x7f",
            "\x9b": "\\x9b",
            "\x85": "\\x85",
            "\u202e": "\\u202e",
            "\u2028": "\\u2028",
            "\ud800": "\\ud800",
            "\U000e0001": "\\U000e0001",
        }
        for char, escape in cases.items():
            with self.subTest(char=ascii(char)):
                self.assertEqual(inert(f"a{char}b"), f"a{escape}b")

    def test_escape_codes_and_line_breaks_stay_on_one_line(self):
        self.assertEqual(
            inert("ls \x1b[2K\x1b[1G\x1b[8m"), "ls \\x1b[2K\\x1b[1G\\x1b[8m",
        )
        self.assertEqual(
            inert("true\n2026-01-01T00:00:00Z  FORGED: all clear"),
            "true\\n2026-01-01T00:00:00Z  FORGED: all clear",
        )
        self.assertEqual(
            inert("Oops \x1b]0;PWNED-TITLE\x07\x1b[2J \x9b31m end"),
            "Oops \\x1b]0;PWNED-TITLE\\x07\\x1b[2J \\x9b31m end",
        )

    def test_printable_text_is_unchanged(self):
        for text in (
            "",
            "ls café → 日本",
            "naïve 😀 ⚠️ test",
            "C:\\temp\\x1b and \\n stay as typed",
            "quotes ' and \" stay",
        ):
            with self.subTest(text=text):
                self.assertEqual(inert(text), text)

    def test_printable_characters_next_to_an_escape_are_unchanged(self):
        self.assertEqual(inert("café → 日本\x1b[2K"), "café → 日本\\x1b[2K")
        self.assertEqual(inert("C:\\temp\n'\""), "C:\\temp\\n'\"")

    def test_every_result_is_printable(self):
        for code in range(0x10000):
            char = chr(code)
            with self.subTest(code=hex(code)):
                result = inert(char)
                self.assertTrue(result.isprintable())
                if char.isprintable():
                    self.assertEqual(result, char)

    def test_a_value_that_is_not_a_str_is_converted(self):
        self.assertEqual(inert(ValueError("bad\x1b[2K")), "bad\\x1b[2K")
        self.assertEqual(inert(404), "404")


if __name__ == "__main__":
    unittest.main()
