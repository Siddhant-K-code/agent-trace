"""Tests for shell command splitting used by validation classifiers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.review import is_test_argv, is_test_command
from agent_trace.shell_split import (
    command_segments,
    is_break,
    is_operator,
    split_tokens,
    strip_comments,
    strip_heredocs,
    write_targets,
)


class TestShellSplit(unittest.TestCase):
    def test_strip_heredocs_all_forms(self):
        for opener in ("<<EOF", "<<'EOF'", '<<"EOF"', "<< EOF", "<<-EOF"):
            with self.subTest(opener=opener):
                text = f"cat > f {opener}\n\tpytest\n\tEOF\nls"
                self.assertEqual(strip_heredocs(text), f"cat > f {opener}\nls")

    def test_strip_heredocs_keeps_here_strings(self):
        self.assertEqual(strip_heredocs("grep x <<< 'EOF'\npytest"), "grep x <<< 'EOF'\npytest")

    def test_two_heredocs_on_one_line(self):
        text = "paste <<A <<B\na body\nA\nb body\nB\nnpm test"
        self.assertEqual(command_segments(text)[-1], ["npm", "test"])

    def test_unterminated_heredoc_drops_the_rest(self):
        self.assertEqual(command_segments("cat <<EOF\npytest\n"), [["cat", "<<", "EOF"]])

    def test_strip_comments_respects_quotes(self):
        self.assertEqual(strip_comments("echo '#x' # y\nls"), "echo '#x' \nls")
        self.assertEqual(strip_comments("echo a#b"), "echo a#b")

    def test_newline_is_a_separator(self):
        self.assertEqual(command_segments("cd repo\npytest -q"), [["cd", "repo"], ["pytest", "-q"]])
        self.assertEqual(command_segments("a &&\nb"), [["a"], ["b"]])

    def test_quoted_newline_is_not_a_separator(self):
        self.assertEqual(command_segments("git commit -m 'a\nb'"), [["git", "commit", "-m", "a\nb"]])

    def test_line_continuation_joins(self):
        self.assertEqual(command_segments("python -m \\\n pytest"), [["python", "-m", "pytest"]])

    def test_unbalanced_quote_falls_back(self):
        tokens, parsed = split_tokens("pytest -k 'oops")
        self.assertFalse(parsed)
        self.assertEqual(tokens[0], "pytest")

    def test_operator_and_break(self):
        self.assertTrue(is_operator(">"))
        self.assertFalse(is_break(">"))
        self.assertTrue(is_break("&&\n"))
        self.assertFalse(is_operator("pytest"))


class TestIsTestCommand(unittest.TestCase):
    def test_single_commands(self):
        self.assertTrue(is_test_command("python -m pytest -q"))
        self.assertTrue(is_test_command("PYTHONPATH=src python3 -m unittest discover -s tests"))
        self.assertFalse(is_test_command("git status"))
        self.assertFalse(is_test_command(""))

    def test_compound_and_multi_line(self):
        self.assertTrue(is_test_command("cd repo && pytest"))
        self.assertTrue(is_test_command("cd repo\nnpm test 2>&1 | tail -4"))
        script = (
            "cd /work && cat > tests/test_a.py <<'EOF'\n"
            "import unittest\n"
            "EOF\n"
            "PYTHONPATH=src python3 -m unittest discover -s tests 2>&1 | tail -4"
        )
        self.assertTrue(is_test_command(script))

    def test_heredoc_body_is_not_a_test_run(self):
        self.assertFalse(is_test_command("cat > run.sh <<'EOF'\npytest -q\nEOF"))
        self.assertFalse(is_test_command("# pytest\nls"))

    def test_is_test_argv(self):
        self.assertTrue(is_test_argv(["CI=1", "pytest"]))
        self.assertFalse(is_test_argv(["echo", "pytest"]))
        self.assertFalse(is_test_argv([]))


if __name__ == "__main__":
    unittest.main()


class TestWriteTargets(unittest.TestCase):
    def test_heredoc_writes_after_absolute_cd(self):
        command = (
            "cd /work/demo && cat > src/pkg/a.py <<'EOF'\n"
            "x = 1 > 0\n"
            "print('>', file=out)\n"
            "EOF\n"
            "cat >> tests/test_a.py <<EOF\n"
            "EOF\n"
            "python3 -m unittest 2>&1 | tail -4"
        )
        self.assertEqual(
            write_targets(command), ["/work/demo/src/pkg/a.py", "/work/demo/tests/test_a.py"]
        )

    def test_redirect_forms(self):
        cases = {
            "echo hi > out.txt": ["out.txt"],
            "echo hi >> log.txt": ["log.txt"],
            "echo hi 1>one.txt": ["one.txt"],
            "make &> all.log": ["all.log"],
            "make &>> all.log": ["all.log"],
            "echo hi >| clobber.txt": ["clobber.txt"],
            "echo a>b": ["b"],
            "cat < in.txt > out.txt": ["out.txt"],
            "echo > './my file.txt'": ["my file.txt"],
            'echo > "/abs/dir/x.md"': ["/abs/dir/x.md"],
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(write_targets(command), expected)

    def test_tee(self):
        self.assertEqual(write_targets("echo x | tee a.txt"), ["a.txt"])
        self.assertEqual(
            write_targets("echo x | tee -a a.txt b.txt > /dev/null"), ["a.txt", "b.txt"]
        )
        self.assertEqual(write_targets("echo x | sudo tee --append /etc/hosts"), ["/etc/hosts"])
        self.assertEqual(write_targets("echo x | tee"), [])

    def test_ignored_targets(self):
        for command in (
            "pytest 2>&1",
            "echo hi >&2",
            "ls > /dev/null 2>&1",
            "ls &> /dev/null",
            "echo > /dev/stdout; echo > /dev/stderr",
            "echo > $OUT",
            'echo > "$DIR/x"',
            "echo > `mktemp`",
            "echo > *.txt",
            "echo > file{1,2}",
            "echo > ~/x",
            "(( a > b ))",
            "[[ a > b ]]",
            "python -c 'print(1 > 0)'",
            "awk '$1 > 5' data.txt",
            "sed -i s/a/b/ f.txt && cp a b && mv c d",
            "echo 'unbalanced > out.txt",
        ):
            with self.subTest(command=command):
                self.assertEqual(write_targets(command), [])

    def test_fd_duplication_does_not_hide_a_file_target(self):
        self.assertEqual(write_targets("pytest > out.txt 2>&1"), ["out.txt"])
        self.assertEqual(write_targets("pytest 2>&1 > out.txt"), ["out.txt"])

    def test_cd_prefix(self):
        self.assertEqual(write_targets("cd sub && echo > a.txt"), ["sub/a.txt"])
        self.assertEqual(
            write_targets("cd /a/b && cd ../c && echo > ../d.txt"), ["/a/d.txt"]
        )
        self.assertEqual(write_targets("cd $X && echo > a.txt; echo > /abs/b.txt"), ["/abs/b.txt"])
        self.assertEqual(write_targets("cd - && echo > a.txt"), [])
        self.assertEqual(write_targets("echo > a.txt; cd /x"), ["a.txt"])
