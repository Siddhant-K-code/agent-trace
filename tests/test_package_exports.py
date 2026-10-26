"""Top-level ``agent_trace`` exports used by the documented Python API."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import agent_trace

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = Path(agent_trace.__file__).resolve().parents[1]
_DOC_IMPORT_RE = re.compile(r"^from agent_trace import (.+)$", re.MULTILINE)


def _documented_names(doc: Path) -> list[str]:
    names: list[str] = []
    for match in _DOC_IMPORT_RE.finditer(doc.read_text(encoding="utf-8")):
        names.extend(n.strip() for n in match.group(1).split(",") if n.strip())
    return names


class TestPackageExports(unittest.TestCase):
    def test_documented_imports_resolve(self):
        for rel in ("README.md", "docs/setup.md"):
            with self.subTest(doc=rel):
                names = _documented_names(_REPO_ROOT / rel)
                self.assertTrue(names, f"no `from agent_trace import` example in {rel}")
                for name in names:
                    self.assertTrue(callable(getattr(agent_trace, name)), name)
                    self.assertIn(name, agent_trace.__all__)

    def test_exports_are_the_decorator_functions(self):
        from agent_trace import decorator

        for name in agent_trace.__all__:
            self.assertIs(getattr(agent_trace, name), getattr(decorator, name))

    def test_from_import_statement_works(self):
        namespace: dict = {}
        exec(
            "from agent_trace import trace_tool, trace_llm_call, start_session, "
            "end_session, log_decision",
            namespace,
        )
        self.assertIn("start_session", namespace)

    def test_unknown_attribute_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            agent_trace.not_a_real_export  # noqa: B018
        self.assertFalse(hasattr(agent_trace, "not_a_real_export"))

    def test_version_is_a_plain_literal(self):
        init = Path(agent_trace.__file__).read_text(encoding="utf-8")
        lines = [line for line in init.splitlines() if line.startswith("__version__")]
        self.assertEqual(lines, [f'__version__ = "{agent_trace.__version__}"'])

    def test_import_does_not_load_decorator_until_accessed(self):
        code = (
            "import sys, agent_trace\n"
            "print('agent_trace.decorator' in sys.modules)\n"
            "print('agent_trace.store' in sys.modules)\n"
            "agent_trace.start_session\n"
            "print('agent_trace.decorator' in sys.modules)\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(_SRC_DIR), env.get("PYTHONPATH", "")) if p
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        self.assertEqual(result.stdout.split(), ["False", "False", "True"])


if __name__ == "__main__":
    unittest.main()
