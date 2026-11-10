"""Subprocess tests for ``agent-strace auto`` start-up instrumentation.

Each test runs the real CLI in a subprocess, which starts the user's command
as another subprocess, and checks what that child interpreter saw.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from agent_trace.models import EventType
from agent_trace.store import TraceStore

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
INTEGRATIONS_DOC = ROOT / "docs" / "integrations.md"
WARNING = "[agent-strace] auto-instrumentation disabled"

# Writes what the child interpreter saw to $PROBE_OUT, then exits $PROBE_EXIT.
PROBE = textwrap.dedent('''\
    import json
    import os
    import sys


    def main():
        agent_trace = sys.modules.get("agent_trace")
        site_module = sys.modules.get("sitecustomize")
        with open(os.environ["PROBE_OUT"], "w") as handle:
            json.dump({
                "instrumented": "agent_trace.auto" in sys.modules,
                "frameworks": os.environ.get("AGENT_STRACE_AUTO_INSTRUMENT"),
                "pythonpath": os.environ.get("PYTHONPATH", ""),
                "pythonstartup": os.environ.get("PYTHONSTARTUP"),
                "bootstrap_existed": os.path.isdir(
                    os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]),
                "sys_path": [os.path.realpath(p or os.curdir) for p in sys.path],
                "agent_trace_file": os.path.realpath(agent_trace.__file__) if agent_trace else None,
                "sitecustomize_runs": getattr(sys, "user_sitecustomize_runs", None),
                "sitecustomize_file": getattr(site_module, "__file__", None),
                "name": __name__,
            }, handle)
        return int(os.environ.get("PROBE_EXIT", "0"))


    if __name__ == "__main__":
        sys.exit(main())
''')

# A sitecustomize the user already has; counts how often it runs in this
# interpreter (an environment variable would leak in from the wrapper).
USER_SITECUSTOMIZE = textwrap.dedent('''\
    import sys
    sys.user_sitecustomize_runs = getattr(sys, "user_sitecustomize_runs", 0) + 1
''')

# Offline stand-in for LiteLLM: the integration patches litellm.completion.
STUB_LITELLM = textwrap.dedent('''\
    class _Usage:
        prompt_tokens = 3
        completion_tokens = 5


    class _Response:
        usage = _Usage()


    def completion(model="", messages=None, **kwargs):
        return _Response()
''')

AGENT_BODY = textwrap.dedent('''\
    import litellm

    litellm.completion(model="stub-model", messages=[{"role": "user", "content": "hi"}])
''')

_STRIPPED_ENV = (
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "AGENT_STRACE_AUTO_INSTRUMENT",
    "AGENT_TRACE_DIR",
    "AGENT_STRACE_ENDPOINT",
    "AGENT_STRACE_WORKSPACE",
    "AGENT_STRACE_TENANT_ID",
)


def _wrapper_code(package_parent: str | None) -> str:
    """Code for ``python -c`` that runs the CLI from this checkout.

    The checkout is put on the wrapper's sys.path only, so the child does not
    inherit it through PYTHONPATH. *package_parent* replaces the directory
    the child loads agent_trace from when it has no installation of its own.
    """
    lines = ["import sys", f"sys.path.insert(0, {str(SRC_DIR)!r})"]
    if package_parent is not None:
        lines += [
            "import agent_trace.auto_bootstrap as bootstrap",
            f"bootstrap.package_parent = lambda: {package_parent!r}",
        ]
    lines += ["from agent_trace.cli import main", "main()"]
    return "\n".join(lines)


def _make_venv(path: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(path)],
        check=True, capture_output=True, timeout=120,
    )
    if os.name == "nt":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def _purelib(python: Path) -> Path:
    out = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return Path(out.stdout.strip())


class AutoCommandTestCase(unittest.TestCase):
    venv_python: Path | None = None
    venv_purelib: Path | None = None

    @classmethod
    def setUpClass(cls):
        cls._venv_tmp = tempfile.TemporaryDirectory()
        try:
            cls.venv_python = _make_venv(Path(cls._venv_tmp.name) / "venv")
            cls.venv_purelib = _purelib(cls.venv_python)
        except (OSError, subprocess.SubprocessError):
            cls.venv_python = None

    @classmethod
    def tearDownClass(cls):
        cls._venv_tmp.cleanup()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.probe = self.tmp / "probe.py"
        self.probe.write_text(PROBE)
        self.probe_out = self.tmp / "probe.json"
        self.traces = self.tmp / "traces"

    def env(self, **extra: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        env.update(
            AGENT_STRACE_TELEMETRY="0",
            DO_NOT_TRACK="1",
            HOME=str(self.tmp),
            AGENT_TRACE_DIR=str(self.traces),
            PROBE_OUT=str(self.probe_out),
        )
        env.update(extra)
        return env

    def run_auto(self, args, env, package_parent=None):
        return subprocess.run(
            [sys.executable, "-c", _wrapper_code(package_parent), *args],
            env=env, cwd=str(self.tmp), capture_output=True, text=True, timeout=120,
        )

    def probe_result(self, proc) -> dict:
        self.assertTrue(
            self.probe_out.exists(),
            f"child did not run\nstdout: {proc.stdout}\nstderr: {proc.stderr}",
        )
        return json.loads(self.probe_out.read_text())

    def require_venv(self) -> Path:
        if self.venv_python is None:
            self.skipTest("python -m venv is unavailable")
        return self.venv_python

    def write_module(self, directory: Path, name: str, source: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.py"
        path.write_text(source)
        return path

    def litellm_events(self, trace_dir: Path) -> list:
        store = TraceStore(str(trace_dir))
        sessions = store.list_sessions()
        self.assertEqual([s.agent_name for s in sessions], ["litellm"])
        return store.load_events(sessions[0].session_id)

    def assert_litellm_events(self, trace_dir: Path) -> None:
        events = self.litellm_events(trace_dir)
        self.assertEqual(
            [e.event_type for e in events],
            [EventType.LLM_REQUEST, EventType.LLM_RESPONSE],
        )
        self.assertEqual(events[0].data["model"], "stub-model")
        self.assertEqual(events[0].data["message_count"], 1)
        self.assertEqual(events[1].data["input_tokens"], 3)
        self.assertEqual(events[1].data["output_tokens"], 5)


class TestAutoInstrumentsChild(AutoCommandTestCase):
    def test_script_with_its_own_agent_trace_is_instrumented(self):
        env = self.env(PYTHONPATH=str(SRC_DIR), PROBE_EXIT="7")
        proc = self.run_auto(
            ["auto", "--framework", "litellm", "--", sys.executable, str(self.probe)], env)
        result = self.probe_result(proc)
        self.assertTrue(result["instrumented"], proc.stderr)
        self.assertEqual(result["frameworks"], "litellm")
        self.assertEqual(result["name"], "__main__")
        self.assertEqual(proc.returncode, 7)
        self.assertNotIn(WARNING, proc.stderr)

    def test_venv_interpreter_without_agent_trace_is_instrumented(self):
        python = self.require_venv()
        proc = self.run_auto(["auto", "--detect", "--", str(python), str(self.probe)], self.env())
        result = self.probe_result(proc)
        self.assertTrue(result["instrumented"], proc.stderr)
        self.assertEqual(result["frameworks"], "detect")
        # Loaded from the launching installation by path...
        self.assertEqual(
            Path(result["agent_trace_file"]).parent.parent, SRC_DIR.resolve())
        # ...without exposing that directory to the child's imports.
        self.assertNotIn(str(SRC_DIR.resolve()), result["sys_path"])
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn(WARNING, proc.stderr)

    def test_python_dash_m_module_is_instrumented(self):
        user_dir = self.tmp / "user-path"
        self.write_module(user_dir, "probe_mod", PROBE)
        env = self.env(PYTHONPATH=str(user_dir), PROBE_EXIT="5")
        proc = self.run_auto(["auto", "--", sys.executable, "-m", "probe_mod"], env)
        result = self.probe_result(proc)
        self.assertTrue(result["instrumented"], proc.stderr)
        self.assertEqual(result["name"], "__main__")
        self.assertEqual(proc.returncode, 5)

    @unittest.skipIf(os.name == "nt", "shebang scripts are POSIX only")
    def test_console_script_and_shebang_script_are_instrumented(self):
        python = self.require_venv()
        if " " in str(python) or " " in sys.executable or " " in str(self.tmp):
            self.skipTest("shebang paths cannot contain spaces")
        # pip writes console scripts as a shebang line plus an import of the
        # entry point from site-packages.
        self.write_module(self.venv_purelib, "agent_strace_probe_cli", PROBE)
        console_script = python.parent / "agent-strace-probe"
        console_script.write_text(
            f"#!{python}\n"
            "import sys\n"
            "from agent_strace_probe_cli import main\n"
            "sys.exit(main())\n"
        )
        shebang_script = self.tmp / "shebang-agent"
        shebang_script.write_text(f"#!{sys.executable}\n{PROBE}")
        try:
            for script in (console_script, shebang_script):
                script.chmod(0o755)
                with self.subTest(script=script.name):
                    self.probe_out.unlink(missing_ok=True)
                    proc = self.run_auto(["auto", "--", str(script)], self.env(PROBE_EXIT="4"))
                    result = self.probe_result(proc)
                    self.assertTrue(result["instrumented"], proc.stderr)
                    self.assertEqual(proc.returncode, 4)
        finally:
            console_script.unlink(missing_ok=True)
            (self.venv_purelib / "agent_strace_probe_cli.py").unlink(missing_ok=True)

    def test_pythonpath_and_pythonstartup_kept_and_bootstrap_removed(self):
        user_dir = self.tmp / "user-path"
        user_dir.mkdir()
        startup = self.tmp / "startup.py"
        env = self.env(PYTHONPATH=str(user_dir), PYTHONSTARTUP=str(startup))
        proc = self.run_auto(["auto", "--", sys.executable, str(self.probe)], env)
        result = self.probe_result(proc)
        self.assertTrue(result["instrumented"], proc.stderr)
        bootstrap, _, rest = result["pythonpath"].partition(os.pathsep)
        self.assertEqual(rest, str(user_dir))
        self.assertTrue(Path(bootstrap).name.startswith("agent-strace-auto-"), bootstrap)
        self.assertTrue(result["bootstrap_existed"])
        self.assertEqual(result["pythonstartup"], str(startup))
        self.assertIn(str(user_dir), result["sys_path"])
        self.assertNotIn(os.path.realpath(bootstrap), result["sys_path"])
        self.assertFalse(Path(bootstrap).exists(), "bootstrap directory was not removed")

    def test_exit_code_is_propagated(self):
        for code in ("0", "1", "42"):
            with self.subTest(code=code):
                self.probe_out.unlink(missing_ok=True)
                proc = self.run_auto(
                    ["auto", "--", sys.executable, str(self.probe)], self.env(PROBE_EXIT=code))
                self.assertTrue(self.probe_result(proc)["instrumented"], proc.stderr)
                self.assertEqual(proc.returncode, int(code))


class TestAutoChainsExistingSitecustomize(AutoCommandTestCase):
    def test_sitecustomize_on_pythonpath_still_runs_once(self):
        user_dir = self.tmp / "user-path"
        user_site = self.write_module(user_dir, "sitecustomize", USER_SITECUSTOMIZE)
        env = self.env(PYTHONPATH=str(user_dir))
        proc = self.run_auto(["auto", "--", sys.executable, str(self.probe)], env)
        result = self.probe_result(proc)
        self.assertTrue(result["instrumented"], proc.stderr)
        self.assertEqual(result["sitecustomize_runs"], 1)
        self.assertEqual(os.path.realpath(result["sitecustomize_file"]), str(user_site))
        self.assertEqual(proc.returncode, 0)

    def test_venv_or_distribution_sitecustomize_still_runs(self):
        python = self.require_venv()
        venv_site = self.write_module(self.venv_purelib, "sitecustomize", USER_SITECUSTOMIZE)
        self.addCleanup(venv_site.unlink, missing_ok=True)
        # Some distributions (Homebrew, Debian) ship a sitecustomize in the
        # standard library directory, which shadows the venv's. Whichever
        # one the interpreter runs on its own must still run under auto.
        baseline = subprocess.run(
            [str(python), str(self.probe)],
            env=self.env(), capture_output=True, text=True, timeout=60,
        )
        expected = self.probe_result(baseline)
        self.assertIsNotNone(expected["sitecustomize_file"])
        self.probe_out.unlink()

        proc = self.run_auto(["auto", "--", str(python), str(self.probe)], self.env())
        result = self.probe_result(proc)
        self.assertTrue(result["instrumented"], proc.stderr)
        self.assertEqual(
            os.path.realpath(result["sitecustomize_file"]),
            os.path.realpath(expected["sitecustomize_file"]),
        )
        self.assertEqual(result["sitecustomize_runs"], expected["sitecustomize_runs"])

    def test_failing_sitecustomize_is_reported_and_program_still_runs(self):
        user_dir = self.tmp / "user-path"
        self.write_module(user_dir, "sitecustomize", "raise RuntimeError('boom')\n")
        env = self.env(PYTHONPATH=str(user_dir), PROBE_EXIT="2")
        proc = self.run_auto(["auto", "--", sys.executable, str(self.probe)], env)
        result = self.probe_result(proc)
        self.assertTrue(result["instrumented"], proc.stderr)
        self.assertIn(
            "Error in sitecustomize; set PYTHONVERBOSE for traceback:\nRuntimeError: boom",
            proc.stderr,
        )
        self.assertEqual(proc.returncode, 2)

    def test_nested_bootstrap_dirs_chain_without_recursion(self):
        from agent_trace import auto_bootstrap

        user_dir = self.tmp / "user-path"
        self.write_module(user_dir, "sitecustomize", USER_SITECUSTOMIZE)
        outer = auto_bootstrap.create_bootstrap_dir()
        self.addCleanup(auto_bootstrap.remove_bootstrap_dir, outer)
        inner = auto_bootstrap.create_bootstrap_dir()
        self.addCleanup(auto_bootstrap.remove_bootstrap_dir, inner)
        env = auto_bootstrap.child_env(
            self.env(PYTHONPATH=os.pathsep.join([outer, str(user_dir), str(SRC_DIR)])),
            inner,
            "detect",
        )
        proc = subprocess.run(
            [sys.executable, str(self.probe)],
            env=env, capture_output=True, text=True, timeout=60,
        )
        result = self.probe_result(proc)
        self.assertTrue(result["instrumented"], proc.stderr)
        self.assertEqual(result["sitecustomize_runs"], 1)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestAutoWhenAgentTraceIsUnimportable(AutoCommandTestCase):
    def test_missing_agent_trace_warns_once_and_program_runs(self):
        python = self.require_venv()
        empty = self.tmp / "no-agent-trace"
        empty.mkdir()
        proc = self.run_auto(
            ["auto", "--framework", "litellm", "--", str(python), str(self.probe)],
            self.env(PROBE_EXIT="3"),
            package_parent=str(empty),
        )
        result = self.probe_result(proc)
        self.assertFalse(result["instrumented"])
        self.assertEqual(proc.stderr.count(WARNING), 1, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(proc.returncode, 3)

    def test_broken_agent_trace_warns_once_and_program_runs(self):
        shadow = self.tmp / "shadow"
        self.write_module(shadow / "agent_trace", "__init__", "raise RuntimeError('broken install')\n")
        env = self.env(PYTHONPATH=str(shadow), PROBE_EXIT="6")
        proc = self.run_auto(["auto", "--", sys.executable, str(self.probe)], env)
        result = self.probe_result(proc)
        self.assertFalse(result["instrumented"])
        self.assertEqual(proc.stderr.count(WARNING), 1, proc.stderr)
        self.assertIn("RuntimeError: broken install", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(proc.returncode, 6)


class TestAutoProducesTraceEvents(AutoCommandTestCase):
    """An instrumented script writes events through a stub LiteLLM module."""

    def setUp(self):
        super().setUp()
        self.stub_dir = self.tmp / "stubs"
        self.write_module(self.stub_dir, "litellm", STUB_LITELLM)
        self.agent = self.tmp / "my_agent.py"
        self.agent.write_text(AGENT_BODY)

    def test_framework_flag_records_llm_events(self):
        env = self.env(PYTHONPATH=str(self.stub_dir))
        proc = self.run_auto(
            ["auto", "--framework", "litellm", "--", sys.executable, str(self.agent)], env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assert_litellm_events(self.traces)

    def test_detect_records_llm_events(self):
        env = self.env(PYTHONPATH=str(self.stub_dir))
        proc = self.run_auto(["auto", "--detect", "--", sys.executable, str(self.agent)], env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assert_litellm_events(self.traces)

    def test_global_trace_dir_flag_selects_the_store(self):
        env = self.env(PYTHONPATH=str(self.stub_dir))
        del env["AGENT_TRACE_DIR"]
        other = self.tmp / "other-traces"
        proc = self.run_auto(
            ["--trace-dir", str(other), "auto", "--framework", "litellm", "--",
             sys.executable, str(self.agent)],
            env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assert_litellm_events(other)
        self.assertFalse((self.tmp / ".agent-traces").exists())


class TestEnvironmentVariableUsage(AutoCommandTestCase):
    """AGENT_STRACE_AUTO_INSTRUMENT without the CLI wrapper, as documented."""

    def setUp(self):
        super().setUp()
        self.stub_dir = self.tmp / "stubs"
        self.write_module(self.stub_dir, "litellm", STUB_LITELLM)
        self.pythonpath = os.pathsep.join([str(SRC_DIR), str(self.stub_dir)])

    def run_python(self, script: Path, **extra: str):
        return subprocess.run(
            [sys.executable, str(script)],
            env=self.env(PYTHONPATH=self.pythonpath, **extra),
            cwd=str(self.tmp), capture_output=True, text=True, timeout=60,
        )

    def documented_snippet(self) -> str:
        blocks = re.findall(r"```python\n(.*?)```", INTEGRATIONS_DOC.read_text(), re.S)
        for block in blocks:
            if re.search(r"^import agent_trace\.auto\b", block, re.M):
                return block
        self.fail("docs/integrations.md has no example that imports agent_trace.auto")

    def test_documented_command_records_llm_events(self):
        doc = INTEGRATIONS_DOC.read_text()
        match = re.search(r"^AGENT_STRACE_AUTO_INSTRUMENT=(\S+) python my_agent\.py$", doc, re.M)
        self.assertIsNotNone(match, "documented env var command not found")
        self.assertIn("litellm", match.group(1).split(","))
        agent = self.tmp / "my_agent.py"
        agent.write_text(self.documented_snippet() + "\n" + AGENT_BODY)
        proc = self.run_python(agent, AGENT_STRACE_AUTO_INSTRUMENT="litellm")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assert_litellm_events(self.traces)

    def test_variable_alone_does_not_instrument(self):
        agent = self.tmp / "my_agent.py"
        agent.write_text(AGENT_BODY)
        proc = self.run_python(agent, AGENT_STRACE_AUTO_INSTRUMENT="litellm")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(self.traces.exists() and TraceStore(str(self.traces)).list_sessions())


if __name__ == "__main__":
    unittest.main()
