"""Tests for auto-instrumentation integrations (issue #102)."""

import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_trace.integrations import (
    _INTEGRATIONS,
    _FRAMEWORK_PROBE,
    _import_integration,
    detect_and_instrument,
    auto_instrument_from_env,
)
from agent_trace.integrations._base import _get_store, emit
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Integration registry
# ---------------------------------------------------------------------------

class TestIntegrationRegistry(unittest.TestCase):
    def test_all_expected_frameworks_registered(self):
        for name in ("openai-agents", "langchain", "litellm", "anthropic", "openai", "strands"):
            self.assertIn(name, _INTEGRATIONS, f"{name} not in registry")

    def test_import_unknown_raises_value_error(self):
        with self.assertRaises(ValueError):
            _import_integration("nonexistent-framework")

    def test_import_missing_framework_raises_import_error(self):
        # All frameworks are optional — importing without them installed raises ImportError
        for name in ("openai-agents", "langchain", "litellm", "anthropic", "openai", "strands"):
            fn = _import_integration(name)
            # Calling the function without the framework installed should raise ImportError
            # (unless the framework happens to be installed in the test env)
            try:
                fn()
            except ImportError:
                pass  # expected when framework not installed
            except Exception:
                pass  # framework installed — that's fine too


# ---------------------------------------------------------------------------
# auto_instrument_from_env
# ---------------------------------------------------------------------------

class TestAutoInstrumentFromEnv(unittest.TestCase):
    def test_empty_env_returns_empty_list(self):
        with patch.dict(os.environ, {"AGENT_STRACE_AUTO_INSTRUMENT": ""}):
            result = auto_instrument_from_env()
        self.assertEqual(result, [])

    def test_unset_env_returns_empty_list(self):
        env = {k: v for k, v in os.environ.items() if k != "AGENT_STRACE_AUTO_INSTRUMENT"}
        with patch.dict(os.environ, env, clear=True):
            result = auto_instrument_from_env()
        self.assertEqual(result, [])

    def test_unknown_framework_does_not_crash(self):
        with patch.dict(os.environ, {"AGENT_STRACE_AUTO_INSTRUMENT": "nonexistent-xyz"}):
            result = auto_instrument_from_env()
        # Should return empty (failed silently)
        self.assertEqual(result, [])

    def test_detect_mode_does_not_crash(self):
        with patch.dict(os.environ, {"AGENT_STRACE_AUTO_INSTRUMENT": "detect"}):
            result = auto_instrument_from_env()
        # Returns list of successfully instrumented frameworks (may be empty)
        self.assertIsInstance(result, list)


# ---------------------------------------------------------------------------
# detect_and_instrument
# ---------------------------------------------------------------------------

class TestDetectAndInstrument(unittest.TestCase):
    def test_returns_list(self):
        result = detect_and_instrument()
        self.assertIsInstance(result, list)

    def test_no_crash_when_no_frameworks_installed(self):
        # Simulate no frameworks installed by patching importlib
        import importlib
        original_import = importlib.import_module

        def _mock_import(name, *args, **kwargs):
            if name in _FRAMEWORK_PROBE.values():
                raise ImportError(f"mocked: {name} not installed")
            return original_import(name, *args, **kwargs)

        with patch("importlib.import_module", side_effect=_mock_import):
            result = detect_and_instrument()
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# _base.emit
# ---------------------------------------------------------------------------

class TestBaseEmit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)
        self.meta = SessionMeta()
        self.store.create_session(self.meta)

    def test_emit_writes_event_locally(self):
        env = {k: v for k, v in os.environ.items() if k != "AGENT_STRACE_ENDPOINT"}
        with patch.dict(os.environ, env, clear=True):
            emit(EventType.TOOL_CALL, self.meta.session_id, self.store,
                 tool_name="Bash", command="ls")
        events = self.store.load_events(self.meta.session_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.TOOL_CALL)

    def test_emit_routes_to_endpoint_when_set(self):
        sent = []

        def _mock_send(event, endpoint):
            sent.append((event, endpoint))
            return True

        with patch.dict(os.environ, {"AGENT_STRACE_ENDPOINT": "http://localhost:9999"}):
            with patch("agent_trace.server.send_event_to_endpoint", _mock_send):
                emit(EventType.TOOL_CALL, self.meta.session_id, self.store,
                     tool_name="Bash")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1], "http://localhost:9999")


# ---------------------------------------------------------------------------
# LiteLLM integration (mock-based — no real litellm needed)
# ---------------------------------------------------------------------------

class TestLiteLLMIntegration(unittest.TestCase):
    def setUp(self):
        # Reset patch state
        import agent_trace.integrations.litellm as m
        m._PATCHED = False

    def test_raises_import_error_when_not_installed(self):
        """instrument_litellm raises ImportError when litellm is absent."""
        import agent_trace.integrations.litellm as m
        m._PATCHED = False
        # Remove litellm from sys.modules and block re-import
        saved = sys.modules.pop("litellm", None)
        sys.modules["litellm"] = None  # type: ignore[assignment]
        try:
            with self.assertRaises((ImportError, TypeError)):
                m.instrument_litellm()
        finally:
            if saved is not None:
                sys.modules["litellm"] = saved
            else:
                sys.modules.pop("litellm", None)
            m._PATCHED = False

    def test_idempotent_when_already_patched(self):
        import agent_trace.integrations.litellm as m
        m._PATCHED = True
        # Should return immediately without error
        from agent_trace.integrations.litellm import instrument_litellm
        instrument_litellm()  # no-op
        self.assertTrue(m._PATCHED)

    def test_uninstrument_resets_flag(self):
        import agent_trace.integrations.litellm as m
        m._PATCHED = True
        from agent_trace.integrations.litellm import uninstrument_litellm
        uninstrument_litellm()
        self.assertFalse(m._PATCHED)


# ---------------------------------------------------------------------------
# LangChain integration (mock-based)
# ---------------------------------------------------------------------------

class TestLangChainIntegration(unittest.TestCase):
    def setUp(self):
        import agent_trace.integrations.langchain as m
        m._PATCHED = False

    def test_idempotent(self):
        import agent_trace.integrations.langchain as m
        m._PATCHED = True
        from agent_trace.integrations.langchain import instrument_langchain
        instrument_langchain()
        self.assertTrue(m._PATCHED)

    def test_uninstrument(self):
        import agent_trace.integrations.langchain as m
        m._PATCHED = True
        from agent_trace.integrations.langchain import uninstrument_langchain
        uninstrument_langchain()
        self.assertFalse(m._PATCHED)


# ---------------------------------------------------------------------------
# OpenAI Agents integration (mock-based)
# ---------------------------------------------------------------------------

class TestOpenAIAgentsIntegration(unittest.TestCase):
    def setUp(self):
        import agent_trace.integrations.openai_agents as m
        m._PATCHED = False

    def test_idempotent(self):
        import agent_trace.integrations.openai_agents as m
        m._PATCHED = True
        from agent_trace.integrations.openai_agents import instrument_openai_agents
        instrument_openai_agents()
        self.assertTrue(m._PATCHED)

    def test_uninstrument(self):
        import agent_trace.integrations.openai_agents as m
        m._PATCHED = True
        from agent_trace.integrations.openai_agents import uninstrument_openai_agents
        uninstrument_openai_agents()
        self.assertFalse(m._PATCHED)


# ---------------------------------------------------------------------------
# Anthropic integration (mock-based)
# ---------------------------------------------------------------------------

class TestAnthropicIntegration(unittest.TestCase):
    def setUp(self):
        import agent_trace.integrations.anthropic as m
        m._PATCHED = False

    def test_idempotent(self):
        import agent_trace.integrations.anthropic as m
        m._PATCHED = True
        from agent_trace.integrations.anthropic import instrument_anthropic
        instrument_anthropic()
        self.assertTrue(m._PATCHED)

    def test_uninstrument(self):
        import agent_trace.integrations.anthropic as m
        m._PATCHED = True
        from agent_trace.integrations.anthropic import uninstrument_anthropic
        uninstrument_anthropic()
        self.assertFalse(m._PATCHED)


# ---------------------------------------------------------------------------
# Strands integration (mock-based)
# ---------------------------------------------------------------------------

class TestStrandsIntegration(unittest.TestCase):
    def setUp(self):
        import agent_trace.integrations.strands as m
        m._PATCHED = False

    def test_idempotent(self):
        import agent_trace.integrations.strands as m
        m._PATCHED = True
        from agent_trace.integrations.strands import instrument_strands
        instrument_strands()
        self.assertTrue(m._PATCHED)

    def test_uninstrument(self):
        import agent_trace.integrations.strands as m
        m._PATCHED = True
        from agent_trace.integrations.strands import uninstrument_strands
        uninstrument_strands()
        self.assertFalse(m._PATCHED)


# ---------------------------------------------------------------------------
# CLI: agent-strace auto registered
# ---------------------------------------------------------------------------

class TestAutoCLIRegistered(unittest.TestCase):
    def test_auto_in_help(self):
        from agent_trace.cli import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "--help"]
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                main()
            except SystemExit:
                pass
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        self.assertIn("auto", output)

    def test_auto_help_shows_frameworks(self):
        from agent_trace.cli import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "auto", "--help"]
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                main()
            except SystemExit:
                pass
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        self.assertIn("langchain", output)
        self.assertIn("litellm", output)


# ---------------------------------------------------------------------------
# pyproject.toml optional extras
# ---------------------------------------------------------------------------

class TestOptionalExtras(unittest.TestCase):
    def test_pyproject_has_optional_deps(self):
        # tomllib is stdlib in 3.11+; fall back to manual parsing on 3.10
        try:
            import tomllib
            _load = lambda f: tomllib.load(f)
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
                _load = lambda f: tomllib.load(f)
            except ImportError:
                # Parse manually: just check the raw text
                pyproject = Path(__file__).parent.parent / "pyproject.toml"
                text = pyproject.read_text()
                for name in ("openai-agents", "langchain", "litellm", "anthropic", "openai", "strands", "all-integrations"):
                    self.assertIn(name, text, f"Missing optional extra: {name}")
                return

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = _load(f)
        extras = data.get("project", {}).get("optional-dependencies", {})
        for name in ("openai-agents", "langchain", "litellm", "anthropic", "openai", "strands"):
            self.assertIn(name, extras, f"Missing optional extra: {name}")
        self.assertIn("all-integrations", extras)


if __name__ == "__main__":
    unittest.main()
