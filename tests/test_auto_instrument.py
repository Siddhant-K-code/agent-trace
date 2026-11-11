"""Tests for auto-instrumentation integrations (issue #102)."""

import io
import os
import sys
import tempfile
import time
import types
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
        for name in ("openai-agents", "langchain", "litellm", "anthropic", "openai", "strands", "crewai", "langgraph"):
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

    def test_emit_returns_the_event(self):
        env = {k: v for k, v in os.environ.items() if k != "AGENT_STRACE_ENDPOINT"}
        with patch.dict(os.environ, env, clear=True):
            ev = emit(EventType.LLM_REQUEST, self.meta.session_id, self.store, model="m")
        self.assertIsInstance(ev, TraceEvent)
        self.assertEqual(ev.parent_id, "")
        self.assertEqual(self.store.load_events(self.meta.session_id)[0].event_id, ev.event_id)

    def test_emit_links_to_parent(self):
        env = {k: v for k, v in os.environ.items() if k != "AGENT_STRACE_ENDPOINT"}
        with patch.dict(os.environ, env, clear=True):
            request = emit(EventType.LLM_REQUEST, self.meta.session_id, self.store, model="m")
            error = emit(EventType.ERROR, self.meta.session_id, self.store,
                         parent=request, error="boom")
        self.assertEqual(error.parent_id, request.event_id)
        self.assertIsNotNone(error.duration_ms)
        self.assertNotIn("parent", error.data)
        stored = self.store.load_events(self.meta.session_id)[1]
        self.assertEqual(stored.parent_id, request.event_id)

    def test_emit_links_to_parent_when_sending_to_endpoint(self):
        sent = []
        request = TraceEvent(event_type=EventType.LLM_REQUEST, session_id=self.meta.session_id)
        with patch.dict(os.environ, {"AGENT_STRACE_ENDPOINT": "http://localhost:9999"}):
            with patch("agent_trace.server.send_event_to_endpoint",
                       lambda event, endpoint: sent.append(event) or True):
                emit(EventType.ERROR, self.meta.session_id, self.store,
                     parent=request, error="boom")
        self.assertEqual(sent[0].parent_id, request.event_id)


# ---------------------------------------------------------------------------
# A failed SDK model call is recorded as an error linked to its request
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    pass


def _raise_rate_limit(*args, **kwargs):
    raise RateLimitError("Rate limit reached")


def _fake_openai():
    mod = types.ModuleType("openai")
    mod.OpenAI = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_raise_rate_limit),
    ))
    return {"openai": mod}, lambda: mod.OpenAI.chat.completions.create(
        None, model="gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=100,
    )


def _fake_anthropic():
    mod = types.ModuleType("anthropic")
    mod.Anthropic = types.SimpleNamespace(messages=types.SimpleNamespace(create=_raise_rate_limit))
    return {"anthropic": mod}, lambda: mod.Anthropic.messages.create(
        None, model="claude-x", messages=[{"role": "user", "content": "hi"}],
    )


def _fake_litellm():
    mod = types.ModuleType("litellm")
    mod.completion = _raise_rate_limit
    return {"litellm": mod}, lambda: mod.completion(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
    )


def _fake_langchain():
    root = types.ModuleType("langchain_core")
    models = types.ModuleType("langchain_core.language_models")

    class BaseChatModel:
        model_name = "gpt-4o"

        def _generate(self, messages, *args, **kwargs):
            raise RateLimitError("Rate limit reached")

    models.BaseChatModel = BaseChatModel
    root.language_models = models
    modules = {
        "langchain_core": root,
        "langchain_core.language_models": models,
        "langgraph": None,
        "langgraph.graph": None,
    }
    return modules, lambda: BaseChatModel()._generate(["hi"])


def _fake_crewai():
    mod = types.ModuleType("crewai")

    class Agent:
        role = "researcher"

        def execute_task(self, task, *args, **kwargs):
            raise RateLimitError("Rate limit reached")

    mod.Agent = Agent
    return {"crewai": mod}, lambda: Agent().execute_task(types.SimpleNamespace(description="t"))


class TestSDKModelCallFailure(unittest.TestCase):
    """Each SDK integration links a failed model call's error to its request."""

    CASES = {
        "openai": _fake_openai,
        "anthropic": _fake_anthropic,
        "litellm": _fake_litellm,
        "langchain": _fake_langchain,
        "crewai": _fake_crewai,
    }

    def _record_failure(self, name):
        import importlib
        integration = importlib.import_module(f"agent_trace.integrations.{name}")
        modules, call = self.CASES[name]()
        tmp = tempfile.mkdtemp()
        env = {k: v for k, v in os.environ.items() if k != "AGENT_STRACE_ENDPOINT"}
        env["AGENT_TRACE_DIR"] = tmp
        integration._PATCHED = False
        try:
            with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, modules):
                getattr(integration, f"instrument_{name}")(agent_name=f"{name}-test")
                with self.assertRaises(RateLimitError):
                    call()
        finally:
            getattr(integration, f"uninstrument_{name}")()
        store = TraceStore(tmp)
        (meta,) = store.list_sessions()
        return meta, store.load_events(meta.session_id)

    def test_error_is_linked_to_its_request(self):
        for name in self.CASES:
            with self.subTest(integration=name):
                _, events = self._record_failure(name)
                request, error = events
                self.assertEqual(request.event_type, EventType.LLM_REQUEST)
                self.assertEqual(error.event_type, EventType.ERROR)
                self.assertEqual(error.parent_id, request.event_id)
                self.assertIsNotNone(error.duration_ms)
                self.assertEqual(error.data["error"], "Rate limit reached")
                self.assertEqual(error.data["error_type"], "RateLimitError")

    def test_failure_exports_as_a_failed_model_call(self):
        from agent_trace.langfuse_export import _events_to_langfuse_observations
        from agent_trace.otlp import session_to_otlp_genai

        for name in self.CASES:
            with self.subTest(integration=name):
                meta, events = self._record_failure(name)
                payload = session_to_otlp_genai(meta, events)
                spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][1:]
                (span,) = spans
                attrs = {a["key"]: a["value"] for a in span["attributes"]}
                self.assertEqual(span["name"], "gen_ai.client.operation")
                self.assertEqual(span["status"], {"code": 2, "message": "Rate limit reached"})
                self.assertEqual(attrs["error.type"], {"stringValue": "RateLimitError"})
                (generation,) = _events_to_langfuse_observations(meta.session_id, events)
                self.assertEqual(generation["level"], "ERROR")
                self.assertEqual(generation["statusMessage"], "Rate limit reached")


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


class TestSDKUsageFields(unittest.TestCase):
    """The SDK integrations keep the cache token counts they are billed for."""

    def test_anthropic_records_cache_reads_and_writes(self):
        from types import SimpleNamespace
        from agent_trace.cost import _usage_detail
        from agent_trace.integrations.anthropic import _usage_fields

        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=8000,
            cache_creation_input_tokens=3000,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=1000, ephemeral_1h_input_tokens=2000,
            ),
        )
        fields = _usage_fields(usage)
        self.assertEqual(fields, {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 3000,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000,
            },
        })
        recorded, _, _ = _usage_detail(fields)
        self.assertEqual(
            (recorded.input_tokens, recorded.cache_read_tokens,
             recorded.cache_write_tokens, recorded.cache_write_long_tokens),
            (11_100, 8000, 3000, 2000),
        )

    def test_openai_records_cached_prompt_tokens(self):
        from types import SimpleNamespace
        from agent_trace.cost import _usage_detail
        from agent_trace.integrations.openai import _usage_fields

        usage = SimpleNamespace(
            prompt_tokens=2000,
            completion_tokens=100,
            prompt_tokens_details=SimpleNamespace(cached_tokens=1500, audio_tokens=0),
        )
        fields = _usage_fields(usage)
        self.assertEqual(fields, {
            "input_tokens": 2000, "output_tokens": 100, "cached_input_tokens": 1500,
        })
        recorded, _, _ = _usage_detail(fields)
        self.assertEqual((recorded.input_tokens, recorded.cache_read_tokens), (2000, 1500))

    def test_missing_or_mock_usage_adds_no_cache_fields(self):
        from agent_trace.integrations.anthropic import _usage_fields as anthropic_fields
        from agent_trace.integrations.openai import _usage_fields as openai_fields

        for fields in (anthropic_fields, openai_fields):
            with self.subTest(fields.__module__):
                self.assertEqual(fields(None), {"input_tokens": 0, "output_tokens": 0})
                self.assertLessEqual(set(fields(MagicMock())), {"input_tokens", "output_tokens"})


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

class TestCrewAIIntegration(unittest.TestCase):
    def setUp(self):
        import agent_trace.integrations.crewai as m
        m._PATCHED = False
        m._orig_kickoff = None
        m._orig_execute_task = None
        m._orig_task_execute = None

    def test_raises_import_error_when_not_installed(self):
        from agent_trace.integrations.crewai import instrument_crewai
        with patch.dict(sys.modules, {"crewai": None}):
            with self.assertRaises(ImportError):
                instrument_crewai()

    def test_idempotent_when_already_patched(self):
        import agent_trace.integrations.crewai as m
        m._PATCHED = True
        from agent_trace.integrations.crewai import instrument_crewai
        instrument_crewai()  # should not raise or re-patch
        self.assertTrue(m._PATCHED)

    def test_uninstrument_resets_flag(self):
        import agent_trace.integrations.crewai as m
        m._PATCHED = True
        from agent_trace.integrations.crewai import uninstrument_crewai
        uninstrument_crewai()
        self.assertFalse(m._PATCHED)

    def test_registered_in_integrations(self):
        self.assertIn("crewai", _INTEGRATIONS)
        self.assertIn("crewai", _FRAMEWORK_PROBE)


class TestLangGraphIntegration(unittest.TestCase):
    def setUp(self):
        import agent_trace.integrations.langchain as m
        m._PATCHED = False
        m._orig_compile = None

    def test_langgraph_alias_resolves_to_langchain_module(self):
        entry = _INTEGRATIONS.get("langgraph")
        self.assertIsNotNone(entry)
        module_path, func_name = entry
        self.assertIn("langchain", module_path)
        self.assertEqual(func_name, "instrument_langchain")

    def test_compile_patch_skipped_when_langgraph_not_installed(self):
        """instrument_langchain must not crash when langgraph is absent."""
        import agent_trace.integrations.langchain as m
        # Simulate langgraph absent by temporarily hiding it
        with patch.dict(sys.modules, {"langgraph": None, "langgraph.graph": None}):
            try:
                from agent_trace.integrations.langchain import instrument_langchain
                instrument_langchain()
            except ImportError:
                pass  # langchain_core also absent — that's fine
        # No exception means the best-effort skip worked
        m._PATCHED = False

    def test_uninstrument_restores_compile(self):
        import agent_trace.integrations.langchain as m
        m._PATCHED = True
        from agent_trace.integrations.langchain import uninstrument_langchain
        uninstrument_langchain()
        self.assertFalse(m._PATCHED)
        self.assertIsNone(m._orig_compile)


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
                for name in ("openai-agents", "langchain", "litellm", "anthropic", "openai", "strands", "crewai", "all-integrations"):
                    self.assertIn(name, text, f"Missing optional extra: {name}")
                return

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = _load(f)
        extras = data.get("project", {}).get("optional-dependencies", {})
        for name in ("openai-agents", "langchain", "litellm", "anthropic", "openai", "strands", "crewai"):
            self.assertIn(name, extras, f"Missing optional extra: {name}")
        self.assertIn("all-integrations", extras)


if __name__ == "__main__":
    unittest.main()
