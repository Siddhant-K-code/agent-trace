"""Fixture-backed OTLP GenAI and OpenInference conformance tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent_trace.langfuse_export import (
    _events_to_langfuse_observations,
    _session_to_langfuse_trace,
)
from agent_trace.models import SessionMeta, TraceEvent
from agent_trace.otlp import session_to_otlp_genai


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FIXTURE = ROOT / "tests" / "fixtures" / "otlp_genai" / "session.json"
OTLP_FIXTURE = ROOT / "examples" / "otlp" / "genai-conformance.json"
LANGFUSE_FIXTURE = ROOT / "examples" / "otlp" / "langfuse-conformance.json"
MAPPING_FIXTURE = ROOT / "examples" / "otlp" / "openinference-mapping.json"
BACKEND_FIXTURE = ROOT / "examples" / "otlp" / "backend-recipes.json"


def _load_source() -> tuple[SessionMeta, list[TraceEvent]]:
    source = json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    meta = SessionMeta.from_json(json.dumps(source["meta"]))
    events = [TraceEvent.from_json(json.dumps(item)) for item in source["events"]]
    return meta, events


def _spans(payload: dict) -> list[dict]:
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _value(value: dict):
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return value[key]
    return None


def _attributes(item: dict) -> dict[str, object]:
    return {
        attribute["key"]: _value(attribute["value"])
        for attribute in item.get("attributes", [])
    }


def _span_for_kind(payload: dict, kind: str) -> dict:
    for span in _spans(payload):
        if _attributes(span).get("openinference.span.kind") == kind:
            return span
    raise AssertionError(f"missing OpenInference span kind {kind}")


class TestOTLPGenAIConformanceFixture(unittest.TestCase):
    def setUp(self):
        self.meta, self.events = _load_source()
        self.payload = session_to_otlp_genai(
            self.meta,
            self.events,
            service_name="agent-trace-conformance",
        )

    def test_payload_matches_checked_in_http_json_fixture(self):
        expected = json.loads(OTLP_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(self.payload, expected)

    def test_payload_generation_is_deterministic(self):
        second = session_to_otlp_genai(
            self.meta,
            self.events,
            service_name="agent-trace-conformance",
        )
        self.assertEqual(self.payload, second)

    def test_project_genai_semantic_convention_contract(self):
        scope = self.payload["resourceSpans"][0]["scopeSpans"][0]["scope"]
        root = _span_for_kind(self.payload, "AGENT")
        llm = _span_for_kind(self.payload, "LLM")
        tool = _span_for_kind(self.payload, "TOOL")
        root_attrs = _attributes(root)
        llm_attrs = _attributes(llm)
        tool_attrs = _attributes(tool)

        self.assertEqual(scope["version"], "genai-semconv-1.27")
        self.assertEqual(root_attrs["gen_ai.agent.id"], self.meta.session_id)
        self.assertEqual(root_attrs["gen_ai.operation.name"], "agent.session")
        self.assertEqual(llm["name"], "gen_ai.client.operation")
        self.assertEqual(llm_attrs["gen_ai.system"], "anthropic")
        self.assertEqual(llm_attrs["gen_ai.request.model"], "claude-3-5-sonnet")
        self.assertEqual(llm_attrs["gen_ai.usage.input_tokens"], "12")
        self.assertEqual(llm_attrs["gen_ai.usage.output_tokens"], "8")
        self.assertEqual(llm_attrs["gen_ai.response.finish_reasons"], "end_turn")
        self.assertEqual(tool_attrs["gen_ai.tool.name"], "Read")
        self.assertEqual(tool_attrs["gen_ai.tool.call.id"], "tool-read-call")

    def test_http_json_types_and_parenting_are_stable(self):
        spans = _spans(self.payload)
        root = spans[0]
        self.assertEqual(root["kind"], 1)
        self.assertIsInstance(root["startTimeUnixNano"], str)
        self.assertIsInstance(root["endTimeUnixNano"], str)
        self.assertEqual(len(root["traceId"]), 32)
        self.assertEqual(len(root["spanId"]), 16)
        for child in spans[1:]:
            self.assertEqual(child["kind"], 3)
            self.assertEqual(child["traceId"], root["traceId"])
            self.assertEqual(child["parentSpanId"], root["spanId"])
            self.assertEqual(
                _attributes(child)["session.id"],
                self.meta.session_id,
            )

    def test_preservation_attributes_cover_required_semantics(self):
        root = _span_for_kind(self.payload, "AGENT")
        llm = _span_for_kind(self.payload, "LLM")
        tool = _span_for_kind(self.payload, "TOOL")
        root_attrs = _attributes(root)
        llm_attrs = _attributes(llm)
        tool_attrs = _attributes(tool)

        self.assertEqual(llm_attrs["llm.provider"], "anthropic")
        self.assertEqual(llm_attrs["llm.model_name"], "claude-3-5-sonnet")
        self.assertEqual(llm_attrs["llm.token_count.prompt"], "12")
        self.assertEqual(llm_attrs["llm.token_count.completion"], "8")
        self.assertEqual(llm_attrs["llm.token_count.total"], "20")
        self.assertAlmostEqual(llm_attrs["llm.cost.total"], 0.000156)
        self.assertEqual(tool_attrs["tool.name"], "Read")
        self.assertEqual(tool_attrs["input.value"], '{"file_path":"README.md"}')
        self.assertEqual(tool_attrs["output.value"], "fixture content")
        self.assertTrue(root_attrs["agent.strace.redacted"])
        self.assertEqual(root_attrs["agent.strace.redacted_event_count"], "1")
        self.assertEqual(root_attrs["agent.strace.evidence.status"], "partial")
        self.assertEqual(root_attrs["agent.strace.evidence.gap_count"], "1")
        self.assertEqual(
            json.loads(root_attrs["agent.strace.evidence.gap_codes"]),
            ["orphan_tool_result"],
        )
        provenance = json.loads(root_attrs["agent.strace.provenance"])
        self.assertEqual(provenance["git_commit"], "0123456")

        llm_metadata = json.loads(llm_attrs["metadata"])["agent_strace"]
        self.assertEqual(
            llm_metadata["request.provenance"]["context_stream"],
            "primary",
        )
        self.assertEqual(
            llm_metadata["response.provenance"]["source_uuid"],
            "upstream-response-1",
        )

    def test_error_span_preserves_exception_and_tool_input(self):
        error = next(
            span
            for span in _spans(self.payload)
            if span.get("status", {}).get("code") == 2
            and _attributes(span).get("tool.name") == "Bash"
        )
        attrs = _attributes(error)
        self.assertEqual(attrs["gen_ai.tool.call.id"], "tool-bash-call")
        self.assertEqual(attrs["input.value"], '{"command":"false"}')
        self.assertEqual(error["events"][0]["name"], "exception")
        exception_attrs = _attributes(error["events"][0])
        self.assertEqual(exception_attrs["exception.type"], "ShellError")
        self.assertEqual(exception_attrs["exception.message"], "command failed")


class TestOpenInferenceMappingRecipe(unittest.TestCase):
    def setUp(self):
        meta, events = _load_source()
        self.payload = session_to_otlp_genai(
            meta,
            events,
            service_name="agent-trace-conformance",
        )
        self.mapping = json.loads(MAPPING_FIXTURE.read_text(encoding="utf-8"))

    def test_span_kind_recipe_matches_payload(self):
        by_scope = {
            "root": _span_for_kind(self.payload, "AGENT"),
            "llm": _span_for_kind(self.payload, "LLM"),
            "tool": _span_for_kind(self.payload, "TOOL"),
        }
        for recipe in self.mapping["span_kinds"]:
            with self.subTest(scope=recipe["scope"]):
                span = by_scope[recipe["scope"]]
                attrs = _attributes(span)
                self.assertEqual(span["kind"], recipe["otlp_span_kind"])
                self.assertEqual(
                    attrs["openinference.span.kind"],
                    recipe["openinference_span_kind"],
                )

    def test_attribute_recipe_names_existing_source_and_destination_fields(self):
        by_scope = {
            "root": _span_for_kind(self.payload, "AGENT"),
            "llm": _span_for_kind(self.payload, "LLM"),
            "tool": _span_for_kind(self.payload, "TOOL"),
        }
        for recipe in self.mapping["attribute_mappings"]:
            with self.subTest(
                scope=recipe["scope"],
                source=recipe["source_attribute"],
                destination=recipe["destination_attribute"],
            ):
                attrs = _attributes(by_scope[recipe["scope"]])
                source = recipe["source_attribute"]
                if source.endswith(".*"):
                    prefix = source[:-1]
                    self.assertTrue(any(key.startswith(prefix) for key in attrs))
                else:
                    self.assertIn(source, attrs)
                self.assertIn(recipe["destination_attribute"], attrs)
                self.assertIn(
                    recipe["representation"],
                    {"renamed", "packed_json"},
                )
                self.assertTrue(recipe["loss"])

    def test_recipe_declares_non_lossless_fields(self):
        losses = {
            item["field"]
            for item in self.mapping["not_losslessly_represented"]
        }
        self.assertEqual(
            losses,
            {
                "multi-turn message events",
                "gen_ai.response.finish_reasons",
                "provider capture blind spots",
                "redacted values",
                "complete hash-chain verification",
            },
        )


class TestBackendRecipeFixtures(unittest.TestCase):
    def test_recipes_use_existing_http_json_exporters(self):
        recipes = json.loads(BACKEND_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(recipes["transport"], "otlp_http_json")
        self.assertNotIn("grpc", json.dumps(recipes).lower())
        self.assertEqual(
            {backend["id"] for backend in recipes["backends"]},
            {"phoenix-openinference", "langfuse", "honeycomb", "datadog"},
        )
        self.assertEqual(
            {backend["exporter"] for backend in recipes["backends"]},
            {"otlp-genai"},
        )
        for fixture in ("otlp_fixture", "langfuse_fixture"):
            self.assertTrue((ROOT / recipes[fixture]).is_file())

    def test_langfuse_fixture_matches_existing_exporter(self):
        meta, events = _load_source()
        actual = {
            "trace": _session_to_langfuse_trace(meta, events),
            "observations": _events_to_langfuse_observations(
                meta.session_id,
                events,
            ),
        }
        expected = json.loads(LANGFUSE_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
