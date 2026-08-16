"""Tests for the provider cost dashboard (issue #208)."""

import argparse
import csv
import datetime
import io
import tempfile
import unittest
from unittest.mock import patch

from agent_trace.cost import (
    LivePricingUnavailable,
    PRICING_SNAPSHOT_DATE,
    PRICING_SOURCES,
    ProviderCostRecord,
    ProviderCostReport,
    build_provider_cost_report,
    cmd_cost,
    format_provider_cost_csv,
    format_provider_cost_report,
    get_model_pricing,
    infer_provider,
    parse_since,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


NOW = datetime.datetime(2026, 8, 16, tzinfo=datetime.timezone.utc).timestamp()


class ProviderCostStore:
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.temp_dir.name)

    def close(self):
        self.temp_dir.cleanup()

    def add_session(self, session_id, started_at, model, request_data=None, response_data=None):
        meta = SessionMeta(session_id=session_id, started_at=started_at)
        self.store.create_session(meta)
        request = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            timestamp=started_at,
            session_id=session_id,
            data={"model": model, **(request_data or {})},
        )
        response_payload = {"model": model, **(response_data or {})}
        response = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            timestamp=started_at + 1,
            session_id=session_id,
            parent_id=request.event_id,
            data=response_payload,
        )
        self.store.append_event(session_id, request)
        self.store.append_event(session_id, response)
        return meta


class TestProviderInferenceAndPricing(unittest.TestCase):
    def test_infers_all_supported_providers(self):
        cases = {
            "claude-opus-4-6": "Anthropic",
            "openai/gpt-4o": "OpenAI",
            "o3": "OpenAI",
            "us.amazon.nova-pro-v1:0": "AWS Bedrock",
            "us.anthropic.claude-sonnet-4-v1:0": "AWS Bedrock",
            "models/gemini-2.5-pro": "Gemini",
        }
        for model, provider in cases.items():
            with self.subTest(model=model):
                self.assertEqual(infer_provider(model), provider)

    def test_snapshot_has_a_rate_for_each_provider(self):
        cases = (
            ("claude-opus-4-6-20260801", "Anthropic"),
            ("gpt-4o-2024-11-20", "OpenAI"),
            ("us.amazon.nova-pro-v1:0", "AWS Bedrock"),
            ("models/gemini-2.5-pro-preview", "Gemini"),
        )
        for model, provider in cases:
            with self.subTest(model=model):
                price = get_model_pricing(model)
                self.assertIsNotNone(price)
                self.assertEqual(price.provider, provider)
                self.assertGreater(price.input_per_million, 0)
                self.assertGreater(price.output_per_million, 0)

    def test_unknown_model_does_not_use_a_silent_fallback(self):
        self.assertIsNone(get_model_pricing("vendor/future-model"))

    def test_pricing_snapshot_is_explicitly_versioned(self):
        datetime.date.fromisoformat(PRICING_SNAPSHOT_DATE)
        self.assertEqual(
            set(PRICING_SOURCES), {"Anthropic", "OpenAI", "AWS Bedrock", "Gemini"},
        )


class TestParseSince(unittest.TestCase):
    def test_relative_days(self):
        self.assertEqual(parse_since("30d", now=NOW), NOW - 30 * 86400)

    def test_relative_weeks(self):
        self.assertEqual(parse_since("2w", now=NOW), NOW - 14 * 86400)

    def test_date_is_midnight_utc(self):
        expected = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc).timestamp()
        self.assertEqual(parse_since("2026-06-01", now=NOW), expected)

    def test_invalid_value_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "relative duration"):
            parse_since("last month", now=NOW)


class TestProviderCostReport(unittest.TestCase):
    def setUp(self):
        self.fixture = ProviderCostStore()

    def tearDown(self):
        self.fixture.close()

    def test_aggregates_provider_model_sessions_and_actual_usage(self):
        # Response usage replaces request-side counters for the same call,
        # rather than double-counting the input tokens.
        self.fixture.add_session(
            "anthropic-1",
            NOW - 86400,
            "claude-opus-4-6",
            request_data={"input_tokens": 123},
            response_data={"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )
        self.fixture.add_session(
            "openai-1",
            NOW - 86400,
            "gpt-4o",
            response_data={"usage": {"prompt_tokens": 1000, "completion_tokens": 2000}},
        )
        self.fixture.add_session(
            "bedrock-1",
            NOW - 86400,
            "us.amazon.nova-pro-v1:0",
            response_data={"usage": {"inputTokens": 3000, "outputTokens": 4000}},
        )
        self.fixture.add_session(
            "gemini-1",
            NOW - 86400,
            "models/gemini-2.5-pro",
            response_data={
                "usageMetadata": {"promptTokenCount": 5000, "candidatesTokenCount": 6000}
            },
        )

        report = build_provider_cost_report(self.fixture.store, since="7d", now=NOW)
        self.assertEqual(len(report.records), 4)
        self.assertEqual(
            {row.provider for row in report.summaries},
            {"Anthropic", "OpenAI", "AWS Bedrock", "Gemini"},
        )
        anthropic = next(row for row in report.records if row.provider == "Anthropic")
        self.assertEqual(anthropic.input_tokens, 1_000_000)
        self.assertEqual(anthropic.output_tokens, 1_000_000)
        self.assertAlmostEqual(anthropic.cost_usd, 30.0)

    def test_filters_old_and_future_sessions(self):
        self.fixture.add_session(
            "included", NOW - 2 * 86400, "gpt-4o",
            response_data={"input_tokens": 100, "output_tokens": 100},
        )
        self.fixture.add_session(
            "old", NOW - 31 * 86400, "gpt-4o",
            response_data={"input_tokens": 100, "output_tokens": 100},
        )
        self.fixture.add_session(
            "future", NOW + 86400, "gpt-4o",
            response_data={"input_tokens": 100, "output_tokens": 100},
        )
        report = build_provider_cost_report(self.fixture.store, since="30d", now=NOW)
        self.assertEqual([row.session_id for row in report.records], ["included"])

    def test_groups_multiple_sessions_for_the_same_provider_model(self):
        for session_id in ("one", "two"):
            self.fixture.add_session(
                session_id, NOW - 1, "gpt-4o",
                response_data={"input_tokens": 100, "output_tokens": 200},
            )
        report = build_provider_cost_report(self.fixture.store, since="7d", now=NOW)
        self.assertEqual(len(report.summaries), 1)
        self.assertEqual(report.summaries[0].provider, "OpenAI")
        self.assertEqual(report.summaries[0].sessions, 2)
        self.assertEqual(report.summaries[0].input_tokens, 200)
        self.assertEqual(report.summaries[0].output_tokens, 400)

    def test_unknown_model_is_visible_with_zero_estimated_cost(self):
        self.fixture.add_session(
            "unknown", NOW - 1, "acme-future-1",
            response_data={"input_tokens": 1000, "output_tokens": 2000},
        )
        report = build_provider_cost_report(self.fixture.store, since="7d", now=NOW)
        self.assertEqual(report.records[0].provider, "Unknown")
        self.assertEqual(report.records[0].cost_usd, 0.0)

    def test_default_mode_makes_no_network_call(self):
        self.fixture.add_session(
            "offline", NOW - 1, "gpt-4o",
            response_data={"input_tokens": 100, "output_tokens": 200},
        )
        with patch("socket.create_connection", side_effect=AssertionError("network called")):
            report = build_provider_cost_report(self.fixture.store, since="7d", now=NOW)
        self.assertEqual(len(report.records), 1)

    def test_live_pricing_fails_instead_of_fabricating_rates(self):
        with self.assertRaisesRegex(LivePricingUnavailable, "compatible pricing API"):
            build_provider_cost_report(
                self.fixture.store, since="7d", now=NOW, live_pricing=True,
            )


class TestProviderCostFormatting(unittest.TestCase):
    def setUp(self):
        self.fixture = ProviderCostStore()
        self.fixture.add_session(
            "csv-session", NOW - 1, "gpt-4o",
            response_data={"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )
        self.report = build_provider_cost_report(self.fixture.store, since="7d", now=NOW)

    def tearDown(self):
        self.fixture.close()

    def test_csv_has_exact_raw_columns(self):
        output = io.StringIO()
        format_provider_cost_csv(self.report, output)
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(
            list(rows[0]),
            ["provider", "model", "session_id", "input_tokens", "output_tokens", "cost_usd"],
        )
        self.assertEqual(rows[0]["session_id"], "csv-session")
        self.assertEqual(rows[0]["input_tokens"], "1000000")

    def test_text_marks_rates_as_offline_snapshot(self):
        output = io.StringIO()
        format_provider_cost_report(self.report, output)
        text = output.getvalue()
        self.assertIn("Cost breakdown — last 7 days", text)
        self.assertIn("Offline estimates", text)
        self.assertIn(PRICING_SNAPSHOT_DATE, text)
        self.assertIn("OpenAI", text)

    def test_text_total_counts_unique_sessions_not_model_rows(self):
        report = ProviderCostReport(
            records=[
                ProviderCostRecord("OpenAI", "gpt-4o", "same", 10, 20, 0.1),
                ProviderCostRecord("Anthropic", "claude-sonnet-4", "same", 10, 20, 0.2),
            ],
            since="7d",
            since_timestamp=NOW - 7 * 86400,
            generated_at=NOW,
        )
        output = io.StringIO()
        format_provider_cost_report(report, output)
        total_line = next(line for line in output.getvalue().splitlines() if line.startswith("Total"))
        self.assertEqual(total_line.split()[-3], "1")

    def test_command_handler_uses_provider_dashboard_path(self):
        args = argparse.Namespace(
            trace_dir=self.fixture.temp_dir.name,
            session_id=None,
            breakdown="provider",
            since="7d",
            csv=True,
            live_pricing=False,
        )
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            self.assertEqual(cmd_cost(args), 0)
        self.assertTrue(stdout.getvalue().startswith("provider,model,session_id"))


if __name__ == "__main__":
    unittest.main()
