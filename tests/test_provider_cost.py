"""Tests for the provider cost dashboard (issue #208)."""

import argparse
import csv
import datetime
import io
import tempfile
import time
import unittest
from unittest.mock import patch

from agent_trace.cli import build_parser
from agent_trace.cost import (
    _session_model_usage,
    LivePricingUnavailable,
    PRICING,
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
    is_bedrock_model,
    parse_since,
)
from agent_trace.inflation import _resolve_factor
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.token_budget import _resolve_limit

try:
    import _escapes
except ImportError:
    from tests import _escapes


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

    def test_usage_without_model_is_reported_without_a_price(self):
        store = self.fixture.store
        store.create_session(SessionMeta(session_id="codex", started_at=NOW - 1))
        store.append_event("codex", TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            timestamp=NOW - 1,
            session_id="codex",
            data={"provider": "codex", "input_tokens": 2500, "output_tokens": 30},
        ))
        report = build_provider_cost_report(store, since="7d", now=NOW)
        self.assertEqual(
            [(r.provider, r.model, r.input_tokens, r.output_tokens, r.cost_usd)
             for r in report.records],
            [("Unknown", "", 2500, 30, 0.0)],
        )
        text = io.StringIO()
        format_provider_cost_report(report, text)
        self.assertIn("(not recorded)", text.getvalue())
        rows = io.StringIO()
        format_provider_cost_csv(report, rows)
        self.assertEqual(next(csv.DictReader(io.StringIO(rows.getvalue())))["model"], "")

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


class TestModelUsagePairing(unittest.TestCase):
    """Usage pairing follows the pairing.py LLM rules."""

    @staticmethod
    def _event(event_type, event_id, parent_id="", **data):
        return TraceEvent(
            event_type=event_type, event_id=event_id, parent_id=parent_id, data=data,
        )

    def test_response_naming_a_request_without_model_counts_on_its_own(self):
        # It used to take over the usage sample of another pending request.
        events = [
            self._event(EventType.LLM_REQUEST, "plain"),
            self._event(EventType.LLM_REQUEST, "gpt", model="gpt-4o", input_tokens=10),
            self._event(EventType.LLM_RESPONSE, "r1", "plain", model="gpt-4o",
                        input_tokens=5, output_tokens=5),
            self._event(EventType.LLM_RESPONSE, "r2", "gpt", output_tokens=7),
        ]
        self.assertEqual(_session_model_usage(events), {"gpt-4o": (15, 12)})

    def test_parentless_response_replaces_newest_same_model_request(self):
        events = [
            self._event(EventType.LLM_REQUEST, "old", model="gpt-4o", input_tokens=10),
            self._event(EventType.LLM_REQUEST, "new", model="gpt-4o", input_tokens=20),
            self._event(EventType.LLM_REQUEST, "other", model="claude-opus-4-6", input_tokens=40),
            self._event(EventType.ASSISTANT_RESPONSE, "a1", model="gpt-4o",
                        input_tokens=25, output_tokens=3),
        ]
        self.assertEqual(
            _session_model_usage(events),
            {"gpt-4o": (35, 3), "claude-opus-4-6": (40, 0)},
        )

    def test_newer_request_without_model_does_not_take_same_model_response(self):
        # The response used to pair with "task", so it counted on its own and
        # the input tokens of "gpt" were counted a second time.
        events = [
            self._event(EventType.LLM_REQUEST, "gpt", model="gpt-4o", input_tokens=10),
            self._event(EventType.LLM_REQUEST, "task"),
            self._event(EventType.LLM_RESPONSE, "r1", model="gpt-4o",
                        input_tokens=30, output_tokens=5),
        ]
        self.assertEqual(_session_model_usage(events), {"gpt-4o": (30, 5)})

    def test_duplicate_response_counts_on_its_own(self):
        events = [
            self._event(EventType.LLM_REQUEST, "a", model="gpt-4o", input_tokens=10),
            self._event(EventType.LLM_RESPONSE, "r1", "a", output_tokens=4),
            self._event(EventType.LLM_RESPONSE, "r2", "a", model="gpt-4o", output_tokens=6),
        ]
        self.assertEqual(_session_model_usage(events), {"gpt-4o": (10, 10)})

    def test_unrequested_response_without_model_reports_its_usage(self):
        # A Codex model request whose model was not reported.
        events = [
            self._event(EventType.LLM_RESPONSE, "r1", model="gpt-5-codex",
                        input_tokens=1200, output_tokens=10),
            self._event(EventType.LLM_RESPONSE, "r2", input_tokens=2500, output_tokens=30),
        ]
        self.assertEqual(
            _session_model_usage(events),
            {"gpt-5-codex": (1200, 10), "": (2500, 30)},
        )

    def test_responses_without_model_or_usage_stay_out(self):
        events = [
            self._event(EventType.LLM_RESPONSE, "r1", status="failed", token_usage=None),
            self._event(EventType.ASSISTANT_RESPONSE, "a1", input_tokens=5, output_tokens=5),
            self._event(EventType.LLM_REQUEST, "plain"),
            self._event(EventType.LLM_RESPONSE, "r2", "plain", input_tokens=5, output_tokens=5),
        ]
        self.assertEqual(_session_model_usage(events), {})


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
        # The first six columns are stable; the cache columns are appended.
        self.assertEqual(
            list(rows[0]),
            [
                "provider", "model", "session_id", "input_tokens", "output_tokens", "cost_usd",
                "cache_read_tokens", "cache_write_tokens",
            ],
        )
        self.assertEqual(rows[0]["session_id"], "csv-session")
        self.assertEqual(rows[0]["input_tokens"], "1000000")
        self.assertEqual(rows[0]["cache_read_tokens"], "0")
        self.assertEqual(rows[0]["cache_write_tokens"], "0")

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


def _record(model, response_data, pricing_table=None):
    """Price one request/response pair and return its only report row."""
    fixture = ProviderCostStore()
    try:
        fixture.add_session("cache-session", NOW - 60, model, response_data=response_data)
        kwargs = {} if pricing_table is None else {"pricing_table": pricing_table}
        report = build_provider_cost_report(fixture.store, since="7d", now=NOW, **kwargs)
    finally:
        fixture.close()
    [record] = report.records
    return record


class TestCachedTokenPricing(unittest.TestCase):
    """Cache reads and writes are billed at the provider's own rates.

    Expected costs are worked out by hand from the snapshot rates (USD per
    1M tokens), so a wrong rate fails as well as a wrong token split.
    """

    def assertRecord(self, record, tokens, cost):
        self.assertEqual(
            (record.input_tokens, record.output_tokens,
             record.cache_read_tokens, record.cache_write_tokens),
            tokens,
        )
        self.assertAlmostEqual(record.cost_usd, cost, places=12)

    def test_openai_chat_cached_tokens_are_part_of_prompt_tokens(self):
        # gpt-4o: input 2.50, cached input 1.25, output 10.
        record = _record("gpt-4o", {"usage": {
            "prompt_tokens": 2000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 1500},
        }})
        self.assertRecord(record, (2000, 100, 1500, 0),
                          (500 * 2.50 + 1500 * 1.25 + 100 * 10) / 1e6)

    def test_openai_responses_cache_writes_by_lifetime(self):
        # gpt-6.1-sol: input 2, cached 0.10, 30m write 2.50, 12h write 5,
        # output 10.  Reasoning tokens are already part of output_tokens.
        record = _record("gpt-6.1-sol", {"usage": {
            "input_tokens": 10_000,
            "output_tokens": 500,
            "output_tokens_details": {"reasoning_tokens": 300},
            "input_tokens_details": {
                "cached_tokens": 6000,
                "cache_write_tokens": 3000,
                "cache_write_tokens_details": {
                    "ttl_30m_tokens": 1000, "ttl_12h_tokens": 2000,
                },
            },
        }})
        self.assertRecord(record, (10_000, 500, 6000, 3000),
                          (1000 * 2 + 6000 * 0.10 + 1000 * 2.50 + 2000 * 5 + 500 * 10) / 1e6)

    def test_anthropic_cache_tokens_are_added_to_input(self):
        # claude-sonnet-4-6: input 3, cache read 0.30, 5m write 3.75,
        # 1h write 6, output 15.
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 3000,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000,
            },
        }
        expected = (100 * 3 + 8000 * 0.30 + 1000 * 3.75 + 2000 * 6 + 50 * 15) / 1e6
        for name, data in (("nested", {"usage": usage}), ("top level", usage)):
            with self.subTest(name):
                record = _record("claude-sonnet-4-6", data)
                self.assertRecord(record, (11_100, 50, 8000, 3000), expected)

    def test_anthropic_cache_write_without_lifetime_split_uses_default_rate(self):
        record = _record("claude-sonnet-4-6", {"usage": {
            "input_tokens": 100, "output_tokens": 0, "cache_creation_input_tokens": 1000,
        }})
        self.assertRecord(record, (1100, 0, 0, 1000), (100 * 3 + 1000 * 3.75) / 1e6)

    def test_bedrock_converse_cache_tokens_are_added_to_input(self):
        # Nova Pro on demand: input 0.80, cache read 0.20, cache write 0,
        # output 3.20.
        record = _record("us.amazon.nova-pro-v1:0", {"usage": {
            "inputTokens": 1000,
            "outputTokens": 100,
            "totalTokens": 7100,
            "cacheReadInputTokens": 4000,
            "cacheWriteInputTokens": 2000,
        }})
        self.assertRecord(record, (7000, 100, 4000, 2000),
                          (1000 * 0.80 + 4000 * 0.20 + 100 * 3.20) / 1e6)

    def test_gemini_cached_content_thoughts_and_tool_prompts(self):
        # gemini-2.5-pro: input 1.25, context caching 0.125, output 10
        # (thinking tokens are billed as output).
        expected = (2050 * 1.25 + 8000 * 0.125 + 500 * 10) / 1e6
        shapes = {
            "camelCase": {"usageMetadata": {
                "promptTokenCount": 10_000,
                "cachedContentTokenCount": 8000,
                "candidatesTokenCount": 200,
                "thoughtsTokenCount": 300,
                "toolUsePromptTokenCount": 50,
                "totalTokenCount": 10_550,
            }},
            "snake_case": {"usage_metadata": {
                "prompt_token_count": 10_000,
                "cached_content_token_count": 8000,
                "candidates_token_count": 200,
                "thoughts_token_count": 300,
                "tool_use_prompt_token_count": 50,
            }},
        }
        for name, data in shapes.items():
            with self.subTest(name):
                record = _record("gemini-2.5-pro", data)
                self.assertRecord(record, (10_050, 500, 8000, 0), expected)

    def test_codex_token_usage_mirrored_at_top_level(self):
        # The Codex hooks copy input/output to the top level and keep the
        # cache split in token_usage.  gpt-5.3-codex: input 1.75, cached
        # 0.175, output 14.
        record = _record("gpt-5.3-codex", {
            "input_tokens": 1010,
            "output_tokens": 6,
            "total_tokens": 1016,
            "token_usage": {
                "input_tokens": 1010,
                "cached_input_tokens": 512,
                "cache_write_input_tokens": 0,
                "output_tokens": 6,
                "reasoning_output_tokens": 0,
                "total_tokens": 1016,
            },
        })
        self.assertRecord(record, (1010, 6, 512, 0),
                          (498 * 1.75 + 512 * 0.175 + 6 * 14) / 1e6)

    def test_codex_cache_writes_are_part_of_input_tokens(self):
        record = _record("gpt-6.1-sol", {"token_usage": {
            "input_tokens": 2000,
            "cached_input_tokens": 500,
            "cache_write_input_tokens": 1000,
            "output_tokens": 10,
        }})
        self.assertRecord(record, (2000, 10, 500, 1000),
                          (500 * 2 + 500 * 0.10 + 1000 * 2.50 + 10 * 10) / 1e6)

    def test_model_without_cache_rates_bills_cached_tokens_at_input_rate(self):
        # Neither model has a published cache rate in the snapshot.
        record = _record("gpt-4.1-nano", {"usage": {
            "prompt_tokens": 2000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 1500},
        }})
        self.assertRecord(record, (2000, 100, 1500, 0), (2000 * 0.10 + 100 * 0.40) / 1e6)
        record = _record("claude-3-5-sonnet", {"usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 3000,
        }})
        self.assertRecord(record, (11_100, 50, 8000, 3000), (11_100 * 3 + 50 * 15) / 1e6)

    def test_pricing_table_without_cache_rates_bills_cache_at_input_rate(self):
        record = _record(
            "gpt-4o",
            {"usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 1500},
            }},
            pricing_table={"gpt-4o": {"input": 1.0, "output": 2.0}},
        )
        self.assertRecord(record, (2000, 100, 1500, 0), (2000 * 1.0 + 100 * 2.0) / 1e6)

    def test_prompt_tokens_are_not_increased_by_anthropic_cache_keys(self):
        # LiteLLM reports an OpenAI-style prompt_tokens that already holds
        # the cache reads, and repeats them under the Anthropic key.
        record = _record("claude-sonnet-4-6", {"usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 800},
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
        }})
        self.assertRecord(record, (1000, 10, 800, 0),
                          (200 * 3 + 800 * 0.30 + 10 * 15) / 1e6)

    def test_top_level_and_nested_usage_are_not_both_counted(self):
        usage = {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 8000}
        record = _record("claude-sonnet-4-6", {**usage, "usage": dict(usage)})
        self.assertRecord(record, (8100, 50, 8000, 0),
                          (100 * 3 + 8000 * 0.30 + 50 * 15) / 1e6)

    def test_cached_tokens_never_exceed_the_input_that_includes_them(self):
        record = _record("gpt-4o", {"usage": {
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 50},
        }})
        self.assertRecord(record, (100, 0, 100, 0), 100 * 1.25 / 1e6)

    def test_summaries_and_csv_carry_cache_tokens(self):
        fixture = ProviderCostStore()
        self.addCleanup(fixture.close)
        usage = {"usage": {
            "prompt_tokens": 2000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 1500, "cache_write_tokens": 200},
        }}
        fixture.add_session("a", NOW - 60, "gpt-4o", response_data=usage)
        fixture.add_session("b", NOW - 30, "gpt-4o", response_data=usage)
        report = build_provider_cost_report(fixture.store, since="7d", now=NOW)
        [summary] = report.summaries
        self.assertEqual((summary.cache_read_tokens, summary.cache_write_tokens), (3000, 400))

        output = io.StringIO()
        format_provider_cost_csv(report, output)
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(
            [(row["input_tokens"], row["cache_read_tokens"], row["cache_write_tokens"])
             for row in rows],
            [("2000", "1500", "200"), ("2000", "1500", "200")],
        )


class TestPricingSnapshotModels(unittest.TestCase):
    def test_current_model_families_resolve_to_their_own_rates(self):
        # model id: (snapshot key, provider, input, output, cache read)
        cases = {
            "claude-opus-4-8": ("claude-opus-4-8", "Anthropic", 5, 25, 0.50),
            "claude-opus-4-20250514": ("claude-opus-4", "Anthropic", 15, 75, 1.50),
            "claude-sonnet-5-5": ("claude-sonnet-5-5", "Anthropic", 2, 10, 0.10),
            "claude-sonnet-5": ("claude-sonnet-5", "Anthropic", 2, 10, 0.20),
            "claude-haiku-4-5-20251001": ("claude-haiku-4-5", "Anthropic", 1, 5, 0.10),
            "gpt-5.5": ("gpt-5.5", "OpenAI", 5, 30, 0.50),
            "gpt-5.5-pro": ("gpt-5.5-pro", "OpenAI", 30, 180, None),
            "gpt-5-mini-2025-08-07": ("gpt-5-mini", "OpenAI", 0.25, 2, 0.025),
            "gpt-5": ("gpt-5", "OpenAI", 1.25, 10, 0.125),
            "gpt-5.3-codex": ("gpt-5.3-codex", "OpenAI", 1.75, 14, 0.175),
            "openai/gpt-6.1-sol": ("gpt-6.1-sol", "OpenAI", 2, 10, 0.10),
            "models/gemini-4-carbon": ("gemini-4-carbon", "Gemini", 4, 20, 0.20),
            "gemini-2.5-flash-lite": ("gemini-2.5-flash-lite", "Gemini", 0.10, 0.40, 0.01),
            "us.amazon.nova-lite-v1:0": ("amazon.nova-lite-v1", "AWS Bedrock", 0.06, 0.24, 0.015),
        }
        for model, (key, provider, inp, out, read) in cases.items():
            with self.subTest(model=model):
                price = get_model_pricing(model)
                self.assertIsNotNone(price)
                self.assertEqual(
                    (price.model_key, price.provider, price.input_per_million,
                     price.output_per_million, price.cache_read_per_million),
                    (key, provider, inp, out, read),
                )

    def test_cache_write_rates_by_lifetime(self):
        price = get_model_pricing("claude-sonnet-4-6")
        self.assertEqual(
            (price.cache_write_per_million, price.cache_write_long_per_million), (3.75, 6.0),
        )
        price = get_model_pricing("gpt-6.1-sol")
        self.assertEqual(
            (price.cache_write_per_million, price.cache_write_long_per_million), (2.5, 5.0),
        )

    def test_every_snapshot_model_resolves_to_itself(self):
        short_names = {"sonnet", "opus", "haiku", "gpt4", "gpt4o"}
        for key, rates in PRICING.items():
            if key in short_names or key == "custom":
                continue
            with self.subTest(key=key):
                price = get_model_pricing(key)
                self.assertIsNotNone(price)
                self.assertEqual(price.model_key, key)
                self.assertNotEqual(price.provider, "Unknown")
                self.assertEqual(price.provider, infer_provider(key))
                self.assertGreater(price.input_per_million, 0)
                self.assertGreater(price.output_per_million, 0)
                if price.cache_read_per_million is not None:
                    self.assertLess(price.cache_read_per_million, price.input_per_million)
                self.assertLessEqual(
                    set(rates), {"input", "output", "cache_read", "cache_write", "cache_write_long"},
                )


# Claude Opus 4.1 and earlier keep the list rate in every Bedrock region.
NO_PREMIUM_KEYS = {
    "claude-opus-4-1", "claude-opus-4", "claude-sonnet-4", "claude-3-7-sonnet",
    "claude-3-5-sonnet", "claude-3-haiku", "claude-3-opus",
}
SONNET_45 = "claude-sonnet-4-5-20250929-v1:0"
US_SONNET_45 = "us.anthropic." + SONNET_45
PROFILE_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/" + US_SONNET_45
FOUNDATION_ARN = "arn:aws:bedrock:us-east-1::foundation-model/anthropic." + SONNET_45
APP_PROFILE_ARN = (
    "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4e5f6"
)
SONNET_45_LIST = (3, 15, 0.3, 3.75, 6)
SONNET_45_REGIONAL = (3.3, 16.5, 0.33, 4.125, 6.6)


def _rates(price):
    return (
        price.input_per_million, price.output_per_million, price.cache_read_per_million,
        price.cache_write_per_million, price.cache_write_long_per_million,
    )


class TestBedrockClaudeModelIds(unittest.TestCase):
    """Bedrock names a Claude model by an inference profile, a base model ID
    or an ARN. Each one gets the Claude price under AWS Bedrock. A geography
    profile of Claude Sonnet 4.5, Haiku 4.5, Opus 4.5 or a later model costs
    1.1 times the list rate."""

    def assertPrice(self, model, key, rates, pricing_table=PRICING):
        price = get_model_pricing(model, pricing_table)
        self.assertIsNotNone(price)
        self.assertEqual((price.model_key, price.provider), (key, "AWS Bedrock"))
        names = ("input", "output", "cache read", "cache write", "long cache write")
        for name, got, want in zip(names, _rates(price), rates):
            with self.subTest(rate=name):
                if want is None:
                    self.assertIsNone(got)
                else:
                    self.assertAlmostEqual(got, want)

    def test_each_geography_profile_costs_ten_percent_more(self):
        for geography in ("us", "eu", "apac", "jp", "au", "us-gov", "ca", "in"):
            with self.subTest(geography=geography):
                self.assertPrice(
                    f"{geography}.anthropic.{SONNET_45}", "claude-sonnet-4-5", SONNET_45_REGIONAL,
                )

    def test_ca_and_in_profiles_of_opus_4_1_keep_the_list_rate(self):
        for geography in ("ca", "in"):
            with self.subTest(geography=geography):
                self.assertPrice(
                    f"{geography}.anthropic.claude-opus-4-1-20250805-v1:0", "claude-opus-4-1",
                    _rates(get_model_pricing("claude-opus-4-1")),
                )

    def test_ca_nova_lite_profile_gets_the_nova_lite_price(self):
        nova_lite = _rates(get_model_pricing("amazon.nova-lite-v1:0"))
        self.assertEqual(nova_lite[:2], (0.06, 0.24))
        for model in ("ca.amazon.nova-lite-v1:0", "  CA.Amazon.Nova-Lite-V1:0\n"):
            with self.subTest(model=model):
                self.assertPrice(model, "amazon.nova-lite-v1", nova_lite)

    def test_geography_profile_inside_a_prefix_or_arn_costs_ten_percent_more(self):
        for model in (
            "bedrock/" + US_SONNET_45,
            "aws/bedrock/" + US_SONNET_45,
            PROFILE_ARN,
            "bedrock/" + PROFILE_ARN,
            "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:inference-profile/us-gov.anthropic."
            + SONNET_45,
        ):
            with self.subTest(model=model):
                self.assertPrice(model, "claude-sonnet-4-5", SONNET_45_REGIONAL)

    def test_id_with_no_geography_uses_the_list_rate(self):
        # Global profiles have no premium. A base model ID or a foundation-model
        # ARN does not show whether the call went to a regional endpoint.
        for model in (
            "global.anthropic." + SONNET_45,
            "bedrock/global.anthropic." + SONNET_45,
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic."
            + SONNET_45,
            "anthropic." + SONNET_45,
            "bedrock/anthropic." + SONNET_45,
            FOUNDATION_ARN,
            "aws/bedrock/" + FOUNDATION_ARN,
        ):
            with self.subTest(model=model):
                self.assertPrice(model, "claude-sonnet-4-5", SONNET_45_LIST)

    def test_claude_in_amazon_bedrock_id_uses_the_list_rate(self):
        self.assertPrice("anthropic.claude-opus-5-5", "claude-opus-5-5", (4, 20, 0.2, 5, 8))

    def test_every_later_claude_key_gets_the_premium(self):
        for key in PRICING:
            if not key.startswith("claude-") or key in NO_PREMIUM_KEYS:
                continue
            with self.subTest(key=key):
                expected = tuple(
                    None if rate is None else rate * 1.1 for rate in _rates(get_model_pricing(key))
                )
                self.assertPrice(f"us.anthropic.{key}-v1:0", key, expected)

    def test_opus_4_1_and_earlier_keep_the_list_rate(self):
        for key in sorted(NO_PREMIUM_KEYS):
            with self.subTest(key=key):
                self.assertPrice(
                    f"us.anthropic.{key}-v1:0", key, _rates(get_model_pricing(key)),
                )
        self.assertPrice(
            "eu.anthropic.claude-3-7-sonnet-20250219-v1:0", "claude-3-7-sonnet",
            (3, 15, None, None, None),
        )

    def test_nova_profiles_have_no_premium(self):
        nova = _rates(get_model_pricing("amazon.nova-pro-v1:0"))
        self.assertEqual(nova[:2], (0.8, 3.2))
        for geography in ("us", "jp", "au"):
            model = f"{geography}.amazon.nova-pro-v1:0"
            with self.subTest(model=model):
                self.assertPrice(model, "amazon.nova-pro-v1", nova)

    def test_premium_applies_to_a_custom_pricing_table(self):
        table = {"claude-sonnet-4-5": {"input": 10, "output": 20}}
        self.assertPrice(US_SONNET_45, "claude-sonnet-4-5", (11, 22, None, None, None), table)
        self.assertPrice(
            "global.anthropic." + SONNET_45, "claude-sonnet-4-5", (10, 20, None, None, None), table,
        )
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        record = _record(US_SONNET_45, usage, pricing_table=table)
        self.assertAlmostEqual(record.cost_usd, 33.0)

    def test_report_prices_cache_tokens_at_the_premium_rates(self):
        # Input 3.3, output 16.5, cache read 0.33, 5-minute write 4.125 and
        # 1-hour write 6.6 dollars per 1M tokens.
        record = _record(US_SONNET_45, {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
            "cache_creation_input_tokens": 2_000_000,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1_000_000,
                "ephemeral_1h_input_tokens": 1_000_000,
            },
        })
        self.assertEqual(record.provider, "AWS Bedrock")
        self.assertAlmostEqual(record.cost_usd, 3.3 + 16.5 + 0.33 + 4.125 + 6.6)

    def test_arn_reads_the_model_id_after_the_resource_type(self):
        cases = {
            "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0":
                ("claude-3-5-sonnet", 3, 15),
            "arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/"
            "anthropic.claude-3-5-sonnet-20240620-v1:0": ("claude-3-5-sonnet", 3, 15),
            "arn:aws-cn:bedrock:cn-north-1::foundation-model/"
            "anthropic.claude-3-haiku-20240307-v1:0": ("claude-3-haiku", 0.25, 1.25),
            "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0":
                ("amazon.nova-pro-v1", 0.8, 3.2),
        }
        for model, (key, inp, out) in cases.items():
            with self.subTest(model=model):
                price = get_model_pricing(model)
                self.assertIsNotNone(price)
                self.assertEqual((price.model_key, price.provider), (key, "AWS Bedrock"))
                self.assertAlmostEqual(price.input_per_million, inp)
                self.assertAlmostEqual(price.output_per_million, out)

    def test_arn_that_names_no_model_gets_no_price(self):
        for model in (
            APP_PROFILE_ARN,
            "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/a1b2c3d4e5f6",
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/"
            "anthropic.claude-3-haiku-20240307-v1:0:200k/a1b2c3d4e5f6",
        ):
            with self.subTest(model=model):
                self.assertEqual(infer_provider(model), "AWS Bedrock")
                self.assertIsNone(get_model_pricing(model))

    def test_bedrock_ids_are_listed_under_aws_bedrock(self):
        for model in (
            "jp.anthropic." + SONNET_45,
            "au.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "jp.amazon.nova-pro-v1:0",
            "anthropic.claude-opus-5-5",
            PROFILE_ARN,
            FOUNDATION_ARN,
            "arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/"
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
        ):
            with self.subTest(model=model):
                self.assertEqual(infer_provider(model), "AWS Bedrock")
        for model in ("claude-sonnet-4-5-20250929", "anthropic/claude-sonnet-4-5"):
            with self.subTest(model=model):
                self.assertEqual(infer_provider(model), "Anthropic")

    def _cost_csv(self, sessions):
        """Return ``cost --breakdown provider --csv`` rows in the shape of *sessions*.

        *sessions* maps a session ID to ``(provider, model, cost_usd)``. Each
        session spends 1M input and 1M output tokens on its model.
        """
        fixture = ProviderCostStore()
        self.addCleanup(fixture.close)
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        for session_id, (_, model, _) in sessions.items():
            fixture.add_session(session_id, time.time() - 60, model, response_data=usage)
        args = build_parser().parse_args([
            "--trace-dir", fixture.temp_dir.name, "cost", "--breakdown", "provider", "--csv",
        ])
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            self.assertEqual(cmd_cost(args), 0)
        rows = csv.DictReader(io.StringIO(stdout.getvalue()))
        return {row["session_id"]: (row["provider"], row["model"], row["cost_usd"]) for row in rows}

    def test_cost_csv_prices_each_bedrock_id(self):
        sessions = {
            "c01direct": ("Anthropic", "claude-sonnet-4-5-20250929", "18.00000000"),
            "c02us": ("AWS Bedrock", US_SONNET_45, "19.80000000"),
            "c03global": ("AWS Bedrock", "global.anthropic." + SONNET_45, "18.00000000"),
            "c04plain": ("AWS Bedrock", "anthropic." + SONNET_45, "18.00000000"),
            "c05mantle": ("AWS Bedrock", "anthropic.claude-opus-5-5", "24.00000000"),
            "c06jp": ("AWS Bedrock", "jp.anthropic." + SONNET_45, "19.80000000"),
            "c07arn": ("AWS Bedrock", PROFILE_ARN, "19.80000000"),
            "c08appprof": ("AWS Bedrock", APP_PROFILE_ARN, "0.00000000"),
            "c09novajp": ("AWS Bedrock", "jp.amazon.nova-pro-v1:0", "4.00000000"),
        }
        self.assertEqual(self._cost_csv(sessions), sessions)

    def test_cost_csv_lists_ca_in_and_new_vendor_ids_under_aws_bedrock(self):
        # Haiku 4.5 lists at $1 input and $5 output per 1M tokens, and an in.
        # profile adds 10%. Nova Lite costs $0.06 and $0.24. DeepSeek has no
        # bundled price.
        sessions = {
            "e1inhaiku": (
                "AWS Bedrock", "in.anthropic.claude-haiku-4-5-20251001-v1:0", "6.60000000",
            ),
            "e2canovalite": ("AWS Bedrock", "ca.amazon.nova-lite-v1:0", "0.30000000"),
            "e3usdeepseek": ("AWS Bedrock", "us.deepseek.r1-v1:0", "0.00000000"),
        }
        self.assertEqual(self._cost_csv(sessions), sessions)


BEDROCK_VENDORS = (
    "ai21", "amazon", "anthropic", "cohere", "deepseek", "luma", "meta", "minimax", "mistral",
    "moonshot", "moonshotai", "nvidia", "openai", "qwen", "stability", "twelvelabs", "writer",
    "xai", "zai",
)
BEDROCK_GEOGRAPHIES = ("us", "us-gov", "eu", "apac", "jp", "au", "ca", "in", "global")


class TestIsBedrockModel(unittest.TestCase):
    """is_bedrock_model decides AWS Bedrock in cost and aws.bedrock in OTLP export."""

    def test_every_vendor_bare_and_after_every_geography(self):
        for vendor in BEDROCK_VENDORS:
            for prefix in ("", *(f"{geography}." for geography in BEDROCK_GEOGRAPHIES)):
                model = f"{prefix}{vendor}.model-v1:0"
                with self.subTest(model=model):
                    self.assertTrue(is_bedrock_model(model))
                    self.assertEqual(infer_provider(model), "AWS Bedrock")

    def test_bedrock_prefixes_and_arns(self):
        for model in (
            "bedrock/custom-model",
            "aws/bedrock/custom-model",
            PROFILE_ARN,
            FOUNDATION_ARN,
            APP_PROFILE_ARN,
            "arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/anthropic." + SONNET_45,
            "arn:aws-cn:bedrock:cn-north-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
        ):
            with self.subTest(model=model):
                self.assertTrue(is_bedrock_model(model))

    def test_ignores_case_and_surrounding_whitespace(self):
        # infer_provider strips and lowers the name first, so these call
        # is_bedrock_model directly.
        for model in (
            "IN.ANTHROPIC.CLAUDE-HAIKU-4-5-20251001-V1:0",
            "Bedrock/Custom-Model",
            APP_PROFILE_ARN.upper(),
            "  ca.amazon.nova-lite-v1:0\n",
            "\tbedrock/custom-model ",
            " " + APP_PROFILE_ARN,
        ):
            with self.subTest(model=model):
                self.assertTrue(is_bedrock_model(model))

    def test_other_names_are_not_bedrock(self):
        for model in (
            "google.gemma-3-27b-it", "google.gemini-2.5-pro", "in.foo-bar", "ca.acme.model-v1:0",
            "xai/grok-4", "x.ai", "mistral-large-latest", "deepseek-chat", "claude-sonnet-4-5",
            "", None,
        ):
            with self.subTest(model=model):
                self.assertFalse(is_bedrock_model(model))
                self.assertNotEqual(infer_provider(model), "AWS Bedrock")

    def test_new_vendor_ids_are_listed_under_aws_bedrock_with_no_price(self):
        for model in (
            "deepseek.r1-v1:0", "us.deepseek.r1-v1:0", "deepseek.v3.2", "luma.ray-v2:0",
            "minimax.minimax-m2", "moonshot.kimi-k2-thinking", "moonshotai.kimi-k2.5",
            "global.moonshotai.kimi-k3", "nvidia.nemotron-nano-3-30b",
            "us-gov.nvidia.nemotron-nano-12b-v2", "stability.sd3-5-large-v1:0",
            "twelvelabs.pegasus-1-2-v1:0", "apac.twelvelabs.pegasus-1-2-v1:0", "xai.grok-4.3",
            "us.xai.grok-4.7", "zai.glm-4.7", "global.zai.glm-5.3",
            # ca. and in. IDs of a vendor with no bundled price
            "in.openai.gpt-5.6-luna", "in.moonshotai.kimi-k3", "ca.meta.llama3-1-70b-instruct-v1:0",
        ):
            with self.subTest(model=model):
                self.assertEqual(infer_provider(model), "AWS Bedrock")
                self.assertIsNone(get_model_pricing(model))


def _entry(model):
    """Return the price key, provider and rates, context limit and tokenizer
    factor that *model* resolves to."""
    price = get_model_pricing(model)
    priced = None if price is None else (price.model_key, price.provider, _rates(price))
    return priced, _resolve_limit(model), _resolve_factor(model)


# A Claude name with a dot in its version, and its dash form.
DOT_DASH_PAIRS = (
    ("claude-sonnet-4.5", "claude-sonnet-4-5"),
    ("claude-sonnet-4.5-20250929", "claude-sonnet-4-5-20250929"),
    ("claude-haiku-4.5", "claude-haiku-4-5"),
    ("claude-opus-4.1", "claude-opus-4-1"),
    ("claude-opus-4.7", "claude-opus-4-7"),
    ("claude-3.5-sonnet", "claude-3-5-sonnet"),
    ("claude-3.7-sonnet", "claude-3-7-sonnet"),
    ("claude-opus-5.5", "claude-opus-5-5"),
)
NAME_FORMS = (
    "{}",
    "anthropic/{}",
    "anthropic/{}:beta",
    "anthropic/{}:thinking",
    "bedrock/{}",
    "anthropic.{}",
    "anthropic.{}-v1:0",
    "us.anthropic.{}-v1:0",
    "eu.anthropic.{}-v1:0",
    "global.anthropic.{}-v1:0",
    "bedrock/us.anthropic.{}-v1:0",
    "aws/bedrock/anthropic.{}-v1:0",
    "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.{}-v1:0",
    "arn:aws:bedrock:us-east-1::foundation-model/anthropic.{}-v1:0",
)


class TestClaudeVersionDot(unittest.TestCase):
    """Copilot and OpenRouter write a Claude version with a dot, as in
    claude-sonnet-4.5. Such a name resolves as its dash form does."""

    def test_dot_name_resolves_as_its_dash_form(self):
        for dot, dash in DOT_DASH_PAIRS:
            for form in NAME_FORMS:
                dot_name, dash_name = form.format(dot), form.format(dash)
                with self.subTest(model=dot_name):
                    self.assertIsNotNone(get_model_pricing(dash_name))
                    self.assertEqual(_entry(dot_name), _entry(dash_name))

    def test_dot_names_get_the_dash_key_and_rates(self):
        # A us. or eu. profile of Sonnet 4.5 or Haiku 4.5 costs 1.1 times the
        # list rate. Opus 4.1 has no premium.
        cases = {
            "claude-sonnet-4.5": ("claude-sonnet-4-5", "Anthropic", 3, 15, 200_000, 1.0),
            "Claude-Sonnet-4.5": ("claude-sonnet-4-5", "Anthropic", 3, 15, 200_000, 1.0),
            "claude-haiku-4.5": ("claude-haiku-4-5", "Anthropic", 1, 5, 200_000, None),
            "claude-opus-4.1": ("claude-opus-4-1", "Anthropic", 15, 75, 200_000, None),
            "claude-opus-4.7": ("claude-opus-4-7", "Anthropic", 5, 25, 1_000_000, 1.38),
            "claude-3.7-sonnet-thought": ("claude-3-7-sonnet", "Anthropic", 3, 15, 200_000, None),
            "anthropic/claude-3.5-sonnet:beta":
                ("claude-3-5-sonnet", "Anthropic", 3, 15, 200_000, None),
            "anthropic/claude-3.7-sonnet:thinking":
                ("claude-3-7-sonnet", "Anthropic", 3, 15, 200_000, None),
            "anthropic.claude-sonnet-4.5":
                ("claude-sonnet-4-5", "AWS Bedrock", 3, 15, 200_000, 1.0),
            "us.anthropic.claude-sonnet-4.5":
                ("claude-sonnet-4-5", "AWS Bedrock", 3.3, 16.5, 200_000, 1.0),
            "us.anthropic.claude-opus-4.1":
                ("claude-opus-4-1", "AWS Bedrock", 15, 75, 200_000, None),
            "bedrock/eu.anthropic.claude-haiku-4.5":
                ("claude-haiku-4-5", "AWS Bedrock", 1.1, 5.5, 200_000, None),
        }
        for model, (key, provider, inp, out, limit, factor) in cases.items():
            with self.subTest(model=model):
                price = get_model_pricing(model)
                self.assertIsNotNone(price)
                self.assertEqual((price.model_key, price.provider), (key, provider))
                self.assertAlmostEqual(price.input_per_million, inp)
                self.assertAlmostEqual(price.output_per_million, out)
                self.assertEqual((_resolve_limit(model), _resolve_factor(model)), (limit, factor))

    def test_every_dot_between_two_digits_reads_as_a_dash(self):
        # Read at its first dot only, claude-sonnet-4.5.1 would use
        # claude-sonnet-4, which has the same rates but no tokenizer factor.
        for model, dash in (
            ("claude-sonnet-4.5.1", "claude-sonnet-4-5"),
            ("claude-opus-4.5.1", "claude-opus-4-5"),
        ):
            with self.subTest(model=model):
                self.assertEqual(_entry(model), _entry(dash))

    def test_dash_form_replaces_the_dot_form(self):
        # claude-opus-4-5.1 starts with claude-opus-4- too. Tried first, the dot
        # form would use claude-opus-4.
        self.assertEqual(_entry("claude-opus-4-5.1"), _entry("claude-opus-4-5"))

    def test_dash_form_of_a_suffix_finds_the_key_it_starts_with(self):
        # claude-sonnet-4-5x starts with claude-sonnet-4- and not claude-sonnet-4-5-.
        self.assertEqual(_entry("claude-sonnet-4.5x"), _entry("claude-sonnet-4-5x"))
        self.assertEqual(get_model_pricing("claude-sonnet-4.5x").model_key, "claude-sonnet-4")

    def test_non_claude_dot_keys_keep_their_entries(self):
        # The rule reads only claude- names. Read in these, the dot would turn
        # gpt-4.1 into gpt-4 and leave gemini-2.5-pro with no price.
        cases = {
            "gpt-4.1": ("gpt-4.1", None, None),
            "gpt-4.1-mini": ("gpt-4.1-mini", None, None),
            "gpt-5.1": ("gpt-5.1", None, None),
            "gemini-2.5-pro": ("gemini-2.5-pro", None, None),
            "gpt-3.5-turbo": (None, 16_385, None),
            "llama-3.1-70b": (None, 131_072, None),
            "gemini-1.5-pro": (None, 1_048_576, 0.95),
        }
        for model, (key, limit, factor) in cases.items():
            with self.subTest(model=model):
                price = get_model_pricing(model)
                self.assertEqual(None if price is None else price.model_key, key)
                self.assertEqual((_resolve_limit(model), _resolve_factor(model)), (limit, factor))

    def test_other_dots_stay_unread(self):
        # A dot that is not between two digits, a name that does not start with
        # claude-, and a version with no key of its own.
        for model in (
            "claude-sonnet-4..5", "claude.sonnet-4.5", "xclaude-sonnet-4.5",
            "claude-3-5-sonnet.v2", "claude-3.5", "claude-2.1", "claude-instant-1.2",
        ):
            with self.subTest(model=model):
                self.assertEqual(_entry(model), (None, None, None))

    def test_cost_csv_prices_dot_names(self):
        fixture = ProviderCostStore()
        self.addCleanup(fixture.close)
        sessions = {
            "c01dot": ("Anthropic", "claude-sonnet-4.5", "18.00000000"),
            "c02openrouter": ("Anthropic", "anthropic/claude-3.5-sonnet:beta", "18.00000000"),
            "c03opus41": ("Anthropic", "claude-opus-4.1", "90.00000000"),
            "c04dash": ("Anthropic", "claude-sonnet-4-5-20250929", "18.00000000"),
            "c05us": ("AWS Bedrock", "us.anthropic.claude-sonnet-4.5", "19.80000000"),
            "c06usopus41": ("AWS Bedrock", "us.anthropic.claude-opus-4.1", "90.00000000"),
        }
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        for session_id, (_, model, _) in sessions.items():
            fixture.add_session(session_id, time.time() - 60, model, response_data=usage)
        args = build_parser().parse_args([
            "--trace-dir", fixture.temp_dir.name, "cost", "--breakdown", "provider", "--csv",
        ])
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            self.assertEqual(cmd_cost(args), 0)
        rows = list(csv.DictReader(io.StringIO(stdout.getvalue())))
        self.assertEqual(
            {row["session_id"]: (row["provider"], row["model"], row["cost_usd"]) for row in rows},
            sessions,
        )


def _provider_report(provider="OpenAI", model="gpt-4o"):
    return ProviderCostReport(
        records=[ProviderCostRecord(provider, model, "s1", 1000, 100, 1.0)],
        since="30d",
        since_timestamp=NOW - 30 * 86400,
        generated_at=NOW,
    )


def _render_provider(report):
    output = io.StringIO()
    format_provider_cost_report(report, output)
    return output.getvalue()


# (label, kind, build): build(value) returns a report with value in the place
# the label names. The kinds are in tests/_escapes.py.
PROVIDER_COST_CASES = [
    ("provider", "text", lambda v: _provider_report(provider=v)),
    ("model", "text", lambda v: _provider_report(model=v)),
]

# (label, cut, build): the dashboard keeps cut characters of the value the
# label names.
PROVIDER_COST_CUTS = [
    ("model", 44, lambda v: _provider_report(model=v)),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """A model name in a trace can hold escape codes and line breaks. The
    provider dashboard shows them as text. The CSV keeps the raw text, except
    that it writes a lone surrogate as its escape."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, PROVIDER_COST_CASES, _render_provider)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, PROVIDER_COST_CASES, _render_provider)

    def test_value_as_long_as_the_cut_shows_in_full(self):
        _escapes.check_each_cut(self, PROVIDER_COST_CUTS, _render_provider)
        self.assertNotIn("\u2026", _render_provider(_provider_report(model="x" * 34 + "\x07" * 10)))

    def test_sgr_code_shows_as_text(self):
        output = _render_provider(_provider_report(model="a\x1b[8mhidden"))
        self.assertIn(r"a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)

    def _csv_rows(self, fixture):
        args = argparse.Namespace(
            trace_dir=fixture.temp_dir.name,
            session_id=None,
            breakdown="provider",
            since="7d",
            csv=True,
            live_pricing=False,
        )
        stdout = _escapes.strict_utf8_stream()
        with patch("sys.stdout", stdout):
            self.assertEqual(cmd_cost(args), 0)
        return list(csv.DictReader(io.StringIO(_escapes.written_text(stdout), newline="")))

    def test_csv_keeps_the_raw_model_name(self):
        fixture = ProviderCostStore()
        self.addCleanup(fixture.close)
        fixture.add_session(
            "s1", time.time() - 60, _escapes.PAYLOAD,
            response_data={"input_tokens": 1000, "output_tokens": 100},
        )
        rows = self._csv_rows(fixture)
        self.assertEqual([row["model"] for row in rows], [_escapes.PAYLOAD])

    def test_csv_writes_a_lone_surrogate_as_its_escape(self):
        fixture = ProviderCostStore()
        self.addCleanup(fixture.close)
        usage = {"input_tokens": 1000, "output_tokens": 100}
        fixture.add_session("s1", time.time() - 60, "claude-" + _escapes.LONE_SURROGATE, response_data=usage)
        fixture.add_session("s2", time.time() - 60, "gpt-4o " + _escapes.PLAIN, response_data=usage)
        rows = self._csv_rows(fixture)
        self.assertEqual(
            {row["session_id"]: row["model"] for row in rows},
            {"s1": "claude-" + _escapes.LONE_SURROGATE_SHOWN, "s2": "gpt-4o " + _escapes.PLAIN},
        )


if __name__ == "__main__":
    unittest.main()
