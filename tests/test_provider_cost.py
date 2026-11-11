"""Tests for the provider cost dashboard (issue #208)."""

import argparse
import csv
import datetime
import io
import tempfile
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
