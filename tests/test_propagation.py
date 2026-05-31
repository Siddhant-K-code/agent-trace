"""Tests for W3C traceparent propagation (issue #142)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.propagation import extract_traceparent, inject_traceparent


class TestInjectTraceparent(unittest.TestCase):
    def test_injects_traceparent_header(self):
        headers = inject_traceparent({}, session_id="abc123def456")
        self.assertIn("traceparent", headers)
        tp = headers["traceparent"]
        parts = tp.split("-")
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "00")          # version
        self.assertEqual(len(parts[1]), 32)        # trace-id
        self.assertEqual(len(parts[2]), 16)        # parent-span-id
        self.assertEqual(parts[3], "01")           # sampled flag

    def test_injects_tracestate_with_session_id(self):
        headers = inject_traceparent({}, session_id="mysession123")
        self.assertIn("tracestate", headers)
        self.assertIn("agent-trace=mysession123", headers["tracestate"])

    def test_reuses_provided_trace_id(self):
        trace_id = "a" * 32
        headers = inject_traceparent({}, session_id="s1", trace_id=trace_id)
        tp = headers["traceparent"]
        self.assertIn(trace_id, tp)

    def test_prepends_to_existing_tracestate(self):
        existing = {"tracestate": "vendor=abc"}
        headers = inject_traceparent(existing, session_id="s1")
        ts = headers["tracestate"]
        self.assertTrue(ts.startswith("agent-trace=s1"))
        self.assertIn("vendor=abc", ts)

    def test_does_not_mutate_input_headers(self):
        original = {"Content-Type": "application/json"}
        inject_traceparent(original, session_id="s1")
        self.assertNotIn("traceparent", original)

    def test_deterministic_trace_id_from_session(self):
        h1 = inject_traceparent({}, session_id="abcdef1234567890abcdef1234567890")
        h2 = inject_traceparent({}, session_id="abcdef1234567890abcdef1234567890")
        self.assertEqual(h1["traceparent"], h2["traceparent"])

    def test_event_id_used_for_span_id(self):
        h1 = inject_traceparent({}, session_id="s1", event_id="aaaa1111bbbb2222")
        h2 = inject_traceparent({}, session_id="s1", event_id="cccc3333dddd4444")
        span1 = h1["traceparent"].split("-")[2]
        span2 = h2["traceparent"].split("-")[2]
        self.assertNotEqual(span1, span2)


class TestExtractTraceparent(unittest.TestCase):
    def _valid_headers(self, session_id="mysession"):
        return {
            "traceparent": f"00-{'a' * 32}-{'b' * 16}-01",
            "tracestate": f"agent-trace={session_id},other=val",
        }

    def test_extracts_valid_traceparent(self):
        ctx = extract_traceparent(self._valid_headers())
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["trace_id"], "a" * 32)
        self.assertEqual(ctx["parent_id"], "b" * 16)
        self.assertEqual(ctx["flags"], "01")

    def test_extracts_session_id_from_tracestate(self):
        ctx = extract_traceparent(self._valid_headers("sess-abc"))
        self.assertEqual(ctx["at_session_id"], "sess-abc")

    def test_returns_none_when_no_header(self):
        ctx = extract_traceparent({"Content-Type": "application/json"})
        self.assertIsNone(ctx)

    def test_returns_none_for_malformed_traceparent(self):
        ctx = extract_traceparent({"traceparent": "not-valid"})
        self.assertIsNone(ctx)

    def test_empty_at_session_id_when_no_tracestate(self):
        headers = {"traceparent": f"00-{'a' * 32}-{'b' * 16}-01"}
        ctx = extract_traceparent(headers)
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["at_session_id"], "")

    def test_case_insensitive_header_lookup(self):
        headers = {
            "Traceparent": f"00-{'c' * 32}-{'d' * 16}-00",
            "Tracestate": "agent-trace=upper-case-session",
        }
        ctx = extract_traceparent(headers)
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["at_session_id"], "upper-case-session")

    def test_roundtrip_inject_then_extract(self):
        injected = inject_traceparent({}, session_id="roundtrip-session-id")
        ctx = extract_traceparent(injected)
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["at_session_id"], "roundtrip-session-id")

    def test_tracestate_with_multiple_vendors(self):
        headers = {
            "traceparent": f"00-{'e' * 32}-{'f' * 16}-01",
            "tracestate": "dd=abc,agent-trace=my-session,other=xyz",
        }
        ctx = extract_traceparent(headers)
        self.assertEqual(ctx["at_session_id"], "my-session")


if __name__ == "__main__":
    unittest.main()
