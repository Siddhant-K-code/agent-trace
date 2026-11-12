"""Every per-session cost surface reports the `agent-strace cost` total.

The sessions are replayed from the Codex capture-matrix fixture. They record
token usage with cache reads, so a flat per-token rate gives a different
figure than the cost command.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cost import estimate_cost, session_cost
from agent_trace.dashboard import (
    build_dashboard,
    build_trend_report,
    format_dashboard,
    format_trend_terminal,
    render_html_dashboard,
    render_html_trend,
)
from agent_trace.eval.dataset import _session_passes_filter, auto_populate, list_entries
from agent_trace.hooks import _PROVIDER_ENV, hook_main
from agent_trace.langfuse_export import (
    OtlpMetricsConfig,
    _session_metrics,
    export_metrics_to_otlp,
)
from agent_trace.lint import LintLevel, _rule_budget_proximity
from agent_trace.models import EventType, TraceEvent
from agent_trace.sample import _score_session, run_sample
from agent_trace.store import TraceStore

try:
    from test_capture_matrix_fixtures import (
        FIXTURES,
        load_fixture,
        matcher_applies,
        registered_hooks,
    )
except ImportError:  # pragma: no cover - package-style test discovery
    from tests.test_capture_matrix_fixtures import (
        FIXTURES,
        load_fixture,
        matcher_applies,
        registered_hooks,
    )


# `agent-strace cost <session>` for the three fixture sessions.
FIXTURE_COSTS = (0.0035202, 0.0037002, 0.0100380)


class TestCostSurfacesMatchCostCommand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = patch.dict(os.environ, {"DO_NOT_TRACK": "1"}, clear=False)
        env.start()
        cls.addClassCleanup(env.stop)
        for name in (
            *_PROVIDER_ENV.values(),
            "AGENT_TRACE_REDACT",
            "AGENT_STRACE_TENANT_ID",
            "AGENT_STRACE_WORKSPACE",
            "AGENT_STRACE_ENDPOINT",
        ):
            os.environ.pop(name, None)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = Path(tmp.name)
        trace_dir = str(cls.tmp / ".agent-traces")
        os.environ["AGENT_TRACE_DIR"] = trace_dir
        registered = registered_hooks("codex")
        for scenario in load_fixture(FIXTURES["codex"])["scenarios"].values():
            for hook in scenario["hooks"]:
                event, payload = hook["event"], hook["payload"]
                if event not in registered or not matcher_applies(
                    registered[event][0], event, payload
                ):
                    continue
                stdin = io.StringIO(json.dumps(payload))
                with patch.object(sys, "stdin", stdin), redirect_stderr(io.StringIO()):
                    hook_main(["--provider", "codex", registered[event][1]])
        cls.store = TraceStore(trace_dir)
        cls.costs = {
            meta.session_id: estimate_cost(cls.store, meta.session_id).total_cost
            for meta in cls.store.list_sessions()
        }

    def test_fixture_sessions_record_cache_reads(self):
        self.assertEqual(len(self.costs), 3)
        for cost, want in zip(sorted(self.costs.values()), FIXTURE_COSTS):
            self.assertAlmostEqual(cost, want, places=10)
        for sid in self.costs:
            self.assertGreater(estimate_cost(self.store, sid).cache_read_tokens, 0)

    def test_session_cost_equals_estimate_cost(self):
        for sid, cost in self.costs.items():
            events = self.store.load_events(sid)
            self.assertEqual(session_cost(events, session_id=sid), cost)
            self.assertEqual(
                session_cost(events, "opus"),
                estimate_cost(self.store, sid, model="opus").total_cost,
            )

    def test_dashboard_session_costs(self):
        report = build_dashboard(self.store)
        self.assertEqual(
            {s.session_id: s.estimated_cost for s in report.summaries}, self.costs
        )
        self.assertAlmostEqual(report.total_cost, sum(self.costs.values()), places=12)
        self.assertEqual(report.unpriced_sessions, 0)

        out = io.StringIO()
        format_dashboard(report, out=out)
        text = out.getvalue()
        self.assertIn(f"Total cost:    ~${report.total_cost:.4f}\n", text)
        page = render_html_dashboard(report)
        for cost in self.costs.values():
            self.assertIn(f"  ${cost:>7.4f}  ", text)
            self.assertIn(f"<td>${cost:.4f}</td>", page)
        self.assertNotIn("n/a", text)
        self.assertNotIn("not priced", page)

    def test_dashboard_trend_costs(self):
        report = build_trend_report(self.store)
        self.assertEqual({p.session_id: p.cost for p in report.points}, self.costs)
        page = render_html_trend(report)
        for cost in self.costs.values():
            self.assertIn(f"<td>${cost:.4f}</td>", page)
        out = io.StringIO()
        format_trend_terminal(report, out=out)
        self.assertNotIn("not priced", out.getvalue())

    def test_otlp_session_cost_metric(self):
        for meta in self.store.list_sessions():
            metrics = _session_metrics(self.store, meta)
            self.assertEqual(
                metrics["agent_strace.session.cost_usd"], self.costs[meta.session_id]
            )

        bodies = []

        def _capture(req, timeout=30):
            bodies.append(json.loads(req.data))
            resp = MagicMock(status=200)
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        config = OtlpMetricsConfig(endpoint="http://localhost:4318")
        with patch("urllib.request.urlopen", side_effect=_capture), \
                redirect_stderr(io.StringIO()):
            self.assertTrue(export_metrics_to_otlp(
                self.store, list(self.costs), config, include_scores=False
            ))
        points = [
            metric["gauge"]["dataPoints"][0]["asDouble"]
            for metric in bodies[0]["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
            if metric["name"] == "agent_strace.session.cost_usd"
        ]
        self.assertEqual(sorted(points), sorted(self.costs.values()))

    def test_sample_cost_estimate(self):
        for meta in self.store.list_sessions():
            events = self.store.load_events(meta.session_id)
            score = _score_session(meta.session_id, meta, events)
            self.assertEqual(score.cost_estimate, self.costs[meta.session_id])

        output = self.tmp / "sample.jsonl"
        with redirect_stderr(io.StringIO()):
            rc = run_sample(
                self.store, "recent", 3, str(output), out=io.StringIO()
            )
        self.assertEqual(rc, 0)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(
            {r["session_id"]: r["score"]["cost_estimate"] for r in records},
            {sid: round(cost, 6) for sid, cost in self.costs.items()},
        )

    def test_lint_budget_proximity_cost(self):
        cfg = {"enabled": True, "level": LintLevel.ERROR, "threshold": 0.90}
        for sid, cost in self.costs.items():
            events = self.store.load_events(sid)

            def _with_budget(budget):
                start = TraceEvent(
                    event_type=EventType.SESSION_START,
                    timestamp=events[0].timestamp,
                    session_id=sid,
                    data={"budget_dollars": budget},
                )
                return [start, *events]

            results = _rule_budget_proximity(_with_budget(cost / 0.95), cfg)
            self.assertEqual(len(results), 1)
            self.assertIn("Session reached 95% of a", results[0].message)
            self.assertIn(f"(estimated ${cost:.4f} spent)", results[0].message)
            self.assertEqual(_rule_budget_proximity(_with_budget(cost / 0.85), cfg), [])

    def test_dataset_cost_above_filter(self):
        for sid, cost in self.costs.items():
            self.assertTrue(
                _session_passes_filter(self.store, sid, f"cost-above:{cost * 0.999:.9f}")
            )
            self.assertFalse(
                _session_passes_filter(self.store, sid, f"cost-above:{cost * 1.001:.9f}")
            )

        dataset = self.tmp / "dataset.jsonl"
        added = auto_populate(self.store, dataset, "cost-above:0.005", since_days=36500)
        self.assertEqual(added, 1)
        [expensive] = [sid for sid, cost in self.costs.items() if cost > 0.005]
        self.assertEqual([e.session_id for e in list_entries(dataset)], [expensive])


if __name__ == "__main__":
    unittest.main()
