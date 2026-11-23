#!/usr/bin/env python3
"""Summarise Weaver live-check reports from the conformance runner.

    python summarize.py output/before-v0.1.0 output/after-v0.1.0

Each argument is a ``--report-dir`` holding one ``<scenario>.json`` per
scenario. Prints Markdown: advice counts per scenario and finding, then every
finding other than ``missing_attribute`` and ``not_stable`` with the span it
is on, then the attribute keys ``missing_attribute`` names (keys outside the
registry). The runner's own ``data.json`` merges findings by span name across
scenarios, so a fixed span that shares a name with an unfixed one does not
show there; these counts are per span.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

LISTED_ELSEWHERE = ("missing_attribute", "not_stable")


def _advice(node, owner=None):
    if isinstance(node, dict):
        if "span" in node and isinstance(node["span"], dict):
            owner = node["span"]
        result = node.get("live_check_result")
        if result:
            for advice in result.get("all_advice", []):
                yield owner, advice
        for value in node.values():
            yield from _advice(value, owner)
    elif isinstance(node, list):
        for value in node:
            yield from _advice(value, owner)


def _reports(report_dir: Path) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(report_dir.glob("*.json"))
        if path.name != "data.json"
    }


def _span_label(span) -> str:
    if span is None:
        return "(resource or scope)"
    status = (span.get("status") or {}).get("code") or "unset"
    return f"`{span['name']}` (status {status})"


def main(argv: list[str]) -> int:
    runs = {Path(arg).name: _reports(Path(arg)) for arg in argv}
    scenarios = sorted({name for reports in runs.values() for name in reports})

    print("| scenario | finding | level | " + " | ".join(runs) + " |")
    print("|---|---|---|" + "---:|" * len(runs))
    for scenario in scenarios:
        counts = {
            run: collections.Counter(
                (advice["id"], advice["level"])
                for _, advice in _advice(reports.get(scenario, {}).get("samples", []))
            )
            for run, reports in runs.items()
        }
        keys = sorted(set().union(*counts.values()), key=lambda k: (k[1] != "violation", k))
        for finding, level in keys:
            cells = " | ".join(str(counts[run][(finding, level)]) for run in runs)
            print(f"| {scenario} | `{finding}` | {level} | {cells} |")
        links = {
            run: sum(len(sample["span"].get("span_links", []))
                     for sample in reports.get(scenario, {}).get("samples", [])
                     if "span" in sample)
            for run, reports in runs.items()
        }
        print(f"| {scenario} | span links received | | "
              + " | ".join(str(links[run]) for run in runs) + " |")

    for run, reports in runs.items():
        print(f"\n### {run}: findings other than {', '.join(LISTED_ELSEWHERE)}\n")
        found = collections.Counter()
        for scenario, report in reports.items():
            for span, advice in _advice(report["samples"]):
                if advice["id"] in LISTED_ELSEWHERE:
                    continue
                context = advice.get("context") or {}
                detail = context.get("missing_attribute") or context.get("attribute_key") or ""
                found[(scenario, _span_label(span), advice["id"], detail)] += 1
        if not found:
            print("None.")
        for (scenario, span, finding, detail), count in sorted(found.items()):
            suffix = f" `{detail}`" if detail else ""
            print(f"- {scenario}: {span}: `{finding}`{suffix} ×{count}")

    for run, reports in runs.items():
        keys = sorted({
            (advice.get("context") or {}).get("attribute_key", "")
            for report in reports.values()
            for _, advice in _advice(report["samples"])
            if advice["id"] == "missing_attribute"
        })
        print(f"\n### {run}: {len(keys)} attribute keys outside the registry\n")
        print(", ".join(f"`{key}`" for key in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
