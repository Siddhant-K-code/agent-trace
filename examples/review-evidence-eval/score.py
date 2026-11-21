"""Score hand-labelled review evidence fixtures without running an agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, object]:
    try:
        return require_object(
            json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object),
            "top level",
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def require_object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def require_step_ids(value: object, where: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{where} must be a list of nonempty step IDs")
    return set(value)


def score_outcome(expected: dict[str, object], actual: object) -> dict[str, str]:
    if actual is None:
        return {"label": "unknown", "reason": "no artifact evidence"}
    actual = require_object(actual, "artifact")
    # Compare JSON types as well as values: True == 1 in Python.
    if json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False,
    ):
        return {"label": "pass", "reason": "artifact matches expected result"}
    return {"label": "fail", "reason": "artifact differs from expected result"}


def score_workflow(
    steps: dict[str, object], observation: dict[str, object],
) -> dict[str, dict[str, str]]:
    observed = require_step_ids(observation["observed_steps"], "observed_steps")
    sufficient = require_step_ids(observation["sufficient_capture_for"], "sufficient_capture_for")
    results = {}
    for step, expectation in steps.items():
        if not isinstance(expectation, str) or expectation not in ("required", "forbidden"):
            raise ValueError(f"invalid expectation for step {step!r}: {expectation!r}")
        if step in observed:
            evidence = "observed"
            label = "pass" if expectation == "required" else "fail"
        elif step in sufficient:
            evidence = "absent"
            label = "fail" if expectation == "required" else "pass"
        else:
            evidence = label = "unknown"
        results[step] = {"expectation": expectation, "evidence": evidence, "label": label}
    return results


def score_cases(cases: dict[str, object], observations: dict[str, object]) -> dict[str, object]:
    missing = sorted(cases.keys() - observations.keys())
    extra = sorted(observations.keys() - cases.keys())
    if missing or extra:
        raise ValueError(f"observation case IDs must match; missing={missing!r}, extra={extra!r}")

    results = {}
    mismatches = []
    for case_id, entry in cases.items():
        try:
            case = require_object(entry, "case")
            observation = require_object(observations[case_id], "observation")
            outcome = score_outcome(
                require_object(case["expected_artifact"], "expected_artifact"), observation["artifact"],
            )
            workflow = score_workflow(require_object(case["steps"], "steps"), observation)
            results[case_id] = {"outcome": outcome, "workflow": workflow}

            labels = require_object(case["expected_labels"], "expected_labels")
            step_labels = require_object(labels["workflow"], "expected workflow labels")
            if step_labels.keys() != workflow.keys():
                raise ValueError("expected workflow labels must cover exactly the case's steps")

            comparisons: list[tuple[str, str | None, object, str]] = [
                ("outcome", None, labels["outcome"], outcome["label"]),
            ]
            comparisons.extend(
                ("workflow", step, step_labels[step], result["label"])
                for step, result in workflow.items()
            )
            for dimension, step, expected, actual in comparisons:
                if expected not in ("pass", "fail", "unknown"):
                    raise ValueError(f"invalid expected label: {expected!r}")
                if expected != actual:
                    mismatches.append({
                        "case": case_id, "dimension": dimension, "step": step,
                        "expected": expected, "actual": actual,
                    })
        except KeyError as exc:
            raise ValueError(f"{case_id!r}: missing field {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"{case_id!r}: {exc}") from exc

    return {"cases": results, "regression_mismatches": mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "observations", type=Path, nargs="?", default=HERE / "observations.json",
        help="candidate observation JSON (defaults to the bundled fixture)",
    )
    args = parser.parse_args()
    try:
        report = score_cases(read_object(HERE / "cases.json"), read_object(args.observations))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 1 if report["regression_mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
