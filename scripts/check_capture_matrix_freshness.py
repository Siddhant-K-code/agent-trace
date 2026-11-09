#!/usr/bin/env python3
"""Report how long ago each capture matrix entry was verified.

    python scripts/check_capture_matrix_freshness.py            report only, exit 0
    python scripts/check_capture_matrix_freshness.py --check    exit 1 when a verified
                                                                entry is stale or its
                                                                last_verified is malformed

A verified entry is stale when its last_verified date is more than
--max-age-days before --today. Unverified entries are listed for information
and never fail. Exit status 2 means bad arguments or an unreadable matrix.

See docs/capture-matrix.md#freshness.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO


DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_MATRIX = (
    Path(__file__).resolve().parents[1] / "src" / "agent_trace" / "capture_matrix.json"
)
# Maintainers record last_verified in their local date while CI runs in UTC,
# so a date one day ahead of the check date is accepted as age 0.
FUTURE_TOLERANCE_DAYS = 1
STATUSES = ("fresh", "stale", "unverified", "malformed")

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class EntryResult:
    provider: str
    name: str
    status: str
    verification_status: Any
    tested_version: Any
    last_verified: Any
    age_days: Optional[int]
    stale_from: Optional[str]
    message: str


def parse_iso_date(value: Any) -> Optional[date]:
    """Return the date for a strict YYYY-MM-DD string, otherwise None."""
    # date.fromisoformat alone also accepts forms such as 20261022 on 3.11+.
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def assess_entry(entry: dict, today: date, max_age_days: int) -> EntryResult:
    provider = str(entry.get("id", "?"))
    status_value = entry.get("verification_status")
    tested_version = entry.get("tested_version")
    last_verified = entry.get("last_verified")

    def result(status, message, age_days=None, stale_from=None):
        return EntryResult(
            provider=provider,
            name=str(entry.get("name", provider)),
            status=status,
            verification_status=status_value,
            tested_version=tested_version,
            last_verified=last_verified,
            age_days=age_days,
            stale_from=stale_from,
            message=message,
        )

    if status_value == "unverified":
        return result("unverified", "no verified release recorded (informational)")
    if status_value != "verified":
        return result("malformed", f"unknown verification_status {status_value!r}")

    verified_on = parse_iso_date(last_verified)
    if verified_on is None:
        return result(
            "malformed",
            f"last_verified {last_verified!r} is not a YYYY-MM-DD date",
        )
    age_days = (today - verified_on).days
    if age_days < -FUTURE_TOLERANCE_DAYS:
        return result(
            "malformed",
            f"last_verified {last_verified} is after the check date {today.isoformat()}",
        )
    age_days = max(age_days, 0)
    stale_from = (verified_on + timedelta(days=max_age_days + 1)).isoformat()
    version = tested_version or "no tested_version"
    if age_days > max_age_days:
        return result(
            "stale",
            f"{version}, last verified {last_verified} ({_days(age_days)} ago, "
            f"more than {_days(max_age_days)})",
            age_days,
            stale_from,
        )
    return result(
        "fresh",
        f"{version}, last verified {last_verified} ({_days(age_days)} ago; "
        f"stale from {stale_from})",
        age_days,
        stale_from,
    )


def assess_matrix(matrix: Any, today: date, max_age_days: int) -> list:
    if not isinstance(matrix, dict) or not isinstance(matrix.get("providers"), list):
        raise ValueError("capture matrix must be an object with a providers list")
    results = []
    for entry in matrix["providers"]:
        if not isinstance(entry, dict):
            raise ValueError("capture matrix provider entries must be objects")
        results.append(assess_entry(entry, today, max_age_days))
    return results


def counts(results: Sequence[EntryResult]) -> dict:
    return {status: sum(r.status == status for r in results) for status in STATUSES}


def has_failures(results: Sequence[EntryResult]) -> bool:
    return any(r.status in ("stale", "malformed") for r in results)


def render_text(results: Sequence[EntryResult], today: date, max_age_days: int) -> str:
    lines = [
        f"Capture matrix freshness on {today.isoformat()} "
        f"(verified entries are stale after {_days(max_age_days)})"
    ]
    width = max((len(r.provider) for r in results), default=0)
    for r in results:
        lines.append(f"{r.provider:<{width}}  {r.status:<10}  {r.message}")
    summary = counts(results)
    lines.append(
        "Summary: " + ", ".join(f"{summary[status]} {status}" for status in STATUSES)
    )
    return "\n".join(lines) + "\n"


def render_json(
    results: Sequence[EntryResult], today: date, max_age_days: int, matrix: Path
) -> str:
    payload = {
        "today": today.isoformat(),
        "max_age_days": max_age_days,
        "matrix": str(matrix),
        "counts": counts(results),
        "failing": has_failures(results),
        "entries": [asdict(r) for r in results],
    }
    return json.dumps(payload, indent=2) + "\n"


def _days(count: int) -> str:
    return f"{count} day" if count == 1 else f"{count} days"


def _date_arg(value: str) -> date:
    parsed = parse_iso_date(value)
    if parsed is None:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = -1
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_capture_matrix_freshness.py",
        description="Report how long ago each capture matrix entry was verified.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX,
        help="capture matrix JSON (default: src/agent_trace/capture_matrix.json)",
    )
    parser.add_argument(
        "--today",
        type=_date_arg,
        default=None,
        help="check date as YYYY-MM-DD (default: current UTC date)",
    )
    parser.add_argument(
        "--max-age-days",
        type=_non_negative_int,
        default=DEFAULT_MAX_AGE_DAYS,
        help=f"days a verification stays fresh (default: {DEFAULT_MAX_AGE_DAYS})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when a verified entry is stale or has a malformed last_verified",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    today = args.today or datetime.now(timezone.utc).date()

    try:
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        results = assess_matrix(matrix, today, args.max_age_days)
    except (OSError, ValueError) as exc:
        stderr.write(f"error: cannot read capture matrix {args.matrix}: {exc}\n")
        return 2

    if args.format == "json":
        stdout.write(render_json(results, today, args.max_age_days, args.matrix))
    else:
        stdout.write(render_text(results, today, args.max_age_days))
    return 1 if args.check and has_failures(results) else 0


if __name__ == "__main__":
    sys.exit(main())
