"""Score one synthetic review-evidence-packet/v0 using a separate evidence file."""

import argparse
import json
from pathlib import Path

from packet_observations import normalize_packet
from score import HERE, read_object, score_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, help="local Phase A packet.json")
    parser.add_argument("binding", type=Path, help="trusted synthetic packet binding")
    parser.add_argument("evidence", type=Path, help="independently supplied synthetic evidence")
    args = parser.parse_args()
    try:
        observation = normalize_packet(
            args.packet.read_bytes(), read_object(args.binding), read_object(args.evidence),
        )
        report = score_cases(read_object(HERE / "packet-cases.json"), {"packet-greeting": observation})
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 1 if report["regression_mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
