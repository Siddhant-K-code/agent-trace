"""Evaluation framework for agent sessions.

Score, compare, and regression-test agent sessions against configurable
scorers. All storage is local JSONL — no external service, no database.
"""

from __future__ import annotations

import argparse
import sys

from .runner import cmd_eval_run, cmd_eval_compare, cmd_eval_ci
from .dataset import cmd_dataset


def cmd_eval(args: argparse.Namespace) -> int:
    eval_command = getattr(args, "eval_command", None)
    if not eval_command:
        sys.stderr.write(
            "Usage: agent-strace eval <run|compare|ci|dataset> ...\n"
            "Run `agent-strace eval --help` for details.\n"
        )
        return 1

    if eval_command == "run":
        return cmd_eval_run(args)
    if eval_command == "compare":
        return cmd_eval_compare(args)
    if eval_command == "ci":
        return cmd_eval_ci(args)
    if eval_command == "dataset":
        return cmd_dataset(args)

    sys.stderr.write(f"Unknown eval subcommand: {eval_command}\n")
    return 1
