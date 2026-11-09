"""Helper for tests that race several Python processes on one file."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def run_concurrently(script: str, args_per_process: list[list[str]]) -> list[str]:
    """Run *script* in one Python process per argument list and return their stdout.

    The script's own arguments start at ``sys.argv[2]``.  It should call
    ``wait_for_start()`` after its imports so that all processes begin the
    racing work at the same moment.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    start_at = str(time.time() + 1.5)
    prelude = (
        "import sys, time\n"
        "def wait_for_start():\n"
        "    time.sleep(max(0.0, float(sys.argv[1]) - time.time()))\n"
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", prelude + script, start_at, *args],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for args in args_per_process
    ]
    outputs, failures = [], []
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        outputs.append(out)
        if proc.returncode != 0:
            failures.append(err or out)
    if failures:
        raise AssertionError("worker failed:\n" + "\n".join(failures))
    return outputs
