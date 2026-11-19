"""Start a child Python that a test can stop with SIGINT.

A process that ignores SIGINT passes that on to the processes it starts, and
a `cmd &` job in a non-interactive shell starts that way. Python then
installs no KeyboardInterrupt handler, so the child never sees a SIGINT that
a test sends it.
"""

import sys

# Use in place of sys.executable. It resets SIGINT to SIG_DFL, then execs
# Python with the same arguments in the same process, so the test's Popen
# still signals it.
SIGINT_DEFAULT = [
    sys.executable, "-c",
    "import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); "
    "os.execv(sys.executable, [sys.executable] + sys.argv[1:])",
]
