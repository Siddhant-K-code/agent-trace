"""Keep the test suite from sending product telemetry.

Tests run agent-strace in this process and in child processes, often with a
fresh HOME and an environment built from scratch.  Such an environment drops
the CI variables and ``DO_NOT_TRACK``, so nothing turns telemetry off, and
without this guard a test run can send real events to the production PostHog
project, each from a new anonymous ID.  This happens in CI too.

Importing this module installs three layers, once per process:

* ``DO_NOT_TRACK=1`` and ``AGENT_STRACE_TELEMETRY=0`` in ``os.environ``, for
  in-process code and for children that inherit the environment.
* A ``subprocess.Popen`` hook that adds the same switches to a child
  environment that sets neither.  A test that sets either one owns its
  telemetry setup; if it leaves telemetry on without its own
  ``AGENT_STRACE_TELEMETRY_HOST``, the hook points the host at a closed local
  port.
* A ``urllib.request.urlopen`` wrapper that refuses PostHog URLs, for
  in-process code that runs after a test clears ``os.environ``.

None of this depends on the disclosure state.  Tests that exercise telemetry
still mock ``urlopen`` or point ``AGENT_STRACE_TELEMETRY_HOST`` at a local
server.

``tests/__init__.py`` imports this module for ``python -m unittest tests.x``
runs.  ``unittest discover -s tests`` does not import the package, so test
modules import it as well.  Discovery imports every module before it runs any
test, so the import in ``test_no_telemetry`` covers a full run.
"""

from __future__ import annotations

import functools
import inspect
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping

TELEMETRY_OFF = {"DO_NOT_TRACK": "1", "AGENT_STRACE_TELEMETRY": "0"}
HOST_ENV = "AGENT_STRACE_TELEMETRY_HOST"
# Nothing listens on the discard port, so a request fails at once.
CLOSED_HOST = "http://127.0.0.1:9"

_MARK = "_agent_strace_no_telemetry"
_TRUE = {"1", "true", "yes", "on", "enabled"}
_FALSE = {"0", "false", "no", "off", "disabled"}


class TelemetryRefusedError(urllib.error.URLError):
    """Raised instead of sending a request to PostHog from a test."""


def without_telemetry(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``env`` for a child process that must not send telemetry."""
    return {**env, **TELEMETRY_OFF}


def _name(key: str | bytes) -> str:
    name = os.fsdecode(key)
    return name.upper() if os.name == "nt" else name


def _child_env(env: Mapping) -> dict | None:
    """Return ``env`` with telemetry kept off, or None if it needs no change."""
    view = {_name(key): os.fsdecode(value) for key, value in env.items()}
    if not any(name in view for name in TELEMETRY_OFF):
        extra = TELEMETRY_OFF
    elif HOST_ENV not in view and not _disabled(view):
        extra = {HOST_ENV: CLOSED_HOST}
    else:
        return None
    if any(isinstance(key, bytes) for key in env):
        extra = {os.fsencode(key): os.fsencode(value) for key, value in extra.items()}
    return {**env, **extra}


def _disabled(view: Mapping[str, str]) -> bool:
    def value(name: str) -> str:
        return view.get(name, "").strip().lower()

    return value("DO_NOT_TRACK") in _TRUE or value("AGENT_STRACE_TELEMETRY") in _FALSE


def _is_posthog(url: object) -> bool:
    try:
        parts = urllib.parse.urlsplit(str(url))
        host = (parts.hostname or "").lower()
        netloc = parts.netloc.lower()
    except ValueError:
        return False
    if host == "posthog.com" or host.endswith(".posthog.com"):
        return True
    telemetry = sys.modules.get("agent_trace.telemetry")
    default = getattr(telemetry, "DEFAULT_POSTHOG_HOST", "")
    return bool(default) and netloc == urllib.parse.urlsplit(default).netloc.lower()


def _guard_popen() -> None:
    original = subprocess.Popen.__init__
    if getattr(original, _MARK, False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        try:
            bound = signature.bind(self, *args, **kwargs)
        except TypeError:
            return original(self, *args, **kwargs)
        env = bound.arguments.get("env")
        child = _child_env(os.environ if env is None else env)
        if child is None:
            return original(self, *args, **kwargs)
        bound.arguments["env"] = child
        return original(*bound.args, **bound.kwargs)

    setattr(__init__, _MARK, True)
    subprocess.Popen.__init__ = __init__


def _guard_urlopen() -> None:
    original = urllib.request.urlopen
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def urlopen(url, *args, **kwargs):
        target = url.full_url if isinstance(url, urllib.request.Request) else url
        if _is_posthog(target):
            raise TelemetryRefusedError("the test suite does not send product telemetry")
        return urlopen.__wrapped__(url, *args, **kwargs)

    setattr(urlopen, _MARK, True)
    urllib.request.urlopen = urlopen


def install() -> None:
    os.environ.update(TELEMETRY_OFF)
    _guard_popen()
    _guard_urlopen()


install()
