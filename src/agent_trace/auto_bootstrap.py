"""Interpreter start-up hook used by ``agent-strace auto``.

``agent-strace auto -- <command>`` writes a ``sitecustomize.py`` into a
private temporary directory, prepends that directory to ``PYTHONPATH`` and
deletes it when the command exits. Python's ``site`` module imports
``sitecustomize`` while the interpreter starts, before any user code, so
every interpreter the command starts is covered: ``python script.py``,
``python -m module``, console scripts, shebang scripts and Python
subprocesses. The generated module:

1. runs the ``sitecustomize`` it shadows, if there is one (a venv's or a
   distribution's), and removes its own directory from ``sys.path``;
2. imports ``agent_trace.auto``, which instruments the frameworks named in
   ``AGENT_STRACE_AUTO_INSTRUMENT``. The interpreter's own ``agent_trace`` is
   used when it has one. Otherwise the package that launched the command is
   loaded by path, without adding its directory (often a whole
   site-packages) to ``sys.path``, so it cannot shadow the child's packages;
3. writes one warning to stderr and lets the program run if that fails.

A ``.pth`` file is not used because ``site`` only reads ``.pth`` files from
site directories, never from ``PYTHONPATH`` entries. Using one would mean
writing into the child's site-packages: that changes a shared environment
for every process using it while the command runs, needs write access, and
is left behind if the wrapper is killed. ``sitecustomize`` on ``PYTHONPATH``
is scoped to the command's environment and needs no write access.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

INSTRUMENT_ENV = "AGENT_STRACE_AUTO_INSTRUMENT"
BOOTSTRAP_PREFIX = "agent-strace-auto-"
_PACKAGE_PARENT_PLACEHOLDER = "__AGENT_STRACE_PACKAGE_PARENT__"

# Kept free of Python 3-only syntax so that a Python 2 interpreter started
# by the command still runs the sitecustomize it shadows and only warns.
_SITECUSTOMIZE_TEMPLATE = r'''# -*- coding: utf-8 -*-
"""Start-up hook written by ``agent-strace auto``; deleted when it exits.

Runs the sitecustomize module this file shadows, then imports
agent_trace.auto, which instruments the frameworks named in
AGENT_STRACE_AUTO_INSTRUMENT. It never stops the program from running.
"""
import os
import sys

_PACKAGE_PARENT = __AGENT_STRACE_PACKAGE_PARENT__


def _real(entry):
    return os.path.normcase(os.path.realpath(entry or os.curdir))


def _run_shadowed_sitecustomize():
    here = _real(os.path.dirname(os.path.abspath(__file__)))
    kept = []
    for entry in sys.path:
        try:
            if _real(entry) == here:
                continue
        except Exception:
            pass
        kept.append(entry)
    sys.path[:] = kept
    this_module = sys.modules.pop("sitecustomize", None)
    try:
        try:
            __import__("sitecustomize")
        except ImportError as exc:
            missing = getattr(exc, "name", None) == "sitecustomize" or (
                str(exc) == "No module named sitecustomize")
            if not missing:
                raise
    except Exception as exc:
        # The report site.py prints when a sitecustomize module fails.
        if sys.flags.verbose:
            sys.excepthook(*sys.exc_info())
        else:
            sys.stderr.write(
                "Error in sitecustomize; set PYTHONVERBOSE for traceback:\n"
                "%s: %s\n" % (exc.__class__.__name__, exc))
    if "sitecustomize" not in sys.modules and this_module is not None:
        sys.modules["sitecustomize"] = this_module


def _import_agent_trace_auto():
    try:
        import agent_trace.auto
        return
    except ImportError as exc:
        if getattr(exc, "name", None) != "agent_trace":
            raise
    # Not installed for this interpreter: load the package that launched
    # the command by path, without putting its directory on sys.path.
    import importlib.machinery
    import importlib.util
    spec = importlib.machinery.PathFinder.find_spec("agent_trace", [_PACKAGE_PARENT])
    if spec is None or spec.loader is None:
        raise ImportError(
            "not installed for this interpreter and not found in %s" % _PACKAGE_PARENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_trace"] = module
    try:
        spec.loader.exec_module(module)
        import agent_trace.auto
    except BaseException:
        for name in list(sys.modules):
            if name == "agent_trace" or name.startswith("agent_trace."):
                del sys.modules[name]
        raise


def _bootstrap():
    try:
        _run_shadowed_sitecustomize()
    except Exception:
        pass
    try:
        _import_agent_trace_auto()
    except Exception as exc:
        try:
            sys.stderr.write(
                "[agent-strace] auto-instrumentation disabled in this Python %d.%d "
                "process: cannot import agent_trace (%s: %s)\n"
                % (sys.version_info[0], sys.version_info[1],
                   exc.__class__.__name__, exc))
        except Exception:
            pass


_bootstrap()
'''


def package_parent() -> str:
    """Return the directory that contains the running ``agent_trace`` package."""
    return str(Path(__file__).resolve().parent.parent)


def render_sitecustomize(parent: str) -> str:
    """Return the ``sitecustomize.py`` source that loads agent_trace from *parent*."""
    return _SITECUSTOMIZE_TEMPLATE.replace(_PACKAGE_PARENT_PLACEHOLDER, repr(parent))


def create_bootstrap_dir() -> str:
    """Create a private temporary directory holding the start-up hook.

    ``mkdtemp`` makes the directory readable and writable only by the current
    user, so no one else can swap the code every child interpreter runs.
    """
    bootstrap_dir = tempfile.mkdtemp(prefix=BOOTSTRAP_PREFIX)
    try:
        path = os.path.join(bootstrap_dir, "sitecustomize.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render_sitecustomize(package_parent()))
    except BaseException:
        remove_bootstrap_dir(bootstrap_dir)
        raise
    return bootstrap_dir


def child_env(base: Mapping[str, str], bootstrap_dir: str, frameworks: str) -> dict[str, str]:
    """Return *base* with the hook prepended to ``PYTHONPATH``.

    Existing ``PYTHONPATH`` entries are kept, after the hook directory.
    """
    env = dict(base)
    env[INSTRUMENT_ENV] = frameworks
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = bootstrap_dir + os.pathsep + existing if existing else bootstrap_dir
    return env


def remove_bootstrap_dir(bootstrap_dir: str) -> None:
    """Delete the hook directory, including any ``__pycache__`` a child wrote."""
    shutil.rmtree(bootstrap_dir, ignore_errors=True)
