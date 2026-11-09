"""Console-script entry point for ``agent-strace``.

Agent CLIs run ``agent-strace hook ...`` synchronously around every tool
call, and Codex also once per model request, so that path must not import
every command module or build the full argument parser.  Plain ``hook``
command lines are parsed here and passed to
:func:`agent_trace.hooks.hook_main`.  Everything else, including help, root
options before ``hook`` and malformed hook arguments, goes to
:func:`agent_trace.cli.main` unchanged, so its output and exit codes stay
those of the full parser.
"""

from __future__ import annotations

import sys

# Choices of ``agent-strace hook --provider``; cli.py uses the same tuple.
HOOK_PROVIDERS = ("claude", "codex", "gemini", "cursor", "copilot")
_PROVIDER_OPTION = "--provider"
# hooks.METADATA_ONLY_FLAG; not imported so that parsing loads nothing.
_METADATA_ONLY_FLAG = "--metadata-only"


def parse_hook_argv(argv: list[str]) -> list[str] | None:
    """Return :func:`hook_main` arguments for a plain ``hook`` command line.

    Accepts ``hook [--provider P | --provider=P | --metadata-only]... [EVENT]``
    in any order, the forms the full parser reads the same way: the last
    provider wins and the default is ``claude``.  Returns None for any other
    command line.
    """
    if not argv or argv[0] != "hook":
        return None
    provider = "claude"
    metadata_only = False
    event = None
    args = argv[1:]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == _METADATA_ONLY_FLAG:
            metadata_only = True
            index += 1
            continue
        if arg == _PROVIDER_OPTION:
            if index + 1 == len(args):
                return None
            value = args[index + 1]
            index += 2
        elif arg.startswith(_PROVIDER_OPTION + "="):
            value = arg[len(_PROVIDER_OPTION) + 1:]
            index += 1
        elif event is None and arg and not arg.startswith("-"):
            event = arg
            index += 1
            continue
        else:
            return None
        if value not in HOOK_PROVIDERS:
            return None
        provider = value
    hook_args = ["--provider", provider]
    if metadata_only:
        hook_args.append(_METADATA_ONLY_FLAG)
    if event:
        hook_args.append(event)
    return hook_args


def main() -> None:
    hook_args = parse_hook_argv(sys.argv[1:])
    if hook_args is None:
        from .cli import main as cli_main

        return cli_main()

    from .hooks import hook_main

    hook_main(hook_args)
    sys.exit(0)


if __name__ == "__main__":
    main()
