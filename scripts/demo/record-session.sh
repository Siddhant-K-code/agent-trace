#!/usr/bin/env bash
# Record the real GitHub Copilot CLI session behind assets/demo.svg.
#
#   scripts/demo/record-session.sh
#
# Creates a throwaway git repo with one small bug, installs the released
# agent-strace into a private venv, registers its Copilot CLI hooks in a
# private COPILOT_HOME, and runs one non-interactive Copilot CLI session that
# fixes the bug and runs the tests. Nothing outside DEMO_DIR and DEMO_TOOLS is
# changed. Your ~/.copilot and your agent-strace install are untouched.
#
# Requires: git, uv, python3, `copilot` on PATH, and a GitHub token with
# Copilot access (COPILOT_GITHUB_TOKEN, else `gh auth token`). The session is a
# real model run, so it uses a few requests and its output varies run to run.
#
#   AGENT_STRACE_VERSION  release to record with (default: __version__ in this repo)
#   DEMO_DIR              throwaway repo (default /tmp/demo, deleted first)
#   DEMO_TOOLS            venv and COPILOT_HOME location (default /tmp/demo-tools)
#
# Then render the image with scripts/demo/render_svg.py.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
version="${AGENT_STRACE_VERSION:-$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$repo_root/src/agent_trace/__init__.py")}"
demo_dir="${DEMO_DIR:-/tmp/demo}"
tools_dir="${DEMO_TOOLS:-/tmp/demo-tools}"

case "$demo_dir" in
  /tmp/*|/private/tmp/*) ;;
  *) echo "DEMO_DIR must be under /tmp so no home path enters the trace" >&2; exit 2 ;;
esac

rm -rf "$demo_dir" "$tools_dir/venv" "$tools_dir/copilot-home"
uv venv -q "$tools_dir/venv"
uv pip install -q --python "$tools_dir/venv/bin/python" "agent-strace==$version"
export PATH="$tools_dir/venv/bin:$PATH"
agent-strace --version

mkdir -p "$demo_dir/textkit" "$demo_dir/tests"
cd "$demo_dir"

cat > textkit/__init__.py <<'EOF'
EOF

cat > textkit/slug.py <<'EOF'
"""Turn titles into URL slugs."""

import re


def slugify(title: str) -> str:
    """Lowercase, drop punctuation, join words with single dashes."""
    text = title.strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return re.sub(r"\s", "-", text)
EOF

cat > tests/__init__.py <<'EOF'
EOF

cat > tests/test_slug.py <<'EOF'
import unittest

from textkit.slug import slugify


class SlugifyTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_punctuation(self):
        self.assertEqual(slugify("Ship it!"), "ship-it")

    def test_repeated_spaces(self):
        self.assertEqual(slugify("Hello,   World"), "hello-world")


if __name__ == "__main__":
    unittest.main()
EOF

cat > README.md <<'EOF'
# textkit

Small text helpers. Run the tests with `python3 -m unittest`.
EOF

git init -q -b main
git -c user.name=demo -c user.email=demo@example.com add .
git -c user.name=demo -c user.email=demo@example.com commit -q -m "textkit: add slugify"
git checkout -q -b fix-slug-spaces

# Hooks in a private Copilot config dir. The trace store stays out of git.
export COPILOT_HOME="$tools_dir/copilot-home"
export COPILOT_GITHUB_TOKEN="${COPILOT_GITHUB_TOKEN:-$(gh auth token)}"
export AGENT_TRACE_DIR="$demo_dir/.agent-traces"
export DO_NOT_TRACK=1
mkdir -p "$COPILOT_HOME"
agent-strace setup --cli copilot >/dev/null
printf '.agent-traces/\n' >> .git/info/exclude

prompt='slugify("Hello,   World") returns "hello---world". Fix textkit/slug.py so runs of spaces become one dash, then run python3 -m unittest.'

copilot -p "$prompt" \
  --no-custom-instructions \
  --no-ask-user \
  --disable-builtin-mcps \
  --allow-tool=write \
  --allow-tool='shell(python3)'

agent-strace list
