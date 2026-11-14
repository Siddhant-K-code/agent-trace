#!/usr/bin/env bash
# Install the agent-strace CLI that matches the ref of this composite Action.
#
# Environment (set through env: in action.yml, never inlined):
#   ACTION_PATH            Action checkout. Defaults to GITHUB_ACTION_PATH.
#   ACTION_REF             github.action_ref. Logged only: it can be empty or
#                          stale inside composite steps (actions/runner#2473).
#   INSTALL_EXTRAS         Comma-separated extras, such as openai,anthropic.
#   AGENT_STRACE_VERSION   Version to install from PyPI instead of the ref's.
#   AGENT_STRACE_INSTALL_ATTEMPTS  PyPI attempts for an exact pin (default 4).
#   AGENT_STRACE_INSTALL_DELAY     Seconds between attempts (default 15).
#
# The ref comes from the checkout path, which the runner lays out as
# _actions/OWNER/REPO/REF. A release tag vX.Y.Z installs agent-strace==X.Y.Z.
# Any other ref (branch, SHA, local uses: ./) installs the checkout itself.
set -euo pipefail

ACTION_PATH="${ACTION_PATH:-${GITHUB_ACTION_PATH:-}}"
ACTION_REF="${ACTION_REF:-}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-}"
AGENT_STRACE_VERSION="${AGENT_STRACE_VERSION:-}"
ATTEMPTS="${AGENT_STRACE_INSTALL_ATTEMPTS:-4}"
DELAY="${AGENT_STRACE_INSTALL_DELAY:-15}"

RELEASE_TAG_RE='^v([0-9]+\.[0-9]+\.[0-9]+)$'
VERSION_RE='^v?([0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?)$'
EXTRAS_RE='^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$'

die() {
  echo "::error::$1" >&2
  exit 1
}

# Print the ref segment of an _actions/OWNER/REPO/REF path, or nothing.
ref_from_path() {
  local path="${1//\\//}"
  path="${path%/}"
  case "$path" in
    */_actions/*) ;;
    *) return 0 ;;
  esac
  local rest="${path##*/_actions/}"
  local owner="${rest%%/*}"
  rest="${rest#"$owner"/}"
  local repo="${rest%%/*}"
  [ "$rest" != "$repo" ] || return 0
  printf '%s' "${rest#"$repo"/}"
}

extras_suffix=""
if [ -n "$INSTALL_EXTRAS" ]; then
  extras="${INSTALL_EXTRAS//[[:space:]]/}"
  [[ "$extras" =~ $EXTRAS_RE ]] || die "install-extras must be a comma-separated list of extra names, got: $INSTALL_EXTRAS"
  extras_suffix="[$extras]"
fi

pip_install() {
  python -m pip install --disable-pip-version-check "$@"
}

install_pinned() {
  local version="$1" attempt=1
  while true; do
    echo "Installing agent-strace${extras_suffix}==${version} from PyPI (attempt ${attempt}/${ATTEMPTS})"
    if pip_install "agent-strace${extras_suffix}==${version}"; then
      return 0
    fi
    [ "$attempt" -lt "$ATTEMPTS" ] || return 1
    attempt=$((attempt + 1))
    sleep "$DELAY"
  done
}

install_checkout() {
  [ -n "$ACTION_PATH" ] || die "Cannot install agent-strace from the Action checkout: GITHUB_ACTION_PATH is empty."
  [ -f "$ACTION_PATH/pyproject.toml" ] || die "Cannot install agent-strace from the Action checkout: no pyproject.toml in $ACTION_PATH."
  echo "Installing agent-strace${extras_suffix} from the Action checkout at $ACTION_PATH"
  pip_install "${ACTION_PATH}${extras_suffix}"
}

report() {
  local source="$1" installed
  installed="$(python -c 'from importlib.metadata import version; print(version("agent-strace"))' 2>/dev/null || true)"
  echo "agent-strace ${installed:-unknown} installed from ${source}"
}

path_ref="$(ref_from_path "$ACTION_PATH")"
echo "Action path: ${ACTION_PATH:-<empty>}"
echo "Action ref from path: ${path_ref:-<none>}; github.action_ref: ${ACTION_REF:-<empty>}"

if [ -n "$AGENT_STRACE_VERSION" ]; then
  [[ "$AGENT_STRACE_VERSION" =~ $VERSION_RE ]] || die "agent-strace-version must be a version such as 0.108.1, got: $AGENT_STRACE_VERSION"
  version="${BASH_REMATCH[1]}"
  install_pinned "$version" || die "agent-strace==${version} is not installable from PyPI after ${ATTEMPTS} attempts."
  report "PyPI (agent-strace-version input)"
elif [[ "$path_ref" =~ $RELEASE_TAG_RE ]]; then
  version="${BASH_REMATCH[1]}"
  if install_pinned "$version"; then
    report "PyPI (Action ref $path_ref)"
  else
    echo "::warning::agent-strace==${version} is not on PyPI yet. Installing the Action checkout for ${path_ref} instead."
    install_checkout
    report "the Action checkout (Action ref $path_ref)"
  fi
else
  install_checkout
  report "the Action checkout (Action ref ${path_ref:-<none>})"
fi
