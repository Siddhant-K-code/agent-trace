#!/usr/bin/env python3
"""Render assets/demo.svg from a recorded session.

Runs `agent-strace replay`, `agent-strace health` and `agent-strace review`
against a trace store, trims the output to fit the README, and writes one
static terminal-style SVG. The text in the image is the commands' real output:
lines are only dropped or cut, never written by hand.

    python3 scripts/demo/render_svg.py \\
        --agent-strace /tmp/demo-tools/venv/bin/agent-strace \\
        --repo /tmp/demo -o assets/demo.svg

Record the session first with scripts/demo/record-session.sh. Stdlib only.
The script fails if the output contains the home directory, user name, host
name, or anything that looks like a token.
"""

from __future__ import annotations

import argparse
import getpass
import html
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

COLS = 100
FONT_SIZE = 13
CHAR_W = FONT_SIZE * 0.6
LINE_H = 17
PAD_X = 16
TITLE_H = 36
TOP = TITLE_H + 24
FONT = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

BG = "#0d1117"
BAR = "#161b22"
FG = "#e6edf3"
MUTED = "#8b949e"
PROMPT = "#58a6ff"
BOLD = ' font-weight="bold"'
# Browsers collapse runs of spaces in SVG text inside <img>; NBSP keeps alignment.
NBSP = "\u00a0"

# ANSI SGR code -> GitHub dark palette.
SGR = {
    "31": "#ff7b72",
    "32": "#3fb950",
    "33": "#d29922",
    "34": "#58a6ff",
    "35": "#d2a8ff",
    "36": "#39c5cf",
    "37": FG,
    "90": MUTED,
}
STATUS = {
    "pass": "#3fb950",
    "present": "#3fb950",
    "healthy": "#3fb950",
    "partial": "#d29922",
    "inconclusive": MUTED,
    "missing": "#ff7b72",
    "fail": "#ff7b72",
    "unknown": "#ff7b72",
    "invalid": "#ff7b72",
}
TOKEN_RE = re.compile(r"(gh[opsur]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|sk-[A-Za-z0-9-]{20,})")
ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")

Segment = tuple[str, str, bool]  # text, color, bold


def run(binary: str, args: list[str], repo: Path, trace_dir: Path) -> str:
    env = dict(os.environ, AGENT_TRACE_DIR=str(trace_dir), DO_NOT_TRACK="1", COLUMNS="200")
    proc = subprocess.run(
        [binary, *args], cwd=repo, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        sys.exit(f"agent-strace {' '.join(args)} failed ({proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def parse_ansi(line: str, default: str = FG) -> list[Segment]:
    segments: list[Segment] = []
    color, bold, dim = default, False, False
    pos = 0
    for match in ANSI_RE.finditer(line):
        if match.start() > pos:
            segments.append((line[pos : match.start()], MUTED if dim else color, bold))
        for code in (match.group(1) or "0").split(";"):
            if code in ("", "0"):
                color, bold, dim = default, False, False
            elif code == "1":
                bold = True
            elif code == "2":
                dim = True
            elif code in SGR:
                color = SGR[code]
        pos = match.end()
    if pos < len(line):
        segments.append((line[pos:], MUTED if dim else color, bold))
    return segments


def plain(line: str, color: str = FG) -> list[Segment]:
    """Plain text with status words colored."""
    segments: list[Segment] = []
    pos = 0
    pattern = re.compile(r"\b(" + "|".join(STATUS) + r")\b")
    for match in pattern.finditer(line):
        word = match.group(1)
        before = line[: match.start()]
        # Only color a status in a status position: after a label or in the findings column.
        if not (before.rstrip().endswith(":") or re.match(r"^\S.*\s{2,}$", before)):
            continue
        segments.append((line[pos : match.start()], color, False))
        segments.append((word, STATUS[word], True))
        pos = match.end()
    segments.append((line[pos:], color, False))
    return [s for s in segments if s[0]]


def width(segments: list[Segment]) -> int:
    return sum(len(text) for text, _, _ in segments)


def truncate(segments: list[Segment], cols: int = COLS) -> list[Segment]:
    if width(segments) <= cols:
        return segments
    out: list[Segment] = []
    left = cols - 1
    for text, color, bold in segments:
        if left <= 0:
            break
        out.append((text[:left], color, bold))
        left -= len(text[:left])
    out.append(("…", MUTED, False))
    return out


def wrap(text: str, indent: int, cols: int = COLS) -> list[str]:
    """Wrap a plain line at word boundaries, keeping its indent."""
    if len(text) <= cols:
        return [text]
    lead = text[: len(text) - len(text.lstrip(" "))]
    lines, current = [], lead
    pad = " " * indent
    for word in text.lstrip(" ").split(" "):
        candidate = f"{current} {word}" if current.strip() else current + word
        if len(candidate) > cols and current.strip():
            lines.append(current)
            current = pad + word
        else:
            current = candidate
    lines.append(current)
    return lines


def replay_lines(out: str) -> list[list[Segment]]:
    """Event stream only: the session summary box is dropped.

    Event lines are cut at the width. Detail lines (one color, indented) wrap
    to two rows first, so a short tool result keeps its ending.
    """
    raw = out.splitlines()
    start = next(i for i, l in enumerate(raw) if ANSI_RE.sub("", l).lstrip().startswith("+"))
    lines: list[list[Segment]] = []
    for line in raw[start:]:
        if not line.strip():
            continue
        segments = parse_ansi(line.rstrip())
        text = "".join(s[0] for s in segments)
        colors = {s[1] for s in segments if s[0].strip()}
        if text.startswith(" ") and len(colors) == 1:
            indent = len(text) - len(text.lstrip())
            parts = wrap(text, indent)
            if len(parts) > 2:
                parts = [parts[0], parts[1] + " " + " ".join(p.strip() for p in parts[2:])]
            color = colors.pop() if colors else MUTED
            lines += [truncate([(p, color, False)]) for p in parts]
            continue
        lines.append(truncate(segments))
    return lines


def health_lines(out: str, keep_limits: int = 3) -> list[list[Segment]]:
    raw = out.splitlines()
    lines: list[list[Segment]] = []
    i = 0
    while i < len(raw):
        line = raw[i]
        if line.startswith("Provider limitations"):
            lines.append(plain(line))
            items = []
            i += 1
            while i < len(raw) and raw[i].startswith("  - "):
                items.append(raw[i])
                i += 1
            lines += [truncate(plain(item, MUTED)) for item in items[:keep_limits]]
            if len(items) > keep_limits:
                lines.append(plain(f"  ... {len(items) - keep_limits} more", MUTED))
            continue
        if line.startswith("Matrix version:"):
            i += 1
            continue
        if line.strip():
            for part in wrap(line, 11):
                lines.append(truncate(plain(part)))
        i += 1
    return lines


def review_lines(out: str) -> list[list[Segment]]:
    """Findings with their details. Source lines are dropped."""
    lines: list[list[Segment]] = []
    for line in out.splitlines():
        if not line.strip() or line.lstrip().startswith("Source:"):
            continue
        if line.startswith("Captured evidence only"):
            lines.append(plain(line, MUTED))
            continue
        color = MUTED if line.startswith("  ") else FG
        for part in wrap(line.rstrip(), 2):
            lines.append(truncate(plain(part, color)))
    return lines


def scrub_check(lines: list[list[Segment]]) -> None:
    text = "\n".join("".join(s[0] for s in line) for line in lines)
    home = str(Path.home())
    banned = {home, f"/Users/{getpass.getuser()}", getpass.getuser(), socket.gethostname().split(".")[0]}
    found = [b for b in banned if b and len(b) > 2 and b in text]
    if TOKEN_RE.search(text):
        found.append("token-like string")
    if found:
        sys.exit(f"refusing to write: output contains {', '.join(found)}")


def svg(title: str, lines: list[list[Segment]]) -> str:
    w = int(PAD_X * 2 + COLS * CHAR_W)
    h = TOP + LINE_H * len(lines) + 4
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'font-family="{FONT}" font-size="{FONT_SIZE}" role="img" aria-label="{html.escape(title)}">',
        f"  <title>{html.escape(title)}</title>",
        f'  <rect width="{w}" height="{h}" rx="8" fill="{BG}"/>',
        f'  <rect width="{w}" height="{TITLE_H}" rx="8" fill="{BAR}"/>',
        f'  <rect y="{TITLE_H - 8}" width="{w}" height="8" fill="{BAR}"/>',
        '  <circle cx="20" cy="18" r="6" fill="#ff5f57"/>',
        '  <circle cx="40" cy="18" r="6" fill="#febc2e"/>',
        '  <circle cx="60" cy="18" r="6" fill="#28c840"/>',
        f'  <text x="{w // 2}" y="23" text-anchor="middle" fill="{MUTED}" font-size="12">'
        f"{html.escape(title)}</text>",
        "  <g>",
    ]
    for n, segments in enumerate(lines):
        y = TOP + n * LINE_H
        spans = "".join(
            f'<tspan fill="{color}"{BOLD if bold else ""}>{html.escape(text, quote=False).replace(" ", NBSP)}</tspan>'
            for text, color, bold in segments
        )
        parts.append(f'    <text x="{PAD_X}" y="{y}">{spans}</text>')
    parts += ["  </g>", "</svg>", ""]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--agent-strace", default="agent-strace", help="binary of the release to show")
    parser.add_argument("--repo", type=Path, default=Path("/tmp/demo"), help="demo repo (cwd for review)")
    parser.add_argument("--trace-dir", type=Path, help="trace store (default REPO/.agent-traces)")
    parser.add_argument("-o", "--output", type=Path, default=Path("assets/demo.svg"))
    args = parser.parse_args()
    trace_dir = args.trace_dir or args.repo / ".agent-traces"

    version = run(args.agent_strace, ["--version"], args.repo, trace_dir).split()[-1]
    blank: list[Segment] = []

    def prompt(cmd: str) -> list[Segment]:
        return [("$ ", PROMPT, False), (cmd, FG, True)]

    lines: list[list[Segment]] = [prompt("agent-strace replay")]
    lines += replay_lines(run(args.agent_strace, ["replay"], args.repo, trace_dir))
    lines += [blank, prompt("agent-strace health")]
    lines += health_lines(run(args.agent_strace, ["health"], args.repo, trace_dir))
    lines += [blank, prompt("agent-strace review")]
    lines += review_lines(run(args.agent_strace, ["review"], args.repo, trace_dir))

    scrub_check(lines)
    title = f"agent-strace {version}: replay, health, review of a real Copilot CLI session (trimmed)"
    args.output.write_text(svg(title, lines), encoding="utf-8")
    print(f"wrote {args.output} ({len(lines)} lines, agent-strace {version})")


if __name__ == "__main__":
    main()
