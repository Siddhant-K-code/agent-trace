"""Assemble plain and packet-assisted views of two synthetic review cases."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from score import HERE, read_object


KIT = HERE / "reviewer-exercise"
PRODUCER_REVISION = "423601ceb6e290b31e8bb290aac67dc5c041313e"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def write_footer_bundle(root: Path) -> None:
    root.mkdir()
    target = {"repo": "demo/site", "pr": 12, "task": "update-footer", "base": "c" * 40, "head": "d" * 40}
    records = [
        ("change", "unified-diff", b"--- a/footer.txt\n+++ b/footer.txt\n@@ -1 +1 @@\n-Draft\n+Ready\n"),
        ("intent", "intent", {
            "by": "fixture author", "summary": "Use the final footer label.",
            "criteria": [{"id": "label", "text": "The footer displays Ready."}],
            "expected_steps": [{"id": "preview", "text": "Preview the rendered footer.", "required": True}],
        }),
        ("format", "validation-receipt", {
            "check_id": "format", "reported_by": "fixture CI", "result": "PASS",
            "assessment": {"by": "fixture maintainer", "relevance": "IRRELEVANT", "criterion_ids": []},
        }),
        ("summary", "session-summary", {
            "session_id": "footer-session", "reported_by": "fixture recorder",
            "capture": {"completeness": "PARTIAL", "limits": ["Only formatting events were retained."]},
            "observations": [],
        }),
    ]
    sources = []
    for source_id, kind, payload in records:
        schema_kind = "receipt" if kind == "validation-receipt" else kind
        raw = payload if isinstance(payload, bytes) else json_bytes(
            {"schema": f"review-evidence-{schema_kind}/v0", "target": target, **payload},
        )
        path = source_id + ".input"
        (root / path).write_bytes(raw)
        sources.append({
            "id": source_id, "type": kind, "version": 0, "path": path, "origin": "fixture author",
            "admission": "SYNTHETIC", "sha256": digest(raw), "lines": [1, raw.count(b"\n")],
        })
    (root / "bundle.json").write_bytes(json_bytes({
        "schema": "review-evidence-input/v0", "target": target, "sources": sources,
        "association": {"by": "fixture operator", "session_id": "footer-session"},
    }))


def verify_parity(baseline: Path, assisted: Path, packet_dir: Path) -> dict[str, str]:
    """Check identical inputs in both arms, including the packet's retained copies."""
    names = {path.name for path in baseline.iterdir()}
    if names != {path.name for path in assisted.iterdir()}:
        raise ValueError("the review arms do not contain the same input files")
    hashes = {}
    for name in sorted(names):
        raw = (baseline / name).read_bytes()
        if raw != (assisted / name).read_bytes():
            raise ValueError(f"review arms differ for {name}")
        hashes[name] = digest(raw)
    packet = read_object(packet_dir / "packet.json")
    if packet["manifest_sha256"] != hashes["bundle.json"]:
        raise ValueError("packet did not use the shared manifest")
    retained = {}
    for source in packet["sources"]:
        raw = (packet_dir / source["retained_path"]).read_bytes()
        if raw != (baseline / source["path"]).read_bytes() or digest(raw) != source["sha256"]:
            raise ValueError(f"packet's retained bytes differ for {source['path']}")
        retained[source["path"]] = source["sha256"]
    if retained != {name: hashes[name] for name in names - {"bundle.json"}}:
        raise ValueError("packet did not retain the same input sources")
    return hashes


def stage_producer(checkout: Path, workspace: Path) -> Path:
    result = subprocess.run(
        ["git", "--no-pager", "-C", str(checkout), "archive",
         "--format=tar", PRODUCER_REVISION, "src/agent_trace"],
        capture_output=True, timeout=15, env=os.environ | {"GIT_NO_LAZY_FETCH": "1"},
    )
    if result.returncode:
        raise ValueError(f"producer checkout must contain {PRODUCER_REVISION}")
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        for member in archive:
            if member.isdir():
                continue
            path = Path(member.name)
            if not member.isfile() or path.parts[:2] != ("src", "agent_trace") or ".." in path.parts:
                raise ValueError(f"unexpected entry in pinned producer archive: {member.name}")
            with archive.extractfile(member) as source:
                destination = workspace / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as copied:
                    shutil.copyfileobj(source, copied)
    return workspace / "src"


def build(checkout: Path, output: Path) -> dict:
    if output.is_relative_to(checkout) or output.exists():
        raise ValueError("output must be a new directory outside the producer checkout")
    report = {"producer_revision": PRODUCER_REVISION, "cases": {}}
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        workspace = Path(temp)
        producer_src = stage_producer(checkout, workspace)
        prepared = workspace / "prepared"
        for case in ("greeting", "footer"):
            plain = prepared / case / "baseline"
            assisted = prepared / case / "packet-assisted"
            baseline = plain / "evidence"
            if case == "greeting":
                shutil.copytree(KIT / "greeting", baseline, ignore=shutil.ignore_patterns(".gitattributes"))
            else:
                plain.mkdir(parents=True)
                write_footer_bundle(baseline)
            shutil.copytree(baseline, assisted / "evidence")
            for arm in (plain, assisted):
                shutil.copyfile(KIT / "worksheet.md", arm / "worksheet.md")
            packet_dir = assisted / "packet"
            result = subprocess.run(
                [sys.executable, "-B", "-S", "-m", "agent_trace.entry", "review-evidence",
                 str(baseline / "bundle.json"), "--output", str(packet_dir)],
                cwd=workspace, text=True, capture_output=True, timeout=30,
                env=os.environ | {
                    "PYTHONPATH": str(producer_src),
                    "DO_NOT_TRACK": "1", "AGENT_STRACE_TELEMETRY": "0",
                },
            )
            if result.returncode not in (0, 2) or not (packet_dir / "packet.json").is_file():
                raise ValueError(f"{case}: producer exit {result.returncode}: {result.stderr.strip()}")
            if case == "greeting" and (
                (packet_dir / "packet.json").read_bytes() != (HERE / "packet-fixture" / "packet.json").read_bytes()
            ):
                raise ValueError("reassembled greeting does not match the pinned packet")
            hashes = verify_parity(baseline, assisted / "evidence", packet_dir)
            report["cases"][case] = {
                "producer_exit": result.returncode, "producer_status": result.stdout.strip(),
                "identical_evidence_sha256": hashes,
            }
        (prepared / "assembly.json").write_bytes(json_bytes(report))
        output.mkdir()
        shutil.copytree(prepared, output, dirs_exist_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True, help="local checkout with the pinned producer commit")
    parser.add_argument("--output", type=Path, required=True, help="new directory with an existing parent")
    args = parser.parse_args()
    try:
        report = build(args.producer.resolve(strict=True), args.output.resolve())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
