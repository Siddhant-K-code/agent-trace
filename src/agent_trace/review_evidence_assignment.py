"""Convert a retained minimized agent-strace assignment ZIP into a review summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .assignment import AssignmentError, BUNDLE_SCHEMA, MAX_ARCHIVE_BYTES, load_assignment_bundle
from .fileutil import fsync_directory, write_atomic
from .inert import inert
from .review_evidence import (
    INPUT_SCHEMA,
    MAX_JSON_BYTES,
    RECORD_SCHEMAS,
    ReviewEvidenceError,
    _HASH,
    _association,
    _directory,
    _enum,
    _object,
    _pattern,
    _read,
    _record,
    _target,
)

SUMMARY_FILENAME = "session-summary.json"


def convert_assignment_summary(
    archive_path: str | Path, bundle_path: str | Path, binding_path: str | Path, output_dir: str | Path,
) -> Path:
    """Check the declared ZIP-to-bundle binding and emit only the minimized summary."""
    archive, bundle, binding, output = (
        Path(path).absolute() for path in (archive_path, bundle_path, binding_path, output_dir)
    )
    if any(".." in path.parts or not path.name for path in (archive, bundle, binding, output)):
        raise ReviewEvidenceError("input and output paths must not traverse parent directories")

    manifest = _object(
        _record(_read(bundle, "bundle", MAX_JSON_BYTES), "bundle"),
        "bundle", "schema target sources association",
    )
    _enum(manifest["schema"], "bundle schema", INPUT_SCHEMA)
    target = _target(manifest["target"])
    if target["base"] == target["head"]:
        raise ReviewEvidenceError("a retained change needs different base and head IDs")
    declared = _association(manifest["association"])["declaration"]

    pin = _object(
        _record(_read(binding, "assignment binding", MAX_JSON_BYTES), "assignment binding"),
        "assignment binding", "archive_sha256 target association",
    )
    expected = _pattern(pin["archive_sha256"], "assignment archive SHA-256", _HASH)
    if _target(pin["target"]) != target:
        raise ReviewEvidenceError("assignment binding target does not match bundle target")
    if _association(pin["association"])["declaration"] != declared:
        raise ReviewEvidenceError("assignment binding association does not match bundle declaration")

    raw = _read(archive, "assignment ZIP", MAX_ARCHIVE_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise ReviewEvidenceError("assignment ZIP SHA-256 does not match binding")
    try:
        load_assignment_bundle(raw)
    except AssignmentError as exc:
        raise ReviewEvidenceError(f"invalid or unsupported {BUNDLE_SCHEMA} ZIP: {exc}") from exc

    summary = {
        "schema": RECORD_SCHEMAS["session-summary"],
        "target": target,
        "session_id": declared["session_id"],
        "reported_by": "agent-strace assignment v1 adapter",
        "capture": {
            "completeness": "PARTIAL",
            "limits": [
                f"Retained {BUNDLE_SCHEMA} ZIP SHA-256: {digest}.",
                "Session, attempt, and change association are operator-declared; the ZIP omits original identifiers.",
                "This format does not provide attributable named-step observations or billing receipts.",
            ],
        },
        "observations": [],
    }
    if "attempt_id" in declared:
        summary["attempt_id"] = declared["attempt_id"]
    text = json.dumps(summary, sort_keys=True, ensure_ascii=True, indent=2) + "\n"

    _directory(output.parent)
    output.mkdir(mode=0o700)
    destination = output / SUMMARY_FILENAME
    complete = False
    try:
        fsync_directory(output.parent)
        write_atomic(destination, text, mode=0o600)
        complete = True
    finally:
        if not complete:
            destination.unlink(missing_ok=True)
            output.rmdir()
    return destination


def cmd_review_evidence_import_assignment(args: argparse.Namespace) -> int:
    try:
        convert_assignment_summary(args.archive, args.bundle, args.binding, args.output)
    except FileExistsError:
        print("review-evidence-import-assignment: output already exists", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"review-evidence-import-assignment: {inert(exc.strerror or 'local file operation failed')}",
              file=sys.stderr)
        return 1
    except (ReviewEvidenceError, KeyboardInterrupt) as exc:
        print(f"review-evidence-import-assignment: {inert(str(exc) or 'interrupted')}", file=sys.stderr)
        return 1
    print(f"Wrote {SUMMARY_FILENAME} (PARTIAL; no named-step observations)")
    return 0
