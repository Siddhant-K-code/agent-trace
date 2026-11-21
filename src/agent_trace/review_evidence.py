"""Assemble a local PR evidence packet from explicitly retained sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assignment import AssignmentError, _check_directory_path, _read_file_by_path, _strict_json_bytes
from .fileutil import fsync_directory, split_lf_lines, write_atomic
from .inert import inert

INPUT_SCHEMA = "review-evidence-input/v0"
PACKET_SCHEMA = "review-evidence-packet/v0"
RECORD_SCHEMAS = {
    "intent": "review-evidence-intent/v0",
    "validation-receipt": "review-evidence-receipt/v0",
    "session-summary": "review-evidence-session-summary/v0",
    "attempt-inventory": "review-evidence-attempt-inventory/v0",
}
DIFF_TYPE = "unified-diff"
MAX_JSON_BYTES = 128 * 1024
MAX_DIFF_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_SOURCES = 64
MAX_ITEMS = 128
MAX_SAFE_INT = 2**53 - 1

_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_HASH = re.compile(r"[0-9a-f]{64}")
_CURRENCY = re.compile(r"[A-Z]{3}")
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_DEVICE = re.compile(r"(?i:con|prn|aux|nul|com[0-9]|lpt[0-9])")
_HUNK = re.compile(
    r"@@ -([0-9]{1,9})(?:,([0-9]{1,9}))? \+([0-9]{1,9})(?:,([0-9]{1,9}))? @@(?: .*)?"
)
_GIT_METADATA = re.compile(
    r"(?:index [0-9a-f]{4,64}\.\.[0-9a-f]{4,64}(?: [0-7]{6})?|"
    r"(?:new file mode|deleted file mode|old mode|new mode) [0-7]{6})"
)


class ReviewEvidenceError(ValueError):
    """Invalid retained evidence or unsafe assembly destination."""


@dataclass(frozen=True)
class _Source:
    metadata: dict[str, Any]
    raw: bytes
    record: dict[str, Any] | None

    @property
    def id(self) -> str:
        return self.metadata["id"]

    def at(self, pointer: str) -> dict[str, Any]:
        return {"source_id": self.id, "pointer": pointer}


def _object(value: Any, label: str, required: str, optional: str = "") -> dict[str, Any]:
    fields = set(required.split())
    if not isinstance(value, dict) or not fields <= value.keys() <= fields | set(optional.split()):
        raise ReviewEvidenceError(f"{label} has missing or unsupported fields")
    return value


def _items(value: Any, label: str, *, minimum: int = 0, maximum: int = MAX_ITEMS) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ReviewEvidenceError(f"{label} has an invalid item count")
    return value


def _enum(value: Any, label: str, *choices: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ReviewEvidenceError(f"{label} has an unsupported value")
    return value


def _pattern(value: Any, label: str, regex: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not regex.fullmatch(value):
        raise ReviewEvidenceError(f"{label} has an invalid format")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ReviewEvidenceError(f"{label} must be nonempty text of at most 512 characters")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int = MAX_SAFE_INT) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ReviewEvidenceError(f"{label} is outside the supported integer range")
    return value


def _target(value: Any) -> dict[str, Any]:
    target = _object(value, "target", "repo pr task base head")
    _pattern(target["repo"], "repo", _REPO)
    _integer(target["pr"], "PR number", 1)
    _pattern(target["task"], "task ID", _ID)
    for key in ("base", "head"):
        _pattern(target[key], key, _OID)
        if not target[key].strip("0"):
            raise ReviewEvidenceError("zero object IDs are not supported")
    if len(target["base"]) != len(target["head"]):
        raise ReviewEvidenceError("base and head use different object ID formats")
    return target


def _record(raw: bytes, label: str) -> Any:
    try:
        return _strict_json_bytes(raw, label=label, max_bytes=MAX_JSON_BYTES, source_text=True)
    except AssignmentError as exc:
        raise ReviewEvidenceError(f"{label}: {exc}") from exc


def _directory(path: Path) -> None:
    try:
        _check_directory_path(path)
    except AssignmentError as exc:
        raise ReviewEvidenceError(str(exc)) from exc


def _read(path: Path, label: str, bound: int) -> bytes:
    _directory(path.parent)
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReviewEvidenceError(f"{label} could not be opened safely") from exc
    if info.st_nlink > 1:
        raise ReviewEvidenceError(f"{label} is multiply linked")
    try:
        return _read_file_by_path(path, max_bytes=bound, min_size=1, label=label)
    except AssignmentError as exc:
        raise ReviewEvidenceError(str(exc)) from exc


def _relative_path(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or len(value) > 240:
        raise ReviewEvidenceError("source path is invalid")
    parts = tuple(value.split("/"))
    if len(parts) > 16 or any(
        not _SEGMENT.fullmatch(part)
        or part.endswith(".")
        or _DEVICE.fullmatch(part.split(".")[0])
        for part in parts
    ):
        raise ReviewEvidenceError("source path must be a portable relative file path")
    return parts


def _diff(lines: list[str]) -> None:
    """Check text hunks without applying the patch or resolving its paths."""
    index = 0
    while index < len(lines):
        if lines[index].startswith("diff --git "):
            if not lines[index][11:].strip():
                raise ReviewEvidenceError("diff has an empty Git file header")
            index += 1
            while index < len(lines) and _GIT_METADATA.fullmatch(lines[index]):
                index += 1
        if (
            index + 2 >= len(lines)
            or not lines[index].startswith("--- ") or len(lines[index]) <= 4
            or not lines[index + 1].startswith("+++ ") or len(lines[index + 1]) <= 4
        ):
            raise ReviewEvidenceError("diff must contain supported unified text file headers")
        index += 2
        hunks = 0
        while index < len(lines) and (hunk := _HUNK.fullmatch(lines[index])):
            index += 1
            hunks += 1
            old_line, old_count, new_line, new_count = hunk.groups()
            old, new = int(old_count or "1"), int(new_count or "1")
            if (old and int(old_line) == 0) or (new and int(new_line) == 0):
                raise ReviewEvidenceError("diff hunk has an invalid start line")
            changed = False
            while old or new:
                prefix = lines[index][:1] if index < len(lines) else ""
                if prefix not in {" ", "-", "+"}:
                    raise ReviewEvidenceError("diff hunk body does not match its header")
                old -= prefix != "+"
                new -= prefix != "-"
                if old < 0 or new < 0:
                    raise ReviewEvidenceError("diff hunk body exceeds its declared counts")
                changed |= prefix != " "
                index += 1
                if index < len(lines) and lines[index] == r"\ No newline at end of file":
                    index += 1
            if not changed:
                raise ReviewEvidenceError("diff hunk has no changes")
        if hunks == 0:
            raise ReviewEvidenceError("diff file has no text hunks")


def _sources(bundle: Path, entries: Any) -> dict[str, _Source]:
    sources: dict[str, _Source] = {}
    paths = {bundle.name.casefold()}
    total = 0
    for value in _items(entries, "sources", minimum=2, maximum=MAX_SOURCES):
        source = _object(value, "source", "id path sha256 type version origin admission lines")
        sid = _pattern(source["id"], "source ID", _ID)
        if sid in sources or sid.endswith(".") or _DEVICE.fullmatch(sid.split(".")[0]):
            raise ReviewEvidenceError("duplicate or unsafe source ID")
        parts = _relative_path(source["path"])
        path_key = "/".join(parts).casefold()
        if path_key in paths:
            raise ReviewEvidenceError("duplicate source path or self-reference")
        paths.add(path_key)
        kind = _enum(source["type"], "source type", DIFF_TYPE, *RECORD_SCHEMAS)
        _integer(source["version"], "source version", 0, 0)
        _text(source["origin"], "source origin")
        _enum(source["admission"], "source admission", "SYNTHETIC", "REVIEWED_NONSENSITIVE")
        expected = _pattern(source["sha256"], "SHA-256", _HASH)
        bound = MAX_DIFF_BYTES if kind == DIFF_TYPE else MAX_JSON_BYTES
        raw = _read(bundle.parent.joinpath(*parts), sid, min(bound, MAX_TOTAL_BYTES - total))
        total += len(raw)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ReviewEvidenceError(f"source {sid}: SHA-256 mismatch")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReviewEvidenceError(f"source {sid}: expected UTF-8") from exc
        line_count = text.count("\n") + (not text.endswith("\n"))
        start, end = _items(source["lines"], "source line locator", minimum=2, maximum=2)
        _integer(start, "source first line", 1, line_count)
        _integer(end, "source last line", start, line_count)
        record = None
        if kind == DIFF_TYPE:
            if "\0" in text:
                raise ReviewEvidenceError("diff contains a NUL byte")
            _diff(split_lf_lines(text))
        else:
            record = _record(raw, sid)
            if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMAS[kind]:
                raise ReviewEvidenceError(f"source {sid}: unsupported record schema")
        sources[sid] = _Source({**source, "retained_path": f"sources/{sid}.txt"}, raw, record)
    return sources


def _intent(source: _Source, target: dict[str, Any]) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    record = _object(source.record, "intent", "schema target by summary criteria", "expected_steps")
    if _target(record["target"]) != target:
        raise ReviewEvidenceError("intent target does not match bundle target")
    by = _text(record["by"], "intent attribution")
    intent = {
        "summary": _text(record["summary"], "intent summary"),
        "by": by,
        "source": source.at("/summary"),
    }
    criteria: dict[str, dict] = {}
    steps: dict[str, dict] = {}
    for index, item in enumerate(_items(record["criteria"], "criteria", minimum=1)):
        value = _object(item, "criterion", "id text")
        cid = _pattern(value["id"], "criterion ID", _ID)
        if cid in criteria:
            raise ReviewEvidenceError("duplicate criterion ID")
        criteria[cid] = {
            "id": cid, "text": _text(value["text"], "criterion text"), "by": by,
            "source": source.at(f"/criteria/{index}"),
            "check_source_ids": [], "stale_check_source_ids": [],
        }
    for index, item in enumerate(_items(record.get("expected_steps", []), "expected steps")):
        value = _object(item, "expected step", "id text required")
        sid = _pattern(value["id"], "expected step ID", _ID)
        if sid in steps or type(value["required"]) is not bool:
            raise ReviewEvidenceError("duplicate step ID or invalid required flag")
        steps[sid] = {
            "id": sid, "text": _text(value["text"], "step text"), "by": by,
            "required": value["required"], "source": source.at(f"/expected_steps/{index}"),
        }
    return intent, criteria, steps


def _check(source: _Source, target: dict[str, Any], criteria: dict[str, dict]) -> dict:
    receipt = _object(
        source.record, "validation receipt", "schema target check_id result reported_by assessment",
    )
    bound = _target(receipt["target"])
    if any(bound[key] != target[key] for key in ("repo", "pr", "task", "base")):
        raise ReviewEvidenceError("receipt repo, PR, task, or base does not match bundle target")
    assessment = _object(receipt["assessment"], "receipt assessment", "by relevance criterion_ids")
    ids = [
        _pattern(value, "mapped criterion ID", _ID)
        for value in _items(assessment["criterion_ids"], "criterion mappings")
    ]
    relevance = _enum(assessment["relevance"], "relevance", "RELEVANT", "IRRELEVANT", "UNASSESSED")
    if len(set(ids)) != len(ids) or not set(ids) <= criteria.keys() or bool(ids) != (relevance == "RELEVANT"):
        raise ReviewEvidenceError("receipt relevance or criterion mappings are inconsistent")
    stale = bound["head"] != target["head"]
    return {
        "id": _pattern(receipt["check_id"], "check ID", _ID),
        "head": bound["head"],
        "binding": "STALE" if stale else "CURRENT",
        "reported_result": _enum(receipt["result"], "reported check result", "PASS", "FAIL"),
        "reported_by": _text(receipt["reported_by"], "check reporter"),
        "assessment": {
            "by": _text(assessment["by"], "relevance attribution"),
            "relevance": relevance, "criterion_ids": ids,
        },
        "coverage_eligible": not stale and relevance == "RELEVANT",
        "source": source.at(""),
    }


def _association(value: Any) -> dict:
    declared = _object(value, "association", "by session_id", "attempt_id")
    _text(declared["by"], "association attribution")
    _pattern(declared["session_id"], "session ID", _ID)
    if "attempt_id" in declared:
        _pattern(declared["attempt_id"], "attempt ID", _ID)
    return {"declaration": declared, "matching_summary": None}


def _summary(
    source: _Source | None, target: dict[str, Any], association: dict | None,
) -> tuple[dict, dict[str, dict]]:
    capture = {"completeness": "UNKNOWN", "limits": [], "reported_by": None, "source": None}
    observations: dict[str, dict] = {}
    if source is None:
        return capture, observations
    if association is None:
        raise ReviewEvidenceError("session summary requires an explicit association")
    record = _object(
        source.record, "session summary", "schema target session_id reported_by capture observations",
        "attempt_id",
    )
    session_id = _pattern(record["session_id"], "summary session ID", _ID)
    attempt_id = _pattern(record["attempt_id"], "summary attempt ID", _ID) if "attempt_id" in record else None
    declaration = association["declaration"]
    if (
        _target(record["target"]) != target
        or session_id != declaration["session_id"] or attempt_id != declaration.get("attempt_id")
    ):
        raise ReviewEvidenceError("session summary identifiers do not match the declared association")
    association["matching_summary"] = source.at("")
    reporter = _text(record["reported_by"], "summary reporter")
    captured = _object(record["capture"], "capture", "completeness limits")
    capture = {
        "completeness": _enum(captured["completeness"], "capture completeness", "COMPLETE", "PARTIAL", "UNKNOWN"),
        "limits": [_text(limit, "capture limit") for limit in _items(captured["limits"], "capture limits")],
        "reported_by": reporter, "source": source.at("/capture"),
    }
    for index, item in enumerate(_items(record["observations"], "observations")):
        value = _object(item, "observation", "step_id note")
        sid = _pattern(value["step_id"], "observed step ID", _ID)
        if sid in observations:
            raise ReviewEvidenceError("duplicate observed step ID")
        observations[sid] = {
            "step_id": sid, "note": _text(value["note"], "observation note"),
            "reported_by": reporter, "source": source.at(f"/observations/{index}"),
        }
    return capture, observations


def _billing(attempt: dict, source: _Source, index: int) -> dict:
    billing = {"completeness": "UNKNOWN", "currency": None, "amount_minor": None, "source": None}
    if "billing" in attempt:
        value = _object(attempt["billing"], "billing", "completeness", "amount_minor currency")
        state = _enum(value["completeness"], "billing completeness", "COMPLETE", "PARTIAL", "UNKNOWN")
        billing.update(completeness=state, source=source.at(f"/attempts/{index}/billing"))
        if state == "COMPLETE":
            billing["currency"] = _pattern(value.get("currency"), "billing currency", _CURRENCY)
            billing["amount_minor"] = _integer(value.get("amount_minor"), "billing amount", 0)
        elif value.keys() != {"completeness"}:
            raise ReviewEvidenceError("incomplete billing must not claim a total or currency")
    return billing


def _aggregate_cost(attempts: list, completeness: str) -> dict:
    totals = [attempt["billing"] for attempt in attempts]
    currencies = {total["currency"] for total in totals}
    if completeness != "COMPLETE":
        reason = "INVENTORY_NOT_COMPLETE"
    elif any(total["completeness"] != "COMPLETE" for total in totals):
        reason = "BILLING_NOT_COMPLETE"
    elif len(currencies) != 1:
        reason = "MIXED_CURRENCY"
    else:
        amount = sum(total["amount_minor"] for total in totals)
        if amount <= MAX_SAFE_INT:
            return {"completeness": "COMPLETE", "currency": currencies.pop(), "amount_minor": amount, "reason": None}
        reason = "TOTAL_OUT_OF_RANGE"
    return {"completeness": "UNKNOWN", "currency": None, "amount_minor": None, "reason": reason}


def _attempt_inventory(source: _Source | None, target: dict, association: dict | None) -> dict:
    attempts = []
    seen = set()
    reporter = None
    completeness = "UNKNOWN"
    if source is not None:
        record = _object(source.record, "attempt inventory", "schema target by completeness attempts")
        if _target(record["target"]) != target:
            raise ReviewEvidenceError("attempt inventory target does not match bundle target")
        reporter = _text(record["by"], "attempt inventory attribution")
        completeness = _enum(record["completeness"], "inventory completeness", "COMPLETE", "PARTIAL", "UNKNOWN")
        for index, item in enumerate(_items(record["attempts"], "attempt inventory", minimum=1)):
            value = _object(item, "attempt", "id", "reported_outcome billing")
            aid = _pattern(value["id"], "inventory attempt ID", _ID)
            if aid in seen:
                raise ReviewEvidenceError("duplicate attempt ID")
            seen.add(aid)
            attempts.append({
                "id": aid,
                "reported_outcome": _enum(value.get("reported_outcome", "UNKNOWN"), "attempt outcome",
                                          "PASS", "FAIL", "UNKNOWN"),
                "source": source.at(f"/attempts/{index}"),
                "billing": _billing(value, source, index),
            })
        declared_id = association["declaration"].get("attempt_id") if association else None
        if completeness == "COMPLETE" and declared_id is not None and declared_id not in seen:
            raise ReviewEvidenceError("complete inventory does not include the declared attempt")
    return {
        "completeness": completeness,
        "by": reporter,
        "source": source.at("") if source else None,
        "attempts": attempts,
        "aggregate_cost": _aggregate_cost(attempts, completeness),
    }


def _packet(bundle_raw: bytes, bundle: Path) -> tuple[dict, list[_Source]]:
    manifest = _object(_record(bundle_raw, "bundle"), "bundle", "schema target sources", "association")
    _enum(manifest["schema"], "bundle schema", INPUT_SCHEMA)
    target = _target(manifest["target"])
    if target["base"] == target["head"]:
        raise ReviewEvidenceError("a retained change needs different base and head IDs")
    by_id = _sources(bundle, manifest["sources"])
    sources = [by_id[sid] for sid in sorted(by_id)]
    typed = {
        kind: [source for source in sources if source.metadata["type"] == kind]
        for kind in (DIFF_TYPE, *RECORD_SCHEMAS)
    }
    if len(typed[DIFF_TYPE]) != 1 or len(typed["intent"]) != 1:
        raise ReviewEvidenceError("expected exactly one diff and one intent")
    if len(typed["session-summary"]) > 1 or len(typed["attempt-inventory"]) > 1:
        raise ReviewEvidenceError("expected at most one session summary and one attempt inventory")
    diff = typed[DIFF_TYPE][0]
    intent, criteria, steps = _intent(typed["intent"][0], target)
    association = _association(manifest["association"]) if "association" in manifest else None
    summaries = typed["session-summary"]
    capture, observations = _summary(summaries[0] if summaries else None, target, association)
    inventories = typed["attempt-inventory"]
    inventory = _attempt_inventory(inventories[0] if inventories else None, target, association)

    checks = []
    gaps = []
    seen = set()
    results = set()
    for source in typed["validation-receipt"]:
        check = _check(source, target, criteria)
        key = (check["id"], check["head"])
        if key in seen:
            raise ReviewEvidenceError("duplicate check ID for the same head")
        seen.add(key)
        checks.append(check)
        if check["binding"] == "STALE":
            gaps.append({"code": "STALE_CHECK", "subject_id": check["id"], "sources": [check["source"]]})
            for cid in check["assessment"]["criterion_ids"]:
                criteria[cid]["stale_check_source_ids"].append(source.id)
        else:
            results.add(check["reported_result"])
            if check["coverage_eligible"]:
                for cid in check["assessment"]["criterion_ids"]:
                    criteria[cid]["check_source_ids"].append(source.id)

    for cid, criterion in criteria.items():
        covered = bool(criterion["check_source_ids"])
        criterion["status"] = "COVERED" if covered else "MISSING"
        if not covered:
            gaps.append({
                "code": "CRITERION_UNCOVERED", "subject_id": cid,
                "sources": [
                    criterion["source"],
                    *(by_id[sid].at("") for sid in criterion["stale_check_source_ids"]),
                ],
            })
    for sid, step in steps.items():
        observation = observations.get(sid)
        step["status"] = "OBSERVED" if observation else "UNOBSERVED"
        step["observation_source"] = observation["source"] if observation else None
        if observation is None and step["required"]:
            gaps.append({
                "code": "STEP_UNOBSERVED", "subject_id": sid,
                "sources": [step["source"]] + ([capture["source"]] if capture["source"] else []),
            })
    missing = any(gap["code"] != "STALE_CHECK" for gap in gaps)
    return {
        "schema": PACKET_SCHEMA,
        "integrity": "VALID",
        "coverage": "INCONCLUSIVE" if missing else "COMPLETE",
        "observed_validation": "MIXED" if len(results) == 2 else next(iter(results), "UNKNOWN"),
        "manifest_sha256": hashlib.sha256(bundle_raw).hexdigest(),
        "target": target,
        "diff": {"binding": "DECLARED", "source": {"source_id": diff.id, "lines": diff.metadata["lines"]}},
        "intent": intent,
        "criteria": list(criteria.values()),
        "checks": checks,
        "association": association,
        "capture": capture,
        "steps": list(steps.values()),
        "observations": list(observations.values()),
        "attempt_inventory": inventory,
        "gaps": gaps,
        "sources": [source.metadata for source in sources],
    }, sources


def _locator(source: dict | None) -> str:
    if source is None:
        return "none"
    if "lines" in source:
        start, end = source["lines"]
        return f"{source['source_id']} lines {start}-{end}"
    return f"{source['source_id']}#{source['pointer']}"


def _review(packet: dict) -> str:
    # Indented code blocks keep attributed text, including URLs and Markdown, inert.
    lines = [
        "# PR evidence review", "",
        "Do not use this report without `packet.json` in the same directory; assembly may be incomplete.",
        "VALID checks local integrity, not authenticity.",
        "Coverage is not acceptance; observed results are reported, not independently reproduced.",
    ]

    def section(title: str, entries: list[str]) -> None:
        lines.extend(["", f"## {title}", "", *("    " + inert(entry) for entry in entries or ["none"])])

    section("Status", [
        f"Integrity: {packet['integrity']}",
        f"Coverage: {packet['coverage']}",
        f"Observed validation (current receipts, including irrelevant ones): {packet['observed_validation']}",
        *(f"{key}: {packet['target'][key]}" for key in ("repo", "pr", "task", "base", "head")),
        f"Diff (declared binding): {_locator(packet['diff']['source'])}",
    ])
    intent = packet["intent"]
    section("Intent and criteria", [
        f"{intent['summary']} (attributed to {intent['by']}; {_locator(intent['source'])})",
        *(
            f"{row['id']}: {row['status']} - {row['text']} ({_locator(row['source'])}); "
            f"current evidence: {', '.join(row['check_source_ids']) or 'none'}; "
            f"stale evidence: {', '.join(row['stale_check_source_ids']) or 'none'}"
            for row in packet["criteria"]
        ),
    ])
    section("Checks", [
        f"{row['id']}: reported {row['reported_result']} by {row['reported_by']}; "
        f"{row['binding']} at {row['head']}; "
        f"{row['assessment']['relevance']} as assessed by {row['assessment']['by']}; "
        f"criteria: {', '.join(row['assessment']['criterion_ids']) or 'none'}; "
        f"counts toward coverage: {'yes' if row['coverage_eligible'] else 'no'} ({_locator(row['source'])})"
        for row in packet["checks"]
    ])
    association = packet["association"]
    capture = packet["capture"]
    association_lines = []
    if association:
        declaration = association["declaration"]
        association_lines = [
            f"Declared by {declaration['by']}: session {declaration['session_id']}; "
            f"attempt {declaration.get('attempt_id', 'not declared')}",
            f"Matching retained identifiers: {_locator(association['matching_summary'])} "
            "(not verification of an external identity or causal link)",
        ]
    section("Association and capture", [
        *(association_lines or ["No session association declared"]),
        f"Capture: {capture['completeness']}; reporter: {capture['reported_by'] or 'unknown'} "
        f"({_locator(capture['source'])})",
        *(f"Limit: {limit}" for limit in capture["limits"]),
        "Unobserved means no retained observation, not proof the step was skipped.",
    ])
    section("Expected steps", [
        f"{row['id']}: {row['status']} "
        f"({'required' if row['required'] else 'optional'}) - {row['text']} "
        f"({_locator(row['source'])}); observation: {_locator(row['observation_source'])}"
        for row in packet["steps"]
    ])
    section("Retained observations", [
        f"{row['step_id']}: {row['note']} (reported by {row['reported_by']}; {_locator(row['source'])})"
        for row in packet["observations"]
    ])
    def cost_amount(billing: dict) -> str:
        if billing["completeness"] != "COMPLETE":
            return "unknown"
        return f"{billing['amount_minor']} minor units {billing['currency']}"

    inventory = packet["attempt_inventory"]
    total = inventory["aggregate_cost"]
    section("Attempt billing", [
        f"Inventory: {inventory['completeness']}; attributed to {inventory['by'] or 'unknown'} "
        f"({_locator(inventory['source'])})",
        *(
            f"{row['id']}: reported outcome {row['reported_outcome']} ({_locator(row['source'])}); "
            f"billing {cost_amount(row['billing'])}, {row['billing']['completeness']} "
            f"({_locator(row['billing']['source'])})"
            for row in inventory["attempts"]
        ),
        f"Aggregate cost: {cost_amount(total)}" + (f" ({total['reason']})" if total["reason"] else ""),
    ])
    section("Evidence gaps and stale receipts", [
        f"{row['code']}: {row['subject_id']} ({', '.join(map(_locator, row['sources']))})"
        for row in packet["gaps"]
    ])
    section("Retained sources", [
        f"{row['id']}: {row['retained_path']}; input {row['path']}; {row['type']} v{row['version']}; "
        f"lines {row['lines'][0]}-{row['lines'][1]}; {row['admission']} from {row['origin']}; "
        f"sha256 {row['sha256']}"
        for row in packet["sources"]
    ])
    return "\n".join(lines) + "\n"


def _publish(output: Path, packet: dict, sources: list[_Source]) -> None:
    packet_text = json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    review = _review(packet)
    _directory(output.parent)
    output.mkdir(mode=0o700)  # Exclusive reservation; never replace even an empty directory.
    complete = False
    try:
        retained = output / "sources"
        retained.mkdir(mode=0o700)
        for source in sources:
            with (output / source.metadata["retained_path"]).open("xb") as stream:
                stream.write(source.raw)
                stream.flush()
                os.fsync(stream.fileno())
        fsync_directory(retained)
        write_atomic(output / "review.md", review, mode=0o600)
        fsync_directory(output.parent)
        # Publish the success marker last. An abrupt interruption can leave an
        # incomplete directory, but never a visible packet with missing sources.
        write_atomic(output / "packet.json", packet_text, mode=0o600)
        complete = True
    finally:
        if not complete:
            (output / "packet.json").unlink(missing_ok=True)
            shutil.rmtree(output)


def assemble_review_evidence(bundle_path: str | Path, output_dir: str | Path) -> dict:
    """Validate, assemble and publish a packet without inspecting a repo or running its evidence."""
    bundle, output = Path(bundle_path).absolute(), Path(output_dir).absolute()
    if ".." in bundle.parts or ".." in output.parts or not output.name:
        raise ReviewEvidenceError("input and output paths must not traverse parent directories")
    raw = _read(bundle, "bundle", MAX_JSON_BYTES)
    packet, sources = _packet(raw, bundle)
    _publish(output, packet, sources)
    return packet


def cmd_review_evidence(args: argparse.Namespace) -> int:
    try:
        packet = assemble_review_evidence(args.bundle, args.output)
    except FileExistsError:
        print("review-evidence: output already exists", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"review-evidence: could not read or assemble evidence: {inert(exc.strerror)}", file=sys.stderr)
        return 1
    except (ReviewEvidenceError, KeyboardInterrupt) as exc:
        print(f"review-evidence: {inert(str(exc) or 'interrupted')}", file=sys.stderr)
        return 1
    print(f"VALID / {packet['coverage']} / {packet['observed_validation']}")
    return 0 if packet["coverage"] == "COMPLETE" else 2
