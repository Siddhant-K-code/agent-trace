"""Normalize the local synthetic greeting packet without reading any task labels."""

from __future__ import annotations

import hashlib
import json

from score import require_object, unique_object


STEP = "review:greeting"


def matching_rows(value: object, field: str, key: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"packet field for {field!r} rows must be an array")
    rows = [require_object(entry, "packet row") for entry in value]
    return [row for row in rows if row.get(field) == key]


def locator(value: object, where: str) -> dict[str, object]:
    obj = require_object(value, where)
    source_id, pointer = obj.get("source_id"), obj.get("pointer")
    if (set(obj) != {"source_id", "pointer"} or not isinstance(source_id, str) or not source_id
            or not isinstance(pointer, str) or not pointer.startswith("/")):
        raise ValueError(f"{where} must contain a source ID and a JSON pointer")
    return obj


def normalize_packet(
    raw: bytes, binding: dict[str, object], evidence: dict[str, object],
) -> dict[str, object]:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != binding.get("packet_sha256"):
        raise ValueError("packet SHA-256 does not match the supplied binding")
    packet = require_object(
        json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object), "packet",
    )
    if packet.get("schema") != "review-evidence-packet/v0" or packet.get("integrity") != "VALID":
        raise ValueError("expected a VALID review-evidence-packet/v0")
    target = require_object(binding.get("target"), "bound target")
    if json.dumps(packet.get("target"), sort_keys=True) != json.dumps(target, sort_keys=True):
        raise ValueError("packet target does not match the supplied binding")

    bound_row = require_object(binding.get("row"), "bound inspect row")
    intent_source = locator(bound_row.get("source"), "bound step locator")
    if "observation_source" not in bound_row:
        raise ValueError("bound inspect row must specify an observation locator or null")
    expected_observation = bound_row["observation_source"]
    if expected_observation is not None:
        expected_observation = locator(expected_observation, "bound observation locator")
    rows = matching_rows(packet.get("steps"), "id", "inspect")
    if (
        bound_row.get("id") != "inspect" or len(rows) != 1
        or rows[0].get("source") != intent_source
        or rows[0].get("observation_source") != expected_observation
        or "observation_source" not in rows[0]
    ):
        raise ValueError("packet inspect row or locator does not match the supplied binding")
    observations = matching_rows(packet.get("observations"), "step_id", "inspect")
    if expected_observation is not None:
        if (rows[0].get("status") != "OBSERVED" or len(observations) != 1
                or observations[0].get("source") != expected_observation):
            raise ValueError("packet inspect observation does not match the bound row")
    elif rows[0].get("status") != "UNOBSERVED" or observations:
        raise ValueError("packet inspect observation conflicts with its bound row")

    if evidence.get("admission") != "SYNTHETIC" or evidence.get("packet_sha256") != digest:
        raise ValueError("independent evidence must be synthetic and reference the bound packet SHA-256")
    artifact = evidence.get("artifact")
    if artifact is not None:
        require_object(artifact, "independent artifact")
    step = require_object(evidence.get("inspection", {}), "independent inspection evidence")
    observed = []
    sufficient = []
    source = step.get("source")
    if step.get("kind") == "greeting-inspection" and isinstance(source, str) and source.strip():
        if (
            step.get("state") == "observed" and expected_observation is not None
            and step.get("corroborates") == expected_observation
        ):
            observed.append(STEP)
        elif (
            step.get("state") == "absent" and step.get("capture") == "sufficient"
            and expected_observation is None
        ):
            sufficient.append(STEP)
    return {"artifact": artifact, "observed_steps": observed, "sufficient_capture_for": sufficient}
