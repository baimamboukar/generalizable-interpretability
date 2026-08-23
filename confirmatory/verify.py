from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "confirmatory" / "protocol.json"
REGISTRY = ROOT / "confirmatory" / "study_registry.csv"
HYPOTHESES = ROOT / "confirmatory" / "hypotheses.csv"
HOLDOUT_PROTOCOL = ROOT / "confirmatory" / "holdout_protocol.json"


def verify() -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text())
    holdout_protocol = json.loads(HOLDOUT_PROTOCOL.read_text())
    errors: list[str] = []
    reports = []
    report_groups = (
        "primary_anchors",
        "positive_controls",
        "baseline_reproducibility_only",
        "supporting_evidence",
    )
    for group in report_groups:
        for study in protocol["corpus"][group]:
            path = ROOT / study["report"]
            reports.append(str(path.relative_to(ROOT)))
            if not path.is_file():
                errors.append(f"missing report: {study['report']}")
    for model in protocol["confirmatory_holdouts"]:
        revision = model["revision"]
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            errors.append(f"unpinned holdout revision: {model['id']}")
        if model["status"] != "sealed_no_task_evaluation":
            errors.append(f"holdout is not sealed: {model['id']}")
    root_holdouts = {
        model["id"]: model["revision"] for model in protocol["confirmatory_holdouts"]
    }
    adapter_holdouts = {
        model["model_id"]: model["revision"]
        for model in holdout_protocol["models"].values()
    }
    if root_holdouts != adapter_holdouts:
        errors.append("holdout adapter registry differs from the root protocol")
    if holdout_protocol["status"] != "adapters_specified_before_task_outcomes":
        errors.append("holdout adapter protocol is not frozen")
    if holdout_protocol["primary_model"] not in root_holdouts:
        errors.append("primary holdout is not in the sealed registry")
    if holdout_protocol["secondary_model"] not in root_holdouts:
        errors.append("secondary holdout is not in the sealed registry")
    for study, adapter in holdout_protocol["study_adapters"].items():
        if "path" not in adapter:
            if adapter.get("status") != "blocked_until_retrospective_analysis_is_locked":
                errors.append(f"unresolved holdout adapter: {study}")
            continue
        path = ROOT / adapter["path"]
        if not path.is_file():
            errors.append(f"missing holdout adapter: {adapter['path']}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != adapter["sha256"]:
            errors.append(f"holdout adapter hash mismatch: {study}")
    with REGISTRY.open(newline="") as handle:
        registry = list(csv.DictReader(handle))
    registered = {row["study_id"] for row in registry}
    expected = {
        study["id"]
        for group in report_groups
        for study in protocol["corpus"][group]
    } | set(protocol["corpus"]["artifact_audit_only"])
    if registered != expected:
        errors.append(
            f"registry mismatch: missing={sorted(expected - registered)} extra={sorted(registered - expected)}"
        )
    with HYPOTHESES.open(newline="") as handle:
        hypotheses = list(csv.DictReader(handle))
    hypothesis_ids = [row["hypothesis_id"] for row in hypotheses]
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        errors.append("duplicate hypothesis_id")
    anchor_ids = {study["id"] for study in protocol["corpus"]["primary_anchors"]}
    hypothesis_studies = {row["study"] for row in hypotheses}
    if not anchor_ids.issubset(hypothesis_studies):
        errors.append(
            f"anchors without hypotheses: {sorted(anchor_ids - hypothesis_studies)}"
        )
    allowed_tiers = {"retrospective_reanalysis", "prospective_confirmatory"}
    invalid_tiers = sorted(
        {row["analysis_tier"] for row in hypotheses} - allowed_tiers
    )
    if invalid_tiers:
        errors.append(f"invalid analysis tiers: {invalid_tiers}")
    result = {
        "schema_version": protocol["schema_version"],
        "frozen_at": protocol["frozen_at"],
        "registered_studies": len(registry),
        "frozen_hypotheses": len(hypotheses),
        "verified_reports": len(reports),
        "sealed_holdouts": len(protocol["confirmatory_holdouts"]),
        "holdout_adapter_protocol": HOLDOUT_PROTOCOL.name,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
