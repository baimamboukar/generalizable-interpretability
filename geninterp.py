#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "data/benchmark.json"

ARTIFACT_GROUPS = [
    "implementation",
    "probes",
    "predictions",
    "metrics",
    "logs",
    "report",
    "trace",
]
EFFICIENCY_FIELDS = [
    "agent_cost_usd",
    "api_time_seconds",
    "wall_time_seconds",
    "mcp_build_time_seconds",
    "human_interventions",
]


def load_json(path: Path) -> dict:
    with path.open(encoding = "utf-8") as handle:
        return json.load(handle)


def benchmark_record(task_id: str, protocol_path: Path = BENCHMARK) -> dict:
    protocol = load_json(protocol_path)
    for task in protocol["tasks"]:
        if task["id"] == task_id:
            return task
    raise ValueError(f"Unknown benchmark task: {task_id}")


def benchmark_task(task_id: str, protocol_path: Path = BENCHMARK) -> dict:
    task = benchmark_record(task_id, protocol_path)
    public = {
        "id": task["id"],
        "title": task["title"],
        "type": task.get("type", "reproduce"),
        "status": task.get("status", "development"),
        "split": task.get("split", "development"),
        **task["public"],
        "submission_contract": {
            "result_ids": [item["id"] for item in task["evaluation"]["results"]],
            "artifact_groups": ARTIFACT_GROUPS,
            "efficiency_fields": EFFICIENCY_FIELDS,
        },
    }
    return public


def ratio(part: int, total: int) -> float:
    return round(part / total, 4) if total else 1.0


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(artifacts: object, root: Path) -> dict:
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    root = root.resolve()
    details = []

    for group in ARTIFACT_GROUPS:
        paths = artifacts.get(group, [])
        valid_paths = []
        if isinstance(paths, list):
            for item in paths:
                if not isinstance(item, str):
                    continue
                path = (root / item).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    continue
                if path.stat().st_size == 0:
                    continue
                valid_paths.append(
                    {
                        "path": str(path.relative_to(root)),
                        "sha256": file_sha256(path),
                    }
                )
        details.append(
            {
                "group": group,
                "passed": bool(valid_paths),
                "files": valid_paths,
            }
        )

    passed = sum(item["passed"] for item in details)
    return {
        "score": ratio(passed, len(details)),
        "passed": passed == len(details),
        "present": passed,
        "required": len(details),
        "details": details,
    }


def validate_protocol(task: dict, submission: dict) -> dict:
    protocol = submission.get("protocol", {})
    errors = []
    results = submission.get("results", {})

    if protocol.get("result_count") != len(results):
        errors.append("protocol.result_count does not match submitted results")
    if not isinstance(protocol.get("scope"), str) or not protocol["scope"].strip():
        errors.append("protocol.scope must be a non-empty string")
    if not isinstance(protocol.get("input_hash"), str) or not protocol["input_hash"].strip():
        errors.append("protocol.input_hash must be a non-empty string")

    public = task["public"]
    expected_hash = public.get("input_sha256")
    if expected_hash and protocol.get("input_hash") != expected_hash:
        errors.append("protocol.input_hash does not match the task input")
    counts = protocol.get("counts", {})
    if not isinstance(counts, dict):
        errors.append("protocol.counts must be an object")
        counts = {}
    for name, expected in public.get("expected_counts", {}).items():
        if counts.get(name) != expected:
            errors.append(f"protocol.counts.{name} does not match the task scope")
    for name, minimum in public.get("minimum_counts", {}).items():
        if not isinstance(counts.get(name), int) or counts[name] < minimum:
            errors.append(f"protocol.counts.{name} is below the task scope")

    return {"passed": not errors, "errors": errors, "reported": protocol}


def validate_efficiency(efficiency: object) -> dict:
    efficiency = efficiency if isinstance(efficiency, dict) else {}
    errors = []
    for field in EFFICIENCY_FIELDS:
        value = efficiency.get(field)
        if not is_number(value) or value < 0:
            errors.append(f"efficiency.{field} must be a non-negative number")
    interventions = efficiency.get("human_interventions")
    if is_number(interventions) and int(interventions) != interventions:
        errors.append("efficiency.human_interventions must be an integer")
    return {"passed": not errors, "errors": errors, "reported": efficiency}


def score_audit(task: dict, submission: dict, audit_map: dict | None) -> dict:
    expected = {item["id"] for item in task["evaluation"]["audit_items"]}
    findings = submission.get("audit_findings", [])
    if not isinstance(findings, list):
        raise ValueError("audit_findings must be a list")
    if not expected:
        return {
            "status": "not_applicable",
            "precision": None,
            "recall": None,
            "f1": None,
            "passed": None,
        }
    if not task["evaluation"].get("audit_required", True):
        return {
            "status": "not_evaluated",
            "precision": None,
            "recall": None,
            "f1": None,
            "passed": None,
        }
    if audit_map is None:
        return {
            "status": "pending_mapping",
            "precision": None,
            "recall": None,
            "f1": None,
            "passed": False,
        }

    matched_findings = set()
    detected_items = set()
    for match in audit_map.get("matches", []):
        finding = match.get("finding")
        item = match.get("audit_item")
        if not isinstance(finding, int) or finding < 0 or finding >= len(findings):
            raise ValueError("Audit map contains an invalid finding index")
        if item not in expected:
            raise ValueError(f"Audit map contains an unknown audit item: {item}")
        matched_findings.add(finding)
        detected_items.add(item)

    unsupported = len(findings) - len(matched_findings)
    precision = ratio(len(matched_findings), len(findings))
    recall = ratio(len(detected_items), len(expected))
    f1 = 0.0 if precision + recall == 0 else round(
        2 * precision * recall / (precision + recall),
        4,
    )
    threshold = task["evaluation"].get("audit_f1_min", 0.5)
    return {
        "status": "scored",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "passed": f1 >= threshold,
        "detected": len(detected_items),
        "expected": len(expected),
        "unsupported": unsupported,
        "threshold": threshold,
    }


def score_submission(
    task: dict,
    submission: dict,
    submission_root: Path,
    audit_map: dict | None = None,
) -> dict:
    if task.get("status") == "candidate":
        raise ValueError("Candidate task is not ready for evaluation")
    if submission.get("task_id") != task["id"]:
        raise ValueError("Submission task_id does not match the selected task")

    evaluation = task["evaluation"]
    expected_results = {item["id"]: item for item in evaluation["results"]}
    submitted_results = submission.get("results", {})
    if not isinstance(submitted_results, dict):
        raise ValueError("results must be an object")
    unknown_results = set(submitted_results) - set(expected_results)
    if unknown_results:
        raise ValueError(f"Unknown result ids: {', '.join(sorted(unknown_results))}")

    result_details = []
    for result_id, expected in expected_results.items():
        actual = submitted_results.get(result_id)
        tolerance = expected.get("tolerance", evaluation["result_tolerance"])
        delta = actual - expected["target"] if is_number(actual) else None
        passed = delta is not None and abs(delta) <= tolerance
        result_details.append(
            {
                "id": result_id,
                "actual": actual,
                "target": expected["target"],
                "tolerance": tolerance,
                "delta": None if delta is None else round(delta, 6),
                "passed": passed,
            }
        )

    passed_results = sum(item["passed"] for item in result_details)
    complete_deltas = [abs(item["delta"]) for item in result_details if item["delta"] is not None]
    scientific_pass = passed_results == len(result_details)
    scientific = {
        "score": ratio(passed_results, len(result_details)),
        "passed": scientific_pass,
        "matched": passed_results,
        "total": len(result_details),
        "mean_absolute_error": (
            round(sum(complete_deltas) / len(complete_deltas), 6)
            if len(complete_deltas) == len(result_details)
            else None
        ),
        "details": result_details,
    }

    protocol = validate_protocol(task, submission)
    evidence = verify_artifacts(submission.get("artifacts"), submission_root)
    efficiency = validate_efficiency(submission.get("efficiency"))
    audit = score_audit(task, submission, audit_map)
    audit_pass = audit["passed"] in (None, True)

    return {
        "task_id": task["id"],
        "task_type": task.get("type", "reproduce"),
        "scientific": scientific,
        "protocol": protocol,
        "audit": audit,
        "evidence": evidence,
        "efficiency": efficiency,
        "task_pass": all(
            [
                scientific_pass,
                protocol["passed"],
                evidence["passed"],
                efficiency["passed"],
                audit_pass,
            ]
        ),
    }


def show_task(task_id: str, protocol_path: Path = BENCHMARK) -> int:
    print(json.dumps(benchmark_task(task_id, protocol_path), indent = 2))
    return 0


def evaluate_benchmark(
    task_id: str,
    submission_path: Path,
    audit_map_path: Path | None = None,
    protocol_path: Path = BENCHMARK,
) -> int:
    task = benchmark_record(task_id, protocol_path)
    submission = load_json(submission_path)
    audit_map = load_json(audit_map_path) if audit_map_path else None
    score = score_submission(task, submission, submission_path.parent, audit_map)
    print(json.dumps(score, indent = 2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog = "geninterp")
    commands = parser.add_subparsers(dest = "command", required = True)
    benchmark = commands.add_parser("benchmark")
    benchmark_commands = benchmark.add_subparsers(
        dest = "benchmark_command",
        required = True,
    )

    show = benchmark_commands.add_parser("show")
    show.add_argument("task_id")
    show.add_argument("--protocol", type = Path, default = BENCHMARK)

    evaluate = benchmark_commands.add_parser("evaluate")
    evaluate.add_argument("task_id")
    evaluate.add_argument("submission", type = Path)
    evaluate.add_argument("--audit-map", type = Path)
    evaluate.add_argument("--protocol", type = Path, default = BENCHMARK)
    args = parser.parse_args()

    try:
        if args.command == "benchmark":
            if args.benchmark_command == "show":
                return show_task(args.task_id, args.protocol)
            if args.benchmark_command == "evaluate":
                return evaluate_benchmark(
                    args.task_id,
                    args.submission,
                    args.audit_map,
                    args.protocol,
                )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"geninterp: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
