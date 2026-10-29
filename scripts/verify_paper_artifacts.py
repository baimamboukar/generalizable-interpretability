#!/usr/bin/env python3
"""Verify the canonical paper tables and figures."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/results/paper-artifacts"
FIGURES = ROOT / "figures/paper-artifacts"
EXPECTED_TABLES = {
    "README.md",
    "artifact_readiness.csv",
    "artifact_coverage.csv",
    "baseline_fidelity.csv",
    "baseline_fidelity_summary.csv",
    "claim_ledger.csv",
    "cohort_accounting.csv",
    "cross_model_extensions.csv",
    "evaluation_contracts.csv",
    "experiment_gaps.csv",
    "open_experiment_status.csv",
    "reproducibility_audit.csv",
    "schema_audit.json",
    "study_inventory.csv",
    "study_verdicts.csv",
    "table_artifact_audits.tex",
    "table_empirical_status.tex",
    "table_evaluation_contracts.tex",
}
EXPECTED_FIGURES = {
    "figure1_study_accounting.png",
    "figure2_baseline_fidelity.png",
    "figure3_cross_model_portability.png",
    "figure4_evaluation_contracts.png",
}
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?:VAST_API_KEY|OPENAI_API_KEY|HF_TOKEN)\s*=", re.I),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_manifest() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["outputs"].items():
        path = DATA / name
        if not path.is_file() or sha256(path) != metadata["sha256"] or path.stat().st_size != metadata["size_bytes"]:
            raise ValueError(f"output hash mismatch: {name}")
    for relative, expected in manifest["source_snapshot"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source hash mismatch: {relative}")


def check_tables() -> dict[str, int]:
    present = {path.name for path in DATA.iterdir() if path.is_file()}
    missing = EXPECTED_TABLES - present
    if missing:
        raise ValueError(f"missing tables: {sorted(missing)}")

    inventory = csv_rows("study_inventory.csv")
    baselines = csv_rows("baseline_fidelity.csv")
    extensions = csv_rows("cross_model_extensions.csv")
    contracts = csv_rows("evaluation_contracts.csv")
    if len(inventory) != 23 or len({row["study_id"] for row in inventory}) != 23:
        raise ValueError("invalid inventory")
    if len(baselines) != 259 or len({row["study_id"] for row in baselines}) != 12:
        raise ValueError("invalid baseline table")
    if len(extensions) != 68 or len({row["study_id"] for row in extensions}) != 10:
        raise ValueError("invalid extension table")
    if len(contracts) != 12:
        raise ValueError("invalid contract table")
    for row in baselines:
        paper = float(row["paper_value"])
        replication = float(row["replication_value"])
        if not 0 <= paper <= 1 or not 0 <= replication <= 1:
            raise ValueError(f"metric outside fraction scale: {row['claim_id']}")
        if abs((replication - paper) - float(row["delta"])) > 1e-8:
            raise ValueError(f"delta mismatch: {row['claim_id']}")
    return {
        "studies": len(inventory),
        "baseline_rows": len(baselines),
        "baseline_studies": len({row["study_id"] for row in baselines}),
        "extension_rows": len(extensions),
        "extension_studies": len({row["study_id"] for row in extensions}),
    }


def check_figures() -> dict[str, dict[str, object]]:
    present = {path.name for path in FIGURES.glob("*.png")}
    if present != EXPECTED_FIGURES:
        raise ValueError(f"figure set mismatch: expected {sorted(EXPECTED_FIGURES)}, found {sorted(present)}")
    output = {}
    for name in sorted(EXPECTED_FIGURES):
        path = FIGURES / name
        with Image.open(path) as image:
            width, height = image.size
            if width < 1800 or height < 1000:
                raise ValueError(f"figure resolution too small: {name} {image.size}")
            rgb = image.convert("RGB")
            corners = [rgb.getpixel((0, 0)), rgb.getpixel((width - 1, 0)), rgb.getpixel((0, height - 1)), rgb.getpixel((width - 1, height - 1))]
            if any(min(pixel) < 248 for pixel in corners):
                raise ValueError(f"figure background is not white: {name} {corners}")
        output[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size, "width": width, "height": height}
    return output


def check_secrets(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix.lower() not in {".csv", ".json", ".md", ".tex", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"possible secret in {path.relative_to(ROOT)}")


def main() -> None:
    check_manifest()
    counts = check_tables()
    figures = check_figures()
    text_paths = [path for path in DATA.iterdir() if path.is_file()] + [
        ROOT / "scripts/build_paper_artifacts.py",
        ROOT / "scripts/plot_paper_artifacts.py",
        ROOT / "scripts/verify_paper_artifacts.py",
    ]
    check_secrets(text_paths)

    release = {
        "schema_version": "1.0.0",
        "status": "verified",
        "counts": counts,
        "tables": {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(DATA.iterdir())
            if path.is_file() and path.name not in {"manifest.json", "release_manifest.json"}
        },
        "figures": figures,
        "scripts": {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in text_paths[-3:]
        },
    }
    (DATA / "release_manifest.json").write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", **counts, "figures": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
    "claim_ledger.csv",
    "cohort_accounting.csv",
    "open_experiment_status.csv",
