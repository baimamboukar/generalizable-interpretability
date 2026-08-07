from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "external" / "prop-probes-iclr"
MANIFEST = json.loads((Path(__file__).parent / "upstream.json").read_text())
DATASETS = {
    "synth": SOURCE
    / "exports/datasets/name_country_food_occupation_basic/dataset.json",
    "para": SOURCE
    / "exports/datasets/name_country_food_occupation_basic/paraphrase/dataset.json",
    "trans": SOURCE
    / "exports/datasets/name_country_food_occupation_basic/es_translation/dataset.json",
}


def commit() -> str | None:
    if not (SOURCE / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()


def inspect() -> None:
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "commit": commit(),
                "expected_commit": MANIFEST["commit"],
                "baseline": {
                    "model": MANIFEST["model"],
                    "table": 1,
                    "settings": ["synth", "para", "trans"],
                    "metrics": ["exact_match", "jaccard"],
                    "domain_probe_layer": 20,
                    "binding_layer": 15,
                },
                "released_artifacts": {
                    "domain_probes": True,
                    "hessian_binding_subspace": True,
                    "archive_bytes": MANIFEST["artifact_bytes"],
                    "archive_sha256": MANIFEST["artifact_sha256"],
                },
                "datasets": {
                    name: {"path": str(path), "exists": path.exists()}
                    for name, path in DATASETS.items()
                },
                "blocker": (
                    "The exact para and trans datasets are absent from the "
                    "repository and released artifact archive."
                ),
            },
            indent=2,
        )
    )


def baseline(config: Path) -> None:
    if commit() != MANIFEST["commit"]:
        raise RuntimeError(f"Expected upstream commit {MANIFEST['commit']}")
    missing = [name for name, path in DATASETS.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing exact released datasets: {', '.join(missing)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE / "src")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coref.probes.evaluate",
            "--config",
            str(config.resolve()),
        ],
        cwd=SOURCE,
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inspect", "baseline"), required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.stage == "inspect":
        inspect()
        return
    if not args.config:
        parser.error("--config is required for the official baseline")
    baseline(args.config)


if __name__ == "__main__":
    main()
