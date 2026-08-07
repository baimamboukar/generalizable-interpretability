from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "external" / "ICR_Probe"
MANIFEST = json.loads((Path(__file__).parent / "upstream.json").read_text())
REQUIRED = [
    "src/icr_score.py",
    "src/icr_probe.py",
    "src/utils.py",
    "src/config.py",
    "scripts/empirical_study.ipynb",
]
MISSING_PIPELINE = [
    "answer generation",
    "hallucination judging and labels",
    "model forward-pass cache creation",
    "ICR feature files",
    "baseline entrypoint",
    "saved checkpoints",
]


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
                "paper_baseline": MANIFEST["paper_settings"],
                "released_files": {
                    path: (SOURCE / path).exists() for path in REQUIRED
                },
                "missing_pipeline": MISSING_PIPELINE,
                "runnable": False,
            },
            indent=2,
        )
    )


def baseline() -> None:
    if commit() != MANIFEST["commit"]:
        raise RuntimeError(f"Expected upstream commit {MANIFEST['commit']}")
    raise RuntimeError(
        "The official release has no executable baseline pipeline. "
        f"Missing: {', '.join(MISSING_PIPELINE)}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inspect", "baseline"), required=True)
    args = parser.parse_args()
    {"inspect": inspect, "baseline": baseline}[args.stage]()


if __name__ == "__main__":
    main()
