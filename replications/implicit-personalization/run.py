from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "external" / "implicit-personalization-stereotypes"
MANIFEST = json.loads((Path(__file__).parent / "upstream.json").read_text())
MODELS = MANIFEST["models"]


def commit() -> str | None:
    if not (SOURCE / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect() -> None:
    files = {
        "conversations": SOURCE / "conversations_250.json",
        "stereotypes": SOURCE / "stereotypes.csv",
        "probe": SOURCE / "train_probe.py",
        "evaluation": SOURCE / "eval_conversations.py",
        "mitigation": SOURCE / "mitigate.py",
        "plots": SOURCE / "visualizations.ipynb",
    }
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "commit": commit(),
                "expected_commit": MANIFEST["commit"],
                "models": MODELS,
                "paper_settings": MANIFEST["paper_settings"],
                "released_files": {
                    name: path.exists() for name, path in files.items()
                },
                "data_hashes": {
                    name: sha256(path)
                    for name, path in files.items()
                    if name in {"conversations", "stereotypes"} and path.exists()
                },
                "released_results": False,
                "model_revisions_pinned": False,
                "baseline_runnable": True,
            },
            indent=2,
        )
    )


def verify() -> None:
    if commit() != MANIFEST["commit"]:
        raise RuntimeError(f"Expected upstream commit {MANIFEST['commit']}")
    for name, expected in MANIFEST["data"].items():
        path = SOURCE / {
            "conversations_sha256": "conversations_250.json",
            "stereotypes_sha256": "stereotypes.csv",
        }[name]
        if sha256(path) != expected:
            raise RuntimeError(f"Unexpected data hash: {path}")


def run(command: list[str], results: Path) -> None:
    verify()
    results.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=SOURCE, env=os.environ.copy(), check=True)


def probe(model: str, results: Path, save: bool, control: bool) -> None:
    command = [
        sys.executable,
        "train_probe.py",
        "--model",
        MODELS[model],
        "--results_dir",
        str(results.resolve()),
    ]
    if save:
        command.append("--save_probe")
    if control:
        command.append("--random")
    run(command, results)


def evaluate(
    model: str,
    demographic: str,
    results: Path,
    batch_size: int,
) -> None:
    command = [
        sys.executable,
        "eval_conversations.py",
        "--model",
        MODELS[model],
        "--results_dir",
        str(results.resolve()),
        "--demo",
        demographic,
        "-n",
        "250",
        "--batch_size",
        str(batch_size),
    ]
    run(command, results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("inspect", "probe", "evaluate"), required=True
    )
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument(
        "--demographic",
        choices=("age", "gender", "race", "socio-economic status"),
    )
    parser.add_argument("--results", type=Path, default=ROOT / "artifacts/implicit-personalization")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-probe", action="store_true")
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    if args.stage == "inspect":
        inspect()
    elif not args.model:
        parser.error("--model is required")
    elif args.stage == "probe":
        probe(args.model, args.results, args.save_probe, args.control)
    elif not args.demographic:
        parser.error("--demographic is required")
    else:
        evaluate(args.model, args.demographic, args.results, args.batch_size)


if __name__ == "__main__":
    main()
