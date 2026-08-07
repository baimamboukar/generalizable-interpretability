from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
UPSTREAM = ROOT / "external" / "disentangle-sycophancy"
ARTIFACTS = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts")) / "disentangle-sycophancy"
COMMIT = json.loads((HERE / "upstream.json").read_text())["commit"]
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LABELS = ("syc", "ga", "pr", "ag", "rand", "syc_all")
DATASETS = {
    "math": "data/factorial/math_factorial.json",
    "cities": "data/factorial/cities_pos_factorial.json",
    "cities-neg": "data/factorial/cities_neg_factorial.json",
    "sp-en-trans": "data/factorial/sp_en_trans_factorial.json",
    "sp-en-trans-pro": "data/factorial/sp_en_trans_pro_factorial.json",
    "larger-than": "data/factorial/larger_than_factorial.json",
    "smaller-than": "data/factorial/smaller_than_factorial.json",
    "claims": "data/factorial/claims_factorial.json",
    "counterfactual": "data/factorial/counterfactual_factorial.json",
}


def source_commit() -> str:
    git_dir = UPSTREAM / ".git"
    if not git_dir.exists():
        return "missing"
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    ref_path = git_dir / ref
    if ref_path.exists():
        return ref_path.read_text().strip()
    for line in (git_dir / "packed-refs").read_text().splitlines():
        commit, name = line.split(maxsplit=1)
        if name == ref:
            return commit
    return "missing"


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(UPSTREAM.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(UPSTREAM)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def data_inventory() -> dict[str, int]:
    return {
        name: len(json.loads((UPSTREAM / relative).read_text()))
        for name, relative in DATASETS.items()
        if (UPSTREAM / relative).exists()
    }


def inspect() -> None:
    requirements = UPSTREAM / "requirements.txt"
    requirement_encoding = None
    if requirements.exists():
        raw = requirements.read_bytes()
        requirement_encoding = "utf-16-le" if raw.startswith(b"\xff\xfe") else "utf-8"
    print(
        json.dumps(
            {
                "paper": "Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs",
                "upstream": str(UPSTREAM),
                "commit": source_commit(),
                "expected_commit": COMMIT,
                "source_digest": source_digest() if UPSTREAM.exists() else None,
                "baseline": {
                    "model": MODEL,
                    "representation": "post-layernorm residual stream at end of sentence",
                    "method": "DiffMean direction with AUROC detection",
                    "labels": LABELS,
                    "dataset": "math_factorial.json",
                    "layers": "all transformer layers, embeddings excluded",
                    "dtype": "float32 (official script)",
                    "batch_size": 32,
                },
                "datasets": data_inventory() if UPSTREAM.exists() else {},
                "release": {
                    "code": True,
                    "controlled_data": True,
                    "checkpoints": False,
                    "paper_result_tables": False,
                    "sycon_bench": False,
                    "requirements_encoding": requirement_encoding,
                },
                "paper_release_count_mismatches": {
                    "sp_en_trans": {"paper": 4000, "release": data_inventory().get("sp-en-trans")},
                    "sp_en_trans_variant": {"paper": 4000, "release": data_inventory().get("sp-en-trans-pro")},
                },
                "known_gaps": [
                    "shell scripts invoke files directly although the source uses relative imports",
                    "steering prompt file referenced by run_all_steering.sh is absent",
                    "run_all_steering.sh uses layer 32 while the paper's Qwen3-30B steering table uses layer 46",
                    "praise.json is used for both training and evaluation by run_roberta.sh",
                    "model revisions are not pinned",
                    "released SPEN translation row counts differ from Appendix B",
                ],
            },
            indent=2,
        )
    )


def preflight() -> None:
    required = [UPSTREAM / "src" / "diffmean_analysis.py"]
    missing = [str(path) for path in required if not path.exists()]
    if source_commit() != COMMIT:
        raise RuntimeError(f"Expected upstream commit {COMMIT}; found {source_commit()}")
    if missing:
        raise RuntimeError(f"Missing upstream files: {missing}")
    print(
        json.dumps(
            {
                "status": "ready",
                "source_commit": source_commit(),
                "module_invocation": "python -m src.diffmean_analysis",
                "direct_script_invocation": "not used: relative imports fail when run as python src/diffmean_analysis.py",
                "baseline_input": str(UPSTREAM / DATASETS["math"]),
                "baseline_output": str(ARTIFACTS / "diffmean" / "qwen3_30b" / "math"),
            },
            indent=2,
        )
    )


def run_diffmean(args: argparse.Namespace) -> None:
    preflight()
    dataset = args.dataset
    input_json = (
        Path(args.input_json).expanduser().resolve()
        if args.input_json
        else UPSTREAM / DATASETS[dataset]
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ARTIFACTS / "diffmean" / "qwen3_30b" / dataset
    )
    command = [
        sys.executable,
        "-m",
        "src.diffmean_analysis",
        "--input_json",
        str(input_json),
        "--model_name",
        args.model,
        "--all_layers",
        "--label_fields",
        *LABELS,
        "--output_dir",
        str(output_dir),
        "--null_ablate",
        "--null_labels",
        *LABELS,
        "--train_batch_size",
        str(args.train_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--dtype",
        args.dtype,
        "--verbose",
    ]
    if args.shard_model:
        command.append("--shard_model")
    print(json.dumps({"cwd": str(UPSTREAM), "command": command}, indent=2))
    subprocess.run(command, cwd=UPSTREAM, check=True)


def aggregate(args: argparse.Namespace) -> None:
    source = Path(args.output_dir)
    rows = []
    for path in sorted(source.glob("diffmean_*_L*.json")):
        payload = json.loads(path.read_text())
        rows.append(
            {
                "study": "Disentangle Sycophancy",
                "figure": "Figure 1",
                "model": payload.get("model_name", MODEL),
                "dataset": Path(payload.get("input_json", "math_factorial.json")).stem,
                "behavior": path.stem.removeprefix("diffmean_").rsplit("_L", 1)[0],
                "layer": payload.get("layer"),
                "metric": "AUROC",
                "paper_value": "",
                "replication_value": payload.get("test_auc_raw"),
                "delta": "",
                "status": "completed",
            }
        )
    target = ROOT / "data" / "results" / "disentangle-sycophancy" / "diffmean_long.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        import csv

        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(target)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inspect", "preflight", "diffmean", "baseline", "aggregate"), required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="math")
    parser.add_argument("--input-json")
    parser.add_argument("--output-dir")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="float32")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--shard-model", action="store_true")
    args = parser.parse_args()
    if args.stage == "inspect":
        inspect()
    elif args.stage == "preflight":
        preflight()
    elif args.stage in {"diffmean", "baseline"}:
        run_diffmean(args)
    else:
        if not args.output_dir:
            parser.error("--output-dir is required for --stage aggregate")
        aggregate(args)


if __name__ == "__main__":
    main()
