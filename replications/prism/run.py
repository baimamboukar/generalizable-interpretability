from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
from statistics import fmean
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external" / "PRISM"
ARTIFACTS = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts")) / "prism"
COMMIT = "b0953654e3181cc3e4a158f0c4313b85c000d290"
ACCELERATE = "0.20.3"
SENTENCEPIECE = "0.1.99"
PROTOBUF = "4.24.3"
MODELS = {
    "llama2-7b": ("meta-llama/Llama-2-7b-chat-hf", "llama2chat7b", 4096),
    "llama2-13b": ("meta-llama/Llama-2-13b-chat-hf", "llama2chat13b", 5120),
    "llama31-8b": ("meta-llama/Llama-3.1-8B-Instruct", "llama31instruct8b", 4096),
    "glm4-9b": ("zai-org/glm-4-9b-chat-hf", "glm4chat9b", 4096),
    "moonlight-16b": (
        "moonshotai/Moonlight-16B-A3B-Instruct",
        "moonlight16ba3binstruct",
        2048,
    ),
    "qwen25-14b": ("Qwen/Qwen2.5-14B-Instruct", "qwen25instruct14b", 5120),
}
EXTENSION_MODELS = {"llama31-8b", "glm4-9b", "moonlight-16b", "qwen25-14b"}
REMOTE_CODE_MODELS = {"moonlight-16b"}
GENERATION_SCRIPTS = (
    "generate_data/gen_true.py",
    "generate_data/gen_true_prompt.py",
    "generate_data/gen_logic.py",
    "generate_data/gen_logic_prompt.py",
)
MODEL_NAMES = {
    "llama2-7b": "Llama-2-7B-Chat",
    "llama2-13b": "Llama-2-13B-Chat",
    "llama31-8b": "Llama-3.1-8B-Instruct",
    "glm4-9b": "GLM-4-9B-Chat",
    "moonlight-16b": "Moonlight-16B-A3B-Instruct",
    "qwen25-14b": "Qwen2.5-14B-Instruct",
}
TRUE_DOMAINS = ["animals", "cities", "companies", "elements", "facts", "inventions"]
LOGIC_STRUCTURES = ["negated", "conjunction", "disjunction"]


def worktree(model: str) -> Path:
    return ARTIFACTS / "worktrees" / f"{COMMIT[:8]}-{model}"


def source_commit() -> str:
    if not (UPSTREAM / ".git").exists():
        return json.loads((Path(__file__).parent / "upstream.json").read_text())["commit"]
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()


def source_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def configured_source(path: str, text: str, model: str) -> str:
    if model != "llama2-13b" or path not in GENERATION_SCRIPTS:
        return text
    source_model, source_dir, _ = MODELS["llama2-7b"]
    target_model, target_dir, _ = MODELS[model]
    text = text.replace(
        f'model_path = "{source_model}"',
        f'model_path = "{target_model}"',
        1,
    )
    text = text.replace(f"/{source_dir}/", f"/{target_dir}/", 1)
    return text


def verify_source(tree: Path, model: str) -> None:
    expected = {
        str(path.relative_to(UPSTREAM)): hashlib.sha256(
            configured_source(
                str(path.relative_to(UPSTREAM)), path.read_text(), model
            ).encode()
        ).hexdigest()
        for path in sorted(UPSTREAM.rglob("*.py"))
    }
    if source_hashes(tree) != expected:
        raise RuntimeError(f"Configured worktree differs in {tree}")


def data_counts() -> dict[str, dict[str, int]]:
    counts = {}
    for dataset in ("true", "logic"):
        counts[dataset] = {
            path.stem: len(json.loads(path.read_text()))
            for path in sorted((UPSTREAM / "raw_data" / dataset).glob("*.json"))
        }
    return counts


def inspect() -> None:
    counts = data_counts()
    print(
        json.dumps(
            {
                "upstream": str(UPSTREAM),
                "commit": source_commit(),
                "expected_commit": COMMIT,
                "raw_rows": {
                    dataset: {"total": sum(rows.values()), "files": rows}
                    for dataset, rows in counts.items()
                },
                "released_activations": False,
                "released_checkpoints": False,
                "released_results": False,
            },
            indent=2,
        )
    )


def preflight(model: str) -> None:
    requirements = (UPSTREAM / "requirements.txt").read_text().splitlines()
    print(
        json.dumps(
            {
                "commit": source_commit(),
                "source_files": len(source_hashes(UPSTREAM)),
                "python": "3.9.19",
                "requirements": requirements,
                "llama2_7b": "released configuration",
                "llama2_13b": "release-provided model/path switches enabled in artifact worktree",
                "extension": {
                    "model": MODELS[model][0],
                    "extractor": "thin AutoModel adapter"
                    if model in EXTENSION_MODELS
                    else "released Llama script",
                    "probe_scripts": "released, unchanged",
                },
                "release_gaps": {
                    "accelerate": {
                        "runtime_version": ACCELERATE,
                        "required_by": "device_map='auto' and low_cpu_mem_usage=True",
                        "listed": any(line.startswith("accelerate") for line in requirements),
                        "installed": importlib.util.find_spec("accelerate") is not None,
                    },
                    "sentencepiece": {
                        "runtime_version": SENTENCEPIECE,
                        "required_by": "LlamaTokenizer",
                        "listed": any(line.startswith("sentencepiece") for line in requirements),
                        "installed": importlib.util.find_spec("sentencepiece") is not None,
                    },
                    "protobuf": {
                        "runtime_version": PROTOBUF,
                        "required_by": "LlamaTokenizer",
                        "listed": any(line.startswith("protobuf") for line in requirements),
                        "installed": importlib.util.find_spec("google.protobuf") is not None,
                    },
                },
                "model_revision_pinned": False,
                "logicstruct_seed_driver_released": False,
            },
            indent=2,
        )
    )


def prepare(model: str) -> Path:
    if source_commit() != COMMIT:
        raise RuntimeError(f"Expected PRISM commit {COMMIT}")
    target = worktree(model)
    if not target.exists():
        shutil.copytree(UPSTREAM, target, ignore=shutil.ignore_patterns(".git"))
        for relative in GENERATION_SCRIPTS:
            path = target / relative
            path.write_text(configured_source(relative, path.read_text(), model))
        metadata = {
            "source": str(UPSTREAM),
            "commit": COMMIT,
            "model": model,
            "source_modified": model == "llama2-13b",
            "configuration": {
                "model": MODELS[model][0],
                "activation_directory": MODELS[model][1],
                "probe_input_size": MODELS[model][2],
            },
        }
        (target / "replication_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    verify_source(target, model)
    print(target)
    return target


def run_logged(command: list[str], cwd: Path, log: Path) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(command))
    lines = []
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout
        for line in process.stdout:
            print(line, end="")
            output.write(line)
            lines.append(line)
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, command)
    return "".join(lines)


def generate(model: str, dataset: str, representation: str) -> None:
    tree = prepare(model)
    log = ARTIFACTS / "logs" / f"{model}_{dataset}_{representation}_generate.log"
    if model in EXTENSION_MODELS:
        script = Path(__file__).parent / "extract.py"
        command = [
            sys.executable,
            str(script),
            "--model",
            MODELS[model][0],
            "--dataset",
            dataset,
            "--representation",
            representation,
            "--raw-data",
            str(tree / "raw_data"),
            "--output",
            str(activation_path(tree, model, dataset, representation)),
        ]
        if model in REMOTE_CODE_MODELS:
            command.append("--trust-remote-code")
        run_logged(command, ROOT, log)
        return
    suffix = "_prompt" if representation == "prompted" else ""
    script = tree / "generate_data" / f"gen_{dataset}{suffix}.py"
    run_logged([sys.executable, script.name], script.parent, log)


def activation_complete(
    model: str, dataset: str, representation: str
) -> bool:
    tree = worktree(model)
    expected = 6 if dataset == "true" else 24
    files = activation_path(tree, model, dataset, representation).glob(
        "*_last_token.json"
    )
    return len(list(files)) == expected


def activation_path(
    tree: Path, model: str, dataset: str, representation: str
) -> Path:
    root = "hd_data_prompt" if representation == "prompted" else "hd_data"
    return tree / root / dataset / MODELS[model][1]


def parse_scores(output: str, method: str) -> dict[str, float]:
    pattern = (
        r"Accuracy on\s*([^:]*):\s*([0-9.]+)"
        if method == "mm"
        else r"Eval on\s*(.*?)\s*-\s*Epoch.*?Accuracy:\s*([0-9.]+)"
    )
    return {name or "affirmative": float(value) for name, value in re.findall(pattern, output)}


def train(model: str, dataset: str, representation: str, method: str) -> None:
    tree = prepare(model)
    data = activation_path(tree, model, dataset, representation)
    expected = 6 if dataset == "true" else 24
    if len(list(data.glob("*_last_token.json"))) != expected:
        raise RuntimeError(f"Missing activations in {data}; run --stage generate first")

    script = tree / "train" / dataset / f"train_{method}.py"
    runs = []
    seeds = range(3) if method == "saplma" else [None]
    train_domains = range(6) if dataset == "true" else [0]
    for seed in seeds:
        for train_domain in train_domains:
            command = [
                sys.executable,
                script.name,
                "--data_path",
                str(data),
                "--train_number",
                str(train_domain),
                "--device",
                "cuda:0",
            ]
            if seed is not None:
                command += [
                    "--setup_seed",
                    str(seed),
                    "--input_size",
                    str(MODELS[model][2]),
                    "--train_epoch",
                    "10",
                    "--batch_size",
                    "32",
                    "--lr",
                    "1e-3",
                    "--wd",
                    "0.0",
                    "--dropout",
                    "0.2",
                ]
            tag = f"seed{seed if seed is not None else 0}_train{train_domain}"
            log = (
                ARTIFACTS
                / "logs"
                / f"{model}_{dataset}_{representation}_{method}_{tag}.log"
            )
            scores = parse_scores(run_logged(command, script.parent, log), method)
            names = TRUE_DOMAINS if dataset == "true" else LOGIC_STRUCTURES
            for test_domain, value in scores.items():
                if dataset == "logic":
                    test_domain = names[["_neg", "_conj", "_disj"].index(test_domain)]
                runs.append(
                    {
                        "model": model,
                        "dataset": "True-False" if dataset == "true" else "LogicStruct",
                        "representation": representation,
                        "method": method.upper(),
                        "seed": seed if seed is not None else 0,
                        "train_domain": names[train_domain] if dataset == "true" else "affirmative",
                        "test_domain": test_domain,
                        "metric": "accuracy",
                        "replication_value": value,
                    }
                )
    save_results(model, dataset, representation, method, runs)


def save_results(
    model: str,
    dataset: str,
    representation: str,
    method: str,
    rows: list[dict],
) -> None:
    output = ARTIFACTS / "results"
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_{dataset}_{representation}_{method}"
    (output / f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["test_domain"], []).append(row["replication_value"])
    values = {name: fmean(scores) for name, scores in grouped.items()}
    values["average"] = fmean(values.values())
    summary = [
        {
            "study": "PRISM",
            "dataset": rows[0]["dataset"],
            "model": MODEL_NAMES[model],
            "method": method.upper(),
            "representation": representation,
            "test_domain": domain,
            "metric": "accuracy",
            "paper_value": "",
            "replication_value": value,
            "delta": "",
            "status": "completed",
        }
        for domain, value in values.items()
    ]
    with (output / f"{stem}_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    print(output / f"{stem}.csv")


def result_path(model: str, dataset: str, representation: str, method: str) -> Path:
    return (
        ARTIFACTS
        / "results"
        / f"{model}_{dataset}_{representation}_{method}_summary.csv"
    )


def baseline(model: str) -> None:
    prepare(model)
    if importlib.util.find_spec("accelerate") is None:
        raise RuntimeError(
            "The released generation scripts require accelerate through "
            "device_map='auto', but it is absent from requirements.txt. "
            "Dependency addition is blocked pending approval."
        )
    for dataset in ("true", "logic"):
        for representation in ("raw", "prompted"):
            if not activation_complete(model, dataset, representation):
                generate(model, dataset, representation)
            for method in ("mm", "saplma"):
                if not result_path(model, dataset, representation, method).exists():
                    train(model, dataset, representation, method)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--stage",
        choices=["inspect", "preflight", "prepare", "generate", "train", "baseline"],
        required=True,
    )
    cli.add_argument("--model", choices=MODELS, default="llama2-7b")
    cli.add_argument("--dataset", choices=["true", "logic"], default="true")
    cli.add_argument("--representation", choices=["raw", "prompted"], default="prompted")
    cli.add_argument("--method", choices=["mm", "saplma"], default="mm")
    return cli


def main() -> None:
    args = parser().parse_args()
    if args.stage == "inspect":
        inspect()
    elif args.stage == "preflight":
        preflight(args.model)
    elif args.stage == "prepare":
        prepare(args.model)
    elif args.stage == "generate":
        generate(args.model, args.dataset, args.representation)
    elif args.stage == "train":
        train(args.model, args.dataset, args.representation, args.method)
    else:
        baseline(args.model)


if __name__ == "__main__":
    main()
