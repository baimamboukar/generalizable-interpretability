from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
UPSTREAM = ROOT / "external" / "RepresentationPoliticalLLM"
ARTIFACTS = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts"))
ARTIFACTS = ARTIFACTS / "political-perspective"
COMMIT = "e746dd1e3838ee6966818fef5a9b2d4e107f1656"
MODELS = {
    "llama2-7b": "Llama-2-7B-Chat",
    "mistral-7b": "Mistral-7B-Instruct-v0.1",
    "vicuna-7b": "Vicuna-7B-v1.5",
}
PAPER_PEAKS = {
    "llama2-7b": (0.8536, 15, 18),
    "mistral-7b": (0.8463, 16, 3),
    "vicuna-7b": (0.8612, 24, 8),
}
NOTEBOOK_SHA256 = "660c8eb787faaab95a4c1ca4320223a95728c12a008846bb6c49bfcee58df574"
BASELINE_CELLS = (1, 2, 3, 5, 6, 9, 11, 13)


def manifest() -> list[dict]:
    return json.loads((HERE / "osf_manifest.json").read_text())


def source_commit() -> str:
    if not (UPSTREAM / ".git").exists():
        return json.loads((HERE / "upstream.json").read_text())["commit"]
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect() -> None:
    notebook = UPSTREAM / "replication_results.ipynb"
    text = notebook.read_text()
    files = manifest()
    print(
        json.dumps(
            {
                "paper": "Linear Representations of Political Perspective Emerge in LLMs",
                "venue": "ICLR 2025",
                "upstream_commit": source_commit(),
                "expected_commit": COMMIT,
                "models": list(MODELS.values()),
                "baseline_targets": ["Figure 2", "Table A3", "Figure 3"],
                "released": {
                    "data": sum(item["group"] == "data" for item in files),
                    "performance": sum(item["group"] == "performance" for item in files),
                    "features": sum(item["group"] == "features" for item in files),
                    "checkpoints": sum(item["group"] == "checkpoint" for item in files),
                },
                "configuration_gaps": {
                    "invalid_hf_token_assignment": "HUGGINGFACE_TOKEN = # ENTER HUGGINGFACE TOKEN"
                    in text,
                    "model_revisions_pinned": False,
                    "executable_script_released": False,
                    "requirements_file_released": False,
                    "scikit_learn_version_released": False,
                },
                "notebook_sha256": sha256(notebook),
            },
            indent=2,
        )
    )


def worktree() -> Path:
    cache = Path(os.environ.get("GI_WORK_DIR", ROOT / ".cache"))
    return cache / "political-perspective" / COMMIT[:8]


def prepare() -> Path:
    source = UPSTREAM / "replication_results.ipynb"
    if sha256(source) != NOTEBOOK_SHA256:
        raise RuntimeError("Unexpected upstream notebook hash")
    tree = worktree()
    if not tree.exists():
        shutil.copytree(UPSTREAM, tree, ignore=shutil.ignore_patterns(".git"))
    notebook = tree / "replication_results.ipynb"
    document = json.loads(source.read_text())
    original = "".join(document["cells"][2]["source"])
    if original != "HUGGINGFACE_TOKEN = # ENTER HUGGINGFACE TOKEN":
        raise RuntimeError("Unexpected Hugging Face token cell")
    configured = 'HUGGINGFACE_TOKEN = __import__("os").environ["HF_TOKEN"]\n'
    document["cells"][2]["source"] = [configured]
    notebook.write_text(json.dumps(document, indent=1) + "\n")
    fetch("data", "all")
    for item in selected("data", "all"):
        target = tree / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ARTIFACTS / "released" / item["path"], target)
    (tree / "results_replication").mkdir(exist_ok=True)
    metadata = {
        "upstream_commit": COMMIT,
        "upstream_notebook_sha256": NOTEBOOK_SHA256,
        "configured_notebook_sha256": sha256(notebook),
        "approved_change": {
            "cell": 2,
            "from": original,
            "to": configured.strip(),
        },
        "executed_cells": list(BASELINE_CELLS),
        "stopped_before": "Intervention",
    }
    (tree / "replication_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "replication_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(tree)
    return tree


def runtime() -> dict[str, str]:
    import einops
    import matplotlib
    import numpy as np
    import pandas
    import scipy
    import sklearn
    import torch
    import tqdm
    import transformers

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "einops": einops.__version__,
        "matplotlib": matplotlib.__version__,
        "tqdm": tqdm.__version__,
        "scikit_learn": sklearn.__version__,
    }
    expected = {
        "torch": "2.3.0",
        "transformers": "4.38.2",
        "numpy": "1.26.4",
        "pandas": "2.2.1",
        "scipy": "1.12.0",
        "einops": "0.8.0",
        "matplotlib": "3.8.3",
        "tqdm": "4.66.2",
    }
    mismatches = {
        key: {"expected": value, "actual": versions[key]}
        for key, value in expected.items()
        if not versions[key].startswith(value)
    }
    if not versions["python"].startswith("3.10.") or mismatches:
        raise RuntimeError(
            json.dumps(
                {"python_expected": "3.10.13", "versions": versions, "mismatches": mismatches},
                indent=2,
            )
        )
    return versions


def summarize(tree: Path) -> None:
    import numpy as np

    rows = []
    paths = {
        item["model"]: tree / item["path"]
        for item in selected("performance", "all")
    }
    for key, path in paths.items():
        array = np.asarray(pickle.load(path.open("rb")))
        layer, head_zero = np.unravel_index(np.nanargmax(array), array.shape)
        paper, paper_layer, paper_head = PAPER_PEAKS[key]
        rows.append(
            {
                "model": MODELS[key],
                "metric": "peak single-head CV Spearman",
                "paper": paper,
                "replication": float(array[layer, head_zero]),
                "delta": float(array[layer, head_zero]) - paper,
                "paper_layer": paper_layer,
                "replication_layer": int(layer),
                "paper_head": paper_head,
                "replication_head": int(head_zero + 1),
            }
        )
    target = ARTIFACTS / "independent_diagnostic.csv"
    released = ARTIFACTS / "independent" / "results_replication"
    released.mkdir(parents=True, exist_ok=True)
    for path in (tree / "results_replication").iterdir():
        if path.is_file():
            shutil.copy2(path, released / path.name)
    with target.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(target)


def execute_baseline() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    from IPython.core.interactiveshell import InteractiveShell

    versions = runtime()
    tree = prepare()
    document = json.loads((tree / "replication_results.ipynb").read_text())
    previous = Path.cwd()
    tree_path = str(tree)
    sys.path.insert(0, tree_path)
    os.chdir(tree)
    try:
        shell = InteractiveShell.instance()
        for index in BASELINE_CELLS:
            print(f"Executing released notebook cell {index}", flush=True)
            result = shell.run_cell("".join(document["cells"][index]["source"]))
            if result.error_before_exec:
                raise result.error_before_exec
            if result.error_in_exec:
                raise result.error_in_exec
            if index == 1:
                resolved = {
                    "accelerate": version("accelerate"),
                    "huggingface_hub": version("huggingface-hub"),
                    "torch": version("torch"),
                    "torchvision": version("torchvision"),
                }
                expected = {
                    "accelerate": "0.27.2",
                    "huggingface_hub": "0.20.3",
                    "torch": "2.3.0",
                    "torchvision": "0.18.0",
                }
                if any(
                    not resolved[key].startswith(value)
                    for key, value in expected.items()
                ):
                    raise RuntimeError(f"Released install changed runtime: {resolved}")
    finally:
        os.chdir(previous)
        sys.path.remove(tree_path)
    (ARTIFACTS / "runtime.json").write_text(json.dumps(versions, indent=2) + "\n")
    summarize(tree)


def selected(group: str, model: str) -> list[dict]:
    files = manifest()
    if group != "all":
        files = [item for item in files if item["group"] == group]
    if model != "all":
        files = [
            item for item in files if item.get("model") in (None, model)
        ]
    return files


def fetch(group: str, model: str) -> None:
    for item in selected(group, model):
        target = ARTIFACTS / "released" / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) == item["sha256"]:
            print(f"verified {target}")
            continue
        with urlopen(item["url"]) as response, target.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if target.stat().st_size != item["size"] or sha256(target) != item["sha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {item['path']}")
        print(f"downloaded {target}")


def released_check() -> None:
    import numpy as np

    fetch("performance", "all")
    rows = []
    for item in selected("performance", "all"):
        array = np.asarray(
            pickle.load((ARTIFACTS / "released" / item["path"]).open("rb"))
        )
        layer, head_zero = np.unravel_index(np.nanargmax(array), array.shape)
        paper, paper_layer, paper_head = PAPER_PEAKS[item["model"]]
        value = float(array[layer, head_zero])
        rows.append(
            {
                "model": MODELS[item["model"]],
                "metric": "peak single-head CV Spearman",
                "paper": paper,
                "released_artifact": value,
                "delta": value - paper,
                "paper_layer": paper_layer,
                "released_layer": int(layer),
                "paper_head": paper_head,
                "released_head": int(head_zero + 1),
            }
        )
    target = ARTIFACTS / "released_diagnostic.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(target)


def baseline() -> None:
    execute_baseline()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("inspect", "fetch", "released-check", "prepare", "baseline"),
        required=True,
    )
    parser.add_argument(
        "--group",
        choices=("data", "performance", "features", "checkpoint", "intervention", "all"),
        default="data",
    )
    parser.add_argument(
        "--model", choices=("llama2-7b", "mistral-7b", "vicuna-7b", "all"), default="all"
    )
    args = parser.parse_args()
    if source_commit() != COMMIT:
        raise RuntimeError(f"Expected upstream commit {COMMIT}")
    {
        "inspect": inspect,
        "fetch": lambda: fetch(args.group, args.model),
        "released-check": released_check,
        "prepare": prepare,
        "baseline": baseline,
    }[args.stage]()


if __name__ == "__main__":
    main()
