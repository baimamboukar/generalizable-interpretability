from __future__ import annotations

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "external/RepresentationPoliticalLLM/generate_figures.ipynb"
WORKDIR = ROOT / "artifacts/political-perspective/independent"
OUTPUT = ROOT / "data/results/political-perspective"
NOTEBOOK_SHA256 = "5abce3209ca24b31fa26f0f4ebc1848a1ca1a08dee813fdd719be929a6794edf"
CELLS = (1, 2, 11, 13)
MODELS = {
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7B-Chat",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral-7B-Instruct-v0.1",
    "lmsys/vicuna-7b-v1.5": "Vicuna-7B-v1.5",
}
PAPER_ENSEMBLE = {
    "Llama-2-7B-Chat": 0.8703,
    "Mistral-7B-Instruct-v0.1": 0.8636,
    "Vicuna-7B-v1.5": 0.8851,
}
PAPER_NEWS = {"Llama-2-7B-Chat": 0.798}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute() -> list[dict]:
    if sha256(NOTEBOOK) != NOTEBOOK_SHA256:
        raise RuntimeError("Unexpected upstream figure notebook hash")
    for path in (WORKDIR / "results_replication").glob("*_features.pkl"):
        if not path.stat().st_size:
            raise RuntimeError(f"Empty feature artifact: {path}")

    document = json.loads(NOTEBOOK.read_text())
    namespace: dict = {}
    previous = Path.cwd()
    stream = io.StringIO()
    (WORKDIR / "figure_replication").mkdir(exist_ok=True)
    os.chdir(WORKDIR)
    try:
        with contextlib.redirect_stdout(stream):
            for index in CELLS:
                exec(
                    compile(
                        "".join(document["cells"][index]["source"]),
                        f"{NOTEBOOK.name}:cell-{index}",
                        "exec",
                    ),
                    namespace,
                )
    finally:
        os.chdir(previous)

    raw = stream.getvalue()
    (WORKDIR / "remaining_claims_stdout.log").write_text(raw)
    values = re.findall(r"^(\S+) 32 ([+-]?\d+(?:\.\d+)?)$", raw, re.MULTILINE)
    if len(values) != 6:
        raise RuntimeError(f"Expected six released-notebook results, got {values}")

    rows = []
    for target, group in (("Table A3 K=32", values[:3]), ("Figure 3 news", values[3:])):
        for model_id, value in group:
            model = MODELS[model_id]
            paper = (
                PAPER_ENSEMBLE.get(model)
                if target.startswith("Table A3")
                else PAPER_NEWS.get(model)
            )
            replication = float(value)
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "metric": "Spearman correlation",
                    "paper": "" if paper is None else paper,
                    "replication": replication,
                    "delta": "" if paper is None else replication - paper,
                }
            )
    return rows


def main() -> None:
    rows = execute()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "remaining_baseline_claims.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(target)


if __name__ == "__main__":
    main()
