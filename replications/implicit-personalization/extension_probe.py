from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "external" / "implicit-personalization-stereotypes"
CONFIG = json.loads((Path(__file__).parent / "extension.json").read_text())
sys.path.insert(0, str(SOURCE))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_probe import get_introductions, get_repr, train_probe
from utils import explicit_indicators, neutral_introductions, templates


def subset(representations: dict, layers: list[int]) -> dict:
    return {
        demographic: {
            value: [[row[layer] for layer in layers] for row in rows]
            for value, rows in values.items()
        }
        for demographic, values in representations.items()
    }


def output_dir(model: str, root: Path) -> Path:
    return root / model


def extract(model: str, root: Path) -> None:
    model_id = CONFIG["models"][model]
    layers = CONFIG["selected_layers"][model]
    output = output_dir(model, root)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    network = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    network.eval()
    introductions = get_introductions(
        templates["introductions"],
        explicit_indicators,
    )
    for demographic in introductions:
        introductions[demographic]["neutral"] = neutral_introductions

    representations, total_layers = get_repr(
        introductions,
        network,
        tokenizer,
    )
    representations = subset(representations, layers)
    with (output / "representations.pkl").open("wb") as handle:
        pickle.dump(representations, handle)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "layers": layers,
                "total_hidden_states": total_layers,
                "representation": "upstream train_probe.get_repr",
                "probe": "upstream train_probe.train_probe",
            },
            indent=2,
        )
    )
    del network, representations
    gc.collect()
    torch.cuda.empty_cache()


def train_part(task: tuple[str, dict, int, bool, Path]) -> Path:
    demographic, values, layers, random, path = task
    train_probe(
        {demographic: values},
        layers,
        path,
        random=random,
    )
    return path


def train(model: str, root: Path, workers: int) -> None:
    output = output_dir(model, root)
    with (output / "representations.pkl").open("rb") as handle:
        representations = pickle.load(handle)
    layers = len(CONFIG["selected_layers"][model])
    parts = output / ".parts"
    shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir()
    tasks = [
        (
            demographic,
            values,
            layers,
            random,
            parts / f"{kind}-{demographic}.pkl",
        )
        for kind, random in (("true", False), ("control", True))
        for demographic, values in representations.items()
    ]
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)),
        mp_context=get_context("spawn"),
    ) as pool:
        paths = list(pool.map(train_part, tasks))

    for kind in ("true", "control"):
        merged = {}
        for path in paths:
            if path.name.startswith(kind):
                with path.open("rb") as handle:
                    merged.update(pickle.load(handle))
        with (output / f"{kind}.pkl").open("wb") as handle:
            pickle.dump(merged, handle)
    shutil.rmtree(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract", "train", "all"), required=True)
    parser.add_argument("--model", choices=CONFIG["models"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts"))
        / "implicit-personalization"
        / "extensions",
    )
    args = parser.parse_args()
    if args.stage in {"extract", "all"}:
        extract(args.model, args.results)
    if args.stage in {"train", "all"}:
        train(args.model, args.results, args.workers)


if __name__ == "__main__":
    main()
