from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "external" / "implicit-personalization-stereotypes"
MANIFEST = json.loads((Path(__file__).parent / "upstream.json").read_text())
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


def run_model(key: str, results: Path) -> None:
    model_id = MANIFEST["models"][key]
    layers = MANIFEST["selected_layers"][key]
    output = results / key
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    introductions = get_introductions(
        templates["introductions"],
        explicit_indicators,
    )
    for demographic in introductions:
        introductions[demographic]["neutral"] = neutral_introductions

    representations, total_layers = get_repr(
        introductions,
        model,
        tokenizer,
    )
    representations = subset(representations, layers)
    with (output / "representations.pkl").open("wb") as handle:
        pickle.dump(representations, handle)

    prefix = model_id.split("/")[-1]
    train_probe(
        representations,
        len(layers),
        output / "true.pkl",
        random=False,
    )
    train_probe(
        representations,
        len(layers),
        output / "control.pkl",
        random=True,
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "layers": layers,
                "total_hidden_states": total_layers,
                "probe": "upstream train_probe.train_probe",
                "prefix": prefix,
            },
            indent=2,
        )
    )
    del model, representations
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=(*MANIFEST["models"], "all"),
        required=True,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            os.environ.get(
                "GI_ARTIFACT_DIR",
                ROOT / "artifacts",
            )
        )
        / "implicit-personalization"
        / "probe-validation",
    )
    args = parser.parse_args()
    models = MANIFEST["models"] if args.model == "all" else (args.model,)
    for model in models:
        run_model(model, args.results)


if __name__ == "__main__":
    main()
