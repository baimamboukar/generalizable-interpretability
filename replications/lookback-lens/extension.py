from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
SOURCE = ROOT / "external" / "Lookback-Lens"
ARTIFACTS = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts"))
RELEASED = ARTIFACTS / "lookback-lens" / "released" / "step1and2" / "dump"
INPUTS = ARTIFACTS / "lookback-lens" / "inputs"
OUTPUT = ARTIFACTS / "lookback-lens" / "extensions"
CONFIG = json.loads((HERE / "extension.json").read_text())["models"]
DATASETS = {
    "nq": {
        "features": RELEASED / "lookback-ratio-nq-7b.pt",
        "released_annotations": RELEASED / "anno-nq-7b.jsonl",
        "annotations": INPUTS / "anno-nq-7b.jsonl",
        "rows": INPUTS / "rows-nq.jsonl",
        "marker": "\n#Answer#:",
    },
    "cnndm": {
        "features": RELEASED / "lookback-ratio-cnndm-7b.pt",
        "released_annotations": RELEASED / "anno-cnndm-7b.jsonl",
        "annotations": INPUTS / "anno-cnndm-7b.jsonl",
        "rows": INPUTS / "rows-cnndm.jsonl",
        "marker": "\n#Summary#:",
    },
}


def inputs_ready() -> bool:
    return all(
        value["annotations"].exists() and value["rows"].exists()
        for value in DATASETS.values()
    )


def ensure_inputs() -> None:
    if inputs_ready():
        return
    subprocess.run(
        [sys.executable, str(HERE / "run.py"), "--stage", "fetch"],
        check=True,
    )
    prepare_inputs()


def prepare_inputs() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    for dataset, paths in DATASETS.items():
        raw = torch.load(
            paths["features"],
            map_location="cpu",
            weights_only=False,
        )
        with paths["rows"].open("w") as stream:
            for row in raw:
                stream.write(
                    json.dumps(
                        {
                            "data_index": row["data_index"],
                            "prompt": row["full_input_text"],
                            "response": row["model_completion"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        shutil.copy2(paths["released_annotations"], paths["annotations"])
        del raw
        gc.collect()
        print(paths["rows"])


def target(model_key: str, dataset: str) -> Path:
    return OUTPUT / model_key / f"lookback-ratio-{dataset}.pt"


def source_rows(dataset: str, limit: int | None) -> list[dict]:
    with DATASETS[dataset]["rows"].open() as stream:
        rows = [json.loads(line) for line in stream]
    return rows[:limit]


def load_model(model_key: str):
    spec = CONFIG[model_key]
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"],
        token=token,
        trust_remote_code=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        token=token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    return tokenizer, model


def model_kwargs(model) -> dict:
    parameters = inspect.signature(model.forward).parameters
    kwargs = {"output_attentions": True, "use_cache": False, "return_dict": True}
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in parameters:
        kwargs["num_logits_to_keep"] = 1
    return kwargs


def lookback_ratios(
    attentions: tuple[torch.Tensor, ...],
    prompt_length: int,
    context_length: int,
    completion_length: int,
) -> torch.Tensor:
    positions = torch.arange(
        prompt_length - 1,
        prompt_length + completion_length - 1,
        device=attentions[0].device,
    )
    layers = []
    for layer in attentions:
        selected = layer[0, :, positions, :].float()
        context = selected[..., :context_length].mean(-1)
        cumulative = selected.cumsum(-1)
        end = cumulative.gather(
            -1,
            positions.view(1, -1, 1).expand(selected.shape[0], -1, 1),
        ).squeeze(-1)
        before = cumulative[..., context_length - 1]
        counts = (positions - context_length + 1).clamp_min(1)
        generated = (end - before) / counts
        layers.append(context / (context + generated).clamp_min(1e-12))
    return torch.stack(layers).cpu()


def extract_dataset(
    model_key: str,
    dataset: str,
    tokenizer,
    model,
    limit: int | None,
) -> Path:
    output = target(model_key, dataset)
    if output.exists() and limit is None:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = DATASETS[dataset]["marker"]
    records = []
    device = next(model.parameters()).device
    started = time.time()
    for row in tqdm(source_rows(dataset, limit), desc=f"{model_key}:{dataset}"):
        prompt = row["prompt"]
        response_ids = tokenizer(
            row["response"],
            add_special_tokens=False,
        )["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        marker_length = len(
            tokenizer(marker, add_special_tokens=False)["input_ids"]
        )
        context_length = len(prompt_ids) - marker_length
        input_ids = torch.tensor(
            [prompt_ids + response_ids],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            result = model(input_ids=input_ids, **model_kwargs(model))
        if not result.attentions or any(value is None for value in result.attentions):
            raise RuntimeError(f"{model_key} did not return eager attention weights")
        ratios = lookback_ratios(
            result.attentions,
            len(prompt_ids),
            context_length,
            len(response_ids),
        )
        records.append(
            {
                "data_index": row["data_index"],
                "model_completion": row["response"],
                "model_completion_ids": torch.tensor(response_ids),
                "attn_scores": ratios,
            }
        )
        del result, input_ids, ratios
    torch.save(records, output)
    metadata = {
        "model": CONFIG[model_key],
        "dataset": dataset,
        "examples": len(records),
        "seconds": time.time() - started,
        "dtype": "bfloat16",
        "attention_implementation": "eager",
        "protocol": "released Llama-2 text and labels, target-model teacher forcing",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


class OffsetTokenizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, text: str, **kwargs):
        result = self.tokenizer(
            text,
            add_special_tokens=False,
            **kwargs,
        )
        result["input_ids"] = [0, *result["input_ids"]]
        if "offset_mapping" in result:
            result["offset_mapping"] = [(0, 0), *result["offset_mapping"]]
        return result

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


def upstream_module():
    spec = importlib.util.spec_from_file_location(
        "lookback_lens_step03",
        SOURCE / "step03_lookback_lens.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def parse(mode: str, text: str, model_key: str) -> list[dict]:
    pattern = re.compile(
        r"^A=([^;]+);B=([^,]+), ([0-9.]+), ([0-9.]+), ([0-9.]+)$",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(text):
        for split, value in zip(
            ("train", "test", "transfer"),
            (float(match.group(3)), float(match.group(4)), float(match.group(5))),
        ):
            rows.append(
                {
                    "model_key": model_key,
                    "model_id": CONFIG[model_key]["model_id"],
                    "family": CONFIG[model_key]["family"],
                    "parameters_b": CONFIG[model_key]["parameters_b"],
                    "setting": mode,
                    "source": match.group(1),
                    "target": match.group(2),
                    "split": split,
                    "metric": "AUROC",
                    "value": value,
                }
            )
    if len(rows) != 6:
        raise RuntimeError(f"Could not parse {mode} output")
    return rows


def evaluate(model_key: str) -> Path:
    missing = [
        dataset for dataset in DATASETS if not target(model_key, dataset).exists()
    ]
    if missing:
        raise RuntimeError(f"Missing features: {', '.join(missing)}")
    module = upstream_module()
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG[model_key]["model_id"],
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
        use_fast=True,
    )
    wrapped = OffsetTokenizer(tokenizer)
    module.transformers.AutoTokenizer.from_pretrained = lambda *args, **kwargs: wrapped
    output = OUTPUT / model_key
    classifiers = output / "classifiers"
    classifiers.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode, sliding in (("predefined", None), ("sliding-window-8", 8)):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            module.main(
                str(DATASETS["nq"]["annotations"]),
                str(target(model_key, "nq")),
                str(DATASETS["cnndm"]["annotations"]),
                str(target(model_key, "cnndm")),
                predefined_span=sliding is None,
                sliding_window=sliding,
                two_fold=True,
                tokenizer_name=CONFIG[model_key]["model_id"],
                output_path=str(classifiers),
            )
        text = stream.getvalue()
        print(text)
        (output / f"{mode}.log").write_text(text)
        rows.extend(parse(mode, text, model_key))
    result = output / "results.csv"
    with result.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CONFIG, required=True)
    parser.add_argument(
        "--stage",
        choices=("inspect", "prepare", "extract", "evaluate", "all"),
        required=True,
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stage == "inspect":
        print(json.dumps(CONFIG[args.model], indent=2))
        return
    if args.stage == "prepare":
        ensure_inputs()
        return
    ensure_inputs()
    if args.stage in {"extract", "all"}:
        tokenizer, model = load_model(args.model)
        names = DATASETS if args.dataset == "all" else (args.dataset,)
        for dataset in names:
            print(extract_dataset(args.model, dataset, tokenizer, model, args.limit))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if args.stage in {"evaluate", "all"}:
        if args.limit:
            raise RuntimeError("Evaluation requires complete feature files")
        print(evaluate(args.model))


if __name__ == "__main__":
    main()
