from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import compat  # noqa: E402
import submission  # noqa: E402

from infra.env import load_project_env  # noqa: E402

SOURCE = ROOT / "external" / "internal-instruction-following"
MANIFEST = json.loads((HERE / "upstream.json").read_text())
DATASET = SOURCE / "data" / "ifeval_simple.jsonl"
REQUIRED = (
    "requirements.txt",
    "data/ifeval_simple.jsonl",
    "notebooks/train_linear_probe.ipynb",
    "run/save_LLMs_activations.py",
    "run/representation_engineering.py",
)
MISSING_ARTIFACTS = (
    "generated model responses",
    "IFEval loose-evaluation labels",
    "saved activations",
    "trained probe checkpoints",
    "reported result files",
)


def source_commit() -> str | None:
    if not (SOURCE / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()


def dataset_audit() -> dict[str, object]:
    if not DATASET.exists():
        return {"exists": False}
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line]
    instructions = Counter(
        instruction
        for row in rows
        for instruction in row.get("instruction_id_list", [])
    )
    return {
        "exists": True,
        "rows": len(rows),
        "instruction_counts": dict(sorted(instructions.items())),
        "rows_with_output": sum("output" in row for row in rows),
    }


def source_audit() -> dict[str, object]:
    notebook = SOURCE / "notebooks" / "train_linear_probe.ipynb"
    activations = SOURCE / "run" / "save_LLMs_activations.py"
    intervention = SOURCE / "run" / "representation_engineering.py"
    notebook_text = notebook.read_text() if notebook.exists() else ""
    activation_text = activations.read_text() if activations.exists() else ""
    intervention_text = intervention.read_text() if intervention.exists() else ""
    return {
        "paper_task_split": "70/30",
        "released_notebook_task_split": "80/20"
        if "split=0.8" in notebook_text
        else "not detected",
        "random_seed_values_recorded": False,
        "tokenizer_keyword_typo": "use_auth_toke=" in activation_text,
        "representation_engineering_missing_registry_import": (
            "instructions_registry.INSTRUCTION_DICT" in intervention_text
            and "import instructions_registry" not in intervention_text
        ),
        "representation_engineering_undefined_training_names": (
            "following_acts, non_following_acts = acts[labels == 1], acts[labels == 0]"
            in intervention_text
        ),
    }


def audit() -> dict[str, object]:
    return {
        "paper": MANIFEST["paper"],
        "venue": MANIFEST["venue"],
        "source": str(SOURCE),
        "commit": source_commit(),
        "expected_commit": MANIFEST["commit"],
        "required_files": {path: (SOURCE / path).exists() for path in REQUIRED},
        "dataset": dataset_audit(),
        "paper_settings": MANIFEST["paper_settings"],
        "source_audit": source_audit(),
        "compatibility": {
            "implemented": True,
            "apple_digest": compat.tree_digest(SOURCE),
            "expected_apple_digest": MANIFEST["source_digest"],
            "ifeval_commit": compat.git_commit(compat.GOOGLE),
            "expected_ifeval_commit": MANIFEST["evaluator"]["commit"],
            "ifeval_digest": compat.tree_digest(compat.IFEVAL),
            "expected_ifeval_digest": MANIFEST["evaluator"]["source_digest"],
        },
        "missing_artifacts": MISSING_ARTIFACTS,
        "baseline_runnable_unchanged": False,
        "baseline_runnable_with_compatibility": True,
    }


def inspect() -> None:
    print(json.dumps(audit(), indent=2))


def preflight(model_key: str | None) -> None:
    load_project_env()
    state = audit()
    if state["commit"] not in (None, state["expected_commit"]):
        raise RuntimeError(
            f"Expected upstream commit {state['expected_commit']}, found {state['commit']}"
        )
    missing = [path for path, exists in state["required_files"].items() if not exists]
    if missing:
        raise RuntimeError(f"Missing upstream files: {', '.join(missing)}")
    sources = compat.verify_sources()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    compat.validate_extension_models()
    compat.validate_control_models()
    models = [compat.MODELS[model_key]] if model_key else list(compat.MODELS.values())
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    access = {}
    for model in models:
        info = api.model_info(model["id"], revision=model["revision"])
        access[model["key"]] = {
            "revision": info.sha == model["revision"],
            "scope": model.get("scope", "baseline"),
            "layers": model["layers"],
        }
        if model.get("scope") in {"extension", "control"}:
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            config = AutoConfig.from_pretrained(
                model["id"], revision=model["revision"], token=os.environ["HF_TOKEN"]
            )
            loader = AutoModelForCausalLM._model_mapping[type(config)]
            tokenizer = AutoTokenizer.from_pretrained(
                model["id"], revision=model["revision"], token=os.environ["HF_TOKEN"]
            )
            if loader.__name__ != model["architecture"]:
                raise RuntimeError(
                    f"Expected {model['architecture']}, found {loader.__name__}"
                )
            access[model["key"]].update(
                {
                    "transformers": transformers.__version__,
                    "architecture": loader.__name__,
                    "tokenizer": type(tokenizer).__name__,
                }
            )
            if model.get("prompt_mode") == "native" and not getattr(
                tokenizer, "chat_template", None
            ):
                raise RuntimeError(f"Native chat template missing for {model['key']}")
            released_layers = compat.upstream_saved_layers(model["num_hidden_layers"])
            access[model["key"]].update(
                {
                    "released_saved_layers": released_layers,
                    "layer_adapter_required": not set(model["layers"]).issubset(
                        released_layers
                    ),
                }
            )
    print(
        json.dumps(
            {
                "status": "ready_for_gpu",
                "sources": sources,
                "model_access": access,
                "hf_bucket_configured": bool(os.environ.get("HF_BUCKET")),
                "protocol_deviations": [
                    "released 510-row artifact used because the paper's exact 500 rows are unidentified",
                    "replacement seeds 0-4 used because the paper's seed values are unavailable",
                    "paper 70/30 split takes precedence over the notebook's 80/20 split",
                    "102 dictionary-valued kwargs rows are wrapped for the official IFEval schema",
                ],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "inspect",
            "preflight",
            "prepare",
            "extract",
            "label",
            "compact",
            "train",
            "select-layer",
            "controls",
            "submission-controls",
            "baseline",
            "aggregate",
            "submission-plan",
            "submission-analysis",
            "submission-compare",
        ),
        required=True,
    )
    parser.add_argument("--model", choices=tuple(compat.MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--normalization",
        choices=("both", *submission.NORMALIZATION_MODES),
        default="both",
    )
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    if args.stage == "inspect":
        inspect()
        return
    if args.stage == "preflight":
        preflight(args.model)
        return
    if args.stage == "aggregate":
        compat.aggregate()
        return
    if args.stage == "submission-plan":
        print(json.dumps(submission.plan(), indent=2))
        return
    if not args.model:
        parser.error(f"--model is required for --stage {args.stage}")
    if args.stage == "prepare":
        print(json.dumps(compat.prepare(args.model), indent=2))
    elif args.stage == "extract":
        compat.extract(args.model, force=args.force)
    elif args.stage == "label":
        compat.label(args.model)
    elif args.stage == "compact":
        compat.compact(args.model, keep_raw=args.keep_raw)
    elif args.stage == "train":
        compat.train(args.model, device=args.device)
    elif args.stage == "select-layer":
        compat.select_layer(args.model, device="cpu")
    elif args.stage == "controls":
        compat.controls(args.model, args.device, args.force, args.keep_raw)
    elif args.stage == "submission-controls":
        compat.controls(args.model, args.device, args.force, args.keep_raw)
        controls = [
            model["key"]
            for model in compat.CONFIG.get("controls", [])
            if model["base_model"] == args.model
        ]
        for model_key in controls:
            submission.analyze(
                model_key,
                device=args.device,
                normalization=args.normalization,
                bootstrap_samples=args.bootstrap_samples,
            )
        print(json.dumps(submission.compare_controls(args.model), indent=2))
    elif args.stage == "submission-analysis":
        print(
            json.dumps(
                submission.analyze(
                    args.model,
                    device=args.device,
                    normalization=args.normalization,
                    bootstrap_samples=args.bootstrap_samples,
                ),
                indent=2,
            )
        )
    elif args.stage == "submission-compare":
        print(json.dumps(submission.compare_controls(args.model), indent=2))
    else:
        compat.baseline(args.model, args.device, args.force, args.keep_raw)


if __name__ == "__main__":
    main()
