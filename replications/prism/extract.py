from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


TRUE_DOMAINS = ["animals", "cities", "companies", "elements", "facts", "inventions"]
LOGIC_DOMAINS = ["animal_class", "cities", "element_symb", "facts", "inventors", "sp_en_trans"]
LOGIC_FORMS = ["", "_neg", "_conj", "_disj"]
PROMPT = "Does the statement '{s}' accurately reflect the truth?"


def domains(dataset: str) -> list[str]:
    if dataset == "true":
        return TRUE_DOMAINS
    return [f"{domain}{form}" for domain in LOGIC_DOMAINS for form in LOGIC_FORMS]


def main() -> None:
    args = parser().parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    ).eval()

    for domain in domains(args.dataset):
        target = output / f"{domain}_last_token.json"
        if target.exists():
            print(f"Skipping {target}")
            continue
        rows = json.loads((args.raw_data / args.dataset / f"{domain}.json").read_text())
        results = []
        for row in tqdm(rows, desc=domain):
            text = row["statement"].strip()
            if args.representation == "prompted":
                text = PROMPT.format(s=text)
            token_ids = tokenizer.encode(text)
            inputs = torch.tensor([token_ids], device=model.device)
            with torch.inference_mode():
                states = model(inputs, output_hidden_states=True).hidden_states[-1]
            results.append({"hd": states[0, -1].float().cpu().tolist(), "label": row["label"]})
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(results))
        temporary.replace(target)

    metadata = {
        "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "dataset": args.dataset,
        "representation": args.representation,
        "dtype": "float32",
        "attention": "eager",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "protocol": "raw tokenizer encoding, final-layer last-token state",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", required=True)
    cli.add_argument("--dataset", choices=["true", "logic"], required=True)
    cli.add_argument("--representation", choices=["raw", "prompted"], required=True)
    cli.add_argument("--raw-data", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--trust-remote-code", action="store_true")
    return cli


if __name__ == "__main__":
    main()
