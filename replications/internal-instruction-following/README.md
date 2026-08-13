# Do LLMs Know Internally When They Follow Instructions?

Source-preserving setup for the authors' ICLR 2025 release, pinned at commit
`7ed7558ecda069ebaa70ea1f39565c07a11ed43f`.

The first target is the paper's probe baseline: predict whether each model will
follow an IFEval-simple instruction from first-, middle-, and last-token hidden
states at three layers. Paper values are frozen in
`data/results/internal-instruction-following/paper_targets.csv`.

```bash
uv run python replications/internal-instruction-following/run.py --stage inspect
uv run python replications/internal-instruction-following/run.py --stage preflight
```

The upstream checkout remains untouched. `compat.py` supplies the missing
orchestration and calls the released activation script plus Google's pinned
IFEval evaluator. It uses the paper's 70/30 task split, five recorded replacement
seeds, all three reported layers and token positions, and the released notebook's
normalization behavior.

```bash
uv run python replications/internal-instruction-following/run.py \
  --stage preflight --model llama2-7b
```

One model's full probe baseline is:

```bash
python replications/internal-instruction-following/run.py \
  --stage baseline --model llama2-7b
```

The baseline model keys are `llama2-7b`, `llama2-13b`, `mistral-7b`, and
`phi3-mini`. Generated responses, deterministic labels, compact activations,
probe directions, per-run metrics, and paper-aligned summaries are written to
`GI_ARTIFACT_DIR/internal-instruction-following/<model>`.

After syncing the run artifacts, generate the plotting-ready tables and figures:

```bash
python replications/internal-instruction-following/plot.py
```

This writes the paper comparison, long-format AUROC table, per-run confusion
matrices and derived metrics, and the baseline/extension figures.

The Vast SDK launch template for the first model is:

```bash
uv run python -m infra vast \
  --label internal-if-llama2-7b-baseline \
  --gpu H100_SXM --min-vram 75 --min-cpu 16 --min-ram 64 \
  --max-price 3 --disk 200 \
  --image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  --external internal-instruction-following \
  --external google-research-ifeval \
  --env-key HF_TOKEN --env-key HF_BUCKET -- \
  bash -lc 'python -m pip install -q -r replications/internal-instruction-following/requirements-compat.txt && python replications/internal-instruction-following/run.py --stage baseline --model llama2-7b'
```

The generic runner syncs `artifacts/` to the configured Hugging Face bucket
whether the command succeeds or fails.

The compatibility protocol is necessarily best-available rather than exact:
the released artifact has 510 rows instead of the paper's 500, the five paper
seed values are unknown, and 102 dictionary-valued `kwargs` rows require a
shape-only adapter for Google's evaluator.

## Extension targets

| Key | Checkpoint | Layers |
| :--- | :--- | :--- |
| `glm4.7-flash` | `zai-org/GLM-4.7-Flash` | 20, 29, 38, 47 |
| `qwen3-14b` | `Qwen/Qwen3-14B` | 16, 24, 32, 40 |
| `gemma4-12b` | `google/gemma-4-12B-it` | 21, 30, 39, 48 |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | 9, 14, 19, 24 |

The four layers are spaced at approximately 40%, 60%, 80%, and 100% relative
depth. The same extraction, labeling, and probe pipeline is used. Each run saves
the full confusion matrix and threshold-derived diagnostic metrics alongside
AUROC. Compute runs are launched explicitly through the shared provider wrapper.
