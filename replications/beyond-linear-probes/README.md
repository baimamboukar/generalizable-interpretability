# Beyond Linear Probes

Source-preserving setup for [Beyond Linear Probes: Dynamic Safety Monitoring
for Language Models](https://arxiv.org/abs/2509.26238), pinned to
`e50d3c4f1bd01c90c08e571c1500d8b4aa62bd13`.

The wrapper executes the released extraction and training scripts unchanged.
It only selects the upstream script, creates the hard-coded activation-cache
path, validates the release, and copies outputs under `GI_ARTIFACT_DIR`.

## Baseline contract

The paper's primary baseline is WildGuardMix with mean-pooled residual-stream
activations. It compares linear, low-rank bilinear, parameter-matched MLP,
early-exit MLP, and fifth-order rank-64 TPC probes. The paper uses five random
seeds, an 80/20 train-validation split, batch size 1024, 50 epochs per joint
run or per progressive term, and the reported AdamW/scheduler configuration.

The four released model conditions and the two Table 1 layers are:

| Model | Hugging Face ID | Layers |
| :--- | :--- | :--- |
| Gemma | `google/gemma-3-27b-it` | 32, 40 |
| Qwen3 | `Qwen/Qwen3-30B-A3B-Base` | 32, 40 |
| GPT-OSS | `openai/gpt-oss-20b` | 16, 20 |
| Llama | `meta-llama/Llama-3.2-3B` | 16, 20 |

Paper targets for Table 1 are frozen in
`data/results/beyond-linear-probes/baseline_targets.csv`. The upstream source
uses the spelling `BeaverTrails`; the paper calls that dataset BeaverTails.

## Commands

```bash
uv run python replications/beyond-linear-probes/run.py --stage inspect
uv run python replications/beyond-linear-probes/run.py --stage preflight
```

Extract all released layers for one model:

```bash
python replications/beyond-linear-probes/run.py \
  --stage extract --model gemma --dataset WildGuard
```

Train one paper-faithful layer after extraction:

```bash
python replications/beyond-linear-probes/run.py \
  --stage train --model gemma --dataset WildGuard --layer 40
```

Run the full WildGuard baseline matrix:

```bash
python replications/beyond-linear-probes/run.py \
  --stage baseline --dataset WildGuard
```

After the synced results arrive, create the paper-versus-replication figure:

```bash
python replications/beyond-linear-probes/plot.py
```

The plot reads the synced results under `artifacts/beyond-linear-probes` and
writes `figures/beyond-linear-probes/paper_vs_replication.png`. Pass `--pdf`
only when a PDF export is needed. A selected-model comparison can be generated
with `--models llama3`.

The same command can be transported through the project Vast adapter. Pass
`--external tpc` so the upstream checkout is bundled, and pass only the
credentials required for the Hugging Face cache and artifact sync:

```bash
uv run python -m infra vast \
  --label tpc-wildguard-baseline \
  --gpu A100_PCIE --min-vram 80 --disk 500 \
  --external tpc \
  --env-key HF_TOKEN --env-key HF_BUCKET --env-key HF_SYNC_ENABLED -- \
  bash -lc \
  'uv pip install --system --no-cache -r replications/beyond-linear-probes/requirements-runtime.txt && \
   python replications/beyond-linear-probes/run.py --stage baseline --dataset WildGuard'
```

The upstream repository has no dependency lockfile, does not pin model
revisions, and hard-codes `/data/scratch/acw663/poly-monitor/activation-cache`.
Its Qwen3 WildGuard extractor also writes `wildguard_train/test` while the
released monitor reads `train/test`; the wrapper reports this mismatch instead
of silently renaming files. These are release diagnostics, not source changes.

## Verified baseline

The Llama-3.2-3B layer-16 WildGuardMix run reproduced all five Table 1 test-F1
values within 0.44 percentage points. The mean absolute difference was 0.20
points. Results, verification hashes, and the paper comparison are recorded in
`data/results/beyond-linear-probes/baseline_replication.csv`,
`artifacts/beyond-linear-probes`, and `reports/beyond-linear-probes.md`.

The released training files require Python 3.12 syntax, but no Python version
or dependency lockfile was published.
