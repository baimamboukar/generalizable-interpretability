# Sycophancy Hides Linearly in the Attention Heads

Upstream-faithful setup for the [official release](https://github.com/rifoagenadi/sycophancy)
at commit `09d51c4474d9a43be0d22bebae37c9fadd38b699`. The wrapper runs copies of
the released probe script without modifying its source.

The first target is the probe-localization baseline: linear probes over every
MHA head, MLP layer, and residual-stream layer of Gemma-3-4B-IT and
Llama-3.2-3B-Instruct on TruthfulQA. This corresponds to Figures 2, 3, 9, and
10. Steering is a separate second stage.

```bash
git clone https://github.com/rifoagenadi/sycophancy external/sycophancy
git -C external/sycophancy checkout 09d51c4474d9a43be0d22bebae37c9fadd38b699

uv run python replications/sycophancy/run.py --stage inspect
```

The wrapper fixes the paper’s Adam configuration: learning rate `1e-5`, batch
size `32`, and `16` epochs. It uses the release’s `sycophancy_challenged`
condition, which matches the paper’s correct-to-incorrect behavior after user
disagreement. Results run on seeds `3407`, `3408`, and `3409` while preserving
the release’s fixed data split. The release dependency file is empty.

Run one paper model through Vast.ai:

```bash
uv run python -m infra vast \
  --instance INSTANCE_ID \
  --sync \
  --external sycophancy \
  -- uv run --no-project --python 3.13 \
  --with torch \
  --with transformers \
  --with datasets \
  --with scikit-learn \
  --with tqdm \
  python replications/sycophancy/run.py \
  --stage baseline \
  --model gemma-3
```

Replace `gemma-3` with `llama-3.2` for the second baseline.
