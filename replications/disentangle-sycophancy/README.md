# Sycophancy Is Not One Thing

Source-preserving setup for the authors' release at commit
`217b89808d07e978a7c8fac37e8ff363e29ad8ef`.

The first target is the probe-focused baseline: DiffMean residual-stream
directions and AUROC by layer on the released factorial data. Cross-behavior
geometry is a post-processing target over those vectors. The wrapper runs the
released module unchanged; it does not rewrite the authors' source. Steering is a later stage because the release
references a missing prompt file and uses a different default layer from the
paper's Qwen3-30B steering table.

```bash
git clone https://github.com/cincynlp/disentangle-sycophancy \
  external/disentangle-sycophancy
git -C external/disentangle-sycophancy checkout \
  217b89808d07e978a7c8fac37e8ff363e29ad8ef

uv run python replications/disentangle-sycophancy/run.py --stage inspect
uv run python replications/disentangle-sycophancy/run.py --stage preflight
```

The paper-faithful baseline command is:

```bash
uv run python replications/disentangle-sycophancy/run.py \
  --stage baseline --shard-model
```

For a GPU provider, run inside the authors' locked environment. The checkout
specifies Python 3.12 in `.python-version` and pins the CUDA 12.6 stack in
`uv.lock` (`torch 2.8.0`, `transformers 4.56.2`).

```bash
uv run python -m infra vast \
  --label sycophancy-qwen3-30b-baseline-locked \
  --gpu H100_SXM --disk 250 \
  --external disentangle-sycophancy \
  --env-key HF_TOKEN --env-key HF_BUCKET --env-key HF_SYNC_ENABLED -- \
  uv run --project external/disentangle-sycophancy --frozen --python 3.12 \
  python replications/disentangle-sycophancy/run.py \
  --stage baseline --shard-model
```

`--frozen` is intentional: current uv reports that the committed upstream lock
needs a metadata refresh, but refreshing it would alter the authors' release.

`requirements-release.txt` is retained only as an exact Linux fallback; it is
not the primary reproduction environment.

Outputs go to `GI_ARTIFACT_DIR/disentangle-sycophancy` (or `artifacts/` when
the variable is unset). Use `--stage aggregate --output-dir ...` after a run
to write the plotting-ready long table.
