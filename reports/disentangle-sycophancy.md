# Sycophancy Is Not One Thing

The paper decomposes sycophancy into sycophantic agreement (SyA), genuine
agreement (GA), and sycophantic praise (SyPr). It uses difference-in-means
directions on post-layernorm residual activations and tests layerwise AUROC,
cross-dataset geometry, and activation-addition steering.

## Initial audit

The public release is pinned at `217b89808d07e978a7c8fac37e8ff363e29ad8ef`.
The controlled factorial data and DiffMean code are available. The first
baseline target is Qwen3-30B on the arithmetic dataset, matching Figure 1;
Figure 2 geometry will be computed from the resulting DiffMean vectors. No
replication result is claimed yet.

The release has several protocol gaps: its shell scripts invoke modules in a way
that breaks their relative imports; the steering prompt file is absent; the
steering script defaults to layer 32 although the paper reports Qwen3-30B at
layer 46; and praise training and evaluation reuse `data/praise.json`.
Model revisions are also unpinned. The released SPEN translation files (one
standard and one `pro` variant) contain 3,760 and 1,880 rows, versus 4,000 and
4,000 rows in Appendix B. These are recorded before any extension.

## Baseline status

**First attempt blocked by our environment mismatch.** The Qwen3-30B run
completed checkpoint loading on one H100, then failed on the first forward
pass with `Expected mat_a to be BFloat16 matrix got Float` in the Qwen3 MoE
kernel. We had installed an unpinned wrapper requirements file under Python
3.11 instead of using the authors' Python 3.12 checkout and `uv.lock`
(`torch 2.8.0+cu126`, `transformers 4.56.2`). Therefore this run is not a
baseline verdict. No AUROC values were produced and no paper-versus-
replication plot is claimed; the baseline is pending a locked-environment
rerun.
The 192 kB failure log is preserved in the HF bucket at
`artifacts/logs/sycophancy-qwen3-30b-baseline-failed.log`. The temporary H100
was destroyed and verified gone. The corrected launch now uses the upstream
lockfile with `uv --frozen` (current uv otherwise requests a lock metadata
refresh) and does not change the authors' source or `--dtype float32` flag.

Targets: `data/results/disentangle-sycophancy/baseline_targets.csv`

Runner: `replications/disentangle-sycophancy/run.py`
