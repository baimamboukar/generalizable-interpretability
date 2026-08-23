# Models Under Pressure confirmatory protocol

## Status

The existing Qwen2.5, Qwen3, and Gemma artifacts are suitable for exploratory
layer screening and metric recomputation. They are not confirmatory evidence:
they contain test-set outputs only, do not record independent seed identities,
and do not pin the model revision or activation provenance.

| Artifact | Result rows | Split | Seed provenance | Confirmatory ready |
|---|---:|---|---|---|
| `qwen25-32b-eval-a100-r11` | 60 | test | missing | no |
| `gemma4-31b-eval-a100-r11` | 180 | test | missing | no |
| `qwen3-32b-extra5-h200-r8` | 60 | test | missing | no |

The Gemma files contain three appended evaluations for every result cell, without
independent seed provenance. They cannot be treated as a three-seed run. All
three artifacts retain row-level IDs, labels, and scores. They can therefore
support threshold-free exploratory metrics without rerunning inference. They
cannot support development-only layer selection, frozen development thresholds,
or between-seed uncertainty.

## Confirmatory design

`submission_protocol.json` freezes the corpus, revisions, layers, three seeds,
dataset hashes, activation site, and analysis before new runs. For each probe and seed, one trained probe is
evaluated on both the development and test datasets. The analysis compares:

1. the layer nearest the prespecified 40% relative depth;
2. the best development layer in the prespecified sparse grid;
3. the best development layer among all screened candidates;
4. the best test layer, reported only as a diagnostic oracle upper bound.

Layer selection maximizes macro-average development AUROC. The decision threshold
maximizes balanced accuracy on pooled development scores and is frozen before
test evaluation. Test outputs include AUROC, AUPRC, accuracy, balanced accuracy,
precision, recall, F1, Brier score, ECE, confusion counts, and TPR at 1% and 5%
FPR. Row-level outputs are retained. Development-selected results receive an
ID-clustered conditional bootstrap interval, and every strategy receives a three-seed
aggregate with raw seed SD and range. For the development-selected strategy, a
full-pipeline bootstrap jointly resamples development and test rows, repeats
layer and threshold selection, and feeds a seed-plus-row hierarchical interval.
The attention-minus-softmax contrast is paired by seed and bootstrap draw. The
released IDs are unique within each dataset, so row resampling currently
reduces to a row bootstrap; no higher-level conversation or trajectory grouping
identifier is available in these artifacts.

Primary activations are the input to each decoder block's `input_layernorm`: the
pre-normalization residual stream specified by the paper. The released source
instead stores the layer-normalization output; this is recorded as an
implementation discrepancy and is not mixed into the primary analysis. A
submission activation is reused only when its sidecar matches the model
revision, dataset hash, layer, and activation site. Legacy activations without
this evidence are re-extracted.

Analysis also requires an artifact `run.json` matching the frozen model ID and
revision, layers, probes, seeds, activation site, protocol digest, dataset
digests, activation provenance coverage, and deterministic upstream source-tree
digest. Result rows must match the expected model, layer, probe, seed-bearing
configuration ID, split, dataset, and row-array lengths. Stale or mixed artifacts
are rejected before selection or metric calculation.

## Verification and execution

Resolve a model configuration without starting compute:

```bash
python replications/models-under-pressure/vast_mup.py \
  --model-key qwen3-32b \
  --stage plan
```

The same wrapper accepts `qwen25-32b` and `gemma4-31b`. Once an explicitly
authorized run has produced an artifact, verify and analyze it with:

```bash
python replications/models-under-pressure/submission_analysis.py audit \
  --artifact artifacts/models-under-pressure/mup-confirmatory-v1-qwen3-32b \
  --model-key qwen3-32b

python replications/models-under-pressure/submission_analysis.py analyze \
  --artifact artifacts/models-under-pressure/mup-confirmatory-v1-qwen3-32b \
  --model-key qwen3-32b \
  --output data/results/model-under-pressure/confirmatory/qwen3-32b
```

Analysis refuses incomplete split, layer, probe, or seed coverage. No confirmatory
result exists until that gate passes.

Existing test-only artifacts cannot pass this contract and remain exploratory.
New runs are required for independent seeds, development selection, frozen
thresholds, exact provenance, and full-pipeline uncertainty. No new inference or
paid compute was run while implementing these checks.
