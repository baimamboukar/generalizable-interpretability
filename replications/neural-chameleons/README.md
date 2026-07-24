# Neural Chameleons Replication

Replication setup for **Neural Chameleons: Language Models Can Learn to Hide
Their Thoughts from Activation Monitors**. The two pinned upstream checkouts are
kept unchanged:

- `external/neural-chameleons` — complete official experimental pipeline at
  `04552285b7c11aae9e117d75af22c5c681e617f9`.
- `external/minimal-neural-chameleons` — author-linked released-checkpoint demo
  at `f077304f6bebd31ded4cc868f2115a8c81067960`.

## Baseline contract

The first runnable check uses the released
`serteal/neural-chameleon-gemma_2_9b-layer_12` checkpoint and all 13 shipped
logistic probes. Modal executes the upstream `demo.py` without modification;
our wrapper only streams and preserves its raw stdout.

This is a release smoke test, not yet an independent reproduction of Figure 2.
The official full pipeline cannot run from a clean checkout because every
probe-checkpoint path selected by `run_full_pipeline.py` is absent. We will
decide how to reconstruct that missing prerequisite only after discussing the
release and paper specifications.

## Commands

```bash
# Cheap release/integrity audit
modal run replications/neural-chameleons/modal_run.py --stage inspect

# Author-linked released checkpoint and probes; A100-80GB
modal run replications/neural-chameleons/modal_run.py --stage released-demo

# Regenerate the main scatter plots from the official repository's shipped data
modal run replications/neural-chameleons/modal_run.py \
  --stage paper-plots \
  --plot-type scatter
```

The released demo writes to the persistent `neural-chameleons` Modal volume:

```text
/cache/results/neural-chameleons/released-demo/
└── released_demo_stdout.log
```

Structured result extraction will be added later as a separate post-processing
step, so it cannot affect or be confused with the released experiment.

The released paper targets are frozen in
`data/results/neural-chameleons/baseline_comparison.csv`; replication and delta
columns remain empty until we independently recreate the baseline. The completed
released-demo run remains separate in
`data/results/neural-chameleons/released_demo_results.csv`.
