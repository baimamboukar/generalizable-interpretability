# Confirmatory submission protocol

This directory freezes the analysis after exploratory results and before evaluating two sealed model revisions. Existing-model analyses are retrospective reanalyses; only the sealed revisions are prospective confirmatory holdouts. The paper tests when an activation-probe result remains reproducible, selectable, and operational across model families. It does not treat target-specific retraining as transfer of a frozen learned monitor.

The three primary anchors isolate different portability boundaries:

1. Models Under Pressure: layer selection, probe architecture, and deployment thresholds.
2. Internal Instruction Following: response policy, label support, prompt format, and normalization.
3. Political Perspective: head and layer selection under equal search budgets.

PRISM, Lookback Lens, and Beyond Linear Probes are positive controls. Among Us and Sycophancy contribute only to baseline reproducibility. Missing-artifact and runtime-blocked studies remain in the artifact audit and cannot count as empirical generalization failures.

## Rules

- Select layers and thresholds using development data only.
- Report oracle test selection only as an upper-bound diagnostic.
- Aggregate within seed before across-seed inference.
- Retain every prespecified condition, including positive and null results.
- Save row identifiers, cluster identifiers, labels, scores, probabilities, predictions, thresholds, seeds, and provenance.
- Keep the two confirmatory model revisions sealed until all adapters and analyses pass without task-outcome inspection.

Run the dependency-free protocol checks with:

```bash
python -m unittest confirmatory.test_metrics
python -m confirmatory.verify
```

The machine-readable specification is in `protocol.json`; `hypotheses.csv` freezes headline comparisons and `study_registry.csv` is the corpus ledger.
