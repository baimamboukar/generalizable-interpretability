# Paper artifact package

This directory contains the canonical, plotting-ready evidence tables for the
replication paper. It is built only from repository reports and retained result
artifacts. Blank values mean unavailable, not zero.

## Tables

- `study_inventory.csv`: all 23 audited studies, including empirical studies,
  artifact audits, in-progress work, and out-of-scope methods.
- `baseline_fidelity.csv`: one row per paper-versus-replication comparison;
  all metrics are normalized to fractions.
- `baseline_fidelity_summary.csv`: within-study fidelity summaries used by
  Figure 2.
- `cross_model_extensions.csv`: the validated 68-row extension synthesis.
- `evaluation_contracts.csv`: what remains fixed or changes when the target
  model changes.
- `reproducibility_audit.csv`: release status and the primary protocol or
  artifact gap for every study.
- `study_verdicts.csv`: baseline and extension verdicts for the complete audit
  corpus.
- `artifact_coverage.csv`: availability of metrics, predictions, confusion
  counts, calibration, low-FPR measurements, and uncertainty.
- `experiment_gaps.csv`: the remaining experiment and documentation ledger.

The three `.tex` files are paper-ready table exports. Their captions preserve
the distinction between empirical failures and studies that could not be run
faithfully.

## Figures

The corresponding figures are in `figures/paper-artifacts/`:

1. study accounting;
2. baseline fidelity;
3. comparable cross-model portability;
4. evaluation-contract matrix.

Figures use a white background and Space Grotesk. Oracle-selected extensions,
failed-baseline diagnostics, and incomparable estimands are excluded from the
normalized-retention figure but remain available in the long-form table.

## Rebuild

```bash
python scripts/build_paper_artifacts.py
python scripts/plot_paper_artifacts.py
python scripts/verify_paper_artifacts.py
```

No new model inference or probe training is performed by these commands.
