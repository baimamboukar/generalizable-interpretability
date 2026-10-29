# When Truthful Representations Flip Under Deceptive Instructions?

The EMNLP 2025 paper studies whether truth-value representations remain stable
when an instruction asks a model to answer deceptively. It trains layer-wise
linear and TTPD probes on the final pre-generation residual-stream state of
Llama-3.1-8B-Instruct and Gemma-2-9B-Instruct.

## Baseline Target

Reproduce Figures 2 and 3: truth-value probe accuracy across layers under
truthful, neutral, and deceptive instructions, including cross-topic transfer.

## Release Audit

| Item | State |
| :--- | :--- |
| Official code | Released |
| Baseline datasets | Omitted from the official repository, but nearly all exact files are recoverable from `sciai-lab/Truth_is_Universal` |
| Preprocessed activations | Not released; regenerable |
| Paper protocol | 5,000 balanced training, 1,000 validation, 5,000 held-out examples; leave-one-topic-pair-out |
| Probe | Per-layer z-scored logistic regression; LBFGS, L2, `C=1`, no intercept, seed 1000 |
| Code–paper agreement | Not exact: prompts, logistic-regression settings, deceptive labels, and activation paths differ |
| Revisions | Model and dependency revisions are not pinned |

The affirmative and negated Figure 2 dataset counts match the paper. Minor
count discrepancies remain for CounterFact and one logical dataset.

## Current Verdict

**Standby: the baseline is recoverable, but the public release is not directly
paper-faithful.** Reproduction would require a documented reconstruction from
the paper specification and upstream datasets rather than simply running the
official repository. No GPU experiment was launched.
