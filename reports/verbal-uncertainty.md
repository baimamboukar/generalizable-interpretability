# Calibrating Verbal Uncertainty As A Linear Feature To Reduce Hallucinations

[Paper](https://aclanthology.org/2025.emnlp-main.187/) · EMNLP 2025 Main, SAC Highlight Award · [Released code](https://github.com/facebookresearch/verbal_uncertainty_feature_calibration)

The paper studies whether a model's verbal uncertainty is represented linearly before it answers. It defines a verbal-uncertainty direction as the normalized difference between the mean residual activations of high- and low-uncertainty answers. It then trains linear regressors on the last question-token hidden states to predict verbal uncertainty and semantic entropy. These signals feed a logistic-regression hallucination detector, while activation steering tests whether the direction is causally useful. The paper reports an average 29.6% relative reduction in confident hallucinations after calibration.

## Evaluation Protocol

| Item | Specification |
| :--- | :--- |
| Venue | EMNLP 2025 Main; SAC Highlight Award |
| Models | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, and Qwen2.5-7B-Instruct; Llama-3.1-70B-Instruct is also used as judge and in mitigation |
| Data | TriviaQA RC, NQ-Open, and PopQA; 10,000 train, 1,000 validation, and 1,000 test questions each |
| Data policy | Questions are fixed benchmarks; answers, uncertainty scores, and hallucination labels are generated separately for each evaluated model |
| Probe | Linear regression over concatenated last-question-token residual states from selected layers |
| Probe targets | Verbal uncertainty and semantic entropy |
| Detection | Logistic regression using semantic uncertainty, verbal uncertainty, or both |
| Primary target | Table 2 probe-predicted hallucination-detection AUROC and accuracy |
| Secondary targets | Table 1 calculated-uncertainty detection, Figure 5 causal validation, and Table 3 mitigation |

On Llama-3.1-8B, the paper reports probe-predicted AUROCs of 73.53% on TriviaQA, 57.15% on NQ-Open, and 74.76% on PopQA.

## Baseline Replication

We regenerated the complete Llama-3.1-8B Table 2 pipeline across 10,000 training, 1,000 validation, and 1,000 test questions per dataset. This includes model answers, Llama-3.1-70B judge outputs, hallucination labels, last-question-token activations, six uncertainty regressors, and the downstream logistic detectors. The resulting combined probe AUROCs were 77.25% on TriviaQA, 56.52% on NQ-Open, and 81.64% on PopQA. Their absolute deviations from the paper were 3.72, 0.63, and 6.88 percentage points. Accuracy was less stable: 77.5%, 57.3%, and 72.3%, respectively, or 2.6, 14.0, and 8.7 points below the paper.

The ranking signal is broadly present, but the full baseline is not a tight numerical reproduction. Because the generated answers and judge labels were not released, the run recreates the protocol rather than the paper's exact sampled evaluation set. The stronger AUROC agreement than accuracy agreement is consistent with a threshold or calibration shift under regenerated data, but does not isolate its cause.

## Release Audit

The repository is pinned at `c445bb678d9c9217a6a28d10656b789473c839dc`. It releases the full methodological source but not the generated answers, judge outputs, merged labels, hidden states, trained regressors, or result tables. Reproducing the baseline therefore requires regenerating every model-dependent artifact. Because these outputs are sampled and judged by another model, seed 42 does not make the released paper artifacts exactly recoverable.

The checked-in shell files are illustrative rather than a complete end-to-end paper matrix. The detection entrypoint loops over models but omits its required dataset argument; the verbal-uncertainty, labeling, and probe-training wrappers cover only narrow examples. Orchestration also assumes the authors' Slurm and Conda setup. Several core package and model revisions are unpinned, and hard-coded checkpoint ranges do not always match the layer ranges stated in the appendix. These are specification gaps that must be recorded rather than silently repaired.

## Decision

The primary Llama-3.1-8B Table 2 replication is complete. It provides supporting evidence rather than a clean replication success: AUROC is broadly recovered, while fixed-threshold accuracy differs materially on NQ-Open and PopQA. Figure 5 and Table 3 remain outside this baseline target.
