# LLM Reasoning As Trajectories

[Paper](https://aclanthology.org/2026.acl-long.1237/) · ACL 2026 Main ·
[Released code](https://github.com/slhleosun/reasoning-trajectory)

The paper treats chain-of-thought reasoning as a trajectory through residual
stream representations. It uses linear probes to test whether reasoning-step
identity becomes linearly separable with depth and whether late-step geometry
predicts final-answer correctness. The central probe claims are that
step-specific regions are strongly separable, transfer among three Llama-based
training regimes, and yield a late-trajectory correctness signal with peak
ROC-AUC 0.87.

## Baseline Target

The baseline target comprises the Instruct condition from Figure 1 and
Figure 2b on GSM8K. It regenerates the complete 7,473-example training split,
extracts the activation immediately before each `Step X:` and answer marker at
all 32 layers, trains the released one-vs-rest logistic regressions, and fits
the released correctness predictors.

## Release Audit

| Item | State |
| :--- | :--- |
| Upstream code | Pinned at `258d1d761bf596c42f0695a6320d8da59d17f99f` |
| Models | Llama-3.1-8B, Llama-3.1-8B-Instruct, and DeepSeek-R1-Distill-Llama-8B |
| Data | Public GSM8K and MATH-500 datasets |
| Generated responses and labels | Not released |
| Activations and probe checkpoints | Not released |
| Reported result files | Not released |
| Dependency versions | Not released |
| Model and dataset revisions | Not pinned upstream; pinned by the replication manifest |
| Released runtime paths | Null model and dataset paths cause the loaders to fail without configuration |
| Correctness seeds | Paper reports 42, 123, and 456; the released predictor CLI does not propagate `--seed` |
| Generation batch size | Released config specifies 8, while the generation CLI defaults to 16 |
| Generation length | Model config and helper defaults specify 512, while the released baseline script hard-codes 1,024 new tokens |

The wrapper does not modify any author Python file. It stages the pinned source,
fills only the required runtime paths with pinned Hugging Face snapshots, and
executes the released scripts. Derived sidecars retain thresholds, confusion
matrices, AUROC, accuracy, balanced accuracy, precision, recall, specificity,
FPR, FNR, F1, and correctness-probe row-level scores.

## Verdict

**Baseline not reproduced; extension skipped.** The unchanged released
generation path exceeded the memory budgets tested at the released settings. Batch 16 exhausted
80 GB and 96 GB H100s; the config-prescribed batch 8 exhausted both a 96 GB
H100 and a 141 GB H200 before saving the first example. The released full
artifact path hard-codes 1,024 generated tokens while its model configuration
specifies 512, and retains per-token, per-layer tensors. This is a
release-runtime reproducibility failure, not evidence against the paper's
scientific claim. A faithful baseline requires author clarification or a
separately documented runtime modification.
