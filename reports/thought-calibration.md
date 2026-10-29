# Thought Calibration

[Paper](https://aclanthology.org/2025.emnlp-main.722/) · EMNLP 2025 Main · [Released code](https://github.com/rmwu/thought-calibration)

The paper uses lightweight probes to decide when a reasoning model can stop generating tokens. It averages the final-layer hidden states within each reasoning step and trains PCA-reduced logistic regressions for four signals: correctness, answer consistency, whether a step attempts a final answer, and whether it adds new information. Learn-then-Test calibration turns the probe scores into stopping thresholds. The reported method preserves accuracy while reducing thinking tokens by up to 60% on held-out s1K-1.1 and up to 20% on AIME 24, GPQA Diamond, and MATH-500.

## Evaluation Protocol

| Item | Specification |
| :--- | :--- |
| Venue | EMNLP 2025 Main |
| Models | DeepSeek-R1-Distill-Qwen-32B, DeepSeek-R1-Distill-Llama-70B, QwQ-32B |
| Representation | Mean final-layer hidden state for each reasoning step |
| Probe | StandardScaler, 256-component PCA, logistic regression |
| Signals | Correct, consistent, novel, leaf |
| Data split | 500 train, 50 test, 450 calibration examples from s1K-1.1, in dataset order |
| Primary result | Figures 3–4: accuracy versus thinking-token reduction |
| Architecture result | Appendix Table 1: linear, MLP, and Transformer AUROC |

## Release Audit

The source is pinned at `bba28b7c76b8e331f3a74cc1cc28cf28ec052441`. Figshare releases the three models' step embeddings and labels (4.54 GB), plus trained linear probes and calibrated thresholds (92.54 MB). The linear notebook implements the reported scaling, PCA, logistic regression, and data split. The release specifies Python 3.12, CUDA 12.6, lmdeploy 0.7.3, four A6000 GPUs, and a single run; other package versions are not locked.

The public code does not implement the MLP or Transformer probes reported in Appendix Table 1. The paper gives architecture grids, but not a complete Transformer optimization and model-selection protocol. In addition, the training-data record omits metadata files consumed by `2-probe.ipynb`; some can be regenerated through the preparation notebook, while the step-level metadata path is commented out. Reconstructing these pieces would introduce new implementation choices.

## Status

Audit complete. The released linear-probe and calibration path is available for a constrained baseline check, but direct end-to-end retraining needs regenerated metadata. Appendix Table 1 is **not source-faithfully reproducible** from the public release, so it cannot currently serve as an independent architecture-stability replication target. No artifact download or paid compute has been started.
