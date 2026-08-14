# Generalization Analysis

This document summarizes why probing methods failed to generalize across models in this project. Only studies where a cross model generalization test was executed are included. Studies where the baseline could not be replicated and the generalization test was therefore skipped are not included.

## Models Under Pressure

The paper proposes attention and softmax probes at layer 31 of Llama 3.3 70B to detect high stakes interactions. The baseline was replicated on six balanced datasets. Attention reached a mean AUROC of 0.9184 and softmax reached 0.9244, both within the reported range. The same probes were then trained and evaluated on Qwen3 32B, DeepSeek R1 Distill Qwen 32B, and GLM 4 32B at multiple relative depths.

Attention generalized in three of nine tested configurations. DeepSeek at relative depths 0.44 and 0.56 and GLM at 0.66 remained within 0.02 of the baseline. All other depths fell below that threshold, including both Qwen depths and deeper DeepSeek and GLM layers. Softmax generalized in only one of six configurations. DeepSeek at 0.69 remained within threshold, while all Qwen and remaining DeepSeek and GLM layers fell substantially below, with Qwen at 0.50 and 0.73 showing the largest decline and a true positive rate at one percent false positive rate of zero.

Four factors explain the pattern. First, dataset variation contributes but does not dominate. The mental health subset is the weakest outlier in both baseline and extensions, yet the gap between models at the same depth exceeds the gap between datasets. Second, policy and training recipe contribute as a secondary factor. Distilled DeepSeek transfers better than base Qwen at comparable depth and scale, which suggests that supervision influences where the relevant signal resides. Third, policy architecture is a primary factor. The optimal relative depth shifts across families. The original optimum at 0.39 does not transfer, and no single depth predicts success across Llama, Qwen, and GLM. Fourth, probe architecture is a primary factor for the softmax family. Softmax probes are consistently less robust than attention probes under the same data and optimization, which indicates that the probe formulation itself does not transfer even when the underlying representation is present.

Full results are in reports/model-under-pressure.md and data/results.

## Among Us

Among Us trains bias free linear probes on Phi-4 at layer 20 to detect deceptive reasoning. In domain learning was strong, but cross dataset generalization fell from a reported mean of 0.8468 to 0.7339 in replication. Cross model generalization was then tested on Llama 3.1 8B, Gemma 2 27B, and Qwen2.5 32B with the same probe at three relative depths. Gemma at 0.48 reached 0.8011 and exceeded the reconstructed Phi-4 baseline, while the best Llama and Qwen configurations remained below it at 0.7617 and 0.7277. In domain AUROC approached 0.98 in every family, while cross dataset AUROC plateaued near 0.66 to 0.75.

The dominant factors are dataset and artifact completeness. Transfer fails across deception corpora even within the original model, and released checkpoints score near chance on the reconstructed cache. Distribution shift between training and evaluation deception datasets explains the plateau. Policy and architecture modulate the magnitude of the plateau but do not restore transfer.

Full results are in reports/among-us.md and data/results/among-us/extension_probe_results.csv.

## Political Perspective

Political Perspective fits ridge regressions on pre layer normalization attention head outputs to predict legislative ideology. The baseline peaks were replicated exactly at 0.8536 for Llama 2 7B, 0.8463 for Mistral 7B, and 0.8612 for Vicuna 7B. Cross model generalization was tested on Qwen3 8B, GLM 4 9B, Gemma 2 9B, Gemma 2 27B, and Phi 4 with the same prompt and regularization.

Generalization was uneven. Gemma 2 9B reached 0.8949 and Phi 4 reached 0.8714, both above the original range, while Qwen3 8B at 0.8252 and GLM 4 9B at 0.8069 fell below it by about 0.03 and 0.05. The peak head location also shifted from layer 15 to layer 33 across families, and scaling Gemma from 9B to 27B did not improve the peak.

The primary factor is policy architecture. Dataset and evaluation protocol were held fixed, and optimization was identical, so the unevenness is attributable to differences in depth and head organization across families rather than to data or training objective alone.

Full results are in reports/political-perspective.md.

## Internal Instruction Following

Internal Instruction Following trains bias-free linear sigmoid probes on
response-token hidden states to predict whether an answer satisfies its
instruction. The four-model baseline reproduces the paper's central split:
strong transfer to unseen tasks but near-chance transfer to unseen instruction
types. We extended the same protocol to Qwen3 14B, GLM 4.7 Flash, and Gemma 4
12B at four relative depths.

Unseen-task transfer remains strong at 0.8971 for Qwen, 0.6867 for GLM, and
0.6975 for Gemma. Failure is concentrated in unseen instruction types, where
paper-aligned early-layer AUROC falls to 0.5223, 0.4267, and 0.4382. Qwen has
only three valid leave-one-type-out folds and Gemma has four: two Qwen
instruction types become entirely positive and one Gemma type becomes entirely
negative.

The primary factor is the interaction between model policy and data. Every
model generates and labels its own answers, so the model-family change also
changes class balance and evaluation support. The probe consequently captures
instruction-specific behavioral correlates rather than a stable cross-family
direction. Layer organization is a secondary factor because later GLM and Gemma
layers recover stronger cells than the paper-aligned early layer. A fixed
Llama-2 prompt template is a plausible additional confound, but the current
results do not isolate it. Fixed-response and native-template controls are
required before attributing the failure solely to architecture.

Full results are in reports/internal-instruction-following.md.

## Scope Note

Studies where the baseline was not reproduced and no cross model test was performed are excluded from this analysis. Studies where cross model generalization was tested and succeeded, including PRISM, Implicit Personalization, and Lookback Lens, are also excluded because they did not fail to generalize.
