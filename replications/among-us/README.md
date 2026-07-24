# Among Us Probe Replication

Replication of the Phi-4 linear probes from **Among Us: A Sandbox for Measuring
and Detecting Agentic Deception**. Game/Elo and SAE experiments are out of scope.
The pinned upstream checkout in `external/AmongUs` remains unchanged.

## Baseline

| Setting | Value |
| :--- | :--- |
| Model | `microsoft/phi-4` |
| Layer | 20/40, residual output |
| Probe | Bias-free linear classifier |
| Training | Adam, LR `0.001`, weight decay `0.001`, 4 epochs, batch 32 |
| Split | 80% train / 20% test |
| Datasets | TruthfulQA, DishonestQA, RepEng, Among Us |

The paper/config uses the final 10 tokens; the released Figure 7 script uses 5.
The runner uses the released dataset/preprocessing classes unchanged and the
literal released Phi-4 loader (no revision or dtype override). It supports the
paper's 10-token path and the released Figure 7 script's 5-token path in
separate caches.

## Commands

```bash
# Public data, then the official GPT-4o-mini labeling script unchanged
modal run replications/among-us/modal_probes.py --stage prepare
modal run replications/among-us/modal_probes.py --stage labels

# Phi-4 L20 activations
modal run replications/among-us/modal_probes.py --stage cache --dataset TruthfulQA
modal run replications/among-us/modal_probes.py --stage cache --dataset DishonestQA
modal run replications/among-us/modal_probes.py --stage cache --dataset RepEng
modal run replications/among-us/modal_probes.py --stage cache --dataset AmongUs

# Train and evaluate
modal run replications/among-us/modal_probes.py --stage inspect
modal run replications/among-us/modal_probes.py --stage train
modal run replications/among-us/modal_probes.py --stage evaluate
```

Add `--profile release-figure7` to training/evaluation to reproduce the released
five-token path. The 5- and 10-token paths use separate caches.

## Cross-model extension

All extensions keep the released five-token probe, datasets, 80/20 ordered
split, optimizer, and evaluation matrix. Layers are selected near 25%, 50%, and
75% depth; all three are captured in the same forward pass. Model inference is
bfloat16 on an A100-80GB, while cached activations and probe training are fp32.

| CLI key | Model | Parameters | Layers |
| :--- | :--- | ---: | :--- |
| `llama-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | 8, 16, 24 / 32 |
| `gemma-27b` | `google/gemma-2-27b-it` | 27B | 10, 22, 34 / 46 |
| `qwen-32b` | `Qwen/Qwen2.5-32B-Instruct` | 32.5B | 16, 32, 48 / 64 |
| `glm4-32b` | `zai-org/GLM-4-32B-0414` | 32B | 15, 30, 45 / 61 |

Run one model at a time:

```bash
modal run replications/among-us/modal_probes.py --stage extension-cache --model llama-8b --dataset all --force
modal run replications/among-us/modal_probes.py --stage extension-inspect --model llama-8b
modal run replications/among-us/modal_probes.py --stage extension-train --model llama-8b
modal run replications/among-us/modal_probes.py --stage extension-evaluate --model llama-8b
```

Replace `llama-8b` with another CLI key for the remaining families.
