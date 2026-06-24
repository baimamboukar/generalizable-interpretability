# Model Under Pressure Replication Report

## June 19, 2026 | Completed Setup.

- It required setting up various infra stuff.. Like R2 (for activations storage),OpenRouter Key, OpenAI key, WandDB keys
- The project comes with a handy tool `mup` (basically a python package with a CLI)
- This helped download all the heavy data locally in a flash with `mup datasets download`
- R2 bucket is `mup-replication`, auth passes, currently empty.

## June 20, 2026 | First activations on Modal.

- Wrote `replications/models-under-pressure/modal_acts.py`: runs `mup acts store` on an H100, uploads to R2.
- Secrets live in a named Modal secret `mup-env` (from `.env`)
- `modal run replications/models-under-pressure/modal_acts.py --model <hf-id> --layers 11 --dataset <path>`
- Extracted activation of layer 11 across Llama-3.3-70B, Llama-3.1-8B, and Gemma-2-12B for the training set.
- Activations are stored in a modal volume, and synced to R2 under `mup-replication/activations/` (See manifest file in R2 for details)

## June 21, 2026 | Paper reading

- Read the paper in detail
- The probing experiments are done on `Llama-3.3-70B@L31`. Layer 31/80 was chosen based on a cross-validation
- Only the other baselines (FT and Prompting) were tested on the other two models (Llama-3.1-8B and Gemma-2-12B)
- They trained various probes architectures (Softmax, Attention, Mean, Max, Max of Rolling Mean)
- For FT, the autoregressive LLMs' final unembedding layer were removed, and a classification head was plugged, before fine-tuning on the training set.

## June 21, 2026 | Headline reproduced

- Extracted Llama-3.3-70B @ L31 for train, val (training test split), and all six balanced test evals.
- Ran their attention probe via their CLI: `mup exp +experiment=evaluate_probe probe=attention model=llama-70b eval_datasets=test_balanced`. We reused their pipeline
- Six-dataset mean AUROC is **0.9189** .The paper reports 0.91+.
  - mts 0.992
  - mt 0.991
  - anthropic 0.957
  - aya 0.909
  - toolace 0.876
  - mental_health 0.789
- mental_health is the weak outlier; everything else high.
- Hook mismatch (code vs paper). The code registers a forward hook on `layers[L].input_layernorm` and stores it, i.e. **post-LN**. But the paper says "residual activations **before** layer norm". We patched the released code to remain faithful to the paper's framing.

## June 22, 2026 | cross-model generalization

- Next is the probe method generalizes to other model families (Gemma, Qwen), not just Llama.
- Extract activations on other models (start Gemma) for train, val, and the six balanced test evals, at a sweep of layers, then probe per layer.
- The extraction hook on layer 31 now captures `input[0]` from `input_layernorm`, i.e. pre-layer-norm residual activations. `ACT_CAPTURE=post` is still available for reproducing the released code if needed.

## Pre-LN Faithful Reproduction

Goal is to reproduce the setup exactly as the paper describes.

1. **Kept the probe hyperparameters on the Figure 2 settings** (all: 200 epochs, AdamW, batch 16, grad-accum 4)
   - Attention: patience 50, LR 5e-3 → 5e-4
   - Softmax: temperature 5, patience 10
   - Mean / Max / Last: patience 50
   - Rolling Max: window 40, patience 50
2. **Re-extract pre-LN L31** for the 8,000 train, 2,000 val, and all six balanced test datasets.

## June 23, 2026 | Attention Probes Replication

## Attention seed 0 result

Seed 0 completed on the paper-faithful Attention setup.

| Dataset                 |  AUROC |
| ----------------------- | -----: |
| anthropic_hh_balanced   | 0.9495 |
| toolace_balanced        | 0.8747 |
| mt_balanced             | 0.9834 |
| mts_balanced            | 0.9924 |
| mental_health_balanced  | 0.8068 |
| aya_redteaming_balanced | 0.9011 |
| Mean                    | 0.9180 |

JSON record:

```json
{
  "probe": "Attention",
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "layer": 31,
  "activation_capture": "pre-LN",
  "hardware": "B200:2",
  "seed": 0,
  "validation": true,
  "use_store": false,
  "results_file": "results/attn_fig2_seed0_v2.jsonl",
  "mean_auroc": 0.9180,
  "per_dataset_auroc": {
    "anthropic_hh_balanced": 0.9495,
    "toolace_balanced": 0.8747,
    "mt_balanced": 0.9834,
    "mts_balanced": 0.9924,
    "mental_health_balanced": 0.8068,
    "aya_redteaming_balanced": 0.9011
  }
}
```

## Attention seed 2 result

| Dataset                 |  AUROC |
| ----------------------- | -----: |
| anthropic_hh_balanced   | 0.9512 |
| toolace_balanced        | 0.8734 |
| mt_balanced             | 0.9844 |
| mts_balanced            | 0.9935 |
| mental_health_balanced  | 0.8096 |
| aya_redteaming_balanced | 0.9019 |
| Mean                    | 0.9190 |

JSON record:

```json
{

  "probe": "Attention",
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "layer": 31,
  "activation_capture": "pre-LN",
  "hardware": "B200:2",
  "seed": 2,
  "validation": true,
  "use_store": false,
  "results_file": "results/attn_fig2_seed2.jsonl",
  "mean_auroc": 0.9190,
  "per_dataset_auroc": {
    "anthropic_hh_balanced": 0.9512,
    "toolace_balanced": 0.8734,
    "mt_balanced": 0.9844,
    "mts_balanced": 0.9935,
    "mental_health_balanced": 0.8096,
    "aya_redteaming_balanced": 0.9019
  }
}
```

## Attention seed 1 result

| Dataset                 |  AUROC |
| ----------------------- | -----: |
| anthropic_hh_balanced   | 0.9497 |
| toolace_balanced        | 0.8750 |
| mt_balanced             | 0.9837 |
| mts_balanced            | 0.9935 |
| mental_health_balanced  | 0.8059 |
| aya_redteaming_balanced | 0.9013 |
| Mean                    | 0.9182 |

JSON record:

```json
{
  "probe": "Attention",
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "layer": 31,
  "activation_capture": "pre-LN",
  "hardware": "B200:2",
  "seed": 1,
  "validation": true,
  "use_store": false,
  "results_file": "results/attn_fig2_seed1.jsonl",
  "mean_auroc": 0.9182,
  "per_dataset_auroc": {
    "anthropic_hh_balanced": 0.9497,
    "toolace_balanced": 0.8750,
    "mt_balanced": 0.9837,
    "mts_balanced": 0.9935,
    "mental_health_balanced": 0.8059,
    "aya_redteaming_balanced": 0.9013
  }
}
```

## Summary

| Seed | Mean AUROC | Results file                       |
| ---- | ---------: | ---------------------------------- |
| 0    |     0.9180 | `results/attn_fig2_seed0_v2.jsonl` |
| 1    |     0.9182 | `results/attn_fig2_seed1.jsonl`    |
| 2    |     0.9190 | `results/attn_fig2_seed2.jsonl`    |

1. Across-seed mean AUROC **0.9184**
2. 95% CI across seed means **0.9161–0.9207**

## June 23, 2026 | Adding Qwen3 Adapter

- Added a `QwenArch` hook path in the extractor for Qwen models.

## Qwen3-32B layer 32 Attention Result

| Dataset                 |  AUROC |
| ----------------------- | -----: |
| anthropic_hh_balanced   | 0.8894 |
| toolace_balanced        | 0.8905 |
| mt_balanced             | 0.9703 |
| mts_balanced            | 0.9557 |
| mental_health_balanced  | 0.5810 |
| aya_redteaming_balanced | 0.8635 |
| Mean                    | 0.8584 |

## Qwen3-32B Layer 47 Attention Result

| Dataset                 |  AUROC |
| ----------------------- | -----: |
| anthropic_hh_balanced   | 0.9173 |
| toolace_balanced        | 0.8964 |
| mt_balanced             | 0.9652 |
| mts_balanced            | 0.9746 |
| mental_health_balanced  | 0.6386 |
| aya_redteaming_balanced | 0.8949 |
| Mean                    | 0.8812 |
