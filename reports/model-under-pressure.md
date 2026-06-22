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
- Verified hook is `input_layernorm` of L31 = residual before layernorm, matches paper.
- Error bars (3 seeds, cache off so each truly retrains). Mean AUROC **0.918 ± 0.001**. Confirmed it is stable.

## June 22, 2026 | cross-model generalization

- Next is the probe method generalizes to other model families (Gemma, Qwen), not just Llama.
- Extract activations on other models (start Gemma) for train, val, and the six balanced test evals, at a sweep of layers, then probe per layer.
