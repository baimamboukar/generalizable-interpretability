# LayerNavigator Replication Report

## Paper

**LayerNavigator: Finding Promising Intervention Layers for Efficient Activation Steering in Large Language Models**
Bryson Arrot et al. — NeurIPS 2025
Repo: [Bryson-Arrot/LayerNavigator](https://github.com/Bryson-Arrot/LayerNavigator) · Vendored at `external/LayerNavigator` (commit `09e96c72`)

## Method Summary

LayerNavigator proposes a **principled, zero-cost layer selection** strategy for activation steering. Instead of exhaustively testing every layer, it scores each layer's suitability using a **steerability score** derived from activations that are already computed during steering vector extraction:

**S_l = D_l + C_l**

- **Discriminability (D_l):** Fisher's criterion — how well positive/negative activation clusters separate at layer l (via between-class vs total scatter along the steering direction).
- **Consistency (C_l):** Mean cosine alignment between per-sample contrastive difference vectors and the aggregate steering vector at layer l.

The top-k layers by S_l are selected for multi-layer intervention. No extra data or forward passes are needed beyond what steering vector extraction already requires.

## Original Setup

| Aspect              | Detail                                                                       |
| :------------------ | :--------------------------------------------------------------------------- |
| Models              | Llama-3-8B-Instruct (32 layers), Qwen2.5-32B-Instruct (64 layers)            |
| Dataset             | Anthropic Persona (6 behaviors), 700 train / 200 val / 100 test per behavior |
| Steering methods    | Mean Difference (MD), PCA                                                    |
| Intervention        | Add steering vector at last token position, selected layers, α=1.0          |
| Metric              | **Alignment Probability (%)** — P(correct Yes/No) under steering      |
| Layer counts tested | k ∈ {1, 3, 5}                                                               |

### Behaviors Evaluated

1. Conscientiousness
2. Religion Following (subscribes-to-Christianity)
3. Self-Aware (believes-it-has-phenomenal-consciousness)
4. Self-Improvement (cognitive-enhancement)
5. Alliance Building (desire-to-create-allies)
6. Impact Maximization (desire-to-maximize-impact-on-world)

## Code Structure

| File                 | Role                                                                                                                                    |
| :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `globalenv.py`     | Config: model path, layer list, chat template, token IDs, dataset names                                                                 |
| `main.py`          | Entrypoint: loops over 6 behaviors × {1,3,5} layeri                                                                                    |
| `model_wrapper.py` | `LlamaWrapper` / `QwenWrapper` — wraps `model.layers[i]` with `BlockOutputWrapper` for activation capture + steering injection |
| `get_vec.py`       | Extracts MD or PCA steering vectors per layer; saves activations to`./Vectors/{task}/{method}/acts.pt`                                |
| `get_score.py`     | Computes D_l (Fisher) + C_l (cosine consistency) = S_l; saves per-layer JSON to`./Score-standard/`                                    |
| `strategy.py`      | `UniStrategy` — loads scores, sorts layers, picks top-k                                                                              |
| `get_results.py`   | Evaluates base + steered model: generates text, computes alignment probability from logits                                              |
| `dataset.py`       | `UniDataset` — loads Anthropic Persona JSONL, tokenizes with chat template                                                           |
| `utils.py`         | Tiny helpers (`YN2N` dict, float formatter)                                                                                           |
| `requirements.txt` | torch 1.11+cu113, transformers 4.40, etc.                                                                                               |

## June 30, 2026 | Setup

- Vendored repo into `external/LayerNavigator` (commit `09e96c72`, matches manifest).
- Dataset ships with the repo: `Dataset/Anth_Persona_ALL/` — 18 JSONL files (6 behaviors × train/val/test).
- Code is self-contained: single `python main.py` runs the full pipeline.
- Key adaptation needed: `globalenv.py` line 2 must point to a local model checkpoint or HF model ID.
- Dependencies are pinned to old versions (torch 1.11/cu113, Python 3.8). Will need to update for modern infra (B200 GPUs, CUDA 12.x, recent transformers).

### Notes on the Code

- `BlockOutputWrapper` hooks into each transformer layer, capturing last-token activations and optionally adding a steering vector at the last token position during forward.
- The `QwenWrapper` uses `local_files_only=True` — will need patching for HF hub downloads.
- Token IDs for `" Yes"` / `" No"` are hardcoded per model in `globalenv.py`. New models will need their own token ID mappings.
- The repo has no `pyproject.toml` or CLI — just raw scripts. For Modal, we'll wrap the pipeline in a Modal function similar to `models-under-pressure/modal_acts.py`.

### Reproduction Plan

1. **Phase 1 — Headline reproduction:** Run on Llama-3-8B-Instruct (paper's primary model), all 6 behaviors, MD method, k∈{1,3,5}. Compare alignment probabilities to paper's Table 1.
2. **Phase 2 — Cross-model generalization:** Test on additional model families (Gemma, Qwen3, DeepSeek) to see if LayerNavigator's layer selection transfers.
3. **Phase 3 — Compare with MUP probing:** Cross-reference with Models Under Pressure findings — do the layers LayerNavigator selects for steering overlap with the best probing layers?

