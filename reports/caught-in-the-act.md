# Caught in the Act: a mechanistic approach to detecting deception

## Setup status

**Status:** setup-ready (queued for first-pass audit + upstream release discovery).

## Paper context

Title: *Caught in the Act: a mechanistic approach to detecting deception* (arXiv:2508.19505, 2025)

The paper proposes linear probing of residual stream activations to distinguish deceptive vs non-deceptive LLM arguments, with per-layer diagnostics and iterative nullspace projection as a reliability check.

## Initial audit notes

- Core methods listed in the paper: linear probe baselines, LLM-as-judge black-box baseline, and INLP iterative null-space projection.
- Main model set in paper text: Qwen 2.5 (1.5B, 3B, 7B, 14B) and DeepSeek-r1 (1.5B, 7B), plus baseline model coverage notes.
- Current repo status: no upstream code/release link confirmed yet from the public page.
- Baseline run plan (initial): OpenRouter-backed judges + probe models `Qwen2.5-3B` and `DeepSeek-R1 1.5B`; optional extension to `DeepSeek-R1 7B`.
- Replication decision: hold off on execution until the release artifacts are verified and a clean runbook is available.

## Baseline target (planned)

| Target model | Probe target | Metric | Planned status |
| --- | --- | --- | --- |
| Qwen2.5-3B + DeepSeek-R1 1.5B (first wave) | Final-token residual-stream linear probe | AUROC | Not started |
| GPT-4o / GPT-4o-mini | Black-box deception judge baseline | Accuracy | Not started |

## Next steps

1. Locate and pin the upstream release (code + generation scripts + dependency lockfile).
2. Reconstruct synthetic argument generation + instruction filtering pipeline exactly as in the paper.
3. Run paper-faithful baseline replication.
4. Add extension study plan: transfer/depth behavior and INLP-based direction-count diagnostics.
