# ICR Probe

ICR Probe detects hallucinations from how residual-stream updates evolve across
layers. For each answer token and layer, it compares the hidden-state update
direction with attention over the top 20 tokens using Jensen-Shannon
divergence. Token-pooled scores from every layer feed a small MLP.

## Baseline target

Reproduce Table 1 AUROC on HaluEval QA, SQuAD, HotpotQA, and TriviaQA
`rc.nocontext` for Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, and Gemma-2-9B-it.
Paper values are frozen in
`data/results/icr-probe/baseline_targets.csv`.

## Release audit

| Item | State |
| :--- | :--- |
| ICR score implementation | Released |
| Probe architecture | Released |
| Main models and datasets | Named, but model revisions and dataset revisions are not pinned |
| Generated answers and labels | Not released |
| Generation and judging protocol | Not released |
| Forward-pass caching pipeline | Not released |
| Baseline runner and checkpoints | Not released |
| Environment lockfile | Not released |
| Paper training setup | 50 epochs, batch 32, Adam at 5e-4 |
| Released config defaults | 100 epochs, batch 16, Adam at 1e-3 |

## Current verdict

**Baseline setup audited; exact reproduction is blocked by the release.** The
repository provides reusable method components, not the end-to-end experiment.
Reconstructing the missing generation, labeling, and caching stages would be an
independent implementation rather than an upstream-faithful replication.
