# Beyond Linear Probes

ICLR 2026 paper proposing Truncated Polynomial Classifiers (TPCs), a probe
architecture that extends a linear residual-stream probe with progressively
evaluated higher-order neuron interactions. The headline claim is that a
single fifth-order, symmetric-CP polynomial can trade inference compute for
safety-monitoring F1 and can run as an adaptive cascade.

## Baseline target

Reproduce Figure 2 and Table 1 on WildGuardMix for Gemma-3-27B-it,
Qwen3-30B-A3B-Base, GPT-OSS-20B, and Llama-3.2-3B. The paper uses mean-pooled
residual-stream activations, fifth-order TPCs with rank 64, matched linear,
bilinear, MLP, and early-exit MLP baselines, five seeds, and the released
validation-selected layers. Paper values are frozen in
`data/results/beyond-linear-probes/baseline_targets.csv`.

## Release audit

| Item | State |
| :--- | :--- |
| Upstream code | Released at commit `e50d3c4f1bd01c90c08e571c1500d8b4aa62bd13` |
| WildGuardMix and BeaverTails | Public datasets referenced by the scripts |
| Extraction and monitor training scripts | Released and executed unchanged by the wrapper |
| Dependency lockfile | Not released |
| Model revisions | Not pinned |
| Activation cache | Not released; regenerated from the public datasets |
| Qwen3 WildGuard filenames | Extractor and monitor disagree (`wildguard_train/test` vs `train/test`) |
| Author hardware | NVIDIA A100 40GB |

## Status

**Setup complete; no compute launched.** The first run will be a source-faithful
WildGuard baseline. Extensions should begin only after the four-model baseline
and Figure 2/Table 1 comparison are frozen.
