# Repository inventory

Updated: 2026-07-24

## Active structure

| Path | Purpose | State |
| :--- | :--- | :--- |
| `replications/models-under-pressure/` | Modal extraction and result synchronization | Completed; preserve |
| `replications/among-us/` | Phi-4 baseline and Llama/Gemma/Qwen extensions | Completed; preserve |
| `replications/neural-chameleons/` | Upstream-faithful release audit | Paused; baseline prerequisites missing |
| `reports/` | Concise paper-versus-replication records | Active |
| `data/results/` | Plotting-ready and raw result exports | Active |
| `data/probe_replication_queue.csv` | Machine-readable next-paper queue | Active |
| `external/` | Pinned upstream repositories | Ignored; do not modify |
| `outputs/` | Presentation artifacts | Generated but durable |

## Historical material

| Path | Reason retained |
| :--- | :--- |
| `replications/layer-navigator/` and `reports/layer-navigator.md` | Paused steering study; outside current probe-only scope |
| `scripts/build_interp_corpus.py`, `find_extensions.py`, `finalize_extensions.py`, `select_interp_targets.py`, `vendor_repos.py` | Earlier venue-mining and vendoring pipeline |
| `data/interp_*.json`, `data/vendored_repos.json` | Inputs and manifest for the earlier literature search |
| `docs/target.papers.shortlist___generalizable.interp.pdf` and matching figure | Historical steering-inclusive shortlist |
| `papers-base/` | Ignored venue corpus used by the historical search |

## Replication state

| Study | Baseline | Extensions | Durable results |
| :--- | :--- | :--- | :--- |
| Models Under Pressure | Reproduced | Qwen, DeepSeek and GLM; multiple layers and probe architectures | Report, figures, raw JSONL exports |
| Among Us | Not reproduced | Llama-8B, Gemma-27B and Qwen-32B; three layers each | Three plotting-ready CSV files |
| Neural Chameleons | Released artifact verified; independent baseline not run | None | Paper targets and released-demo CSV files |
| LayerNavigator | Not run | None | Setup only; paused |

## Cleanup policy

- Keep official repositories unchanged under `external/`.
- Keep code provider-independent where possible; infrastructure wrappers remain
  separate from experiment code.
- Store paper, released-checkpoint, replication, and extension values separately.
- Use long-format CSVs for results and concise Markdown reports for conclusions.
- Do not commit caches, temporary PDF renders, inspection traces, model weights,
  activations, or provider-specific runtime state.

## Remaining consolidation

1. Convert the Models Under Pressure raw JSONL exports into one long-format CSV.
2. Move the verbose chronological section of its report to an archive after the
   CSV has preserved every result.
3. Add the missing GLM Among Us extension only if its existing result artifact
   can be recovered without new GPU work.
4. Start PRISM in a new replication folder after its official release is pinned.
