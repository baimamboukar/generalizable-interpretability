# Research Data

## Saved Collections

| Location | Contents |
| :--- | :--- |
| `interp_corpus.json` | Historical paper-search corpus |
| `interp_initial_selection.json` and `interp_final_selection.json` | Earlier paper selections |
| `probe_replication_queue.csv` | Study queue |
| `vendored_repos.json` | Upstream repository locations and revisions |
| `screening/papers.csv` | Final screening table with paper links, code availability, decisions and evidence levels |
| `screening/datasets.csv` | Dataset variants, response sources and label definitions |
| `screening/dataset_overlap.csv` | Exact overlap findings and separately labelled dataset-lineage relationships |
| `protocols/` | Historical experiment specifications, hypotheses and study registry |
| `results/<study>/` | Published targets, replication measurements and model-extension results |
| `papers-base/` | Downloaded conference metadata, excluded from Git |

## Result Records

Published targets and measured results remain separate. Retain model identifiers and revisions, dataset partitions, response and label provenance, feature sites, fitting settings, random seeds and metric definitions with each experiment. Prediction records should retain row identifiers, labels, scores and thresholds when available.

Nested experiment directories, including `results/model-under-pressure/confirmatory/`, contain outputs from a specific evaluation. Their manifests and configuration fields identify how the measurements were produced. They form part of the experimental record. Large activation caches and temporary execution files belong in local artifact storage.

`results/paper-artifacts/` and `results/paper-synthesis/` contain derived cross-study tables for analysis and plotting. Historical manifests retain original file locations and hashes; those records describe the original computation, not the current directory layout.

The specifications in `protocols/` record an earlier analysis plan. Their original freeze dates and model-status fields remain unchanged. They do not constitute a new preregistration for the proposed benchmark. The historical registry includes a later additional study beyond the original protocol corpus.

## Screening Scope

The final screen, dated 29 September 2026, contains 75 detailed records and 174 paper–dataset usage records. It identifies 27 candidates, 23 holds, 19 background references and six exclusions. Review depth varies from abstract inspection to paper/repository inspection and historical local reproduction evidence. Candidate status does not establish executability or benchmark admission.

The overlap table contains 14 relationships. Three exact checks found 527 shared statements between PRISM True-False and Neural Chameleons RepE facts, 543 between PRISM LogicStruct and RepE, and 480 between the two PRISM collections. Normalization removed surrounding whitespace only. Other rows describe source lineage or shared dataset families, not verified identical examples. Stored-label conflicts remain unadjudicated; shared data alone do not establish train/test leakage.

Screening source notes, planning drafts and presentation files are archived outside the repository. The three CSV files retain the final screening results.
