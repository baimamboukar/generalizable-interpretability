
# Monitoring Latent World States with Propositional Probes

The paper introduces compositional probes that decode propositions from
Tulu-2-13B activations. Layer-20 domain probes identify names, countries,
occupations, and foods; a Hessian-derived binding subspace at layer 15 binds
the detected values into propositions.

## Baseline target

Reproduce the standard columns of Table 1 using the released probes and
Hessian: `synth`, paraphrased English `para`, and Spanish `trans`, evaluated
with exact match and Jaccard. Paper targets are frozen in
`data/results/propositional-probes/baseline_targets.csv`.

## Release audit

| Item                              | State                                                                   |
| :-------------------------------- | :---------------------------------------------------------------------- |
| Tulu-2-13B                        | Released code loads`allenai/tulu-2-13b`; revision is not pinned       |
| Domain probes                     | Released for Tulu and Llama                                             |
| Hessian binding subspace          | Released for Tulu and Llama                                             |
| Proposition labels and`synth`   | Deterministically regenerable from the released template and prompt IDs |
| Exact`para` and `trans` texts | Not released                                                            |
| Evaluation configs                | Generated interactively by a notebook; not committed                    |
| Environment                       | Torch 2.1.0, Transformers 4.35.2, custom bundled TransformerLens        |
| Paper compute                     | At most four A100 80GB; Hessian about five hours                        |

## Current verdict

**Closed: exact baseline not reproducible from the public release.** We checked
all remote refs, the complete Git history, the authors' public repositories,
the arXiv source bundle, Hugging Face, and the official Drive archive. The
labels can be regenerated exactly, but the GPT-3.5-generated `para` and
`trans` texts that determine the reported scores are unavailable. A new
generation would be a different evaluation set, so no GPU run was launched.
