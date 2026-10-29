# Generalizable Interpretability

This project studies the reproducibility and generalization of activation-probing methods in language models. Each study distinguishes the original-model comparison from extensions using newly fitted probes.

## Repository Structure

| Directory | Contents | Git Policy |
| :--- | :--- | :--- |
| `data/` | Paper inventories, final screening tables, experiment specifications and numerical results | Tracked, except raw conference downloads |
| `figures/` | Figures derived from recorded results | Tracked |
| `scripts/` | Corpus preparation, result aggregation, metric calculation and plotting | Tracked |
| `reports/` | Formal study reports | Tracked |
| `replications/` | Local experiment wrappers, environments and execution helpers | Code excluded |
| `external/` | Original author repositories | Local only |
| `infra/` | Compute-provider and artifact-storage helpers | Local only |
| `artifacts/` | Local run outputs, caches and larger intermediate files | Local only |

The [data guide](data/README.md) describes the saved collections. Experiment settings and provenance remain with the data because they are required to interpret or regenerate results. Manuscripts, submissions, slides and planning drafts are maintained outside this repository.

## Study Reports

The research corpus contains 22 studies. Eight selected findings form the primary comparison, four studies provide complementary experiments, and ten provide artifact or execution audits. Six primary comparisons support the selected finding with qualifications; two original-model discrepancies remain unresolved. LayerNavigator is retained separately as an earlier steering study.

| Study | Evidence | Report |
| :--- | :--- | :--- |
| Detecting High-Stakes Interactions with Activation Probes | Qualified baseline and exploratory model comparisons | [Report](reports/model-under-pressure.md) |
| Among Us: A Sandbox for Measuring and Detecting Agentic Deception | Original-model discrepancy; exploratory extensions | [Report](reports/among-us.md) |
| PRISM: Prompt-Guided Internal States for Hallucination Detection | Prompting benefit recovered with numerical qualifications | [Report](reports/prism.md) |
| Sycophancy Hides Linearly in the Attention Heads | Original-model accuracy discrepancy unresolved | [Report](reports/sycophancy.md) |
| Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization | Reduced explicit-identity comparison and extensions | [Report](reports/implicit-personalization.md) |
| Monitoring Latent World States in Language Models with Propositional Probes | Missing evaluation texts | [Report](reports/propositional-probes.md) |
| ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs | Incomplete experimental release | [Report](reports/icr-probe.md) |
| LayerNavigator: Finding Promising Intervention Layers for Efficient Activation Steering | Historical steering study outside the 22-study corpus | [Report](reports/layer-navigator.md) |
| Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Activation Monitors | Released demonstration verified; full baseline blocked | [Report](reports/neural-chameleons.md) |
| Do LLMs “Know” Internally When They Follow Instructions? | Qualified baseline and response controls | [Report](reports/internal-instruction-following.md) |
| Beyond Linear Probes: Dynamic Safety Monitoring for Language Models | Selected baseline and model extensions | [Report](reports/beyond-linear-probes.md) |
| Lookback Lens: Detecting Contextual Hallucinations in Large Language Models Using Only Attention Maps | Released-feature baseline and fixed-response extensions | [Report](reports/lookback-lens.md) |
| Linear Representations of Political Perspective Emerge in Large Language Models | Original-model comparisons and exploratory extensions | [Report](reports/political-perspective.md) |
| When Truthful Representations Flip Under Deceptive Instructions | Paper–code discrepancies | [Report](reports/truthful-representation-flip.md) |
| Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs | Initial execution blocked by environment mismatch | [Report](reports/disentangle-sycophancy.md) |
| Caught in the Act: A Mechanistic Approach to Detecting Deception | Matching implementation unverified | [Report](reports/caught-in-the-act.md) |
| LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals | Released generation exceeds tested memory budgets | [Report](reports/reasoning-trajectory.md) |
| The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness | Selected Qwen3 accuracy comparison recovered | [Report](reports/test-awareness-steering.md) |
| Probing for Arithmetic Errors in Language Models | Missing artifacts and disabled baseline components | [Report](reports/arithmetic-error-probing.md) |
| Thought Calibration: Efficient and Confident Test-Time Scaling | Incomplete retraining metadata and nonlinear implementations | [Report](reports/thought-calibration.md) |
| ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability | Evaluation and protocol blockers | [Report](reports/redeep.md) |
| LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations | Qualified exact-answer baseline and response-controlled extensions | [Report](reports/llms-know.md) |
| Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations | Regenerated Table 2 baseline with mixed numerical fidelity | [Report](reports/verbal-uncertainty.md) |

The [cross-study analysis](reports/generalization.md) summarizes the findings and their limitations. The [training-data audit](reports/mup-training-data-audit.md) documents the high-stakes dataset provenance.

## Analysis Code

`scripts/build_interp_corpus.py` reads local conference metadata from `data/papers-base/`. Selection and repository-discovery scripts operate on the resulting inventories. Result aggregation and plotting scripts read saved study outputs from `data/results/` and write figures to `figures/`.

`scripts/metrics.py` supplies shared metric calculations. Its numerical checks run with `python -m unittest scripts.test_metrics`. Individual experiments require local wrappers, upstream repositories and study-specific dependencies. The tracked reports and results do not constitute a self-contained release of every experiment.
