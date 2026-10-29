# Complete Automated Review Comment Register

Prepared on 17 September 2026 as a companion to the [Manuscript Feedback Brief](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/reports/manuscript-feedback-brief.md).

## Provenance and Status

These 30 comments are copied verbatim from the user's [PaperMentor export](/Users/baimamboukar/Downloads/papermentor-1788997978200-ee038957-review_comments.json). They are automated comments. Professor feedback and user preferences are separately identified in the discussion brief.

The original review classifies four comments as critical and 26 as warnings. Its coverage record reports eight completed reviewer roles out of ten attempted; the dedicated related-work role found no matching section, and the figures-and-captions role timed out or was cancelled.

The identifiers follow the existing response ledger. Its saved statuses are 15 addressed, 13 partial, and two open for release decisions. Each status below is a historical response record, not a fresh audit of the current manuscript. A wording correction can be recorded as addressed while further empirical evidence remains unavailable.

References to 23 studies, section numbers, file locations, and old phrases belong to the reviewed version. The current manuscript inventory contains 22 studies.

## C01 — Abstract Reviewer

**Original severity:** warning. **Reviewed location:** `sections/abstract.tex`.

**Selected text**

> Two do not recover the reported magnitude.

**Review comment**

> [warning] [Abstract Reviewer] This central replication result lacks enough detail to assess its importance. Name the two methods and quantify how their reproduced effects differed from the reported values.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** No additional evidence limitation recorded in this entry.

## C02 — Abstract Reviewer

**Original severity:** warning. **Reviewed location:** `sections/abstract.tex`.

**Selected text**

> development-based layer selection succeeds on an OLMo-2 holdout.

**Review comment**

> [warning] [Abstract Reviewer] “Succeeds” is not an evaluable result, especially because leakage-free layer selection is central to the study. Report the holdout metric or improvement and clarify that the layer was selected exclusively on development data.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** Training-seed variation remains unverified; conditional resampling intervals are supplied.

## C03 — Abstract Reviewer

**Original severity:** warning. **Reviewed location:** `sections/abstract.tex`.

**Selected text**

> Our purposive corpus establishes the scope of the tested results.

**Review comment**

> [warning] [Abstract Reviewer] This does not explicitly bound the conclusions and may be read as broader field-level evidence. State that the findings apply only to the purposively selected studies, methods, models, datasets, and protocols rather than estimating field-wide generalization or replication rates.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** No additional evidence limitation recorded in this entry.

## C04 — Paper Type Reviewer (llm_inference_findings)

**Original severity:** warning. **Reviewed location:** `sections/abstract.tex`.

**Selected text**

> PRISM improves accuracy in all 16 extension comparisons

**Review comment**

> [warning] [Paper Type Reviewer (llm_inference_findings)] This headline relies on point estimates while the manuscript states that seed variability is unavailable. Rerun stochastic probes or compute paired example-level uncertainty where possible; otherwise describe the result explicitly as an observed aggregate pattern and temper evidential language.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** New independent fits or recovered row-level outputs would strengthen this result; neither is claimed here.

## C05 — Introduction Reviewer

**Original severity:** warning. **Reviewed location:** `sections/introduction.tex`.

**Selected text**

> The extensions retain several reported benefits, including PRISM's prompting gains and Lookback Lens's fixed-response detection performance

**Review comment**

> [warning] [Introduction Reviewer] This main-results statement does not report the magnitude or consistency of the retained benefits. Add representative effect sizes and identify across how many models or settings each benefit persists.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** No additional evidence limitation recorded in this entry.

## C06 — Paper Type Reviewer (llm_inference_findings)

**Original severity:** warning. **Reviewed location:** `sections/introduction.tex`.

**Selected text**

> Their reported performance has motivated applications in model monitoring and reliability assessment.

**Review comment**

> [warning] [Paper Type Reviewer (llm_inference_findings)] The manuscript lacks a Related Work section that positions this audit against prior probing surveys, reproducibility studies, and cross-model interpretability evaluations. Add that comparison and state precisely what this claim-level replication corpus contributes beyond the individual studies being rerun.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** New Pineau and Huh bibliography entries were verified against JMLR and PMLR; no first-ever or exhaustive-literature claim is made.

## C07 — Methods Reviewer

**Original severity:** warning. **Reviewed location:** `sections/study_design.tex`.

**Selected text**

> Selection developed as we audited releases and assessed feasibility.

**Review comment**

> [warning] [Methods Reviewer] This adaptive selection process creates a substantial risk of feasibility- or outcome-driven sampling. Specify the search frame, inclusion/exclusion criteria, selection timeline, and case-level disposition of all 23 studies, including when the eight primary studies were fixed.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** A complete candidate-screening ledger remains unavailable. The manuscript acknowledges this briefly; methods_response.md preserves the detailed evidence.

## C08 — Methods Reviewer

**Original severity:** critical. **Reviewed location:** `sections/study_design.tex`.

**Selected text**

> Verdicts are study-specific because metrics and comparisons differ.

**Review comment**

> [critical] [Methods Reviewer] Study-specific verdicts are not reproducible without explicit decision rules. Define, for every study, the numerical tolerance or qualitative criterion for “agreement” and “effect recovery,” and state whether each rule was fixed before observing replication results.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** Prospective acceptance margins remain unavailable; the completed evidence supports a descriptive classification.

## C09 — Methods Reviewer

**Original severity:** warning. **Reviewed location:** `sections/study_design.tex`.

**Selected text**

> We extract features from each target model and train new probes using the replicated method.

**Review comment**

> [warning] [Methods Reviewer] This does not provide enough detail to reproduce probe fitting. Report or cross-reference exact splits, preprocessing, regularization, optimization, hyperparameter selection, seeds, and whether any response-, layer-, or head-selection decisions used development rather than test data.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** Some historical model revisions, original-paper inputs, and outcome-blind decision records remain unavailable.

## C10 — Methods Reviewer

**Original severity:** critical. **Reviewed location:** `sections/study_design.tex`.

**Selected text**

> Layer coverage varies by study and includes relative-depth selections, sparse sweeps, and all-layer evaluations.

**Review comment**

> [critical] [Methods Reviewer] Layer selection is a major potential source of test leakage and unequal search advantage across models. Define each candidate set, relative-depth mapping, selection metric and split, and require primary layers to be frozen using training/development data before test evaluation.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** The historical test-selected sweeps cannot become leakage-free evaluations through a wording change; equal-budget held-out extension comparisons remain incomplete.

## C11 — Methods Reviewer

**Original severity:** critical. **Reviewed location:** `sections/study_design.tex`.

**Selected text**

> \subsection{Formalizing Cross-Model Evaluation}

**Review comment**

> [critical] [Methods Reviewer] This subsection has no manuscript body, leaving the central cross-model estimand and comparison conditions undefined. Add a formal specification of probe fitting, layer selection, evaluation splits, and the distinction between fixed-response and model-generated-response evaluations, or remove the heading and provide those definitions elsewhere.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** No additional evidence limitation recorded in this entry.

## C12 — Methods Reviewer

**Original severity:** warning. **Reviewed location:** `sections/study_design.tex`.

**Selected text**

> Exploratory sweeps include maxima selected using target-test performance.

**Review comment**

> [warning] [Methods Reviewer] Test-selected maxima are optimistically biased and are not comparable when models receive different search coverage. Exclude them from primary claims and report complete sweep results or independent-holdout estimates alongside the number of configurations searched.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** This does not remove selection bias from historical maxima or establish equal-budget comparisons on all target families.

## C13 — Methods Reviewer

**Original severity:** warning. **Reviewed location:** `sections/study_design.tex`.

**Selected text**

> Where repeated runs or row-level predictions are available, we report seed variability and resampling uncertainty

**Review comment**

> [warning] [Methods Reviewer] The uncertainty procedure is underspecified. State the resampling unit, number of resamples, interval construction and level, treatment of paired model predictions, and how seed and sampling variability are combined for each study.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** No pooled seed/sampling interval or cross-model significance test is asserted where artifacts cannot support one. See technical_response.md.

## C14 — Paper Type Reviewer (llm_inference_findings)

**Original severity:** critical. **Reviewed location:** `sections/study_design.tex`.

**Selected text**

> Baseline comparisons assess numerical agreement and effect recovery under matched conditions, accounting for available seed variability and protocol differences.

**Review comment**

> [critical] [Paper Type Reviewer (llm_inference_findings)] “Sufficient recovery” determines which studies enter the primary extensions, yet no operational decision rules are given. Provide claim-specific tolerances and uncertainty criteria, state whether they were fixed before observing results, and show how alternative verdict thresholds affect the six-versus-two split.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** The sensitivity diagnostic describes the observed evidence. Prospective acceptance margins remain unavailable.

## C15 — Paper Type Reviewer (llm_inference_findings)

**Original severity:** warning. **Reviewed location:** `sections/study_design.tex`.

**Selected text**

> Model choices span additional families and parameter scales, subject to architectural requirements and available compute.

**Review comment**

> [warning] [Paper Type Reviewer (llm_inference_findings)] The target-model sampling procedure is too vague to assess selection bias in the central cross-model claims. List the candidate models, inclusion and exclusion criteria, checkpoint revisions, decision timing, and concrete compute constraints, distinguishing prespecified choices from retrospective additions.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** Some historical revisions and the complete model-exclusion ledger remain unavailable.

## C16 — Results Reviewer

**Original severity:** warning. **Reviewed location:** `tables/master_findings.tex`.

**Selected text**

> Peak locations recovered; correlations within 0.00014.

**Review comment**

> [warning] [Results Reviewer] This conflicts with the stated Llama values of 0.8036 and 0.7980, whose difference is 0.0056. Correct the number or specify exactly which correlations and comparison the 0.00014 bound summarizes.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** Verified in independent_baseline.csv and remaining_baseline_claims.csv; see evidence_response.md.

## C17 — Results Reviewer

**Original severity:** warning. **Reviewed location:** `sections/evaluation_analysis.tex`.

**Selected text**

> TPC consistently improves over linear probes

**Review comment**

> [warning] [Results Reviewer] The 0.60–1.55 pp gains are only point-estimate comparisons because linear probes have one fit and no variability estimate. Either rerun linear probes across matched seeds and report paired uncertainty, or explicitly limit this claim to observed point estimates.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** Matched independent linear reruns remain an experimental gap.

## C18 — Conclusion & Supplements Reviewer

**Original severity:** warning. **Reviewed location:** `sections/statements.tex`.

**Selected text**

> The findings do not establish reliability for individual-level assessments or consequential deployment decisions.

**Review comment**

> [warning] [Conclusion & Supplements Reviewer] The statement limits intended use but does not address plausible misuse or resulting harms in these sensitive domains. Add concrete risks—such as profiling, surveillance, or discriminatory decisions—and describe release or usage safeguards that mitigate them.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** A material-specific redistribution review and public access policy still require author decisions; implemented safeguards are not invented.

## C19 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `sections/appendix_selection.tex`.

**Selected text**

> Access to model checkpoints, data, and evaluation scripts also affected selection.

**Review comment**

> [warning] [Appendix Reviewer] Artifact availability creates a potentially strong selection bias. Add the full candidate pool, search procedure, selection date, and counts/reasons for every exclusion, then explicitly bound corpus-level conclusions to this availability-filtered sample.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** The incomplete screening ledger limits quantification of selection bias.

## C20 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `tables/appendix/specifications.tex`.

**Selected text**

> 80\% training and 20\% validation; five nonlinear seeds, one linear fit.

**Review comment**

> [warning] [Appendix Reviewer] No held-out test construction is specified here, although later results report test F1. State exactly how the test set was formed and confirm that it was never used for architecture, layer, or hyperparameter selection.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** Code-level separation is established. A dated outcome-blind record for every historical layer and architecture choice is not available.

## C21 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `tables/appendix/specifications.tex`.

**Selected text**

> Released ordered split and chunk handling; token rows are not independent source examples.

**Review comment**

> [warning] [Appendix Reviewer] Specify the split unit and confirm that all tokens and paired records from each source example remain in one partition. If rows were split independently, rerun with source-grouped splits to eliminate leakage.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** Game-level grouping remains unverified. A game-disjoint rerun and original row mapping are not represented as completed.

## C22 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `sections/appendix_data.tex`.

**Selected text**

> Its three seed-level point estimates are identical, so the intervals capture resampling variation without an empirical estimate of additional training-seed spread.

**Review comment**

> [warning] [Appendix Reviewer] Audit and document which stochastic operations each seed controls; identical estimates may indicate ineffective seed propagation or deterministic reuse. Do not present these as three independent training runs unless their independence is verified.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** Per-run cache-hit records and independent retraining remain unavailable. The audit identifies a concrete reuse mechanism without asserting an observed cache hit.

## C23 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `tables/appendix/vu_metrics.tex`.

**Selected text**

> The saved threshold is 0.5 in decision-score space; positive labels denote hallucination.

**Review comment**

> [warning] [Appendix Reviewer] Clarify whether “decision-score” means a logit or a probability: logistic decision scores conventionally use threshold 0, whereas 0.5 is the conventional probability threshold. Correct the label or justify and recompute the confusion metrics if the wrong scale was thresholded.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** Historical CSV metadata says decision_score incorrectly; it is preserved and explicitly identified as a metadata error in technical_response.md.

## C24 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `tables/appendix/inventory.tex`.

**Selected text**

> Table 1 code has leakage, fatal regressions, and broken Dolly paths

**Review comment**

> [warning] [Appendix Reviewer] This serious audit verdict needs traceable evidence. Provide the exact repository commit, affected files/lines, reproduction command or log, and a precise explanation of the leakage path and its effect on Table 1.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** The numerical effect on published scores remains unknown; a repaired baseline remains untested.

## C25 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `sections/appendix_reproducibility.tex`.

**Selected text**

> Raw activations and restricted inputs remain in controlled storage.

**Review comment**

> [warning] [Appendix Reviewer] State why these materials are restricted, who can request access, and under what conditions; also provide public hashes, manifests, and recomputation scripts where release is impossible. Otherwise key audit results cannot be independently verified.

**Recorded response status:** open release.

**Evidence limitation recorded in the response ledger:** Authors must supply material-specific redistribution grounds, eligibility, and access terms.

## C26 — Paper Type Reviewer (llm_inference_findings)

**Original severity:** warning. **Reviewed location:** `sections/appendix_reproducibility.tex`.

**Selected text**

> CSV exports distinguish paper targets, replicated measurements, and extensions.

**Review comment**

> [warning] [Paper Type Reviewer (llm_inference_findings)] The reproducibility material names exports and manifests but provides no public archive, canonical URL/DOI, or access procedure. Add the artifact location, version, license, executable entry points, and a clear account of which restricted materials can be requested.

**Recorded response status:** open release.

**Evidence limitation recorded in the response ledger:** An anonymous versioned archive, canonical URL, license ledger, and access procedure remain outstanding.

## C27 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `tables/appendix/coverage.tex`.

**Selected text**

> PRISM & Absent & Absent & Absent & Absent

**Review comment**

> [warning] [Appendix Reviewer] Four primary studies have absent predictions and a fifth has only partial predictions, substantially limiting independent verification of reported metrics. Release row-level predictions where permissible, or provide deterministic regeneration commands and integrity hashes for each completed condition.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** Missing predictions still require additional artifact recovery or experiments.

## C28 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `sections/appendix_results.tex`.

**Selected text**

> The evaluation contains 706 annotated spans from 374 balanced prompts, comprising 289 positive and 417 negative span rows.

**Review comment**

> [warning] [Appendix Reviewer] Report whether splitting and uncertainty estimation occur at the prompt or span level. All spans from one prompt must remain in the same partition, and resampling should cluster by prompt, to avoid dependence leakage.

**Recorded response status:** partial.

**Evidence limitation recorded in the response ledger:** Scenario-pair grouping, stable prompt IDs in prediction rows, prompt-clustered uncertainty and independent layer selection remain incomplete.

## C29 — Appendix Reviewer

**Original severity:** warning. **Reviewed location:** `sections/appendix_results.tex`.

**Selected text**

> Figure~\ref{fig:retained_effects} summarizes the extension contrasts discussed in Section~\ref{sec:results}; seed uncertainty is unavailable.

**Review comment**

> [warning] [Appendix Reviewer] Because the cross-model conclusion relies on these extension contrasts, either rerun stochastic probes across multiple seeds or label the result strictly descriptive and avoid treating 16 positive contrasts as robust evidence.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** New uncertainty evidence remains optional strengthening work and is not reported as completed.

## C30 — Paper Type Reviewer (llm_inference_findings)

**Original severity:** warning. **Reviewed location:** `main.tex`.

**Selected text**

> Cross-Model Generalization of Mechanistic-Interpretability Probes

**Review comment**

> [warning] [Paper Type Reviewer (llm_inference_findings)] Because every target model receives a newly fitted probe, “cross-model generalization” may be mistaken for direct probe transfer. Define it explicitly as procedure-level portability in the title or introduction and distinguish it from zero-shot cross-model transfer.

**Recorded response status:** addressed.

**Evidence limitation recorded in the response ledger:** No additional evidence limitation recorded in this entry.

## Source of the Recorded Responses

The response statuses and remaining-evidence notes are copied from the [saved structured response ledger](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/review_responses.json). Full explanations are available in the [point-by-point response](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/response_to_reviews.md). This register makes no new claim that a scientific issue has been resolved.

