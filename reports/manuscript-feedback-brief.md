# Manuscript Feedback and Professor Discussion Brief

Prepared on 17 September 2026. This is a discussion and handoff document. Neither manuscript was edited while preparing it.

## 1. What We Need to Decide

The main decision is the paper's contribution. The professor's latest feedback proposes a reproduction and verification pipeline, followed by its empirical application and a substantial discussion of the findings. We need to agree on how prominently agent assistance should feature and what evidence would support that emphasis.

Three questions should lead the meeting:

1. Is the main contribution the empirical evaluation of probing methods, a reusable researcher-supervised verification workflow, or an evaluated agent-based system?
2. What single conclusion should a reader remember about the evaluated probing methods?
3. Which remaining evidence gaps must be closed before submission, and which can be handled through appropriately scoped claims?

The sections below consolidate the user's instructions, professor and mentor comments supplied in the conversation, the saved writing skill, and the 30 comments in the exported PaperMentor review. Repeated instructions are merged. New recommendations are identified as proposals. Unexported comments currently inside Overleaf are outside this collection.

## 2. Scientific Context to Preserve

### Project Purpose

The project examines whether selected, influential interpretability findings can be reproduced on their original models and recovered when the probing procedure is applied to additional model families. Probes are the main focus. Steering studies became eligible where their selected claims fit the research question and available artifacts permit a meaningful evaluation.

The original motivation was concern about conclusions drawn from narrow model coverage. The interpretation must follow the observed results. Successful extensions, unresolved baselines, and restricted evaluations all contribute to the study.

This project is separate from the lab's work on explaining Apollo deception-probe failures and developing better deception probes. It is also separate from parameter-space symmetry research. Each target model generally receives a newly fitted probe in this project; this fitting procedure needs to be explicit whenever the manuscript discusses cross-model generalization.

### Current Evidence Groups

The current manuscript inventory contains 22 studies:

| Evidence group | Current scope |
| --- | --- |
| Primary comparisons | Eight selected baseline findings: PRISM, Lookback Lens, Political Perspective, Beyond Linear Probes, Models Under Pressure, Instruction Following, Among Us, and Sycophancy Hides Linearly. |
| Complementary experiments | Four studies: Reading Between the Prompts, LLMs Know More Than They Show, Test Awareness, and Verbal Uncertainty. |
| Artifact or execution audits | Ten studies whose reported scope remains primarily an audit. |

The primary analysis records six qualified recoveries and two unresolved original-model discrepancies. Recovery refers to the selected finding and its documented conditions. Complete reproduction of every claim in each paper remains a separate question.

Earlier feedback refers to inventories of 18 or 23 studies. These are version-specific comments. The current inventory is the reference for manuscript counts. The user's approved presentation treats the included studies as one corpus and explains their selection and experimental scope without narrating the project's calendar.

### Interpretation Boundaries

- An artifact audit establishes what can be checked from a release. Missing inputs or an execution failure limit the available scientific conclusion.
- A baseline discrepancy concerns performance on an original paper model. Its cause can remain unresolved after source and metric checks.
- A cross-model extension assesses the procedure after target-model fitting. Its interpretation depends on the retained task, inputs, responses, labels, and selection procedure.
- A target-test maximum describes an exploratory search. Independent evaluation requires selecting configurations on separate data.
- Changes in generated responses can change labels, class balance, valid evaluation folds, and the prediction problem.
- Architecture comparisons require clear seed coverage and uncertainty reporting. The retained Beyond Linear Probes comparisons have unequal fitting coverage.
- The response-source controls change response content and success labels jointly. Their separate causal effects remain unresolved.
- AUROC summarizes ranking. Thresholded errors require their own evaluation, including threshold selection and the development and test distributions.
- The corpus supports conclusions about its evaluated cases. A field-wide success or failure rate would require a different sampling basis.

These distinctions should be expressed in direct, affirmative prose. Essential qualifications should remain visible near the claims they affect.

## 3. Professor and Mentor Feedback

The quotations in this section were supplied by the user. The interpretation column is a working reading for discussion.

### Contribution, Headline, and Structure

| ID | Feedback supplied | Working interpretation and clarification |
| --- | --- | --- |
| P01 | “the paper is missing a good headline / conclusion” | Agree on one empirical takeaway that can organize the abstract, results, and discussion. |
| P02 | “it should not be an effort to find the graphs that clearly show: ‘Works on paper-model, fails on others’” | Put matched comparisons and the overview early. Clarify whether the desired headline concerns baseline recovery, cross-model change, or the requirements for interpreting those changes. The two largest unresolved discrepancies concern original models. |
| P03 | “Put the first part on pipeline of reproduction/verification, best advocate for automated, agent-driven reproducibility methods” | Describe the actual verification workflow and agent-supported steps. Clarify the intended level of automation claim. |
| P04 | “as in it could be automated to reproduce/verify these work” | Separate the workflow demonstrated here from a prospective autonomous system evaluation. Ask what evaluation the professor expects for the latter. |
| P05 | “the flow should be we develop a pipleine and then use the pipeline to replicate these work and found most of them XXX” | Pipeline → application → findings → implications. The final wording of “most” needs a clear reference to the selected findings and their qualifications. |
| P06 | “discussion should have at least 3 paragraph of subtantive contents, not counting limitations or conclusions” | Develop the meaning of the observations, the requirements for credible comparisons, and the implications for future verification. |
| P07 | “remember numbers are less important than findings and trends” | Lead each results paragraph with its finding, then give the few measurements needed to support it. Preserve full numerical evidence in tables and the supplement. |

The earlier lab discussion considered a broad generalization-failure paper, continued replication, and a focused synthesis of recurring evaluation issues. The user proposed a targeted search for additional strong studies and an evidence-led synthesis. No arbitrary fixed number of additional papers should be treated as an agreed completion rule.

Johannes's Models Under Pressure comments also matter for the story. He asked for aggregate performance across datasets and broader layer coverage before interpreting an initially weak model as a pipeline failure. The 85% AUROC screen was a pragmatic decision in that investigation. Its scope should remain attached to that investigation.

### Abstract and Introduction

| ID | Feedback supplied | Writing implication |
| --- | --- | --- |
| P08 | The abstract “presupposes that everyone has read these papers.” | Describe prediction tasks and findings before using paper names as shorthand. |
| P09 | “it reads like you are calling them out” | Describe discrepancies and implementation uncertainty neutrally. Keep attribution factual and avoid implying negligence. |
| P10 | “difference between audit (22) and ‘attempt to reproduce’ (8) still not clear” | Define the activities and explain the scope of the primary, complementary, and audit groups. |
| P11 | The final sentence is important and should be “bigger/more prevalent.” | Carry the main takeaway through the introduction, discussion, and conclusion. |
| P12 | “‘interaction stakes’ is not a property of the behavior” | Define probes as predicting properties of inputs, outputs, or internal states as appropriate. Describe stakes as a property of the interaction or prompt. |

The user approved the following takeaway in the earlier abstract:

> Several probing benefits persist across model families. Our comparisons show how response generation, activation selection, and decision thresholds shape the evidence for generalization.

The meeting should clarify how this empirical conclusion should connect to the pipeline-led framing.

### Methods, Evidence, and Reader Accessibility

| ID | Feedback supplied | Writing implication |
| --- | --- | --- |
| P13 | Table 1 should explain the claims to readers unfamiliar with the papers. | State the selected task, what the original result demonstrated, and what our comparison establishes. |
| P14 | “The introduction of 2.3 is great!” | Preserve the accessible motivation for the formal evaluation setup when reorganizing sections. |
| P15 | “fitting procedure could be F, candidate set could be S” | Use intuitive notation, define every symbol, and connect equations to the actual procedure. |
| P16 | “is activation site a layer?” | Explain that sites can be layer residual states, MLP outputs, attention heads, or other study-specific features. |
| P17 | Table 2 leaves “exploratory” and “diagnostic” unclear. | Define target-test selection and baseline-limited diagnostics beside the table and in the methods. |
| P18 | For Among Us and Sycophancy, “is there any way to verify and communicate that your implementation is not flawed?” | Report the checks performed and their evidential scope. Ask what additional reference-output or evaluator-parity check is necessary for the chosen conclusion. |
| P19 | Read with the mindset of someone with some mechanistic-interpretability knowledge who has not read all eight papers. | Define tasks and terms locally. Each result should explain why the reader should care and what it supports. |

### Figures and Tables

| ID | Feedback supplied | Writing and presentation implication |
| --- | --- | --- |
| P20 | “why are some numbers in the middle of the bars?” | Use consistent value-label placement. The latest preference is above the bars with adequate headroom. |
| P21 | Avoid the large OLMo-2 figure that conveys too little for its size. | Reserve main-text space for the central argument. Keep useful detailed diagnostics in the supplement. |
| P22 | A main table with model rows, prior-work columns, green checks, and red crosses. | Provide a scannable model-level overview. The user subsequently approved a compact study-first table with baseline and extension model rosters and defined outcome symbols. |
| P23 | Figures 5 and 6 are too large; discussion is the most important part. | Rebalance space toward substantive interpretation while retaining necessary evidence and accessible figure references. |

## 4. Settled User Writing and Presentation Preferences

### Prose and Flow

1. Use clean, direct scientific English. State the substantive finding early in each paragraph.
2. Keep one main message per paragraph. Define terms before reusing them, and make transitions build on the preceding argument.
3. Follow the clarity and flow of the user's chosen exemplar, [Learning Hierarchical Features with Matryoshka Sparse Autoencoders](https://arxiv.org/html/2503.17547v1). The exemplar guides structure and tone; the manuscript's content remains its own.
4. Keep the introduction concise. Contributions should be simple statements without mini-headings and repeated explanations.
5. Avoid rhetorical questions, generic announcements, inflated novelty, and claims that exceed the retained evidence.
6. Avoid opening sentences with “Together.” Remove filler such as “Answering this question requires more than comparing scores.”
7. Avoid repeated contrastive constructions such as “X, not Y,” “rather than,” and “instead of.” Express the required scientific distinction directly.
8. Avoid mechanical “X: Y” explanatory sentences, slash-separated concepts, and using a plus sign for “and.” Normal punctuation and mathematical notation remain appropriate where needed.
9. Avoid manuscript narration about when studies were added, internal file management, or the research conversation.
10. Keep figures, tables, and text complementary. Explain what a result means in the prose and retain sufficient numerical support nearby.
11. Keep findings, limitations, and proposed explanations distinguishable. A control that changes multiple factors supports a joint comparison.
12. Respect the authors' methodological choices when describing their pipelines. Define on-policy responses relative to the feature model and identify who generated the data.

Examples of the preferred direction:

| Earlier wording the user rejected | Preferred approach |
| --- | --- |
| “Compute constraints were study-specific rather than a prospectively fixed common budget.” | “Compute constraints were study-specific.” Add any necessary comparison-budget qualification where that comparison is interpreted. |
| “Table A8 reports local retention, not public accessibility.” | “Table A8 summarizes retained artifacts.” Explain release availability separately. |
| “GPT-4o/manual agreement.” | “Agreement between GPT-4o labels and human annotations.” |
| “We ask: do probing methods retain their reported benefits?” | “We evaluate whether the reported probing benefits persist after refitting across model families.” |

### Study Design and Inclusion

- Present a coherent inclusion-criteria account covering candidate identification, eligibility, and artifact screening. Avoid fragmenting it into many small sections.
- Explain the search of major-conference proceedings and the preference for probing relevance, usable releases, cross-model applicability, and safety or alignment relevance.
- Historical search preferences emphasized recent, high-caliber studies, limited original model-family coverage, and claims that had received little independent stress testing. The user repeatedly excluded jailbreak-focused work and studies already tested across many families. The current eligibility statement should reflect the actual retained corpus and documented exceptions.
- Distinguish the documented search process from the intended criteria. Use the retained records to support claims about its completeness or timing.
- Explain each study's selected claim and why the available protocol permits that comparison.
- Keep model-selection constraints explicit and concise. Family, scale, architecture, access, and compute can affect eligibility.
- Preserve the unified corpus presentation. Keep provenance and decision records in the accompanying audit material.
- Retain the exclusion of LayerNavigator from the study inventory.

### Figures

- White background; Space Grotesk throughout figure text.
- Clear titles in Title Case. Use a subtitle only when it adds necessary context.
- Prefer readable bar charts for discrete comparisons. Use curves when layer-wise structure is the result. Preserve the data being summarized.
- Use sufficiently wide bars, balanced panel sizes, restrained grids, and consistent numeric-label placement.
- Put value labels above bars with enough space for uncertainty whiskers. Reduce font size when needed and check every label at paper scale.
- Label multi-panel figures with a), b), c), and d). Provide clear legends for paper, replication, extensions, and model identities.
- Use lively, readable palettes and consistent model-family colors where applicable. Figure-specific palette choices can differ.
- Display percentage signs with percentage values and define the metric in the label or caption. Use up to two decimal places where informative. Keep correlation metrics on their own named scale.
- Remove unnecessary 50% reference lines. Keep dashes and grid lines subtle.
- Put methodological explanations, bootstrap definitions, and interpretive qualifications in captions. Keep internal filenames and process notes out of images.
- Visually preview every figure before delivery and check the compiled page for overlap, clipping, and legibility.
- The approved PRISM gains and Models Under Pressure layer-coverage charts are the bar-chart style references. Earlier rejected designs should remain superseded.

### Tables and Main Evidence Overview

- Use clean, centered, legible tables with minimal rules and sensible grouping.
- Avoid large grids dominated by unrun conditions. The approved compact overview groups completed model entries by study and separates original-model baselines from extensions.
- Use green checks for qualified selected-finding recovery and red crosses for unresolved discrepancies. Use orange delta symbols for diagnostic results, with distinct, defined exploratory and mixed symbols.
- Symbols need a short legend. Study-specific qualifications belong in captions or nearby prose.
- Main-table cells should remain readable without bibliography clutter. Cite each study in adjacent prose or another appropriate manuscript location.
- Choose tables for compact exact comparisons and figures for patterns that benefit from visualization. Figure area must earn its place in the main argument.

### Supplement, References, and Delivery

- Use the official ICLR 2027 author kit without layout workarounds. Keep the main structure compact, with the user's preference for no more than six principal sections.
- Use Title Case for section headings such as “Supplementary Material.”
- Organize the supplement with a detailed, professional contents page. The user's structural exemplar is *MatAnyone: Stable Video Matting with Consistent Memory Propagation*; our content stays specific to this study.
- Keep the full document at or below the user's requested 30-page ceiling, ideally closer to 25 where scientific completeness permits. This is a project preference; conference submission requirements must also be checked.
- Keep reproducibility prose brief and factual. Put exact commands, scripts, long hashes, and detailed manifests in the artifact documentation.
- Use readable model names, optionally linked to the exact verified checkpoint. Avoid visible long revisions and code paths.
- Keep the data-provenance table's dataset names plain, following the user's later instruction. Add links elsewhere only when they reach the exact relevant resource.
- Prefer official proceedings or journal references, then arXiv, then other sources. Link a readable title and verify study identity. ACL Anthology can itself be the official proceedings source.
- Cite all included studies appropriately. Preserve the approved disclosure of Claude and Codex assistance.
- Provide editable LaTeX and a self-contained Overleaf ZIP with simple filenames. Inspect the compiled PDF and test a clean extraction when packaging is requested.

### Revision Boundaries

- Discuss proposed substantive changes before editing when the user requests approval.
- Work section by section when requested. Keep prose drafts, figure previews, and review notes separable.
- Keep communication concise and free of decorative emojis. Preserve the explicitly approved scientific symbols in the results overview.
- Preserve numerical evidence and protocol qualifications during style changes.
- Do not initiate additional experiments merely to strengthen wording. Experimental gaps should be identified for a scientific decision.
- Preserve previous manuscript versions and clearly identify the version under discussion.

## 5. Automated Review and Evidence Gaps

The exported PaperMentor review contains 30 comments, including four classified as critical. These are automated reviewer comments. The complete verbatim register is in [manuscript-review-comment-register.md](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/reports/manuscript-review-comment-register.md).

The retained response ledger labels 15 addressed, 13 partial, and two open for release decisions. These are recorded revision statuses. They do not establish that all underlying empirical questions have been resolved, and they have not been independently re-audited for this brief.

| Issue group | Comment IDs | Point to preserve in the discussion |
| --- | --- | --- |
| Claim strength and scope | C01–C05, C14, C29–C30 | Define the selected effect, denominators, observed magnitude, target-model refitting, and available uncertainty. |
| Sampling and verdict rules | C07–C08, C15, C19 | Explain inclusion and model selection. Missing prospective acceptance margins or screening records limit the claims available. |
| Fitting, selection, and formal setup | C09–C12, C20–C21 | Specify data partitions, preprocessing, candidate sites, selection data, search coverage, and grouping units. |
| Uncertainty and measurement | C13, C16–C17, C22–C23, C28 | Check metric definitions, numerical scopes, seed independence, threshold scales, and dependent observations. |
| Reproducibility and release | C24–C27 | Supply traceable audit evidence, retained outputs, access terms, and a versioned release where feasible. |
| Positioning and responsible use | C06, C18 | Explain the contribution relative to prior reproducibility work and address plausible profiling or monitoring harms. |

Evidence limitations recorded in the revision materials include missing prospective recovery margins, incomplete screening histories, missing row-level predictions, unequal independent seed coverage, and incomplete parity checks against author reference outputs. These require either additional evidence or carefully bounded conclusions.

The discrepancy implementation review records reuse of author components, comparisons with paper hyperparameters, compatibility checks, and reconstruction of retained score summaries. These checks should be explained concretely. Direct numerical parity with the original reference activations and evaluators remains a separate verification target.

## 6. Saved Writing Skill and Future Use

The user-supplied skill is preserved at [Research Paper Writing](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/tools/research-paper-writing/SKILL.md). Its [companion README](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/tools/research-paper-writing/README.md) records project-specific usage.

The skill is project-local. It must be explicitly loaded when continuing the work; storing it here alone does not ensure automatic inclusion in every future conversation. The section guides and example bank named under `references/` were not supplied. We should apply the saved workflow directly and identify any future additions as project-authored guidance.

### Required Writing Workflow

1. Agree on the section's message and its role in the paper.
2. Build a compact outline of three to seven points.
3. Draft one clear message per paragraph, with an informative opening sentence.
4. Reverse-outline the draft and check how each paragraph supports the section's claim.
5. Map each substantive claim to evidence and record its supported scope.
6. Review contribution, clarity, experimental strength, evaluation completeness, and method soundness.
7. Revise unsupported claims and verify stable terminology, logical transitions, citations, and layout.

For section revisions, retain an outline, paragraph roles, a short clarity checklist, and a claim–evidence map in the review notes. Only the approved scientific prose belongs in the manuscript.

### Reusable Handoff Instruction

> Before working on this manuscript, read the manuscript feedback brief and the project-local Research Paper Writing skill. Identify the manuscript version and the authorized scope of the edit. Preserve the empirical distinctions between artifact audits, original-model replication, cross-model refitting, exploratory searches, and independent evaluations. Apply the user's direct scientific style and approved figure conventions. Prepare a short outline and claim–evidence check before revising prose. Keep review scaffolding in separate notes. Seek clarification on the headline and automation claim before broad restructuring.

## 7. Questions to Send to the Professor

The following is a proposed meeting agenda. It is not an approved rewrite plan.

1. **Main contribution.** “Would you like the paper's main contribution to be the empirical replication study, the reusable verification pipeline, or an evaluated agent-based reproducibility system? This determines which evidence we need to foreground. Which manuscript version should we use as the starting point?”
2. **Automation claim.** “We can document the agent-assisted steps and researcher decisions. Are you proposing this as the workflow used in the study, or do you expect a separate evaluation of agent success rates, time, cost, and human intervention?”
3. **Headline.** “Several selected findings persist after refitting, and two original-model discrepancies remain unresolved. Would you support a headline about what a credible cross-model verification procedure must establish?”
4. **Decisive evidence.** “Which one or two matched comparisons should a reader encounter first? How should we present the baseline discrepancies alongside the strongest successful extensions?”
5. **Verification standard.** “For Among Us and Sycophancy, what additional check would make the discrepancy sufficiently well established? Would parity with an author evaluator on shared activations be the priority?”
6. **Discussion priorities.** “Should the substantive discussion focus on reference artifacts, response and label provenance, activation selection, and operating thresholds? Which three points should lead?”
7. **Submission threshold.** “Which remaining gaps are essential to resolve before submission? Can we agree on a short list of required checks and scope the other conclusions to the evidence already available?”

Suggested opening for the meeting:

> I have consolidated the writing and methodological feedback. I understand the proposed flow as a verification pipeline, its application to published probing studies, and a discussion of the resulting findings. Before restructuring further, I would like to agree on the main contribution, the level of automation we can claim, and the minimum additional verification needed for the headline.

## 8. Source Record

- **User and professor feedback:** the manuscript discussions in this conversation, including the most recent pipeline and discussion comments. Unquoted interpretations in this brief are editorial synthesis.
- **Automated review:** [original PaperMentor export](/Users/baimamboukar/Downloads/papermentor-1788997978200-ee038957-review_comments.json), containing 30 comments.
- **Earlier response ledger:** [point-by-point response](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/response_to_reviews.md) and [structured statuses](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/review_responses.json). These describe an earlier revision and include historical counts.
- **Reader-focused feedback:** [reader revision review](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/reader_revision/reviewer_response.md).
- **Approved prose reference:** [abstract review](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/abstract_revision_review.md) and [prose review](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/prose_revision_review.md).
- **Approved presentation decisions:** [compact table review](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/table_2_concepts/draft_e_review.md) and [integration review](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/approved_integration/writing_review.md). The later integration replaces the diagnostic letter with the approved delta symbol.
- **Reference presentation:** [study reference audit](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/review_revision/review/reference_audit/study_reference_review.md). Publication metadata were not rechecked for this discussion brief.
- **Separate pipeline-led version:** [editorial plan](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/pipeline_revision/review/editorial_plan.md), [abstract](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/pipeline_revision/sections/abstract.tex), and [current inventory](/Users/baimamboukar/Desktop/RESEARCH/neuroscience/JinesisLab/generalizable-interpretability/paper/pipeline_revision/tables/appendix/inventory.tex). Its framing remains a subject for the professor discussion.

This document records writing guidance and open decisions. It adds no experimental result, changes no verdict, and authorizes no new run or release.
