# Cross-Study Analysis

The research corpus separates baseline fidelity, target-model decodability, selection procedures and operating thresholds. These are distinct outcomes. Recovering a predictive probe after fitting it to a new model does not establish that an original layer-selection rule or threshold transfers.

## Baseline Fidelity

The primary comparisons contain six qualified recoveries of selected findings and two unresolved original-model discrepancies. Among Us and Sycophancy Hides contribute baseline discrepancies. Their results cannot establish a general failure of probes on new models. Artifact and execution audits remain part of the corpus even when they produce no performance measurement.

The recovered findings have different scopes. Lookback Lens includes a replay of released features. Implicit Personalization tests explicit demographic identity in a reduced setting. Beyond Linear Probes covers one original model and layer. These boundaries prevent interpreting selected comparisons as complete paper replications.

## Generalization Findings

High-stakes probes recover predictive signals across several model families, but their scores and relative rankings depend on layer choice. Expanded test-layer searches improve some initially weak results. These maxima describe available signals; they do not evaluate an independent selection procedure.

Internal Instruction Following preserves the distinction between unseen tasks and unseen instruction types. Model-generated responses change both labels and activations, with single-label folds making some AUROCs undefined. The GLM response control changes unseen-type performance while holding the model fixed, but changes in responses and labels are not separately identified.

Political Perspective shows mixed exploratory results across models. Different numbers of available heads give models unequal search budgets, limiting architecture-level comparisons. PRISM, Lookback Lens and Beyond Linear Probes support successful target-specific fitting in their tested settings, with qualifications concerning data, probe ranking and selection.

The complementary LLMs Know More Than They Show experiment finds predictive exact-answer correctness signals across additional models. Fixed-response controls alter individual scores and selected layers, showing that response provenance affects the measured result.

## Interpretation

The evidence supports a conditional research claim: probe methods often recover decodable signals after target-specific fitting, while selection, response policy, label support and calibration affect their reliability. Comparisons should retain their original aggregation, valid-fold counts, fitting repetitions and feature definitions. Development-only selection and held-out operating-threshold evaluation are required for stronger claims about a reusable monitoring procedure.

Historical experiment specifications describe earlier analyses. They do not establish that every planned condition was completed or that a new benchmark has been preregistered.
