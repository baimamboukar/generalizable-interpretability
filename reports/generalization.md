# Cross-paper generalization analysis

This analysis uses the complete frozen corpus rather than selecting only
failures. It separates four questions: whether a paper baseline reproduces,
whether the published probe family remains decodable after target-model
retraining, whether a layer or head selection rule ports without test-set
search, and whether a development-frozen operating threshold remains useful.
Previously inspected models are retrospective evidence. The two sealed model
revisions in `confirmatory/protocol.json` are reserved for prospective tests.

## Current pattern

The evidence does not support a universal claim that probes fail on new model
families. Target-specific retraining often recovers strong ranking performance.
The recurring weakness is the portability of the surrounding measurement
procedure: layer or head selection, prompt and response policy, label support,
probe-family ranking, and operating thresholds. The confirmatory analyses are
therefore designed to distinguish missing target-model decodability from a
selection or calibration failure.

## Models Under Pressure

The Llama-3.3-70B headline was reproduced, with an important specification
gap: the paper describes pre-layer-normalization residual activations while the
released hook records the layer-normalization output. Exploratory extension
results initially made Qwen3 look weak at layers 32 and 47. Expanding the
screen changed that conclusion: Qwen3 reached 0.9290 attention AUROC at layer
54 and 0.9012 softmax AUROC at layer 42. Qwen2.5 and Gemma also had sampled
layers above 0.91 for both probes.

These oracle-style peaks show that the signal is decodable, but they do not
show that the paper's layer choice ports. The best layer shifts across models,
and softmax is especially layer-sensitive. The frozen reanalysis therefore
compares a prespecified relative-depth rule, an equal sparse grid,
development-only selection, and a test-oracle diagnostic across three seeds.
It also evaluates development-frozen thresholds at 1% and 5% FPR.

## Internal Instruction Following

The baseline reproduces the paper's qualitative split: probes transfer well to
new tasks within known instruction types but remain near chance on unseen
instruction types. On Qwen, GLM, and Gemma, some held-out types collapse to a
single label because each model generates and labels its own answers. This
makes response policy and label support part of the estimand rather than a
minor nuisance.

The completed GLM control supports this interpretation. Substituting fixed
Llama-2-13B answers increased unseen-type AUROC from 48.37% to 55.02%, whereas
changing only the prompt template did not recover performance. Train-fitted
normalization changed Qwen and Gemma unseen-type means by less than 0.41
percentage points, excluding the release's split-wise normalization as the main
driver. Their factorial response-policy and prompt-template controls remain
confirmatory work; these hold the architecture fixed while crossing target
versus fixed responses with released versus native prompt templates.

## Political Perspective

The three original-model baselines reproduce at paper precision. Exploratory
extensions are mixed: Gemma and Phi match or exceed the original range, while
Qwen and GLM are lower. Those peak comparisons are not yet architecture-fair,
because models expose different numbers of layers and heads and therefore
receive unequal test-set search budgets. The frozen reanalysis uses nested
selection and the same candidate budget for every architecture. Until that
analysis is complete, the extension supports uneven decodability but not a
clean architecture-level failure claim.

## Positive controls and denominator

PRISM and Lookback Lens reproduce and transfer successfully in their tested
settings. Beyond Linear Probes also transfers after retraining, although its
proposed TPC is not the best probe on every extension model. The reduced
Implicit Personalization experiment confirms explicit demographic-identity
decodability across models, but does not test the paper's full stereotype or
steering claim. Among Us and Sycophancy contribute baseline-reproducibility
failures only. Missing-artifact and runtime-blocked studies remain in the audit
denominator and are not counted as cross-model failures.

## Submission claim boundary

The defensible emerging claim is conditional: probe methods often retain
target-model decodability, while the choices required to turn a probe into a
monitor do not automatically port. Final claims will use development-only or
nested selection, equal search budgets, clustered uncertainty, complete metric
records, and the sealed holdouts. Test-oracle results remain diagnostics and
cannot support the headline conclusion.
