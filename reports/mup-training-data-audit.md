# High-Stakes Probe Training Data Audit

The probe training data are synthetic input texts generated and labelled by GPT-4o. The feature-extraction model encodes these texts but does not generate them. This distinction separates the source of supervision from the model whose representations are measured.

## Dataset Provenance

The audited local copy corresponds to the `prompts_4x` training collection in the authors' `Arrrlex/models-under-pressure` release. It contains 8,000 training examples and 2,000 held-out synthetic examples. The training set is balanced between 4,000 high-stakes and 4,000 low-stakes labels, with 3,687 distinct pair identifiers.

Each training input is a single text string. The generation procedure creates paired situations, expands them into queries and assigns stakes and confidence scores. Examples with intermediate stakes scores or low confidence are filtered out. All 8,000 training rows identify GPT-4o as the label model.

| Stakes score | Examples |
| :--- | ---: |
| 1 | 53 |
| 2 | 1,310 |
| 3 | 2,637 |
| 8 | 2,070 |
| 9 | 1,835 |
| 10 | 95 |

## Training Composition

| Attribute | Recorded composition |
| :--- | :--- |
| Topics | 27 |
| Language | English 4,862; Hindi 1,078; German 1,052; French 1,008 |
| Tone | Casual 2,295; overly polite 2,033; vulnerable 1,851; angry 1,821 |
| Style | Third person 1,924; character perspective 1,680; question 1,518; response to situation 1,501; prompt to LLM 1,377 |
| Length category | Very short 2,940; short 2,359; medium 1,664; long 1,037 |

The recorded composition counts agree with the corresponding paper tables. This agreement supports consistency of the dataset description; it does not establish byte-level identity with the authors' experimental copy.

## Evaluation Distribution

Evaluation includes human-written conversations, medical records and dialogues, social-media statements, red-team prompts and synthetic tool-use conversations. ToolACE is synthetic. The evaluation collections include both single-turn and multi-turn inputs, with different lengths, domains and annotation ambiguity from the training set.

The training text is external to the model being probed. Describing this as off-policy input encoding is useful only with that definition: the target is a property of an input context, not the probed model's generated behaviour. Synthetic-to-external distribution shift is a separate issue from the identity of the text generator.

## Limitations And Implications

GPT-4o supplies both generation and labelling, introducing potential correlated biases. The training collection lacks explicit tool-use and psychological-harm coverage, while these domains appear in evaluation. Such gaps are plausible contributors to errors, but the existing results do not isolate their causal effects.

Fixed training texts support comparisons across feature-extraction models. They do not, by themselves, attribute differences to representation geometry: activation sites, tokenization, fitting, layer selection and calibration also vary. A controlled data-coverage study would hold these choices fixed while varying the generator or adding matched development examples.

The audit uses the local training artifact and the paper's dataset description, especially Appendices C.1 and C.7. It does not establish the current state of the remote dataset release.
