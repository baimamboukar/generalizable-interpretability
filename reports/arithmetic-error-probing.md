# Probing For Arithmetic Errors In Language Models

[Paper](https://arxiv.org/abs/2507.12379) · EMNLP 2025 · [Released code](https://github.com/yuchen9-5un/arithmetic-error-probing)

The paper probes Gemma-2-2B-IT residual-stream activations at the equals-sign token while it performs three-digit addition. Circular, ridge, multiclass logistic, and one-hidden-layer MLP probes decode the model's predicted hundreds digit and the correct digit across all 26 layers. Five lightweight error detectors then compare or jointly model those signals. Figure 3 reports over 90% error-detection accuracy in late layers, while Figure 5 reports approximately 85% transfer from direct arithmetic to structured addition-only GSM8K reasoning.

## Release Audit

The release is not yet executable as a strict baseline. Its main pure-arithmetic trainer enables only the ridge probe and disables every error detector, whereas Figure 3 requires four probe families and five detectors. The paper states 10,000 training epochs for each probe, but source defaults use 1,000 for MLP, logistic, and error-detector training. The paper selects the first incorrect CoT equation, or the first equation when all are correct; the released generator randomly chooses among eligible equations.

The source also writes to an `arithmetic_error_probing` package path that does not match the pinned checkout, omits the paper's bfloat16 setting in the primary loaders, and does not pin the Gemma checkpoint revision or dependency versions. Only the symbolic GSM8K templates are released; the exact generated outputs, 800-example pure set, 685-step CoT set, activations, checkpoints, and result tables are absent.

## Status

Audit complete; strict baseline blocked before compute. Running it faithfully would require either additional author artifacts or an explicitly approved compatibility implementation, which would no longer be an unchanged execution of the released baseline.
