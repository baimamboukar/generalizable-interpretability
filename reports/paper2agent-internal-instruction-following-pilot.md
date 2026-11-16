# Internal Instruction Following Agent Pilot

## Objective

The task compares a Raw Agent with an MCP-assisted agent using the same Phi-3
activations, responses, labels and data splits. Both conditions fit linear
probes and evaluate generalization to held-out tasks and instruction types.

## Primary Result

| Evaluation | Raw Agent | MCP-Assisted Agent |
| :--- | ---: | ---: |
| Held-out tasks, first token, layer 14 | 0.8849 AUROC | 0.8849 AUROC |
| Held-out instruction types, first token, layer 14 | 0.5611 AUROC | 0.5611 AUROC |

The matched results recover the study's main distinction: performance is high
on held-out tasks and close to chance on held-out instruction types.

The Raw Agent also tested an instruction-type-only baseline, which reached
0.916 AUROC. Within-type performance was approximately 0.63 AUROC. These
controls suggest that instruction-type base rates explain much of the pooled
held-out-task score. They were not part of the matched primary evaluation.

## Scope And Evidence

The Raw Agent fitted 162 probes and produced runnable code, fitted probes,
predictions, metrics, a log and a detailed report. The MCP fitted 57 probes and
produced fitted probes, predictions, metrics, a log and a short report. The
conditions therefore agree on the primary result but differ in evaluation
breadth and reporting detail.

## Efficiency

| Measure | Raw Agent | MCP-Assisted Agent |
| :--- | ---: | ---: |
| Claude cost | $1.0772 | $0.2088 |
| Claude API time | 178.5 s | 20.4 s |
| Recorded wall time | 354.1 s | 120.1 s |
| Human interventions | 0 | 0 |

MCP assistance reduced run-level Claude cost by 80.6% and API time by 88.6%.
These figures exclude MCP construction and should be interpreted with the
scope difference above.

## Conclusion

Both conditions completed the primary comparison and produced the same two
summary AUROCs. Future runs should require the same probe grid and reporting
artifacts before making a strict efficiency comparison.
