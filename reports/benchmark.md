# Agent Benchmark

**Version:** 0.1  
**Status:** Pilot  
**Updated:** 13 November 2026

## Objective

The benchmark evaluates two questions separately:

1. Which probing procedures generalize across models, datasets and response
   regimes?
2. Can an agent reproduce and audit these evaluations faithfully and
   efficiently?

Each task provides a paper, pinned repository, labels, data splits, cached
activations and a concise scientific objective. Raw and MCP-assisted agents
receive the same task inputs and budget. Reference results and audit keys are
used only for evaluation.

## Current Evidence

Three paired evaluations have complete scientific results for both conditions.
The finalized LLMs Know rerun used the same 90 layer-selection cells and 18
final evaluations for both conditions. Each agent matched all 18 reference
AUROCs and all 18 selected layers.

| Study | Result | Raw Agent | MCP-Assisted Agent |
| :--- | :--- | ---: | ---: |
| LLMs Know | Mean test AUROC | 0.9099 | 0.9099 |
| Instruction Following | Held-out task AUROC | 0.8849 | 0.8849 |
| Instruction Following | Held-out instruction-type AUROC | 0.5611 | 0.5611 |
| Political Perspective | Peak Spearman correlation | 0.8252 | 0.8252 |

The Instruction Following Raw Agent also found that instruction type alone
reached 0.916 AUROC. This control was not part of the matched primary result.

The Lookback Lens pair is incomplete. The Raw Agent returned all 12 requested
AUROC values, while the MCP tool stopped during sliding-window processing with
a NumPy indexing error. Its two partial probe files are not benchmark results.

## Efficiency

| Study | Raw cost | MCP cost | Raw API time | MCP API time |
| :--- | ---: | ---: | ---: | ---: |
| LLMs Know | $1.0234 | $0.4227 | 171.1 s | 54.2 s |
| Instruction Following | $1.0772 | $0.2088 | 178.5 s | 20.4 s |
| Political Perspective | $0.7398 | $0.2371 | 109.8 s | 31.2 s |

Across these three pairs, MCP assistance reduced run-level Claude cost by
59--81% and Claude API time by 68--89%. These figures exclude MCP construction
and do not establish equal end-to-end workload. Workload is now recorded using
task-specific counts, such as candidate-layer evaluations and final test cells,
because a single probe count was ambiguous across implementations.

The LLMs Know MCP session took 1,357 seconds of wall time because the tool ran
twice. The duplicate outputs were identical, and only the second is retained.

## Benchmark Status

`data/benchmark.json` contains four evaluable tasks and four candidate task
definitions. A task definition is not evidence that an agent run was completed.

| Study | Current status |
| :--- | :--- |
| LLMs Know | Verified paired run; audit not evaluated |
| Instruction Following | Verified paired run |
| Political Perspective | Verified paired run |
| Lookback Lens | Incomplete paired run |
| PRISM | Candidate task only |
| Models Under Pressure | Candidate task only |
| Beyond Linear Probes | Candidate task only |
| Test Awareness Steering | Candidate task only |

The verified measurements are stored in
`data/results/agent-pilot/current_paired_runs.csv`. Earlier perfect-score
scale-up records were removed because they did not represent independent agent
reproductions.

## Next Steps

1. Apply the finalized task and scoring format to Instruction Following.
2. Repair and rerun the Lookback Lens MCP workflow.
3. Apply the shared task contract to the remaining candidate tasks.
4. Run candidate tasks without exposing reference outputs to the agents.
5. Score scientific fidelity, generalization, audit quality, evidence
   completeness, failures, cost, API time, wall time and human intervention
   separately.
