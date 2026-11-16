# Benchmark Task And Scoring Contract

## Benchmark Item

Each benchmark item has a public task and a private evaluation record. Agents
receive only the public task.

The public task contains:

- task identifier, title, track, status and split;
- paper, repository, commit and input descriptions;
- scientific objective and concise instruction;
- input hash when available;
- expected or minimum workload counts when fixed;
- required result identifiers;
- required artifact and efficiency fields.

The private evaluation record contains:

- human-verified reference values;
- numerical tolerances;
- known audit items;
- the minimum audit F1 when an audit is required.

`python3 geninterp.py benchmark show TASK_ID` exports the public task without
the reference values or audit keys. The complete benchmark file must not be
copied into an agent workspace.

## Agent Submission

```json
{
  "task_id": "political-perspective-qwen3-generalization",
  "results": {
    "peak_spearman": 0.8252,
    "peak_layer": 17,
    "peak_head": 8,
    "num_probes": 1152
  },
  "protocol": {
    "result_count": 4,
    "counts": {
      "evaluation_cells": 1152
    },
    "scope": "All 36 layers and 32 heads with two-fold ridge evaluation",
    "input_hash": "<sha256>"
  },
  "audit_findings": [],
  "artifacts": {
    "implementation": ["code/pipeline.py"],
    "probes": ["probes/probes.npz"],
    "predictions": ["predictions/predictions.npz"],
    "metrics": ["metrics/results.json"],
    "logs": ["logs/run.log"],
    "report": ["REPORT.md"],
    "trace": ["trace/session.json"]
  },
  "efficiency": {
    "agent_cost_usd": 0.74,
    "api_time_seconds": 110,
    "wall_time_seconds": 156,
    "mcp_build_time_seconds": 0,
    "human_interventions": 0
  }
}
```

Artifact paths are relative to the submission file. The evaluator verifies that
every required file exists, is non-empty and remains inside the submission
directory. It records a SHA-256 hash for each verified file.

## Scoring

The evaluator reports separate components:

1. **Scientific fidelity:** proportion of reference values within tolerance,
   with per-result errors and mean absolute error.
2. **Protocol validity:** result count, workload counts, scope and input hash.
3. **Audit quality:** precision, recall and F1 after an evaluator maps the
   agent's free-text findings to the private audit items.
4. **Evidence completeness:** verified artifact groups present.
5. **Efficiency:** validated cost, API time, wall time, MCP construction time
   and human intervention.

An overall pass requires all scientific results within tolerance, a valid
protocol declaration, every artifact group, complete efficiency metadata and,
when applicable, audit F1 above the task threshold. Efficiency does not
compensate for an incorrect scientific result.

Audit scoring is enabled only when the public task explicitly asks the agent
to audit the study. Other runs report the audit component as not evaluated.

Candidate tasks cannot be scored until their task scope and evaluation record
are reviewed.
