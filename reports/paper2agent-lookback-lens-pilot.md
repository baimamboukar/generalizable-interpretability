# Lookback Lens Agent Pilot

## Objective

The task asks a Raw Agent and an MCP-assisted agent to reproduce the 12 Table 2
AUROC values from the released annotations and lookback-ratio features. Neither
condition receives the reference values.

## Current Status

The paired evaluation is incomplete.

The Raw Agent fitted the probes and returned all 12 requested values:

| Setting | Source | Train AUROC | Test AUROC | Transfer AUROC |
| :--- | :--- | ---: | ---: | ---: |
| Predefined span | NQ | 0.9868 | 0.9144 | 0.8526 |
| Predefined span | CNN/DM | 0.9846 | 0.8722 | 0.8207 |
| Sliding window 8 | NQ | 0.8856 | 0.8665 | 0.6589 |
| Sliding window 8 | CNN/DM | 0.8652 | 0.8475 | 0.6621 |

The MCP tool fitted two predefined-span probes, then stopped during
sliding-window processing with:

```text
only integer scalar arrays can be converted to a scalar index
```

It did not produce predictions, metrics, logs, runnable code or a report. The
agent session completed normally because it reported the tool failure, but the
scientific task did not complete.

## Run Record

| Measure | Raw Agent | MCP-Assisted Agent |
| :--- | ---: | ---: |
| Scientific outcome | Complete | Incomplete |
| Claude cost | $1.6104 | $0.2003 |
| Claude API time | 220.8 s | 24.2 s |
| Human interventions | 0 | 0 |

No efficiency comparison is reported because the MCP condition did not
complete the requested work.

## Next Step

Repair the sliding-window indexing path and rerun the MCP condition with the
same inputs. The partial probe files must not be scored.
