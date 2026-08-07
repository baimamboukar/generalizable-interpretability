# ICR Probe

Upstream-faithful setup for the ACL 2025 paper
[ICR Probe](https://arxiv.org/abs/2507.16488).

```bash
uv run python replications/icr-probe/run.py --stage inspect
```

The official repository is pinned at the commit recorded in `upstream.json`.
It releases the ICR calculation and probe classes, but not the complete data
generation, labeling, caching, training, or evaluation pipeline. The wrapper
therefore audits the release and does not recreate the missing baseline.
