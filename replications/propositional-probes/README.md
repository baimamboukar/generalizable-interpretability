# Monitoring Latent World States with Propositional Probes

Upstream-faithful wrapper for the [official release](https://github.com/jiahai-feng/prop-probes-iclr)
at commit `8e45e1c24c375bba01c208249e9ae14791176694`.

The first target is Table 1's standard Tulu-2-13B baseline: evaluate the
released layer-20 domain probes and Hessian binding subspace on `synth`,
`para`, and Spanish `trans`, reporting exact match and Jaccard.

```bash
uv run python replications/propositional-probes/run.py --stage inspect
```

The official checkpoints and deterministic proposition labels are recoverable.
The exact GPT-3.5-generated `para` and `trans` texts are absent from the
repository, its complete Git history, and the artifact archive. Recreating
those texts would be a proxy rather than an exact baseline replication.

If the exact datasets become available, the wrapper runs the authors'
entrypoint unchanged:

```bash
uv run python replications/propositional-probes/run.py \
  --stage baseline \
  --config PATH_TO_OFFICIAL_CONFIG
```
