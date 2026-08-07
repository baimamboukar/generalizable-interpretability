# Implicit personalization

Upstream-faithful setup for
[Reading Between the Prompts](https://arxiv.org/abs/2505.16467), pinned to the
commit in `upstream.json`.

```bash
uv run python replications/implicit-personalization/run.py --stage inspect
```

The wrapper executes the released scripts unchanged. Train cross-validated
probe results, saved probes, and control probes separately:

```bash
python replications/implicit-personalization/run.py --stage probe --model llama
python replications/implicit-personalization/run.py --stage probe --model llama --save-probe
python replications/implicit-personalization/run.py --stage probe --model llama --control
```

Then evaluate one released demographic condition:

```bash
python replications/implicit-personalization/run.py \
  --stage evaluate --model llama --demographic gender
```

The compute-reduced validation uses five interior layers per model while calling
the released representation and probe functions unchanged:

```bash
python replications/implicit-personalization/selected_probe.py --model llama
python replications/implicit-personalization/selected_probe.py --model gemma
python replications/implicit-personalization/selected_probe.py --model olmo
python replications/implicit-personalization/plot_probe_validation.py
```

Selected layers are `12, 18, 24, 30, 36` for Gemma and
`9, 14, 18, 23, 28` for Llama and OLMo. This reproduces the trends in Figures
2 and 9–11, not every point in the original curves. The probe-only environment
uses `requirements-probe.txt`; `baukit` is required only for mitigation.

Use `requirements-runtime.txt` for the portable subset of the versions in the
authors' environment. The original `requirements.txt` contains machine-local
Conda paths and cannot be installed on another host as written.
