# Caught in the Act Replication

Repository setup for the study:

- **Caught in the Act: a mechanistic approach to detecting deception**
- arXiv: [2508.19505](https://arxiv.org/abs/2508.19505)

The wrapper is intentionally release-faithful: once the upstream repo and commit are
confirmed, the pinned checkout is cloned unchanged and the replication stages run
against that frozen source.

```bash
git clone <upstream-repository-url> external/caught-in-the-act
git -C external/caught-in-the-act checkout <upstream-commit>

uv run python replications/caught-in-the-act/run.py --stage inspect
```

## Baseline target (paper-faithful scope)

| Step | Details |
| :--- | :--- |
| Data generation | Synthetic argument generation from MMLU prompts (filter + verifier steps) |
| Features | Final-position residual stream activations |
| Probes | Linear classifiers for deception / non-deception |
| Baselines | GPT-4o and GPT-4o-mini text judges |
| Reliability checks | INLP projections and saliency attribution |
| Probe models | Qwen 2.5 (3B), DeepSeek-r1 (1.5B) |
| Optional probe model extension | DeepSeek-r1 (7B) |
| Baseline judges | GPT-4o and GPT-4o-mini via OpenRouter |

## Current setup contract

- `upstream.json` stores repository URL, pinned commit, and metadata.
- `run.py` is the single wrapper entrypoint for preflight and stage execution.
- `reports/caught-in-the-act.md` contains all logged target values and decisions.

```bash
uv run python replications/caught-in-the-act/run.py --stage inspect
uv run python replications/caught-in-the-act/run.py --stage preflight

# OpenRouter judges default to GPT-4o and GPT-4o-mini
uv run python replications/caught-in-the-act/run.py --stage baseline

# Try the two requested models only
uv run python replications/caught-in-the-act/run.py --stage baseline \
  --models qwen2.5-3b,deepseek-r1-1.5b

# Use the 7B DeepSeek-R1 alternative when you want it instead
uv run python replications/caught-in-the-act/run.py --stage baseline \
  --models qwen2.5-3b,deepseek-r1-7b
```

## Setup blockers (must be resolved before baseline run)

1. Confirm official upstream repository URL (not included in arXiv metadata).
2. Pin commit that matches release scripts, prompts, and generated artifacts.
3. Capture lockfile/environment if the release does not fully pin requirements.
