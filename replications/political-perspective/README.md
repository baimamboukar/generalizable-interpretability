# Linear Representations of Political Perspective

Source-preserving baseline scaffold for the ICLR 2025 paper at commit
`e746dd1e3838ee6966818fef5a9b2d4e107f1656`.

```bash
git clone https://github.com/JunsolKim/RepresentationPoliticalLLM \
  external/RepresentationPoliticalLLM
git -C external/RepresentationPoliticalLLM checkout \
  e746dd1e3838ee6966818fef5a9b2d4e107f1656

uv run python replications/political-perspective/run.py --stage inspect
uv run --with numpy==1.26.4 python \
  replications/political-perspective/run.py --stage released-check
```

The baseline target is Figure 2 and Table A3: attention-head ridge probes for
Llama-2-7B-Chat, Mistral-7B-Instruct-v0.1, and Vicuna-7B-v1.5. Figure 3 tests
the top-32 ensemble on lawmakers and transfers the same probes to news outlets.

The release is notebook-only. Its invalid token assignment is replaced by an
approved one-line `HF_TOKEN` lookup in a hashed artifact copy. Cells 1–13 are
executed unchanged and execution stops before the separate intervention
experiment. The pinned upstream checkout is not changed.

The release omits the scikit-learn version despite importing it. We use
release-era `scikit-learn==1.4.1.post1` and record this as an environment
specification gap. Its unpinned `baukit` install also requires torchvision;
`torchvision==0.18.0` is paired with the released PyTorch 2.3.0 to prevent
current torchvision releases from replacing the paper runtime.
The notebook also installs `accelerate` without a version. Its first launch
resolved a version incompatible with `pyvene`'s exact Hub dependency, so the
reconstruction pins `accelerate==0.27.2` and `huggingface-hub==0.20.3`.

Figure 2, Table A3, and Figure 3 are reproduced from independently extracted
activations. `remaining_claims.py` executes the relevant cells from the
released figure notebook unchanged.

## Extension

The extension keeps the released prompts, labels, activation site, probe, and
evaluation protocol while adapting the attention hook to additional model
architectures. It evaluates every attention head in every layer. New
extractions retain row identifiers, both source datasets, the resolved model
revision, prompt and activation lineage, zero-based head indices, and data
hashes.

```bash
uv run --with-requirements \
  replications/political-perspective/requirements-extension.txt \
  python replications/political-perspective/extension.py \
  --stage extract --model qwen3-8b
```

Each extension revision is pinned in `confirmatory_protocol.json`; the
extractor rejects any conflicting override.

## Confirmatory analysis

The submission analysis separates head selection from evaluation. Within each
outer politician fold, heads are ranked only by inner-fold performance. It
reports the selected top head and top-32 ensemble on held-out politicians, then
selects on all politician data before evaluating the independent Ad Fontes news
set. Every architecture receives the same layer-balanced candidate budget.

```bash
uv run --with-requirements \
  replications/political-perspective/requirements-extension.txt \
  python replications/political-perspective/confirmatory.py \
  --stage analyze --model qwen3-8b
```

The default candidate budget, seeds, top-k, folds, ridge alpha, bootstrap
counts, index base, and selection metric live in
`confirmatory_protocol.json`. `--protocol` can select another protocol file;
individual CLI options can override its values. The protocol SHA256 and every
explicit override are retained in the run manifest.

The output retains politician- and document-level predictions, nested head
rankings selected by mean fold-wise Spearman, Spearman, Pearson, MAE and RMSE,
and both conditional and full-pipeline clustered bootstrap intervals. The
fixed-budget head search is repeated under protocol seeds 0, 1 and 2. A hashed
manifest records every selection and uncertainty choice. Before combining
model results, audit that all runs used the same protocol:

```bash
uv run --with-requirements \
  replications/political-perspective/requirements-extension.txt \
  python replications/political-perspective/confirmatory.py \
  --stage prepare-released --model llama2-7b
```

This converts the authors' released baseline tensors without model inference.
Their model revisions were not released, so the manifest preserves that gap.
Then run the same `analyze` command with `--model llama2-7b`.

```bash
uv run --with-requirements \
  replications/political-perspective/requirements-extension.txt \
  python replications/political-perspective/confirmatory.py \
  --stage audit-panel \
  --manifest artifacts/political-perspective/extensions/qwen3-8b/confirmatory/manifest.json \
  --manifest artifacts/political-perspective/extensions/glm4-9b/confirmatory/manifest.json
```

The existing `all_heads.csv` files are sufficient only for the earlier oracle
peak summary. Nested selection and row-level uncertainty require politician
feature tensors. Independent news transfer also requires news feature tensors.
The current extension artifacts do not contain those tensors locally, so the
Qwen, GLM, Gemma and Phi confirmatory outputs require new pinned-revision
inference. The three original-model baselines can reuse the released tensors.
No GPU run is started by these scripts without an explicit command.
