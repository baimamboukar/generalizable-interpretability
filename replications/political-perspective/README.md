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
architectures. It evaluates every attention head in every layer.

```bash
uv run --with-requirements \
  replications/political-perspective/requirements-extension.txt \
  python replications/political-perspective/extension.py \
  --stage all --model qwen3-8b
```
