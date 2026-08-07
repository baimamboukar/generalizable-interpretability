# PRISM

Faithful baseline setup for [PRISM](https://github.com/fujie-math/PRISM) at
commit `b0953654e3181cc3e4a158f0c4313b85c000d290`.

The runner executes byte-for-byte copies of the released generation and probe
scripts. Outputs live under `GI_ARTIFACT_DIR/prism`. For 13B, the wrapper
enables the model and output-path switches already included as comments in the
release, inside a verified artifact worktree. The pinned checkout is unchanged.

```bash
git clone https://github.com/fujie-math/PRISM external/PRISM
git -C external/PRISM checkout b0953654e3181cc3e4a158f0c4313b85c000d290
```

```bash
uv run python replications/prism/run.py --stage inspect
uv run python replications/prism/run.py --stage preflight

uv run python -m infra local -- \
  uv run --no-project --python 3.9 \
  --with-requirements external/PRISM/requirements.txt \
  python replications/prism/run.py \
  --stage generate --model llama2-7b --dataset true --representation prompted

uv run python -m infra local -- \
  uv run --no-project --python 3.9 \
  --with-requirements external/PRISM/requirements.txt \
  python replications/prism/run.py \
  --stage train --model llama2-7b --dataset true \
  --representation prompted --method saplma
```

The resumable 7B wrapper runs every released raw/prompted, True-False/
LogicStruct, MM/SAPLMA condition:

```bash
uv run python -m infra vast --instance INSTANCE_ID --sync --external PRISM -- \
  uv run --no-project --python 3.9 \
  --with-requirements external/PRISM/requirements.txt \
  --with accelerate==0.20.3 \
  --with sentencepiece==0.1.99 \
  --with protobuf==4.24.3 \
  python replications/prism/run.py --stage baseline --model llama2-7b
```

Use the same matrix for the paper's 13B condition:

```bash
uv run python -m infra vast --instance INSTANCE_ID --sync --external PRISM -- \
  uv run --no-project --python 3.9 \
  --with-requirements external/PRISM/requirements.txt \
  --with accelerate==0.20.3 \
  --with sentencepiece==0.1.99 \
  --with protobuf==4.24.3 \
  python replications/prism/run.py --stage baseline --model llama2-13b
```

Run the same command through Modal:

```bash
uv run python -m infra modal --gpu A100-80GB --external PRISM -- \
  uv run --no-project --python 3.9 \
  --with-requirements external/PRISM/requirements.txt \
  python replications/prism/run.py \
  --stage generate --model llama2-7b --dataset true --representation prompted
```

Or through an existing Vast.ai instance:

```bash
uv run python -m infra vast --instance INSTANCE_ID --sync --external PRISM -- \
  uv run --no-project --python 3.9 \
  --with-requirements external/PRISM/requirements.txt \
  python replications/prism/run.py \
  --stage generate --model llama2-7b --dataset true --representation prompted
```

`HF_TOKEN` must grant access to the gated Llama-2 repositories. Reproduce all
paper cells by running both models, both datasets, both representations, and
both probe methods.

The release omits `accelerate`, `sentencepiece`, and `protobuf`, although its
generation scripts require them. The wrapper adds `accelerate==0.20.3`,
`sentencepiece==0.1.99`, and `protobuf==4.24.3` at runtime. The release does
not pin model revisions and provides no LogicStruct seed driver.
