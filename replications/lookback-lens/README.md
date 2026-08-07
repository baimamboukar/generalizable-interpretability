# Lookback Lens

Source-preserving baseline scaffold for the EMNLP 2024 paper at commit
`e0a1fa3a898fbf6512af7be5567dea8ffe7a6620`.

```bash
git clone https://github.com/voidism/Lookback-Lens external/Lookback-Lens
git -C external/Lookback-Lens checkout \
  e0a1fa3a898fbf6512af7be5567dea8ffe7a6620

uv run python replications/lookback-lens/run.py --stage inspect
uv run python replications/lookback-lens/run.py --stage fetch
```

The first target is Table 2: fit the released logistic-regression probe on all
layers and attention heads from Llama-2-7B-Chat, using both predefined spans
and sliding windows of eight tokens on Natural Questions and CNN/DailyMail.

Run the authors' script unchanged:

```bash
uv run --no-project --python 3.9 \
  --with-requirements external/Lookback-Lens/requirements.txt \
  --with-editable external/Lookback-Lens/transformers-4.32.0 \
  python replications/lookback-lens/run.py --stage baseline

uv run --with matplotlib --with fonttools \
  python replications/lookback-lens/plot.py
```

The linked release contains the exact 7B lookback ratios and GPT-4o
annotations for Table 2. It does not contain the 13B features, NQ-train
features, or head-mapping matrix needed for Table 4.

## Cross-model extension

The extension reuses the released generations and labels, computes target-model
lookback ratios by teacher forcing, and reruns the authors' probe protocol.

```bash
python replications/lookback-lens/extension.py \
  --stage all \
  --model llama-3.1-8b
```
