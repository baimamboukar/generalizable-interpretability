# Compute adapters

Copy `.env.example` to the ignored root `.env` and fill the credentials used by
the project. The adapters load it automatically. Vast receives no credentials
by default; pass only required names with `--env-key`. Uploads and commands use
the Vast SDK without SSH. Modal injects job credentials through
`Secret.from_dict`.

Set `HF_BUCKET`, run `python -m infra.artifacts create`, and enable
`HF_SYNC_ENABLED` after `python -m infra.artifacts check` succeeds. Modal and Vast jobs then sync
`GI_ARTIFACT_DIR` after successful runs. New experiments can initialize W&B
through `infra.tracking.init_run`.

Replication code should expose an ordinary command and write persistent outputs
under `GI_ARTIFACT_DIR`. The same command can run through any provider.

```bash
uv run python -m infra local -- python path/to/run.py --stage baseline

uv run python -m infra local --sync-artifacts -- \
  python path/to/run.py --stage baseline

uv run python -m infra modal \
  --gpu A100-80GB \
  -- python path/to/run.py --stage baseline

uv run python -m infra vast \
  --label experiment-name \
  --gpu A100_PCIE \
  --min-vram 75 \
  --min-ram 128 \
  --env-key HF_TOKEN \
  --env-key HF_BUCKET \
  --env-key HF_SYNC_ENABLED \
  -- python path/to/run.py --stage baseline
```

`--min-vram` and `--min-ram` are specified in GiB; `--disk` is specified in
GB. The adapter converts memory requirements to the units used by the Vast
offer API before selecting a machine.

Add only the required pinned upstream checkout:

```bash
uv run python -m infra vast \
  --label experiment-name \
  --external models-under-pressure \
  -- python path/to/run.py --stage baseline
```

Modal uses `App.run()` and `Function.remote()`. Vast uses `VastAI.execute()` for
commands and chunked uploads. No SSH or rsync transport is used. The same
`--external` option works with both providers.

Existing paper-specific Modal entrypoints remain supported. New replications
should keep experiment logic in a normal Python CLI and use this package only
for transport, caching, and artifact paths.
