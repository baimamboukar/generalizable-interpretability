#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential
exec uv run \
  --python 3.12 \
  --with-requirements replications/beyond-linear-probes/requirements-runtime.txt \
  python replications/beyond-linear-probes/extension.py \
  --stage pipeline \
  --model "$1"
