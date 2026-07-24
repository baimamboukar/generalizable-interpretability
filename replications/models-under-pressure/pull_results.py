"""Pull a probe-eval results jsonl from R2 into the local repo so it can be read/plotted.

Usage:
    python replications/models-under-pressure/pull_results.py llama70b_layer31_softmax
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import boto3

REPO = Path(__file__).resolve().parents[2] / "external" / "models-under-pressure"
load_dotenv(REPO / ".env")

run_id = sys.argv[1] if len(sys.argv) > 1 else "llama70b_layer31_softmax"
key = f"results/{run_id}.jsonl"

dest_dir = REPO / "data" / "results" / "evaluate_probes"
dest_dir.mkdir(parents=True, exist_ok=True)
dest = dest_dir / f"results_{run_id}.jsonl"

c = boto3.client(
    "s3",
    endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)
bucket = os.environ["R2_ACTIVATIONS_BUCKET"]
c.download_file(bucket, key, str(dest))
print(f"pulled r2://{bucket}/{key} -> {dest}")
print(f"bytes: {dest.stat().st_size}")
