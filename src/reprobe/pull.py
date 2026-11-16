"""Download a study's public inputs and its pinned repository, then verify them.

    uv run --extra reprobe python src/reprobe/pull.py lookback-lens
"""

import subprocess
import sys

from constants import ARTIFACTS, HF_BUCKET, TASKS
from geninterp import benchmark_record
from helpers import load_yaml


def main(study: str) -> None:
    task_id = load_yaml("studies.yaml")[study]["task_id"]
    public = benchmark_record(task_id)["public"]
    folder = ARTIFACTS / study
    repository = folder / "reference" / "repository"
    checksums = TASKS / study / "inputs.sha256"

    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(["hf", "buckets", "sync", f"{HF_BUCKET}/reprobe/{study}", str(folder)], check=True)

    if not repository.exists():
        subprocess.run(["git", "clone", "--quiet", public["repository"], str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "checkout", "--quiet", public["commit"]], check=True)

    subprocess.run(["shasum", "-a", "256", "--check", "--quiet", str(checksums)], cwd=folder, check=True)
    print(f"{study}: inputs verified, repository at {public['commit'][:10]}, in {folder}")


if __name__ == "__main__":
    main(sys.argv[1])
