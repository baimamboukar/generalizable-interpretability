from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
SOURCE = ROOT / "external" / "Lookback-Lens"
MANIFEST = json.loads((HERE / "upstream.json").read_text())
ARTIFACTS = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts"))
ARTIFACTS = ARTIFACTS / "lookback-lens"
ARCHIVE = ARTIFACTS / "released" / "step1and2.tar.bz"
RELEASED = ARTIFACTS / "released" / "step1and2" / "dump"
FILES = {
    "nq_annotation": RELEASED / "anno-nq-7b.jsonl",
    "nq_features": RELEASED / "lookback-ratio-nq-7b.pt",
    "cnndm_annotation": RELEASED / "anno-cnndm-7b.jsonl",
    "cnndm_features": RELEASED / "lookback-ratio-cnndm-7b.pt",
}
PAPER = {
    ("predefined", "nq-7b", "cnndm-7b"): (0.983, 0.912, 0.853),
    ("predefined", "cnndm-7b", "nq-7b"): (0.977, 0.888, 0.820),
    ("sliding-window-8", "nq-7b", "cnndm-7b"): (0.883, 0.871, 0.661),
    ("sliding-window-8", "cnndm-7b", "nq-7b"): (0.862, 0.853, 0.660),
}


def commit() -> str | None:
    if not (SOURCE / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_archive() -> bool:
    return (
        ARCHIVE.exists()
        and ARCHIVE.stat().st_size == MANIFEST["artifact_bytes"]
        and sha256(ARCHIVE) == MANIFEST["artifact_sha256"]
    )


def inspect() -> None:
    print(
        json.dumps(
            {
                "paper": "Lookback Lens",
                "venue": MANIFEST["venue"],
                "source": str(SOURCE),
                "commit": commit(),
                "expected_commit": MANIFEST["commit"],
                "baseline": {
                    "target": "Table 2",
                    "model": "meta-llama/Llama-2-7b-chat-hf",
                    "datasets": ["Natural Questions", "CNN/DailyMail"],
                    "feature": "lookback ratio from every layer and attention head",
                    "probe": "sklearn LogisticRegression(max_iter=1000)",
                    "validation": "two-fold, random_state=42",
                    "settings": ["predefined spans", "sliding window 8"],
                },
                "released_table_2_artifacts": {
                    name: {"path": str(path), "exists": path.exists()}
                    for name, path in FILES.items()
                },
                "release_gaps": {
                    "scikit_learn_version": "not specified",
                    "scipy_version": "not specified",
                    "model_revision": "not pinned",
                    "table_4_13b_features": "not released",
                    "table_4_nq_train_features": "not released",
                    "table_4_head_mapping": "not released",
                },
            },
            indent=2,
        )
    )


def extract() -> None:
    target = RELEASED.parent
    if all(FILES[name].exists() for name in FILES):
        return
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:bz2") as archive:
        root = target.resolve()
        if any(
            root not in (target / member.name).resolve().parents
            for member in archive.getmembers()
        ):
            raise RuntimeError("Unsafe path in released artifact")
        archive.extractall(target)


def fetch() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not valid_archive():
        temporary = ARCHIVE.with_suffix(".part")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(MANIFEST["artifact_url"]) as source:
                    with temporary.open("wb") as destination:
                        shutil.copyfileobj(source, destination, 1024 * 1024)
                break
            except OSError:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        temporary.replace(ARCHIVE)
    if not valid_archive():
        raise RuntimeError("Released artifact hash mismatch")
    extract()
    missing = [name for name, path in FILES.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing released artifacts: {', '.join(missing)}")
    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "sha256": sha256(ARCHIVE),
                "files": {name: str(path) for name, path in FILES.items()},
            },
            indent=2,
        )
    )


def command(sliding: bool, output: Path) -> list[str]:
    args = [
        sys.executable,
        str(SOURCE / "step03_lookback_lens.py"),
        "--anno_1",
        str(FILES["nq_annotation"]),
        "--anno_2",
        str(FILES["cnndm_annotation"]),
        "--lookback_ratio_1",
        str(FILES["nq_features"]),
        "--lookback_ratio_2",
        str(FILES["cnndm_features"]),
        "--tokenizer_name",
        "meta-llama/Llama-2-7b-chat-hf",
        "--output_path",
        str(output),
    ]
    if sliding:
        args.extend(["--sliding_window", "8"])
    return args


def parse(mode: str, text: str) -> list[dict]:
    pattern = re.compile(
        r"^A=([^;]+);B=([^,]+), ([0-9.]+), ([0-9.]+), ([0-9.]+)$",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(text):
        source, target = match.group(1), match.group(2)
        values = tuple(float(match.group(index)) for index in range(3, 6))
        papers = PAPER[(mode, source, target)]
        for split, paper, result in zip(
            ("train", "test", "transfer"), papers, values
        ):
            rows.append(
                {
                    "model": "Llama-2-7B-Chat",
                    "setting": mode,
                    "source": source,
                    "target": target,
                    "split": split,
                    "metric": "AUROC",
                    "paper": paper,
                    "replication": result,
                    "delta": result - paper,
                }
            )
    if len(rows) != 6:
        raise RuntimeError(f"Could not parse official {mode} output")
    return rows


def baseline() -> None:
    if commit() != MANIFEST["commit"]:
        raise RuntimeError(f"Expected upstream commit {MANIFEST['commit']}")
    fetch()
    output = ARTIFACTS / "baseline"
    classifiers = output / "classifiers"
    classifiers.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "torch": version("torch"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "transformers": version("transformers"),
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    rows = []
    for mode, sliding in (("predefined", False), ("sliding-window-8", True)):
        result = subprocess.run(
            command(sliding, classifiers),
            cwd=SOURCE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        print(result.stdout)
        (output / f"{mode}.log").write_text(result.stdout)
        rows.extend(parse(mode, result.stdout))
    target = output / "table2.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("inspect", "fetch", "baseline"), required=True
    )
    args = parser.parse_args()
    {"inspect": inspect, "fetch": fetch, "baseline": baseline}[args.stage]()


if __name__ == "__main__":
    main()
