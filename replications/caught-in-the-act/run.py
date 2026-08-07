from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
UPSTREAM = ROOT / "external" / "caught-in-the-act"
ARTIFACTS = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts")) / "caught-in-the-act"
MANIFEST = json.loads((HERE / "upstream.json").read_text())
COMMIT = MANIFEST.get("commit", "")
REPOSITORY = MANIFEST.get("repository", "")
ENTRYPOINT = MANIFEST.get("default_openrouter_entrypoint", "")
DEFAULT_PROBE_MODELS = tuple(MANIFEST.get("default_probe_models", []))
DEFAULT_JUDGE_MODELS = tuple(MANIFEST.get("judge_models", []))

MODEL_INDEX = {
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

JUDGE_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}


def source_files_digest() -> str | None:
    if not UPSTREAM.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(UPSTREAM.rglob("*")):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(UPSTREAM)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def source_commit() -> str | None:
    if not UPSTREAM.exists():
        return None
    if not (UPSTREAM / ".git").exists():
        return "stripped"
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def upstream_ready() -> bool:
    return bool(REPOSITORY and COMMIT) and UPSTREAM.exists()


def inspect() -> None:
    print(
        json.dumps(
            {
                "paper": "Caught in the Act: a mechanistic approach to detecting deception",
                "paper_url": MANIFEST.get("paper"),
                "paper_version": MANIFEST.get("paper_version"),
                "venue": MANIFEST.get("venue"),
                "repository": REPOSITORY or None,
                "commit": source_commit(),
                "expected_commit": COMMIT,
                "source_digest": source_files_digest() if UPSTREAM.exists() else None,
                "artifacts_root": str(ARTIFACTS),
                "status": "ready" if upstream_ready() else "blocked-upstream-not-pinned",
            },
            indent=2,
        )
    )


def preflight() -> None:
    if not REPOSITORY:
        raise RuntimeError(
            "Cannot run preflight: upstream repository URL is not yet populated in "
            "replications/caught-in-the-act/upstream.json"
        )
    if not COMMIT:
        raise RuntimeError(
            "Cannot run preflight: upstream commit has not been pinned in "
            "replications/caught-in-the-act/upstream.json."
        )
    if not UPSTREAM.exists():
        raise RuntimeError(
            "Cannot run preflight: clone upstream into external/caught-in-the-act first."
        )
    if COMMIT and source_commit() != COMMIT:
        raise RuntimeError(
            f"Expected upstream commit {COMMIT}, found {source_commit() or 'unknown'}"
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "source": str(UPSTREAM),
                "source_commit": source_commit(),
                "expected_commit": COMMIT or None,
                "entrypoint": (UPSTREAM / ENTRYPOINT).as_posix() if ENTRYPOINT else None,
                "hf_router_key": bool(os.environ.get("OPENROUTER_API_KEY"))
                or bool(os.environ.get("OPEN_ROUTER_API_KEY")),
            },
            indent=2,
        )
    )


def parse_choices(raw: str, allowed: dict[str, str]) -> tuple[str, ...]:
    choices = []
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        key = key.lower()
        if key not in allowed:
            raise ValueError(f"Unknown model key {key!r}; choose from {sorted(allowed)}")
        choices.append(key)
    if not choices:
        raise ValueError("No model keys provided")
    return tuple(choices)


def baseline(models: str, judges: str) -> None:
    if not REPOSITORY or not COMMIT:
        raise RuntimeError(
            "Blocker: pin upstream repo + commit before baseline run. "
            "Populate upstream.json and keep the checkout in external/caught-in-the-act."
        )
    if not UPSTREAM.exists():
        raise RuntimeError("Missing external/caught-in-the-act checkout.")
    if source_commit() != COMMIT:
        raise RuntimeError(
            f"Expected upstream commit {COMMIT}, found {source_commit() or 'unknown'}"
        )
    probe_keys = parse_choices(models, MODEL_INDEX)
    judge_keys = parse_choices(judges, JUDGE_MODELS)
    resolved_judges = [JUDGE_MODELS[key] for key in judge_keys]
    resolved_models = [MODEL_INDEX[key] for key in probe_keys]
    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API_KEY")):
        raise RuntimeError(
            "Missing OpenRouter key: set OPENROUTER_API_KEY (or OPEN_ROUTER_API_KEY) "
            "before running baseline."
        )
    if not ENTRYPOINT:
        raise RuntimeError(
            "Blocked: default openrouter entrypoint is unset in "
            "replications/caught-in-the-act/upstream.json."
        )
    entrypoint = UPSTREAM / ENTRYPOINT
    if not entrypoint.exists():
        raise FileNotFoundError(f"Expected upstream entrypoint {entrypoint}, not found.")
    command = [
        sys.executable,
        str(entrypoint),
        "--stage",
        "baseline",
        "--models",
        *resolved_models,
        "--judge-models",
        *resolved_judges,
        "--artifacts-dir",
        str(ARTIFACTS),
    ]
    print(
        json.dumps(
            {
                "status": "ready",
                "command": command,
                "probe_models": resolved_models,
                "judge_models": resolved_judges,
                "notes": "Run command manually after upstream code is pinned and synced in external/caught-in-the-act.",
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("inspect", "preflight", "baseline"),
        required=True,
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_PROBE_MODELS)
        if DEFAULT_PROBE_MODELS
        else ",".join(MODEL_INDEX.keys()),
        help="Comma-separated probe model keys.",
    )
    parser.add_argument(
        "--judge-models",
        default=",".join(DEFAULT_JUDGE_MODELS)
        if DEFAULT_JUDGE_MODELS
        else ",".join(JUDGE_MODELS.keys()),
        help="Comma-separated judge model keys.",
    )
    args = parser.parse_args()

    if args.stage == "inspect":
        inspect()
    elif args.stage == "preflight":
        preflight()
    else:
        baseline(args.models, args.judge_models)


if __name__ == "__main__":
    main()
