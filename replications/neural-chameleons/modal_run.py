"""Modal entrypoints for the Neural Chameleons replication setup.

Examples:
    modal run replications/neural-chameleons/modal_run.py --stage inspect
    modal run replications/neural-chameleons/modal_run.py --stage released-demo
    modal run replications/neural-chameleons/modal_run.py --stage paper-plots
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess
import sys

import modal


HERE = Path(__file__).resolve().parent
REMOTE_MINIMAL = Path("/minimal")
REMOTE_OFFICIAL = Path("/official")
REMOTE_CACHE = Path("/cache")
PROJECT_ROOT = HERE.parents[1] if len(HERE.parents) > 1 else None
LOCAL_MINIMAL = (
    PROJECT_ROOT / "external" / "minimal-neural-chameleons"
    if PROJECT_ROOT is not None
    else REMOTE_MINIMAL
)
LOCAL_OFFICIAL = (
    PROJECT_ROOT / "external" / "neural-chameleons"
    if PROJECT_ROOT is not None
    else REMOTE_OFFICIAL
)

source_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(LOCAL_MINIMAL, str(REMOTE_MINIMAL), copy=True)
    .add_local_dir(LOCAL_OFFICIAL, str(REMOTE_OFFICIAL), copy=True)
)

# Exact direct dependency versions from the author-linked minimal reproduction's
# uv.lock at commit f077304. Notebook-only packages are intentionally omitted.
demo_image = source_image.pip_install(
    "accelerate==1.10.1",
    "hf-transfer==0.1.9",
    "huggingface-hub==0.35.3",
    "numpy==2.3.4",
    "pandas==2.3.3",
    "scikit-learn==1.7.2",
    "torch==2.9.0",
    "transformers==4.57.1",
)

# The official plot script imports these modules at import time even when only
# the scatter plot is requested. Versions are left unconstrained where the
# official setup.py leaves them unconstrained.
plot_image = demo_image.pip_install(
    "matplotlib==3.10.7",
    "seaborn==0.13.2",
    "python-dotenv",
    "sentence-transformers",
)

app = modal.App("neural-chameleons-replication")
secret = modal.Secret.from_name("mup-env")
volume = modal.Volume.from_name("neural-chameleons", create_if_missing=True)
mounts = {str(REMOTE_CACHE): volume}


@app.function(image=source_image, cpu=2, memory=8192, timeout=30 * 60)
def inspect_release() -> dict:
    """Verify the pinned artifacts and report the full-pipeline blocker."""
    demo = REMOTE_MINIMAL / "demo.py"
    probe_files = sorted((REMOTE_MINIMAL / "probes").glob("*_weights.pt"))
    full_pipeline = (
        REMOTE_OFFICIAL
        / "experiments"
        / "self_obfuscation_pipeline"
        / "run_full_pipeline.py"
    )
    release_config = (
        REMOTE_OFFICIAL
        / "experiments"
        / "self_obfuscation_pipeline"
        / "model_config_21preset.json"
    )
    for path in (demo, full_pipeline, release_config):
        if not path.exists():
            raise FileNotFoundError(path)
    if len(probe_files) != 13:
        raise ValueError(f"Expected 13 released probes, found {len(probe_files)}")

    checkpoint_refs = sorted(
        set(
            re.findall(
                r'probe_checkpoint_regular\s*=\s*"([^"]+)"',
                full_pipeline.read_text(),
            )
        )
    )
    missing_refs = [ref for ref in checkpoint_refs if not (REMOTE_OFFICIAL / ref).exists()]
    result = {
        "official_commit": "04552285b7c11aae9e117d75af22c5c681e617f9",
        "minimal_commit": "f077304f6bebd31ded4cc868f2115a8c81067960",
        "released_demo": str(demo),
        "released_probe_count": len(probe_files),
        "full_pipeline_probe_refs": len(checkpoint_refs),
        "missing_full_pipeline_probe_refs": len(missing_refs),
        "full_pipeline_runnable_as_released": not missing_refs,
        "release_config": json.loads(release_config.read_text()),
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=demo_image,
    secrets=[secret],
    volumes=mounts,
    gpu="A100-80GB",
    memory=65536,
    ephemeral_disk=524288,
    timeout=6 * 60 * 60,
)
def released_demo() -> dict:
    """Run the author-linked demo unchanged and preserve its raw output."""
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(REMOTE_CACHE / "huggingface"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    output_dir = REMOTE_CACHE / "results" / "neural-chameleons" / "released-demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "released_demo_stdout.log"

    process = subprocess.Popen(
        [sys.executable, "demo.py"],
        cwd=REMOTE_MINIMAL,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with stdout_path.open("w") as output:
        for line in process.stdout:
            print(line, end="")
            output.write(line)
            output.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, [sys.executable, "demo.py"])

    volume.commit()
    result = {
        "command": f"{sys.executable} demo.py",
        "upstream_script": str(REMOTE_MINIMAL / "demo.py"),
        "stdout": str(stdout_path),
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=plot_image,
    volumes=mounts,
    cpu=4,
    memory=16384,
    timeout=60 * 60,
)
def paper_plots(plot_type: str = "scatter") -> dict:
    """Regenerate plots from the result files shipped in the official repo."""
    valid = {
        "all",
        "scatter",
        "violin",
        "lmeval",
        "multiprobe",
        "probe-types",
        "cosine",
        "cosine-ensemble",
        "cosine-ensemble-avg",
    }
    if plot_type not in valid:
        raise ValueError(f"Unknown plot type {plot_type!r}; choose from {sorted(valid)}")

    output_dir = (
        REMOTE_CACHE / "results" / "neural-chameleons" / "paper-plots" / plot_type
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.self_obfuscation_pipeline.generate_paper_plots",
            "--plot-type",
            plot_type,
            "--output-folder",
            str(output_dir),
        ],
        cwd=REMOTE_OFFICIAL,
        check=True,
    )
    files = sorted(str(path) for path in output_dir.rglob("*") if path.is_file())
    volume.commit()
    result = {"plot_type": plot_type, "files": files}
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(stage: str = "inspect", plot_type: str = "scatter") -> None:
    if stage == "inspect":
        print(inspect_release.remote())
    elif stage == "released-demo":
        print(released_demo.remote())
    elif stage == "paper-plots":
        print(paper_plots.remote(plot_type))
    else:
        raise ValueError("stage must be inspect, released-demo, or paper-plots")
