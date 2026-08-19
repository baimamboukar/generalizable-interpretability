from __future__ import annotations

from pathlib import Path
import os
import tarfile
from unittest.mock import Mock, patch

from infra.vast import VastRunner


def test_archive_preserves_upstream_source_tree(tmp_path: Path) -> None:
    root = tmp_path / "project"
    upstream = root / "external" / "upstream"
    (root / "infra").mkdir(parents=True)
    (root / "infra" / "runner.py").write_text("pass\n")
    for relative in ("data/example.csv", "docs/readme.md", "figures/main.png"):
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    for relative in (".git/config", "__pycache__/cache.pyc", "results/run.pkl"):
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)

    archive = tmp_path / "project.tar.gz"
    VastRunner._archive(root, archive, ["upstream"])

    with tarfile.open(archive) as bundle:
        names = set(bundle.getnames())
    assert "external/upstream/data/example.csv" in names
    assert "external/upstream/docs/readme.md" in names
    assert "external/upstream/figures/main.png" in names
    assert not any(".git" in name for name in names)
    assert not any("__pycache__" in name for name in names)
    assert not any("results" in name for name in names)


def test_launch_does_not_return_instance_credentials(tmp_path: Path) -> None:
    runner = VastRunner()
    runner.client = Mock()
    runner.client.search_offers.return_value = [
        {
            "id": 7,
            "gpu_name": "H100 NVL",
            "dph_total": 2.0,
        }
    ]
    runner.client.create_instance.return_value = {
        "success": True,
        "new_contract": 42,
        "instance_api_key": "secret",
    }
    with (
        patch.dict(os.environ, {"HF_BUCKET": "hf://buckets/test/bucket"}),
        patch.object(runner, "_archive"),
        patch.object(runner, "_script", return_value="true"),
        patch.object(__import__("infra.vast", fromlist=["subprocess"]).subprocess, "run"),
    ):
        result = runner.launch(
            root=tmp_path,
            command=["true"],
            external=[],
            gpu="H100_NVL",
            min_vram=75,
            min_cpu=16,
            min_ram=64,
            min_cuda=12.6,
            max_price=3,
            disk=150,
            image="image",
            label="test",
            env_keys=[],
        )
    assert result["new_contract"] == 42
    assert "instance_api_key" not in result
