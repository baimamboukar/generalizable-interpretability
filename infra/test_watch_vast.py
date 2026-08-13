from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infra import watch_vast


def artifact_result(stage: str, seen: list[str]):
    def run(command, **_kwargs):
        seen.append(command[-2])
        Path(command[-1]).write_text(f'{{"stage": "{stage}"}}')
        return SimpleNamespace(returncode=0)

    return run


def test_hf_artifact_requires_terminal_stage() -> None:
    seen: list[str] = []
    artifact = "artifacts/internal-instruction-following/phi3-mini/run.json"
    with (
        patch.dict(os.environ, {"HF_BUCKET": "hf://buckets/org/bucket"}),
        patch.object(
            watch_vast.subprocess,
            "run",
            side_effect=artifact_result("prepared", seen),
        ),
    ):
        assert not watch_vast.hf_artifact_ready(artifact)
    with (
        patch.dict(os.environ, {"HF_BUCKET": "hf://buckets/org/bucket"}),
        patch.object(
            watch_vast.subprocess,
            "run",
            side_effect=artifact_result("trained", seen),
        ),
    ):
        assert watch_vast.hf_artifact_ready(artifact)
    assert all("/artifacts/" not in target for target in seen)


def test_destroy_verifies_instance_removal(tmp_path: Path) -> None:
    client = Mock()
    client.show_instance.side_effect = [{"id": 42}, None]
    with patch.object(watch_vast.time, "sleep"):
        assert watch_vast.destroy(client, 42, tmp_path / "watch.log")
    assert client.show_instance.call_count == 2
