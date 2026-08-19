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
    with (
        patch.dict(os.environ, {"HF_BUCKET": "hf://buckets/org/bucket"}),
        patch.object(
            watch_vast.subprocess,
            "run",
            side_effect=artifact_result("test_extracted", seen),
        ),
    ):
        assert not watch_vast.hf_artifact_ready(artifact)
    assert all("/artifacts/" not in target for target in seen)


def test_destroy_verifies_instance_removal(tmp_path: Path) -> None:
    client = Mock()
    client.show_instance.side_effect = [{"id": 42}, None]
    with patch.object(watch_vast.time, "sleep"):
        assert watch_vast.destroy(client, 42, tmp_path / "watch.log")
    assert client.show_instance.call_count == 2


def test_normalize_logs_accepts_structured_responses() -> None:
    assert watch_vast.normalize_logs(None) == ""
    assert watch_vast.normalize_logs("ready") == "ready"
    assert watch_vast.normalize_logs({"status": "ready"}) == '{"status": "ready"}'


def test_phase_reports_layer_extraction_progress() -> None:
    logs = "train-all-layers:  37%|###7      | 8000/21690"
    assert "37%" in watch_vast.phase(logs)


def test_monitor_finishes_when_instance_is_absent(tmp_path: Path) -> None:
    client = Mock()
    client.show_instance.return_value = None
    args = SimpleNamespace(
        artifact=[],
        hf_artifact=[],
        instance=[42],
        interval_minutes=1,
        log=tmp_path / "watch.log",
        max_checks=None,
        max_idle_checks=None,
        once=False,
    )
    with (
        patch.dict(os.environ, {"VAST_API_KEY": "test"}),
        patch.object(watch_vast, "VastAI", return_value=client),
        patch.object(watch_vast, "load_project_env"),
    ):
        watch_vast.monitor(args)
    assert "destroyed=true" in args.log.read_text()


def test_monitor_stops_a_restarted_terminal_job(tmp_path: Path) -> None:
    client = Mock()
    client.show_instance.return_value = {"id": 42, "actual_status": "running"}
    client.logs.return_value = "results saved\nSync completed."
    args = SimpleNamespace(
        artifact=[],
        hf_artifact=[],
        instance=[42],
        interval_minutes=1,
        log=tmp_path / "watch.log",
        max_checks=None,
        max_idle_checks=None,
        once=False,
    )
    with (
        patch.dict(os.environ, {"VAST_API_KEY": "test"}),
        patch.object(watch_vast, "VastAI", return_value=client),
        patch.object(watch_vast, "load_project_env"),
        patch.object(watch_vast, "destroy", return_value=True) as destroy,
    ):
        watch_vast.monitor(args)
    destroy.assert_called_once_with(client, 42, args.log)
    assert "terminal_sync=true success=True" in args.log.read_text()


def test_monitor_stops_only_on_complete_hf_artifact(tmp_path: Path) -> None:
    client = Mock()
    client.show_instance.return_value = {"id": 42, "actual_status": "running"}
    client.logs.return_value = "extracting"
    args = SimpleNamespace(
        artifact=[],
        hf_artifact=["42=run.json"],
        instance=[42],
        interval_minutes=1,
        log=tmp_path / "watch.log",
        max_checks=None,
        max_idle_checks=None,
        once=False,
    )
    with (
        patch.dict(os.environ, {"VAST_API_KEY": "test"}),
        patch.object(watch_vast, "VastAI", return_value=client),
        patch.object(watch_vast, "load_project_env"),
        patch.object(watch_vast, "hf_artifact_ready", return_value=True),
        patch.object(watch_vast, "destroy", return_value=True) as destroy,
    ):
        watch_vast.monitor(args)
    destroy.assert_called_once_with(client, 42, args.log)
    assert "artifact_ready=true" in args.log.read_text()
