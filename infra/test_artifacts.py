from pathlib import Path
import os
from unittest.mock import patch

from infra import artifacts


def test_bucket_check_uses_supported_cli() -> None:
    with (
        patch.dict(os.environ, {"HF_BUCKET": "hf://buckets/org/results"}),
        patch.object(artifacts.subprocess, "run") as run,
    ):
        artifacts.check()

    run.assert_called_once_with(
        ["hf", "buckets", "info", "org/results"], check=True
    )


def test_bucket_sync_respects_feature_gate(tmp_path: Path) -> None:
    with (
        patch.dict(
            os.environ,
            {
                "HF_BUCKET": "hf://buckets/org/results",
                "HF_SYNC_ENABLED": "true",
            },
        ),
        patch.object(artifacts.subprocess, "run") as run,
    ):
        artifacts.sync(tmp_path)

    run.assert_called_once_with(
        ["hf", "buckets", "sync", str(tmp_path), "hf://buckets/org/results"],
        check=True,
    )
