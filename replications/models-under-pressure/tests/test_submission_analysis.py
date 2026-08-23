from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "submission_analysis.py"
SPEC = importlib.util.spec_from_file_location("mup_submission_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)

WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "mup_vast_wrapper", MODULE_PATH.parent / "vast_mup.py"
)
assert WRAPPER_SPEC and WRAPPER_SPEC.loader
wrapper = importlib.util.module_from_spec(WRAPPER_SPEC)
sys.modules[WRAPPER_SPEC.name] = wrapper
WRAPPER_SPEC.loader.exec_module(wrapper)


def test_activation_hook_captures_pre_layer_norm_input() -> None:
    cache = {}
    value = Mock()
    detached = value.detach.return_value
    expected = detached.cpu.return_value

    wrapper.pre_layer_norm_hook(cache, 7)(None, (value,), Mock())

    assert cache[7] is expected


def test_olmo2_adapter_targets_decoder_block_input() -> None:
    class Layers:
        def __init__(self):
            self.layers = [Mock(), Mock()]

    network = Mock()
    network.model = Layers()
    args = Mock(architecture_adapter="olmo2_decoder_block_input")
    registry = type("Registry", (), {"_architectures": {}})
    base = type("Architecture", (), {})

    wrapper.register_architecture_adapter(args, registry, base)
    architecture = registry._architectures["olmo2"]()
    assert architecture.get_layer_norm(network, 1) is network.model.layers[1]
    assert architecture.get_layers(network) is network.model.layers


def test_holdout_plan_is_frozen_without_running_task_data(capsys) -> None:
    protocol = MODULE_PATH.parent / "holdout_protocol.json"
    original = sys.argv
    try:
        sys.argv = [
            "vast_mup.py",
            "--stage",
            "plan",
            "--protocol",
            str(protocol),
            "--model-key",
            "olmo2-32b",
        ]
        wrapper.main()
    finally:
        sys.argv = original

    plan = json.loads(capsys.readouterr().out)
    assert plan["layers"] == [22, 25, 30, 38, 46, 54]
    assert plan["architecture_adapter"] == "olmo2_decoder_block_input"
    assert "decoder block input" in plan["activation_site"]


def test_common_metrics() -> None:
    metrics = analysis.calculate_metrics(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.4, 0.35, 0.8],
        threshold=0.5,
    )

    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx(5 / 6)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 1, 1)
    assert metrics["brier"] == pytest.approx(0.158125)


def test_threshold_is_fitted_on_development_labels() -> None:
    threshold = analysis.choose_threshold(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.2, 0.8, 0.9],
    )

    assert threshold == 0.8


def test_low_fpr_thresholds_are_selected_on_development_data() -> None:
    labels = [0] * 100 + [1, 1]
    scores = [index / 100 for index in range(100)] + [0.995, 0.98]

    thresholds = analysis.development_thresholds(labels, scores)
    one_percent = analysis.operating_rates(labels, scores, thresholds["fpr_0_01"])
    five_percent = analysis.operating_rates(labels, scores, thresholds["fpr_0_05"])

    assert one_percent == pytest.approx((0.5, 0.0))
    assert five_percent == pytest.approx((1.0, 0.02))


def _record(split: str, layer: int, labels: list[int], scores: list[float]) -> object:
    return analysis.ResultRecord(
        split=split,
        model="model",
        probe="attention",
        layer=layer,
        seed=41,
        dataset="dataset",
        ids=tuple(str(index) for index in range(len(labels))),
        labels=tuple(labels),
        scores=tuple(scores),
        config_id=f"run_attention_L{layer}_seed41",
        source=Path("result.jsonl"),
    )


def test_oracle_is_separate_from_dev_selection() -> None:
    records = [
        _record("dev", 1, [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
        _record("dev", 2, [0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8]),
        _record("test", 1, [0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8]),
        _record("test", 2, [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
    ]
    protocol = {
        "probes": ["attention"],
        "seeds": [41],
        "dev_datasets": ["dataset"],
        "test_datasets": ["dataset"],
        "selection": {"relative_depth": 0.5},
        "models": {
            "model": {
                "num_hidden_layers": 4,
                "candidate_layers": [1, 2],
                "sparse_grid_layers": [1, 2],
            }
        },
    }

    selected = analysis._select_layers(records, protocol, "model")
    by_strategy = {row["strategy"]: row for row in selected}

    assert by_strategy["dev_selected"]["layer"] == 1
    assert by_strategy["dev_selected"]["diagnostic_only"] is False
    assert by_strategy["oracle_test"]["layer"] == 2
    assert by_strategy["oracle_test"]["diagnostic_only"] is True


def test_protocol_requires_revisions_and_three_seeds(tmp_path: Path) -> None:
    protocol = json.loads((MODULE_PATH.parent / "submission_protocol.json").read_text())
    protocol["seeds"] = [42]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))

    with pytest.raises(ValueError, match="at least three seeds"):
        analysis.load_protocol(path)


def test_end_to_end_analysis_retains_rows_and_marks_oracle(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    protocol = {
        "schema_version": 1,
        "analysis_id": "test",
        "probes": ["attention"],
        "seeds": [41, 42, 43],
        "activation_site": "site",
        "upstream_source_tree": {"sha256": "c" * 64, "file_count": 1},
        "dataset_sha256": {"data.jsonl": "b" * 64},
        "selection": {
            "objective": "macro_mean_auroc",
            "relative_depth": 0.5,
            "threshold_objective": "balanced_accuracy",
            "oracle_is_diagnostic_only": True,
        },
        "uncertainty": {
            "bootstrap_replicates": 20,
            "confidence_level": 0.95,
            "random_seed": 1,
            "cluster_key": "dataset_and_id",
        },
        "dev_datasets": ["dataset"],
        "test_datasets": ["dataset"],
        "models": {
            "model": {
                "model_id": "model",
                "revision": "a" * 40,
                "model_config": "model",
                "num_hidden_layers": 4,
                "candidate_layers": [1, 2],
                "sparse_grid_layers": [1, 2],
            }
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    (artifact / "submission_protocol.json").write_bytes(protocol_path.read_bytes())
    (artifact / "run.json").write_text(
        json.dumps(
            {
                "model": "model",
                "model_revision": "a" * 40,
                "model_config": "model",
                "layers": [1, 2],
                "seeds": [41, 42, 43],
                "probes": ["attention"],
                "activation_site": "site",
                "dataset_sha256": protocol["dataset_sha256"],
                "protocol_sha256": protocol_hash,
                "upstream_source_tree": {"sha256": "c" * 64, "file_count": 1},
                "activation_provenance_sha256": "d" * 64,
                "activation_provenance_rows": 2,
            }
        )
    )
    labels = [0, 0, 1, 1]
    for seed in protocol["seeds"]:
        for split in ("dev", "test"):
            for layer in (1, 2):
                perfect = (split == "dev" and layer == 1) or (split == "test" and layer == 2)
                scores = [0.1, 0.2, 0.8, 0.9] if perfect else [0.1, 0.9, 0.2, 0.8]
                row = {
                    "config": {
                        "id": f"run_attention_L{layer}_seed{seed}",
                        "model_name": "model",
                        "layer": layer,
                        "probe_spec": {"name": "attention"},
                    },
                    "dataset_name": "dataset",
                    "dataset_path": f"data/evals/{split}/dataset.jsonl",
                    "output_scores": scores,
                    "ground_truth_labels": labels,
                    "ids": ["a", "b", "c", "d"],
                }
                path = artifact / f"results_run_{split}_attention_L{layer}_seed{seed}.jsonl"
                path.write_text(json.dumps(row) + "\n")

    output = tmp_path / "output"
    manifest = analysis.analyze(artifact, output, protocol_path, "model")

    assert manifest["coverage"]["complete"] is True
    selection_text = (output / "selections.csv").read_text()
    assert "oracle_test" in selection_text
    assert "True" in selection_text
    assert (output / "selection_aggregate.csv").exists()
    assert (output / "metrics.csv").exists()
    assert len((output / "rows.jsonl").read_text().splitlines()) == 48
    first_row = json.loads((output / "rows.jsonl").read_text().splitlines()[0])
    assert first_row["probability"] == first_row["score"]
    assert first_row["threshold_source"] == "pooled_development_balanced_accuracy"
    assert (output / "seed_aggregate.csv").exists()
    assert (output / "full_pipeline_seed.csv").exists()
    assert (output / "full_pipeline_aggregate.csv").exists()
    assert "uncertainty_note" in (output / "seed_aggregate.csv").read_text().splitlines()[0]
    assert "hierarchical_ci_low" in (
        output / "full_pipeline_aggregate.csv"
    ).read_text().splitlines()[0]


def test_result_completion_requires_each_dev_and_test_dataset_once(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "config": {
                "id": "run_attention_L7_seed41",
                "model_name": "model",
                "layer": 7,
                "probe_spec": {"name": "attention"},
            },
            "dataset_name": dataset.stem,
            "dataset_path": str(dataset.relative_to(wrapper.PROJECT)),
            "output_scores": [0.2, 0.8],
            "ground_truth_labels": [0, 1],
            "ids": ["a", "b"],
        }
        for dataset in wrapper.EVALS
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert wrapper.result_is_complete(
        path,
        model="model",
        layer=7,
        probe="attention",
        config_id="run_attention_L7_seed41",
    ) is True

    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    assert wrapper.result_is_complete(
        path,
        model="model",
        layer=7,
        probe="attention",
        config_id="run_attention_L7_seed41",
    ) is False


def test_result_completion_rejects_stale_config(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "config": {
                "id": "run_attention_L7_seed41",
                "model_name": "wrong-model",
                "layer": 7,
                "probe_spec": {"name": "attention"},
            },
            "dataset_name": dataset.stem,
            "dataset_path": str(dataset.relative_to(wrapper.PROJECT)),
            "output_scores": [0.2, 0.8],
            "ground_truth_labels": [0, 1],
            "ids": ["a", "b"],
        }
        for dataset in wrapper.EVALS
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert wrapper.result_is_complete(
        path,
        model="model",
        layer=7,
        probe="attention",
        config_id="run_attention_L7_seed41",
    ) is False


def test_artifact_contract_rejects_revision_mismatch(tmp_path: Path) -> None:
    protocol = json.loads((MODULE_PATH.parent / "submission_protocol.json").read_text())
    model_key = "qwen3-32b"
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    (artifact / "submission_protocol.json").write_bytes(protocol_path.read_bytes())
    model = protocol["models"][model_key]
    (artifact / "run.json").write_text(
        json.dumps(
            {
                "model": model["model_id"],
                "model_revision": "f" * 40,
                "model_config": model["model_config"],
                "layers": model["candidate_layers"],
                "seeds": protocol["seeds"],
                "probes": protocol["probes"],
                "activation_site": protocol["activation_site"],
                "dataset_sha256": protocol["dataset_sha256"],
                "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "upstream_source_tree": {"sha256": "c" * 64, "file_count": 1},
                "activation_provenance_sha256": "d" * 64,
                "activation_provenance_rows": len(protocol["dataset_sha256"])
                * len(model["candidate_layers"]),
            }
        )
    )

    with pytest.raises(ValueError, match="model_revision"):
        analysis.validate_artifact_contract(
            artifact, protocol, protocol_path, model_key
        )


def test_source_tree_digest_changes_with_source(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "pyproject.toml").write_text("name = 'x'\n")
    source = tmp_path / "src" / "module.py"
    source.write_text("VALUE = 1\n")
    first = wrapper.source_tree_digest(tmp_path)
    source.write_text("VALUE = 2\n")
    second = wrapper.source_tree_digest(tmp_path)

    assert first["sha256"] != second["sha256"]


def test_activation_restore_contract_rejects_incomplete_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "model": "model",
        "model_revision": "a" * 40,
        "dataset": "data.jsonl",
        "dataset_sha256": "b" * 64,
        "layer": 7,
        "activation_site": wrapper.ACTIVATION_SITE,
    }
    monkeypatch.setattr(wrapper, "expected_activation_provenance", lambda _args: [expected])
    (tmp_path / "manifest.json").write_text('{"rows": []}\n')
    (tmp_path / "submission_provenance.json").write_text(
        json.dumps({"schema_version": 1, "rows": []})
    )

    with pytest.raises(ValueError, match="incomplete or stale"):
        wrapper.validate_activation_provenance(object(), tmp_path)

    (tmp_path / "submission_provenance.json").write_text(
        json.dumps({"schema_version": 1, "rows": [expected]})
    )
    wrapper.validate_activation_provenance(object(), tmp_path)
