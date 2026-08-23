from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "confirmatory.py"
SPEC = importlib.util.spec_from_file_location("political_confirmatory", MODULE_PATH)
assert SPEC and SPEC.loader
confirmatory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = confirmatory
SPEC.loader.exec_module(confirmatory)

analyze = confirmatory.analyze
assert_feature_compatibility = confirmatory.assert_feature_compatibility
audit_panel = confirmatory.audit_panel
batch_candidate_predictions = confirmatory.batch_candidate_predictions
cluster_bootstrap = confirmatory.cluster_bootstrap
equal_budget_candidates = confirmatory.equal_budget_candidates
full_pipeline_bootstrap = confirmatory.full_pipeline_bootstrap
independent_news_predictions = confirmatory.independent_news_predictions
nested_predictions = confirmatory.nested_predictions
rank_candidates = confirmatory.rank_candidates
rowwise_spearman = confirmatory.rowwise_spearman
sha256 = confirmatory.sha256
validate_analysis_args = confirmatory.validate_analysis_args


def synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(60, 4, 3, 3))
    labels = features[:, 2, 1, 0] * 1.8 + features[:, 2, 1, 1] * -0.4
    labels += rng.normal(scale=0.05, size=len(labels))
    identifiers = [f"p-{index:03d}" for index in range(len(labels))]
    return features, labels, identifiers


def test_equal_budget_candidates_are_deterministic_and_layer_balanced():
    first = equal_budget_candidates(40, 32, 512, 42)
    second = equal_budget_candidates(40, 32, 512, 42)
    assert first == second
    assert len(first) == len(set(first)) == 512
    counts = np.bincount([layer for layer, _ in first], minlength=40)
    assert counts.min() >= 12
    assert counts.max() <= 13


def test_nested_predictions_are_complete_and_select_signal():
    features, labels, identifiers = synthetic()
    rows, selections = nested_predictions(
        features,
        labels,
        identifiers,
        identifiers,
        equal_budget_candidates(4, 3, 12, 42),
        top_k=3,
        outer_folds=3,
        inner_folds=2,
        seed=42,
        alpha=1.0,
    )
    assert len(rows) == len(labels) * 2
    for method in ("top1", "top3"):
        subset = [row for row in rows if row["selection"] == method]
        assert sorted(row["row_index"] for row in subset) == list(range(len(labels)))
    fold_winners = [
        row for row in selections if row["rank"] == 1
    ]
    assert all((row["layer"], row["head"]) == (2, 1) for row in fold_winners)


def test_news_predictions_do_not_use_news_labels_for_selection():
    features, labels, identifiers = synthetic()
    news_features, news_labels, news_ids = synthetic(seed=11)
    candidates = equal_budget_candidates(4, 3, 12, 42)
    first_rows, first_selection = independent_news_predictions(
        features,
        labels,
        identifiers,
        news_features,
        news_labels,
        news_ids,
        news_ids,
        candidates,
        top_k=3,
        inner_folds=3,
        seed=42,
        alpha=1.0,
    )
    second_rows, second_selection = independent_news_predictions(
        features,
        labels,
        identifiers,
        news_features,
        news_labels[::-1],
        news_ids,
        news_ids,
        candidates,
        top_k=3,
        inner_folds=3,
        seed=42,
        alpha=1.0,
    )
    assert first_selection == second_selection
    assert [row["prediction"] for row in first_rows] == [
        row["prediction"] for row in second_rows
    ]
    assert [row["rank"] for row in first_selection] == list(
        range(1, len(candidates) + 1)
    )
    assert sum(row["selected_top_k"] for row in first_selection) == 3


def test_ranking_uses_mean_foldwise_spearman():
    features, labels, identifiers = synthetic()
    candidates = [(2, 1)]
    folds = confirmatory.grouped_folds(identifiers, 3, 17)
    expected = []
    for train, test in folds:
        predictions = confirmatory.fit_predict(
            features, labels, train, features[test], candidates, 1.0
        )
        expected.append(confirmatory.regression_metrics(labels[test], predictions)["spearman"])
    ranking = rank_candidates(
        features, labels, np.asarray(identifiers), candidates, 3, 17, 1.0
    )
    assert np.isclose(ranking[0]["selection_spearman_mean"], np.mean(expected))
    assert ranking[0]["selection_valid_folds"] == 3


def test_batched_candidate_ridge_matches_independent_fits():
    features, labels, identifiers = synthetic()
    candidates = equal_budget_candidates(4, 3, 8, 9)
    train, test = confirmatory.grouped_folds(identifiers, 3, 17)[0]
    batched = batch_candidate_predictions(
        features, labels, train, features[test], candidates, 1.0
    )
    independent = np.stack(
        [
            confirmatory.fit_predict(
                features, labels, train, features[test], [candidate], 1.0
            )
            for candidate in candidates
        ]
    )
    assert np.allclose(batched, independent)
    expected = np.array(
        [
            confirmatory.regression_metrics(labels[test], values)["spearman"]
            for values in independent
        ]
    )
    assert np.allclose(rowwise_spearman(labels[test], batched), expected)


def test_cluster_bootstrap_is_reproducible():
    rng = np.random.default_rng(4)
    labels = rng.normal(size=40)
    predictions = labels + rng.normal(scale=0.2, size=40)
    groups = np.array([f"g-{index // 2}" for index in range(40)])
    first = cluster_bootstrap(labels, predictions, groups, 100, 42)
    second = cluster_bootstrap(labels, predictions, groups, 100, 42)
    assert first == second
    for metric in first.values():
        assert metric["ci_low"] <= metric["estimate"] <= metric["ci_high"]


def test_full_pipeline_bootstrap_refits_selection():
    features, labels, identifiers = synthetic()
    news_features, news_labels, news_ids = synthetic(seed=11)
    rows = full_pipeline_bootstrap(
        features,
        labels,
        identifiers,
        identifiers,
        news_features,
        news_labels,
        news_ids,
        news_ids,
        equal_budget_candidates(4, 3, 6, 0),
        top_k=2,
        outer_folds=3,
        inner_folds=2,
        replicates=2,
        seed=42,
        alpha=1.0,
    )
    assert len(rows) == 16
    assert all(row["uncertainty_type"] == "full_pipeline_cluster_bootstrap" for row in rows)
    assert all(row["bootstrap_replicates"] == 2 for row in rows)


def test_analyze_retains_predictions_metrics_and_provenance(tmp_path, monkeypatch):
    features, labels, identifiers = synthetic()
    news_features, news_labels, news_ids = synthetic(seed=11)
    root = tmp_path / "extensions"
    model = root / "qwen3-8b"
    model.mkdir(parents=True)
    common = {
        "model": "example/model",
        "requested_revision": "a" * 40,
        "resolved_revision": "a" * 40,
        "activation_site": "test hook",
        "activation_lineage": "final prompt-token pre-projection attention-head outputs",
        "prompt_protocol": "released paper prompts v1",
        "prompt_template": "test prompt",
        "head_index_base": 0,
        "dataset_sha256": "b" * 64,
    }
    torch.save(
        {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "sample_ids": identifiers,
            "cluster_ids": identifiers,
            "dataset": "HS116_members",
            **common,
        },
        model / "features.pt",
    )
    torch.save(
        {
            "features": torch.from_numpy(news_features),
            "labels": torch.from_numpy(news_labels),
            "sample_ids": news_ids,
            "cluster_ids": news_ids,
            "dataset": "adfontesmedia",
            **common,
        },
        model / "news_features.pt",
    )
    monkeypatch.setattr(confirmatory, "ARTIFACTS", root)
    output = analyze(
        "qwen3-8b",
        candidate_budget=12,
        top_k=3,
        outer_folds=3,
        inner_folds=2,
        bootstrap_replicates=20,
        seed=42,
        alpha=1.0,
        candidate_seeds=[0, 1],
        pipeline_bootstrap_replicates=2,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["resolved_revision"] == "a" * 40
    assert manifest["selection"]["news_labels_used_for_selection"] is False
    assert manifest["selection"]["candidate_budget"] == 12
    assert manifest["selection"]["candidate_seeds"] == [0, 1]
    assert manifest["selection"]["selection_metric"] == "mean fold-wise Spearman correlation"
    assert manifest["confirmatory_protocol"]["sha256"] == sha256(
        Path(confirmatory.__file__).with_name("confirmatory_protocol.json")
    )
    assert manifest["confirmatory_protocol"]["overrides"]["candidate_budget"] == 12
    assert (output / "row_predictions.csv").read_text().count("\n") == 481
    assert (output / "pipeline_uncertainty.csv").exists()
    metrics = (output / "metrics.csv").read_text()
    for metric in ("spearman", "pearson", "mae", "rmse"):
        assert f",{metric}," in metrics
    assert "n_rows,n_clusters" in metrics

    second = root / "second-model" / "confirmatory"
    shutil.copytree(output, second)
    for filename in (
        "row_predictions.csv",
        "nested_selections.csv",
        "news_selection.csv",
        "metrics.csv",
        "pipeline_uncertainty.csv",
    ):
        frame = pd.read_csv(second / filename)
        frame["model_key"] = "second-model"
        frame["model"] = "example/second-model"
        frame.to_csv(second / filename, index=False)
    second_manifest = json.loads((second / "manifest.json").read_text())
    second_manifest["model_key"] = "second-model"
    second_manifest["model"] = "example/second-model"
    second_manifest["outputs"] = {
        filename: sha256(second / filename)
        for filename in second_manifest["outputs"]
    }
    (second / "manifest.json").write_text(json.dumps(second_manifest, indent=2) + "\n")
    audit_panel([output / "manifest.json", second / "manifest.json"])
    second_manifest["confirmatory_protocol"]["sha256"] = "c" * 64
    (second / "manifest.json").write_text(json.dumps(second_manifest, indent=2) + "\n")
    with np.testing.assert_raises(RuntimeError):
        audit_panel([output / "manifest.json", second / "manifest.json"])


def test_compatibility_and_argument_validation_fail_fast():
    common = {
        "model": "example/model",
        "requested_revision": "a" * 40,
        "resolved_revision": "a" * 40,
        "activation_site": "test hook",
        "activation_lineage": "final prompt-token pre-projection attention-head outputs",
        "prompt_protocol": "released paper prompts v1",
        "prompt_template": "test prompt",
        "head_index_base": 0,
    }
    politician = {**common, "dataset": "HS116_members"}
    news = {**common, "dataset": "adfontesmedia", "resolved_revision": "c" * 40}
    with np.testing.assert_raises(ValueError):
        assert_feature_compatibility(politician, news, (10, 4, 3, 3), (10, 4, 3, 3))
    with np.testing.assert_raises(ValueError):
        validate_analysis_args(4, 5, 3, 2, 10, 2, 42, 1.0, [0, 1, 2])
