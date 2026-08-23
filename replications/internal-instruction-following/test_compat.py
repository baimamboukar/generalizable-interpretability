from __future__ import annotations

import csv
import os
import pickle
import tempfile
import unittest

import compat
import numpy as np
import submission
import torch


class CompatibilityTest(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        metrics = compat.classification_metrics(
            torch.tensor([0, 0, 1, 1]), torch.tensor([0.1, 0.8, 0.7, 0.2])
        )
        self.assertEqual(
            {key: metrics[key] for key in ("tn", "fp", "fn", "tp")},
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )
        for key in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall_tpr",
            "specificity_tnr",
            "f1",
        ):
            self.assertEqual(metrics[key], 0.5)

    def test_submission_metrics_and_frozen_threshold(self) -> None:
        labels = [0, 0, 1, 1]
        probabilities = [0.1, 0.4, 0.6, 0.9]
        thresholds = submission.frozen_thresholds(labels, probabilities)
        metrics = submission.metric_values(labels, probabilities, thresholds)
        self.assertEqual(set(metrics), set(submission.METRICS))
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["auprc"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["brier"], 0.085)
        self.assertEqual(
            metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"], 4
        )

    def test_submission_low_fpr_thresholds_are_development_frozen(self) -> None:
        labels = [0] * 100 + [1, 1]
        probabilities = [index / 100 for index in range(100)] + [0.995, 0.98]
        thresholds = submission.frozen_thresholds(labels, probabilities)
        metrics = submission.metric_values(labels, probabilities, thresholds)

        self.assertEqual(metrics["test_tpr_at_dev_fpr_0_01_threshold"], 0.5)
        self.assertEqual(metrics["test_fpr_at_dev_fpr_0_01_threshold"], 0.0)
        self.assertEqual(metrics["test_tpr_at_dev_fpr_0_05_threshold"], 1.0)
        self.assertEqual(metrics["test_fpr_at_dev_fpr_0_05_threshold"], 0.02)

    def test_weighted_bootstrap_metrics_match_repeated_rows(self) -> None:
        labels = np.array([0, 0, 1, 1, 0, 1])
        probabilities = np.array([0.1, 0.8, 0.7, 0.2, 0.4, 0.9])
        thresholds = submission.frozen_thresholds(labels, probabilities)
        weights = np.array([[2, 0, 1, 3, 1, 2], [1, 2, 2, 1, 3, 1]])
        draws = submission.weighted_metric_draws(
            labels, probabilities, weights, thresholds
        )
        for row, weight in enumerate(weights):
            indices = np.repeat(np.arange(len(labels)), weight)
            expected = submission.metric_values(
                labels[indices], probabilities[indices], thresholds
            )
            for metric, value in expected.items():
                if value != "":
                    self.assertAlmostEqual(float(draws[metric][row]), float(value))
        intervals = submission.cluster_bootstrap(
            labels, probabilities, np.arange(len(labels)) % 3, thresholds, 10, 1
        )
        self.assertEqual({row["metric"] for row in intervals}, set(submission.METRICS))

    def test_normalization_control_is_leakage_free(self) -> None:
        train = torch.tensor([[0.0], [2.0], [4.0]])
        development = torch.tensor([[10.0], [12.0]])
        test = torch.tensor([[20.0], [22.0]])
        released = submission.normalize_partitions(
            train, development, test, "released_splitwise"
        )
        train_fitted = submission.normalize_partitions(
            train, development, test, "train_fitted"
        )
        self.assertAlmostEqual(float(released[2].mean()), 0.0)
        self.assertGreater(float(train_fitted[2].mean()), 5.0)

    def test_extension_layer_configuration(self) -> None:
        compat.validate_extension_models()
        compat.validate_control_models()
        expected = {
            "glm4.7-flash": [20, 29, 38, 47],
            "qwen3-14b": [16, 24, 32, 40],
            "gemma4-12b": [21, 30, 39, 48],
            "gpt-oss-20b": [9, 14, 19, 24],
        }
        self.assertEqual(
            {
                key: model["layers"]
                for key, model in compat.MODELS.items()
                if model.get("scope") == "extension"
            },
            expected,
        )
        source = (compat.HERE / "runtime" / "upstream_entrypoint.py").read_text()
        self.assertIn("GI_TARGET_LAYERS", source)

    def test_crossed_controls_are_complete(self) -> None:
        for base_model in ("glm4.7-flash", "qwen3-14b", "gemma4-12b"):
            controls = [
                model
                for model in compat.CONFIG["controls"]
                if model["base_model"] == base_model
            ]
            self.assertEqual(len(controls), 4)
            self.assertEqual(
                {
                    (
                        model["prompt_mode"],
                        "fixed"
                        if model["response_policy"].startswith("fixed")
                        else "on_policy",
                    )
                    for model in controls
                },
                {
                    ("released_llama2", "fixed"),
                    ("released_llama2", "on_policy"),
                    ("native", "fixed"),
                    ("native", "on_policy"),
                },
            )

    def test_factorial_control_contrasts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("GI_ARTIFACT_DIR")
            os.environ["GI_ARTIFACT_DIR"] = directory
            try:
                controls = [
                    model
                    for model in compat.CONFIG["controls"]
                    if model["base_model"] == "qwen3-14b"
                ]
                for model in controls:
                    prompt_effect = 0.1 if model["prompt_mode"] == "native" else 0.0
                    policy_effect = (
                        0.2 if model["response_policy"].startswith("fixed") else 0.0
                    )
                    rows = []
                    for normalization, normalization_effect in (
                        ("released_splitwise", 0.0),
                        ("train_fitted", 0.05),
                    ):
                        value = 0.5 + prompt_effect + policy_effect + normalization_effect
                        rows.append(
                            {
                                "experiment": "task_generalization",
                                "token_position": "last",
                                "split": "0",
                                "heldout_instruction": "",
                                "normalization": normalization,
                                **{metric: value for metric in submission.METRICS},
                            }
                        )
                    path = submission.output_paths(model)["runs"]
                    submission.write_csv(path, rows)
                result = submission.compare_controls("qwen3-14b")
                self.assertGreater(result["contrast_rows"], 0)
                with open(result["summary"], newline="") as handle:
                    summary = list(csv.DictReader(handle))
                effects = {
                    row["contrast"]: float(row["mean_delta"])
                    for row in summary
                    if row["metric"] == "auroc"
                    and row["normalization"] == "train_fitted"
                }
                self.assertAlmostEqual(effects["response_policy"], 0.2)
                self.assertAlmostEqual(effects["prompt_template"], 0.1)
                self.assertAlmostEqual(effects["prompt_response_interaction"], 0.0)
                normalization = next(
                    row
                    for row in summary
                    if row["metric"] == "auroc"
                    and row["contrast"] == "normalization"
                )
                self.assertAlmostEqual(float(normalization["mean_delta"]), 0.05)
            finally:
                if previous is None:
                    os.environ.pop("GI_ARTIFACT_DIR", None)
                else:
                    os.environ["GI_ARTIFACT_DIR"] = previous

    def test_teacher_forced_positions_match_generation_steps(self) -> None:
        self.assertEqual(
            compat.released_prompt("test"),
            "<s>[INST] <<SYS>>\n You are a helpful assistant.\n \n"
            "<</SYS>>\n\n test [/INST] ",
        )
        self.assertEqual(
            compat.teacher_forced_positions(5, 4),
            {"first": 4, "middle": 6, "last": 7},
        )

    def test_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("GI_ARTIFACT_DIR")
            os.environ["GI_ARTIFACT_DIR"] = directory
            try:
                model = compat.MODELS["llama2-7b"]
                target = compat.paths(model)
                target["raw_activations"].mkdir(parents=True)
                for index in range(510):
                    record = {}
                    for token in compat.TOKENS:
                        length = 4 if token == "first" else 1
                        record[f"output_token_{token}"] = {
                            f"layer_{layer}": torch.randn(1, length, 8)
                            for layer in model["layers"]
                        }
                    with (target["raw_activations"] / f"sample_{index}.pkl").open(
                        "wb"
                    ) as handle:
                        pickle.dump(record, handle)
                compat.compact("llama2-7b")
                payload = torch.load(
                    target["compact"], map_location="cpu", weights_only=False
                )
                self.assertEqual(tuple(payload["activations"].shape), (510, 3, 3, 8))
                self.assertFalse(target["raw_activations"].exists())
            finally:
                if previous is None:
                    os.environ.pop("GI_ARTIFACT_DIR", None)
                else:
                    os.environ["GI_ARTIFACT_DIR"] = previous

    def test_probe_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("GI_ARTIFACT_DIR")
            os.environ["GI_ARTIFACT_DIR"] = directory
            epochs = compat.CONFIG["paper_settings"]["epochs"]
            seeds = compat.CONFIG["paper_settings"]["replication_seeds"]
            compat.CONFIG["paper_settings"]["epochs"] = 1
            compat.CONFIG["paper_settings"]["replication_seeds"] = [0]
            try:
                model = compat.MODELS["llama2-7b"]
                target = compat.paths(model)
                target["root"].mkdir(parents=True)
                block = compat.CONFIG["dataset"]["released_rows_per_instruction"]
                source = [
                    {
                        "instruction_id_list": [f"instruction_{instruction}"],
                        "prompt": f"prompt_{instruction}_{task}",
                    }
                    for instruction in range(
                        compat.CONFIG["dataset"]["instruction_types"]
                    )
                    for task in range(block)
                ]
                labels = []
                values = []
                generator = torch.Generator().manual_seed(0)
                for index, row in enumerate(source):
                    label = bool((index % 102) % 2)
                    labels.append(
                        {
                            "instruction_id_list": row["instruction_id_list"],
                            "prompt": row["prompt"],
                            "response": "test",
                            "follow_all_instructions": label,
                            "follow_instruction_list": [label],
                        }
                    )
                    sample = torch.randn(3, 3, 8, generator=generator)
                    sample[..., 0] += 2 if label else -2
                    values.append(sample)
                compat.write_jsonl(target["labels"], labels)
                torch.save(
                    {
                        "model": model,
                        "tokens": compat.TOKENS,
                        "layers": model["layers"],
                        "activations": torch.stack(values),
                    },
                    target["compact"],
                )
                compat.train("llama2-7b", device="cpu")
                compat.select_layer("llama2-7b", device="cpu")
                manifest = submission.analyze(
                    "llama2-7b",
                    device="cpu",
                    normalization="train_fitted",
                    bootstrap_samples=0,
                )
                submission_paths = submission.output_paths(model)
                self.assertTrue(submission_paths["runs"].exists())
                self.assertTrue(submission_paths["predictions"].exists())
                self.assertEqual(
                    manifest["analysis"]["normalization_modes_run"], ["train_fitted"]
                )
                with submission_paths["runs"].open() as handle:
                    submission_runs = list(csv.DictReader(handle))
                self.assertTrue(submission_runs)
                self.assertTrue(
                    all(
                        row["threshold_source"] == "development_balanced_accuracy"
                        for row in submission_runs
                    )
                )
                self.assertEqual(
                    set(submission.METRICS),
                    {key for key in submission_runs[0] if key in submission.METRICS},
                )
                with target["summary"].open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 18)
                with target["metrics_summary"].open() as handle:
                    metric_rows = list(csv.DictReader(handle))
                self.assertEqual(
                    {row["metric"] for row in metric_rows}, set(compat.METRICS)
                )
                with target["runs"].open() as handle:
                    runs = list(csv.DictReader(handle))
                self.assertTrue(runs)
                for row in runs:
                    self.assertEqual(
                        int(row["tn"])
                        + int(row["fp"])
                        + int(row["fn"])
                        + int(row["tp"]),
                        int(row["n_test"]),
                    )
                with target["predictions"].open() as handle:
                    self.assertTrue(list(csv.DictReader(handle)))
                self.assertTrue(compat.training_complete(model, target))
                with target["layer_selection_summary"].open() as handle:
                    selection = list(csv.DictReader(handle))
                self.assertEqual(len(selection), 6)
                self.assertIn(int(selection[-1]["selected_layer"]), model["layers"])
                compat.baseline("llama2-7b", "cpu", False, False)
                self.assertEqual(compat.read_jsonl(target["labels"]), labels)
            finally:
                compat.CONFIG["paper_settings"]["epochs"] = epochs
                compat.CONFIG["paper_settings"]["replication_seeds"] = seeds
                if previous is None:
                    os.environ.pop("GI_ARTIFACT_DIR", None)
                else:
                    os.environ["GI_ARTIFACT_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
