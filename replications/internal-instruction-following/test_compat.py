from __future__ import annotations

import csv
import os
import pickle
import tempfile
import unittest

import compat
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

    def test_extension_layer_configuration(self) -> None:
        compat.validate_extension_models()
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
                source = compat.read_jsonl(
                    compat.APPLE / "data" / "ifeval_simple.jsonl"
                )
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
