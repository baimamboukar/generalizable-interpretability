import math
import unittest

from confirmatory.metrics import (
    apply_frozen_threshold,
    auprc,
    auroc,
    binary_metrics,
    cluster_bootstrap,
    operating_point_at_fpr,
    select_youden_threshold,
)
from confirmatory.verify import verify


class MetricsTest(unittest.TestCase):
    def test_perfect_binary_metrics(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual((result.tn, result.fp, result.fn, result.tp), (2, 0, 0, 2))
        self.assertEqual(result.auroc, 1.0)
        self.assertEqual(result.auprc, 1.0)
        self.assertEqual(result.balanced_accuracy, 1.0)
        self.assertEqual(result.f1, 1.0)
        self.assertAlmostEqual(result.brier, 0.025)

    def test_tied_auroc(self):
        self.assertEqual(auroc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(auprc([0, 1], [0.5, 0.5]), 0.5)

    def test_low_fpr_operating_point(self):
        point = operating_point_at_fpr([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9], 0.0)
        self.assertEqual(point.fpr, 0.0)
        self.assertEqual(point.tpr, 1.0)
        self.assertEqual(point.threshold, 0.3)
        self.assertEqual(point.fp, 0)
        self.assertEqual(point.n_negative, 2)
        self.assertEqual(point.fpr_resolution, 0.5)

    def test_frozen_development_threshold_is_applied_without_test_retuning(self):
        development = operating_point_at_fpr(
            [0, 0, 1, 1], [0.1, 0.2, 0.3, 0.9], 0.0
        )
        result = apply_frozen_threshold(
            [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], development.threshold
        )
        self.assertEqual(result["threshold"], 0.3)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fpr"], 0.5)
        self.assertEqual(result["tpr"], 1.0)

    def test_decision_threshold_does_not_switch_to_probability_scale(self):
        result = binary_metrics(
            [0, 1], [-1.0, 1.0], probabilities=[0.9, 0.1], threshold=0.0
        )
        self.assertEqual((result.tn, result.fp, result.fn, result.tp), (1, 0, 0, 1))
        self.assertEqual(result.threshold_space, "decision_score")

    def test_single_class_ranking_metrics_are_undefined(self):
        result = binary_metrics([1, 1], [0.2, 0.8])
        self.assertIsNone(result.auroc)
        self.assertIsNone(result.roc_tpr_at_fpr_0_01)
        self.assertEqual(result.n_negative, 0)

    def test_youden_threshold(self):
        self.assertEqual(select_youden_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 0.8)

    def test_cluster_bootstrap_is_deterministic(self):
        first = cluster_bootstrap(
            [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["a", "a", "b", "b"], auroc,
            replicates=50, seed=7,
        )
        second = cluster_bootstrap(
            [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["a", "a", "b", "b"], auroc,
            replicates=50, seed=7,
        )
        self.assertEqual(first, second)
        self.assertTrue(math.isfinite(first["estimate"]))

    def test_protocol_registry(self):
        self.assertEqual(verify()["registered_studies"], 18)


if __name__ == "__main__":
    unittest.main()
