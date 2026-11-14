"""Independent checks of statistical definitions and leakage exclusion invariants."""
import unittest
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from orthobench.reference_stats import agreement, paired_reference_scores
from orthobench.reference_eval import split_masks


class ReferenceTests(unittest.TestCase):
    def test_four_category_kappa_matches_independent_implementation(self):
        a = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3])
        b = np.array([0, 1, 2, 1, 1, 2, 0, 3, 0])
        result = agreement(a, b, n_boot=1000)
        self.assertAlmostEqual(result["kappa"], cohen_kappa_score(a, b, labels=[0, 1, 2, 3]))
        self.assertEqual(result["agreements"], 5)

    def test_not_assessable_agreement_is_in_denominator(self):
        result = agreement([0, 0, 3, 3], [0, 1, 3, 3], n_boot=1000)
        self.assertEqual(result["raw_agreement"], .75)
        self.assertEqual(result["matrix"][3][3], 2)

    def test_absent_class_still_contributes_to_macro_f1(self):
        result = paired_reference_scores([0, 0, 1], [0, 0, 1], [0, 0, 1], n_boot=100)
        self.assertAlmostEqual(result["consensus"]["macro_f1"], 2 / 3)
        self.assertEqual(result["delta_ci"], [0., 0.])

    def test_paired_score_delta_matches_sklearn(self):
        a, b, p = [0, 1, 1, 2, 2], [0, 0, 1, 2, 0], [0, 1, 2, 2, 0]
        result = paired_reference_scores(a, b, p, n_boot=100)
        expected = f1_score(b, p, labels=[0, 1, 2], average="macro") - f1_score(a, p, labels=[0, 1, 2], average="macro")
        self.assertAlmostEqual(result["delta_consensus_minus_manifest"], expected)

    def test_audit_pilot_exclusions_and_duplicates_never_train(self):
        pids = np.array(["train", "train", "test", "pilot", "excluded", "duplicate"])
        tr, te = split_masks(pids, {"test", "pilot", "excluded"}, {"test"}, {"duplicate"})
        self.assertEqual(pids[tr].tolist(), ["train", "train"])
        self.assertEqual(pids[te].tolist(), ["test"])

    def test_quarantine_removes_test_patients_too(self):
        pids = np.array(["train", "test", "test_dup"])
        tr, te = split_masks(pids, {"test", "test_dup"}, {"test", "test_dup"}, {"test_dup"})
        self.assertEqual(pids[te].tolist(), ["test"])


if __name__ == "__main__":
    unittest.main()
