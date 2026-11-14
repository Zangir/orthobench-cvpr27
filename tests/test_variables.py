"""Checks for the CVPR 2027 leaderboard statistics."""
import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orthobench import variables as V  # noqa: E402
from orthobench.zoo import confident_flags  # noqa: E402


class TestVariables(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = rng.integers(0, 3, 150)
        self.p = np.where(rng.random(150) < 0.5, self.y, rng.integers(0, 3, 150))

    def test_scores_match_sklearn(self):
        f1, bacc = V.scores(self.y, self.p)
        self.assertAlmostEqual(f1[0], f1_score(self.y, self.p, average="macro", labels=[0, 1, 2]), places=12)
        self.assertAlmostEqual(bacc[0], balanced_accuracy_score(self.y, self.p), places=12)

    def test_bootstrap_rows_match_resampling(self):
        w = V.draw_weights(len(self.y), 5, 7)
        rng = np.random.default_rng(7)
        idx = rng.integers(0, len(self.y), (5, len(self.y)))
        f1 = V.scores(self.y, self.p, w)[0]
        for b in range(5):
            ref = f1_score(self.y[idx[b]], self.p[idx[b]], average="macro", labels=[0, 1, 2], zero_division=0)
            self.assertAlmostEqual(f1[b], ref, places=12)

    def test_identical_references_have_no_shift(self):
        w = V.draw_weights(len(self.y), 50, 1)
        preds = [[self.p], [self.y], [np.zeros_like(self.y)]]
        out = V.leaderboard_shift(["a", "b", "c"], preds, self.y, self.y, w)
        self.assertEqual(out["n_flips"], 0)
        self.assertEqual(out["n_significant_shifts"], 0)
        self.assertAlmostEqual(out["kendall_tau_point"], 1.0)

    def test_auroc(self):
        s = np.random.default_rng(3).random(80)
        pos = s + np.random.default_rng(4).normal(0, 0.3, 80) > 0.5
        self.assertAlmostEqual(V.auroc(s, pos), roc_auc_score(pos, s), places=12)

    def test_confident_flags_detect_obvious_noise(self):
        probs = np.array([[0.9, 0.05, 0.05]] * 10 + [[0.05, 0.9, 0.05]] * 10)
        given = np.array([0] * 9 + [1] + [1] * 10)
        flags, _ = confident_flags(probs, given)
        self.assertTrue(flags[9])
        self.assertEqual(int(flags.sum()), 1)


if __name__ == "__main__":
    unittest.main()
