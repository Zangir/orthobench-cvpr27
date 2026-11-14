"""Leaderboard statistics under two evaluation variables (vectorised bootstrap).

All scores of all models share the same patient-bootstrap draws, so differences
between models and between reference versions are paired.
"""
import itertools
import numpy as np
from scipy.stats import kendalltau


def draw_weights(n, n_boot, seed):
    """Multiplicity matrix (n_boot x n) of a patient bootstrap; row b = draw b."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    w = np.zeros((n_boot, n))
    np.add.at(w, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1)
    return w


def scores(truth, pred, w=None):
    """Macro-F1 and balanced accuracy; with w, one value per bootstrap row."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    w = np.ones((1, len(truth))) if w is None else w
    f1s, recalls = [], []
    for c in range(3):
        t, p = truth == c, pred == c
        tp, fp, fn = w @ (t & p), w @ (~t & p), w @ (t & ~p)
        den = 2 * tp + fp + fn
        f1s.append(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0))
        sup = tp + fn
        recalls.append(np.divide(tp, sup, out=np.full_like(tp, np.nan), where=sup > 0))
    f1 = np.mean(f1s, 0)
    bacc = np.nanmean(np.stack(recalls), 0)
    return f1, bacc


def ci(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def entry_scores(truth, runs, w=None, metric=0):
    """Score of one leaderboard entry = mean over its runs (1 frozen fit or several seeds)."""
    return np.mean([scores(truth, p, w)[metric] for p in runs], 0)


def leaderboard_shift(names, preds, ref_a, ref_b, w, metric=0):
    """Compare leaderboards scored against ref_a and ref_b on identical predictions.

    preds[k] is a list of prediction vectors for entry k (one per training seed).
    """
    point_a = np.array([entry_scores(ref_a, p, None, metric)[0] for p in preds])
    point_b = np.array([entry_scores(ref_b, p, None, metric)[0] for p in preds])
    boot_a = np.stack([entry_scores(ref_a, p, w, metric) for p in preds], 1)  # n_boot x models
    boot_b = np.stack([entry_scores(ref_b, p, w, metric) for p in preds], 1)
    taus = np.array([kendalltau(a, b).statistic for a, b in zip(boot_a, boot_b)])
    top_a, top_b = boot_a.argmax(1), boot_b.argmax(1)
    pairs = []
    for i, j in itertools.combinations(range(len(names)), 2):
        da, db = point_a[i] - point_a[j], point_b[i] - point_b[j]
        d = (boot_b[:, i] - boot_b[:, j]) - (boot_a[:, i] - boot_a[:, j])
        lo, hi = ci(d)
        pairs.append({"a": names[i], "b": names[j], "diff_ref_a": float(da), "diff_ref_b": float(db),
                      "flip": bool(np.sign(da) != np.sign(db) and da != 0 and db != 0),
                      "shift": float(db - da), "shift_ci": [lo, hi],
                      "significant_shift": bool(lo > 0 or hi < 0)})
    rank_a = (-point_a).argsort().argsort() + 1
    rank_b = (-point_b).argsort().argsort() + 1
    return {
        "models": [{"name": n, "score_ref_a": float(a), "score_ref_b": float(b),
                    "rank_ref_a": int(ra), "rank_ref_b": int(rb),
                    "ci_ref_a": ci(boot_a[:, k]), "ci_ref_b": ci(boot_b[:, k]),
                    "delta": float(b - a), "delta_ci": ci(boot_b[:, k] - boot_a[:, k])}
                   for k, (n, a, b, ra, rb) in enumerate(zip(names, point_a, point_b, rank_a, rank_b))],
        "kendall_tau_point": float(kendalltau(point_a, point_b).statistic),
        "kendall_tau_ci": ci(taus),
        "top_ref_a": names[int(point_a.argmax())], "top_ref_b": names[int(point_b.argmax())],
        "p_same_top_over_draws": float((top_a == top_b).mean()),
        "n_pairs": len(pairs), "n_flips": sum(p["flip"] for p in pairs),
        "n_significant_shifts": sum(p["significant_shift"] for p in pairs),
        "n_significant_flips": sum(p["flip"] and p["significant_shift"] for p in pairs),
        "pairs": pairs,
    }


def auroc(score, positive):
    """Mann-Whitney AUROC with ties counted as one half."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def auroc_boot(score, positive, n_boot=2000, seed=20261108):
    rng = np.random.default_rng(seed)
    n, vals = len(score), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = auroc(np.asarray(score)[i], np.asarray(positive)[i])
        if not np.isnan(v):
            vals.append(v)
    return ci(np.array(vals))
