"""Nominal reviewer agreement and fixed-three-class evaluation statistics."""
import numpy as np
from scipy.stats import binomtest, chi2, norm


def wilson(successes, total):
    """Two-sided 95% Wilson interval for a binomial proportion."""
    z = norm.ppf(0.975)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [float(centre - half), float(centre + half)]


def matrix(a, b, n_classes):
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if len(a) != len(b) or not len(a):
        raise ValueError("Nonempty, aligned observations required")
    if min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= n_classes:
        raise ValueError("Label outside the declared category set")
    return np.bincount(a * n_classes + b, minlength=n_classes**2).reshape(n_classes, n_classes)


def coefficients(m):
    """Vectorized nominal kappa and AC1; final two axes form each contingency table."""
    n = m.sum(axis=(-2, -1))
    rows, cols = m.sum(axis=-1), m.sum(axis=-2)
    observed = np.trace(m, axis1=-2, axis2=-1) / n
    expected = (rows * cols).sum(axis=-1) / n**2
    pooled = (rows + cols) / (2 * n[..., None])
    ac_chance = (pooled * (1 - pooled)).sum(axis=-1) / (m.shape[-1] - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = (observed - expected) / (1 - expected)
        ac1 = (observed - ac_chance) / (1 - ac_chance)
    return observed, expected, kappa, ac1


def agreement(a, b, n_classes=4, n_boot=20000, seed=20260906):
    """Pair bootstrap via multinomial cell counts, equivalent to resampling cases."""
    m = matrix(a, b, n_classes)
    n = int(m.sum())
    raw, expected, kappa, ac1 = coefficients(m)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, m.ravel() / n, size=n_boot).reshape(-1, n_classes, n_classes)
    _, _, ks, acs = coefficients(draws)
    per_class = []
    for c in range(n_classes):
        nr, nc, both = int(m[c].sum()), int(m[:, c].sum()), int(m[c, c])
        b1, b2 = nr - both, nc - both
        per_class.append({
            "reviewer1_n": nr, "reviewer2_n": nc, "both": both,
            "marginal_difference": (nc - nr) / n,
            "positive_agreement": 2 * both / (nr + nc) if nr + nc else None,
            "negative_agreement": 2 * (n - nr - nc + both) / (2 * n - nr - nc)
            if 2 * n - nr - nc else None,
            "mcnemar_exact_p": float(binomtest(b1, b1 + b2).pvalue) if b1 + b2 else 1.,
        })
    bowker, df = 0., 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            if m[i, j] + m[j, i]:
                bowker += float((m[i, j] - m[j, i])**2 / (m[i, j] + m[j, i]))
                df += 1
    v = -(m + m.T).astype(float)
    np.fill_diagonal(v, m.sum(0) + m.sum(1) - 2 * m.diagonal())
    d = (m.sum(1) - m.sum(0))[:-1]
    rank = int(np.linalg.matrix_rank(v[:-1, :-1]))
    sm = float(d @ np.linalg.pinv(v[:-1, :-1]) @ d)
    return {
        "n": n, "matrix": m.tolist(), "agreements": int(m.trace()),
        "raw_agreement": float(raw), "raw_agreement_ci": wilson(int(m.trace()), n),
        "expected_agreement": float(expected), "kappa": float(kappa),
        "kappa_ci": np.nanquantile(ks, [0.025, 0.975]).tolist(),
        "undefined_bootstrap_kappas": int(np.isnan(ks).sum()),
        "ac1": float(ac1), "ac1_ci": np.nanquantile(acs, [0.025, 0.975]).tolist(),
        "brennan_prediger": float((raw - 1 / n_classes) / (1 - 1 / n_classes)),
        "per_class": per_class,
        "bowker": {"statistic": bowker, "df": df, "p": float(chi2.sf(bowker, df))},
        "stuart_maxwell": {"statistic": sm, "df": rank, "p": float(chi2.sf(sm, rank))},
    }


def f1_from_matrix(m):
    """Three classes always contribute, including absent classes in bootstrap draws."""
    denom = m.sum(-1) + m.sum(-2)
    diag = np.diagonal(m, axis1=-2, axis2=-1)
    return np.divide(2 * diag, denom, out=np.zeros_like(diag, dtype=float), where=denom != 0)


def paired_reference_scores(original, consensus, pred, n_boot=2000, seed=20260910):
    """Score unchanged predictions against two labels with patient-paired uncertainty."""
    original, consensus, pred = map(np.asarray, (original, consensus, pred))
    n = len(pred)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, (n_boot, n))
    distributions, result = [], {}
    for name, truth in (("manifest", original), ("consensus", consensus)):
        m = matrix(truth, pred, 3)
        codes = truth[draws] * 3 + pred[draws]
        counts = np.stack([(codes == c).sum(1) for c in range(9)], axis=1).reshape(-1, 3, 3)
        fs = f1_from_matrix(counts).mean(-1)
        distributions.append(fs)
        result[name] = {
            "macro_f1": float(f1_from_matrix(m).mean()),
            "macro_f1_ci": np.quantile(fs, [0.025, 0.975]).tolist(),
            "accuracy": float((truth == pred).mean()),
            "per_class_f1": f1_from_matrix(m).tolist(), "confusion": m.tolist(),
        }
    result.update({"n_test": n, "changed_labels": int((original != consensus).sum()),
                   "delta_consensus_minus_manifest": result["consensus"]["macro_f1"] - result["manifest"]["macro_f1"],
                   "delta_ci": np.quantile(distributions[1] - distributions[0], [0.025, 0.975]).tolist()})
    return result
