#!/usr/bin/env python3
"""Repeated patient-level 5-fold CV of the 30 frozen entries (plan amendment 2026-11-17, round 9).

Pool: the reference-arm training patients (unaudited, not quarantined), source labels. The same
folds are used for every entry. Tiers: corrected repeated k-fold t-test (Nadeau & Bengio), two-sided
5%, tier = current best entry plus every entry not significantly below it, repeated on the rest.

  python scripts/repeated_cv.py --workbook W    -> results/repeated_cv.json   (CPU, cached features)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_cvpr27 as R  # noqa: E402
from orthobench import zoo, variables as V  # noqa: E402
from orthobench.reference_eval import split_masks  # noqa: E402

K, REPEATS = 5, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path, required=True)
    a = ap.parse_args()
    ctx = zoo.context(R.read_cases(a.workbook), R.CACHE)
    encs = [e for e in R.available("intraoral")]
    ref = zoo.load_cache(R.CACHE, encs[0], "intraoral")
    pids0 = ref["patient_id"].astype(str)
    tr0, _ = split_masks(pids0, ctx["audit_ids"], ctx["test_ids"], ctx["quarantine"])
    pool = np.array(sorted(set(pids0[tr0])))
    y0 = ref["label_numeric"].astype(int)
    plab = np.array([int(y0[pids0 == p][0]) for p in pool])
    folds = []
    for r in range(REPEATS):
        for f, (a_, b_) in enumerate(StratifiedKFold(K, shuffle=True, random_state=r).split(pool, plab)):
            folds.append((r, f, set(pool[a_]), pool[b_]))
    scores, t0 = {}, time.monotonic()
    for e in encs:
        F = zoo.load_cache(R.CACHE, e, "intraoral")
        pids, y, X = F["patient_id"].astype(str), F["label_numeric"].astype(int), F["features"]
        for h in zoo.HEADS:
            name = f"{zoo.PRETTY[e]} / {h}"
            per_fold, pooled = [], {r: ([], []) for r in range(REPEATS)}
            for r, f, trset, te in folds:
                mtr = np.isin(pids, list(trset)); mte = np.isin(pids, te)
                prob = zoo._head(h, X[mtr], y[mtr], X[mte])
                agg = np.stack([prob[pids[mte] == p].mean(0) for p in te])
                truth = np.array([int(y[pids == p][0]) for p in te])
                pred = agg.argmax(1)
                per_fold.append(float(V.scores(truth, pred)[0][0]))
                pooled[r][0].append(truth); pooled[r][1].append(pred)
            rep = [float(V.scores(np.concatenate(t), np.concatenate(p))[0][0]) for t, p in pooled.values()]
            scores[name] = {"fold_macro_f1": per_fold, "mean": float(np.mean(per_fold)), "sd_folds": float(np.std(per_fold, ddof=1)),
                            "repeat_macro_f1": rep, "mean_repeats": float(np.mean(rep)), "sd_repeats": float(np.std(rep, ddof=1))}
            print(f"[cv] {name}: {scores[name]['mean']:.3f} +- {scores[name]['sd_folds']:.3f} ({time.monotonic() - t0:.0f}s)", flush=True)
    # tiers: corrected repeated k-fold t-test against the best remaining entry
    n_tr, n_te = len(pool) * (K - 1) / K, len(pool) / K
    corr = 1 / (K * REPEATS) + n_te / n_tr
    remaining, tiers = set(scores), []
    while remaining:
        best = max(remaining, key=lambda n: scores[n]["mean"])
        tier = [best]
        for n in sorted(remaining - {best}, key=lambda n: -scores[n]["mean"]):
            d = np.array(scores[best]["fold_macro_f1"]) - np.array(scores[n]["fold_macro_f1"])
            var = d.var(ddof=1)
            t = d.mean() / np.sqrt(corr * var) if var > 0 else np.inf
            p = 2 * stats.t.sf(abs(t), df=len(d) - 1)
            scores[n].setdefault("vs_tier_leader", []).append({"leader": best, "mean_diff": float(d.mean()), "p": float(p)})
            if p >= 0.05:
                tier.append(n)
        tiers.append(tier)
        remaining -= set(tier)
    for i, tier in enumerate(tiers):
        for n in tier:
            scores[n]["tier"] = i + 1
    out = {"provenance": "run", "plan_amendment": "experiments/cvpr27_plan.json amendments[5] (2026-11-17)",
           "pool_patients": int(len(pool)), "k": K, "repeats": REPEATS, "n_entries": len(scores),
           "tier_test": "corrected repeated k-fold t-test (Nadeau-Bengio), two-sided 0.05", "variance_correction": corr,
           "tiers": tiers, "entries": scores}
    R.save("repeated_cv.json", out)


if __name__ == "__main__":
    main()
