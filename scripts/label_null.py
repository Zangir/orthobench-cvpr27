#!/usr/bin/env python3
"""Null distributions for the label-version reordering (plan amendment 2026-11-17).

N1: Kendall tau_b between the 33-entry macro-F1 leaderboards of two independent patient-bootstrap
    resamples under the same label version (sampling noise alone).
N2: Kendall tau_b between the source-label leaderboard and the leaderboard after 64 randomly chosen
    test patients receive a random other class (any 64 label changes).
Both are compared with the observed source-vs-adjudicated tau_b. Stored per-patient predictions only.

  python scripts/label_null.py --workbook W   -> results/label_version_null.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_cvpr27 as R  # noqa: E402
from orthobench import zoo, variables as V  # noqa: E402

N_DRAWS, SEED, N_CHANGE = 1000, 20261117, 64


def entries(store, test):
    ent = {}
    for e in R.available("intraoral"):
        for h in zoo.HEADS:
            ent[f"{zoo.PRETTY[e]} / {h}"] = [R._align(store, f"intraoral/{e}/{h}", test).argmax(1)]
    archs = sorted({k.split("/")[1] for k in store if k.startswith("finetune/") and not k.endswith("#ids")})
    for a in archs:
        runs = sorted(k for k in store if k.startswith(f"finetune/{a}/") and not k.endswith("#ids"))
        ent[f"{zoo.PRETTY.get(a, a)} / fine-tuned"] = [R._align(store, k, test).argmax(1) for k in runs]
    return ent


def board(ent, truth, w=None):
    """Macro-F1 of every entry (rows: entries; columns: bootstrap draws if w is given)."""
    return np.stack([V.entry_scores(truth, runs, w, 0) for runs in ent.values()])


def summ(x, obs):
    x = np.asarray(x)
    return {"median": float(np.median(x)), "p2_5": float(np.quantile(x, 0.025)),
            "p97_5": float(np.quantile(x, 0.975)), "share_le_observed": float(np.mean(x <= obs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path, required=True)
    a = ap.parse_args()
    ctx = zoo.context(R.read_cases(a.workbook), R.CACHE)
    store = R._load_preds()
    test = np.array(sorted(str(x) for x in store["intraoral/dinov2/logreg#ids"]))
    src, adj, _ = zoo.reference_labels(ctx, test)
    ent = entries(store, test)
    assert len(ent) == 33, len(ent)
    obs = float(kendalltau(board(ent, src)[:, 0], board(ent, adj)[:, 0]).statistic)
    rng = np.random.default_rng(SEED)
    n = len(test)
    out = {"provenance": "run", "plan_amendment": "experiments/cvpr27_plan.json amendments[date=2026-11-17]",
           "n_test": n, "n_entries": len(ent), "n_draws": N_DRAWS, "seed": SEED, "n_changed": int((src != adj).sum()),
           "observed_tau_b": obs}
    for nm, truth in (("source", src), ("adjudicated", adj)):
        wa, wb = V.draw_weights(n, N_DRAWS, rng.integers(1 << 31)), V.draw_weights(n, N_DRAWS, rng.integers(1 << 31))
        ba, bb = board(ent, truth, wa), board(ent, truth, wb)
        taus = [kendalltau(ba[:, i], bb[:, i]).statistic for i in range(N_DRAWS)]
        out[f"N1_bootstrap_pairs_{nm}"] = summ(taus, obs)
    base = board(ent, src)[:, 0]
    taus = []
    for _ in range(N_DRAWS):
        t = src.copy()
        idx = rng.choice(n, N_CHANGE, replace=False)
        t[idx] = (t[idx] + rng.integers(1, 3, N_CHANGE)) % 3
        taus.append(kendalltau(base, board(ent, t)[:, 0]).statistic)
    out["N2_random_relabel_64"] = summ(taus, obs)
    R.save("label_version_null.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
