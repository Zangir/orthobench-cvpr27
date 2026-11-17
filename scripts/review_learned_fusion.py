#!/usr/bin/env python3
"""Post-plan learned trimodal fusion on the 285 fully paired patients.

Plan amendment 2026-11-17 'B1_learned_fusion' in ../cvpr27_plan.json, logged before running.
Same cohort, patient vectors (L2-normalised mean image feature), source labels, 10 stratified
70/30 patient splits (seeds 0-9) and logistic probe as the complete-case concatenation fusion in
results/fusion.json (orthobench.benchmark.fusion). Per split it scores:
  photos      photographs only (logistic probe)                      [as fusion.json]
  concat      concatenation of the three blocks (logistic probe)     [as fusion.json]
  stack       per-modality logistic heads; a multinomial logistic stacking layer on their
              out-of-fold log-probabilities (inner 5-fold CV on the training patients)
  weighted    per-modality logistic heads; modality weights on the simplex (step 0.1) that
              maximise out-of-fold macro-F1 on the training patients; weighted mean probability
Nothing is selected on the test patients.

  python scripts/review_learned_fusion.py --features rad --seed 0
-> data/review_runs/fusion_<features>_s<seed>.json (resumable: skipped if it exists)
  python scripts/review_learned_fusion.py --aggregate   -> results/review_learned_fusion.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orthobench import benchmark, splits, variables as V  # noqa: E402
from orthobench.probes import logreg_probe  # noqa: E402

CACHE, RUNS = ROOT / "data/feat_cache", ROOT / "data/review_runs"  # per-patient predictions stay in gitignored data/
MODS = ("intraoral", "opg", "cephalogram")
FEATURES = {"dinov2": {"intraoral": "dinov2", "opg": "dinov2", "cephalogram": "dinov2"},
            "rad": {"intraoral": "dinov2", "opg": "rad_dino", "cephalogram": "rad_dino"}}
METHODS = ("photos", "concat", "stack", "weighted")
SEEDS, N_BOOT, BOOT_SEED = tuple(range(10)), 2000, 20261108
GRID = [w for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=3) if abs(sum(w) - 1) < 1e-9]


def cohort(fs):
    blocks = {m: benchmark._patient_matrix(dict(np.load(CACHE / f"{FEATURES[fs][m]}_{m}.npz", allow_pickle=True)))
              for m in MODS}
    pats = sorted(set.intersection(*(set(blocks[m][0].tolist()) for m in MODS)))
    pats = np.array(pats)
    B, y = {}, None
    for m in MODS:
        p_all, M, ya = blocks[m]
        sel = np.searchsorted(p_all, pats)
        B[m] = M[sel]
        assert y is None or (ya[sel] == y).all()
        y = ya[sel]
    return pats, B, y


def f1(y, pred):
    return float(V.scores(y, pred)[0][0])


def oof(B, y, tr, seed):
    """Out-of-fold per-modality probabilities on the training patients."""
    out = {m: np.zeros((len(tr), 3)) for m in MODS}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]):
        for m in MODS:
            out[m][b] = logreg_probe(B[m][tr[a]], y[tr[a]], B[m][tr[b]])
    return out


def run(fs, seed):
    pats, B, y = cohort(fs)
    tr, te = splits.patient_level_split(pats.astype(str), y, seed=seed, test_frac=0.3)
    prob = {m: logreg_probe(B[m][tr], y[tr], B[m][te]) for m in MODS}
    pred = {"photos": prob["intraoral"].argmax(1),
            "concat": logreg_probe(np.hstack([B[m][tr] for m in MODS]), y[tr],
                                   np.hstack([B[m][te] for m in MODS])).argmax(1)}
    O = oof(B, y, tr, seed)
    eps = 1e-6
    Z = np.hstack([np.log(O[m] + eps) for m in MODS])
    meta = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Z, y[tr])
    pred["stack"] = meta.predict_proba(np.hstack([np.log(prob[m] + eps) for m in MODS])).argmax(1)
    uni = np.full(3, 1 / 3)
    scored = [(f1(y[tr], sum(w[k] * O[m] for k, m in enumerate(MODS)).argmax(1)), -np.linalg.norm(np.array(w) - uni), w)
              for w in GRID]
    best = max(scored, key=lambda t: (t[0], t[1]))
    wbest = best[2]
    pred["weighted"] = sum(wbest[k] * prob[m] for k, m in enumerate(MODS)).argmax(1)
    return {"provenance": "run", "status": "post-plan", "features": fs, "seed": seed,
            "encoders": FEATURES[fs], "n_patients": len(pats), "n_train": len(tr), "n_test": len(te),
            "test_patients_index": te.tolist(), "y_test": y[te].tolist(),
            "pred": {k: v.tolist() for k, v in pred.items()},
            "macro_f1": {k: f1(y[te], v) for k, v in pred.items()},
            "unimodal_macro_f1": {m: f1(y[te], prob[m].argmax(1)) for m in MODS},
            "weights_selected": dict(zip(MODS, map(float, wbest))), "weights_oof_macro_f1": float(best[0]),
            "stack_coef_abs_mean_by_modality": {m: float(np.abs(meta.coef_[:, 3 * k:3 * k + 3]).mean())
                                                for k, m in enumerate(MODS)}}


def aggregate():
    out = {"provenance": "run", "status": "post-plan",
           "plan_amendment": "experiments/cvpr27_plan.json amendments[B1_learned_fusion]",
           "labels": "source (manifest) labels, as results/fusion.json", "seeds": list(SEEDS), "test_frac": 0.3,
           "bootstrap": f"{N_BOOT} patient draws per split (seed {BOOT_SEED}+split seed); a draw of the 10-split mean "
                        "averages draw b of every split", "feature_sets": {}}
    for fs in FEATURES:
        runs = [json.loads((RUNS / f"fusion_{fs}_s{s}.json").read_text()) for s in SEEDS
                if (RUNS / f"fusion_{fs}_s{s}.json").exists()]
        if not runs:
            continue
        boot = {k: [] for k in METHODS}
        for r in runs:
            w = V.draw_weights(len(r["y_test"]), N_BOOT, BOOT_SEED + r["seed"])
            for k in METHODS:
                boot[k].append(V.scores(np.array(r["y_test"]), np.array(r["pred"][k]), w)[0])
        boot = {k: np.mean(v, 0) for k, v in boot.items()}
        per = {k: np.array([r["macro_f1"][k] for r in runs]) for k in METHODS}
        rec = {"n_splits": len(runs), "complete": len(runs) == len(SEEDS), "encoders": runs[0]["encoders"],
               "n_patients": runs[0]["n_patients"], "methods": {}, "differences": {}}
        for k in METHODS:
            rec["methods"][k] = {"mean": float(per[k].mean()), "sd": float(per[k].std(ddof=1)) if len(runs) > 1 else 0.0,
                                 "per_split": per[k].tolist(), "ci": V.ci(boot[k])}
        for a, b in (("stack", "concat"), ("stack", "photos"), ("weighted", "concat"), ("weighted", "photos"),
                     ("concat", "photos")):
            d = per[a] - per[b]
            rec["differences"][f"{a}_minus_{b}"] = {"mean": float(d.mean()), "ci": V.ci(boot[a] - boot[b]),
                                                    "splits_won": int((d > 0).sum()), "splits_tied": int((d == 0).sum())}
        rec["unimodal_mean"] = {m: float(np.mean([r["unimodal_macro_f1"][m] for r in runs])) for m in MODS}
        rec["weights_selected_mean"] = {m: float(np.mean([r["weights_selected"][m] for r in runs])) for m in MODS}
        rec["weights_selected_per_split"] = [r["weights_selected"] for r in runs]
        out["feature_sets"][fs] = rec
    hist = json.loads((ROOT / "results/fusion.json").read_text())["tracks"]
    out["reproduces_fusion_json"] = {
        "photos": abs(out["feature_sets"]["dinov2"]["methods"]["photos"]["mean"] - hist["intraoral"]["macro_f1"]["mean"]) < 1e-9,
        "concat": abs(out["feature_sets"]["dinov2"]["methods"]["concat"]["mean"]
                      - hist["intraoral+opg+cephalogram"]["macro_f1"]["mean"]) < 1e-9} if "dinov2" in out["feature_sets"] else None
    (ROOT / "results/review_learned_fusion.json").write_text(json.dumps(out, indent=1) + "\n")
    print("[saved] results/review_learned_fusion.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", choices=sorted(FEATURES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        return aggregate()
    path = RUNS / f"fusion_{a.features}_s{a.seed}.json"
    if path.exists():
        print(f"[skip] {path.name} exists"); return
    res = run(a.features, a.seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res) + "\n")
    print(f"[fusion] {a.features} s{a.seed}: " + " ".join(f"{k} {v:.3f}" for k, v in res["macro_f1"].items()), flush=True)


if __name__ == "__main__":
    main()
