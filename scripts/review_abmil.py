#!/usr/bin/env python3
"""Post-plan gated attention-MIL baseline (Ilse et al., ICML 2018) over cached frame embeddings.

Plan amendment 2026-11-17 'A1_attention_mil' in ../cvpr27_plan.json, logged before running.
A patient is a bag of intraoral frame embeddings (DINOv2-S 384-d, SAM ViT-B 768-d ViT state).
Reference split as the leaderboard: fitted on the 593 unaudited, non-quarantined training patients
(source labels), scored once on the 189 test patients against both label versions with the
shared 2,000 patient-bootstrap draws. Hyperparameters come from patient-grouped 5-fold CV on the
training patients only. CPU, one config or one seed per call, resumable.

  python scripts/review_abmil.py cv    --workbook W --encoder dinov2 --config 3
  python scripts/review_abmil.py final --workbook W --encoder dinov2 --seed 0
  python scripts/review_abmil.py aggregate --workbook W        -> results/review_abmil.json
Per-run files (with per-patient probabilities) go to the gitignored data/review_runs/.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_cvpr27 as R  # noqa: E402
from orthobench import zoo, variables as V  # noqa: E402
from orthobench.reference_eval import split_masks  # noqa: E402

RUNS = ROOT / "data/review_runs"
ENC = {"dinov2": ("dinov2_intraoral.npz", "features", "DINOv2-S"),
       "sam": ("sam_vitb_intraoral.npz", "features_vit", "SAM ViT-B")}
GRID = [dict(lr=lr, wd=wd, dropout=dp) for lr, wd, dp in itertools.product((1e-4, 1e-3), (1e-4, 1e-2), (0.0, 0.25))]
CHECKPOINTS = (20, 50, 100)   # epochs evaluated inside each CV run
HID, ATT, BATCH, SEEDS = 128, 64, 16, (0, 1, 2)


def load(enc, ctx):
    f = dict(np.load(R.CACHE / ENC[enc][0], allow_pickle=True))
    pids, y = f["patient_id"].astype(str), f["label_numeric"].astype(int)
    tr, te = split_masks(pids, ctx["audit_ids"], ctx["test_ids"], ctx["quarantine"])
    return f[ENC[enc][1]].astype(np.float32), pids, y, tr, te


def bags(X, pids, order):
    idx = [np.where(pids == p)[0] for p in order]
    K = max(len(i) for i in idx)
    B = np.zeros((len(order), K, X.shape[1]), np.float32)
    M = np.zeros((len(order), K), bool)
    for n, i in enumerate(idx):
        B[n, :len(i)], M[n, :len(i)] = X[i], True
    return B, M


def model(d, dropout):
    import torch
    nn = torch.nn

    class GatedABMIL(nn.Module):
        """Ilse et al. (2018) gated attention pooling: a_k = softmax(w^T (tanh(V h_k) * sigm(U h_k)))."""

        def __init__(self):
            super().__init__()
            self.fe = nn.Sequential(nn.Linear(d, HID), nn.ReLU(), nn.Dropout(dropout))
            self.V, self.U, self.w = nn.Linear(HID, ATT), nn.Linear(HID, ATT), nn.Linear(ATT, 1)
            self.cls = nn.Linear(HID, 3)

        def forward(self, x, m):
            h = self.fe(x)
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
            a = torch.softmax(a.masked_fill(~m, float("-inf")), 1)
            return self.cls((a.unsqueeze(-1) * h).sum(1)), a
    return GatedABMIL()


def fit_predict(Btr, Mtr, ytr, Bte, Mte, cfg, seed, epochs=CHECKPOINTS):
    """Train; return {epoch: test probabilities} at each checkpoint epoch."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = model(Btr.shape[2], cfg["dropout"])
    cnt = np.bincount(ytr, minlength=3).astype(np.float64)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    Xt, Mt, Yt = torch.from_numpy(Btr), torch.from_numpy(Mtr), torch.from_numpy(ytr).long()
    Xe, Me = torch.from_numpy(Bte), torch.from_numpy(Mte)
    out = {}
    for ep in range(1, max(epochs) + 1):
        net.train()
        order = rng.permutation(len(ytr))
        for k in range(0, len(order), BATCH):
            b = torch.from_numpy(order[k:k + BATCH])
            opt.zero_grad()
            lossf(net(Xt[b], Mt[b])[0], Yt[b]).backward()
            opt.step()
        if ep in epochs:
            net.eval()
            with torch.no_grad():
                out[ep] = torch.softmax(net(Xe, Me)[0], 1).numpy()
    return out


def setup(workbook, enc):
    ctx = zoo.context(R.read_cases(workbook), R.CACHE)
    X, pids, y, tr, te = load(enc, ctx)
    train = np.array(sorted(set(pids[tr])))
    ytrain = np.array([int(y[pids == p][0]) for p in train])
    sc = StandardScaler().fit(X[tr])  # training frames only, as the linear probe
    Xs = np.zeros_like(X); Xs[tr | te] = sc.transform(X[tr | te])
    return ctx, Xs, pids, y, tr, te, train, ytrain


def ids_hash(ids):
    import hashlib
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def cmd_cv(a):
    path = RUNS / f"abmil_cv_{a.encoder}_c{a.config}.json"
    if path.exists():
        print(f"[skip] {path.name} exists"); return
    _, X, pids, _, tr, _, train, ytrain = setup(a.workbook, a.encoder)
    cfg, t0 = GRID[a.config], time.monotonic()
    scores = {ep: [] for ep in CHECKPOINTS}
    for fa, fb in StratifiedKFold(5, shuffle=True, random_state=0).split(train, ytrain):
        Btr, Mtr = bags(X, pids, train[fa]); Bte, Mte = bags(X, pids, train[fb])
        probs = fit_predict(Btr, Mtr, ytrain[fa], Bte, Mte, cfg, seed=0)
        for ep, p in probs.items():
            scores[ep].append(float(V.scores(ytrain[fb], p.argmax(1))[0][0]))
    res = {"encoder": a.encoder, "config_index": a.config, "config": cfg,
           "cv_fold_macro_f1": {str(ep): v for ep, v in scores.items()},
           "cv_mean_macro_f1": {str(ep): float(np.mean(v)) for ep, v in scores.items()},
           "n_train_patients": len(train), "seconds": time.monotonic() - t0}
    RUNS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[abmil-cv] {a.encoder} c{a.config} {cfg}: " + " ".join(f"ep{k} {v:.3f}" for k, v in res["cv_mean_macro_f1"].items()), flush=True)


def selected(enc):
    rows = []
    for c in range(len(GRID)):
        p = RUNS / f"abmil_cv_{enc}_c{c}.json"
        if not p.exists():
            raise SystemExit(f"CV incomplete: {p.name} missing")
        r = json.loads(p.read_text())
        rows += [(v, -c, -int(ep), c, int(ep)) for ep, v in r["cv_mean_macro_f1"].items()]
    best = max(rows)  # highest CV macro-F1; ties -> earlier config, fewer epochs
    return {"config_index": best[3], "config": GRID[best[3]], "epochs": best[4], "cv_mean_macro_f1": best[0]}


def cmd_final(a):
    path = RUNS / f"abmil_final_{a.encoder}_s{a.seed}.json"
    if path.exists():
        print(f"[skip] {path.name} exists"); return
    sel = selected(a.encoder)
    _, X, pids, _, tr, te, train, ytrain = setup(a.workbook, a.encoder)
    test = np.array(sorted(set(pids[te])))
    Btr, Mtr = bags(X, pids, train); Bte, Mte = bags(X, pids, test)
    t0 = time.monotonic()
    prob = fit_predict(Btr, Mtr, ytrain, Bte, Mte, sel["config"], seed=a.seed, epochs=(sel["epochs"],))[sel["epochs"]]
    res = {"encoder": a.encoder, "seed": a.seed, "selected": sel, "n_train_patients": len(train),
           "n_test_patients": len(test), "test_ids_sha256": ids_hash(test),
           "prob": prob.tolist(), "seconds": time.monotonic() - t0}
    RUNS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res) + "\n")
    print(f"[abmil-final] {a.encoder} s{a.seed}: pred counts {np.bincount(prob.argmax(1), minlength=3).tolist()}", flush=True)


def cmd_aggregate(a):
    import postplan_baselines as PB
    rows = R.read_cases(a.workbook)
    ctx = zoo.context(rows, R.CACHE)
    store = R._load_preds()
    test = np.array(sorted(str(x) for x in store["intraoral/dinov2/logreg#ids"]))
    manifest, final, _ = zoo.reference_labels(ctx, test)
    w = V.draw_weights(len(test), R.N_BOOT, R.SEED)
    zr = json.loads((R.RESULTS / "zoo_reference.json").read_text())["macro_f1"]["models"]
    lead = [R._align(store, f"finetune/dinov2/s{s}", test).argmax(1) for s in range(3)]
    probe = {"dinov2": [R._align(store, "intraoral/dinov2/logreg", test).argmax(1)],
             "sam": [PB.sam_probe(ctx, test)[1].argmax(1)]}
    out = {"provenance": "run", "status": "post-plan",
           "plan_amendment": "experiments/cvpr27_plan.json amendments[A1_attention_mil]",
           "n_test": len(test), "bootstrap_replicates": R.N_BOOT, "bootstrap_seed": R.SEED,
           "model": {"hidden": HID, "attention": ATT, "batch_bags": BATCH, "optimizer": "Adam",
                     "loss": "class-weighted cross-entropy", "grid": GRID, "checkpoint_epochs": list(CHECKPOINTS)},
           "entries": {}}
    for enc in ENC:
        runs = [RUNS / f"abmil_final_{enc}_s{s}.json" for s in SEEDS]
        if not all(p.exists() for p in runs):
            out["entries"][ENC[enc][2]] = {"complete": False, "n_seeds_done": sum(p.exists() for p in runs)}
            continue
        R_ = [json.loads(p.read_text()) for p in runs]
        assert all(r["test_ids_sha256"] == ids_hash(test) for r in R_), "test patient order differs"
        preds = [np.array(r["prob"]).argmax(1) for r in R_]
        cv = [json.loads((RUNS / f"abmil_cv_{enc}_c{c}.json").read_text()) for c in range(len(GRID))]
        rec = {"complete": True, "selected": R_[0]["selected"], "n_train_patients": R_[0]["n_train_patients"],
               "cv_grid": [{"config": c["config"], "cv_mean_macro_f1": c["cv_mean_macro_f1"]} for c in cv],
               "pred_class_counts_per_seed": [np.bincount(p, minlength=3).tolist() for p in preds]}
        for r, t in (("source", manifest), ("adjudicated", final)):
            mb, pb, lb = (V.entry_scores(t, preds, w, 0), V.entry_scores(t, probe[enc], w, 0),
                          V.entry_scores(t, lead, w, 0))
            m0, p0, l0 = (V.entry_scores(t, preds)[0], V.entry_scores(t, probe[enc])[0], V.entry_scores(t, lead)[0])
            others = sorted((m["score_ref_a" if r == "source" else "score_ref_b"] for m in zr), reverse=True)
            rec[r] = {"macro_f1": float(m0), "macro_f1_ci": V.ci(mb),
                      "per_seed_macro_f1": [float(V.scores(t, p)[0][0]) for p in preds],
                      "balanced_accuracy": float(V.entry_scores(t, preds, None, 1)[0]),
                      "probe_macro_f1": float(p0), "probe_macro_f1_ci": V.ci(pb),
                      "mil_minus_probe": float(m0 - p0), "mil_minus_probe_ci": V.ci(mb - pb),
                      "mil_beats_probe_share_of_draws": float((mb > pb).mean()),
                      "leader_minus_mil": float(l0 - m0), "leader_minus_mil_ci": V.ci(lb - mb),
                      "rank_if_inserted_among_33": int(1 + sum(s > m0 for s in others))}
        out["entries"][ENC[enc][2]] = rec
    out["probe_definition"] = {"DINOv2-S": "leaderboard entry 'DINOv2-S / logreg' (stored predictions)",
                               "SAM ViT-B": "post-plan entry 'SAM ViT-B / logreg' (features_vit), refit deterministically"}
    R.save("review_abmil.json", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["cv", "final", "aggregate"])
    ap.add_argument("--workbook", type=Path, required=True)
    ap.add_argument("--encoder", choices=sorted(ENC))
    ap.add_argument("--config", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    {"cv": cmd_cv, "final": cmd_final, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
