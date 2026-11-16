#!/usr/bin/env python3
"""Post-plan baselines P1 (SAM ViT-B linear probe) and P2 (zero-shot CLIP-B/16, BiomedCLIP).

Plan amendment 2026-11-16 in ../cvpr27_plan.json, logged before any run. Selection uses only the
593 unaudited training patients (source labels); the 189 audit-held-out test patients are scored
once. These entries are NOT added to the pre-registered 33-entry statistics.

  python scripts/postplan_baselines.py --workbook W [--skip-sam]   (CPU; local models only)
-> results/postplan_baselines.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_cvpr27 as R  # noqa: E402
from orthobench import zoo, variables as V  # noqa: E402
from orthobench.probes import logreg_probe  # noqa: E402
from orthobench.reference_eval import split_masks  # noqa: E402

PLAN = json.loads((HERE.parents[1] / "cvpr27_plan.json").read_text())
AMEND = [a for a in PLAN["amendments"] if a.get("date") == "2026-11-16"][0]
TEMPLATES, PCLS = AMEND["prompt_templates"], AMEND["prompt_classes"]


def patient_f1(prob, pids, order, truth):
    agg = np.stack([prob[pids == p].mean(0) for p in order])
    return V.scores(truth, agg.argmax(1))[0][0], agg


def masks(f, ctx):
    pids = f["patient_id"].astype(str)
    tr, te = split_masks(pids, ctx["audit_ids"], ctx["test_ids"], ctx["quarantine"])
    return pids, tr, te


def train_truth(f, pids, tr):
    y = f["label_numeric"].astype(int)
    order = np.array(sorted(set(pids[tr])))
    return order, np.array([int(y[pids == p][0]) for p in order])


# ------------------------------------------------------------------ P2 zero-shot
def text_embeddings(enc):
    import open_clip
    import torch
    if enc == "clip_b16":
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        tok = open_clip.get_tokenizer("ViT-B-16")
    else:
        hub = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, _, _ = open_clip.create_model_and_transforms(hub)
        tok = open_clip.get_tokenizer(hub)
    model.eval()
    out = []
    with torch.no_grad():
        for t in TEMPLATES:
            key = "desc" if "{desc}" in t else "cls"
            txt = [t.format(**{key: c}) for c in PCLS[key]]
            e = model.encode_text(tok(txt)).float()
            out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
        scale = float(model.logit_scale.exp())
    return out, scale


def zero_shot(enc, ctx, test, refs):
    f = zoo.load_cache(R.CACHE, enc, "intraoral")
    pids, tr, te = masks(f, ctx)
    x = f["features"].astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    T, scale = text_embeddings(enc)
    order, ytr = train_truth(f, pids, tr)
    val = []
    for t in T:
        logit = scale * x[tr] @ t.T
        p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        val.append(float(patient_f1(p, pids[tr], order, ytr)[0]))
    best = int(np.argmax(val))
    logit = scale * x[te] @ T[best].T
    p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    agg = np.stack([p[pids[te] == q].mean(0) for q in test])
    return {"selected_template": TEMPLATES[best], "selected_template_index": best,
            "validation_macro_f1_per_template": val, "n_validation_patients": len(order),
            "logit_scale": scale}, agg


# ------------------------------------------------------------------ P1 SAM linear probe
def sam_probe(ctx, test):
    f = dict(np.load(R.CACHE / "sam_vitb_intraoral.npz", allow_pickle=True))
    pids, tr, te = masks(f, ctx)
    y = f["label_numeric"].astype(int)
    order, ytr = train_truth(f, pids, tr)
    cv = {}
    for key in ("features", "features_vit"):
        X = f[key]
        scores = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(order, ytr):
            ma, mb = np.isin(pids, order[a]) & tr, np.isin(pids, order[b]) & tr
            prob = logreg_probe(X[ma], y[ma], X[mb], class_weight="balanced", C=1.0)
            scores.append(float(patient_f1(prob, pids[mb], order[b], ytr[b])[0]))
        cv[key] = {"fold_macro_f1": scores, "mean": float(np.mean(scores))}
    sel = "features" if cv["features"]["mean"] >= cv["features_vit"]["mean"] else "features_vit"
    prob = logreg_probe(f[sel][tr], y[tr], f[sel][te], class_weight="balanced", C=1.0)
    agg = np.stack([prob[pids[te] == q].mean(0) for q in test])
    return {"cv_patient_macro_f1": cv, "selected_feature": sel,
            "selected_feature_desc": {"features": "mean-pooled 256-d neck output",
                                      "features_vit": "mean-pooled 768-d ViT hidden state"}[sel],
            "n_train_patients": len(order), "n_train_images": int(tr.sum()),
            "n_unreadable": int(f.get("n_unreadable", 0))}, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path, required=True)
    ap.add_argument("--skip-sam", action="store_true")
    a = ap.parse_args()
    rows = R.read_cases(a.workbook)
    ctx = zoo.context(rows, R.CACHE)
    store = R._load_preds()
    test = np.array(sorted(str(x) for x in store["intraoral/dinov2/logreg#ids"]))
    manifest, final, _ = zoo.reference_labels(ctx, test)
    w = V.draw_weights(len(test), R.N_BOOT, R.SEED)
    lead_runs = [R._align(store, f"finetune/dinov2/s{s}", test).argmax(1) for s in range(3)]
    lead = {r: (V.entry_scores(t, lead_runs, w, 0), V.entry_scores(t, lead_runs, w, 1), V.entry_scores(t, lead_runs)[0])
            for r, t in (("source", manifest), ("adjudicated", final))}
    zr = json.loads((R.RESULTS / "zoo_reference.json").read_text())["macro_f1"]["models"]

    def score(pred):
        out = {}
        for r, t in (("source", manifest), ("adjudicated", final)):
            f1b, bab = V.scores(t, pred, w)
            f1, ba = V.scores(t, pred)
            others = sorted((m["score_ref_a" if r == "source" else "score_ref_b"] for m in zr), reverse=True)
            out[r] = {"macro_f1": float(f1[0]), "macro_f1_ci": V.ci(f1b), "bacc": float(ba[0]),
                      "bacc_ci": V.ci(bab), "leader_minus_entry": float(lead[r][2] - f1[0]),
                      "leader_minus_entry_ci": V.ci(lead[r][0] - f1b),
                      "rank_if_inserted_among_33": int(1 + sum(s > f1[0] for s in others)),
                      "beats_leader_point": bool(f1[0] > lead[r][2])}
        out["pred_class_counts"] = np.bincount(pred, minlength=3).tolist()
        return out

    res = {"provenance": "run", "plan_amendment": "experiments/cvpr27_plan.json amendments[date=2026-11-16]",
           "n_test": len(test), "bootstrap_replicates": R.N_BOOT, "bootstrap_seed": R.SEED,
           "leader": {"name": "DINOv2-S / fine-tuned", "source_macro_f1": float(lead["source"][2]),
                      "adjudicated_macro_f1": float(lead["adjudicated"][2])},
           "entries": {}, "not_run": []}
    for enc, nm in (("clip_b16", "CLIP-B/16 / zero-shot"), ("biomedclip", "BiomedCLIP / zero-shot")):
        meta, agg = zero_shot(enc, ctx, test, (manifest, final))
        res["entries"][nm] = {**meta, **score(agg.argmax(1))}
        print(nm, json.dumps({k: res["entries"][nm][k]["macro_f1"] for k in ("source", "adjudicated")}), flush=True)
    if not a.skip_sam and (R.CACHE / "sam_vitb_intraoral.npz").exists():
        meta, agg = sam_probe(ctx, test)
        res["entries"]["SAM ViT-B / logreg"] = {**meta, **score(agg.argmax(1))}
        print("SAM", json.dumps({k: res["entries"]["SAM ViT-B / logreg"][k]["macro_f1"] for k in ("source", "adjudicated")}))
    else:
        res["not_run"].append("SAM ViT-B / logreg (feature cache missing)")
    res["not_run"].append("generative VLM (P3): not run")
    R.save("postplan_baselines.json", res)


if __name__ == "__main__":
    main()
