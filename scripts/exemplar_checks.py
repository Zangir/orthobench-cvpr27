#!/usr/bin/env python3
"""CPU checks added by the 2026-11-14 plan amendment (../cvpr27_plan.json, "amendments").

Runs only on cached features and stored per-patient predictions. No model is refit for
selection and no audited label is used for fitting. Only aggregates are written to
results/; patient identifiers never leave memory.

  python scripts/exemplar_checks.py --workbook ../../updates/OrthoLabel_EDA.xlsx

Outputs: results/rank_stability.json, expert_subset.json, expert_gap_ci.json, feature_distance.json,
modality_paired.json, family_summary.json, fig2_frame_probs.json
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from orthobench import zoo, variables as V  # noqa: E402
from orthobench.reference_audit import read_cases, LABELS  # noqa: E402
from orthobench.reference_eval import split_masks  # noqa: E402
from orthobench.probes import logreg_probe  # noqa: E402
import run_cvpr27 as R  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data/feat_cache"
FAMILY = {"resnet18": "ImageNet-supervised", "resnet50": "ImageNet-supervised",
          "convnext_tiny": "ImageNet-supervised", "convnext_large": "ImageNet-supervised",
          "dinov2": "DINOv2", "dinov2_base": "DINOv2", "dinov2_large": "DINOv2",
          "rad_dino": "Radiology", "siglip2": "Image-text", "clip_b16": "Image-text",
          "biomedclip": "Image-text"}
from make_figures import EXEMPLAR_PATIENT as PRIMER_PATIENT  # record already published in fig9_primer; never printed  # noqa: E402


def save(name, obj):
    obj = {"provenance": "run", "plan": "experiments/cvpr27_plan.json (amendment 2026-11-14)", **obj}
    (RESULTS / name).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
    print(f"[saved] results/{name}", flush=True)


def entries_and_labels(ctx):
    store = R._load_preds()
    test = np.array(sorted(str(x) for x in store["intraoral/dinov2/logreg#ids"]))
    manifest, final, agreed = zoo.reference_labels(ctx, test)
    entries = {}
    for e in zoo.ENCODERS:
        if f"intraoral/{e}/logreg" not in store:
            continue
        for h in zoo.HEADS:
            entries[(e, h)] = [R._align(store, f"intraoral/{e}/{h}", test)]
    for arch in ("dinov2", "resnet50", "convnext_tiny"):
        runs = sorted(k for k in store if k.startswith(f"finetune/{arch}/") and not k.endswith("#ids"))
        entries[(arch, "fine-tuned")] = [R._align(store, k, test) for k in runs]
    return store, test, manifest, final, agreed, entries


def name(key):
    e, h = key
    return f"{zoo.PRETTY[e]} / {h}"


def rank_stability(test, manifest, final, entries):
    w = V.draw_weights(len(test), R.N_BOOT, R.SEED)          # identical draws to zoo_reference.json
    keys = list(entries)
    preds = [[p.argmax(1) for p in entries[k]] for k in keys]
    ba = np.stack([V.entry_scores(manifest, p, w) for p in preds], 1)
    bb = np.stack([V.entry_scores(final, p, w) for p in preds], 1)
    ta, tb = ba.argmax(1), bb.argmax(1)
    out = {}
    for i, k in enumerate(keys):
        out[name(k)] = {"p_rank1_source": float((ta == i).mean()), "p_rank1_adjudicated": float((tb == i).mean()),
                        "p_rank1_both": float(((ta == i) & (tb == i)).mean())}
    top = sorted(out, key=lambda n: -out[n]["p_rank1_source"])[:5]
    save("rank_stability.json", {"bootstrap_replicates": R.N_BOOT, "bootstrap_seed": R.SEED,
                                 "per_entry": out, "top5_by_source": top,
                                 "p_same_top_any_entry": float((ta == tb).mean())})


def expert_subset(ctx, test, manifest, entries):
    pr = ctx["primary"]
    ft = [p.argmax(1) for p in entries[("dinov2", "fine-tuned")]]
    rec = {"n_test": len(test)}
    for r in ("expert1", "expert2"):
        raw = [pr[p][f"{r}_label"] for p in test]
        ex = np.array([LABELS.index(v) if v in LABELS[:3] else 3 for v in raw])
        ok = ex < 3
        f_all = float(V.scores(manifest, ex)[0][0])
        f_sub = float(V.scores(manifest[ok], ex[ok])[0][0])
        m_all = float(np.mean([V.scores(manifest, q)[0][0] for q in ft]))
        m_sub = float(np.mean([V.scores(manifest[ok], q[ok])[0][0] for q in ft]))
        rec[r] = {"n_not_assessable": int((~ok).sum()), "n_assessable": int(ok.sum()),
                  "expert_macro_f1_all_na_as_error": f_all, "expert_macro_f1_assessable_only": f_sub,
                  "dinov2s_ft_macro_f1_all": m_all, "dinov2s_ft_macro_f1_same_assessable_subset": m_sub,
                  "gap_expert_minus_model_all": f_all - m_all, "gap_expert_minus_model_assessable": f_sub - m_sub}
    weaker = min(("expert1", "expert2"), key=lambda r: rec[r]["expert_macro_f1_all_na_as_error"])
    rec["weaker_reviewer_by_all_score"] = weaker
    save("expert_subset.json", rec)


def _cos_nn(X, pids, labels):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    same_p = pids[:, None] == pids[None, :]
    same_c = labels[:, None] == labels[None, :]
    def nearest(mask):
        M = np.where(mask, S, -np.inf)
        v = M.max(1)
        return v
    d_sp = 1 - nearest(same_p)
    d_oc = 1 - nearest(~same_p & same_c)
    d_ox = 1 - nearest(~same_p & ~same_c)
    nn = S.argmax(1)
    has_sp = np.isfinite(d_sp)
    frac = float((pids[nn] == pids).mean())
    frac_multi = float((pids[nn] == pids)[has_sp].mean())
    _, k = np.unique(pids, return_counts=True)
    chance = float((k * (k - 1)).sum() / (len(pids) * (len(pids) - 1)))
    return d_sp[has_sp], d_oc, d_ox, frac, frac_multi, int(has_sp.sum()), chance


def _q(x):
    return {"median": float(np.median(x)), "q1": float(np.quantile(x, 0.25)), "q3": float(np.quantile(x, 0.75)),
            "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95)), "n": int(len(x))}


def feature_distance():
    out = {}
    for m in ("intraoral", "opg", "cephalogram"):
        for e in zoo.ENCODERS:
            p = CACHE / f"{e}_{m}.npz"
            if not p.exists():
                continue
            f = zoo.load_cache(CACHE, e, m)
            X = f["features"].astype(np.float32)
            pids = f["patient_id"].astype(str)
            lab = f["label_numeric"].astype(int)
            d_sp, d_oc, d_ox, frac, frac_multi, n_sp, chance = _cos_nn(X, pids, lab)
            out[f"{m}/{zoo.PRETTY[e]}"] = {
                "n_images": int(len(X)), "n_images_with_same_patient_neighbour": n_sp,
                "nearest_same_patient": _q(d_sp), "nearest_other_patient_same_class": _q(d_oc),
                "nearest_other_patient_other_class": _q(d_ox),
                "frac_nn_same_patient": frac, "chance_frac_nn_same_patient": chance, "frac_nn_same_patient_among_multi_image": frac_multi}
            print(f"[dist] {m}/{e}: nn-same-patient {frac:.3f}", flush=True)
    save("feature_distance.json", {"metric": "cosine distance (1 - cos) to the nearest other image",
                                   "labels": "source labels", "per_encoder": out})


def modality_paired(store, test, final):
    pos = {p: i for i, p in enumerate(test)}
    out, enc = {}, {}
    for mod in ("opg", "cephalogram"):
        rows = []
        for e in zoo.ENCODERS:
            if e == "rad_dino" or f"{mod}/{e}/logreg" not in store:
                continue
            ids = [str(x) for x in store[f"{mod}/{e}/logreg#ids"]]
            ids = [p for p in ids if p in pos]
            idx = np.array([pos[p] for p in ids])
            io = R._align(store, f"intraoral/{e}/logreg", test)[idx].argmax(1)
            ra = R._align(store, f"{mod}/{e}/logreg", np.array(ids)).argmax(1)
            fi = final[idx]
            s_io, s_ra = float(V.scores(fi, io)[0][0]), float(V.scores(fi, ra)[0][0])
            rows.append((s_io, s_ra))
            enc[f"{mod}/{zoo.PRETTY[e]}"] = {"n": len(ids), "photo": s_io, "radiograph": s_ra, "family": FAMILY[e]}
        a = np.array(rows)
        out[mod] = {"n_encoders": len(rows), "zoo_mean_photo": float(a[:, 0].mean()),
                    "zoo_mean_radiograph": float(a[:, 1].mean())}
    rd = {}
    for mod in ("opg", "cephalogram"):
        ids = [str(x) for x in store[f"{mod}/rad_dino/logreg#ids"]]
        idx = np.array([pos[p] for p in ids if p in pos])
        ra = R._align(store, f"{mod}/rad_dino/logreg", np.array([p for p in ids if p in pos])).argmax(1)
        rd[mod] = float(V.scores(final[idx], ra)[0][0])
    save("modality_paired.json", {"label": "adjudicated", "head": "linear", "summary": out,
                                  "per_encoder": enc, "rad_dino_radiograph": rd})


def family_summary(manifest, final, entries):
    fam = {}
    for (e, h), runs in entries.items():
        preds = [p.argmax(1) for p in runs]
        f = "Fine-tuned" if h == "fine-tuned" else FAMILY[e]
        key = f"{f} / {h}"
        s = float(V.entry_scores(manifest, preds)[0]); t = float(V.entry_scores(final, preds)[0])
        fam.setdefault(key, []).append((s, t))
    out = {k: {"n_entries": len(v), "mean_source": float(np.mean([a for a, _ in v])),
               "mean_adjudicated": float(np.mean([b for _, b in v]))} for k, v in sorted(fam.items())}
    save("family_summary.json", {"metric": "patient macro-F1, 189 test patients", "groups": out})


def expert_gap_ci(ctx, test, manifest, final, entries):
    """C11 + C12 (amendment 2, 2026-11-14): paired intervals for the expert gap and the lead."""
    w = V.draw_weights(len(test), R.N_BOOT, R.SEED)          # identical draws to zoo_reference.json
    pr = ctx["primary"]
    ft = [p.argmax(1) for p in entries[("dinov2", "fine-tuned")]]
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    rec = {"bootstrap_replicates": R.N_BOOT, "bootstrap_seed": R.SEED, "n_test": len(test)}
    for r in ("expert1", "expert2"):
        raw = [pr[p][f"{r}_label"] for p in test]
        ex = np.array([LABELS.index(v) if v in LABELS[:3] else 3 for v in raw])
        ok = (ex < 3).astype(float)
        out = {}
        for rule, ww in (("all_na_as_error", w), ("assessable_only", w * ok[None, :])):
            e_b = V.scores(manifest, ex, ww)[0]
            m_b = V.entry_scores(manifest, ft, ww)
            g = e_b - m_b
            out[rule] = {"gap_ci95": ci(g), "share_draws_model_at_or_above_reviewer": float((g <= 0).mean())}
        rec[r] = out
    others = {"convnext_t_ft": entries[("convnext_tiny", "fine-tuned")],
              "resnet18_linear": entries[("resnet18", "logreg")]}
    lead = {}
    for nm, runs in others.items():
        for lab, truth in (("source", manifest), ("adjudicated", final)):
            d = V.entry_scores(truth, ft, w) - V.entry_scores(truth, [p.argmax(1) for p in runs], w)
            pt = float(V.entry_scores(truth, ft)[0] - V.entry_scores(truth, [p.argmax(1) for p in runs])[0])
            lead[f"dinov2s_ft_minus_{nm}_{lab}"] = {"point": pt, "ci95": ci(d)}
    rec["lead"] = lead
    save("expert_gap_ci.json", rec)


def fig2_frame_probs(ctx):
    rec = {"record": "the redacted record already shown in the primer figure (identifier withheld)",
           "in_reference_test_189": PRIMER_PATIENT in ctx["test_ids"],
           "audited": PRIMER_PATIENT in ctx["audit_ids"]}
    for m in ("intraoral", "opg", "cephalogram"):
        f = zoo.load_cache(CACHE, "dinov2", m)
        pids, y = f["patient_id"].astype(str), f["label_numeric"].astype(int)
        tr, _ = split_masks(pids, ctx["audit_ids"], ctx["test_ids"], ctx["quarantine"])
        sel = pids == PRIMER_PATIENT
        rec[f"{m}_in_training_pool"] = bool((tr & sel).any())
        trn = tr & ~sel                                   # never fit on the shown record
        order = np.argsort(f["record_id"][sel].astype(str))
        prob = logreg_probe(f["features"][trn], y[trn], f["features"][sel], class_weight="balanced", C=1.0)
        rec[f"{m}_frame_probs"] = [[float(v) for v in row] for row in prob[order]]
        rec["source_label"] = LABELS[int(y[sel][0])]
    rec["class_order"] = LABELS[:3]
    save("fig2_frame_probs.json", rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--only-gap", action="store_true", help="run only C11/C12 (amendment 2)")
    a = ap.parse_args()
    ctx = zoo.context(read_cases(a.workbook), CACHE)
    store, test, manifest, final, agreed, entries = entries_and_labels(ctx)
    expert_gap_ci(ctx, test, manifest, final, entries)
    if a.only_gap:
        return
    rank_stability(test, manifest, final, entries)
    expert_subset(ctx, test, manifest, entries)
    modality_paired(store, test, final)
    family_summary(manifest, final, entries)
    fig2_frame_probs(ctx)
    feature_distance()


if __name__ == "__main__":
    main()
