#!/usr/bin/env python3
"""CVPR 2027 experiments: encoder zoo under two evaluation variables.

  predict  --workbook W   fit every cached encoder x head (+ pruned, radiograph tracks)
                          on the reference split; per-patient predictions -> data/preds/
  analyze  --workbook W   leaderboard shift, label analyses, fusion -> results/*.json
  split                   image- vs patient-level split for the zoo -> results/split_zoo.json

Frozen analysis plan: ../cvpr27_plan.json (written before any score was computed).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orthobench import zoo, variables as V  # noqa: E402
from orthobench.reference_audit import read_cases, LABELS  # noqa: E402

CACHE = ROOT / "data/feat_cache"
PREDS = Path(os.environ.get("ZOO_PREDS", ROOT / "data/preds"))
RESULTS = Path(os.environ.get("ZOO_RESULTS", ROOT / "results"))
SEED = 20261108
N_BOOT = 2000


def available(modality):
    only = os.environ.get("ZOO_ONLY")  # debugging: comma-separated encoder subset
    encs = only.split(",") if only else zoo.ENCODERS
    return [e for e in zoo.ENCODERS if e in encs and (CACHE / f"{e}_{modality}.npz").exists()]


def save(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
    print(f"[saved] results/{name}", flush=True)


def cmd_predict(a):
    rows = read_cases(a.workbook)
    ctx = zoo.context(rows, CACHE)
    PREDS.mkdir(parents=True, exist_ok=True)
    out_path = PREDS / "zoo_preds.npz"
    store = dict(np.load(out_path, allow_pickle=True)) if out_path.exists() else {}
    info = json.loads((PREDS / "zoo_info.json").read_text()) if (PREDS / "zoo_info.json").exists() else {}
    for m in zoo.MODALITIES:
        for e in available(m):
            f = zoo.load_cache(CACHE, e, m)
            heads = zoo.HEADS if m == "intraoral" else ("logreg",)
            for h in heads:
                key = f"{m}/{e}/{h}"
                if key in store and not a.force:
                    continue
                t0 = time.monotonic()
                test, prob, meta = zoo.predict(f, h, ctx)
                store[key + "#ids"], store[key] = test, prob
                info[key] = {**meta, "n_test": len(test), "seconds": time.monotonic() - t0}
                print(f"[predict] {key}: {len(test)} test, {time.monotonic() - t0:.1f}s", flush=True)
            if m == "intraoral" and (f"{m}/{e}/pruned" not in store or a.force):
                test, prob, meta = zoo.predict_pruned(f, ctx)
                store[f"{m}/{e}/pruned#ids"], store[f"{m}/{e}/pruned"] = test, prob
                info[f"{m}/{e}/pruned"] = meta
                print(f"[predict] {m}/{e}/pruned: pruned {meta['n_pruned_patients']}", flush=True)
            np.savez(out_path, **store)
            (PREDS / "zoo_info.json").write_text(json.dumps(info, indent=1))


def _load_preds():
    store = dict(np.load(PREDS / "zoo_preds.npz", allow_pickle=True))
    ft = PREDS / "finetune_reference_preds.npz"
    if ft.exists():
        store.update(dict(np.load(ft, allow_pickle=True)))
    return store


def _align(store, key, order):
    ids = [str(x) for x in store[key + "#ids"]]
    pos = {p: i for i, p in enumerate(ids)}
    return np.stack([store[key][pos[p]] for p in order])


def cmd_analyze(a):
    rows = read_cases(a.workbook)
    ctx = zoo.context(rows, CACHE)
    store = _load_preds()
    info = json.loads((PREDS / "zoo_info.json").read_text())
    test = np.array(sorted(str(x) for x in store["intraoral/dinov2/logreg#ids"]))
    manifest, final, agreed = zoo.reference_labels(ctx, test)
    w = V.draw_weights(len(test), N_BOOT, SEED)
    changed = manifest != final

    # ---- leaderboard entries (intraoral track) ----
    entries, names = {}, []
    for e in available("intraoral"):
        for h in zoo.HEADS:
            entries[f"{zoo.PRETTY[e]} / {h}"] = [_align(store, f"intraoral/{e}/{h}", test)]
    ft_archs = sorted({k.split("/")[1] for k in store if k.startswith("finetune/") and not k.endswith("#ids")})
    for arch in ft_archs:
        runs = sorted(k for k in store if k.startswith(f"finetune/{arch}/") and not k.endswith("#ids"))
        entries[f"{zoo.PRETTY.get(arch, arch)} / fine-tuned"] = [_align(store, k, test) for k in runs]
    names = list(entries)
    preds = [[p.argmax(1) for p in entries[n]] for n in names]
    probs = [np.mean(entries[n], 0) for n in names]

    shift_f1 = V.leaderboard_shift(names, preds, manifest, final, w, metric=0)
    shift_bacc = V.leaderboard_shift(names, preds, manifest, final, w, metric=1)
    lin = [i for i, n in enumerate(names) if n.endswith("/ logreg") or n.endswith("fine-tuned")]
    shift_lin = V.leaderboard_shift([names[i] for i in lin], [preds[i] for i in lin], manifest, final, w)
    for blk in (shift_f1, shift_bacc, shift_lin):
        blk["pairs_significant"] = [p for p in blk["pairs"] if p["significant_shift"]]
        del blk["pairs"]
    for m in shift_f1["models"]:
        k = names.index(m["name"])
        p = preds[k]
        m["n_runs"] = len(p)
        m["changed_pred_manifest"] = float(np.mean([(q[changed] == manifest[changed]).mean() for q in p]))
        m["changed_pred_final"] = float(np.mean([(q[changed] == final[changed]).mean() for q in p]))
        m["acc_final_agreed"] = float(np.mean([(q[agreed] == final[agreed]).mean() for q in p]))
        m["acc_final_adjudicated"] = float(np.mean([(q[~agreed] == final[~agreed]).mean() for q in p]))
        if len(p) > 1:
            per_seed = [V.scores(final, q)[0][0] for q in p]
            m["final_per_seed"] = [float(x) for x in per_seed]
            m["manifest_per_seed"] = [float(V.scores(manifest, q)[0][0]) for q in p]

    # zoo-mean changed-case decomposition with bootstrap over changed patients
    ch_idx = np.where(changed)[0]
    pm = np.array([[(q[ch_idx] == manifest[ch_idx]).mean(0) for q in p] for p in preds], dtype=object)
    M = np.stack([np.mean([(q[ch_idx] == manifest[ch_idx]) for q in p], 0) for p in preds])  # models x changed
    F = np.stack([np.mean([(q[ch_idx] == final[ch_idx]) for q in p], 0) for p in preds])
    wc = V.draw_weights(len(ch_idx), N_BOOT, SEED + 1)
    bm, bf = (wc @ M.mean(0)) / len(ch_idx), (wc @ F.mean(0)) / len(ch_idx)
    A = np.stack([np.mean([(q == final) for q in p], 0) for p in preds]).mean(0)
    wa = V.draw_weights(len(test), N_BOOT, SEED + 2)
    agr = (wa * agreed) @ A / np.maximum(wa @ agreed, 1)
    adj = (wa * ~agreed) @ A / np.maximum(wa @ ~agreed, 1)
    decomposition = {
        "n_changed_test": int(changed.sum()), "n_test": len(test),
        "n_agreed_test": int(agreed.sum()), "n_adjudicated_test": int((~agreed).sum()),
        "zoo_mean_pred_equals_manifest": float(M.mean()), "ci_manifest": V.ci(bm),
        "zoo_mean_pred_equals_final": float(F.mean()), "ci_final": V.ci(bf),
        "zoo_mean_pred_neither": float(1 - M.mean() - F.mean()),
        "models_closer_to_final": int((F.mean(1) > M.mean(1)).sum()), "n_models": len(names),
        "zoo_mean_acc_final_agreed": float(A[agreed].mean()),
        "zoo_mean_acc_final_adjudicated": float(A[~agreed].mean()),
        "agreed_minus_adjudicated_ci": V.ci(agr - adj),
    }

    # ---- label-error detection: does low p(manifest) find the expert changes? ----
    det = {}
    lr_keys = [n for n in names if n.endswith("/ logreg") or n.endswith("fine-tuned")]
    for n in lr_keys:
        pr = probs[names.index(n)]
        s = 1 - pr[np.arange(len(test)), manifest]
        det[n] = {"auroc": V.auroc(s, changed), "auroc_ci": V.auroc_boot(s, changed)}
    ens_keys = [n for n in names if n.endswith("/ logreg")]
    ens = np.mean([probs[names.index(n)] for n in ens_keys], 0)
    s_ens = 1 - ens[np.arange(len(test)), manifest]
    # confident-learning flags on test cases with thresholds from out-of-fold training probabilities
    cl = {}
    for e in available("intraoral"):
        f = zoo.load_cache(CACHE, e, "intraoral")
        _, plab, oof = zoo.oof_train_probs(f, ctx)
        thr = np.array([oof[plab == c, c].mean() for c in range(3)])
        pr = probs[names.index(f"{zoo.PRETTY[e]} / logreg")]
        flag = np.array([(np.where(p >= thr)[0][np.argmax(p[p >= thr])] != g) if (p >= thr).any() else False
                         for p, g in zip(pr, manifest)])
        tp = int((flag & changed).sum())
        cl[zoo.PRETTY[e]] = {"n_flagged": int(flag.sum()), "true_positive": tp,
                             "precision": tp / max(int(flag.sum()), 1), "recall": tp / int(changed.sum())}
    detection = {"per_model": det, "ensemble_members": ens_keys,
                 "ensemble_auroc": V.auroc(s_ens, changed), "ensemble_auroc_ci": V.auroc_boot(s_ens, changed),
                 "base_rate_changed": float(changed.mean()), "confident_learning_flags": cl,
                 "cl_mean_precision": float(np.mean([v["precision"] for v in cl.values()])),
                 "cl_mean_recall": float(np.mean([v["recall"] for v in cl.values()]))}

    # ---- noise-robust training: CL-pruned refit vs unpruned (logreg) ----
    prune = {}
    dm, df = [], []
    for e in available("intraoral"):
        base = _align(store, f"intraoral/{e}/logreg", test).argmax(1)
        pru = _align(store, f"intraoral/{e}/pruned", test).argmax(1)
        b_m, b_f = V.scores(manifest, base, w)[0], V.scores(final, base, w)[0]
        p_m, p_f = V.scores(manifest, pru, w)[0], V.scores(final, pru, w)[0]
        dm.append(p_m - b_m); df.append(p_f - b_f)
        prune[zoo.PRETTY[e]] = {
            "manifest_base": float(V.scores(manifest, base)[0][0]), "manifest_pruned": float(V.scores(manifest, pru)[0][0]),
            "final_base": float(V.scores(final, base)[0][0]), "final_pruned": float(V.scores(final, pru)[0][0]),
            "delta_manifest_ci": V.ci(p_m - b_m), "delta_final_ci": V.ci(p_f - b_f),
            "gap_final_minus_manifest_ci": V.ci((p_f - b_f) - (p_m - b_m)),
            **{k: v for k, v in info[f"intraoral/{e}/pruned"].items() if k != "thresholds"}}
        prune[zoo.PRETTY[e]]["delta_manifest"] = prune[zoo.PRETTY[e]]["manifest_pruned"] - prune[zoo.PRETTY[e]]["manifest_base"]
        prune[zoo.PRETTY[e]]["delta_final"] = prune[zoo.PRETTY[e]]["final_pruned"] - prune[zoo.PRETTY[e]]["final_base"]
    dm, df = np.mean(dm, 0), np.mean(df, 0)
    pruning = {"per_encoder": prune,
               "zoo_mean_delta_manifest": float(np.mean([v["delta_manifest"] for v in prune.values()])),
               "zoo_mean_delta_final": float(np.mean([v["delta_final"] for v in prune.values()])),
               "zoo_mean_delta_manifest_ci": V.ci(dm), "zoo_mean_delta_final_ci": V.ci(df),
               "zoo_mean_gap_ci": V.ci(df - dm),
               "encoders_final_gain_exceeds_manifest_gain": int(sum(v["delta_final"] > v["delta_manifest"] for v in prune.values()))}

    # ---- oracle ceiling and human reference on all primary classifiable and on test ----
    pr = ctx["primary"]
    allp = sorted(pr)
    def lab(p, k):
        v = pr[p][k]
        return LABELS.index(v) if v in LABELS[:3] else 3  # not assessable -> wrong for every class
    human = {}
    for pop, ids in (("primary_221", allp), ("test_189", list(test))):
        man = np.array([lab(p, "manifest_label") for p in ids])
        fin = np.array([lab(p, "final_label") for p in ids])
        wp = V.draw_weights(len(ids), N_BOOT, SEED + 3)
        rec = {"n": len(ids)}
        o = V.scores(man, fin, wp)
        rec["oracle_final_vs_manifest_macro_f1"] = float(V.scores(man, fin)[0][0])
        rec["oracle_final_vs_manifest_ci"] = V.ci(o[0])
        rec["oracle_final_vs_manifest_bacc"] = float(V.scores(man, fin)[1][0])
        for r in ("expert1", "expert2"):
            ex = np.array([lab(p, f"{r}_label") for p in ids])
            rec[f"{r}_vs_manifest_macro_f1"] = float(V.scores(man, ex)[0][0])
            rec[f"{r}_vs_manifest_ci"] = V.ci(V.scores(man, ex, wp)[0])
            rec[f"{r}_not_assessable"] = int((ex == 3).sum())
        e1 = np.array([lab(p, "expert1_label") for p in ids]); e2 = np.array([lab(p, "expert2_label") for p in ids])
        both = (e1 < 3) & (e2 < 3)
        rec["expert1_vs_expert2_macro_f1_both_assessable"] = float(V.scores(e1[both], e2[both])[0][0])
        rec["n_both_assessable"] = int(both.sum())
        human[pop] = rec

    # ---- exploratory (added after the frozen plan): partial correction dose-response ----
    from scipy.stats import kendalltau
    rng = np.random.default_rng(SEED + 5)
    base = np.array([V.entry_scores(manifest, p)[0] for p in preds])
    top0 = int(base.argmax())
    dose = []
    for q in np.linspace(0, 1, 11):
        taus, same = [], []
        for _ in range(200):
            ref = manifest.copy()
            pick = ch_idx[rng.random(len(ch_idx)) < q] if 0 < q < 1 else (ch_idx if q == 1 else ch_idx[:0])
            ref[pick] = final[pick]
            sc = np.array([V.entry_scores(ref, p)[0] for p in preds])
            taus.append(kendalltau(base, sc).statistic); same.append(int(sc.argmax()) == top0)
        dose.append({"fraction_corrected": float(q), "kendall_tau_mean": float(np.mean(taus)),
                     "kendall_tau_ci": V.ci(np.array(taus)), "p_same_top": float(np.mean(same))})
    partial = {"status": "exploratory, added after the frozen plan", "draws_per_level": 200, "curve": dose}

    best_manifest = max(shift_f1["models"], key=lambda m: m["score_ref_a"])
    best_final = max(shift_f1["models"], key=lambda m: m["score_ref_b"])
    save("zoo_reference.json", {
        "provenance": "run", "plan": "experiments/cvpr27_plan.json", "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED, "n_test": len(test), "n_entries": len(names),
        "test_manifest_support": np.bincount(manifest, minlength=3).tolist(),
        "test_final_support": np.bincount(final, minlength=3).tolist(),
        "best_manifest": {"name": best_manifest["name"], "score": best_manifest["score_ref_a"]},
        "best_final": {"name": best_final["name"], "score": best_final["score_ref_b"]},
        "ref_a": "manifest", "ref_b": "final",
        "config": {"input_px": {e: __import__("orthobench.features", fromlist=["x"]).BACKBONES[e][3] for e in zoo.ENCODERS},
                   "logreg": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"}, "knn_k": [1, 5],
                   "finetune": {"epochs": 8, "batch": 16, "batch_planned": 32, "weight_decay": 0.05, "cache_px": 256, "crop_px": 224,
                                "lr_head": 1e-3, "lr_backbone": {"dinov2": 1e-5, "resnet50": 1e-4, "convnext_tiny": 1e-4},
                                "seeds": [0, 1, 2]},
                   "quarantine_hamming": 6, "cl_folds": 5},
        "macro_f1": shift_f1, "balanced_accuracy": shift_bacc, "linear_and_finetuned_only": shift_lin,
        "changed_case_decomposition": decomposition, "partial_correction": partial, "training_info": {k: v for k, v in info.items() if k.startswith("intraoral/")},
    })
    save("label_analysis.json", {"provenance": "run", "plan": "experiments/cvpr27_plan.json",
                                 "human_and_oracle": human, "label_error_detection": detection,
                                 "confident_learning_pruning": pruning})
    cmd_fusion(ctx, store, test, manifest, final, w)


def cmd_fusion(ctx, store, test, manifest, final, w):
    out = {"provenance": "run", "per_encoder": {}}
    rad = {m: [str(x) for x in store.get(f"{m}/dinov2/logreg#ids", [])] for m in ("opg", "cephalogram")}
    have = {m: set(v) for m, v in rad.items()}
    out["n_test_with_opg"] = int(sum(p in have["opg"] for p in test))
    out["n_test_with_ceph"] = int(sum(p in have["cephalogram"] for p in test))
    gains_m, gains_f = [], []
    for e in available("intraoral"):
        if not all(f"{m}/{e}/logreg" in store for m in ("opg", "cephalogram")):
            continue
        io = _align(store, f"intraoral/{e}/logreg", test)
        maps = {m: dict(zip([str(x) for x in store[f"{m}/{e}/logreg#ids"]], store[f"{m}/{e}/logreg"]))
                for m in ("opg", "cephalogram")}
        fused = np.stack([np.mean([io[i]] + [maps[m][p] for m in maps if p in maps[m]], 0)
                         for i, p in enumerate(test)])
        rec = {}
        for nm, ref in (("manifest", manifest), ("final", final)):
            a0, a1 = V.scores(ref, io.argmax(1), w)[0], V.scores(ref, fused.argmax(1), w)[0]
            rec[nm] = {"intraoral": float(V.scores(ref, io.argmax(1))[0][0]),
                       "late_fusion": float(V.scores(ref, fused.argmax(1))[0][0]),
                       "delta_ci": V.ci(a1 - a0)}
            rec[nm]["delta"] = rec[nm]["late_fusion"] - rec[nm]["intraoral"]
            (gains_m if nm == "manifest" else gains_f).append(a1 - a0)
        out["per_encoder"][zoo.PRETTY[e]] = rec
    if gains_f:
        out["zoo_mean_delta_final"] = float(np.mean([v["final"]["delta"] for v in out["per_encoder"].values()]))
        out["zoo_mean_delta_final_ci"] = V.ci(np.mean(gains_f, 0))
        out["zoo_mean_delta_manifest"] = float(np.mean([v["manifest"]["delta"] for v in out["per_encoder"].values()]))
        out["zoo_mean_delta_manifest_ci"] = V.ci(np.mean(gains_m, 0))
        out["encoders_fusion_better_final"] = int(sum(v["final"]["delta"] > 0 for v in out["per_encoder"].values()))
    # modality headroom on identical patients (zoo-mean logreg)
    head = {}
    for m in ("opg", "cephalogram"):
        encs = [e for e in available("intraoral") if f"{m}/{e}/logreg" in store]
        if not encs:
            continue
        ids = [p for p in test if p in have[m]]
        idx = np.array([list(test).index(p) for p in ids])
        man, fin = manifest[idx], final[idx]
        wm = V.draw_weights(len(ids), N_BOOT, SEED + 4)
        sio, srad = [], []
        for e in encs:
            io = _align(store, f"intraoral/{e}/logreg", np.array(ids)).argmax(1)
            rd = _align(store, f"{m}/{e}/logreg", np.array(ids)).argmax(1)
            sio.append(V.scores(fin, io, wm)[0]); srad.append(V.scores(fin, rd, wm)[0])
        head[m] = {"n_patients": len(ids), "n_encoders": len(encs),
                   "zoo_mean_intraoral_final": float(np.mean([V.scores(fin, _align(store, f"intraoral/{e}/logreg", np.array(ids)).argmax(1))[0][0] for e in encs])),
                   f"zoo_mean_{m}_final": float(np.mean([V.scores(fin, _align(store, f"{m}/{e}/logreg", np.array(ids)).argmax(1))[0][0] for e in encs])),
                   "delta_radiograph_minus_intraoral_ci": V.ci(np.mean(srad, 0) - np.mean(sio, 0)),
                   "best_encoder_" + m: max(encs, key=lambda e: V.scores(fin, _align(store, f"{m}/{e}/logreg", np.array(ids)).argmax(1))[0][0])}
    out["modality_headroom"] = head
    # every radiograph-track entry on its own test cohort (includes RAD-DINO, radiographs only)
    rt = {}
    for m in ("opg", "cephalogram"):
        for e in zoo.ENCODERS:
            k = f"{m}/{e}/logreg"
            if k not in store:
                continue
            ids = np.array([str(x) for x in store[k + "#ids"]])
            pr = ctx["primary"]
            fin = np.array([LABELS.index(pr[p]["final_label"]) for p in ids])
            man = np.array([LABELS.index(pr[p]["manifest_label"]) for p in ids])
            wr = V.draw_weights(len(ids), N_BOOT, SEED + 6)
            pred = store[k].argmax(1)
            rt[f"{m}/{zoo.PRETTY[e]}"] = {"n_test": len(ids), "final_macro_f1": float(V.scores(fin, pred)[0][0]),
                                          "final_ci": V.ci(V.scores(fin, pred, wr)[0]),
                                          "manifest_macro_f1": float(V.scores(man, pred)[0][0])}
    out["radiograph_tracks"] = rt
    save("fusion_late.json", out)


def cmd_split(a):
    from orthobench import splits, probes, evaluate as ev
    res = {"provenance": "run", "plan": "experiments/cvpr27_plan.json", "seeds": [0, 1, 2, 3, 4],
           "test_frac": 0.3, "scoring": "per image, source labels", "entries": {}}
    for e in available("intraoral"):
        f = zoo.load_cache(CACHE, e, "intraoral")
        X, pid, y = f["features"], f["patient_id"].astype(str), f["label_numeric"].astype(int)
        for h in zoo.HEADS:
            img, pat = [], []
            for s in range(5):
                tr, te = splits.patient_level_split(pid, y, seed=s, test_frac=0.3)
                pat.append(ev.image_metrics(y[te], probes.run_probe("logreg" if h == "logreg" else "knn", X[tr], y[tr], X[te], k=int(h[3:]) if h != "logreg" else 1).argmax(1))["macro_f1"])
                tri, tei = splits.image_level_split(y, seed=1000 + s, test_frac=0.3)
                img.append(ev.image_metrics(y[tei], probes.run_probe("logreg" if h == "logreg" else "knn", X[tri], y[tri], X[tei], k=int(h[3:]) if h != "logreg" else 1).argmax(1))["macro_f1"])
            d = np.array(img) - np.array(pat)
            res["entries"][f"{zoo.PRETTY[e]} / {h}"] = {
                "image_split": float(np.mean(img)), "patient_split": float(np.mean(pat)),
                "image_split_per_seed": [float(x) for x in img], "patient_split_per_seed": [float(x) for x in pat],
                "inflation": float(d.mean()), "inflation_sd": float(d.std(ddof=1)), "all_seeds_positive": bool((d > 0).all())}
            print(f"[split] {e}/{h}: img {np.mean(img):.3f} pat {np.mean(pat):.3f}", flush=True)
    from scipy.stats import kendalltau
    names = list(res["entries"])
    A = np.array([res["entries"][n]["image_split"] for n in names])
    B = np.array([res["entries"][n]["patient_split"] for n in names])
    per_seed = [kendalltau([res["entries"][n]["image_split_per_seed"][s] for n in names],
                           [res["entries"][n]["patient_split_per_seed"][s] for n in names]).statistic for s in range(5)]
    rank_img, rank_pat = (-A).argsort().argsort() + 1, (-B).argsort().argsort() + 1
    for n, ri, rp in zip(names, rank_img, rank_pat):
        res["entries"][n]["rank_image_split"], res["entries"][n]["rank_patient_split"] = int(ri), int(rp)
    infl = {h: [res["entries"][n]["inflation"] for n in names if n.endswith("/ " + h)] for h in zoo.HEADS}
    res.update({"kendall_tau_point": float(kendalltau(A, B).statistic), "kendall_tau_per_seed": [float(t) for t in per_seed],
                "top_image_split": names[int(A.argmax())], "top_patient_split": names[int(B.argmax())],
                "mean_inflation_by_head": {h: float(np.mean(v)) for h, v in infl.items()},
                "min_inflation_by_head": {h: float(np.min(v)) for h, v in infl.items()},
                "max_inflation_by_head": {h: float(np.max(v)) for h, v in infl.items()},
                "top10_image_split_knn_count": int(sum(names[i].endswith(("knn1", "knn5")) for i in (-A).argsort()[:10])),
                "top10_patient_split_knn_count": int(sum(names[i].endswith(("knn1", "knn5")) for i in (-B).argsort()[:10])),
                "n_entries": len(names)})
    save("split_zoo.json", res)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["predict", "analyze", "split"])
    ap.add_argument("--workbook", type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"predict": cmd_predict, "analyze": cmd_analyze, "split": cmd_split}[a.cmd](a)


if __name__ == "__main__":
    main()
