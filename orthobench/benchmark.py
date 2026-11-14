"""Benchmark experiments: leakage inflation, baselines, ablations.

Every function consumes an in-memory feature dict (features, patient_id,
label_numeric, ...) and returns a JSON-serialisable results dict. No restricted
image is touched here -- only cached frozen features and opaque IDs.
"""
import numpy as np
from . import splits, probes, evaluate as ev


def _pl(feat):
    return feat["features"].astype(np.float64), feat["patient_id"].astype(str), \
        feat["label_numeric"].astype(int)


def _patient_label_map(pid, y):
    return {p: int(y[pid == p][0]) for p in np.unique(pid)}


def leakage_experiment(feat, seeds=(0, 1, 2, 3, 4), test_frac=0.3):
    """Image-level vs patient-level splits for logreg, 1-NN, 5-NN probes."""
    X, pid, y = _pl(feat)
    plabel = _patient_label_map(pid, y)
    configs = [("logreg", {}), ("knn1", {"k": 1}), ("knn5", {"k": 5})]
    out = {"n_images": int(len(y)), "n_patients": int(len(plabel)),
           "seeds": list(seeds), "test_frac": test_frac, "probes": {}}
    for name, kw in configs:
        rec = {"image_level": [], "patient_level": [], "patient_agg": []}
        for s in seeds:
            # patient-level (correct)
            tr, te = splits.patient_level_split(pid, y, seed=s, test_frac=test_frac)
            prob = probes.run_probe("logreg" if name == "logreg" else "knn",
                                    X[tr], y[tr], X[te], **kw)
            pred = prob.argmax(1)
            m_img = ev.image_metrics(y[te], pred)
            at, ap = ev.aggregate_to_patient(prob, pid[te], plabel)
            m_agg = ev.image_metrics(at, ap)
            # image-level (naive/leaky), paired but disjoint seed stream
            tri, tei = splits.image_level_split(y, seed=1000 + s, test_frac=test_frac)
            probi = probes.run_probe("logreg" if name == "logreg" else "knn",
                                     X[tri], y[tri], X[tei], **kw)
            m_lk = ev.image_metrics(y[tei], probi.argmax(1))
            rec["image_level"].append(m_lk)
            rec["patient_level"].append(m_img)
            rec["patient_agg"].append(m_agg)
        summ = {}
        for regime in rec:
            f1s = [r["macro_f1"] for r in rec[regime]]
            accs = [r["accuracy"] for r in rec[regime]]
            summ[regime] = {
                "macro_f1": ev.mean_std(f1s),
                "accuracy": ev.mean_std(accs),
                # Per-seed values persisted so the appendix per-seed table is
                # regenerable from this JSON (results-persistence rule).
                "per_seed": {"seed": list(seeds),
                             "macro_f1": [round(float(x), 6) for x in f1s],
                             "accuracy": [round(float(x), 6) for x in accs]}}
        il = summ["image_level"]["macro_f1"]["mean"]
        pl = summ["patient_level"]["macro_f1"]["mean"]
        ila = summ["image_level"]["accuracy"]["mean"]
        pla = summ["patient_level"]["accuracy"]["mean"]
        summ["inflation"] = {
            "macro_f1_abs": round(il - pl, 4),
            "macro_f1_rel_pct": round((il - pl) / pl * 100, 1) if pl else None,
            "accuracy_abs": round(ila - pla, 4),
            "accuracy_rel_pct": round((ila - pla) / pla * 100, 1) if pla else None}
        out["probes"][name] = summ
    return out


def baselines(feat_by_backbone, seeds=(0, 1, 2, 3, 4), test_frac=0.3):
    """Patient-level linear-probe baselines per backbone, plus trivial baselines,
    with patient-bootstrap CIs on a representative seed."""
    out = {"seeds": list(seeds), "backbones": {}, "trivial": {}}
    ref = next(iter(feat_by_backbone.values()))
    _, pid, y = _pl(ref)
    plabel = _patient_label_map(pid, y)
    # trivial baselines at patient level (seed 0 split)
    tr, te = splits.patient_level_split(pid, y, seed=0, test_frac=test_frac)
    te_pid = pid[te]
    at = np.array([plabel[p] for p in np.unique(te_pid)])
    maj = np.bincount(y[tr]).argmax()
    out["trivial"]["majority_class"] = ev.image_metrics(at, np.full_like(at, maj))
    rng = np.random.RandomState(0)
    prior = np.bincount(y[tr]) / len(y[tr])
    rand = rng.choice(3, size=len(at), p=prior)
    out["trivial"]["stratified_random"] = ev.image_metrics(at, rand)
    for bk, feat in feat_by_backbone.items():
        X, pidb, yb = _pl(feat)
        pmap = _patient_label_map(pidb, yb)
        f1s, accs, pcs = [], [], []
        boot = None
        for s in seeds:
            tr, te = splits.patient_level_split(pidb, yb, seed=s, test_frac=test_frac)
            prob = probes.logreg_probe(X[tr], yb[tr], X[te])
            a_t, a_p = ev.aggregate_to_patient(prob, pidb[te], pmap)
            m = ev.image_metrics(a_t, a_p)
            f1s.append(m["macro_f1"]); accs.append(m["accuracy"])
            pcs.append(list(m["per_class_f1"]))
            if s == seeds[0]:
                boot = ev.patient_bootstrap_ci(a_t, a_p)
                per_class = m["per_class_f1"]
                pc_boot = ev.per_class_bootstrap_ci(a_t, a_p)
                conf = ev.confusion(a_t, a_p)
                # Persist the two derived views the paper actually prints, so the
                # number audit traces them instead of matching a count by accident.
                conf_pct = [[round(100.0 * c / max(sum(row), 1), 1) for c in row]
                            for row in conf]
                ci_half = [round((r["f1_ci"][1] - r["f1_ci"][0]) / 2.0, 4)
                           for r in pc_boot]
        # Per-class F1 over ALL seeds: single-seed per-class numbers on a 115-patient
        # minority class are far too noisy to carry a claim.
        pca = np.asarray(pcs, dtype=float)
        out["backbones"][bk] = {
            "patient_macro_f1": ev.mean_std(f1s),
            "patient_accuracy": ev.mean_std(accs),
            "per_class_f1_seed0": per_class,
            "per_class_f1_mean": [round(float(v), 4) for v in pca.mean(axis=0)],
            "per_class_f1_std": [round(float(v), 4) for v in pca.std(axis=0)],
            "per_class_f1_per_seed": [[round(float(v), 4) for v in row] for row in pca],
            "patient_bootstrap_ci_seed0": boot,
            "per_class_bootstrap_ci_seed0": pc_boot,
            "per_class_ci_halfwidth_seed0": ci_half,
            "confusion_seed0": conf,
            "confusion_row_pct_seed0": conf_pct,
            # Seeds index the patient partition identically across backbones, so
            # these vectors support PAIRED comparisons; overlapping error bars on
            # unpaired means understate what the suite can actually resolve.
            "per_seed_macro_f1": [round(float(x), 6) for x in f1s]}
    ref = max(out["backbones"], key=lambda b: out["backbones"][b]["patient_macro_f1"]["mean"])
    out["paired_vs_best"] = {"best": ref, "comparisons": {}}
    a = np.array(out["backbones"][ref]["per_seed_macro_f1"])
    for bk, v in out["backbones"].items():
        if bk == ref:
            continue
        d = a - np.array(v["per_seed_macro_f1"])
        out["paired_vs_best"]["comparisons"][bk] = {
            "mean_gap": round(float(d.mean()), 4), "std_gap": round(float(d.std()), 4),
            "wins": int((d > 0).sum()), "n_seeds": int(len(d)),
            "exceeds_paired_noise": bool(abs(d.mean()) > 2 * float(d.std()))}
    return out


def ablations(feat_dino, feat_resnet=None, seeds=(0, 1, 2, 3, 4)):
    X, pid, y = _pl(feat_dino)
    pmap = _patient_label_map(pid, y)
    res = {}

    def pat_lp(feat, class_weight="balanced", seedset=seeds):
        Xa, pa, ya = _pl(feat); pm = _patient_label_map(pa, ya)
        f1 = []
        for s in seedset:
            tr, te = splits.patient_level_split(pa, ya, seed=s)
            prob = probes.logreg_probe(Xa[tr], ya[tr], Xa[te], class_weight=class_weight)
            at, ap = ev.aggregate_to_patient(prob, pa[te], pm)
            f1.append(ev.image_metrics(at, ap)["macro_f1"])
        return ev.mean_std(f1)

    # A) class weighting
    res["class_weight"] = {"balanced": pat_lp(feat_dino, "balanced"),
                           "none": pat_lp(feat_dino, None)}
    # B) feature extractor
    res["feature_extractor"] = {"dinov2": pat_lp(feat_dino)}
    if feat_resnet is not None:
        res["feature_extractor"]["resnet50"] = pat_lp(feat_resnet)
    # C) aggregation: single-image vs mean-prob (patient-level split)
    single, agg = [], []
    for s in seeds:
        tr, te = splits.patient_level_split(pid, y, seed=s)
        prob = probes.logreg_probe(X[tr], y[tr], X[te])
        single.append(ev.image_metrics(y[te], prob.argmax(1))["macro_f1"])
        at, ap = ev.aggregate_to_patient(prob, pid[te], pmap)
        agg.append(ev.image_metrics(at, ap)["macro_f1"])
    res["aggregation"] = {"single_image": ev.mean_std(single),
                          "mean_prob_patient": ev.mean_std(agg)}
    # D) data efficiency (patients per class in train)
    res["data_efficiency"] = {}
    for npc in [10, 25, 50, 100]:
        f1 = []
        for s in seeds:
            tr, te = splits.few_shot_patient_split(pid, y, seed=s, n_per_class=npc)
            if len(tr) < 10:
                continue
            prob = probes.logreg_probe(X[tr], y[tr], X[te])
            at, ap = ev.aggregate_to_patient(prob, pid[te], pmap)
            f1.append(ev.image_metrics(at, ap)["macro_f1"])
        if f1:
            res["data_efficiency"][f"{npc}_per_class"] = ev.mean_std(f1)
    return res


def modality_tracks(feat_by_modality, seeds=(0, 1, 2, 3, 4), test_frac=0.3):
    """One benchmark track per imaging modality, scored under the same protocol.

    This is what turns a single-modality headline into a multimodal benchmark: the
    Angle-class task is defined identically on intraoral photographs, panoramic
    OPG and lateral cephalograms, so per-modality difficulty is comparable. It
    also isolates the leakage mechanism, because the modalities differ sharply in
    images per patient -- the quantity the same-patient shortcut needs.
    """
    out = {"seeds": list(seeds), "test_frac": test_frac, "modalities": {}}
    for mod, feat in feat_by_modality.items():
        X, pid, y = _pl(feat)
        pmap = _patient_label_map(pid, y)
        n_pat = len(pmap)
        rec = {"n_images": int(len(y)), "n_patients": int(n_pat),
               "images_per_patient": round(len(y) / n_pat, 3),
               "label_distribution": {int(c): int((np.array(list(pmap.values())) == c).sum())
                                      for c in range(3)}}
        f1_pat, f1_agg, f1_img, boot = [], [], [], None
        for s in seeds:
            tr, te = splits.patient_level_split(pid, y, seed=s, test_frac=test_frac)
            prob = probes.logreg_probe(X[tr], y[tr], X[te])
            f1_pat.append(ev.image_metrics(y[te], prob.argmax(1))["macro_f1"])
            at, ap = ev.aggregate_to_patient(prob, pid[te], pmap)
            m = ev.image_metrics(at, ap)
            f1_agg.append(m["macro_f1"])
            if s == seeds[0]:
                boot = ev.patient_bootstrap_ci(at, ap)
                per_class = m["per_class_f1"]
            tri, tei = splits.image_level_split(y, seed=1000 + s, test_frac=test_frac)
            probi = probes.logreg_probe(X[tri], y[tri], X[tei])
            f1_img.append(ev.image_metrics(y[tei], probi.argmax(1))["macro_f1"])
        rec["patient_level_macro_f1"] = ev.mean_std(f1_pat)
        rec["patient_agg_macro_f1"] = ev.mean_std(f1_agg)
        rec["image_level_macro_f1"] = ev.mean_std(f1_img)
        rec["per_class_f1_seed0"] = per_class
        rec["patient_bootstrap_ci_seed0"] = boot
        diff = np.array(f1_img) - np.array(f1_pat)
        inf = float(diff.mean())
        # The per-seed spread of the DIFFERENCE is what decides whether an
        # inflation is real. With ~1 image per patient the two split protocols
        # coincide, so the gap should sit inside this spread.
        rec["leakage_inflation"] = {
            "macro_f1_abs": round(inf, 4), "macro_f1_std": round(float(diff.std()), 4),
            "macro_f1_rel_pct": round(100 * inf / rec["patient_level_macro_f1"]["mean"], 1)
            if rec["patient_level_macro_f1"]["mean"] else None,
            "exceeds_seed_noise": bool(abs(inf) > 2 * float(diff.std()))}
        rec["per_seed_macro_f1"] = {
            "seed": list(seeds),
            "patient_level": [round(float(x), 6) for x in f1_pat],
            "patient_agg": [round(float(x), 6) for x in f1_agg],
            "image_level": [round(float(x), 6) for x in f1_img]}
        out["modalities"][mod] = rec
    return out


def _patient_matrix(feat):
    """Collapse a modality's image features to one L2-normalised vector per patient."""
    X, pid, y = _pl(feat)
    pats = np.array(sorted(set(pid.tolist())))
    M = np.zeros((len(pats), X.shape[1]))
    for i, p in enumerate(pats):
        M[i] = X[pid == p].mean(0)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    lab = {p: int(y[pid == p][0]) for p in pats}
    return pats, M, np.array([lab[p] for p in pats])


def fusion(feat_by_modality, seeds=tuple(range(10)), test_frac=0.3):
    """Multimodal fusion and missing-modality study on fully-paired patients.

    Every track is scored on the SAME patient set (those carrying all three
    modalities) so unimodal and fused numbers are directly comparable rather than
    confounded by cohort size. Missing-modality rows mean-impute the absent block
    from the training fold, which is the deployment case where one record is not
    on file.
    """
    mods = [m for m in ("intraoral", "opg", "cephalogram") if m in feat_by_modality]
    blocks = {m: _patient_matrix(feat_by_modality[m]) for m in mods}
    common = set(blocks[mods[0]][0].tolist())
    for m in mods[1:]:
        common &= set(blocks[m][0].tolist())
    pats = np.array(sorted(common))
    B, y = {}, None
    for m in mods:
        p_all, M, ya = blocks[m]
        sel = np.searchsorted(p_all, pats)
        B[m] = M[sel]
        y = ya[sel] if y is None else y
    out = {"seeds": list(seeds), "test_frac": test_frac, "modalities": mods,
           "n_patients_fully_paired": int(len(pats)),
           "label_distribution": {int(c): int((y == c).sum()) for c in range(3)},
           "tracks": {}}
    combos = [(m,) for m in mods] + [
        tuple(c) for c in [("intraoral", "opg"), ("intraoral", "cephalogram"),
                           ("opg", "cephalogram"), tuple(mods)]
        if all(x in mods for x in c) and len(c) > 1]
    pid_dummy = pats.astype(str)
    for combo in combos:
        f1, boot, per_class = [], None, None
        for s in seeds:
            tr, te = splits.patient_level_split(pid_dummy, y, seed=s, test_frac=test_frac)
            Xtr = np.hstack([B[m][tr] for m in combo])
            Xte = np.hstack([B[m][te] for m in combo])
            prob = probes.logreg_probe(Xtr, y[tr], Xte)
            m_ = ev.image_metrics(y[te], prob.argmax(1))
            f1.append(m_["macro_f1"])
            if s == seeds[0]:
                boot = ev.patient_bootstrap_ci(y[te], prob.argmax(1))
                per_class = m_["per_class_f1"]
        out["tracks"]["+".join(combo)] = {
            "macro_f1": ev.mean_std(f1), "per_class_f1_seed0": per_class,
            "patient_bootstrap_ci_seed0": boot, "n_blocks": len(combo),
            "per_seed_macro_f1": [round(float(x), 6) for x in f1]}
    # missing-modality at test time: train on all three, blank one block
    out["missing_modality"] = {}
    for drop in mods:
        f1 = []
        for s in seeds:
            tr, te = splits.patient_level_split(pid_dummy, y, seed=s, test_frac=test_frac)
            Xtr = np.hstack([B[m][tr] for m in mods])
            imp = {m: B[m][tr].mean(0) for m in mods}
            Xte = np.hstack([np.tile(imp[m], (len(te), 1)) if m == drop else B[m][te]
                             for m in mods])
            prob = probes.logreg_probe(Xtr, y[tr], Xte)
            f1.append(ev.image_metrics(y[te], prob.argmax(1))["macro_f1"])
        out["missing_modality"][f"drop_{drop}"] = ev.mean_std(f1)
    full = out["tracks"]["+".join(tuple(mods))]["macro_f1"]["mean"]
    best_uni = max(out["tracks"][m]["macro_f1"]["mean"] for m in mods)
    out["fusion_gain_over_best_unimodal"] = {
        "abs": round(full - best_uni, 4),
        "rel_pct": round(100 * (full - best_uni) / best_uni, 1) if best_uni else None}
    return out


def redaction_ablation(feat_plain, feat_redacted, seeds=(0, 1, 2, 3, 4), test_frac=0.3):
    """Does a release-safe border redaction cost benchmark accuracy?

    If blanking the frame margin -- where burned-in identifiers live -- leaves the
    patient-level score unchanged, the cohort can be released margin-redacted
    without waiting for per-image PHI adjudication. That makes this ablation a
    release-policy decision, not just a robustness check.
    """
    res = {"seeds": list(seeds), "variants": {}}
    for name, feat in (("plain", feat_plain), ("margin_redacted", feat_redacted)):
        X, pid, y = _pl(feat)
        pmap = _patient_label_map(pid, y)
        f1 = []
        for s in seeds:
            tr, te = splits.patient_level_split(pid, y, seed=s, test_frac=test_frac)
            prob = probes.logreg_probe(X[tr], y[tr], X[te])
            at, ap = ev.aggregate_to_patient(prob, pid[te], pmap)
            f1.append(ev.image_metrics(at, ap)["macro_f1"])
        res["variants"][name] = {"patient_macro_f1": ev.mean_std(f1),
                                 "n_images": int(len(y)),
                                 "redact_margin": float(feat.get("redact_margin", 0) or 0),
                                 "per_seed": [round(float(x), 6) for x in f1]}
    a = res["variants"]["plain"]
    b = res["variants"]["margin_redacted"]
    # Seeds are paired (same patient partitions), so the per-seed difference is the
    # right statistic for "does redaction cost anything".
    d = np.array(b["per_seed"]) - np.array(a["per_seed"])
    res["delta"] = {
        "macro_f1_abs": round(float(d.mean()), 4),
        "macro_f1_std": round(float(d.std()), 4),
        "per_seed": [round(float(x), 6) for x in d],
        "seeds_not_worse": int((d >= -1e-6).sum()), "n_seeds": int(len(d)),
        "within_one_seed_sd": bool(abs(d.mean()) <= a["patient_macro_f1"]["std"])}
    return res


def significance(leakage, finetune, modality):
    """Paired significance of every leakage inflation we report.

    Seeds index the patient partition identically in both regimes, so the per-seed
    DIFFERENCE is the paired statistic. We report three things per arm, because at
    3-5 seeds no single one is sufficient on its own:
      - a paired t-test on the per-seed differences (parametric, small n),
      - an exact paired sign test (distribution-free, but weak at n=3),
      - an exact paired permutation test over all 2^n sign flips (the honest
        small-sample choice: it makes no normality assumption and is exact).
    All are one-sided against H0: inflation <= 0.
    """
    import itertools
    import math

    def _tests(d):
        d = [float(x) for x in d]
        n = len(d)
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
        sd = math.sqrt(var)
        se = sd / math.sqrt(n) if sd > 0 else 0.0
        t = mean / se if se > 0 else float("inf")
        # scipy ships with scikit-learn, which is already a hard requirement
        from scipy import stats as _st
        p_t = float(_st.t.sf(t, n - 1)) if se > 0 and n > 1 else 0.0
        # exact one-sided paired permutation test over sign flips
        obs = sum(d)
        ge = sum(1 for signs in itertools.product((1, -1), repeat=n)
                 if sum(s * x for s, x in zip(signs, d)) >= obs)
        p_perm = ge / (2 ** n)
        n_pos = sum(1 for x in d if x > 0)
        p_sign = sum(math.comb(n, k) for k in range(n_pos, n + 1)) / (2 ** n)
        return {"n_seeds": n, "mean_diff": round(mean, 6), "sd_diff": round(sd, 6),
                "t_stat": round(t, 3) if se > 0 else None,
                "p_t_one_sided": round(p_t, 5),
                "p_permutation_one_sided": round(p_perm, 5),
                "p_sign_one_sided": round(p_sign, 5),
                "seeds_positive": f"{n_pos}/{n}", "per_seed_diff": [round(x, 6) for x in d]}

    out = {"note": "One-sided paired tests of H0: leakage inflation <= 0. The exact "
                   "permutation test over sign flips is the primary statistic; its "
                   "floor at n seeds is 2^-n (0.031 at 5 seeds, 0.125 at 3).",
           "arms": {}}
    for name, key in (("linear_probe", "logreg"), ("knn5", "knn5"), ("knn1", "knn1")):
        pr = leakage["probes"][key]
        d = [i - p for i, p in zip(pr["image_level"]["per_seed"]["macro_f1"],
                                   pr["patient_level"]["per_seed"]["macro_f1"])]
        out["arms"][name] = _tests(d)
    r = finetune["regimes"]
    out["arms"]["finetuned"] = _tests(
        [i - p for i, p in zip(r["image_level"]["per_seed"]["macro_f1"],
                               r["patient_level"]["per_seed"]["macro_f1"])])
    for m, rec in modality["modalities"].items():
        ps = rec["per_seed_macro_f1"]
        out["arms"]["modality_" + m] = _tests(
            [i - p for i, p in zip(ps["image_level"], ps["patient_level"])])
        out["arms"]["modality_" + m]["images_per_patient"] = rec["images_per_patient"]
    return out
