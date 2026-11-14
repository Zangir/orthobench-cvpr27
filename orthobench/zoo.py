"""Encoder-zoo predictions on the audit-held-out reference split (CVPR 2027 revision).

Reuses the exact partition of `reference_eval`: every audited patient is withheld
from fitting and all cross-patient pHash candidates are quarantined. Training uses
original manifest labels of unaudited patients; test predictions are produced once
and later scored against both reference versions. Per-patient predictions are kept
in the gitignored data/ directory; only aggregates go to results/.
"""
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .probes import logreg_probe, knn_probe
from .reference_audit import LABELS, EXCLUDED
from .reference_eval import quarantine_candidates, split_masks

ENCODERS = ("resnet18", "resnet50", "convnext_tiny", "convnext_large", "dinov2", "dinov2_base",
            "dinov2_large", "rad_dino", "siglip2", "clip_b16", "biomedclip")
HEADS = ("logreg", "knn1", "knn5")
MODALITIES = ("intraoral", "opg", "cephalogram")
PRETTY = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-T",
          "convnext_large": "ConvNeXt-L", "dinov2": "DINOv2-S", "dinov2_base": "DINOv2-B",
          "dinov2_large": "DINOv2-L", "rad_dino": "RAD-DINO", "siglip2": "SigLIP-2",
          "clip_b16": "CLIP-B/16", "biomedclip": "BiomedCLIP"}


def load_cache(cache_dir, backbone, modality):
    return dict(np.load(Path(cache_dir) / f"{backbone}_{modality}.npz", allow_pickle=True))


def context(rows, cache_dir):
    """Audit sets, quarantine and per-patient labels shared by every experiment."""
    caches = {m: load_cache(cache_dir, "dinov2", m) for m in MODALITIES}
    quarantine, _ = quarantine_candidates(caches)
    primary = {r["patient_id"]: r for r in rows
               if r["batch"] == "Audit batch" and r["final_label"] != EXCLUDED}
    return {"audit_ids": {r["patient_id"] for r in rows}, "quarantine": quarantine,
            "primary": primary, "test_ids": set(primary)}


def _head(head, xtr, ytr, xte):
    if head == "logreg":
        return logreg_probe(xtr, ytr, xte, class_weight="balanced", C=1.0)
    return knn_probe(xtr, ytr, xte, k=int(head[3:]))


def _aggregate(prob, pids, order):
    return np.stack([prob[pids == p].mean(0) for p in order])


def predict(f, head, ctx):
    """Fit on unaudited non-quarantined patients; return (test patients, probs, train info)."""
    pids, y = f["patient_id"].astype(str), f["label_numeric"].astype(int)
    tr, te = split_masks(pids, ctx["audit_ids"], ctx["test_ids"], ctx["quarantine"])
    test = np.array(sorted(set(pids[te])))
    prob = _head(head, f["features"][tr], y[tr], f["features"][te])
    train = sorted(set(pids[tr]))
    return test, _aggregate(prob, pids[te], test), {"n_train_patients": len(train),
                                                     "n_train_images": int(tr.sum())}


def oof_train_probs(f, ctx, n_folds=5, seed=0):
    """Patient-grouped out-of-fold logreg probabilities for the training pool."""
    pids, y = f["patient_id"].astype(str), f["label_numeric"].astype(int)
    tr, _ = split_masks(pids, ctx["audit_ids"], ctx["test_ids"], ctx["quarantine"])
    train = np.array(sorted(set(pids[tr])))
    plab = np.array([int(y[pids == p][0]) for p in train])
    out = np.zeros((len(train), 3))
    for a, b in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(train, plab):
        ma, mb = np.isin(pids, train[a]) & tr, np.isin(pids, train[b]) & tr
        prob = logreg_probe(f["features"][ma], y[ma], f["features"][mb], class_weight="balanced", C=1.0)
        out[b] = _aggregate(prob, pids[mb], train[b])
    return train, plab, out


def confident_flags(probs, given):
    """Confident-learning off-diagonal flags with per-class mean self-confidence thresholds."""
    thr = np.array([probs[given == c, c].mean() for c in range(3)])
    flags = np.zeros(len(given), dtype=bool)
    for i, p in enumerate(probs):
        above = np.where(p >= thr)[0]
        if len(above):
            flags[i] = above[np.argmax(p[above])] != given[i]
    return flags, thr


def predict_pruned(f, ctx, thr_source="oof"):
    """Refit logreg after removing confidently mislabelled training patients."""
    pids, y = f["patient_id"].astype(str), f["label_numeric"].astype(int)
    train, plab, oof = oof_train_probs(f, ctx)
    flags, thr = confident_flags(oof, plab)
    keep = set(train[~flags])
    tr, te = split_masks(pids, ctx["audit_ids"], ctx["test_ids"], ctx["quarantine"])
    tr = tr & np.isin(pids, list(keep))
    test = np.array(sorted(set(pids[te])))
    prob = logreg_probe(f["features"][tr], y[tr], f["features"][te], class_weight="balanced", C=1.0)
    return test, _aggregate(prob, pids[te], test), {
        "n_pruned_patients": int(flags.sum()), "n_train_patients_after": len(keep),
        "pruned_by_given_class": np.bincount(plab[flags], minlength=3).tolist(),
        "thresholds": thr.tolist()}


def reference_labels(ctx, test):
    pr = ctx["primary"]
    manifest = np.array([LABELS.index(pr[p]["manifest_label"]) for p in test])
    final = np.array([LABELS.index(pr[p]["final_label"]) for p in test])
    agreed = np.array([pr[p]["final_source"] != "senior" for p in test])
    return manifest, final, agreed
