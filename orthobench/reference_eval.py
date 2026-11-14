"""Retrospective, audit-held-out evaluation using frozen cached representations.

The entire audit (including pilot and exclusions) is withheld from fitting. All
patients in any cross-patient pHash candidate at the pre-existing threshold are
quarantined from both fitting and scoring. No model/hyperparameter selection uses
audit labels. This is not a prospectively untouched or externally sampled test set.
"""
from pathlib import Path
from itertools import combinations
from importlib.metadata import version
import hashlib
import numpy as np
from .dedup import hamming_matrix_pairs
from .probes import logreg_probe
from .reference_audit import LABELS, EXCLUDED
from .reference_stats import paired_reference_scores

BACKBONES = ("resnet50", "dinov2", "dinov2_large", "siglip2", "biomedclip")
MODALITIES = ("intraoral", "opg", "cephalogram")


def quarantine_candidates(caches, max_dist=6):
    """Union all cross-patient candidate endpoints without guessing patient identity."""
    quarantine, counts = set(), {}
    for name, f in caches.items():
        pairs = hamming_matrix_pairs(f["phash"], max_dist=max_dist)
        pids = f["patient_id"].astype(str)
        cross = [(pids[i], pids[j]) for edges in pairs.values() for i, j in edges if pids[i] != pids[j]]
        involved = {p for pair in cross for p in pair}
        quarantine.update(involved)
        counts[name] = {"cross_patient_candidate_pairs": len(cross), "patients_involved": len(involved)}
    return quarantine, counts


def split_masks(patient_ids, audit_ids, test_ids, quarantine):
    """Fail closed: audited/quarantined patients cannot enter training."""
    tr = np.array([p not in audit_ids and p not in quarantine for p in patient_ids])
    te = np.array([p in test_ids and p not in quarantine for p in patient_ids])
    if set(patient_ids[tr]) & (audit_ids | set(patient_ids[te]) | quarantine):
        raise ValueError("Reference evaluation split overlaps")
    if not tr.any() or not te.any():
        raise ValueError("Empty reference evaluation partition")
    return tr, te


def run_evaluation(rows, cache_dir, n_boot=2000, seed=20260910):
    cache_dir = Path(cache_dir)
    provenance = {}

    def load(backbone, modality):
        path = cache_dir / f"{backbone}_{modality}.npz"
        # Existing trusted local cache only; clinical images are never copied.
        f = dict(np.load(path, allow_pickle=True))
        provenance[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return f

    caches = {m: load("dinov2", m) for m in MODALITIES}
    quarantine, near_counts = quarantine_candidates(caches)
    audit_ids = {r["patient_id"] for r in rows}
    primary = {r["patient_id"]: r for r in rows if r["batch"] == "Audit batch" and r["final_label"] != EXCLUDED}
    test_ids = set(primary)
    metrics, partition_stats = {}, {}

    def score(name, ids, pred, n_train, train_labels):
        original = [LABELS.index(primary[p]["manifest_label"]) for p in ids]
        consensus = [LABELS.index(primary[p]["final_label"]) for p in ids]
        metrics[name] = paired_reference_scores(original, consensus, pred, n_boot, seed)
        metrics[name]["n_train_patients"] = int(n_train)
        metrics[name]["test_consensus_support"] = np.bincount(consensus, minlength=3).tolist()
        metrics[name]["test_manifest_support"] = np.bincount(original, minlength=3).tolist()
        metrics[name]["train_manifest_support"] = np.bincount(train_labels, minlength=3).tolist()

    def single(f, name):
        pids, y = f["patient_id"].astype(str), f["label_numeric"].astype(int)
        tr, te = split_masks(pids, audit_ids, test_ids, quarantine)
        # Check the cache's clinical targets match the workbook before scoring.
        for r in rows:
            labs = y[pids == r["patient_id"]]
            if len(labs) and not np.all(labs == LABELS.index(r["manifest_label"])):
                raise ValueError("Cache/manifest labels conflict")
        train_ids, test = sorted(set(pids[tr])), sorted(set(pids[te]))
        if set(y[tr]) != {0, 1, 2}:
            raise ValueError("A training class is absent")
        prob = logreg_probe(f["features"][tr], y[tr], f["features"][te], C=1., class_weight="balanced")
        pred = [int(prob[pids[te] == p].mean(0).argmax()) for p in test]
        train_labels = [int(y[pids == p][0]) for p in train_ids]
        score(name, test, pred, len(train_ids), train_labels)
        partition_stats[name] = {"train_images": int(tr.sum()), "test_images": int(te.sum()),
                                 "train_test_patient_overlap": len(set(train_ids) & set(test)),
                                 "audited_training_patients": len(set(train_ids) & audit_ids)}
        majority = int(np.bincount(train_labels, minlength=3).argmax())
        score(name + "/majority", test, [majority] * len(test), len(train_ids), train_labels)
        print(f"[reference] {name}: {len(train_ids)} train / {len(test)} test", flush=True)

    for b in BACKBONES:
        single(caches["intraoral"] if b == "dinov2" else load(b, "intraoral"), b + "/intraoral")
    for m in MODALITIES[1:]:
        single(caches[m], "dinov2/" + m)

    # Mean embeddings per modality, concatenate blocks on an identical complete-case cohort.
    maps, plabels = {}, {}
    for m, f in caches.items():
        pids = f["patient_id"].astype(str)
        maps[m] = {p: f["features"][pids == p].mean(0) for p in np.unique(pids)}
        for p in np.unique(pids):
            lab = int(f["label_numeric"][pids == p][0])
            if p in plabels and plabels[p] != lab:
                raise ValueError("Cross-modality clinical label conflict")
            plabels[p] = lab
    paired = np.array(sorted(set.intersection(*(set(v) for v in maps.values()))))
    tr, te = split_masks(paired, audit_ids, test_ids, quarantine)
    y = np.array([plabels[p] for p in paired])
    blocks = {m: np.stack([maps[m][p] for p in paired]) for m in MODALITIES}
    for k in (1, 2, 3):
        for subset in combinations(MODALITIES, k):
            x = np.concatenate([blocks[m] for m in subset], axis=1)
            prob = logreg_probe(x[tr], y[tr], x[te], C=1., class_weight="balanced")
            name = "paired/" + "+".join(subset)
            score(name, paired[te], prob.argmax(1), tr.sum(), y[tr])
    # Missing-modality ablation uses the same learned full model via an identical fit.
    x = np.concatenate([blocks[m] for m in MODALITIES], axis=1)
    starts = np.cumsum([0] + [blocks[m].shape[1] for m in MODALITIES])
    for i, m in enumerate(MODALITIES):
        xt = x[te].copy()
        xt[:, starts[i]:starts[i + 1]] = x[tr, starts[i]:starts[i + 1]].mean(0)
        prob = logreg_probe(x[tr], y[tr], xt, C=1., class_weight="balanced")
        score("missing/" + m, paired[te], prob.argmax(1), tr.sum(), y[tr])
    return {
        "provenance": "run", "design": "retrospective audit-held-out evaluation; hyperparameters fixed before execution",
        "n_audit_withheld": len(audit_ids), "n_primary_classifiable": len(primary),
        "n_quarantined_patients": len(quarantine), "n_audit_quarantined": len(audit_ids & quarantine),
        "n_primary_classifiable_quarantined": len(test_ids & quarantine),
        "near_duplicate_threshold": 6, "near_duplicate_by_modality": near_counts,
        "bootstrap_replicates": n_boot, "bootstrap_seed": seed,
        "hyperparameters": {"C": 1., "class_weight": "balanced", "max_iter": 2000},
        "training_labels": "original manifests, only unaudited patients",
        "test_labels": "primary audit consensus; excluded cases and pilot not scored",
        "noise_axis": "patient-bootstrap on fixed predictions; no training-seed replication claimed",
        "frozen_feature_cache_sha256": provenance, "partitions": partition_stats, "metrics": metrics,
        "library_versions": {k: version(k) for k in ("numpy", "scipy", "scikit-learn", "openpyxl")},
        "limitations": ["Retrospective sample with unverified sampling mechanism", "Earlier model-family choices used this cohort",
                        "Frozen pretraining contamination cannot be ruled out", "Hash quarantine is a screen, not clinical identity reconciliation",
                        "Training targets remain unaudited", "No fine-tuned checkpoint is reused because prior training may overlap the audit"],
    }
