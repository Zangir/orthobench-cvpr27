#!/usr/bin/env python3
"""Aggregate cohort statistics for Tab. 1 (user request 2026-11-16). CPU, seconds.

Reads the restricted manifests in place, the audit workbook (case-level, read in place) and the
existing feature caches (patient ids only, to reproduce the reference-arm split masks). Writes
ONLY aggregate counts to results/dataset_table_stats.json. Cells of a cross-tabulation below
5 patients are suppressed (stored as null, printed as "<5"). No demographic field exists.

  python scripts/dataset_table_stats.py --data-root PATH --workbook W
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_cvpr27 as R  # noqa: E402
from orthobench import zoo  # noqa: E402
from orthobench.reference_eval import split_masks  # noqa: E402

MODS = ("intraoral", "opg", "cephalogram")
CLASSES = ("Class I", "Class II", "Class III")
K_MIN = 5


def sup(n):
    n = int(n)
    return n if n >= K_MIN or n == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--workbook", type=Path, required=True)
    a = ap.parse_args()
    root = Path(a.data_root)
    img = pd.read_csv(root / "manifests/images.csv")
    pat = pd.read_csv(root / "manifests/patients.csv")
    rows = R.read_cases(a.workbook)
    ctx = zoo.context(rows, R.CACHE)
    out = {"provenance": "run: aggregates of the restricted manifests, audit workbook and reference-arm split masks",
           "small_cell_rule": f"cross-tab cells with 1..{K_MIN - 1} patients suppressed (null)",
           "n_patients": int(len(pat)), "n_images": int(len(img)), "per_modality": {}, "per_class": {},
           "reference_split": {}}
    for m in MODS:
        sub = img[img.modality == m]
        f = zoo.load_cache(R.CACHE, "dinov2", m)
        pids = f["patient_id"].astype(str)
        tr, te = split_masks(pids, ctx["audit_ids"], ctx["test_ids"], ctx["quarantine"])
        out["per_modality"][m] = {
            "patients": int(sub.patient_id.nunique()), "images": int(len(sub)),
            "images_per_patient_mean": float(sub.groupby("patient_id").size().mean()),
            "train_patients": int(len(set(pids[tr]))), "test_patients": int(len(set(pids[te]))),
            "train_images": int(tr.sum()), "test_images": int(te.sum()),
            "patients_by_source_class": {c: sup(sub[sub.label == c].patient_id.nunique()) for c in CLASSES},
            "images_by_source_class": {c: sup((sub.label == c).sum()) for c in CLASSES}}
    lab = pat.set_index("patient_id")["label"]
    f = zoo.load_cache(R.CACHE, "dinov2", "intraoral")
    pids = f["patient_id"].astype(str)
    tr, te = split_masks(pids, ctx["audit_ids"], ctx["test_ids"], ctx["quarantine"])
    train, test = sorted(set(pids[tr])), sorted(set(pids[te]))
    used = set(train) | set(test)
    unused = [p for p in pat.patient_id if p not in used]
    audited = ctx["audit_ids"]
    manifest, final, _ = zoo.reference_labels(ctx, np.array(test))
    out["reference_split"] = {
        "train_patients": len(train), "test_patients": len(test), "unused_patients": len(unused),
        "unused_audited": int(sum(p in audited for p in unused)),
        "unused_unaudited_quarantined": int(sum(p not in audited for p in unused)),
        "train_by_source_class": {c: sup(sum(lab[p] == c for p in train)) for c in CLASSES},
        "test_by_source_class": {c: sup((manifest == i).sum()) for i, c in enumerate(CLASSES)},
        "test_by_adjudicated_class": {c: sup((final == i).sum()) for i, c in enumerate(CLASSES)},
        "unused_by_source_class": {c: sup(sum(lab[p] == c for p in unused)) for c in CLASSES}}
    out["per_class"] = {c: {"patients": int((pat.label == c).sum()), "images": int((img.label == c).sum())}
                        for c in CLASSES}
    R.save("dataset_table_stats.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
