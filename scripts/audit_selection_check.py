#!/usr/bin/env python3
"""Post-plan descriptive checks (plan amendment 2026-11-17, 'D1_audit_selection', 'C0_resnet50_collapse_note').

D1: source-label mix of the 240 audited patients versus all unaudited patients (2x3 Pearson chi-square).
C0: predicted-class counts of the three stored fine-tuned ResNet-50 runs on the 189 test patients,
    next to a constant Class-I prediction on the same patients. Nothing is fitted.

  python scripts/audit_selection_check.py --workbook W   -> results/audit_selection.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_cvpr27 as R  # noqa: E402
from orthobench import zoo, variables as V  # noqa: E402
from orthobench.reference_audit import read_cases, LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path, required=True)
    a = ap.parse_args()
    rows = read_cases(a.workbook)
    audited = {r["patient_id"]: LABELS.index(r["manifest_label"]) for r in rows}
    lab = {}
    for m in zoo.MODALITIES:
        f = zoo.load_cache(R.CACHE, "dinov2", m)
        for p, y in zip(f["patient_id"].astype(str), f["label_numeric"].astype(int)):
            assert lab.setdefault(p, int(y)) == int(y), "patient label differs across modalities"
    assert all(lab[p] == y for p, y in audited.items() if p in lab), "export and cache labels differ"
    aud = np.bincount(list(audited.values()), minlength=3)
    una = np.bincount([y for p, y in lab.items() if p not in audited], minlength=3)
    chi2, p, dof, _ = chi2_contingency(np.stack([aud, una]), correction=False)
    out = {"provenance": "run", "status": "post-plan",
           "plan_amendment": "experiments/cvpr27_plan.json amendments[D1_audit_selection]",
           "audit_selection": {
               "n_audited": int(aud.sum()), "n_unaudited": int(una.sum()), "n_patients": len(lab),
               "audited_source_counts_I_II_III": aud.tolist(), "unaudited_source_counts_I_II_III": una.tolist(),
               "audited_share": (aud / aud.sum()).tolist(), "unaudited_share": (una / una.sum()).tolist(),
               "chi2": float(chi2), "dof": int(dof), "p": float(p),
               "sampling_rule_recorded_in_export": False}}
    # C0: ResNet-50 fine-tuned runs on the reference split
    ctx = zoo.context(rows, R.CACHE)
    store = R._load_preds()
    test = np.array(sorted(str(x) for x in store["intraoral/dinov2/logreg#ids"]))
    manifest, final, _ = zoo.reference_labels(ctx, test)
    runs = {}
    for s in range(3):
        pred = R._align(store, f"finetune/resnet50/s{s}", test).argmax(1)
        runs[f"s{s}"] = {"pred_class_counts_I_II_III": np.bincount(pred, minlength=3).tolist(),
                         "source_macro_f1": float(V.scores(manifest, pred)[0][0]),
                         "adjudicated_macro_f1": float(V.scores(final, pred)[0][0]),
                         "adjudicated_balanced_accuracy": float(V.scores(final, pred)[1][0])}
    const = np.zeros(len(test), dtype=int)
    out["resnet50_collapse"] = {
        "n_test": len(test), "runs": runs,
        "constant_class_I": {"source_macro_f1": float(V.scores(manifest, const)[0][0]),
                             "adjudicated_macro_f1": float(V.scores(final, const)[0][0]),
                             "adjudicated_balanced_accuracy": float(V.scores(final, const)[1][0])},
        "historical_majority_class_macro_f1": json.loads(
            (R.RESULTS / "historical_controls.json").read_text())["controls"]["majority_class"]["macro_f1"],
        "historical_majority_class_n_test": 289}
    R.save("audit_selection.json", out)


if __name__ == "__main__":
    main()
