"""Read the supplied audit workbook in place and independently audit its rows.

Only aggregate statistics leave this module. Spreadsheet summary/formula cells are
cross-checks, never the input to statistical calculations. Original inputs are immutable.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib
import numpy as np
from .reference_stats import agreement, matrix, wilson

LABELS = ["Class I", "Class II", "Class III", "Not assessable"]
EXCLUDED = "Excluded (not assessable)"


def read_cases(workbook):
    import openpyxl
    w = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    values = iter(w["Analysis Data"].values)
    headers = next(values)
    rows = [dict(zip(headers, r)) for r in values if r[0] is not None]
    w.close()
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate patient IDs in reference export")
    for r in rows:
        required = ("patient_id", "batch", "status", "final_source", "final_label", "manifest_label",
                    "expert1_user", "expert2_user", "expert1_at", "expert2_at")
        if any(r.get(k) in (None, "") for k in required):
            raise ValueError("Missing required provenance field")
        if r["batch"] not in ("Audit batch", "Pilot batch"):
            raise ValueError("Unknown audit batch")
        if any(r[f"expert{i}_confidence"] not in ("High", "Medium", "Low") for i in (1, 2)):
            raise ValueError("Invalid or missing confidence")
        for key in ("expert1_label", "expert2_label"):
            if r[key] not in LABELS:
                raise ValueError(f"Invalid {key}")
        if r["final_label"] not in LABELS[:3] + [EXCLUDED] or r["manifest_label"] not in LABELS[:3]:
            raise ValueError("Invalid final or manifest label")
        trigger = r["expert1_label"] != r["expert2_label"] or "Not assessable" in (r["expert1_label"], r["expert2_label"])
        if (r["final_source"] == "senior") != trigger:
            raise ValueError("Adjudication trigger/source mismatch")
        if r["status"] != ("adjudicated" if trigger else "agreed"):
            raise ValueError("Nonfinal or inconsistent workflow status")
        if not trigger and r["final_label"] != r["expert1_label"]:
            raise ValueError("Agreement finalization differs from expert label")
        if trigger:
            decision = str(r["senior_decisions"])
            expected = "Not assessable" if r["final_label"] == EXCLUDED else r["final_label"]
            if not decision.startswith("adjudication=" + expected + " @"):
                raise ValueError("Senior and final labels differ")
        if r["final_label"] != EXCLUDED and int(r["manifest_final_agree"]) != int(r["manifest_label"] == r["final_label"]):
            raise ValueError("Manifest agreement flag mismatch")
        if r["final_label"] == EXCLUDED and r["manifest_final_agree"] is not None:
            raise ValueError("Excluded case must not carry a manifest-match flag")
        if bool(r["requires_senior_by_protocol"]) != trigger:
            raise ValueError("Derived senior-trigger field mismatch")
    return rows


def timestamp(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def summarize(rows, n_boot=20000, seed=20260906):
    a = [LABELS.index(r["expert1_label"]) for r in rows]
    b = [LABELS.index(r["expert2_label"]) for r in rows]
    assessed = [r for r in rows if r["final_label"] != EXCLUDED]
    orig = [LABELS.index(r["manifest_label"]) for r in assessed]
    final = [LABELS.index(r["final_label"]) for r in assessed]
    senior = [r for r in rows if r["final_source"] == "senior"]
    any_na = [r for r in rows if "Not assessable" in (r["expert1_label"], r["expert2_label"])]
    durations = []
    gaps = []
    for r in rows:
        t1, t2 = timestamp(r["expert1_at"]), timestamp(r["expert2_at"])
        if t2 < t1:
            raise ValueError("Second review precedes first")
        gaps.append((t2 - t1).total_seconds() / 3600)
        if r["final_source"] == "senior":
            ts = timestamp(str(r["senior_decisions"]).split(" @", 1)[1])
            if ts < t2:
                raise ValueError("Senior review precedes second review")
            durations.append((ts - t2).total_seconds() / 3600)
    mfinal = [3 if r["final_label"] == EXCLUDED else LABELS.index(r["final_label"]) for r in rows]
    manifest_matrix = matrix([LABELS.index(r["manifest_label"]) for r in rows], mfinal, 4)[:3]
    changed = sum(x != y for x, y in zip(orig, final))
    alignment = []
    for r in senior:
        label = "Not assessable" if r["final_label"] == EXCLUDED else r["final_label"]
        x, y = label == r["expert1_label"], label == r["expert2_label"]
        actual = "Matches both experts" if x and y else "Matches Expert 1" if x else "Matches Expert 2" if y else "Matches neither expert"
        if actual != r["senior_alignment"]:
            raise ValueError("Workbook senior-alignment field is inconsistent")
        alignment.append(actual)
    pace = {}
    for reviewer in ("expert1", "expert2", "senior"):
        times = sorted(timestamp(str(r["senior_decisions"]).split(" @", 1)[1]) if reviewer == "senior"
                       else timestamp(r[reviewer + "_at"]) for r in (senior if reviewer == "senior" else rows))
        intervals = np.array([(b - a).total_seconds() for a, b in zip(times, times[1:])])
        within = intervals[intervals <= 1800]
        pace[reviewer] = {"submissions": len(times), "sessions": 1 + int((intervals > 1800).sum()),
                          "median_interval_seconds": float(np.median(within)) if len(within) else None,
                          "interval_iqr_seconds": np.quantile(within, [.25, .75]).tolist() if len(within) else None}
    return {
        "n": len(rows), "reviewer_agreement": agreement(a, b, 4, n_boot, seed),
        "final_labels": dict(Counter(r["final_label"] for r in rows)),
        "manifest_labels": dict(Counter(r["manifest_label"] for r in rows)),
        "manifest_vs_final_matrix": manifest_matrix.tolist(),
        "manifest_agreement": agreement(orig, final, 3, n_boot, seed),
        "changed_labels": changed, "changed_fraction": changed / len(assessed),
        "changed_fraction_ci": wilson(changed, len(assessed)),
        "n_assessable": len(assessed), "n_excluded": len(rows) - len(assessed),
        "senior_n": len(senior), "senior_fraction": len(senior) / len(rows),
        "agreement_resolved_n": len(rows) - len(senior),
        "senior_final_labels": dict(Counter(r["final_label"] for r in senior)),
        "senior_alignment": dict(Counter(alignment)), "submission_pace": pace,
        "any_na": len(any_na), "both_na": sum(x == y == 3 for x, y in zip(a, b)),
        "na_retained": sum(r["final_label"] != EXCLUDED for r in any_na),
        "na_excluded": sum(r["final_label"] == EXCLUDED for r in any_na),
        "new_senior_exclusions": sum(r["final_label"] == EXCLUDED and r not in any_na for r in senior),
        "confidence": {f"expert{i}": dict(Counter(r[f"expert{i}_confidence"] for r in rows)) for i in (1, 2)},
        "expert_comments": sum(bool(r[f"expert{i}_comment"]) for r in rows for i in (1, 2)),
        "n_expert_reviews": len(rows) * 2, "discussion_comments": sum(int(r["n_comments"]) for r in rows),
        "review_gap_hours_median": float(np.median(gaps)),
        "senior_turnaround_hours": {"min": min(durations), "q1": float(np.quantile(durations, .25)),
                                     "median": float(np.median(durations)), "q3": float(np.quantile(durations, .75)),
                                     "max": max(durations), "mean": float(np.mean(durations))},
    }


def audit(workbook, data_root, n_boot=20000, seed=20260906):
    import pandas as pd
    rows = read_cases(workbook)
    patients = pd.read_csv(Path(data_root) / "manifests/patients.csv").set_index("patient_id")
    missing = sum(r["patient_id"] not in patients.index for r in rows)
    mismatch = sum(r["patient_id"] in patients.index and patients.loc[r["patient_id"], "label"] != r["manifest_label"] for r in rows)
    if missing or mismatch:
        raise ValueError(f"Cohort linkage failed: {missing} missing IDs, {mismatch} source-label mismatches")
    return {
        "provenance": "run: independent recomputation from user-provided case-level workbook; clinical reviews not repeated",
        "workbook_sha256": hashlib.sha256(Path(workbook).read_bytes()).hexdigest(),
        "source_sheet": "Analysis Data", "bootstrap_replicates": n_boot, "bootstrap_seed": seed,
        "bootstrap_method": "multinomial paired-case bootstrap; percentile kappa/AC1 intervals; Wilson proportion intervals",
        "cohort_linkage": {"n_source_patients": len(patients), "n_linked": len(rows), "missing_ids": missing,
                           "manifest_label_mismatches": mismatch, "audited_fraction": len(rows) / len(patients)},
        "combined": summarize(rows, n_boot, seed),
        "primary": summarize([r for r in rows if r["batch"] == "Audit batch"], n_boot, seed),
        "pilot": summarize([r for r in rows if r["batch"] == "Pilot batch"], n_boot, seed),
        "limitations": ["Sampling procedure not evidenced by export", "Reviewer qualifications and blinding not verified by generic accounts",
                        "No dedicated senior rationale field", "No AI predictions in input workbook", "No established linkage to a 134-case post-AI cohort"],
    }
