"""Load OrthoBench manifests and compute the curation audit (dataset_stats).

All numbers are regenerated from the live CSV manifests so the paper's headline
counts are traceable. Pre-packaging provenance (exact/near duplicates removed,
cross-modality conflicts resolved) is read from audit/package_audit.json, which
records the cleaning that produced the packaged cohort.
"""
import os
import json
import numpy as np
import pandas as pd


def data_root(cli_value=None):
    root = cli_value or os.environ.get("ORTHOBENCH_DATA_ROOT")
    if not root:
        raise SystemExit(
            "No data root. Pass --data-root /abs/path/to/orthobenchmark-dataset "
            "or set ORTHOBENCH_DATA_ROOT. Restricted data is read in place; see README.")
    if not os.path.isdir(os.path.join(root, "manifests")):
        raise SystemExit(f"Data root has no manifests/: {root}")
    return root


def load_manifests(root):
    img = pd.read_csv(os.path.join(root, "manifests", "images.csv"))
    pat = pd.read_csv(os.path.join(root, "manifests", "patients.csv"))
    return img, pat


def _iqr(series):
    return [float(series.median()), float(series.quantile(0.25)),
            float(series.quantile(0.75)), int(series.min()), int(series.max())]


def curation_audit(root):
    img, pat = load_manifests(root)
    stats = {}
    stats["n_patients"] = int(pat["patient_id"].nunique())
    stats["n_images"] = int(len(img))
    stats["images_by_modality"] = {m: int((img.modality == m).sum())
                                   for m in sorted(img.modality.unique())}
    stats["patient_label_distribution"] = {k: int(v) for k, v in
                                           pat["label"].value_counts().to_dict().items()}
    # imbalance ratio Class I : Class III
    d = stats["patient_label_distribution"]
    if d.get("Class III"):
        stats["imbalance_I_to_III"] = round(d.get("Class I", 0) / d["Class III"], 2)
    # images per patient per modality
    stats["images_per_patient"] = {}
    for m in stats["images_by_modality"]:
        per = img[img.modality == m].groupby("patient_id").size()
        stats["images_per_patient"][m] = _iqr(per)
    # modality co-presence (from patient manifest)
    stats["modalities_present"] = {k: int(v) for k, v in
                                   pat["modalities_present"].value_counts().to_dict().items()}
    stats["fully_paired_patients"] = int(
        (pat["modalities_present"] == "intraoral+opg+cephalogram").sum())
    # exact-duplicate audit within modality on the PACKAGED (cleaned) cohort
    stats["redundant_exact_within_modality"] = {}
    for m in stats["images_by_modality"]:
        sub = img[img.modality == m]
        stats["redundant_exact_within_modality"][m] = int(len(sub) - sub["sha256"].nunique())
    # cross-patient exact-content groups remaining in packaged cohort
    g = img.groupby("sha256")["patient_id"].nunique()
    stats["cross_patient_exact_groups_remaining"] = int((g > 1).sum())
    # resolution medians/ranges
    stats["resolution"] = {}
    for m in stats["images_by_modality"]:
        sub = img[img.modality == m]
        stats["resolution"][m] = {
            "median_wh": [int(sub.width.median()), int(sub.height.median())],
            "min_wh": [int(sub.width.min()), int(sub.height.min())],
            "max_wh": [int(sub.width.max()), int(sub.height.max())]}
    # phi/exif provenance
    stats["source_files_with_exif"] = int((img["source_had_exif"] == 1).sum())
    stats["phi_pixel_review_pending"] = int((img["phi_pixel_review"] == "pending").sum())
    # pre-packaging provenance from the build audit, if present
    ap = os.path.join(root, "audit", "package_audit.json")
    if os.path.exists(ap):
        with open(ap) as f:
            pa = json.load(f)
        stats["pre_packaging_provenance"] = {
            "input_exact_unique_images": pa.get("input_exact_unique_images"),
            "excluded_cross_patient_exact_groups": pa.get("excluded_cross_patient_exact_groups"),
            "cross_modality_groups_resolved": pa.get("cross_modality_groups_resolved"),
            "cross_modality_combinations": pa.get("cross_modality_combinations"),
            "source_files_with_sensitive_info_keys": pa.get("source_files_with_sensitive_info_keys"),
            "output_files_with_sensitive_metadata": pa.get("output_files_with_sensitive_metadata"),
            "pixel_equivalence_verified": pa.get("pixel_equivalence_verified"),
        }
    return stats


def intraoral_table(root):
    """Return a dataframe of intraoral images with patient-level labels."""
    img, _ = load_manifests(root)
    io = img[img.modality == "intraoral"].copy().reset_index(drop=True)
    return io[["record_id", "patient_id", "relative_path", "label",
               "label_numeric", "width", "height"]]


def modality_table(root, modality):
    """Rows for one modality (intraoral|opg|cephalogram) with patient-level labels."""
    img, _ = load_manifests(root)
    sub = img[img.modality == modality].copy().reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"No rows for modality {modality!r} in {root}")
    return sub[["record_id", "patient_id", "relative_path", "label",
                "label_numeric", "width", "height"]]


def all_images_table(root):
    """Every packaged image with modality, for cohort-wide audits (e.g. pixel PHI)."""
    img, _ = load_manifests(root)
    return img[["record_id", "patient_id", "relative_path", "modality",
                "label", "label_numeric"]].copy().reset_index(drop=True)
