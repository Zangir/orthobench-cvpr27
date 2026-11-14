#!/usr/bin/env python3
"""Build results/comparison_matrix.json: OrthoBench vs incumbent dental/ortho
datasets, one row per dataset with capability flags and a quantitative column.

Every incumbent fact is *reported provenance*: it is quoted from a cited source
in the literature review (source_id = arXiv id or OpenAlex/Crossref id in
litreview/papers.json). No fact here is invented; unknown/unreported values are
recorded as null and rendered as "n/r" downstream. The OrthoBench row is *run*
provenance, regenerated from results/dataset_stats.json.

Capability columns:
  patient_linked_multimodal  intraoral+OPG+ceph tied to the same patient
  angle_class_labels         patient-level Angle Class I/II/III labels
  patient_level_split        a patient-level (grouped) split is defined/possible
"""
import os
import json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def _dataset_stats():
    p = os.path.join(RES, "dataset_stats.json")
    if os.path.exists(p):
        return json.load(open(p))
    return None


# Incumbent rows. Each numeric fact carries the source_id it is quoted from.
# The linked flag refers specifically to the intraoral+OPG+cephalogram combination.
# A task description does not establish the absence of patient-level splitting:
# the split column is unknown unless the source evidence documents that property.
INCUMBENTS = [
    {"name": "ISBI-2015", "cite_key": "wang2016benchmark", "source_id": "td995ce1163",
     "modalities": "Lateral ceph", "population": "Single-center (Asia)",
     "task": "Landmark detection", "n_patients": None, "n_images": 400},
    {"name": "Aariz", "cite_key": "khalid2023aariz", "source_id": "2302.07797",
     "modalities": "Lateral ceph", "population": "Pakistan",
     "task": "Landmark detection + CVM", "n_patients": None, "n_images": 1000},
    {"name": "CL-Detection 2023", "cite_key": "zhang2024techniques", "source_id": "2409.15834",
     "modalities": "Lateral ceph", "population": "Multi-center (3 centers)",
     "task": "Landmark detection", "n_patients": None, "n_images": 600},
    {"name": "Tufts", "cite_key": "panetta2022tufts", "source_id": "ta4b42dbfce",
     "modalities": "Panoramic", "population": "USA",
     "task": "Diagnostic segmentation", "n_patients": None, "n_images": 1000},
    {"name": "DENTEX", "cite_key": "hamamci2023dentex", "source_id": "2305.19112",
     "modalities": "Panoramic", "population": "Europe",
     "task": "Enumeration / pathosis", "n_patients": None, "n_images": 3903,
     # DENTEX challenge paper (arXiv 2305.19112v1, Sec. 2.1): 693 quadrant-only + 634
     # quadrant-enumeration + 1005 fully labelled + 1571 unlabelled panoramic X-rays = 3,903
     # (amendment 2026-11-14; previously left n/r)
     "n_images_note": "693 + 634 + 1005 + 1571 unlabelled panoramic X-rays"},
    {"name": "UFBA-UESC", "cite_key": "silva2018automatic", "source_id": "t709c1581ad",
     "modalities": "Panoramic", "population": "Brazil",
     "task": "Tooth segmentation", "n_patients": None, "n_images": 1500},
]

CAP_FALSE = {"patient_linked_multimodal": False, "angle_class_labels": False,
             "patient_level_split": None}


def build():
    ds = _dataset_stats()
    if ds is None:
        raise FileNotFoundError("Run the cohort audit before rendering the comparison")
    rows = []
    for inc in INCUMBENTS:
        row = dict(inc)
        row.update(CAP_FALSE)
        row["provenance"] = "reported"
        rows.append(row)
    ortho = {
        "name": "OrthoBench (ours)", "cite_key": None, "source_id": "ours",
        "modalities": "Intraoral + OPG + ceph", "population": "single source cohort (withheld for review)",
        "task": "Patient-level Angle class",
        "patient_linked_multimodal": True, "angle_class_labels": True,
        "patient_level_split": True, "provenance": "run",
        "n_patients": ds["n_patients"],
        "n_images": ds["n_images"],
        "paired_patients": ds["fully_paired_patients"],
    }
    rows.append(ortho)
    out = {
        "description": ("OrthoBench vs representative public dental/orthodontic "
                        "datasets. Incumbent facts are reported provenance (quoted "
                        "from the cited source in litreview/papers.json); the "
                        "OrthoBench row is regenerated from results/dataset_stats.json."),
        "quantitative_column": "n_images",
        "capability_columns": ["patient_linked_multimodal", "angle_class_labels",
                               "patient_level_split"],
        "null_render": "n/r",
        "datasets": rows,
    }
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "comparison_matrix.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {path} ({len(rows)} rows)")


if __name__ == "__main__":
    build()
