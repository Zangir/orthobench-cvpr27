#!/usr/bin/env python3
"""Generate a tiny, license-clean SYNTHETIC OrthoBench-shaped sample for smoke tests.
No real patient data. Manifest columns mirror the real schema (subset used by the code).
Run:  python sample_data/make_sample.py
"""
import os
import csv
import hashlib
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
MAN = os.path.join(HERE, "manifests")
os.makedirs(MAN, exist_ok=True)

rng = np.random.RandomState(0)
LABELS = ["Class I", "Class II", "Class III"]
img_rows, pat_rows = [], []
rec = 0
for pi in range(9):
    pid = f"SYN{pi:03d}"
    c = pi % 3
    mods = []
    plan = {"intraoral": rng.randint(3, 6), "opg": rng.randint(0, 2),
            "cephalogram": rng.randint(0, 2)}
    for modality, n in plan.items():
        for _ in range(n):
            rec += 1
            recid = f"SYNI{rec:06d}"
            sub = os.path.join(IMG, modality, pid)
            os.makedirs(sub, exist_ok=True)
            w, h = int(rng.randint(200, 400)), int(rng.randint(200, 400))
            arr = (rng.rand(h, w, 3) * 255).astype("uint8")
            rel = f"images/{modality}/{pid}/{recid}.png"
            Image.fromarray(arr).save(os.path.join(HERE, rel))
            sha = hashlib.sha256(arr.tobytes()).hexdigest()
            img_rows.append({"record_id": recid, "patient_id": pid, "relative_path": rel,
                             "modality": modality, "label": LABELS[c], "label_numeric": c,
                             "width": w, "height": h, "sha256": sha,
                             "pixel_sha256": sha, "source_had_exif": 0,
                             "phi_pixel_review": "pending", "split": "unassigned"})
        if n > 0:
            mods.append(modality)
    pat_rows.append({"patient_id": pid, "label": LABELS[c], "label_numeric": c,
                     "n_images": sum(plan.values()),
                     "n_intraoral": plan["intraoral"], "n_opg": plan["opg"],
                     "n_cephalogram": plan["cephalogram"],
                     "modalities_present": "+".join(mods), "split": "unassigned"})

with open(os.path.join(MAN, "images.csv"), "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(img_rows[0].keys())); wtr.writeheader(); wtr.writerows(img_rows)
with open(os.path.join(MAN, "patients.csv"), "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(pat_rows[0].keys())); wtr.writeheader(); wtr.writerows(pat_rows)
print(f"wrote {len(img_rows)} synthetic images / {len(pat_rows)} patients to {HERE}")
