#!/usr/bin/env python3
"""Complete seed-zero control intervals without retraining historical backbones."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orthobench import evaluate, splits


def main():
    cache = ROOT / "data/feat_cache/dinov2_intraoral.npz"
    with np.load(cache, allow_pickle=False) as features:
        pid = features["patient_id"].astype(str)
        labels = features["label_numeric"].astype(int)
    train, test = splits.patient_level_split(pid, labels, seed=0, test_frac=0.3)
    patient_labels = {p: int(labels[pid == p][0]) for p in np.unique(pid)}
    true = np.array([patient_labels[p] for p in np.unique(pid[test])])
    counts = np.bincount(labels[train], minlength=3)
    predictions = {
        "majority_class": np.full_like(true, counts.argmax()),
        "stratified_random": np.random.RandomState(0).choice(
            3, size=len(true), p=counts / counts.sum()),
    }
    historical = json.loads((ROOT / "results/baselines.json").read_text())["trivial"]
    result = {
        "provenance": "run: recomputed historical seed-zero controls",
        "n_test_patients": len(true), "split_seed": 0,
        "bootstrap_seed": 0, "bootstrap_replicates": 1000,
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "controls": {},
    }
    for name, pred in predictions.items():
        metrics = evaluate.image_metrics(true, pred)
        if not np.isclose(metrics["macro_f1"], historical[name]["macro_f1"]):
            raise ValueError(f"Historical {name} point estimate did not reproduce")
        result["controls"][name] = {
            **metrics, **evaluate.patient_bootstrap_ci(true, pred),
        }
    output = ROOT / "results/historical_controls.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["controls"], indent=2))


if __name__ == "__main__":
    main()
