#!/usr/bin/env python3
"""Post-plan rerun of fine-tuned ResNet-50 with a lower backbone rate and linear warmup.

Plan amendment 2026-11-17 'C1_resnet50_warmup' in ../cvpr27_plan.json, logged before running.
Same split, pixel cache, augmentation, batch 16, 8 epochs, head rate 1e-3 and weight decay as the
protocol fine-tuning (scripts/finetune_reference.py); changed: backbone rate 1e-4 -> 1e-5 and one
epoch of linear warmup before the cosine decay. One seed per call; resumable (skips a seed whose
result file exists). Predictions go to the gitignored data/preds/, never into the 33-entry store.

  python scripts/review_resnet50_warmup.py --workbook W --seed 0
-> results/review_runs/resnet50_warmup_s0.json, data/preds/review_resnet50_warmup_s0.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from orthobench import finetune, zoo, variables as V  # noqa: E402
from orthobench.reference_audit import read_cases  # noqa: E402
from finetune_reference import PixelReader  # noqa: E402

LR_BACKBONE, BATCH, EPOCHS = 1e-5, 16, 8


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", required=True, type=Path)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    out = ROOT / f"results/review_runs/resnet50_warmup_s{a.seed}.json"
    if out.exists():
        print(f"[skip] {out.name} exists"); return
    cache = ROOT / "data/feat_cache/pixels_intraoral_256.u8"
    meta = dict(np.load(str(cache) + ".meta.npz", allow_pickle=True))
    arr = PixelReader(str(cache), int(meta["n"]))
    ctx = zoo.context(read_cases(a.workbook), ROOT / "data/feat_cache")
    pids = meta["patient_id"].astype(str)
    tr_mask = np.array([p not in ctx["audit_ids"] and p not in ctx["quarantine"] for p in pids])
    te_mask = np.array([p in ctx["test_ids"] and p not in ctx["quarantine"] for p in pids])
    assert not (set(pids[tr_mask]) & (ctx["audit_ids"] | ctx["quarantine"] | set(pids[te_mask])))
    tr, te = np.where(tr_mask)[0], np.where(te_mask)[0]
    test = np.array(sorted(set(pids[te])))
    manifest, final, _ = zoo.reference_labels(ctx, test)
    warmup = len(tr) // BATCH + 1  # one epoch of steps, as counted by finetune.train_eval
    t0 = time.monotonic()
    prob = finetune.train_eval(arr, meta, tr, te, seed=a.seed, epochs=EPOCHS, bs=BATCH,
                               lr_bb=LR_BACKBONE, arch="resnet50", warmup_steps=warmup)
    agg = np.stack([prob[pids[te] == p].mean(0) for p in test])
    pred_dir = ROOT / "data/preds"; pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f"review_resnet50_warmup_s{a.seed}.npz", prob=agg, ids=test)
    pred = agg.argmax(1)
    res = {"provenance": "run", "status": "post-plan",
           "plan_amendment": "experiments/cvpr27_plan.json amendments[C1_resnet50_warmup]",
           "arch": "resnet50", "seed": a.seed, "lr_backbone": LR_BACKBONE, "lr_head": 1e-3, "batch": BATCH,
           "epochs": EPOCHS, "warmup_steps": warmup, "schedule": "linear warmup then cosine to 0",
           "n_train_images": int(len(tr)), "n_train_patients": int(len(set(pids[tr]))),
           "n_test_patients": int(len(test)),
           "pred_class_counts_I_II_III": np.bincount(pred, minlength=3).tolist(),
           "manifest_macro_f1": float(V.scores(manifest, pred)[0][0]),
           "final_macro_f1": float(V.scores(final, pred)[0][0]),
           "final_balanced_accuracy": float(V.scores(final, pred)[1][0]),
           "wall_seconds": time.monotonic() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[resnet50-warmup] s{a.seed}: source {res['manifest_macro_f1']:.3f} adjudicated "
          f"{res['final_macro_f1']:.3f} counts {res['pred_class_counts_I_II_III']} ({res['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
