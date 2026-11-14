#!/usr/bin/env python3
"""End-to-end fine-tuning on the audit-held-out reference split (CVPR 2027 revision).

Streams the existing uint8 pixel cache (no clinical image is re-read or copied).
Training patients = unaudited, non-quarantined; test = primary-audit classifiable,
non-quarantined (identical to the frozen-probe partition). Fixed recipes, no early
stopping, no test-based selection (experiments/cvpr27_plan.json).

Usage: python scripts/finetune_reference.py --workbook W --arch resnet50 --seeds 0 1 2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from orthobench import finetune, zoo, variables as V  # noqa: E402
from orthobench.reference_audit import read_cases  # noqa: E402

LR_BACKBONE = {"dinov2": 1e-5, "resnet50": 1e-4, "convnext_tiny": 1e-4}


class PixelReader:
    """Row reader for the uint8 pixel cache using pread: no persistent memory map, so the
    1.4 GB cache never becomes resident (keeps the job under the machine's RSS cap)."""
    ROW = 256 * 256 * 3

    def __init__(self, path, n):
        self.fd, self.n = os.open(path, os.O_RDONLY), n

    def __getitem__(self, idx):
        idx = np.atleast_1d(idx)
        out = np.empty((len(idx), 256, 256, 3), dtype=np.uint8)
        for k, i in enumerate(idx):
            out[k] = np.frombuffer(os.pread(self.fd, self.ROW, int(i) * self.ROW), dtype=np.uint8).reshape(256, 256, 3)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", required=True, type=Path)
    ap.add_argument("--arch", required=True, choices=sorted(LR_BACKBONE))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
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
    out_npz = ROOT / "data/preds/finetune_reference_preds.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    store = dict(np.load(out_npz, allow_pickle=True)) if out_npz.exists() else {}
    res_path = ROOT / "results/finetune_reference.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {
        "provenance": "run", "plan": "experiments/cvpr27_plan.json", "epochs": a.epochs,
        "n_train_images": int(len(tr)), "n_train_patients": int(len(set(pids[tr]))),
        "n_test_images": int(len(te)), "n_test_patients": int(len(test)), "runs": {}}
    for s in a.seeds:
        key = f"finetune/{a.arch}/s{s}"
        if key in store:
            continue
        t0 = time.monotonic()
        prob = finetune.train_eval(arr, meta, tr, te, seed=s, epochs=a.epochs, bs=a.batch,
                                   lr_bb=LR_BACKBONE[a.arch], arch=a.arch)
        agg = np.stack([prob[pids[te] == p].mean(0) for p in test])
        store[key], store[key + "#ids"] = agg, test
        np.savez(out_npz, **store)
        pred = agg.argmax(1)
        res["runs"][key] = {"arch": a.arch, "seed": s, "lr_backbone": LR_BACKBONE[a.arch], "batch": a.batch,
                            "manifest_macro_f1": float(V.scores(manifest, pred)[0][0]),
                            "final_macro_f1": float(V.scores(final, pred)[0][0]),
                            "final_balanced_accuracy": float(V.scores(final, pred)[1][0]),
                            "wall_seconds": time.monotonic() - t0}
        res_path.write_text(json.dumps(res, indent=1) + "\n")
        print(f"[ft-ref] {key}: manifest {res['runs'][key]['manifest_macro_f1']:.3f} "
              f"final {res['runs'][key]['final_macro_f1']:.3f} ({time.monotonic() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
