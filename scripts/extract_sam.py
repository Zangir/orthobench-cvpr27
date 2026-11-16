#!/usr/bin/env python3
"""Frozen SAM ViT-B image-encoder features for the post-plan entry P1 (plan amendment 2026-11-16).

Usage: python scripts/extract_sam.py --data-root PATH [--modality intraoral --limit N --batch 4]

The image encoder of SAM (facebook/sam-vit-base) is run frozen with SAM's own preprocessing
(longest side 1024, pad, SAM normalisation). Two image features are kept from the same forward
pass, both mean-pooled over the 64x64 grid:
  features       256-d neck output (the image embedding SAM's mask decoder reads)
  features_vit   768-d final ViT hidden state before the neck
The choice between them is made on training patients only (scripts/postplan_baselines.py).
Restricted images are read in place; only the feature cache is written (gitignored data/).
Resumable: partial progress is saved every --save-every images.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from orthobench import manifest, features  # noqa: E402

HUB = "facebook/sam-vit-base"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root")
    ap.add_argument("--modality", default="intraoral", choices=["intraoral", "opg", "cephalogram"])
    ap.add_argument("--cache-dir", default=str(ROOT / "data/feat_cache"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=400)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor
    Image.MAX_IMAGE_PIXELS = None

    root = manifest.data_root(a.data_root)
    table = manifest.modality_table(root, a.modality)
    if a.limit:
        table = table.head(a.limit)
    out = Path(a.cache_dir) / f"sam_vitb_{a.modality}{'_limit' + str(a.limit) if a.limit else ''}.npz"
    part = out.with_suffix(".partial.npz")
    if out.exists():
        print(f"[sam] cache exists: {out}"); return
    done = dict(np.load(part, allow_pickle=True)) if part.exists() else None
    acc = {k: list(done[k]) for k in ("features", "features_vit", "record_id", "patient_id", "label",
                                      "label_numeric")} if done else \
        {k: [] for k in ("features", "features_vit", "record_id", "patient_id", "label", "label_numeric")}
    seen = set(acc["record_id"])
    device = features._device()
    proc = SamProcessor.from_pretrained(HUB)
    enc = SamModel.from_pretrained(HUB).vision_encoder.eval().to(device)
    rows = [r for r in table.itertuples(index=False) if r.record_id not in seen]
    print(f"[sam] device={device} todo={len(rows)} done={len(seen)}", flush=True)
    t0, bad, since = time.monotonic(), 0, 0

    def save(path):
        np.savez(path, **{k: np.stack(v).astype(np.float32) if k.startswith("features") else np.array(v)
                          for k, v in acc.items()}, backbone="sam_vitb", hub=HUB, n_unreadable=bad)

    for i in range(0, len(rows), a.batch):
        chunk, ims = rows[i:i + a.batch], []
        keep = []
        for r in chunk:
            try:
                ims.append(Image.open(os.path.join(root, r.relative_path)).convert("RGB")); keep.append(r)
            except Exception:
                bad += 1
        if not ims:
            continue
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            o = enc(pixel_values=px, output_hidden_states=True)
        neck = o.last_hidden_state.float().mean(dim=(2, 3)).cpu().numpy()          # B x 256
        vit = o.hidden_states[-1].float().mean(dim=(1, 2)).cpu().numpy()             # B x 768
        for k, r in enumerate(keep):
            acc["features"].append(neck[k]); acc["features_vit"].append(vit[k])
            acc["record_id"].append(r.record_id); acc["patient_id"].append(r.patient_id)
            acc["label"].append(r.label); acc["label_numeric"].append(int(r.label_numeric))
        since += len(keep)
        if since >= a.save_every:
            save(part); since = 0
            n = len(acc["record_id"])
            print(f"[sam] {n}/{len(table)} ({time.monotonic() - t0:.0f}s, bad={bad})", flush=True)
    save(out)
    if part.exists():
        part.unlink()
    print(f"[sam] wrote {len(acc['record_id'])} rows to {out} in {time.monotonic() - t0:.0f}s (bad={bad})")


if __name__ == "__main__":
    main()
