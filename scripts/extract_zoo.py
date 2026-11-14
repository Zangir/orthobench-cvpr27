#!/usr/bin/env python3
"""Extract frozen features for the encoder zoo, decoding each restricted image once.

Usage: python scripts/extract_zoo.py --data-root PATH --modality opg \
           --backbones resnet50,dinov2_base,rad_dino [--limit N --cache-dir DIR]
Restricted images are read in place; only feature caches are written (gitignored).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from orthobench import manifest, features  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root")
    ap.add_argument("--modality", required=True, choices=["intraoral", "opg", "cephalogram"])
    ap.add_argument("--backbones", required=True)
    ap.add_argument("--cache-dir", default=str(ROOT / "data/feat_cache"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    root = manifest.data_root(a.data_root)
    table = manifest.modality_table(root, a.modality)
    if a.limit:
        table = table.head(a.limit)
    os.makedirs(a.cache_dir, exist_ok=True)
    features.extract_many(table, root, a.backbones.split(","), a.cache_dir, a.modality, batch=a.batch)


if __name__ == "__main__":
    main()
