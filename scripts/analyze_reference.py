#!/usr/bin/env python3
"""Recompute the expert audit; optionally evaluate audit-held-out frozen classifiers."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orthobench.reference_audit import audit, read_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data/feat_cache")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = audit(args.workbook, args.data_root)
    (args.output_dir / "reference_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("[reference] verified workbook rows and cohort linkage", flush=True)
    if args.evaluate:
        from orthobench.reference_eval import run_evaluation
        started = time.monotonic()
        result = run_evaluation(read_cases(args.workbook), args.cache_dir)
        result["evaluation_wall_seconds"] = time.monotonic() - started
        result["workbook_sha256"] = __import__("hashlib").sha256(args.workbook.read_bytes()).hexdigest()
        (args.output_dir / "reference_evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("[reference] saved aggregate evaluation results", flush=True)


if __name__ == "__main__":
    main()
