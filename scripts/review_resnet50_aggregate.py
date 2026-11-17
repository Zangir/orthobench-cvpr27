#!/usr/bin/env python3
"""Aggregate the post-plan ResNet-50 warmup rerun next to the protocol ResNet-50 entry.

Plan amendment 2026-11-17 'C1_resnet50_warmup'. Entry = mean of the seeds' scores on the shared
2,000 patient-bootstrap draws (as every fine-tuned leaderboard entry). Nothing is refitted.

  python scripts/review_resnet50_aggregate.py --workbook W   -> results/review_resnet50_warmup.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_cvpr27 as R  # noqa: E402
from orthobench import zoo, variables as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path, required=True)
    a = ap.parse_args()
    ctx = zoo.context(R.read_cases(a.workbook), R.CACHE)
    store = R._load_preds()
    test = np.array(sorted(str(x) for x in store["intraoral/dinov2/logreg#ids"]))
    manifest, final, _ = zoo.reference_labels(ctx, test)
    w = V.draw_weights(len(test), R.N_BOOT, R.SEED)
    ref = json.loads((R.RESULTS / "finetune_reference.json").read_text())["runs"]
    zr = json.loads((R.RESULTS / "zoo_reference.json").read_text())["macro_f1"]["models"]

    def aligned(path):
        z = np.load(path, allow_pickle=True)
        pos = {str(p): i for i, p in enumerate(z["ids"])}
        return np.stack([z["prob"][pos[p]] for p in test]).argmax(1)

    done = [s for s in range(3) if (R.RESULTS / f"review_runs/resnet50_warmup_s{s}.json").exists()]
    groups = {"protocol": ([R._align(store, f"finetune/resnet50/s{s}", test).argmax(1) for s in range(3)],
                           [{"seed": s, **{k: ref[f"finetune/resnet50/s{s}"][k] for k in
                                           ("manifest_macro_f1", "final_macro_f1", "final_balanced_accuracy")}} for s in range(3)]),
              "warmup": ([aligned(R.ROOT / f"data/preds/review_resnet50_warmup_s{s}.npz") for s in done],
                         [json.loads((R.RESULTS / f"review_runs/resnet50_warmup_s{s}.json").read_text()) for s in done])}
    lead = [R._align(store, f"finetune/dinov2/s{s}", test).argmax(1) for s in range(3)]
    out = {"provenance": "run", "status": "post-plan" + ("" if len(done) == 3 else f" (partial: {len(done)}/3 seeds)"),
           "plan_amendment": "experiments/cvpr27_plan.json amendments[C1_resnet50_warmup]",
           "n_test": len(test), "bootstrap_replicates": R.N_BOOT, "bootstrap_seed": R.SEED, "seeds_done": done}
    boots = {}
    for g, (preds, meta) in groups.items():
        rec = {"per_seed": [{**{k: m[k] for k in ("seed", "manifest_macro_f1", "final_macro_f1", "final_balanced_accuracy")},
                             "pred_class_counts_I_II_III": np.bincount(p, minlength=3).tolist()} for m, p in zip(meta, preds)]}
        for r, t in (("source", manifest), ("adjudicated", final)):
            b = V.entry_scores(t, preds, w, 0)
            boots[(g, r)] = b
            others = sorted((m["score_ref_a" if r == "source" else "score_ref_b"] for m in zr), reverse=True)
            s0 = float(V.entry_scores(t, preds)[0])
            rec[r] = {"macro_f1": s0, "macro_f1_ci": V.ci(b),
                      "balanced_accuracy": float(V.entry_scores(t, preds, None, 1)[0]),
                      "leader_minus_entry": float(V.entry_scores(t, lead)[0] - s0),
                      "leader_minus_entry_ci": V.ci(V.entry_scores(t, lead, w, 0) - b),
                      "rank_if_inserted_among_33": int(1 + sum(s > s0 for s in others))}
        out[g] = rec
    for r in ("source", "adjudicated"):
        d = boots[("warmup", r)] - boots[("protocol", r)]
        out[f"warmup_minus_protocol_{r}"] = {"diff": out["warmup"][r]["macro_f1"] - out["protocol"][r]["macro_f1"], "ci": V.ci(d)}
    R.save("review_resnet50_warmup.json", out)


if __name__ == "__main__":
    main()
