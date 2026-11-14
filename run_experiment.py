#!/usr/bin/env python3
"""OrthoBench benchmark entry point.

Subcommands
-----------
  smoke                      end-to-end pipeline on synthetic data (<2 min, no downloads)
  audit    --data-root PATH  curation audit  -> results/dataset_stats.json
  dedup    --data-root PATH  near-duplicate detection (needs cached features) -> results/neardup.json
  extract  --data-root PATH  frozen features -> data/feat_cache/<backbone>_<modality>.npz
  leakage                    image-level vs patient-level leakage -> results/leakage.json
  baselines                  patient-level baselines (all cached backbones) -> results/baselines.json
  ablations                  ablation grid -> results/ablations.json
  modality                   one Angle-class track per modality -> results/modality_tracks.json
  fusion                     multimodal fusion + missing modality -> results/fusion.json
  redaction                  release-safe margin-redaction ablation -> results/redaction.json
  finetune --data-root PATH  end-to-end fine-tune, both split protocols -> results/finetune.json
  phi-audit --data-root PATH burned-in pixel-PHI OCR triage -> results/phi_audit.json
  all      --data-root PATH  extract + audit + dedup + every experiment

Restricted clinical data is read in place via --data-root (or ORTHOBENCH_DATA_ROOT);
it is never copied into the repo. Results are aggregate-only.
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from orthobench import manifest, dedup, benchmark  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CACHE_DIR = os.path.join(HERE, "data", "feat_cache")


def _load_result(name):
    path = os.path.join(RESULTS, name)
    return json.load(open(path)) if os.path.exists(path) else None


def _save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[saved] {path}")


MODALITIES = ("intraoral", "opg", "cephalogram")
SOTA_BACKBONES = ("resnet50", "dinov2", "dinov2_large", "siglip2", "biomedclip")


def _cache_path(backbone, modality="intraoral"):
    return os.path.join(CACHE_DIR, f"{backbone}_{modality}.npz")


def _load_feat(backbone, modality="intraoral"):
    p = _cache_path(backbone, modality)
    if not os.path.exists(p):
        raise SystemExit(f"No feature cache for {backbone}/{modality} at {p}. Run: "
                         f"python run_experiment.py extract --backbone {backbone} "
                         f"--modality {modality} --data-root PATH")
    return dict(np.load(p, allow_pickle=True))


def _cached_backbones(modality="intraoral"):
    return [b for b in SOTA_BACKBONES if os.path.exists(_cache_path(b, modality))]


def cmd_extract(args):
    from orthobench import features
    root = manifest.data_root(args.data_root)
    table = manifest.modality_table(root, args.modality)
    if args.limit:
        table = table.head(args.limit)
    tag = args.modality if not args.redact_margin else \
        f"{args.modality}-redact{int(args.redact_margin * 100)}"
    feat = features.extract(table, root, backbone=args.backbone,
                            cache_path=_cache_path(args.backbone, tag),
                            redact_margin=args.redact_margin, limit=args.limit)
    print(f"[extract] {args.backbone}/{tag}: {feat['features'].shape} "
          f"(unreadable={feat.get('n_unreadable', 0)})")


def cmd_audit(args):
    root = manifest.data_root(args.data_root)
    _save("dataset_stats.json", manifest.curation_audit(root))


def cmd_dedup(args):
    feat = _load_feat(args.backbone)
    rep = dedup.neardup_report(feat["phash"], feat["patient_id"].astype(str),
                               max_dist=args.max_dist)
    _save("neardup.json", rep)


def cmd_leakage(args):
    feat = _load_feat(args.backbone)
    _save("leakage.json", benchmark.leakage_experiment(feat, seeds=tuple(range(args.seeds))))


def cmd_baselines(args):
    feats = {b: _load_feat(b) for b in _cached_backbones()}
    if not feats:
        raise SystemExit("No cached intraoral features; run `extract` first.")
    _save("baselines.json", benchmark.baselines(feats, seeds=tuple(range(args.seeds))))


def cmd_modality(args):
    feats = {m: _load_feat(args.backbone, m) for m in MODALITIES
             if os.path.exists(_cache_path(args.backbone, m))}
    if len(feats) < 2:
        raise SystemExit(f"Need >=2 modality caches for {args.backbone}; have {list(feats)}")
    _save("modality_tracks.json",
          benchmark.modality_tracks(feats, seeds=tuple(range(args.seeds))))


def cmd_fusion(args):
    feats = {m: _load_feat(args.backbone, m) for m in MODALITIES
             if os.path.exists(_cache_path(args.backbone, m))}
    if len(feats) < 3:
        raise SystemExit(f"Fusion needs all three modality caches; have {list(feats)}")
    _save("fusion.json", benchmark.fusion(feats, seeds=tuple(range(args.fusion_seeds))))


def cmd_redaction(args):
    tag = f"intraoral-redact{int(args.redact_margin * 100 or 12)}"
    if not os.path.exists(_cache_path(args.backbone, tag)):
        raise SystemExit(f"Missing {tag} cache; run `extract --redact-margin 0.12` first.")
    _save("redaction.json", benchmark.redaction_ablation(
        _load_feat(args.backbone), _load_feat(args.backbone, tag),
        seeds=tuple(range(args.seeds))))


def cmd_finetune(args):
    from orthobench import finetune
    root = manifest.data_root(args.data_root)
    table = manifest.modality_table(root, args.modality)
    arr, meta = finetune.cache_pixels(
        table, root, os.path.join(CACHE_DIR, f"pixels_{args.modality}_256.u8"))
    _save("finetune.json", finetune.experiment(
        arr, meta, seeds=tuple(range(args.ft_seeds)), epochs=args.epochs))


def cmd_phi_audit(args):
    from orthobench import phi_audit
    root = manifest.data_root(args.data_root)
    table = manifest.all_images_table(root)
    res = phi_audit.audit(table, root, workers=args.workers, limit=args.limit)
    res["summary"]["recall_probe"] = phi_audit.recall_probe(
        table, root, n=args.probe_n, workers=args.workers)
    _save("phi_audit.json", res["summary"])
    _save("phi_audit_per_image.json", res["per_image"])


def cmd_significance(args):
    lk, ft, mt = (_load_result("leakage.json"), _load_result("finetune.json"),
                  _load_result("modality_tracks.json"))
    missing = [n for n, v in (("leakage", lk), ("finetune", ft),
                              ("modality_tracks", mt)) if v is None]
    if missing:
        raise SystemExit(f"Missing results: {missing}. Run those stages first.")
    _save("significance.json", benchmark.significance(lk, ft, mt))


def cmd_ablations(args):
    dino = _load_feat("dinov2")
    resnet = _load_feat("resnet50") if os.path.exists(_cache_path("resnet50")) else None
    _save("ablations.json", benchmark.ablations(dino, resnet, seeds=tuple(range(args.seeds))))


def cmd_all(args):
    from orthobench import features
    root = manifest.data_root(args.data_root)
    for bk in args.backbones:
        for mod in MODALITIES if bk == "dinov2" else ("intraoral",):
            features.extract(table_for(root, mod), root, backbone=bk,
                             cache_path=_cache_path(bk, mod))
    features.extract(table_for(root, "intraoral"), root, backbone="dinov2",
                     cache_path=_cache_path("dinov2", "intraoral-redact12"),
                     redact_margin=0.12)
    cmd_audit(args)
    cmd_dedup(args)
    cmd_leakage(args)
    cmd_baselines(args)
    cmd_ablations(args)
    cmd_significance(args)
    cmd_modality(args)
    cmd_fusion(args)
    cmd_redaction(args)
    cmd_phi_audit(args)
    cmd_finetune(args)


def table_for(root, modality):
    return manifest.modality_table(root, modality)


def cmd_smoke(args):
    """Synthetic end-to-end run: no restricted data, no model downloads."""
    rng = np.random.RandomState(0)
    n_pat, dim = 30, 384
    pids, labs, feats, phs = [], [], [], []
    for pi in range(n_pat):
        c = pi % 3
        center = rng.randn(dim) * (1.0 + 0.4 * c)  # class-separable + patient-specific
        for _ in range(rng.randint(3, 8)):
            pids.append(f"P{pi:03d}"); labs.append(c)
            feats.append(center + rng.randn(dim) * 0.5)
            phs.append(np.uint64(rng.randint(0, 2 ** 63)))
    feat = {"features": np.array(feats, np.float32),
            "patient_id": np.array(pids), "label_numeric": np.array(labs),
            "label": np.array([["Class I", "Class II", "Class III"][c] for c in labs]),
            "phash": np.array(phs, np.uint64)}
    lk = benchmark.leakage_experiment(feat, seeds=(0, 1))
    bl = benchmark.baselines({"synthetic": feat}, seeds=(0, 1))
    ab = benchmark.ablations(feat, seeds=(0, 1))
    dd = dedup.neardup_report(feat["phash"], feat["patient_id"], max_dist=6)
    _save("smoke.json", {"leakage": lk, "baselines": bl, "ablations": ab, "neardup": dd})
    print("[smoke] OK: leakage inflation (logreg macro-F1 abs) =",
          lk["probes"]["logreg"]["inflation"]["macro_f1_abs"])


def main():
    ap = argparse.ArgumentParser(description="OrthoBench benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"smoke": cmd_smoke, "audit": cmd_audit, "dedup": cmd_dedup,
            "extract": cmd_extract, "leakage": cmd_leakage, "baselines": cmd_baselines,
            "ablations": cmd_ablations, "modality": cmd_modality, "fusion": cmd_fusion,
            "redaction": cmd_redaction, "finetune": cmd_finetune,
            "phi-audit": cmd_phi_audit, "significance": cmd_significance,
            "all": cmd_all}
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument("--data-root", default=None)
        p.add_argument("--backbone", default="dinov2", choices=list(SOTA_BACKBONES))
        p.add_argument("--backbones", nargs="+", default=list(SOTA_BACKBONES))
        p.add_argument("--modality", default="intraoral", choices=list(MODALITIES))
        p.add_argument("--seeds", type=int, default=5)
        p.add_argument("--fusion-seeds", type=int, default=10)
        p.add_argument("--ft-seeds", type=int, default=3)
        p.add_argument("--epochs", type=int, default=8)
        p.add_argument("--redact-margin", type=float, default=0.0)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--probe-n", type=int, default=250)
        p.add_argument("--max-dist", type=int, default=10)
        p.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
