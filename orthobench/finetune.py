"""End-to-end fine-tuning baseline, run under both split protocols.

The frozen probes are deliberately weak; this module supplies the high-capacity
reference point a benchmark suite needs, and doubles as the test of the paper's
central claim that probe-measured leakage inflation is a *lower bound*: a network
that can update every weight is free to memorise patients, so fine-tuning under
an image-level split should inflate more than a linear probe does, not less.

Decoded 256-px crops are cached once into a uint8 memmap (the restricted images
are read in place and never copied as images); every epoch then streams from
that memmap, which keeps a full run within a single 16 GB GPU budget.
"""
import os
import numpy as np

SIZE, CACHE_SIZE = 224, 256


def cache_pixels(table, root, cache_path, log=print):
    """Decode every row once into a uint8 memmap of CACHE_SIZE-square crops."""
    meta_path = cache_path + ".meta.npz"
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        m = dict(np.load(meta_path, allow_pickle=True))
        arr = np.memmap(cache_path, dtype=np.uint8, mode="r",
                        shape=(int(m["n"]), CACHE_SIZE, CACHE_SIZE, 3))
        log(f"[finetune] using pixel cache {cache_path} {arr.shape}")
        return arr, m
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    rows = list(table.itertuples(index=False))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    arr = np.memmap(cache_path, dtype=np.uint8, mode="w+",
                    shape=(len(rows), CACHE_SIZE, CACHE_SIZE, 3))
    pids, ys, keep = [], [], 0
    for i, r in enumerate(rows):
        try:
            im = Image.open(os.path.join(root, r.relative_path)).convert("RGB")
            w, h = im.size
            s = CACHE_SIZE / min(w, h)
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
            w, h = im.size
            l, t = (w - CACHE_SIZE) // 2, (h - CACHE_SIZE) // 2
            arr[keep] = np.asarray(im.crop((l, t, l + CACHE_SIZE, t + CACHE_SIZE)))
        except Exception:
            continue
        pids.append(r.patient_id); ys.append(int(r.label_numeric)); keep += 1
        if (i + 1) % 1000 == 0:
            log(f"[finetune] cached {i+1}/{len(rows)}")
    arr.flush()
    m = {"n": keep, "patient_id": np.array(pids), "label_numeric": np.array(ys)}
    np.savez(meta_path, **m)
    log(f"[finetune] pixel cache {cache_path}: {keep}/{len(rows)} images")
    return np.memmap(cache_path, dtype=np.uint8, mode="r",
                     shape=(keep, CACHE_SIZE, CACHE_SIZE, 3)), m


def _model(device, n_classes=3, arch="dinov2"):
    """Backbone + linear head. arch: dinov2 (ViT-S/14, historical), resnet50, convnext_tiny."""
    import torch
    if arch == "dinov2":
        torch.hub.set_dir(torch.hub.get_dir())
        net, dim = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False), 384
    elif arch == "resnet50":
        import torchvision.models as M
        net, dim = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2), 2048
        net.fc = torch.nn.Identity()
    elif arch == "convnext_tiny":
        import torchvision.models as M
        net, dim = M.convnext_tiny(weights=M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1), 768
        net.classifier[2] = torch.nn.Identity()
    else:
        raise ValueError(arch)
    head = torch.nn.Linear(dim, n_classes)
    torch.nn.init.zeros_(head.bias); torch.nn.init.normal_(head.weight, std=0.01)
    return net.to(device), head.to(device)


def _batches(idx, bs, rng=None):
    order = idx.copy()
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


def _prep(arr, sel, train, rng, mean, std, device):
    import torch
    x = np.asarray(arr[sel], dtype=np.float32) / 255.0
    if train:
        o = rng.randint(0, CACHE_SIZE - SIZE + 1, size=2)
        x = x[:, o[0]:o[0] + SIZE, o[1]:o[1] + SIZE]
        x = np.clip(x * (0.9 + 0.2 * rng.rand()), 0, 1)
    else:
        o = (CACHE_SIZE - SIZE) // 2
        x = x[:, o:o + SIZE, o:o + SIZE]
    t = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(device)
    return (t - mean) / std


def train_eval(arr, meta, tr, te, seed, epochs=8, bs=32, lr_bb=1e-5, lr_head=1e-3,
               device=None, log=print, arch="dinov2", warmup_steps=0):
    """Fine-tune on `tr`, return test probabilities on `te` (image order preserved).

    warmup_steps=0 is the protocol recipe (cosine decay from the first step). warmup_steps>0
    (post-plan, 2026-11-17) ramps both learning rates linearly from 0 over that many steps and
    then applies the same cosine decay over the remaining steps."""
    import torch
    from .features import _device, _IN_MEAN, _IN_STD
    device = device or _device()
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    y = meta["label_numeric"].astype(int)
    net, head = _model(device, arch=arch)
    mean = torch.tensor(_IN_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(_IN_STD).view(1, 3, 1, 1).to(device)
    cnt = np.bincount(y[tr], minlength=3).astype(np.float64)
    w = torch.tensor((cnt.sum() / (3 * np.maximum(cnt, 1))), dtype=torch.float32).to(device)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([{"params": net.parameters(), "lr": lr_bb},
                             {"params": head.parameters(), "lr": lr_head}],
                            weight_decay=0.05)
    steps = max(1, epochs * (len(tr) // bs + 1))
    if warmup_steps > 0:
        import math
        rest = max(1, steps - warmup_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warmup_steps if k < warmup_steps
                                                  else 0.5 * (1 + math.cos(math.pi * min(k - warmup_steps, rest) / rest)))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for ep in range(epochs):
        net.train(); head.train()
        tot, nb = 0.0, 0
        for sel in _batches(tr, bs, rng):
            sel = np.sort(sel)
            x = _prep(arr, sel, True, rng, mean, std, device)
            t = torch.from_numpy(y[sel]).long().to(device)
            opt.zero_grad()
            loss = lossf(head(net(x)), t)
            loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        log(f"[finetune] seed {seed} epoch {ep+1}/{epochs} loss {tot/max(nb,1):.4f}")
    net.eval(); head.eval()
    probs = np.zeros((len(te), 3), dtype=np.float32)
    order = np.argsort(te)
    with torch.no_grad():
        for k in range(0, len(te), bs):
            pos = order[k:k + bs]
            sel = te[pos]
            x = _prep(arr, np.sort(sel), False, rng, mean, std, device)
            probs[pos[np.argsort(sel)]] = torch.softmax(head(net(x)), 1).cpu().numpy()
    del net, head, opt
    if device == "mps":
        torch.mps.empty_cache()
    return probs


def experiment(arr, meta, seeds=(0, 1, 2), epochs=8, test_frac=0.3, log=print):
    """Fine-tune under image-level and patient-level splits; report the inflation."""
    from . import splits, evaluate as ev
    pid = meta["patient_id"].astype(str)
    y = meta["label_numeric"].astype(int)
    pmap = {p: int(y[pid == p][0]) for p in np.unique(pid)}
    rec = {"image_level": [], "patient_level": [], "patient_agg": []}
    for s in seeds:
        tr, te = splits.patient_level_split(pid, y, seed=s, test_frac=test_frac)
        prob = train_eval(arr, meta, tr, te, seed=s, epochs=epochs, log=log)
        rec["patient_level"].append(ev.image_metrics(y[te], prob.argmax(1)))
        at, ap = ev.aggregate_to_patient(prob, pid[te], pmap)
        rec["patient_agg"].append(ev.image_metrics(at, ap))
        tri, tei = splits.image_level_split(y, seed=1000 + s, test_frac=test_frac)
        probi = train_eval(arr, meta, tri, tei, seed=s, epochs=epochs, log=log)
        rec["image_level"].append(ev.image_metrics(y[tei], probi.argmax(1)))
        log(f"[finetune] seed {s}: img {rec['image_level'][-1]['macro_f1']:.4f} "
            f"pat {rec['patient_level'][-1]['macro_f1']:.4f} "
            f"pat-agg {rec['patient_agg'][-1]['macro_f1']:.4f}")
    out = {"model": "dinov2_vits14_finetuned", "epochs": epochs, "seeds": list(seeds),
           "test_frac": test_frac, "n_images": int(len(y)),
           "n_patients": int(len(pmap)), "regimes": {}}
    for k, v in rec.items():
        f1 = [r["macro_f1"] for r in v]
        acc = [r["accuracy"] for r in v]
        # Per-class F1 over ALL seeds: a single seed on the 115-patient minority
        # class is far too noisy to carry a claim.
        pca = np.asarray([r["per_class_f1"] for r in v], dtype=float)
        out["regimes"][k] = {
            "macro_f1": ev.mean_std(f1), "accuracy": ev.mean_std(acc),
            "per_class_f1_seed0": v[0]["per_class_f1"],
            "per_class_f1_mean": [round(float(x), 4) for x in pca.mean(axis=0)],
            "per_class_f1_std": [round(float(x), 4) for x in pca.std(axis=0)],
            "per_class_f1_per_seed": [[round(float(x), 4) for x in row] for row in pca],
            "per_seed": {"seed": list(seeds),
                         "macro_f1": [round(float(x), 6) for x in f1],
                         "accuracy": [round(float(x), 6) for x in acc]}}
    il = out["regimes"]["image_level"]["macro_f1"]["mean"]
    pl = out["regimes"]["patient_level"]["macro_f1"]["mean"]
    ila = out["regimes"]["image_level"]["accuracy"]["mean"]
    pla = out["regimes"]["patient_level"]["accuracy"]["mean"]
    # Seeds index the patient partition identically in both regimes, so the spread of
    # the per-seed DIFFERENCE is the statistic that decides whether the inflation is
    # resolved; persist it so the paper can quote it without recomputation.
    diff = (np.array(out["regimes"]["image_level"]["per_seed"]["macro_f1"])
            - np.array(out["regimes"]["patient_level"]["per_seed"]["macro_f1"]))
    out["inflation"] = {"macro_f1_abs": round(il - pl, 4),
                        "macro_f1_std": round(float(diff.std()), 4),
                        "macro_f1_per_seed": [round(float(x), 6) for x in diff],
                        "macro_f1_rel_pct": round((il - pl) / pl * 100, 1) if pl else None,
                        "exceeds_seed_noise": bool(abs(diff.mean()) > 2 * float(diff.std())),
                        "accuracy_abs": round(ila - pla, 4),
                        "accuracy_rel_pct": round((ila - pla) / pla * 100, 1) if pla else None}
    return out
