"""Frozen feature extraction for every OrthoBench modality.

Backbones span three families so the baseline suite is not a single-model
sanity check: an ImageNet-supervised CNN (ResNet-50), self-supervised ViTs at
two scales (DINOv2 ViT-S/14 and ViT-L/14), a modern image-text encoder
(SigLIP-2 base) and a biomedical vision-language encoder (BiomedCLIP). Each
carries its own input size and normalisation, declared in BACKBONES.

Features are cached under a gitignored cache dir keyed by (backbone, modality);
the restricted images are read in place and never copied into the repo. A
64-bit DCT perceptual hash is computed in the same pass for near-duplicate
analysis.
"""
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
_IN_MEAN, _IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_HALF = (0.5, 0.5, 0.5)

# backbone -> (family, hub id, feature dim, input size, mean, std)
BACKBONES = {
    "resnet50": ("torchvision", "resnet50", 2048, 224, _IN_MEAN, _IN_STD),
    "dinov2": ("torchhub", "dinov2_vits14", 384, 224, _IN_MEAN, _IN_STD),
    "dinov2_large": ("hf_vision", "facebook/dinov2-large", 1024, 224, _IN_MEAN, _IN_STD),
    "siglip2": ("hf_siglip", "google/siglip2-base-patch16-224", 768, 224, _HALF, _HALF),
    "biomedclip": ("open_clip",
                   "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                   512, 224, (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
    # CVPR 2027 zoo extension: more scales, a radiology-pretrained ViT, modern CNNs,
    # and a general-domain image-text encoder.
    "dinov2_base": ("torchhub", "dinov2_vitb14", 768, 224, _IN_MEAN, _IN_STD),
    "rad_dino": ("hf_vision", "microsoft/rad-dino", 768, 518, (0.5307,) * 3, (0.2583,) * 3),
    "convnext_tiny": ("torchvision", "convnext_tiny", 768, 224, _IN_MEAN, _IN_STD),
    "convnext_large": ("torchvision", "convnext_large", 1536, 224, _IN_MEAN, _IN_STD),
    "resnet18": ("torchvision", "resnet18", 512, 224, _IN_MEAN, _IN_STD),
    "clip_b16": ("open_clip_named", "ViT-B-16|openai", 512, 224,
                 (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
}

_TV_WEIGHTS = {"resnet50": "ResNet50_Weights.IMAGENET1K_V2", "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
               "convnext_tiny": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
               "convnext_large": "ConvNeXt_Large_Weights.IMAGENET1K_V1"}


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_rgb(path, size=224, resize=None, redact_margin=0.0):
    """Load, optionally blank the frame border, then resize + centre-crop.

    `redact_margin` blacks out the outer fraction of the original frame before
    resizing. That border band is where acquisition software burns in identifiers,
    so this is the release-safe variant of the input: it lets us measure what a
    blanket-redacted release would cost the benchmark.
    """
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    resize = resize or int(round(size * 256 / 224))
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if redact_margin > 0:
        mw, mh = int(w * redact_margin), int(h * redact_margin)
        d = ImageDraw.Draw(im)
        for box in [(0, 0, w, mh), (0, h - mh, w, h), (0, 0, mw, h), (w - mw, 0, w, h)]:
            d.rectangle(box, fill=(0, 0, 0))
    s = resize / min(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return im.crop((l, t, l + size, t + size))


def phash64(im, hs=8, hf=32):
    """64-bit DCT perceptual hash (no scipy dependency)."""
    g = np.asarray(im.convert("L").resize((hf, hf)), dtype=np.float32)

    def dct1(x):
        n = x.shape[-1]
        k = np.arange(n)
        basis = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
        return x @ basis.T
    d = dct1(dct1(g).T).T
    low = d[:hs, :hs]
    med = np.median(low[1:, 1:])
    bits = (low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return np.uint64(val)


class Backbone:
    """Frozen encoder wrapper: exposes .dim/.size/.mean/.std and .embed(batch)."""

    def __init__(self, name, device):
        import torch
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name}; choose from {sorted(BACKBONES)}")
        fam, hub, dim, size, mean, std = BACKBONES[name]
        self.name, self.family, self.dim, self.size = name, fam, dim, size
        self.mean, self.std, self.device = mean, std, device
        os.environ.setdefault("SSL_CERT_FILE", _certifi())
        if fam == "torchhub":
            self.model = torch.hub.load("facebookresearch/dinov2", hub, verbose=False)
        elif fam == "torchvision":
            import torchvision.models as M
            enum, member = _TV_WEIGHTS[hub].split(".")
            net = getattr(M, hub)(weights=getattr(getattr(M, enum), member))
            if hub.startswith("resnet"):
                net.fc = torch.nn.Identity()
            else:  # ConvNeXt: keep the pre-classifier LayerNorm + flatten
                net.classifier[2] = torch.nn.Identity()
            self.model = net
        elif fam == "hf_vision":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(hub)
        elif fam == "hf_siglip":
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(hub)
        elif fam == "open_clip":
            import open_clip
            self.model, _, _ = open_clip.create_model_and_transforms(hub)
        elif fam == "open_clip_named":
            import open_clip
            arch, tag = hub.split("|")
            self.model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag)
        else:
            raise ValueError(fam)
        self.model = self.model.eval().to(device)

    def embed(self, x):
        import torch
        with torch.no_grad():
            if self.family in ("torchhub", "torchvision"):
                f = self.model(x)
            elif self.family == "hf_vision":
                f = self.model(pixel_values=x).last_hidden_state[:, 0]
            elif self.family == "hf_siglip":
                f = self.model(pixel_values=x).pooler_output
            else:  # open_clip / open_clip_named
                f = self.model.encode_image(x)
        return f.float().cpu().numpy()


def _certifi():
    try:
        import certifi
        return certifi.where()
    except Exception:
        return ""


def build_model(backbone, device):
    """Back-compatible shim: returns (module, dim) for the frozen encoder."""
    bb = Backbone(backbone, device)
    return bb.model, bb.dim


def cache_name(backbone, modality="intraoral"):
    return f"{backbone}_{modality}.npz"


def extract(table, root, backbone="dinov2", cache_path=None, batch=32,
            with_phash=True, limit=None, redact_margin=0.0, log=print):
    """Extract frozen features for a modality `table` (from manifest.modality_table).
    Caches to cache_path (.npz). Returns the loaded dict if the cache exists."""
    import torch
    if cache_path and os.path.exists(cache_path):
        log(f"[features] using cache {cache_path}")
        return dict(np.load(cache_path, allow_pickle=True))
    device = _device()
    bb = Backbone(backbone, device)
    mean = torch.tensor(bb.mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std = torch.tensor(bb.std, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    rows = list(table.itertuples(index=False))
    if limit:
        rows = rows[:limit]
    feats, recs, pids, labs, labn, phs = [], [], [], [], [], []
    buf, meta = [], []
    bad = 0

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device)
        x = (x - mean) / std
        f = bb.embed(x)
        for k, mt in enumerate(meta):
            feats.append(f[k]); recs.append(mt[0]); pids.append(mt[1])
            labs.append(mt[2]); labn.append(mt[3]); phs.append(mt[4])
        buf.clear(); meta.clear()

    for i, r in enumerate(rows):
        p = os.path.join(root, r.relative_path)
        try:
            im = _load_rgb(p, size=bb.size, redact_margin=redact_margin)
            ph = phash64(im) if with_phash else np.uint64(0)
            arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        except Exception:
            bad += 1
            continue
        buf.append(arr)
        meta.append((r.record_id, r.patient_id, r.label, int(r.label_numeric), ph))
        if len(buf) >= batch:
            flush()
        if (i + 1) % 1000 == 0:
            log(f"[features] {i+1}/{len(rows)} processed (bad={bad})")
    flush()
    out = {"features": np.stack(feats).astype(np.float32),
           "record_id": np.array(recs), "patient_id": np.array(pids),
           "label": np.array(labs), "label_numeric": np.array(labn),
           "phash": np.array(phs, dtype=np.uint64), "dim": bb.dim, "backbone": backbone,
           "redact_margin": redact_margin, "n_unreadable": bad}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **out)
        log(f"[features] wrote {out['features'].shape} to {cache_path} (bad={bad})")
    return out


def extract_many(table, root, backbones, cache_dir, tag, batch=32, log=print):
    """Decode each restricted image ONCE and embed it with several frozen encoders.

    Writes one cache per backbone with the same schema as `extract` (the pHash is
    computed on the 224-px crop, identical to `extract`). Backbones whose cache
    already exists are skipped. Images are read in place and never copied.
    """
    import torch
    todo = [b for b in backbones if not os.path.exists(os.path.join(cache_dir, cache_name(b, tag)))]
    if not todo:
        log("[features] all caches present"); return
    device = _device()
    nets = {b: Backbone(b, device) for b in todo}
    norm = {b: (torch.tensor(n.mean).view(1, 3, 1, 1).to(device), torch.tensor(n.std).view(1, 3, 1, 1).to(device))
            for b, n in nets.items()}
    sizes = sorted({n.size for n in nets.values()})
    feats = {b: [] for b in todo}
    meta, buf, bad = [], {s: [] for s in sizes}, 0

    def flush():
        if not meta or not buf[sizes[0]]:
            return
        stacked = {s: torch.from_numpy(np.stack(buf[s])).to(device) for s in sizes}
        for b, n in nets.items():
            m, sd = norm[b]
            feats[b].append(n.embed((stacked[n.size] - m) / sd))
        for s in sizes:
            buf[s].clear()

    rows = list(table.itertuples(index=False))
    for i, r in enumerate(rows):
        try:
            crops = {s: _load_rgb(os.path.join(root, r.relative_path), size=s) for s in sizes}
            c224 = crops[224] if 224 in crops else _load_rgb(os.path.join(root, r.relative_path), size=224)
            ph = phash64(c224)
        except Exception:
            bad += 1
            continue
        for s in sizes:
            buf[s].append(np.asarray(crops[s], dtype=np.float32).transpose(2, 0, 1) / 255.0)
        meta.append((r.record_id, r.patient_id, r.label, int(r.label_numeric), ph))
        if len(buf[sizes[0]]) >= batch:
            flush()
        if (i + 1) % 500 == 0:
            log(f"[features] {tag}: {i+1}/{len(rows)} (bad={bad})")
    flush()
    rec, pid, lab, labn, phs = (np.array(v) for v in zip(*meta))
    for b in todo:
        out = {"features": np.concatenate(feats[b]).astype(np.float32), "record_id": rec,
               "patient_id": pid, "label": lab, "label_numeric": labn,
               "phash": phs.astype(np.uint64), "dim": nets[b].dim, "backbone": b,
               "redact_margin": 0.0, "n_unreadable": bad}
        path = os.path.join(cache_dir, cache_name(b, tag))
        np.savez_compressed(path, **out)
        log(f"[features] wrote {out['features'].shape} to {path}")
