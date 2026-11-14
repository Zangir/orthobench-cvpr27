"""Burned-in pixel-PHI audit: OCR triage + redaction verification.

Restricted images are read in place. This module never persists, prints, or
returns recognised text: only per-image token counts, risk categories, and
normalised bounding boxes leave the worker process. The point is triage --
reducing a whole-cohort manual PHI review to the small flagged subset a human
can actually inspect -- plus a check that blacking out the detected boxes
removes every detection on re-OCR.
"""
import os
import re
import numpy as np

CONF_MIN = 60
MAX_DIM = 1400
TESS_CONFIG = "--psm 11 --oem 3"
BAND_MAX_DIM = 2400  # native-ish resolution for the margin second pass
MARGIN = 0.12  # outer fraction of the frame where acquisition software burns overlays

# Text that legitimately appears burned into dental/orthodontic images and is
# not a patient identifier: laterality and view markers, exposure parameters,
# scale/units, and detector/manufacturer strings.
_BENIGN = {
    "l", "r", "lt", "rt", "left", "right", "a", "p", "ap", "pa", "lat", "lateral",
    "ceph", "cephalometric", "pano", "panoramic", "opg", "opt", "occlusal",
    "buccal", "frontal", "upper", "lower", "max", "mand", "mm", "cm", "kv", "kvp",
    "ma", "mas", "sec", "s", "dap", "mgy", "dose", "std", "adult", "child",
    "sirona", "planmeca", "planmeca", "vatech", "carestream", "kodak", "morita",
    "instrumentarium", "orthophos", "proline", "pax", "romexis", "dolphin",
    "nikon", "canon", "sony", "iphone", "samsung", "eos", "dslr", "flash",
    "zoom", "auto", "copy", "scan", "film", "image", "img", "dicom", "jpeg", "png",
}
_DATE = re.compile(
    r"^(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}[-/.](?:jan|feb|mar|apr|may|jun|"
    r"jul|aug|sep|oct|nov|dec)[a-z]*[-/.]?\d{0,4}|(?:jan|feb|mar|apr|may|jun|jul|"
    r"aug|sep|oct|nov|dec)[a-z]*\s?\d{1,2},?\s?\d{2,4})$", re.I)
_ID = re.compile(r"^(?=(?:.*\d){4,})[A-Za-z0-9#/-]{4,}$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z'\-]{2,}$")
_TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")


def classify(text):
    """Category for one recognised token. Text stays local to the caller."""
    t = text.strip().strip(".,;:()[]")
    low = t.lower()
    if len(t) < 3 and not _TIME.match(t):
        return "too_short"
    if low in _BENIGN:
        return "benign_annotation"
    if _DATE.match(t) or _TIME.match(t):
        return "date_or_time"
    if _ID.match(t):
        return "identifier_like"
    if _NAME.match(t):
        return "name_like"
    return "unclassified_text"


RISK = ("date_or_time", "identifier_like", "name_like")


def _ocr(im):
    """Return (texts, confs, boxes) for tokens above CONF_MIN on a PIL image."""
    import pytesseract
    d = pytesseract.image_to_data(im, config=TESS_CONFIG,
                                  output_type=pytesseract.Output.DICT)
    out = []
    for i in range(len(d["text"])):
        txt = d["text"][i]
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < CONF_MIN or not txt.strip() or not re.search(r"[A-Za-z0-9]", txt):
            continue
        out.append((txt, conf, (d["left"][i], d["top"][i], d["width"][i], d["height"][i])))
    return out


def _load(path, max_dim=MAX_DIM):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("L")
    w, h = im.size
    s = min(1.0, max_dim / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return im


def _margin_pass(path, margin=None, max_dim=BAND_MAX_DIM):
    """Second pass: OCR the four frame margins at higher resolution.

    Overlay identifiers are burned into the border and are often only 1-2% of
    frame height, which survives the whole-frame downscale poorly. Re-reading
    just the border bands at up to BAND_MAX_DIM recovers those small strings at a
    fraction of the cost of a full-resolution scan. Boxes come back in
    whole-frame normalised coordinates so the two passes can be merged.
    """
    margin = MARGIN if margin is None else margin
    try:
        im = _load(path, max_dim=max_dim)
    except Exception:
        return []
    W, H = im.size
    mh, mw = max(1, int(H * margin)), max(1, int(W * margin))
    bands = [(0, 0, W, mh), (0, H - mh, W, H), (0, 0, mw, H), (W - mw, 0, W, H)]
    out = []
    for (x0, y0, x1, y1) in bands:
        crop = im.crop((x0, y0, x1, y1))
        for txt, conf, (bx, by, bw, bh) in _ocr(crop):
            out.append((txt, conf, ((x0 + bx) / W, (y0 + by) / H, bw / W, bh / H)))
    return out


def _tokens(path):
    """Both detector passes, as (text, conf, normalised box) in frame coordinates."""
    im = _load(path)
    W, H = im.size
    toks = [(t, c, (x / W, y / H, bw / W, bh / H))
            for t, c, (x, y, bw, bh) in _ocr(im)]
    n_full = len(toks)
    seen = {(round(b[0], 2), round(b[1], 2)) for _, _, b in toks}
    for t, c, b in _margin_pass(path):
        if (round(b[0], 2), round(b[1], 2)) not in seen:
            toks.append((t, c, b))
    return im, toks, n_full


def scan_one(args):
    """Worker: OCR one image, return counts/boxes only (never the text)."""
    record_id, modality, path, verify_redaction = args
    try:
        im, toks, n_full = _tokens(path)
    except Exception:
        return {"record_id": record_id, "modality": modality, "unreadable": 1}
    cats, boxes = {}, []
    for txt, conf, (x, y, bw, bh) in toks:
        c = classify(txt)
        cats[c] = cats.get(c, 0) + 1
        if c in RISK:
            boxes.append([round(x, 4), round(y, 4), round(bw, 4), round(bh, 4), int(conf)])
    rec = {"record_id": record_id, "modality": modality, "unreadable": 0,
           "n_tokens": len(toks), "n_tokens_full_frame_pass": n_full,
           "categories": cats,
           "n_risk": sum(cats.get(c, 0) for c in RISK), "risk_boxes": boxes}
    if verify_redaction and boxes:
        rec["residual_after_redaction"] = _residual(im, boxes)
    return rec


def _residual(im, boxes, pad=0.01):
    """Black out the risk boxes (padded) and re-OCR; return residual risk tokens."""
    from PIL import ImageDraw
    im2 = im.copy()
    W, H = im2.size
    d = ImageDraw.Draw(im2)
    for x, y, bw, bh, _ in boxes:
        x0 = max(0, int((x - pad) * W)); y0 = max(0, int((y - pad) * H))
        x1 = min(W, int((x + bw + pad) * W)); y1 = min(H, int((y + bh + pad) * H))
        d.rectangle([x0, y0, x1, y1], fill=0)
    return sum(1 for txt, _, _ in _ocr(im2) if classify(txt) in RISK)


def audit(table, root, workers=8, limit=None, verify_redaction=True, log=print):
    """OCR-triage every image in `table`. Returns per-image records + summary.

    A thread pool is used deliberately: pytesseract shells out to the tesseract
    binary, so threads give real parallelism and avoid the macOS spawn-pickling
    trap that a process pool hits when the driver is not an importable module.
    """
    from multiprocessing.pool import ThreadPool
    rows = [(r.record_id, r.modality, os.path.join(root, r.relative_path),
             verify_redaction) for r in table.itertuples(index=False)]
    if limit:
        rows = rows[:limit]
    recs = []
    with ThreadPool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(scan_one, rows, chunksize=16)):
            recs.append(rec)
            if (i + 1) % 1000 == 0:
                log(f"[phi] {i+1}/{len(rows)} scanned")
    return {"per_image": recs, "summary": summarize(recs)}


def summarize(recs):
    ok = [r for r in recs if not r.get("unreadable")]
    mods = sorted({r["modality"] for r in ok})
    cats = {}
    for r in ok:
        for c, n in r.get("categories", {}).items():
            cats[c] = cats.get(c, 0) + n
    flagged = [r for r in ok if r.get("n_risk", 0) > 0]
    resid = [r["residual_after_redaction"] for r in flagged
             if "residual_after_redaction" in r]
    per_mod = {}
    for m in mods:
        sub = [r for r in ok if r["modality"] == m]
        fl = [r for r in sub if r.get("n_risk", 0) > 0]
        per_mod[m] = {
            "images": len(sub),
            "with_any_text": sum(1 for r in sub if r["n_tokens"] > 0),
            "flagged_risk": len(fl),
            "flagged_pct": round(100 * len(fl) / len(sub), 2) if sub else None,
            "risk_tokens": int(sum(r.get("n_risk", 0) for r in sub))}
    return {
        "images_scanned": len(ok), "unreadable": len(recs) - len(ok),
        "conf_min": CONF_MIN, "max_dim": MAX_DIM, "tesseract_config": TESS_CONFIG,
        "images_with_any_text": sum(1 for r in ok if r["n_tokens"] > 0),
        "images_flagged_risk": len(flagged),
        "flagged_pct_of_cohort": round(100 * len(flagged) / len(ok), 2) if ok else None,
        "token_categories": cats,
        "risk_tokens_total": int(sum(r.get("n_risk", 0) for r in ok)),
        "risk_boxes_total": int(sum(len(r.get("risk_boxes", [])) for r in ok)),
        "by_modality": per_mod,
        "margin_tier": margin_stats(recs),
        "redaction_verified_images": len(resid),
        "redaction_residual_risk_tokens": int(sum(resid)),
        "redaction_clean_images": int(sum(1 for x in resid if x == 0)),
        "flagged_record_ids": sorted(r["record_id"] for r in flagged),
        "risk_box_area_pct": _box_area_stats(flagged),
    }


def _box_area_stats(flagged):
    a = [sum(b[2] * b[3] for b in r.get("risk_boxes", [])) * 100 for r in flagged]
    if not a:
        return None
    a = np.array(a)
    return {"median": round(float(np.median(a)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "max": round(float(a.max()), 3)}


def margin_stats(recs):
    """Split flagged images by whether their risk boxes sit in the frame margin.

    Burned-in overlay identifiers are almost always written into the outer border
    of the frame by the acquisition software; a detection in the middle of the
    anatomy is far more likely to be OCR noise on a tooth, bracket, or mirror.
    Reporting both tiers keeps the flagged count honest about false positives.
    """
    flagged = [r for r in recs if r.get("n_risk", 0) > 0]
    def in_margin(b):
        x, y, w, h = b[:4]
        return (y + h <= MARGIN or y >= 1 - MARGIN
                or x + w <= MARGIN or x >= 1 - MARGIN)
    any_m = sum(1 for r in flagged if any(in_margin(b) for b in r["risk_boxes"]))
    all_m = sum(1 for r in flagged if all(in_margin(b) for b in r["risk_boxes"]))
    return {"margin_fraction": MARGIN, "flagged": len(flagged),
            "any_box_in_margin": any_m, "all_boxes_in_margin": all_m,
            "no_box_in_margin": len(flagged) - any_m}


_FAKE_FIRST = ["AMINA", "KHALID", "NOURA", "YOUSSEF", "LAYLA", "OMAR", "SARA", "TARIQ"]
_FAKE_LAST = ["ALMANSOURI", "ALHASHIMI", "ALBALUSHI", "ALKAABI", "ALSUWAIDI", "ALNAQBI"]


def recall_probe(table, root, n=200, seed=0, workers=8, log=print):
    """Positive control: burn a SYNTHETIC identifier into real frames and re-detect.

    Names, dates and MRNs here are generated, not patient data. The synthesised
    frame is written to a scratch file and pushed through the same two-pass
    detector used on the cohort, so the reported sensitivity is the pipeline's,
    not an easier in-memory variant. Sensitivity is broken out by overlay font
    size because small strings are the hard case.
    """
    import tempfile
    from multiprocessing.pool import ThreadPool
    from PIL import ImageDraw, ImageFont
    idx = np.random.RandomState(seed).choice(
        len(table), size=min(n, len(table)), replace=False)
    rows = list(table.iloc[idx].itertuples(index=False))
    font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    tmpdir = tempfile.mkdtemp(prefix="phi_probe_")

    def one(job):
        k, r = job
        rng = np.random.RandomState(seed * 100003 + k)
        try:
            im = _load(os.path.join(root, r.relative_path), max_dim=BAND_MAX_DIM)
        except Exception:
            return None
        W, H = im.size
        txt = (f"{_FAKE_FIRST[rng.randint(len(_FAKE_FIRST))]} "
               f"{_FAKE_LAST[rng.randint(len(_FAKE_LAST))]}  "
               f"{rng.randint(1,29):02d}/{rng.randint(1,13):02d}/{rng.randint(1990,2026)}  "
               f"MRN{rng.randint(1000000,9999999)}")
        frac = 0.010 + 0.025 * rng.rand()          # 1.0-3.5% of frame height
        size = max(8, int(H * frac))
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            font = ImageFont.load_default()
        d = ImageDraw.Draw(im)
        bb = d.textbbox((0, 0), txt, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = max(2, int(0.01 * min(W, H)))
        corner = rng.randint(4)
        x = pad if corner in (0, 2) else max(pad, W - tw - pad)
        y = pad if corner in (0, 1) else max(pad, H - th - 3 * pad)
        patch = np.asarray(im.crop((x, y, min(W, x + tw), min(H, y + th))),
                           dtype=np.float32)
        fill = 255 if (patch.size and patch.mean() < 128) else 0
        d.text((x, y), txt, fill=fill, font=font)
        path = os.path.join(tmpdir, f"probe_{k:05d}.png")
        im.save(path)
        try:
            _, toks, _ = _tokens(path)
            risk = sum(1 for t, _, _ in toks if classify(t) in RISK)
        finally:
            os.remove(path)
        return {"detected": int(risk > 0), "risk_tokens": risk,
                "font_px": size, "font_frac_pct": round(100 * frac, 2),
                "corner": int(corner)}

    with ThreadPool(workers) as pool:
        out = [r for r in pool.map(one, list(enumerate(rows))) if r]
    os.rmdir(tmpdir)
    det = sum(r["detected"] for r in out)
    bands = {}
    for lo, hi, name in [(0, 1.5, "1.0-1.5pct"), (1.5, 2.5, "1.5-2.5pct"),
                         (2.5, 99, "2.5-3.5pct")]:
        sub = [r for r in out if lo <= r["font_frac_pct"] < hi]
        if sub:
            bands[name] = {"n": len(sub),
                           "sensitivity": round(sum(r["detected"] for r in sub) / len(sub), 4)}
    return {"n_probed": len(out), "detected": det,
            "sensitivity": round(det / len(out), 4) if out else None,
            "font_px_median": int(np.median([r["font_px"] for r in out])) if out else None,
            "sensitivity_by_font_size": bands,
            "note": "synthetic identifiers burned into real frames; no patient text used"}
