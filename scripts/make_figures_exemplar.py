#!/usr/bin/env python3
"""Figures and LaTeX tables of the CVPR 2027 exemplar rewrite (EXEMPLAR_PLAN.md, Part 3).

Every value is read from results/*.json. The only image content is the set of redacted,
downsampled thumbnails already published in the primer figure (paper/figures/fig9_primer.pdf);
they are read from that PDF at render time and never written to the repository.

  python scripts/make_figures_exemplar.py [--paper ../../paper]

Writes paper/figures/ex_*.pdf|png and paper/sections/generated/ex_*.tex
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Ellipse  # noqa: E402

RES = HERE.parent / "results"
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, GRAY = ("#0072B2", "#E69F00", "#009E73", "#D55E00",
                                               "#CC79A7", "#56B4E9", "#8a8a8a")
HEAD_COLOR = {"logreg": BLUE, "knn1": VERM, "knn5": ORANGE, "fine-tuned": GREEN}
HEAD_NAME = {"knn1": "1-NN", "knn5": "5-NN", "logreg": "linear", "fine-tuned": "fine-tuned"}
CLS_COLOR = [GREEN, ORANGE, VERM]
FAMILY = {"ResNet-18": "ImageNet", "ResNet-50": "ImageNet", "ConvNeXt-T": "ImageNet",
          "ConvNeXt-L": "ImageNet", "DINOv2-S": "DINOv2", "DINOv2-B": "DINOv2", "DINOv2-L": "DINOv2",
          "RAD-DINO": "Radiology", "SigLIP-2": "Image-text", "CLIP-B/16": "Image-text",
          "BiomedCLIP": "Image-text"}
PANEL_EDGE = "#7F9FBF"
PANEL_FACE = "#F3F7FB"


def load(name):
    return json.loads((RES / name).read_text())


def setup():
    ps.style()
    # Arial is narrower than DejaVu Sans, so >= 7 pt labels fit the same panels (review 2026-11-14)
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "font.size": 7.3, "xtick.labelsize": 7.3, "ytick.labelsize": 7.3,
                         "axes.labelsize": 7.3, "axes.titlesize": 7.8, "legend.fontsize": 7.3,
                         "lines.markersize": 3, "axes.grid": True})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[fig] {name}")


def primer_thumbnails(pdf):
    """Redacted thumbnails already published in the primer figure, ordered by position."""
    import fitz
    from PIL import Image
    doc = fitz.open(pdf)
    page = doc[0]
    items = sorted(page.get_image_info(xrefs=True), key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))
    ims = [Image.open(io.BytesIO(doc.extract_image(b["xref"])["image"])).convert("RGB") for b in items]
    # first row: OPG, cephalogram; second row: intraoral frames 1-5 (left to right)
    return {"opg": ims[0], "cephalogram": ims[1], "intraoral": ims[2:7]}


# ----------------------------------------------------------------- legibility (review 2026-11-14)
# Every label must print at >= 7 pt (0.7 x the 10 pt body). Figures 1, 2, 5, 6 print at about
# 0.98 x their 7.0 in width, Fig. 3 at about 0.97 x (3.25 in into one column), Fig. 4 at about
# 1.17 x. FS is the smallest size used in the full-width figures (7.3 x 0.98 = 7.15 pt).
FS = 7.3


def panel(fig, x, y, w, h, title, num, fig_h):
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, linewidth=1.0))
    aspect = (w * 7.0) / (h * fig_h)                       # keep the number badge round
    ax.add_patch(Ellipse((0.075, 0.93), 0.095, 0.095 * aspect, facecolor="white", edgecolor="#222222", lw=0.8,
                         clip_on=False))
    ax.text(0.075, 0.928, str(num), ha="center", va="center", fontsize=FS, fontweight="bold")
    ax.text(0.145, 0.93, title, ha="left", va="center", fontsize=7.8, fontweight="bold")
    return ax


def molar(ax, x, y, w, h, col, up=True):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=col, edgecolor="#333333", lw=0.6, zorder=3))
    cy = y + h if up else y
    for cx in (x + 0.26 * w, x + 0.74 * w):
        ax.add_patch(Ellipse((cx, cy), 0.34 * w, 0.30 * h, facecolor=col, edgecolor="#333333", lw=0.6, zorder=3))


# ----------------------------------------------------------------- Fig 1 (teaser)
SUB = "#5a5a5a"
TILE_EDGE = "#B8C7D6"


def tile(ax, x, y, w, h, edge=TILE_EDGE, lw=0.7):
    """White inner tile (SiteShift Fig. 1 style) inside a LUWA-style rounded panel."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.025",
                                facecolor="white", edgecolor=edge, lw=lw, zorder=1))


def subtitle(ax, text):
    ax.text(0.06, 0.855, text, ha="left", va="center", fontsize=FS, color=SUB)


def fig1(out, thumbs):
    """Teaser. Surgical revision 2026-11-15: same facts; panels 2-4 redrawn with grey
    subtitles, white tiles, stat chips and cleaner slope charts (LUWA Fig. 1 panels,
    SiteShift Fig. 1 "glimpse" tiles)."""
    ds, ra, sz, zr = load("dataset_stats.json"), load("reference_audit.json"), load("split_zoo.json"), load("zoo_reference.json")
    pr = ra["primary"]
    FH = 2.6
    fig = plt.figure(figsize=(7.0, FH))
    W, G = 0.243, 0.0093
    xs = [G * 0.5 + i * (W + G) for i in range(4)]
    Y0, H = 0.02, 0.96
    asp = W * 7.0 / (H * FH)                               # panel x-unit / y-unit in inches

    def sub(i, u0, v0, du, dv):                            # panel coords -> figure axes
        return fig.add_axes([xs[i] + W * u0, Y0 + H * v0, W * du, H * dv])
    # (1) linked records (content unchanged; subtitle added)
    ax = panel(fig, xs[0], Y0, W, H, "Linked clinical records", 1, FH)
    subtitle(ax, "every record type of a patient")
    for j, im in enumerate(thumbs["intraoral"]):
        x0 = 0.05 + j * 0.182
        ax.imshow(np.asarray(im), extent=(x0, x0 + 0.17, 0.595, 0.74), aspect="auto", zorder=2)
    ax.text(0.5, 0.765, "intraoral photo series", ha="center", fontsize=FS, color="#333333")
    ax.imshow(np.asarray(thumbs["opg"]), extent=(0.07, 0.47, 0.32, 0.50), aspect="auto", zorder=2)
    ax.imshow(np.asarray(thumbs["cephalogram"]), extent=(0.57, 0.91, 0.32, 0.50), aspect="auto", zorder=2)
    ax.text(0.27, 0.525, "panoramic (OPG)", ha="center", fontsize=FS, color="#333333")
    ax.text(0.74, 0.525, "cephalogram", ha="center", fontsize=FS, color="#333333")
    n = ds["images_by_modality"]
    ax.text(0.5, 0.25, f"{n['intraoral']:,} photos, {n['opg']:,} OPGs,", ha="center", va="center", fontsize=FS)
    ax.text(0.5, 0.19, f"{n['cephalogram']:,} cephalograms", ha="center", va="center", fontsize=FS)
    ax.text(0.5, 0.115, f"{ds['n_patients']} patients", ha="center", va="center", fontsize=FS, fontweight="bold")
    ax.text(0.5, 0.055, f"{ds['fully_paired_patients']} with all three", ha="center", va="center", fontsize=FS)

    # (2) Angle class target: three class tiles with molar schematics + distribution bar
    ax = panel(fig, xs[1], Y0, W, H, "Angle class target", 2, FH)
    subtitle(ax, "sagittal molar relation, per patient")
    lab = ds["patient_label_distribution"]
    names = ("Class I", "Class II", "Class III")
    shifts = (0.0, 0.30, -0.30)                            # lower molar distal (II) / mesial (III)
    TW, TX = 0.285, [0.05, 0.3575, 0.665]
    for j, (name, shift) in enumerate(zip(names, shifts)):
        x0 = TX[j]
        tile(ax, x0, 0.36, TW, 0.445)
        ax.text(x0 + TW / 2, 0.755, name, ha="center", va="center", fontsize=FS, fontweight="bold", color=CLS_COLOR[j])
        mw, mh = 0.15, 0.075
        mx = x0 + (TW - mw) / 2
        molar(ax, mx, 0.585, mw, mh, "#C8C8C8", up=False)
        molar(ax, mx + shift * mw, 0.46, mw, mh, CLS_COLOR[j], up=True)
        ax.plot([mx + mw * 0.26] * 2, [0.43, 0.69], color="#666666", lw=0.5, ls=(0, (2, 1.5)), zorder=4)
        ax.text(x0 + TW / 2, 0.395, f"{lab[name]}", ha="center", va="center", fontsize=FS, fontweight="bold")
    tot = sum(lab[nm] for nm in names)
    left = 0.05
    for j, nm in enumerate(names):
        w = 0.90 * lab[nm] / tot
        ax.add_patch(Rectangle((left, 0.215), w, 0.055, facecolor=CLS_COLOR[j], edgecolor="white", lw=0.6))
        left += w
    ax.text(0.5, 0.165, "share of patients (source labels)", ha="center", va="center", fontsize=FS, color=SUB)
    ax.text(0.5, 0.075, f"imbalance I : III = {ds['imbalance_I_to_III']:.2f}", ha="center", va="center", fontsize=FS)

    # (3) expert audit: reading flow + transition matrix + stat chips
    ax = panel(fig, xs[2], Y0, W, H, "Expert audit", 3, FH)
    subtitle(ax, f"{pr['reviewer_agreement']['n']} primary patients re-read")
    for x0, t in ((0.06, "Reviewer A"), (0.53, "Reviewer B")):
        tile(ax, x0, 0.715, 0.41, 0.085, edge=SKY, lw=0.9)
        ax.text(x0 + 0.205, 0.7575, t, ha="center", va="center", fontsize=FS)
    tile(ax, 0.26, 0.565, 0.48, 0.085, edge=BLUE, lw=0.9)
    ax.text(0.50, 0.6075, f"Adjudicator ({pr['senior_n']})", ha="center", va="center", fontsize=FS)
    for x0 in (0.265, 0.735):
        ax.annotate("", xy=(0.50 + (x0 - 0.5) * 0.35, 0.655), xytext=(x0, 0.712),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555555", mutation_scale=6))
    m = np.array(pr["manifest_vs_final_matrix"])[:, :3]
    ax.text(0.33, 0.505, "label changes", ha="center", va="center", fontsize=FS, color=SUB)
    hx = sub(2, 0.17, 0.13, 0.36, 0.34)
    hx.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
    for i in range(3):
        for j in range(3):
            hx.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=FS,
                    color="white" if m[i, j] > m.max() * 0.55 else "#222222")
    hx.set_xticks(range(3), ["I", "II", "III"], fontsize=FS); hx.set_yticks(range(3), ["I", "II", "III"], fontsize=FS)
    hx.set_xlabel("adjudicated", fontsize=FS, labelpad=1)
    hx.set_ylabel("source", fontsize=FS, labelpad=1)
    hx.tick_params(length=0, pad=1); hx.grid(False)
    for sp in hx.spines.values():
        sp.set_visible(False)
    ag = pr["reviewer_agreement"]
    for y0, big, small in ((0.335, f"{100 * ag['raw_agreement']:.1f}%", "agreement"),
                           (0.135, f"{100 * pr['changed_fraction']:.1f}%", "labels revised")):
        tile(ax, 0.585, y0, 0.37, 0.165)
        ax.text(0.77, y0 + 0.108, big, ha="center", va="center", fontsize=8.6, fontweight="bold", color=BLUE)
        ax.text(0.77, y0 + 0.045, small, ha="center", va="center", fontsize=FS, color=SUB)

    # (4) evaluation variables: two slope charts in white tiles
    ax = panel(fig, xs[3], Y0, W, H, "Evaluation variables", 4, FH)
    subtitle(ax, f"same models, {len(sz['entries'])}/{zr['n_entries']} entries")  # QA 2026-11-16: split panel plots 30
    tile(ax, 0.05, 0.215, 0.43, 0.60)
    tile(ax, 0.52, 0.215, 0.43, 0.60)
    a1 = sub(3, 0.175, 0.31, 0.27, 0.36)
    a2 = sub(3, 0.645, 0.31, 0.27, 0.36)
    for nme, v in sz["entries"].items():
        h = nme.split(" / ")[1]
        a1.plot([0, 1], [v["patient_split"], v["image_split"]], color=HEAD_COLOR[h], lw=0.55, alpha=0.8,
                marker="o", ms=1.4, mew=0)
    for mm in zr["macro_f1"]["models"]:
        h = mm["name"].split(" / ")[1]
        a2.plot([0, 1], [mm["score_ref_a"], mm["score_ref_b"]], color=HEAD_COLOR[h], lw=0.55, alpha=0.8,
                marker="o", ms=1.4, mew=0)
    a1.set_xticks([0, 1], ["pat.", "img."], fontsize=7.8)
    a2.set_xticks([0, 1], ["src", "adj."], fontsize=7.8)
    for a, t, tau in ((a1, "split unit", sz["kendall_tau_point"]), (a2, "label version", zr["macro_f1"]["kendall_tau_point"])):
        a.set_xlim(-0.3, 1.3); a.set_ylim(0.24, 0.62)
        a.set_yticks([0.3, 0.4, 0.5, 0.6]); a.grid(axis="y", alpha=0.35, lw=0.4); a.grid(axis="x", visible=False)
        a.tick_params(labelsize=7.8, pad=1, length=1.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    a2.set_yticklabels([])
    a1.set_yticklabels(["", ".4", "", ".6"])
    for x0, t, tau in ((0.265, "split unit", sz["kendall_tau_point"]), (0.735, "label version", zr["macro_f1"]["kendall_tau_point"])):
        ax.text(x0, 0.775, t, ha="center", va="center", fontsize=FS, fontweight="bold")
        ax.text(x0, 0.725, f"$\\tau_b$ = {tau:.2f}", ha="center", va="center", fontsize=FS, color=SUB)
    hs = [("knn1", "1-NN"), ("knn5", "5-NN"), ("logreg", "linear"), ("fine-tuned", "fine-tuned")]
    for k, (h, t) in enumerate(hs):
        x0, y0 = 0.10 + (k % 2) * 0.45, 0.135 - (k // 2) * 0.07
        ax.plot([x0, x0 + 0.07], [y0, y0], color=HEAD_COLOR[h], lw=1.6)
        ax.text(x0 + 0.09, y0, t, va="center", fontsize=FS)
    save(fig, out, "ex_fig1_teaser")


# ----------------------------------------------------------------- Fig 2 (record)
def fig2(out, thumbs):
    """One record and its probe reading (surgical revision 2026-11-15: the left column is a
    LUWA Fig. 2-style sample column; it now carries the record's per-record-type reading as
    100%-stacked bars instead of a text box; the grid has group headers and row labels)."""
    fp = load("fig2_frame_probs.json")
    ims = list(thumbs["intraoral"]) + [thumbs["opg"], thumbs["cephalogram"]]
    probs = fp["intraoral_frame_probs"] + fp["opg_frame_probs"] + fp["cephalogram_frame_probs"]
    heads = [f"frame {i + 1}" for i in range(5)] + ["OPG", "ceph."]
    CL = ["I", "II", "III"]
    FH = 2.2
    fig = plt.figure(figsize=(7.0, FH))
    # ---- left column: record card
    L0, LW = 0.0, 0.255
    ax = fig.add_axes([L0, 0.02, LW, 0.96]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.01), 0.96, 0.98, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, lw=0.9))
    ax.text(0.08, 0.915, "One linked record", fontsize=7.8, fontweight="bold", va="center")
    ax.text(0.08, 0.835, "5 photos, 1 OPG, 1 cephalogram", fontsize=FS, color="#555555", va="center")
    ax.add_patch(FancyBboxPatch((0.08, 0.625), 0.84, 0.15, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="white", edgecolor=GREEN, lw=0.8))
    src = fp["source_label"].replace("Class ", "")
    ax.text(0.50, 0.700, f"source label: Class {src}\n(not audited)", fontsize=FS, color=GREEN,
            ha="center", va="center", fontweight="bold", linespacing=1.15)
    ax.text(0.08, 0.545, "probe reading by record type", fontsize=FS, va="center", style="italic", color="#333333")
    rows = [("photo mean", np.mean(fp["intraoral_frame_probs"], 0)),
            ("OPG", np.asarray(fp["opg_frame_probs"][0])),
            ("ceph.", np.asarray(fp["cephalogram_frame_probs"][0]))]
    bx = fig.add_axes([L0 + LW * 0.36, 0.20, LW * 0.46, 0.27])
    for i, (nm, pv) in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0.0
        for k in range(3):
            bx.barh(y, pv[k], left=left, color=CLS_COLOR[k], height=0.62, edgecolor="white", lw=0.4)
            left += pv[k]
        k = int(np.argmax(pv))
        bx.text(1.04, y, CL[k], va="center", ha="left", fontsize=FS, fontweight="bold", color=CLS_COLOR[k])
    bx.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=FS)
    bx.set_xlim(0, 1); bx.set_xticks([0, 0.5, 1], ["0", ".5", "1"]); bx.grid(False)
    bx.tick_params(labelsize=FS, pad=1, length=1.5)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    ax.text(0.08, 0.075, "class", fontsize=FS, va="center", color="#555555")
    for k in range(3):                                     # class legend
        x = 0.30 + k * 0.21
        ax.add_patch(Rectangle((x, 0.053), 0.06, 0.045, facecolor=CLS_COLOR[k], lw=0))
        ax.text(x + 0.08, 0.075, CL[k], fontsize=FS, va="center")
    # ---- grid: images (row 1) and probe probabilities (row 2)
    X0, GAP = 0.305, 0.016
    wcol = (1.0 - X0 - GAP) / len(ims)
    xs = [X0 + j * wcol + (GAP if j >= 5 else 0) for j in range(len(ims))]
    IM_Y, IM_H, B_Y, B_H = 0.47, 0.33, 0.14, 0.25
    for j in range(len(ims)):
        a = fig.add_axes([xs[j] + 0.003, IM_Y, wcol - 0.006, IM_H])
        a.imshow(np.asarray(ims[j]), aspect="auto"); a.axis("off")
        a.set_title(heads[j], fontsize=FS, pad=2)
        b = fig.add_axes([xs[j] + 0.016, B_Y, wcol - 0.026, B_H])
        pv = probs[j]
        k = int(np.argmax(pv))
        b.bar(range(3), pv, color=CLS_COLOR, width=0.72)
        b.text(k, pv[k] + 0.04, f"{pv[k]:.2f}", ha="center", va="bottom", fontsize=FS)
        b.set_ylim(0, 1.18); b.set_xlim(-0.95, 2.6); b.set_xticks(range(3), CL, fontsize=FS)
        b.set_yticks([0, 0.5, 1.0])
        b.tick_params(labelsize=FS, pad=1, length=1.5)
        b.set_yticklabels(["0", ".5", "1"] if j == 0 else [])
        b.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)
    # group headers (LUWA-style column groups) and rotated row labels
    for (j0, j1, t) in ((0, 4, "intraoral photo series"), (5, 6, "radiographs")):
        xa, xb = xs[j0] + 0.006, xs[j1] + wcol - 0.006
        fig.lines.append(plt.Line2D([xa, xb], [0.925, 0.925], transform=fig.transFigure, color="#555555", lw=0.7))
        fig.text((xa + xb) / 2, 0.945, t, ha="center", va="bottom", fontsize=FS, fontweight="bold")
    fig.text(X0 - 0.030, IM_Y + IM_H / 2, "image", rotation=90, ha="center", va="center", fontsize=FS)
    fig.text(X0 - 0.030, B_Y + B_H / 2, "p(class)", rotation=90, ha="center", va="center", fontsize=FS)
    save(fig, out, "ex_fig2_record")


# ----------------------------------------------------------------- Fig 3 (feature distance)
def fig_samples(out, thumbs):
    """Benchmark images of one linked record (user request 2026-11-16, round 5).

    Sample-panel style (coloured group headers, dark in-image tags, captions under each tile).
    PRIVACY: uses only the redacted, OCR-screened thumbnails already published in the primer
    figure (primer_thumbnails); the output is a paper-only figure and is never part of a code
    release (the thumbnails are restricted, like ex_fig1_teaser and fig9_primer).
    """
    fp = load("fig2_frame_probs.json")
    CL = ["I", "II", "III"]
    W, H = 7.0, 1.62
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    tiles = [(im, f"frame {i + 1}", fp["intraoral_frame_probs"][i], BLUE) for i, im in enumerate(thumbs["intraoral"])]
    tiles += [(thumbs["opg"], "OPG", fp["opg_frame_probs"][0], ORANGE),
              (thumbs["cephalogram"], "ceph.", fp["cephalogram_frame_probs"][0], PURPLE)]
    gap, ggap = 0.05, 0.09
    ratio = sum(im.size[0] / im.size[1] for im, *_ in tiles)
    th = (W - 0.02 - gap * 4 - ggap * 2) / ratio      # tile height so the row fills the width
    y0 = 0.50
    x, pos = 0.01, []
    for k, (im, tag, pv, col) in enumerate(tiles):
        if k in (5, 6):
            x += ggap - gap
        w = th * im.size[0] / im.size[1]
        ax.imshow(np.asarray(im), extent=(x, x + w, y0, y0 + th), zorder=2, interpolation="lanczos")
        ax.add_patch(Rectangle((x, y0), w, th, fill=False, edgecolor=col, lw=1.0, zorder=3))
        ax.text(x + 0.03, y0 + th - 0.03, tag, fontsize=FS, color="white", va="top", ha="left", zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.6, lw=0))
        c = int(np.argmax(pv))
        ax.text(x + w / 2, y0 - 0.1, f"probe: Class {CL[c]}", fontsize=FS, ha="center", va="center",
                color=CLS_COLOR[c], fontweight="bold")
        pos.append((x, w, col))
        x += w + gap
    groups = [(0, 4, "intraoral photograph series (5 frames)"), (5, 5, "panoramic (OPG)"),
              (6, 6, "cephalogram")]
    for a, b, t in groups:
        xa, xb, col = pos[a][0], pos[b][0] + pos[b][1], pos[a][2]
        ax.add_patch(Rectangle((xa, y0 + th + 0.04), xb - xa, 0.2, color=col, lw=0, zorder=2))
        ax.text((xa + xb) / 2, y0 + th + 0.14, t, fontsize=FS, color="white", fontweight="bold",
                ha="center", va="center", zorder=3)
    src = fp["source_label"]
    ax.text(0.01, 0.2, f"One patient's linked record.  Source label: {src}.  Expert audit: not audited.  "
            "De-identified and border-redacted.", fontsize=FS, color="#222222", va="center")
    ax.text(0.01, 0.05, "probe = frozen DINOv2-S linear probe trained without this record; label colour = predicted class",
            fontsize=FS, color=SUB, va="center")
    save(fig, out, "ex_fig_samples")


def fig3(out):
    fd = load("feature_distance.json")["per_encoder"]
    encs = ["ResNet-18", "ConvNeXt-T", "DINOv2-S", "BiomedCLIP"]
    kinds = [("nearest_same_patient", "same patient", BLUE),
             ("nearest_other_patient_same_class", "other patient, same class", ORANGE),
             ("nearest_other_patient_other_class", "other patient, other class", GREEN)]
    fs3 = 7.4                                               # prints at about 0.97 x in one column
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    stats, pos, cols = [], [], []
    for i, e in enumerate(encs):
        d = fd[f"intraoral/{e}"]
        for k, (key, _, c) in enumerate(kinds):
            q = d[key]
            stats.append({"med": q["median"], "q1": q["q1"], "q3": q["q3"], "whislo": q["p05"],
                          "whishi": q["p95"], "fliers": []})
            pos.append(i * 4 + k); cols.append(c)
        ax.text(i * 4 + 1, 0.352, f"{100 * d['frac_nn_same_patient']:.0f}%", ha="center", fontsize=fs3,
                fontweight="bold", color=BLUE)
    bp = ax.bxp(stats, positions=pos, widths=0.75, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor("white"); patch.set_edgecolor(c); patch.set_linewidth(1.0)
    for k, c in enumerate(cols):
        for part in ("whiskers", "caps"):
            for ln in bp[part][2 * k:2 * k + 2]:
                ln.set_color(c); ln.set_linewidth(0.8)
        bp["medians"][k].set_color(c); bp["medians"][k].set_linewidth(1.2)
    ax.set_xticks([i * 4 + 1 for i in range(len(encs))], encs, fontsize=fs3)
    ax.tick_params(axis="y", labelsize=fs3)
    ax.set_ylabel("cosine distance to\nnearest other photo", fontsize=fs3)
    ax.set_ylim(0, 0.38)
    ax.set_title("nearest photo from the same patient", fontsize=fs3, color=BLUE, pad=3)
    for key, lab, c in kinds:
        ax.plot([], [], color=c, lw=1.4, label=lab)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=fs3, handlelength=1.3,
              columnspacing=1.0, handletextpad=0.4, labelspacing=0.3)
    ax.grid(axis="x", visible=False)
    save(fig, out, "ex_fig3_featdist")



# ----------------------------------------------------------------- Fig 4 (factor analysis)
def _boxes(ax, groups, xpos, color, width=0.18):
    data = [g for g in groups]
    bp = ax.boxplot(data, positions=xpos, widths=width, patch_artist=True, showfliers=False,
                    medianprops=dict(color=color, lw=1.1), whiskerprops=dict(color=color, lw=0.7),
                    capprops=dict(color=color, lw=0.7), boxprops=dict(edgecolor=color, facecolor="white", lw=0.9))
    rng = np.random.default_rng(0)
    for x, g in zip(xpos, data):
        ax.scatter(x + rng.uniform(-width / 3, width / 3, len(g)), g, s=3.5, color=color, alpha=0.75, lw=0, zorder=3)
    return bp


def fig4(out):
    zr, sz, fr, fl, mp = (load("zoo_reference.json"), load("split_zoo.json"), load("finetune_reference.json"),
                          load("fusion_late.json"), load("modality_paired.json"))
    models = {m["name"]: m for m in zr["macro_f1"]["models"]}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 0.8, 1.05, 0.8], "wspace": 0.10})
    heads = ["knn1", "knn5", "logreg", "fine-tuned"]
    off = {"knn1": -0.27, "knn5": -0.09, "logreg": 0.09, "fine-tuned": 0.27}
    # (a) pretraining family x adaptation, adjudicated label
    fams = ["ImageNet", "DINOv2", "Image-text"]
    ax = axes[0]
    for h in heads:
        groups, xs = [], []
        for i, f in enumerate(fams):
            if h == "fine-tuned":
                vals = []
                for nme, mm in models.items():
                    if nme.endswith("fine-tuned") and FAMILY[nme.split(" / ")[0]] == f:
                        vals += mm.get("final_per_seed", [mm["score_ref_b"]])
            else:
                vals = [mm["score_ref_b"] for nme, mm in models.items()
                        if nme.endswith("/ " + h) and FAMILY[nme.split(" / ")[0]] == f]
            if vals:
                groups.append(vals); xs.append(i + off[h])
        _boxes(ax, groups, xs, HEAD_COLOR[h], 0.15)
    ax.set_xticks(range(3), ["ImageNet\nsupervised", "DINOv2", "image-\ntext"])
    ax.set_title("(a) Pretraining and adaptation", loc="left")
    ax.set_ylabel("macro-F1")
    # (b) split unit, per-image, source labels, entries x seeds
    ax = axes[1]
    for h in ["knn1", "knn5", "logreg"]:
        g_p, g_i = [], []
        for nme, v in sz["entries"].items():
            if nme.endswith("/ " + h):
                g_p += v["patient_split_per_seed"]; g_i += v["image_split_per_seed"]
        _boxes(ax, [g_p, g_i], [0 + off[h] * 1.2, 1 + off[h] * 1.2], HEAD_COLOR[h], 0.17)
    ax.set_xticks([0, 1], ["patient\nsplit", "image\nsplit"])
    ax.set_title("(b) Split unit", loc="left")
    ax.text(0.0, 0.605, "per image", ha="center", va="top", fontsize=6.3, color="#555555")
    # (c) modality, linear head, adjudicated label, same patients
    ax = axes[2]
    pe = mp["per_encoder"]
    photo_o = [v["photo"] for k, v in pe.items() if k.startswith("opg/")]
    opg = [v["radiograph"] for k, v in pe.items() if k.startswith("opg/")]
    photo_c = [v["photo"] for k, v in pe.items() if k.startswith("cephalogram/")]
    ceph = [v["radiograph"] for k, v in pe.items() if k.startswith("cephalogram/")]
    photo_all = [models[f"{e} / logreg"]["score_ref_b"] for e in FAMILY if f"{e} / logreg" in models]
    fus = [v["final"]["late_fusion"] for v in fl["per_encoder"].values()]
    _boxes(ax, [photo_o, opg], [-0.17, 0.17], BLUE, 0.26)
    _boxes(ax, [photo_c, ceph], [0.83, 1.17], BLUE, 0.26)
    _boxes(ax, [photo_all, fus], [1.83, 2.17], BLUE, 0.26)
    rd = mp["rad_dino_radiograph"]
    ax.scatter([0.17, 1.17], [rd["opg"], rd["cephalogram"]], marker="D", s=9, color=PURPLE, zorder=4,
               label="RAD-DINO")
    ax.set_xticks([0, 1, 2], ["photo |\nOPG", "photo |\nceph", "photo |\nfused"], fontsize=6.2)
    for x, t in ((0, f"n={fl['n_test_with_opg']}"), (1, f"n={fl['n_test_with_ceph']}"), (2, f"n={zr['n_test']}")):
        ax.text(x, 0.215, t, ha="center", fontsize=6.3, color="#555555")
    ax.legend(loc="upper right", fontsize=6.3, handletextpad=0.2, borderpad=0.1)
    ax.set_title("(c) Modality (linear)", loc="left")
    # (d) label version
    ax = axes[3]
    for h in heads:
        a = [mm["score_ref_a"] for n, mm in models.items() if n.endswith("/ " + h) or (h == "fine-tuned" and n.endswith("fine-tuned"))]
        b = [mm["score_ref_b"] for n, mm in models.items() if n.endswith("/ " + h) or (h == "fine-tuned" and n.endswith("fine-tuned"))]
        _boxes(ax, [a, b], [0 + off[h] * 1.2, 1 + off[h] * 1.2], HEAD_COLOR[h], 0.15)
    ax.set_xticks([0, 1], ["source", "adjud."])
    ax.set_title("(d) Label version", loc="left")
    for a in axes:
        a.grid(axis="x", visible=False)
        a.set_ylim(0.2, 0.62)
    hs = [plt.Line2D([], [], color=HEAD_COLOR[h], lw=1.4, label=HEAD_NAME[h]) for h in heads]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=6.4)
    save(fig, out, "ex_fig4_factors")


# ----------------------------------------------------------------- Fig 5 (experts)
def fig5(out):
    la, zr = load("label_analysis.json"), load("zoo_reference.json")
    t = la["human_and_oracle"]["test_189"]
    models = zr["macro_f1"]["models"]
    best = zr["best_manifest"]
    bm = next(m for m in models if m["name"] == best["name"])
    fig = plt.figure(figsize=(7.0, 2.45))
    B, T = 0.20, 0.88
    axes = [fig.add_axes([0.150, B, 0.190, T - B]), fig.add_axes([0.435, B, 0.135, T - B]),
            fig.add_axes([0.770, B, 0.225, T - B])]
    ax = axes[0]
    rows = [("adjudicated label", t["oracle_final_vs_manifest_macro_f1"], t["oracle_final_vs_manifest_ci"], VERM),
            ("Reviewer A", t["expert1_vs_manifest_macro_f1"], t["expert1_vs_manifest_ci"], ORANGE),
            ("Reviewer B", t["expert2_vs_manifest_macro_f1"], t["expert2_vs_manifest_ci"], ORANGE),
            ("best of 33 entries", best["score"], bm["ci_ref_a"], BLUE)]
    for i, (nme, v, ci, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, color=c, height=0.62)
        ax.errorbar(v, y, xerr=[[v - ci[0]], [ci[1] - v]], color="#222222", lw=0.7, capsize=2)
        ax.text(0.76, y, f"{v:.3f}", va="center", fontsize=FS, fontweight="bold")
    xr = t["expert1_vs_expert2_macro_f1_both_assessable"]
    ax.axvline(xr, color="#555555", ls=(0, (3, 2)), lw=0.7)
    ax.text(xr - 0.015, 3.95, "reviewers vs\neach other", fontsize=FS, ha="right", va="center", color="#555555",
            linespacing=1.05)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=FS)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, 4.45)
    ax.set_xlabel("macro-F1 vs source label")
    ax.set_title("(a) Expert reference", loc="left")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    for m in models:
        h = m["name"].split(" / ")[1]
        ax.plot([0, 1], [m["acc_final_agreed"], m["acc_final_adjudicated"]], color=HEAD_COLOR[h], lw=0.55, alpha=0.6)
    cd = zr["changed_case_decomposition"]
    ax.plot([0, 1], [cd["zoo_mean_acc_final_agreed"], cd["zoo_mean_acc_final_adjudicated"]], color="black", lw=1.6,
            marker="s", ms=3.5)
    ax.set_xticks([0, 1], [f"agreed\n(n={cd['n_agreed_test']})", f"adjud.\n(n={cd['n_adjudicated_test']})"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("accuracy vs adjudicated")
    ax.set_title("(b) Where models fail", loc="left")
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    det = la["label_error_detection"]
    items = sorted(det["per_model"].items(), key=lambda kv: kv[1]["auroc"])
    items = [("ensemble (10 linear)", {"auroc": det["ensemble_auroc"], "auroc_ci": det["ensemble_auroc_ci"]})] + items[::-1]
    for i, (nme, v) in enumerate(items):
        y = len(items) - 1 - i
        h = "fine-tuned" if "fine-tuned" in nme else "logreg"
        c = "black" if i == 0 else HEAD_COLOR[h]
        ax.errorbar(v["auroc"], y, xerr=[[v["auroc"] - v["auroc_ci"][0]], [v["auroc_ci"][1] - v["auroc"]]],
                    fmt="o", ms=2.6, color=c, lw=0.7, capsize=1.5)
    ax.axvline(0.5, color=GRAY, ls=(0, (3, 2)), lw=0.7)
    labels = [n.replace(" / logreg", "").replace(" / fine-tuned", " (FT)") for n, _ in items]
    ax.set_yticks(range(len(items)), labels[::-1], fontsize=FS)
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.set_xlabel("AUROC for changed labels")
    ax.set_title("(c) Finding the changes", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, out, "ex_fig5_experts")


# ----------------------------------------------------------------- Fig 6 (protocol)
def _box(ax, x, y, w, h, text, face="white", bold=False, fs=FS):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015", facecolor=face,
                                edgecolor="#9AA9B8", lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.6, color="#555555"))


def fig6(out):
    fr, zr, sz = load("finetune_reference.json"), load("zoo_reference.json"), load("split_zoo.json")
    cd = zr["changed_case_decomposition"]
    fig = plt.figure(figsize=(7.0, 2.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for (x0, w, title) in ((0.005, 0.470, "Reference arm: one prediction, two label versions"),
                           (0.485, 0.280, "Split arm: two split units"),
                           (0.775, 0.220, "Evaluation card")):
        ax.add_patch(FancyBboxPatch((x0, 0.03), w, 0.80, boxstyle="round,pad=0,rounding_size=0.02",
                                    facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, lw=0.9))
        ax.add_patch(FancyBboxPatch((x0 + 0.01, 0.86), w - 0.02, 0.12,
                                    boxstyle="round,pad=0,rounding_size=0.02", facecolor=BLUE, edgecolor="none"))
        ax.text(x0 + 0.018, 0.92, title, color="white", fontsize=7.6, fontweight="bold", va="center")
    bw, gap = 0.104, 0.0123
    xs = [0.015 + i * (bw + gap) for i in range(4)]
    txt = [f"train\n{fr['n_train_patients']} unaudited\npatients\n{fr['n_train_images']:,} images",
           f"predict once\n{fr['n_test_patients']} audit-\nheld-out\npatients\n{fr['n_test_images']:,} images",
           f"score twice:\nsource and\nadjudicated\ndiffer on {cd['n_changed_test']}",
           f"{zr['bootstrap_replicates']:,} shared\nbootstrap\ndraws: $\\tau_b$,\npaired shifts"]
    for i, (x, t) in enumerate(zip(xs, txt)):
        _box(ax, x, 0.22, bw, 0.56, t)
        if i:
            _arrow(ax, xs[i - 1] + bw, 0.50, x, 0.50)
    ax.text(0.24, 0.115, f"perceptual-hash quarantine (Hamming $\\leq$ {zr['config']['quarantine_hamming']}) before fitting",
            ha="center", fontsize=FS, style="italic", color="#444444")
    n_img = load("dataset_stats.json")["images_by_modality"]["intraoral"]
    txt2 = [f"{sz['n_entries']} frozen entries\n{n_img:,} photos, source labels",
            f"image | patient split\ntest {sz['test_frac']}, {len(sz['seeds'])} seeds",
            "per-image macro-F1:\ninflation, $\\tau_b$"]
    ys2 = [0.585, 0.325, 0.065]
    for i, (y, t) in enumerate(zip(ys2, txt2)):
        _box(ax, 0.497, y, 0.256, 0.205, t)
        if i:
            _arrow(ax, 0.625, ys2[i - 1], 0.625, y + 0.205)
    card = ["split unit: patient", "label version: both", "training: unaudited only", "leakage screen: pHash",
            "scoring unit: patient", "CIs: paired bootstrap", "ceiling: expert reviewers",
            "release: paired predictions"]
    for k, c in enumerate(card):
        ax.text(0.783, 0.765 - k * 0.0915, "• " + c, fontsize=FS, va="center")
    save(fig, out, "ex_fig6_protocol")


# ----------------------------------------------------------------- supplementary comparison
def fig_comparison(out):
    """Fig. 2, compact single-column comparison (user request 2026-11-16).

    Layout follows a booktabs comparison table as in LUWA Tab. 1 and SiteShift Tab. 1: one row
    per dataset, name on the first line and record type and task in grey on the second, three
    check columns and an image-count bar. Facts and marks come from comparison_matrix.json
    unchanged (not reported renders as a cross, as before); only display labels are shortened.
    """
    cm = load("comparison_matrix.json")
    rows = cm["datasets"]
    MOD = {"Lateral ceph": "ceph", "Panoramic": "OPG", "Intraoral + OPG + ceph": "photos + OPG + ceph"}
    TASK = {"Landmark detection": "landmarks", "Landmark detection + CVM": "landmarks + CVM",
            "Diagnostic segmentation": "diagnostic segmentation", "Enumeration / pathosis": "enumeration, pathosis",
            "Tooth segmentation": "tooth segmentation", "Patient-level Angle class": "Angle class"}
    W, RH, HD = 3.3, 0.255, 0.34
    H = HD + RH * len(rows) + 0.06
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    xc = {"patient_linked_multimodal": 1.83, "angle_class_labels": 2.18, "patient_level_split": 2.53}
    xb, bmax = 2.74, 0.24
    top = H - 0.02
    ax.plot([0, W], [top, top], color="black", lw=0.8)
    hy = top - HD / 2
    ax.text(0.0, hy, "Dataset", fontsize=FS, fontweight="bold", va="center")
    for (k, x), t in zip(xc.items(), ("Linked", "Angle\nlabels", "Patient\nsplit")):
        ax.text(x, hy, t, fontsize=FS, fontweight="bold", va="center", ha="center", linespacing=0.95)
    ax.text(xb, hy, "Images", fontsize=FS, fontweight="bold", va="center")
    ax.plot([0, W], [top - HD, top - HD], color="black", lw=0.5)
    mx = max(r["n_images"] for r in rows)
    for i, r in enumerate(rows):
        yc = top - HD - RH * (i + 0.5)
        ours = r["source_id"] == "ours"
        if ours:
            ax.add_patch(Rectangle((0, yc - RH / 2 + 0.01), W, RH - 0.02, color=BLUE, alpha=0.08, lw=0))
        ax.text(0.0, yc + 0.055, r["name"], fontsize=FS, va="center", fontweight="bold" if ours else "normal")
        ax.text(0.0, yc - 0.06, f"{MOD[r['modalities']]} · {TASK[r['task']]}", fontsize=7.0, va="center", color=SUB)
        for k, x in xc.items():
            v = r[k]
            ax.text(x, yc, "✓" if v else "✗", fontsize=8, ha="center", va="center",
                    color=GREEN if v else VERM, fontweight="bold")
        bl = bmax * r["n_images"] / mx
        ax.add_patch(Rectangle((xb, yc - 0.045), bl, 0.09, color=BLUE if ours else "#BFBFBF", lw=0))
        ax.text(xb + bl + 0.03, yc, f"{r['n_images']:,}", fontsize=FS, va="center",
                fontweight="bold" if ours else "normal")
    ax.plot([0, W], [top - HD - RH * len(rows), top - HD - RH * len(rows)], color="black", lw=0.8)
    save(fig, out, "ex_fig_comparison")


# ----------------------------------------------------------------- supplementary map (surgical revision 2026-11-15)
def fig_map(out):
    """Where the compared dental datasets come from (SiteShift Figs. 2 and 7 style).

    Country outlines: Natural Earth 1:110m (public domain), vendored in map_data/. Dataset
    origins and their sources: dataset_origins.json (unverified origins are not drawn).
    """
    from matplotlib.patches import Polygon as MPoly
    geo = json.loads((HERE / "map_data/ne_110m_countries_min.geojson").read_text())
    org = json.loads((HERE / "dataset_origins.json").read_text())["datasets"]
    cm = {r["name"]: r for r in load("comparison_matrix.json")["datasets"]}
    LAND, EDGE, PUB, OURS = "#E4E2DD", "white", "#1F3A5F", VERM
    fig = plt.figure(figsize=(7.0, 2.2))   # round 4: legend moved to the bottom, less height
    ax = fig.add_axes([0.0, 0.0, 0.70, 1.0])
    for f in geo["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ax.add_patch(MPoly(np.asarray(poly[0]), closed=True, facecolor=LAND, edgecolor=EDGE, lw=0.35))
    ax.set_xlim(-105, 168); ax.set_ylim(-45, 66); ax.set_aspect("equal"); ax.axis("off")
    # label anchor (lon, lat, ha) per dataset: layout only, chosen to keep labels clear of each other
    pos = {"ISBI-2015": (130, 40, "left"), "CL-Detection 2023": (100, -8, "left"), "Aariz": (60, 11, "right"),
           "DENTEX": (-12, 47, "right"), "UFBA-UESC": (-28, -22, "left"), "OrthoBench (ours)": (60, 57, "left")}
    desc = {"ISBI-2015": "Taipei, Taiwan", "CL-Detection 2023": "Shenzhen, China",
            "Aariz": "Islamabad, Pakistan", "DENTEX": "three hospitals, Türkiye",
            "UFBA-UESC": "Bahia, Brazil", "OrthoBench (ours)": "Gulf region\nintraoral + OPG + ceph"}
    for r in org:
        if r["lon"] is None:
            continue
        ours = r["cite_key"] is None
        c = OURS if ours else PUB
        if ours:
            # QA 2026-11-16: halo spans the whole Gulf and its shores, not one country
            ax.add_patch(Ellipse((r["lon"], r["lat"]), 24, 15, facecolor=OURS, alpha=0.20, lw=0, zorder=2))
            ax.scatter(r["lon"], r["lat"], marker="*", s=80, color=OURS, edgecolor="white", lw=0.4, zorder=5)
        else:
            ax.scatter(r["lon"], r["lat"], s=13, color=PUB, edgecolor="white", lw=0.4, zorder=5)
        lx, ly, ha = pos[r["name"]]
        ax.plot([r["lon"], lx + (1.5 if ha == "right" else -1.5)], [r["lat"], ly + 1.5], color=c, lw=0.5, zorder=4)
        ax.text(lx, ly + 0.6, r["name"], ha=ha, va="bottom", fontsize=FS, fontweight="bold", color=c, zorder=6)
        ax.text(lx, ly, desc[r["name"]], ha=ha, va="top", fontsize=FS, color=c if ours else "#333333",
                linespacing=1.1, zorder=6)
    ax.scatter([], [], s=13, color=PUB, label="public dental dataset (Tab. 1c)")
    ax.scatter([], [], marker="*", s=70, color=OURS, label="OrthoBench (ours), Gulf region")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=FS, frameon=False, handletextpad=0.3,
              labelspacing=0.3, borderaxespad=0.2)
    # right: datasets by region and record type
    bx = fig.add_axes([0.80, 0.30, 0.185, 0.68])
    regions = ["East Asia", "South Asia", "Türkiye", "South America", "not verified", "Gulf region"]
    rlab = {"not verified": "not verified\n(" + ", ".join(r["name"] for r in org if r["lon"] is None) + ")"}
    kinds = [("Lateral ceph", "cephalograms", "#9DB4CE"), ("Panoramic", "panoramic", "#C9C6BF"),
             ("Intraoral + OPG + ceph", "linked IO + OPG + ceph", OURS)]
    for i, rg in enumerate(regions):
        y = len(regions) - 1 - i
        left = 0
        for key, _, col in kinds:
            k = sum(1 for r in org if r["region"] == rg and cm[r["name"]]["modalities"] == key)
            if k:
                bx.barh(y, k, left=left, color=col, height=0.62, edgecolor="white", lw=0.5)
                left += k
        bx.text(left + 0.08, y, str(left), va="center", fontsize=FS,
                color=OURS if rg == "Gulf region" else "#333333", fontweight="bold" if rg == "Gulf region" else "normal")
    bx.set_yticks(range(len(regions)), [rlab.get(r, r) for r in regions[::-1]], fontsize=FS, linespacing=1.0)
    for t in bx.get_yticklabels():
        if t.get_text() == "Gulf region":
            t.set_color(OURS); t.set_fontweight("bold")
    bx.set_xlim(0, 2.6); bx.set_xticks([0, 1, 2]); bx.set_xlabel("datasets in Tab. 1c", fontsize=FS, labelpad=1)
    bx.tick_params(labelsize=7.8, pad=1, length=1.5); bx.grid(axis="y", visible=False)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    hd = [Rectangle((0, 0), 1, 1, color=col) for _, _, col in kinds]
    # record-type legend at the bottom of the figure (user request 2026-11-16, round 4)
    fig.legend(hd, [t for _, t, _ in kinds], loc="lower center", bbox_to_anchor=(0.62, -0.02), ncol=3, fontsize=FS,
               frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1)
    save(fig, out, "ex_fig_map")


# ----------------------------------------------------------------- tables
def f3(x):
    return f"{x:.3f}"


def tab1(gen):
    """Tab. 1 (round 5, user request 2026-11-16): the earlier combined layout. Left: (a) cohort by
    record type, (b) primary expert audit, (c) comparison with dental datasets (marks and verified
    facts from comparison_matrix.json; not reported renders as a cross). Right: (d) patients per
    class, split and label version, (e) reference split by record type."""
    ds, ra, cm = load("dataset_stats.json"), load("reference_audit.json"), load("comparison_matrix.json")
    ts = load("dataset_table_stats.json")
    pr = ra["primary"]; ag = pr["reviewer_agreement"]
    mods = (("intraoral", "Photographs"), ("opg", "OPG"), ("cephalogram", "Cephalogram"))
    L = [r"\begin{tabular}[t]{@{}lcccc@{}}\toprule",
         r"\multicolumn{5}{@{}l}{\textbf{(a) Cohort by record type}}\\",
         r"Record type & Patients & Images & Images/patient & Median resolution\\\midrule",
         *[f"{nm} & {ts['per_modality'][m]['patients']} & {ts['per_modality'][m]['images']:,} & "
           f"{ds['images_per_patient'][m][0]:.0f} ({ds['images_per_patient'][m][1]:.0f}--{ds['images_per_patient'][m][2]:.0f}), "
           f"{ds['images_per_patient'][m][3]}--{ds['images_per_patient'][m][4]} & "
           f"${ds['resolution'][m]['median_wh'][0]}\\times{ds['resolution'][m]['median_wh'][1]}$\\\\"
           for m, nm in mods],
         f"All (linked: {ds['fully_paired_patients']}) & {ds['n_patients']} & {ds['n_images']:,} & & \\\\",
         r"\midrule\multicolumn{5}{@{}l}{\textbf{(b) Expert audit (primary batch)}}\\",
         r"Patients & Agreement & $\kappa$ ($95\%$ CI) & Adjudicated / excl. & Revised\\\midrule",
         f"{ag['n']} & ${100 * ag['raw_agreement']:.1f}\\%$ & {ag['kappa']:.3f} ({ag['kappa_ci'][0]:.3f}--{ag['kappa_ci'][1]:.3f}) & "
         f"{pr['senior_n']} / {pr['n_excluded']} & {pr['changed_labels']}/{pr['n_assessable']} (${100 * pr['changed_fraction']:.1f}\\%$)\\\\",
         r"\midrule\multicolumn{5}{@{}l}{\textbf{(c) Comparison with dental datasets}}\\",
         r"Dataset & Images & Linked & Angle labels & Patient split\\\midrule"]
    for r in cm["datasets"]:
        nm = r"\textbf{OrthoBench (ours)}" if r["source_id"] == "ours" else f"{r['name']}~\\cite{{{r['cite_key']}}}"
        mk = lambda v: r"\cmark" if v else r"\xmark"  # noqa: E731
        L.append(f"{nm} & {r['n_images']:,} & {mk(r['patient_linked_multimodal'])} & {mk(r['angle_class_labels'])} & "
                 f"{mk(r['patient_level_split'])}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (gen / "ex_tab1.tex").write_text("\n".join(L) + "\n")
    tab1_right(gen, ts, pr)


def tab1_right(gen, ts, pr):
    """Tab. 1 (d) patients per class, split and label version, (e) reference split by record type."""
    cls = ("Class I", "Class II", "Class III")
    rs = ts["reference_split"]
    c = lambda v: r"$<$5" if v is None else f"{v:,}"  # noqa: E731  (small-cell suppression)
    fin = {k.replace("Excluded (not assessable)", "excl"): v for k, v in pr["final_labels"].items()}

    def row(name, d, tot):
        return f"{name} & " + " & ".join(c(d[k]) for k in cls) + f" & {tot}\\\\"
    L = [r"\begin{tabular}[t]{@{}lcccc@{}}\toprule",
         r"\multicolumn{5}{@{}l}{\textbf{(d) Patients per class, split and label version}}\\",
         r"Set (label version) & Class~I & Class~II & Class~III & Total\\\midrule",
         row("All patients (source)", {k: ts["per_class"][k]["patients"] for k in cls}, ts["n_patients"]),
         row("All images (source)", {k: ts["per_class"][k]["images"] for k in cls}, f"{ts['n_images']:,}"),
         row("Train (source)", rs["train_by_source_class"], rs["train_patients"]),
         row("Test (source)", rs["test_by_source_class"], rs["test_patients"]),
         row("Test (adjudicated)", rs["test_by_adjudicated_class"], rs["test_patients"]),
         row("Not used$^\\dagger$ (source)", rs["unused_by_source_class"], rs["unused_patients"]),
         row("Primary audit (source)", pr["manifest_labels"], pr["n"]),
         row("Primary audit (adjudicated)", fin, f"{pr['n_assessable']} + {fin['excl']} excl."),
         r"\midrule\multicolumn{5}{@{}l}{\textbf{(e) Reference split by record type}}\\",
         r"Record type & Train pat. & Test pat. & Train img. & Test img.\\\midrule"]
    for m, nm in (("intraoral", "Photographs"), ("opg", "OPG"), ("cephalogram", "Cephalogram")):
        v = ts["per_modality"][m]
        L.append(f"{nm} & {v['train_patients']} & {v['test_patients']} & {v['train_images']:,} & {v['test_images']:,}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (gen / "ex_tab1_right.tex").write_text("\n".join(L) + "\n")


def tab2(gen):
    """Leaderboard without empty cells (surgical revision 2026-11-15).

    (a) is the full grid of the ten photograph encoders: every head in both arms, plus the
    radiograph tracks. Entries that exist for only some settings get their own blocks:
    (b) the three fine-tuned backbones (reference arm only), (c) the radiograph-only encoder
    RAD-DINO, (d) the expert reference, scored against the source label only, because the
    reviewers' readings define the adjudicated label.
    """
    zr, sz, fl, la = load("zoo_reference.json"), load("split_zoo.json"), load("fusion_late.json"), load("label_analysis.json")
    M = {m["name"]: m for m in zr["macro_f1"]["models"]}
    S = sz["entries"]
    R = fl["radiograph_tracks"]
    order = [["ResNet-18", "ResNet-50"], ["ConvNeXt-T", "ConvNeXt-L"], ["DINOv2-S", "DINOv2-B", "DINOv2-L"],
             ["CLIP-B/16", "SigLIP-2", "BiomedCLIP"]]

    def val(enc, col):
        kind, h, ref = col
        if kind == "ref":
            return M[f"{enc} / {h}"]["score_ref_a" if ref == "src" else "score_ref_b"]
        if kind == "split":
            return S[f"{enc} / {h}"]["patient_split" if ref == "pat" else "image_split"]
        return R[f"{h}/{enc}"]["final_macro_f1"]
    cols = [("ref", "logreg", "src"), ("ref", "logreg", "adj"), ("ref", "knn1", "src"), ("ref", "knn1", "adj"),
            ("ref", "knn5", "src"), ("ref", "knn5", "adj"),
            ("split", "logreg", "pat"), ("split", "logreg", "img"), ("split", "knn1", "pat"), ("split", "knn1", "img"),
            ("split", "knn5", "pat"), ("split", "knn5", "img"),
            ("rad", "opg", ""), ("rad", "cephalogram", "")]
    allenc = [e for g in order for e in g]
    best = {c: max(val(e, c) for e in allenc) for c in cols}

    def cell(v, b):
        s = f3(v)
        return f"\\textbf{{{s}}}" if b else s
    L = [r"\begin{tabular}{@{}l*{6}{c}@{\hspace{7pt}}*{6}{c}@{\hspace{7pt}}cc@{}}\toprule",
         r"\textbf{(a) Frozen} & \multicolumn{6}{c}{Reference arm: patient macro-F1, $189$ patients} & "
         r"\multicolumn{6}{c}{Split arm: per-image macro-F1, source label} & \multicolumn{2}{c}{Radiographs (adj.)}\\",
         r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}\cmidrule(l){14-15}",
         r" & \multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{$1$-NN} & \multicolumn{2}{c}{$5$-NN} & "
         r"\multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{$1$-NN} & \multicolumn{2}{c}{$5$-NN} & OPG & Ceph.\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
         r"Encoder & src & adj & src & adj & src & adj & pat. & img. & pat. & img. & pat. & img. & "
         f"($n={fl['n_test_with_opg']}$) & ($n={fl['n_test_with_ceph']}$)\\\\\\midrule"]
    for gi, g in enumerate(order):
        for e in g:
            # QA 2026-11-16: bold every entry that ties the best at the printed precision
            L.append(f"{e} & " + " & ".join(cell(val(e, c), f3(val(e, c)) == f3(best[c])) for c in cols) + r"\\")
        if gi < len(order) - 1:
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}")
    (gen / "ex_tab2.tex").write_text("\n".join(L) + "\n")

    # (b) fine-tuned backbones, reference arm
    ft = [("DINOv2-S", "DINOv2-S"), ("ConvNeXt-T", "ConvNeXt-T"), ("ResNet-50", "ResNet-50")]
    fb = {k: max(M[f"{e} / fine-tuned"][k] for e, _ in ft) for k in ("score_ref_a", "score_ref_b")}
    B = [r"\begin{tabular}[t]{@{}lcc@{}}\toprule",
         r"\multicolumn{3}{@{}l}{\textbf{(b) Fine-tuned, reference arm}}\\",
         r"Backbone & src & adj\\\midrule"]
    for e, nm in ft:
        m = M[f"{e} / fine-tuned"]
        B.append(f"{nm} & {cell(m['score_ref_a'], m['score_ref_a'] == fb['score_ref_a'])} & "
                 f"{cell(m['score_ref_b'], m['score_ref_b'] == fb['score_ref_b'])}\\\\")
    B.append(r"\bottomrule\end{tabular}")
    (gen / "ex_tab2b.tex").write_text("\n".join(B) + "\n")

    # (c) radiograph-only encoder, linear head
    C = [r"\begin{tabular}[t]{@{}lccc@{}}\toprule",
         r"\multicolumn{4}{@{}l}{\textbf{(c) RAD-DINO, linear head}}\\",
         r"Record & $n$ & src & adj\\\midrule"]
    for k, nm in (("opg", "OPG"), ("cephalogram", "Ceph.")):
        r = R[f"{k}/RAD-DINO"]
        C.append(f"{nm} & {r['n_test']} & {f3(r['manifest_macro_f1'])} & {f3(r['final_macro_f1'])}\\\\")
    C.append(r"\bottomrule\end{tabular}")
    (gen / "ex_tab2c.tex").write_text("\n".join(C) + "\n")

    # (d) expert reference, source label only
    t = la["human_and_oracle"]["test_189"]
    D = [r"\begin{tabular}[t]{@{}lc@{}}\toprule",
         r"\multicolumn{2}{@{}l}{\textbf{(d) Expert reference, $189$ patients}}\\",
         r"Reader & src ($95\%$ CI)\\\midrule"]
    for nme, v, ci in (("Reviewer A", t["expert1_vs_manifest_macro_f1"], t["expert1_vs_manifest_ci"]),
                       ("Reviewer B", t["expert2_vs_manifest_macro_f1"], t["expert2_vs_manifest_ci"]),
                       ("Adjudicated label", t["oracle_final_vs_manifest_macro_f1"], t["oracle_final_vs_manifest_ci"])):
        D.append(f"{nme} & {f3(v)} ({f3(ci[0])}--{f3(ci[1])})\\\\")
    D.append(r"\bottomrule\end{tabular}")
    (gen / "ex_tab2d.tex").write_text("\n".join(D) + "\n")


def postplan_tables(gen):
    """Post-plan entries (plan amendment 2026-11-16): Tab. 2 (e) and the appendix detail table."""
    pp = RES / "postplan_baselines.json"
    if not pp.exists():
        print("[skip] postplan_baselines.json missing"); return
    pp = json.loads(pp.read_text())
    order = [k for k in ("SAM ViT-B / logreg", "CLIP-B/16 / zero-shot", "BiomedCLIP / zero-shot") if k in pp["entries"]]
    nm = lambda k: k.replace(" / logreg", " linear").replace(" / zero-shot", " zero-shot")  # noqa: E731
    E = [r"\begin{tabular}[t]{@{}lcc@{}}\toprule",
         r"\multicolumn{3}{@{}l}{\textbf{(e) Post-plan entries}}\\",
         r"Entry & src & adj\\\midrule"]
    for k in order:
        v = pp["entries"][k]
        E.append(f"{nm(k)} & {f3(v['source']['macro_f1'])} & {f3(v['adjudicated']['macro_f1'])}\\\\")
    E.append(r"\bottomrule\end{tabular}")
    (gen / "ex_tab2e.tex").write_text("\n".join(E) + "\n")
    S = [r"\begin{tabular}{@{}lcccccc@{}}\toprule",
         r" & \multicolumn{3}{c}{Source label} & \multicolumn{3}{c}{Adjudicated label}\\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Entry & macro-F1 ($95\%$ CI) & bal.\ acc. & leader $-$ entry ($95\%$ CI) & macro-F1 ($95\%$ CI) & bal.\ acc. & leader $-$ entry ($95\%$ CI)\\\midrule"]
    for k in order:
        v = pp["entries"][k]
        cells = []
        for r in ("source", "adjudicated"):
            x = v[r]
            cells += [f"{f3(x['macro_f1'])} ({f3(x['macro_f1_ci'][0])}--{f3(x['macro_f1_ci'][1])})", f3(x["bacc"]),
                      f"{f3(x['leader_minus_entry'])} ({f3(x['leader_minus_entry_ci'][0])}--{f3(x['leader_minus_entry_ci'][1])})"]
        S.append(f"{nm(k)} & " + " & ".join(cells) + "\\\\")
    S.append(r"\bottomrule\end{tabular}")
    (gen / "ex_supp_postplan.tex").write_text("\n".join(S) + "\n")

def supp_cohort_class(gen):
    """Appendix B.2: patients and images per source class and record type (aggregate, small cells suppressed)."""
    ts = load("dataset_table_stats.json")
    cls = ("Class I", "Class II", "Class III")
    c = lambda v: r"$<$5" if v is None else f"{v:,}"  # noqa: E731
    L = [r"\begin{tabular}{@{}lcccccc@{}}\toprule",
         r" & \multicolumn{3}{c}{Patients (source label)} & \multicolumn{3}{c}{Images (source label)}\\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Record type & Class~I & Class~II & Class~III & Class~I & Class~II & Class~III\\\midrule"]
    for m, nm in (("intraoral", "Photographs"), ("opg", "OPG"), ("cephalogram", "Cephalogram")):
        v = ts["per_modality"][m]
        L.append(f"{nm} & " + " & ".join(c(v["patients_by_source_class"][k]) for k in cls) + " & "
                 + " & ".join(c(v["images_by_source_class"][k]) for k in cls) + "\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (gen / "ex_supp_cohort_class.tex").write_text("\n".join(L) + "\n")

def supp_tables(gen):
    rs, es, fd = load("rank_stability.json"), load("expert_subset.json"), load("feature_distance.json")
    L = [r"\begin{tabular}{@{}lccc@{}}\toprule",
         r"Entry & $P$(rank 1), source & $P$(rank 1), adjudicated & $P$(rank 1), both\\\midrule"]
    top = sorted(rs["per_entry"].items(), key=lambda kv: -(kv[1]["p_rank1_source"] + kv[1]["p_rank1_adjudicated"]))
    for n, v in top:
        if v["p_rank1_source"] + v["p_rank1_adjudicated"] == 0:
            continue
        L.append(f"{n.replace(' / logreg', ' (linear)').replace(' / fine-tuned', ' (fine-tuned)').replace(' / knn1', ' (1-NN)').replace(' / knn5', ' (5-NN)')}"
                 f" & {v['p_rank1_source']:.3f} & {v['p_rank1_adjudicated']:.3f} & {v['p_rank1_both']:.3f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (gen / "ex_supp_rank.tex").write_text("\n".join(L) + "\n")
    L = [r"\begin{tabular}{@{}lcccccc@{}}\toprule",
         r" & \multicolumn{3}{c}{All $189$ (not assessable = error)} & \multicolumn{3}{c}{Reviewer's assessable patients}\\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Reviewer & reviewer & DINOv2-S (FT) & gap & $n$ & reviewer & DINOv2-S (FT) \& gap\\\midrule"]
    L = L[:3] + [r"Reviewer & reviewer & DINOv2-S (FT) & gap & reviewer & DINOv2-S (FT) & gap\\\midrule"]
    for r, nm in (("expert1", "A"), ("expert2", "B")):
        v = es[r]
        L.append(f"{nm} ($n$={es['n_test']} / {v['n_assessable']}) & {v['expert_macro_f1_all_na_as_error']:.3f} & {v['dinov2s_ft_macro_f1_all']:.3f} & "
                 f"{v['gap_expert_minus_model_all']:.3f} & {v['expert_macro_f1_assessable_only']:.3f} & "
                 f"{v['dinov2s_ft_macro_f1_same_assessable_subset']:.3f} & {v['gap_expert_minus_model_assessable']:.3f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (gen / "ex_supp_expert.tex").write_text("\n".join(L) + "\n")
    L = [r"\begin{tabular}{@{}llccccc@{}}\toprule",
         r"Record & Encoder & Images & NN same patient & Chance & \multicolumn{2}{c}{Median distance: same class / other class}\\\midrule"]
    for k, v in fd["per_encoder"].items():
        m, e = k.split("/", 1)
        L.append(f"{ {'intraoral': 'Photo', 'opg': 'OPG', 'cephalogram': 'Ceph.'}[m]} & {e} & {v['n_images']:,} & "
                 f"${100 * v['frac_nn_same_patient']:.1f}\\%$ & ${100 * v['chance_frac_nn_same_patient']:.2g}\\%$ & "
                 f"{v['nearest_other_patient_same_class']['median']:.3f} & {v['nearest_other_patient_other_class']['median']:.3f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (gen / "ex_supp_featdist.tex").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(HERE.parents[2] / "paper"))
    a = ap.parse_args()
    paper = Path(a.paper)
    out, gen = paper / "figures", paper / "sections/generated"
    out.mkdir(parents=True, exist_ok=True); gen.mkdir(parents=True, exist_ok=True)
    setup()
    primer = paper / "figures/fig9_primer.pdf"
    if primer.exists():   # restricted: the redacted thumbnails exist only in the authors' copy
        thumbs = primer_thumbnails(primer)
        fig1(out, thumbs); fig2(out, thumbs); fig_samples(out, thumbs)
    else:
        print("[skip] Figs 1-2 need the redacted record thumbnails (not released); rendering the rest")
    fig3(out); fig4(out); fig5(out); fig6(out); fig_comparison(out); fig_map(out)
    tab1(gen); tab2(gen); supp_tables(gen); postplan_tables(gen); supp_cohort_class(gen)


if __name__ == "__main__":
    main()
