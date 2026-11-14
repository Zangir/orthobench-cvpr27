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
def fig1(out, thumbs):
    ds, ra, sz, zr = load("dataset_stats.json"), load("reference_audit.json"), load("split_zoo.json"), load("zoo_reference.json")
    pr = ra["primary"]
    FH = 2.6
    fig = plt.figure(figsize=(7.0, FH))
    W, G = 0.243, 0.0093
    xs = [G * 0.5 + i * (W + G) for i in range(4)]
    Y0, H = 0.02, 0.96

    def sub(i, u0, v0, du, dv):                            # panel coords -> figure axes
        return fig.add_axes([xs[i] + W * u0, Y0 + H * v0, W * du, H * dv])
    # (1) linked records
    ax = panel(fig, xs[0], Y0, W, H, "Linked clinical records", 1, FH)
    for j, im in enumerate(thumbs["intraoral"]):
        x0 = 0.05 + j * 0.182
        ax.imshow(np.asarray(im), extent=(x0, x0 + 0.17, 0.635, 0.795), aspect="auto", zorder=2)
    ax.text(0.5, 0.825, "intraoral photo series", ha="center", fontsize=FS, color="#333333")
    ax.imshow(np.asarray(thumbs["opg"]), extent=(0.07, 0.47, 0.335, 0.53), aspect="auto", zorder=2)
    ax.imshow(np.asarray(thumbs["cephalogram"]), extent=(0.57, 0.91, 0.335, 0.53), aspect="auto", zorder=2)
    ax.text(0.27, 0.555, "panoramic (OPG)", ha="center", fontsize=FS, color="#333333")
    ax.text(0.74, 0.555, "cephalogram", ha="center", fontsize=FS, color="#333333")
    n = ds["images_by_modality"]
    ax.text(0.5, 0.25, f"{n['intraoral']:,} photos, {n['opg']:,} OPGs,", ha="center", va="center", fontsize=FS)
    ax.text(0.5, 0.19, f"{n['cephalogram']:,} cephalograms", ha="center", va="center", fontsize=FS)
    ax.text(0.5, 0.115, f"{ds['n_patients']} patients", ha="center", va="center", fontsize=FS, fontweight="bold")
    ax.text(0.5, 0.055, f"{ds['fully_paired_patients']} with all three", ha="center", va="center", fontsize=FS)
    # (2) Angle class target
    ax = panel(fig, xs[1], Y0, W, H, "Angle class target", 2, FH)
    lab = ds["patient_label_distribution"]
    for j, (name, shift) in enumerate((("Class I", 0.0), ("Class II", 0.3), ("Class III", -0.3))):
        cx = 0.06 + j * 0.31
        molar(ax, cx + 0.06, 0.58, 0.16, 0.08, "#BFBFBF", up=False)
        molar(ax, cx + 0.06 + shift * 0.16, 0.43, 0.16, 0.08, CLS_COLOR[j], up=True)
        ax.plot([cx + 0.06, cx + 0.06], [0.39, 0.72], color="#888888", lw=0.5, ls=(0, (2, 2)))
        ax.text(cx + 0.14, 0.78, name, ha="center", fontsize=FS, fontweight="bold", color=CLS_COLOR[j])
        ax.text(cx + 0.14, 0.30, f"{lab[name]}", ha="center", fontsize=FS, fontweight="bold")
    ax.text(0.5, 0.20, "patients (source labels)", ha="center", fontsize=FS)
    ax.text(0.5, 0.08, "sagittal molar relation", ha="center", fontsize=FS, style="italic", color="#444444")
    # (3) expert audit
    ax = panel(fig, xs[2], Y0, W, H, "Expert audit", 3, FH)
    for y, t in ((0.76, "Reviewer A"), (0.62, "Reviewer B")):
        ax.add_patch(Ellipse((0.075, y), 0.06, 0.06 * W * 7.0 / (H * FH), facecolor=SKY, edgecolor="#333333", lw=0.5))
        ax.text(0.125, y, t, va="center", fontsize=FS)
    ax.add_patch(Ellipse((0.075, 0.40), 0.06, 0.06 * W * 7.0 / (H * FH), facecolor=BLUE, edgecolor="#333333", lw=0.5))
    ax.text(0.125, 0.40, "Adjudicator", va="center", fontsize=FS)
    ax.annotate("", xy=(0.075, 0.44), xytext=(0.075, 0.58), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#555555"))
    m = np.array(pr["manifest_vs_final_matrix"])[:, :3]
    hx = sub(2, 0.56, 0.42, 0.34, 0.34)
    hx.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
    for i in range(3):
        for j in range(3):
            hx.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=FS,
                    color="white" if m[i, j] > m.max() * 0.55 else "#222222")
    hx.set_xticks(range(3), ["I", "II", "III"], fontsize=FS); hx.set_yticks(range(3), ["I", "II", "III"], fontsize=FS)
    hx.set_xlabel("adjudicated", fontsize=FS, labelpad=1)
    hx.yaxis.set_label_position("right"); hx.set_ylabel("source", fontsize=FS, labelpad=2, rotation=270, va="bottom")
    hx.tick_params(length=0, pad=1); hx.grid(False)
    for s in hx.spines.values():
        s.set_visible(False)
    ag = pr["reviewer_agreement"]
    ax.text(0.5, 0.215, f"{100 * ag['raw_agreement']:.1f}% agreement", ha="center", fontsize=FS)
    ax.text(0.5, 0.150, f"({ag['n']} primary patients)", ha="center", fontsize=FS)
    ax.text(0.5, 0.060, f"{100 * pr['changed_fraction']:.1f}% of labels revised", ha="center",
            fontsize=FS, fontweight="bold")
    # (4) evaluation variables
    ax = panel(fig, xs[3], Y0, W, H, "Evaluation variables", 4, FH)
    e = sz["entries"]
    a1 = sub(3, 0.20, 0.36, 0.33, 0.37)
    a2 = sub(3, 0.62, 0.36, 0.33, 0.37)
    for nme, v in e.items():
        h = nme.split(" / ")[1]
        a1.plot([0, 1], [v["patient_split"], v["image_split"]], color=HEAD_COLOR[h], lw=0.5, alpha=0.75)
    for mm in zr["macro_f1"]["models"]:
        h = mm["name"].split(" / ")[1]
        a2.plot([0, 1], [mm["score_ref_a"], mm["score_ref_b"]], color=HEAD_COLOR[h], lw=0.5, alpha=0.75)
    a1.set_xticks([0, 1], ["patient", "image"], fontsize=FS)
    a2.set_xticks([0, 1], ["source", "adjud."], fontsize=FS)
    a1.set_title(f"split\n$\\tau_b$ = {sz['kendall_tau_point']:.2f}", fontsize=FS, pad=2, linespacing=1.1)
    a2.set_title(f"label\n$\\tau_b$ = {zr['macro_f1']['kendall_tau_point']:.2f}", fontsize=FS, pad=2, linespacing=1.1)
    for a in (a1, a2):
        a.set_xlim(-0.35, 1.35); a.set_ylim(0.24, 0.62); a.grid(False)
        a.set_yticks([0.3, 0.4, 0.5, 0.6])
        a.tick_params(labelsize=FS, pad=1, length=2)
    a2.set_yticklabels([])
    ax.text(0.5, 0.215, f"{zr['n_entries']} entries, same models", ha="center", fontsize=FS)
    hs = [("knn1", "1-NN"), ("knn5", "5-NN"), ("logreg", "linear"), ("fine-tuned", "fine-tuned")]
    for k, (h, t) in enumerate(hs):
        x0, y0 = 0.10 + (k % 2) * 0.43, 0.135 - (k // 2) * 0.07
        ax.plot([x0, x0 + 0.07], [y0, y0], color=HEAD_COLOR[h], lw=1.6)
        ax.text(x0 + 0.09, y0, t, va="center", fontsize=FS)
    save(fig, out, "ex_fig1_teaser")


# ----------------------------------------------------------------- Fig 2 (record)
def fig2(out, thumbs):
    fp = load("fig2_frame_probs.json")
    ims = list(thumbs["intraoral"]) + [thumbs["opg"], thumbs["cephalogram"]]
    probs = fp["intraoral_frame_probs"] + fp["opg_frame_probs"] + fp["cephalogram_frame_probs"]
    heads = [f"frame {i + 1}" for i in range(5)] + ["OPG", "cephalogram"]
    fig = plt.figure(figsize=(7.0, 2.2))
    # left: record schematic
    ax = fig.add_axes([0.0, 0.05, 0.135, 0.90]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, lw=0.9))
    ax.add_patch(Ellipse((0.5, 0.83), 0.30, 0.11, facecolor=SKY, edgecolor="#333333", lw=0.6))
    ax.text(0.5, 0.68, "one patient", ha="center", fontsize=FS, fontweight="bold")
    for k, t in enumerate(("5 photos", "1 OPG", "1 cephalogram")):
        ax.text(0.5, 0.56 - k * 0.095, t, ha="center", fontsize=FS)
    ax.text(0.5, 0.17, "source label:\nClass I\n(not audited)", ha="center", va="center", fontsize=FS,
            color=GREEN, fontweight="bold", linespacing=1.15)
    n = len(ims)
    x0, wcol = 0.17, 0.83 / n
    for j in range(n):
        a = fig.add_axes([x0 + j * wcol + 0.004, 0.53, wcol - 0.008, 0.36])
        a.imshow(np.asarray(ims[j]), aspect="auto"); a.axis("off")
        a.set_title(heads[j], fontsize=FS, pad=2)
        b = fig.add_axes([x0 + j * wcol + 0.016, 0.17, wcol - 0.028, 0.29])
        p = probs[j]
        b.bar(range(3), p, color=CLS_COLOR, width=0.75)
        b.set_ylim(0, 1); b.set_xticks(range(3), ["I", "II", "III"], fontsize=FS)
        b.set_yticks([0, 0.5, 1.0])
        b.tick_params(labelsize=FS, pad=1, length=1.5)
        if j == 0:
            b.set_yticklabels(["0", ".5", "1"])
            b.set_ylabel("p(class)", fontsize=FS, labelpad=1)
        else:
            b.set_yticklabels([])
        b.grid(axis="y", alpha=0.3)
    mean_io = np.mean(fp["intraoral_frame_probs"], 0)
    fig.text(x0 + 2.5 * wcol, 0.012,
             f"photo-series mean: Class {['I', 'II', 'III'][int(mean_io.argmax())]}"
             f" (p = {mean_io.max():.2f})", ha="center", fontsize=FS, color=CLS_COLOR[int(mean_io.argmax())],
             fontweight="bold")
    save(fig, out, "ex_fig2_record")


# ----------------------------------------------------------------- Fig 3 (feature distance)
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
    cm = load("comparison_matrix.json")
    rows = cm["datasets"]
    fig, ax = plt.subplots(figsize=(7.0, 0.30 * len(rows) + 0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, len(rows) + 1); ax.axis("off")
    cols = [("Dataset", 0.0), ("Modalities", 0.19), ("Task", 0.395), ("Linked\nIO+OPG+ceph", 0.645),
            ("Angle\nlabels", 0.745), ("Patient split\ndocumented", 0.83), ("Images", 0.905)]
    for t, x in cols:
        ax.text(x, len(rows) + 0.45, t, fontsize=6.2, fontweight="bold", va="center",
                ha="left" if x < 0.5 or x > 0.85 else "center")
    mx = max(r["n_images"] for r in rows)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i + 0.5
        ours = r["source_id"] == "ours"
        if ours:
            ax.add_patch(Rectangle((-0.005, y - 0.4), 1.015, 0.8, color=BLUE, alpha=0.08, lw=0))
        fw = "bold" if ours else "normal"
        ax.text(0.0, y, r["name"], fontsize=6.6, va="center", fontweight=fw)
        ax.text(0.19, y, r["modalities"], fontsize=6.3, va="center")
        ax.text(0.395, y, r["task"], fontsize=6.3, va="center")
        for x, key in ((0.645, "patient_linked_multimodal"), (0.745, "angle_class_labels"), (0.83, "patient_level_split")):
            v = r[key]
            s, c = ("✓", GREEN) if v else (("✗", VERM) if v is False else ("n/r", GRAY))
            ax.text(x, y, s, fontsize=8 if s != "n/r" else 6.2, ha="center", va="center", color=c, fontweight="bold")
        bl = 0.045 * r["n_images"] / mx
        ax.add_patch(Rectangle((0.905, y - 0.15), bl, 0.3, color=BLUE if ours else "#BFBFBF", lw=0))
        ax.text(0.905 + bl + 0.005, y, f"{r['n_images']:,}", fontsize=6.2, va="center")
    save(fig, out, "ex_fig_comparison")


# ----------------------------------------------------------------- tables
def f3(x):
    return f"{x:.3f}"


def tab1(gen):
    ds, ra, cm = load("dataset_stats.json"), load("reference_audit.json"), load("comparison_matrix.json")
    pr = ra["primary"]; ag = pr["reviewer_agreement"]
    L = [r"\begin{tabular}{@{}lcccc@{}}\toprule",
         r"\multicolumn{5}{@{}l}{\textbf{(a) Cohort}}\\",
         r"Patients & Images & Modalities & Photos/patient & Fully linked\\\midrule",
         f"{ds['n_patients']} & {ds['n_images']:,} & 3 & {ds['images_per_patient']['intraoral'][0]:.0f} (median) & {ds['fully_paired_patients']}\\\\",
         r"\midrule\multicolumn{5}{@{}l}{\textbf{(b) Expert audit (primary batch)}}\\",
         r"Patients & Agreement & $\kappa$ ($95\%$ CI) & Adjudicated / excl. & Revised\\\midrule",
         f"{ag['n']} & ${100 * ag['raw_agreement']:.1f}\\%$ & {ag['kappa']:.3f} ({ag['kappa_ci'][0]:.3f}--{ag['kappa_ci'][1]:.3f}) & "
         f"{pr['senior_n']} / {pr['n_excluded']} & {pr['changed_labels']}/{pr['n_assessable']} (${100 * pr['changed_fraction']:.1f}\\%$)\\\\",
         r"\midrule\multicolumn{5}{@{}l}{\textbf{(c) Comparison with dental datasets}}\\",
         r"Dataset & Images & Linked & Angle labels & Patient split\\\midrule"]
    for r in cm["datasets"]:
        nm = r"\textbf{OrthoBench (ours)}" if r["source_id"] == "ours" else f"{r['name']}~\\cite{{{r['cite_key']}}}"
        mk = lambda v: r"\cmark" if v else (r"\xmark" if v is False else "n/r")  # noqa: E731
        L.append(f"{nm} & {r['n_images']:,} & {mk(r['patient_linked_multimodal'])} & {mk(r['angle_class_labels'])} & "
                 f"{mk(r['patient_level_split'])}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (gen / "ex_tab1.tex").write_text("\n".join(L) + "\n")


def tab2(gen):
    zr, sz, fl, la = load("zoo_reference.json"), load("split_zoo.json"), load("fusion_late.json"), load("label_analysis.json")
    M = {m["name"]: m for m in zr["macro_f1"]["models"]}
    S = sz["entries"]
    R = fl["radiograph_tracks"]
    order = [["ResNet-18", "ResNet-50"], ["ConvNeXt-T", "ConvNeXt-L"], ["DINOv2-S", "DINOv2-B", "DINOv2-L"],
             ["RAD-DINO"], ["CLIP-B/16", "SigLIP-2", "BiomedCLIP"]]
    cols = []
    def val(enc, col):
        kind, h, ref = col
        if kind == "ref":
            m = M.get(f"{enc} / {h}")
            return None if m is None else m["score_ref_a" if ref == "src" else "score_ref_b"]
        if kind == "split":
            e = S.get(f"{enc} / {h}")
            return None if e is None else e["patient_split" if ref == "pat" else "image_split"]
        r = R.get(f"{h}/{enc}")
        return None if r is None else r["final_macro_f1"]
    cols = [("ref", "logreg", "src"), ("ref", "logreg", "adj"), ("ref", "knn1", "src"), ("ref", "knn1", "adj"),
            ("ref", "knn5", "src"), ("ref", "knn5", "adj"), ("ref", "fine-tuned", "src"), ("ref", "fine-tuned", "adj"),
            ("split", "logreg", "pat"), ("split", "logreg", "img"), ("split", "knn1", "pat"), ("split", "knn1", "img"),
            ("rad", "opg", ""), ("rad", "cephalogram", "")]
    allenc = [e for g in order for e in g]
    best = {}
    for c in cols:
        vs = [val(e, c) for e in allenc]
        vs = [v for v in vs if v is not None]
        best[c] = max(vs)
    L = [r"\begin{tabular}{@{}l*{8}{c}@{\hspace{7pt}}*{4}{c}@{\hspace{7pt}}cc@{}}\toprule",
         r" & \multicolumn{8}{c}{Reference arm: patient macro-F1, $189$ audit-held-out patients} & "
         r"\multicolumn{4}{c}{Split arm: per-image macro-F1} & \multicolumn{2}{c}{Radiographs (adj.)}\\",
         r"\cmidrule(lr){2-9}\cmidrule(lr){10-13}\cmidrule(l){14-15}",
         r" & \multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{$1$-NN} & \multicolumn{2}{c}{$5$-NN} & "
         r"\multicolumn{2}{c}{Fine-tuned} & \multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{$1$-NN} & OPG & Ceph.\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
         r"Encoder & src & adj & src & adj & src & adj & src & adj & pat. & img. & pat. & img. & "
         f"($n={fl['n_test_with_opg']}$) & ($n={fl['n_test_with_ceph']}$)\\\\\\midrule"]
    for gi, g in enumerate(order):
        for e in g:
            cells = []
            for c in cols:
                v = val(e, c)
                if v is None:
                    cells.append("--")
                else:
                    s = f3(v)
                    cells.append(f"\\textbf{{{s}}}" if abs(v - best[c]) < 1e-12 else s)
            L.append(f"{e} & " + " & ".join(cells) + r"\\")
        L.append(r"\midrule")
    t = la["human_and_oracle"]["test_189"]
    for nme, v, ci in (("Reviewer A", t["expert1_vs_manifest_macro_f1"], t["expert1_vs_manifest_ci"]),
                       ("Reviewer B", t["expert2_vs_manifest_macro_f1"], t["expert2_vs_manifest_ci"]),
                       ("Adjudicated label", t["oracle_final_vs_manifest_macro_f1"], t["oracle_final_vs_manifest_ci"])):
        L.append(f"{nme} & \\multicolumn{{8}}{{c}}{{vs source label: {f3(v)} ({f3(ci[0])}--{f3(ci[1])})}} & "
                 r"-- & -- & -- & -- & -- & --\\")
    L.append(r"\bottomrule\end{tabular}")
    (gen / "ex_tab2.tex").write_text("\n".join(L) + "\n")


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
        fig1(out, thumbs); fig2(out, thumbs)
    else:
        print("[skip] Figs 1-2 need the redacted record thumbnails (not released); rendering the rest")
    fig3(out); fig4(out); fig5(out); fig6(out); fig_comparison(out)
    tab1(gen); tab2(gen); supp_tables(gen)


if __name__ == "__main__":
    main()
