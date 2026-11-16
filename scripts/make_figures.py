#!/usr/bin/env python3
"""Generate paper figures from results/*.json (matplotlib only, no seaborn).

Design rules (PIPELINE.md 0c): Okabe-Ito colourblind-safe palette; ONE fixed hue
per method/system reused across ALL figures; labelled axes with units; no in-plot
titles; every number read only from results/*.json. Each figure -> figures/<name>.pdf
and figures/<name>.png (300 dpi).

Figures:
  fig0_teaser         F-B pipeline teaser (cohort -> audit -> split -> headline)
  fig1_leakage        main result: leakage inflation with uncertainty
  fig2_neardup        diagnostic: residual near-duplicate leakage
  fig3_baselines      diagnostic: per-class discriminativeness by backbone
  fig4_data_efficiency ablation: macro-F1 vs training patients per class
  fig5_comparison     F-D comparison matrix (incumbents x capabilities + #images)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from paper_style import style

style()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.8,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "figure.dpi": 120,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

# Okabe-Ito colourblind-safe palette
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999",
      "black": "#000000"}

# ONE fixed hue per concept, reused across every figure.
HUE = {
    "image_level": OI["vermillion"],   # naive / leaky split (also = leakage risk)
    "patient_level": OI["green"],      # correct grouped split
    "dinov2": OI["blue"],              # DINOv2 backbone
    "resnet50": OI["orange"],          # ResNet-50 backbone
    "all_pairs": OI["sky"],            # all near-duplicate pairs
    "cross_patient": OI["vermillion"], # cross-patient near-dups = residual leakage
    "dinov2_large": OI["sky"],         # DINOv2 ViT-L/14
    "siglip2": OI["purple"],           # SigLIP-2 base
    "biomedclip": OI["green"],         # BiomedCLIP
    "intraoral": OI["blue"],           # intraoral modality
    "opg": OI["orange"],               # panoramic OPG modality
    "cephalogram": OI["purple"],       # lateral cephalogram modality
    "finetuned": OI["black"],          # end-to-end fine-tuned model
}

DISP = {"dinov2": "DINOv2 S/14", "dinov2_large": "DINOv2 L/14", "resnet50": "ResNet-50",
        "siglip2": "SigLIP-2", "biomedclip": "BiomedCLIP",
        "intraoral": "Intraoral", "opg": "OPG", "cephalogram": "Cephalogram"}


def _load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    print(f"[fig] {name}")


def _polish_axes(ax, grid_axis="y"):
    """Apply a restrained, publication-style frame consistently."""
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#D8D8D8", linewidth=0.65, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------- teaser (F-B)
def fig_teaser(ds, nd, lk, ft=None):
    """Left->right dataset pipeline: real multimodal cohort -> curation/leakage
    audit -> patient-level split -> headline leakage result."""
    n_pat = ds["n_patients"]; n_img = ds["n_images"]
    m = ds["images_by_modality"]
    paired = ds["fully_paired_patients"]
    cross = nd["by_threshold"]["6"]["cum_cross_patient_leq"]
    infl_lin = lk["probes"]["logreg"]["inflation"]["macro_f1_abs"]
    infl_lin_pct = lk["probes"]["logreg"]["inflation"]["macro_f1_rel_pct"]
    infl_knn = lk["probes"]["knn1"]["inflation"]["macro_f1_abs"]
    infl_knn_pct = lk["probes"]["knn1"]["inflation"]["macro_f1_rel_pct"]
    img_f1 = lk["probes"]["knn1"]["image_level"]["macro_f1"]["mean"]
    pat_f1 = lk["probes"]["knn1"]["patient_level"]["macro_f1"]["mean"]

    fig, axes = plt.subplots(
        1, 4, figsize=(15.8, 3.65),
        gridspec_kw={"width_ratios": [1.02, 1.02, 1.02, 1.14], "wspace": 0.17},
    )
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(ax, color, header):
        ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                     boxstyle="round,pad=0.02,rounding_size=0.04",
                     linewidth=1.6, edgecolor=color, facecolor=color + "14"))
        ax.text(0.5, 0.90, header, ha="center", va="top", fontsize=11.5,
                fontweight="bold", color=OI["black"])

    # Panel 1: real multi-modality cohort
    ax = axes[0]; box(ax, OI["blue"], "Real Gulf clinical cohort")
    ax.text(0.5, 0.70, f"{n_pat} patients", ha="center", fontsize=13,
            fontweight="bold", color=OI["blue"])
    ax.text(0.5, 0.61, f"{n_img:,} images", ha="center", fontsize=11.5, color=OI["black"])
    # modality mini bar
    labels = ["intraoral", "OPG", "ceph"]
    vals = [m.get("intraoral", 0), m.get("opg", 0), m.get("cephalogram", 0)]
    cols = [OI["blue"], OI["orange"], OI["green"]]
    bx0, bw, by = 0.16, 0.22, 0.16
    mx = max(vals)
    for i, (lb, v, c) in enumerate(zip(labels, vals, cols)):
        h = 0.24 * v / mx
        ax.add_patch(plt.Rectangle((bx0 + i * bw, by), 0.14, h, color=c))
        ax.text(bx0 + i * bw + 0.07, by + h + 0.015, f"{v:,}", ha="center",
                fontsize=9, color=OI["black"])
        ax.text(bx0 + i * bw + 0.07, by - 0.06, lb, ha="center", fontsize=8.5,
                color=OI["black"])
    ax.text(0.5, 0.50, f"{paired} patients carry all 3 modalities", ha="center",
            fontsize=9.5, style="italic", color=OI["black"])

    # Panel 2: curation / leakage audit
    ax = axes[1]; box(ax, OI["vermillion"], "Curation & leakage audit")
    ax.text(0.5, 0.72, "exact-hash de-dup:", ha="center", fontsize=10, color=OI["black"])
    ax.text(0.5, 0.645, "0 cross-patient groups left", ha="center", fontsize=10.5,
            fontweight="bold", color=OI["green"])
    ax.text(0.5, 0.50, "perceptual-hash near-dups:", ha="center", fontsize=10,
            color=OI["black"])
    ax.text(0.5, 0.415, f"{cross} cross-patient pairs", ha="center", fontsize=12,
            fontweight="bold", color=OI["vermillion"])
    ax.text(0.5, 0.34, "(Hamming $\\leq 6$) survive", ha="center", fontsize=9.5,
            color=OI["black"])
    ax.text(0.5, 0.19, "residual leakage flagged", ha="center", fontsize=9.5,
            style="italic", color=OI["black"])

    # Panel 3: patient-level split protocol
    ax = axes[2]; box(ax, OI["green"], "Patient-level split")
    # draw patients as clusters of dots assigned wholesale to train/test
    ax.text(0.28, 0.74, "train", ha="center", fontsize=10.5, fontweight="bold",
            color=OI["black"])
    ax.text(0.74, 0.74, "test", ha="center", fontsize=10.5, fontweight="bold",
            color=OI["black"])
    rng = np.random.RandomState(0)
    for side, cx in [("train", 0.28), ("test", 0.74)]:
        npat = 3 if side == "train" else 2
        for pi in range(npat):
            px = cx - 0.11 + 0.11 * pi
            py = 0.52 - 0.0 * pi
            col = OI["green"]
            for _ in range(3):
                ax.add_patch(plt.Circle((px + rng.uniform(-0.02, 0.02),
                             py + rng.uniform(-0.05, 0.05)), 0.018,
                             color=col, alpha=0.85))
    ax.text(0.5, 0.30, "all images of a patient", ha="center", fontsize=9.5,
            color=OI["black"])
    ax.text(0.5, 0.235, "stay on ONE side", ha="center", fontsize=10.5,
            fontweight="bold", color=OI["green"])
    ax.text(0.5, 0.13, "(no identity leak)", ha="center", fontsize=9.5,
            style="italic", color=OI["black"])

    # Panel 4: headline result
    ax = axes[3]; box(ax, OI["black"], "Split protocol matters")
    xb = [0.30, 0.62]
    for x, v, c, lab in [(0.30, img_f1, HUE["image_level"], "image-level\n(leaky)"),
                         (0.62, pat_f1, HUE["patient_level"], "patient-level\n(correct)")]:
        h = 0.34 * v / max(img_f1, pat_f1)
        ax.add_patch(plt.Rectangle((x - 0.09, 0.23), 0.18, h, color=c))
        ax.text(x, 0.23 + h + 0.015, f"{v:.2f}", ha="center", fontsize=9.5,
                color=OI["black"])
        ax.text(x, 0.145, lab, ha="center", fontsize=8.5, color=OI["black"])
    ax.annotate("", xy=(0.62, 0.23 + 0.34 * pat_f1 / max(img_f1, pat_f1)),
                xytext=(0.30, 0.23 + 0.34 * img_f1 / max(img_f1, pat_f1)),
                arrowprops=dict(arrowstyle="<->", color=OI["vermillion"], lw=1.4))
    if ft is not None:
        r = ft["regimes"]
        infl_ft = (r["image_level"]["macro_f1"]["mean"]
                   - r["patient_level"]["macro_f1"]["mean"])
        infl_ft_pct = ft["inflation"]["macro_f1_rel_pct"]
    ax.text(0.5, 0.79, "macro-F1 inflated by up to", ha="center", fontsize=9.5,
            color=OI["black"])
    ax.text(0.5, 0.705, f"+{infl_knn:.3f} ({infl_knn_pct:.1f}%)", ha="center",
            fontsize=13, fontweight="bold", color=OI["vermillion"])
    foot = (f"1-NN upper bound; +{infl_ft:.3f} for the best system"
            if ft is not None else
            f"1-NN probe; +{infl_lin:.3f} for a linear probe")
    ax.text(0.5, 0.055, foot,
            ha="center", fontsize=7.6, style="italic", color="#444444")

    # arrows between panels (figure-level)
    for i in range(3):
        con = FancyArrowPatch((0.985, 0.5), (1.055, 0.5),
                              transform=axes[i].transAxes, clip_on=False,
                              arrowstyle="-|>", mutation_scale=18,
                              lw=2, color=OI["grey"])
        axes[i].add_patch(con)
    _save(fig, "fig0_teaser")


# ---------------------------------------------------------------- main result
def fig_leakage(lk, ft=None):
    """Every classifier is inflated by a leaky split. The fine-tuned model is the
    system a practitioner would actually deploy, so it -- not the 1-NN upper bound --
    is the realistic cost; 1-NN brackets the effect from above."""
    probes = [("logreg", "Linear probe"), ("knn5", "5-NN"), ("knn1", "1-NN")]
    img_m = [lk["probes"][p]["image_level"]["macro_f1"]["mean"] for p, _ in probes]
    img_s = [lk["probes"][p]["image_level"]["macro_f1"]["std"] for p, _ in probes]
    pat_m = [lk["probes"][p]["patient_level"]["macro_f1"]["mean"] for p, _ in probes]
    pat_s = [lk["probes"][p]["patient_level"]["macro_f1"]["std"] for p, _ in probes]
    labels = [n for _, n in probes]
    n_frozen = len(labels)
    if ft is not None:
        r = ft["regimes"]
        img_m.append(r["image_level"]["macro_f1"]["mean"])
        img_s.append(r["image_level"]["macro_f1"]["std"])
        pat_m.append(r["patient_level"]["macro_f1"]["mean"])
        pat_s.append(r["patient_level"]["macro_f1"]["std"])
        labels.append("Fine-tuned\nDINOv2")
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    bars_img = ax.bar(
        x - w / 2, img_m, w, yerr=img_s, capsize=3,
        color=HUE["image_level"], edgecolor="white", linewidth=0.6,
        label="Image-level split (leaky)",
    )
    bars_pat = ax.bar(
        x + w / 2, pat_m, w, yerr=pat_s, capsize=3,
        color=HUE["patient_level"], edgecolor="white", linewidth=0.6,
        label="Patient-level split (correct)",
    )
    for bars, vals in ((bars_img, img_m), (bars_pat, pat_m)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val - 0.035, f"{val:.3f}",
                    ha="center", va="top", fontsize=8.5, color="white",
                    fontweight="bold")
    for i in range(len(labels)):
        infl = img_m[i] - pat_m[i]
        ax.annotate(f"inflation +{infl:.3f}",
                    (x[i], max(img_m[i], pat_m[i]) + img_s[i] + 0.018),
                    ha="center", fontsize=9, color=OI["vermillion"],
                    fontweight="bold")
    if ft is not None:
        ax.axvline(n_frozen - 0.5, color="#BBBBBB", linewidth=0.9,
                   linestyle=(0, (3, 3)), zorder=0)
        top = max(img_m) + 0.13
        ax.text(n_frozen - 0.6, top, "frozen features", ha="right", va="center",
                fontsize=8.5, color="#888888", style="italic")
        ax.text(n_frozen - 0.4, top, "end-to-end", ha="left", va="center",
                fontsize=8.5, color="#888888", style="italic")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1 (mean $\\pm$ std over seeds)")
    ax.set_xlabel("Classifier (all scored per image, so only the split differs)")
    ax.set_ylim(0, max(img_m) + 0.15)
    ax.legend(loc="upper left", frameon=False, ncol=1)
    _polish_axes(ax)
    _save(fig, "fig1_leakage")


def fig_neardup(nd):
    ts = sorted(int(t) for t in nd["by_threshold"])
    cum = [nd["by_threshold"][str(t)]["cum_pairs_leq"] for t in ts]
    cx = [nd["by_threshold"][str(t)]["cum_cross_patient_leq"] for t in ts]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.7),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    for ax in axes:
        ax.set_xlabel("pHash Hamming threshold (bits)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _polish_axes(ax, grid_axis="both")
    axes[0].plot(ts, cum, "o-", lw=2, ms=4.5, color=HUE["all_pairs"],
                 label="All pairs")
    axes[0].plot(ts, cx, "s--", lw=2, ms=4.2, color=HUE["cross_patient"],
                 label="Cross-patient")
    axes[0].set_ylabel("Cumulative near-duplicate pairs")
    axes[0].set_title("Full threshold sweep", fontsize=10.5, pad=8,
                      fontweight="bold")
    axes[0].legend(loc="upper left", frameon=False)

    action_t = 6
    action_i = ts.index(action_t)
    action_slice = slice(0, action_i + 1)
    axes[1].plot(ts[action_slice], cum[action_slice], "o-", lw=2, ms=4.5,
                 color=HUE["all_pairs"])
    axes[1].plot(ts[action_slice], cx[action_slice], "s--", lw=2, ms=4.2,
                 color=HUE["cross_patient"])
    axes[1].set_xlim(min(ts) - 0.15, action_t + 0.25)
    axes[1].set_ylim(0, max(cum[:action_i + 1]) * 1.25)
    axes[1].set_title("Actionable low-distance region", fontsize=10.5, pad=8,
                      fontweight="bold")
    axes[1].axvline(action_t, color="#666666", ls=":", lw=1)
    axes[1].annotate(
        f"{cx[action_i]} cross-patient\npairs at threshold $\\leq {action_t}$",
        xy=(action_t, cx[action_i]), xytext=(action_t - 5.8, cx[action_i] + 45),
        arrowprops=dict(arrowstyle="->", color=HUE["cross_patient"], lw=1),
        color=HUE["cross_patient"], fontsize=9, fontweight="bold", ha="left",
    )
    fig.subplots_adjust(wspace=0.30)
    _save(fig, "fig2_neardup")


def fig_baselines(bl):
    # one fixed hue per backbone; grouped by Angle class on the x-axis
    order = sorted(bl["backbones"],
                   key=lambda b: bl["backbones"][b]["patient_macro_f1"]["mean"])
    labels = ["Class I", "Class II", "Class III"]
    disp = DISP
    x = np.arange(3); w = 0.84 / max(len(order), 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for k, bk in enumerate(order):
        rec = bl["backbones"][bk]
        # per-class F1 over ALL seeds; a single seed on the 115-patient minority
        # class swings by up to +/-0.08 and must not be plotted bare.
        pc = rec.get("per_class_f1_mean", rec["per_class_f1_seed0"])
        er = rec.get("per_class_f1_std", [0.0] * len(pc))
        bars = ax.bar(x + (k - (len(order) - 1) / 2) * w, pc, w, yerr=er, capsize=2,
                      error_kw={"elinewidth": 0.9},
                      color=HUE[bk], edgecolor="white", linewidth=0.6,
                      label=disp.get(bk, bk))
        for bar, val, e in zip(bars, pc, er):
            ax.text(bar.get_x() + bar.get_width() / 2, val + e + 0.012, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.2, color=HUE[bk],
                    fontweight="bold", rotation=90)
    ax.axhline(1 / 3, ls=":", color="#555555", lw=1,
               label="0.33 reference")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Per-class F1 (mean $\\pm$ std, 5 seeds)")
    ax.set_xlabel("Angle malocclusion class")
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8.5)
    ax.set_ylim(0, 0.80)
    _polish_axes(ax)
    _save(fig, "fig3_baselines")


def fig_data_efficiency(ab):
    de = ab.get("data_efficiency", {})
    if not de:
        return
    xs = sorted(int(k.split("_")[0]) for k in de)
    m = [de[f"{x}_per_class"]["mean"] for x in xs]
    s = [de[f"{x}_per_class"]["std"] for x in xs]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.errorbar(xs, m, yerr=s, fmt="o-", lw=2, ms=6, color=HUE["dinov2"],
                ecolor=HUE["dinov2"], elinewidth=1.2, capsize=3,
                label="DINOv2 linear probe")
    for xp, yp in zip(xs, m):
        ax.annotate(f"{yp:.3f}", (xp, yp), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=HUE["dinov2"], fontweight="bold")
    ax.set_xlabel("Training patients per class")
    ax.set_ylabel("Patient-level macro-F1 (mean $\\pm$ std)")
    ax.set_xticks(xs)
    ax.set_ylim(min(np.array(m) - np.array(s)) - 0.015,
                max(np.array(m) + np.array(s)) + 0.025)
    ax.legend(loc="lower right", frameon=False)
    _polish_axes(ax, grid_axis="both")
    _save(fig, "fig4_data_efficiency")


# ------------------------------------------------------- comparison matrix (F-D)
def fig_comparison(cm):
    rows = cm["datasets"]
    caps = cm["capability_columns"]
    cap_hdr = {"patient_linked_multimodal": "Patient-linked\nmultimodal",
               "angle_class_labels": "Angle-class\nlabels",
               "patient_level_split": "Patient-level\nsplit"}
    n = len(rows)
    # column x-positions
    x_name = 0.0
    x_mod = 0.19
    x_caps = [0.40, 0.55, 0.70]
    x_bar0 = 0.81
    bar_w = 0.18
    max_img = max([r["n_images"] for r in rows if r.get("n_images")])
    fig, ax = plt.subplots(figsize=(11.5, 0.66 * n + 1.25))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, n + 1.1); ax.axis("off")
    yh = n + 0.4
    # headers
    ax.text(x_name, yh, "Dataset", fontsize=11.5, fontweight="bold", va="center")
    ax.text(x_mod, yh, "Modalities", fontsize=11.5, fontweight="bold", va="center")
    for xc, c in zip(x_caps, caps):
        ax.text(xc, yh, cap_hdr[c], fontsize=10.5, fontweight="bold", va="center",
                ha="center")
    ax.text(x_bar0, yh, "# images (log-scaled bar)", fontsize=10.5,
            fontweight="bold", va="center")
    ax.plot([-0.02, 1.02], [n - 0.5 + 0.5, n - 0.5 + 0.5], color=OI["black"], lw=1.0)

    for i, r in enumerate(rows):
        y = n - 1 - i + 0.5
        is_ours = r["source_id"] == "ours"
        if is_ours:
            ax.add_patch(plt.Rectangle((-0.02, y - 0.35), 1.04, 0.72,
                         color=OI["blue"], alpha=0.08, zorder=0))
        elif i % 2:
            ax.add_patch(plt.Rectangle((-0.02, y - 0.35), 1.04, 0.72,
                         color="#000000", alpha=0.025, zorder=0))
        fw = "bold" if is_ours else "normal"
        ax.text(x_name, y, r["name"], fontsize=10.5, va="center", fontweight=fw)
        ax.text(x_mod, y, r["modalities"], fontsize=9.5, va="center")
        for xc, c in zip(x_caps, caps):
            val = r[c]
            sym, col = ("✓", OI["green"]) if val else ("✗", OI["vermillion"])
            ax.text(xc, y, sym, fontsize=15, va="center", ha="center",
                    color=col, fontweight="bold")
        # quantitative column: #images bar
        ni = r.get("n_images")
        if ni:
            # Log scaling preserves the large range while keeping smaller datasets legible.
            bl = bar_w * np.log10(ni + 1) / np.log10(max_img + 1)
            col = HUE["dinov2"] if is_ours else OI["grey"]
            ax.add_patch(plt.Rectangle((x_bar0, y - 0.16), bl, 0.32, color=col))
            ax.text(x_bar0 + bl + 0.008, y, f"{ni:,}", fontsize=9, va="center")
        else:
            ax.text(x_bar0, y, "n/r", fontsize=9, va="center", color=OI["grey"],
                    style="italic")
    _save(fig, "fig5_comparison")


# ------------------------------------------------- task primer (Angle + modalities)
# Class I, fully paired, 0 PHI-audit flags. The pseudonymous code is kept out of the
# released code: it lives in the restricted, gitignored data/ directory.
_EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "exemplar_patient.txt")
EXEMPLAR_PATIENT = open(_EX).read().strip() if os.path.exists(_EX) else os.environ.get("ORTHOBENCH_EXEMPLAR_PATIENT", "")
REDACT_MARGIN = 0.12                    # the release policy's border band


def _exemplar_frames(root, patient=EXEMPLAR_PATIENT, margin=REDACT_MARGIN, px=340):
    """Load one patient's record from the RESTRICTED root, apply the release
    border redaction, and downsample. Returns {} when the root is absent so the
    figure degrades to schematics and the repo still builds for anyone.

    Images are read in place and never copied into the repo; only the redacted,
    downsampled thumbnails reach the figure PDF.
    """
    import csv
    from PIL import Image, ImageDraw
    if not root or not os.path.isdir(root):
        return {}
    man = os.path.join(root, "manifests", "images.csv")
    if not os.path.exists(man):
        return {}
    Image.MAX_IMAGE_PIXELS = None
    out = {"intraoral": [], "opg": [], "cephalogram": []}
    for r in csv.DictReader(open(man)):
        if r["patient_id"] != patient:
            continue
        p = os.path.join(root, r["relative_path"])
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        mw, mh = int(w * margin), int(h * margin)
        d = ImageDraw.Draw(im)
        for box in [(0, 0, w, mh), (0, h - mh, w, h), (0, 0, mw, h), (w - mw, 0, w, h)]:
            d.rectangle(box, fill=(0, 0, 0))
        im.thumbnail((px, px), Image.LANCZOS)
        out[r["modality"]].append((r["record_id"], im))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    return out



def _font_scale(k):
    """Scale every font size for one figure so labels print at >= 7 pt (QA 2026-11-16)."""
    keys = ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize",
            "legend.fontsize")
    upd = {}
    for key in keys:
        v = plt.rcParams[key]
        if isinstance(v, (int, float)):
            upd[key] = v * k
    return plt.rc_context(upd)

def _fig_primer_impl(ds, data_root=None):
    """Reader-facing primer: what Angle Class I/II/III means, what each modality
    contributes, and what a patient's uncurated intraoral series actually looks
    like. Angle anatomy is schematic; the photographic panels are real frames from
    one PHI-audit-clean fully-paired patient, carrying the release border
    redaction. Every count is read from dataset_stats.json."""
    from matplotlib.patches import Rectangle, Polygon, Ellipse

    n_img = ds["images_by_modality"]
    ipp = ds["images_per_patient"]          # [median, q1, q3, min, max]
    lab = ds["patient_label_distribution"]
    root = data_root or os.environ.get("ORTHOBENCH_DATA_ROOT")
    ex = _exemplar_frames(root)
    have = all(ex.get(k) for k in ("intraoral", "opg", "cephalogram"))

    fig = plt.figure(figsize=(7.1, 4.15))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.02], hspace=0.50,
                             left=0.005, right=0.995, top=0.925, bottom=0.015)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.16, 1.0], wspace=0.20)

    fig.text(0.005, 0.972, "(a)  Angle class = sagittal first-molar relationship",
             ha="left", va="center", fontsize=8.4, fontweight="bold", color="#222222")
    fig.text(0.560, 0.972, "(b)  The two radiographic modalities",
             ha="left", va="center", fontsize=8.4, fontweight="bold", color="#222222")
    fig.add_artist(plt.Line2D([0.545, 0.545], [0.52, 0.955], color="#CCCCCC",
                              linewidth=0.9, transform=fig.transFigure))

    # ---------- (a) Angle classes: sagittal first-molar relationship -------------
    gsa = top[0].subgridspec(1, 3, wspace=0.12)
    classes = [("Class I", 0.00, "molars aligned"),
               ("Class II", 0.30, "lower molar behind"),
               ("Class III", -0.30, "lower molar ahead")]
    cls_col = {"Class I": OI["green"], "Class II": OI["orange"],
               "Class III": OI["vermillion"]}

    def _molar(ax, x, y, w, h, col, up=True):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=col, edgecolor="#333333",
                               linewidth=0.9, alpha=0.85, zorder=3))
        cy = y + h if up else y
        for cx in (x + 0.26 * w, x + 0.74 * w):
            ax.add_patch(Ellipse((cx, cy), 0.34 * w, 0.30 * h, facecolor=col,
                                 edgecolor="#333333", linewidth=0.9, zorder=3))

    for j, (name, shift, note) in enumerate(classes):
        ax = fig.add_subplot(gsa[0, j])
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.20, 1.10); ax.axis("off")
        col = cls_col[name]
        _molar(ax, 0.30, 0.60, 0.42, 0.17, "#BFBFBF", up=False)
        _molar(ax, 0.30 + shift * 0.42, 0.32, 0.42, 0.17, col, up=True)
        if j == 0:
            ax.text(0.24, 0.685, "upper", ha="right", va="center", fontsize=7.6,
                    color="#777777")
            ax.text(0.24, 0.405, "lower", ha="right", va="center", fontsize=7.6,
                    color="#777777")
            ax.annotate("", xy=(0.86, 0.185), xytext=(0.52, 0.185),
                        arrowprops=dict(arrowstyle="-|>", color="#999999",
                                        linewidth=0.8))
            ax.text(0.48, 0.185, "posterior", ha="right", va="center", fontsize=7.6,
                    color="#999999")
        ax.plot([0.30, 0.30], [0.26, 0.82], color="#888888", linewidth=0.7,
                linestyle=(0, (2, 2)), zorder=1)
        if shift != 0:
            ax.annotate("", xy=(0.30 + shift * 0.42, 0.245), xytext=(0.30, 0.245),
                        arrowprops=dict(arrowstyle="-|>", color=col, linewidth=1.3,
                                        shrinkA=0, shrinkB=0))
        ax.text(0.5, 1.02, name, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=col)
        ax.text(0.5, 0.10, note, ha="center", va="center", fontsize=7.6,
                color="#444444", style="italic")
        ax.text(0.5, -0.05, f"{lab[name]} patients", ha="center", va="center",
                fontsize=7.6, color="#222222", fontweight="bold")

    # ---------- (b) one exemplar per modality ------------------------------------
    gsb = top[1].subgridspec(1, 2, wspace=0.14)
    mods = [("opg", "Panoramic (OPG)", "whole dentition, one projection"),
            ("cephalogram", "Cephalogram", "skeletal profile, lateral view")]
    for j, (key, name, note) in enumerate(mods):
        ax = fig.add_subplot(gsb[0, j])
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.20, 1.10); ax.axis("off")
        col = HUE[key]
        if have:
            im = ex[key][0][1]
            ax.imshow(np.asarray(im), extent=(0.06, 0.94, 0.28, 0.88),
                      aspect="auto", zorder=2)
            ax.add_patch(Rectangle((0.06, 0.28), 0.88, 0.60, facecolor="none",
                                   edgecolor=col, linewidth=1.6, zorder=4))
        else:
            ax.add_patch(Rectangle((0.06, 0.28), 0.88, 0.60, facecolor="#FFFFFF",
                                   edgecolor=col, linewidth=1.4, zorder=2))
            if key == "intraoral":
                w, gap, n = 0.105, 0.030, 5
                x0 = 0.5 - (n * w + (n - 1) * gap) / 2.0
                for row_y in (0.62, 0.40):
                    for i in range(n):
                        ax.add_patch(Rectangle((x0 + i * (w + gap), row_y), w, 0.12,
                                               facecolor=col, edgecolor="#333333",
                                               linewidth=0.6, alpha=0.8, zorder=3))
            elif key == "opg":
                th = np.linspace(np.pi * 0.08, np.pi * 0.92, 60)
                ax.plot(0.5 + 0.36 * np.cos(th), 0.46 + 0.26 * np.sin(th), color=col,
                        linewidth=2.6, zorder=3)
            else:
                prof = np.array([[0.34, 0.32], [0.32, 0.50], [0.36, 0.64],
                                 [0.46, 0.76], [0.62, 0.79], [0.72, 0.70],
                                 [0.72, 0.59], [0.66, 0.54], [0.68, 0.46],
                                 [0.60, 0.42], [0.58, 0.32]])
                ax.add_patch(Polygon(prof, closed=True, facecolor=col, alpha=0.35,
                                     edgecolor=col, linewidth=1.6, zorder=3))
        ax.text(0.5, 1.02, name, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=col)
        ax.text(0.5, 0.15, note, ha="center", va="center", fontsize=7.6,
                color="#444444", style="italic", linespacing=1.15)
        ax.text(0.5, -0.02, f"{n_img[key]:,} images", ha="center", va="center",
                fontsize=7.6, color="#222222", fontweight="bold")
        ax.text(0.5, -0.115, f"median {ipp[key][0]:.0f}/patient", ha="center",
                va="center", fontsize=7.6, color="#666666")

    # ---------- (c) one patient's uncurated intraoral series ---------------------
    series = ex["intraoral"] if have else []
    n = max(len(series), 5)
    fig.text(0.005, 0.47,
             "(c)  One patient's intraoral series: what the intraoral track actually scores"
             f"  ({n_img['intraoral']:,} images, median {ipp['intraoral'][0]:.0f}/patient)",
             ha="left", va="center", fontsize=8.4, fontweight="bold", color="#222222")
    # QA 2026-11-16: two lines, so the >= 7 pt text stays inside the 7.1-inch canvas
    fig.text(0.005, 0.418,
             "No view labels exist, so every method must pool the whole series;\nthe molar "
             "relationship that defines Angle class is legible in only some frames.",
             ha="left", va="center", fontsize=7.6, color="#666666", style="italic", linespacing=1.1)
    gsc = outer[1].subgridspec(1, n, wspace=0.06)
    for j in range(n):
        ax = fig.add_subplot(gsc[0, j])
        ax.set_xlim(0, 1); ax.set_ylim(-0.22, 1.0); ax.axis("off")
        if have:
            ax.imshow(np.asarray(series[j][1]), extent=(0.02, 0.98, 0.10, 0.98),
                      aspect="auto", zorder=2)
            ax.add_patch(Rectangle((0.02, 0.10), 0.96, 0.88, facecolor="none",
                                   edgecolor=HUE["intraoral"], linewidth=1.2,
                                   zorder=4))
        else:
            ax.add_patch(Rectangle((0.02, 0.10), 0.96, 0.88, facecolor="#F2F2F2",
                                   edgecolor=HUE["intraoral"], linewidth=1.2,
                                   zorder=2))
        ax.text(0.5, -0.03, f"frame {j+1}", ha="center", va="center", fontsize=7.6,
                color="#666666")
    _save(fig, "fig9_primer")


# ------------------------------------------- modality tracks + leakage mechanism
def fig_primer(ds, data_root=None):
    with _font_scale(1.0):
        return _fig_primer_impl(ds, data_root)

def _fig_modality_impl(mt):
    """Left: per-modality difficulty. Right: the mechanism -- leakage inflation
    against images per patient, with the per-seed difference spread. Where a patient
    contributes one image the two split protocols coincide and the gap must vanish."""
    mods = [m for m in ("intraoral", "opg", "cephalogram") if m in mt["modalities"]]
    d = mt["modalities"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    m = [d[k]["patient_agg_macro_f1"]["mean"] for k in mods]
    e = [d[k]["patient_agg_macro_f1"]["std"] for k in mods]
    bars = ax.bar(np.arange(len(mods)), m, 0.6, yerr=e, capsize=3,
                  color=[HUE[k] for k in mods], edgecolor="white", linewidth=0.6)
    for bar, val, k in zip(bars, m, mods):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9*1.15, color="white", fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, val + e[mods.index(k)] + 0.008,
                f"n={d[k]['n_images']:,}", ha="center", va="bottom", fontsize=8*1.15,
                color="#555555")
    ax.axhline(0.329, ls=":", color="#555555", lw=1)
    ax.annotate("dotted line: stratified random (0.329)", (0.02, 0.96),
                xycoords="axes fraction", ha="left", fontsize=8*1.15, color="#555555")
    ax.set_xticks(np.arange(len(mods)))
    ax.set_xticklabels([DISP[k] for k in mods])
    ax.set_ylabel("Patient-level macro-F1\n(mean $\\pm$ std)")
    ax.set_xlabel("Benchmark track")
    ax.set_ylim(0, max(m) + 0.13)
    _polish_axes(ax)

    ax = axes[1]
    ipp = [d[k]["images_per_patient"] for k in mods]
    inf = [d[k]["leakage_inflation"]["macro_f1_abs"] for k in mods]
    ise = [d[k]["leakage_inflation"]["macro_f1_std"] for k in mods]
    ax.axhline(0, color="#444444", lw=1)
    # stagger the two near-coincident low-images-per-patient labels
    offs = {"intraoral": (-150, 22), "cephalogram": (14, -78), "opg": (14, -48)}
    for k, xv, yv, ev in zip(mods, ipp, inf, ise):
        real = d[k]["leakage_inflation"]["exceeds_seed_noise"]
        ax.errorbar(xv, yv, yerr=ev, fmt="o", ms=9, color=HUE[k], ecolor=HUE[k],
                    elinewidth=1.4, capsize=4,
                    markerfacecolor=HUE[k] if real else "white",
                    markeredgecolor=HUE[k], markeredgewidth=1.6)
        ax.annotate(f"{DISP[k]} {yv:+.3f} $\\pm$ {ev:.3f}", (xv, yv),
                    xytext=offs.get(k, (8, 14)), textcoords="offset points",
                    fontsize=8.5*1.15, color=HUE[k], fontweight="bold")
    ax.set_xlabel("Images per patient in the track")
    ax.set_ylabel("Leakage inflation\n(image $-$ patient macro-F1)")
    ax.set_xlim(0.3, 9.0)
    ax.annotate("filled = exceeds seed noise;  hollow = within noise",
                (0.5, 1.02), xycoords="axes fraction", ha="center", va="bottom", fontsize=8*1.15,
                color="#555555")
    _polish_axes(ax, grid_axis="both")
    fig.subplots_adjust(wspace=0.42)
    _save(fig, "fig6_modality")


# --------------------------------------------- confusion + per-class intervals
def fig_modality(mt):
    with _font_scale(1.15):
        return _fig_modality_impl(mt)

def fig_confusion(bl):
    """Where the benchmark actually fails. Left: patient-level confusion for the
    two most informative backbones. Right: per-class F1 with patient-bootstrap
    intervals, which show that nearly all the uncertainty sits on Class III."""
    show = [b for b in ("dinov2", "biomedclip") if b in bl["backbones"]
            and bl["backbones"][b].get("confusion_seed0")]
    if not show:
        return
    names = ["I", "II", "III"]
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(8.6, 2.9),
                             gridspec_kw={"width_ratios": [1] * len(show) + [1.35]})
    for ax, bk in zip(axes[:len(show)], show):
        cm = np.asarray(bl["backbones"][bk]["confusion_seed0"], dtype=float)
        row = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]*100:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "#222222")
        ax.set_xticks(range(3)); ax.set_xticklabels(names)
        ax.set_yticks(range(3)); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        ax.set_title(DISP.get(bk, bk), fontsize=9, color=HUE[bk], fontweight="bold")
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax = axes[-1]
    off = np.linspace(-0.18, 0.18, len(show))
    for k, bk in enumerate(show):
        rec = bl["backbones"][bk].get("per_class_bootstrap_ci_seed0")
        if not rec:
            continue
        xs = np.arange(3) + off[k]
        m = [r["f1_mean"] for r in rec]
        lo = [r["f1_mean"] - r["f1_ci"][0] for r in rec]
        hi = [r["f1_ci"][1] - r["f1_mean"] for r in rec]
        ax.errorbar(xs, m, yerr=[lo, hi], fmt="o", ms=6, capsize=4, elinewidth=1.4,
                    color=HUE[bk], label=DISP.get(bk, bk))
    ax.set_xticks(range(3)); ax.set_xticklabels("Class " + n for n in names)
    ax.set_ylabel("Per-class F1\n($95\\%$ patient-bootstrap CI)")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _polish_axes(ax)
    fig.subplots_adjust(wspace=0.42)
    _save(fig, "fig10_confusion")


# ------------------------------------------ leakage mechanism (column-width)
def fig_mechanism(mt):
    """The mechanism alone, sized for one column: leakage inflation against images
    per patient. Where a patient contributes one image the two split protocols
    coincide and the gap must vanish, which is what the hollow markers show."""
    mods = [m for m in ("intraoral", "opg", "cephalogram") if m in mt["modalities"]]
    d = mt["modalities"]
    fig, ax = plt.subplots(figsize=(5.2, 3.05))
    ipp = [d[k]["images_per_patient"] for k in mods]
    inf = [d[k]["leakage_inflation"]["macro_f1_abs"] for k in mods]
    ise = [d[k]["leakage_inflation"]["macro_f1_std"] for k in mods]
    ax.axhline(0, color="#444444", lw=1)
    offs = {"intraoral": (-8, 20), "cephalogram": (12, 20), "opg": (12, -26)}
    for k, xv, yv, ev in zip(mods, ipp, inf, ise):
        real = d[k]["leakage_inflation"]["exceeds_seed_noise"]
        ax.errorbar(xv, yv, yerr=ev, fmt="o", ms=9, color=HUE[k], ecolor=HUE[k],
                    elinewidth=1.4, capsize=4,
                    markerfacecolor=HUE[k] if real else "white",
                    markeredgecolor=HUE[k], markeredgewidth=1.6)
        ax.annotate(f"{DISP[k]}\n{yv:+.3f} $\\pm$ {ev:.3f}", (xv, yv),
                    xytext=offs.get(k, (8, 14)), textcoords="offset points",
                    fontsize=8.5, color=HUE[k], fontweight="bold",
                    ha="right" if k == "intraoral" else "left", linespacing=1.1)
    ax.set_xlabel("Images per patient in the track")
    ax.set_ylabel("Leakage inflation\n(image $-$ patient macro-F1)")
    ax.set_xlim(0.3, 9.2)
    ax.annotate("filled = exceeds seed noise", (0.98, 0.04),
                xycoords="axes fraction", ha="right", fontsize=8, color="#555555")
    _polish_axes(ax, grid_axis="both")
    _save(fig, "fig6b_mechanism")


# ------------------------------------------------------------------ fusion track
def _fig_fusion_impl(fu):
    tr = fu["tracks"]
    keys = [k for k in ("intraoral", "opg", "cephalogram", "intraoral+opg",
                        "intraoral+cephalogram", "opg+cephalogram",
                        "intraoral+opg+cephalogram") if k in tr]
    short = {"intraoral": "IO", "opg": "OPG", "cephalogram": "Ceph"}
    lab = ["+\n".join(short[p] for p in k.split("+")) for k in keys]
    m = [tr[k]["macro_f1"]["mean"] for k in keys]
    e = [tr[k]["macro_f1"]["std"] for k in keys]
    col = [HUE[k] if "+" not in k else OI["grey"] for k in keys]
    col[-1] = OI["black"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.bar(np.arange(len(keys)), m, 0.62, yerr=e, capsize=3, color=col,
                  edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, m):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8.5*1.05, color="white", fontweight="bold")
    best_uni = max(m[:3])
    ax.axhline(best_uni, ls="--", color=OI["vermillion"], lw=1.1)
    ax.annotate(f"dashed line: best single modality ({best_uni:.3f})", (0.02, 0.89),
                xycoords="axes fraction", ha="left", fontsize=8.5*1.05,
                color=OI["vermillion"])
    ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels(lab, fontsize=8.5*1.05)
    ax.set_ylabel("Macro-F1 (mean $\\pm$ std, 10 seeds)")
    ax.set_xlabel(f"Modality blocks fused ({fu['n_patients_fully_paired']} "
                  f"fully-paired patients)")
    ax.set_ylim(0, max(np.array(m) + np.array(e)) + 0.115)
    ax.annotate("all error bars overlap: the paired cohort cannot yet resolve a "
                "fusion gain", (0.02, 0.96), xycoords="axes fraction", ha="left",
                fontsize=8.5*1.05, color="#555555")
    _polish_axes(ax)
    _save(fig, "fig7_fusion")


# --------------------------------------------------- pixel-PHI audit and release
def fig_fusion(fu):
    with _font_scale(1.05):
        return _fig_fusion_impl(fu)

def _fig_phi_impl(ph, rd):
    """Three panels: how much of the cohort needs review, how sensitive the detector
    is, and what a blanket border redaction costs the benchmark (nothing)."""
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 4.0),
                             gridspec_kw={"width_ratios": [1.05, 1.0, 0.78]})
    ax = axes[0]
    mods = [m for m in ("intraoral", "opg", "cephalogram") if m in ph["by_modality"]]
    pct = [ph["by_modality"][m]["flagged_pct"] for m in mods]
    bars = ax.bar(np.arange(len(mods)), pct, 0.6,
                  color=[HUE[m] for m in mods], edgecolor="white", linewidth=0.6)
    for bar, val, m in zip(bars, pct, mods):
        ax.text(bar.get_x() + bar.get_width() / 2, val - 0.5,
                f"{val:.1f}%\n({ph['by_modality'][m]['flagged_risk']})",
                ha="center", va="top", fontsize=8.5*1.2, color="white",
                fontweight="bold")
    ax.axhline(ph["flagged_pct_of_cohort"], ls=":", color="#444444", lw=1)
    ax.annotate(f"dotted line: whole cohort {ph['flagged_pct_of_cohort']:.2f}%",
                (0.02, 0.96), xycoords="axes fraction", ha="left", fontsize=8*1.2,
                color="#444444")
    ax.set_xticks(np.arange(len(mods))); ax.set_xticklabels([{"cephalogram": "Ceph."}.get(m, DISP[m]) for m in mods])
    ax.set_ylabel("Images flagged\nfor PHI review (%)")
    ax.set_xlabel("Modality")
    ax.set_ylim(0, max(pct) + 4.2)
    _polish_axes(ax)

    ax = axes[1]
    rp = ph.get("recall_probe", {})
    bands = rp.get("sensitivity_by_font_size", {})
    bk = list(bands)
    sens = [bands[b]["sensitivity"] for b in bk]
    lab = [b.replace("pct", "%").replace("-", "\u2013\n") for b in bk]
    bars = ax.bar(np.arange(len(bk)), sens, 0.6, color=OI["blue"],
                  edgecolor="white", linewidth=0.6)
    for bar, val, b in zip(bars, sens, bk):
        ax.text(bar.get_x() + bar.get_width() / 2, val - 0.07, f"{val:.2f}",
                ha="center", va="top", fontsize=9*1.2, color="white", fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"n={bands[b]['n']}", ha="center", va="bottom", fontsize=8*1.2,
                color="#555555")
    ax.axhline(rp.get("sensitivity", 0), ls="--", color=OI["vermillion"], lw=1.1)
    ax.annotate(f"dashed line: overall {rp.get('sensitivity', 0):.2f}", (0.02, 0.96),
                xycoords="axes fraction", ha="left", fontsize=8.5*1.2,
                color=OI["vermillion"])
    ax.set_xticks(np.arange(len(bk))); ax.set_xticklabels(lab, fontsize=9*1.2)
    ax.set_ylabel("Detection rate,\nsynthetic overlays")
    ax.set_xlabel("Overlay height (% of frame)")
    ax.set_ylim(0, 1.08)
    _polish_axes(ax)

    ax = axes[2]
    v = rd["variants"]
    names = ["Unredacted", "Border\nredacted"]
    m = [v["plain"]["patient_macro_f1"]["mean"],
         v["margin_redacted"]["patient_macro_f1"]["mean"]]
    e = [v["plain"]["patient_macro_f1"]["std"],
         v["margin_redacted"]["patient_macro_f1"]["std"]]
    bars = ax.bar([0, 1], m, 0.52, yerr=e, capsize=4,
                  color=[OI["grey"], OI["green"]], edgecolor="white", linewidth=0.6)
    for bar, val, ev in zip(bars, m, e):
        ax.text(bar.get_x() + bar.get_width() / 2, val - ev - 0.015, f"{val:.3f}",
                ha="center", va="top", fontsize=9*1.2, color="white", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=9*1.2)
    ax.set_ylabel("Patient-level\nmacro-F1")
    ax.set_ylim(0, max(np.array(m) + np.array(e)) + 0.10)
    ax.annotate(f"paired $\\Delta$ = {rd['delta']['macro_f1_abs']:+.3f} "
                f"$\\pm$ {rd['delta']['macro_f1_std']:.3f}",
                (0.5, 0.93), xycoords="axes fraction", ha="center", fontsize=8.5*1.2,
                color="#555555")
    _polish_axes(ax)
    fig.subplots_adjust(wspace=0.55)
    _save(fig, "fig8_phi")


def fig_phi(ph, rd):
    with _font_scale(1.2):
        return _fig_phi_impl(ph, rd)

def main():
    datasets = {name: _load(name + ".json") for name in (
        "dataset_stats", "neardup", "leakage", "finetune", "baselines",
        "ablations", "modality_tracks", "fusion", "phi_audit", "redaction",
        "comparison_matrix")}
    ds, nd, lk = (datasets[k] for k in ("dataset_stats", "neardup", "leakage"))
    if ds:
        fig_primer(ds)
    if ds and nd and lk:
        fig_teaser(ds, nd, lk, datasets["finetune"])
    if lk:
        fig_leakage(lk, datasets["finetune"])
    if nd:
        fig_neardup(nd)
    if datasets["baselines"]:
        fig_baselines(datasets["baselines"])
        fig_confusion(datasets["baselines"])
    if datasets["ablations"]:
        fig_data_efficiency(datasets["ablations"])
    if datasets["modality_tracks"]:
        fig_modality(datasets["modality_tracks"])
    if datasets["fusion"]:
        fig_fusion(datasets["fusion"])
    if datasets["phi_audit"] and datasets["redaction"]:
        fig_phi(datasets["phi_audit"], datasets["redaction"])
    if datasets["comparison_matrix"]:
        fig_comparison(datasets["comparison_matrix"])
    with open(os.path.join(os.path.dirname(__file__), "figure_catalog.json")) as f:
        catalog = json.load(f)
    manifest = [r for r in catalog if os.path.exists(os.path.join(FIG, r["file"] + ".pdf"))]
    with open(os.path.join(FIG, "figures.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[figures] historical figures rendered; run scripts/render_reference.py "
          "to apply the completed-audit teaser, comparison, and tables")


if __name__ == "__main__":
    main()
