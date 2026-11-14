#!/usr/bin/env python3
"""CVPR 2027 figures and LaTeX tables, rendered only from persisted results/*.json.

  python scripts/make_figures_cvpr27.py [--results DIR] [--out DIR] [--tables DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, GRAY = ("#0072B2", "#E69F00", "#009E73", "#D55E00",
                                               "#CC79A7", "#56B4E9", "#8a8a8a")
HEAD_COLOR = {"logreg": BLUE, "knn1": VERM, "knn5": ORANGE, "fine-tuned": GREEN}


def head_of(name):
    return name.split(" / ")[1]


def load(d, name):
    return json.loads((Path(d) / name).read_text())


def slope(ax, names, left, right, ltxt, rtxt, label_names=(), title=""):
    for n, a, b in zip(names, left, right):
        c = HEAD_COLOR[head_of(n)]
        lw, al = (1.6, 1.0) if n in label_names else (0.6, 0.45)
        ax.plot([0, 1], [a, b], color=c, lw=lw, alpha=al, marker="o", ms=1.8 if n not in label_names else 3)
    ax.set_xlim(-0.9, 1.9)
    ax.set_xticks([0, 1], [ltxt, rtxt])
    ax.grid(axis="x", visible=False)
    ax.set_title(title, fontsize=8, loc="left", fontweight="bold")


def teaser(split, ref, lab, out):
    ps.style()
    plt.rcParams.update({"font.size": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 7.5,
                         "axes.titlesize": 8, "lines.markersize": 3})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.0], "wspace": 0.5})
    # (a) split variable
    e = split["entries"]
    names = list(e)
    A = [e[n]["patient_split"] for n in names]
    B = [e[n]["image_split"] for n in names]
    top_img, top_pat = split["top_image_split"], split["top_patient_split"]
    slope(axes[0], names, A, B, "patient\nsplit", "image\nsplit", {top_img, top_pat},
          f"(a) Split unit ($\\tau_b$={split['kendall_tau_point']:.2f})")
    axes[0].set_ylabel("per-image macro-F1 (source labels)")
    for n, side, val in ((top_pat, 0, e[top_pat]["patient_split"]), (top_img, 1, e[top_img]["image_split"])):
        axes[0].annotate(n.replace(" / ", "\n"), (side, val), xytext=(-4 if side == 0 else 4, 0), textcoords="offset points",
                         ha="right" if side == 0 else "left", va="center", fontsize=6, color=HEAD_COLOR[head_of(n)], fontweight="bold")
    # (b) reference variable: highlight the winner and the significant swap among linear probes
    mods = ref["macro_f1"]["models"]
    nm = [m["name"] for m in mods]
    L = [m["score_ref_a"] for m in mods]
    R = [m["score_ref_b"] for m in mods]
    tb = ref["macro_f1"]["top_ref_b"]
    swap = [p for p in ref["macro_f1"]["pairs_significant"]
            if p["flip"] and p["a"].endswith("logreg") and p["b"].endswith("logreg")]
    hl = {tb} | ({swap[0]["a"], swap[0]["b"]} if swap else set())
    slope(axes[1], nm, L, R, "source\nlabel", "expert\nlabel", hl,
          f"(b) Label version ($\\tau_b$={ref['macro_f1']['kendall_tau_point']:.2f})")
    axes[1].set_ylabel("patient macro-F1 (same predictions)")
    for n in hl:
        m = mods[nm.index(n)]
        a_, b_ = m["score_ref_a"], m["score_ref_b"]
        lab_ = n.replace(" / ", "\n").replace("logreg", f"#{m['rank_ref_a']} \u2192 #{m['rank_ref_b']}")
        dy = 0
        if swap and n in (swap[0]["a"], swap[0]["b"]):
            other = swap[0]["b"] if n == swap[0]["a"] else swap[0]["a"]
            dy = 5 if b_ >= mods[nm.index(other)]["score_ref_b"] else -5
        axes[1].annotate(lab_, (1, b_), xytext=(4, dy), textcoords="offset points", ha="left", va="center",
                         fontsize=5.6, color=HEAD_COLOR[head_of(n)], fontweight="bold")
    # (c) label ceiling
    h = lab["human_and_oracle"]["test_189"]
    best = max(mods, key=lambda m: m["score_ref_a"])
    rows = [("best of %d\nmodels" % len(mods), best["score_ref_a"], None, BLUE),
            ("reviewer B", h["expert2_vs_manifest_macro_f1"], h["expert2_vs_manifest_ci"], ORANGE),
            ("reviewer A", h["expert1_vs_manifest_macro_f1"], h["expert1_vs_manifest_ci"], ORANGE),
            ("adjudicated\nlabel", h["oracle_final_vs_manifest_macro_f1"], h["oracle_final_vs_manifest_ci"], VERM)]
    ax = axes[2]
    y = np.arange(len(rows))
    for i, (t, v, c, col) in enumerate(rows):
        err = None if c is None else [[v - c[0]], [c[1] - v]]
        ax.barh(i, v, color=col, height=0.62, xerr=err, capsize=2.5, error_kw={"lw": 0.8})
        ax.text(v + (0.0 if c is None else c[1] - v) + 0.02, i, f"{v:.3f}", va="center", fontsize=6.5, fontweight="bold")
    ax.axvline(1.0, color=GRAY, lw=0.8, ls="--")
    ax.text(0.985, 3.45, "perfect", ha="right", va="center", fontsize=6, color=GRAY)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.6, 3.7)
    ax.set_xlabel("macro-F1 vs source label")
    ax.grid(axis="y", visible=False)
    ax.set_title("(c) Ceiling of the source label", fontsize=8, loc="left", fontweight="bold")
    handles = [plt.Line2D([], [], color=c, marker="o", lw=1.2, ms=3, label=k) for k, c in HEAD_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.33, -0.15), fontsize=6.5)
    ps.save(fig, str(Path(out) / "fig_teaser_cvpr27"))


def labels_fig(ref, lab, out):
    ps.style()
    plt.rcParams.update({"font.size": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.labelsize": 6.5})
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0), gridspec_kw={"wspace": 0.9, "width_ratios": [1, 1.05]})
    mods = ref["macro_f1"]["models"]
    ax = axes[0]
    ag = [m["acc_final_agreed"] for m in mods]
    ad = [m["acc_final_adjudicated"] for m in mods]
    for a, b, m in zip(ag, ad, mods):
        ax.plot([0, 1], [a, b], color=HEAD_COLOR[head_of(m["name"])], alpha=0.5, lw=0.8, marker="o", ms=2.5)
    d = ref["changed_case_decomposition"]
    ax.plot([0, 1], [d["zoo_mean_acc_final_agreed"], d["zoo_mean_acc_final_adjudicated"]], color="k", lw=1.5, marker="s", ms=3)
    ax.set_xticks([0, 1], [f"agreed\n(n={d['n_agreed_test']})", f"adjudicated\n(n={d['n_adjudicated_test']})"])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylabel("accuracy vs expert label")
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Expert agreement", fontsize=6.8, loc="left", fontweight="bold")
    ax = axes[1]
    det = lab["label_error_detection"]["per_model"]
    items = sorted(det.items(), key=lambda kv: kv[1]["auroc"])
    items.append((f"ensemble ({len(lab['label_error_detection']['ensemble_members'])} linear)", {"auroc": lab["label_error_detection"]["ensemble_auroc"],
                                           "auroc_ci": lab["label_error_detection"]["ensemble_auroc_ci"]}))
    for i, (n, v) in enumerate(items):
        col = "k" if n.startswith("ensemble") else HEAD_COLOR[head_of(n)]
        ax.errorbar(v["auroc"], i, xerr=[[v["auroc"] - v["auroc_ci"][0]], [v["auroc_ci"][1] - v["auroc"]]],
                    fmt="o", color=col, ms=3, lw=0.8, capsize=1.5)
    ax.axvline(0.5, color=GRAY, ls="--", lw=0.8)
    ax.set_yticks(range(len(items)), [n.replace(" / logreg", "").replace(" / fine-tuned", " (FT)") for n, _ in items], fontsize=5.2)
    ax.set_xlabel("AUROC, expert-changed labels")
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) Change detection", fontsize=6.8, loc="left", fontweight="bold")
    ps.save(fig, str(Path(out) / "fig_labels_cvpr27"))


def dose_fig(ref, out):
    """Exploratory: leaderboard agreement as a growing share of the expert changes is applied."""
    ps.style()
    plt.rcParams.update({"font.size": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.labelsize": 6.5})
    c = ref["partial_correction"]["curve"]
    x = [100 * r["fraction_corrected"] for r in c]
    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    lo = [r["kendall_tau_ci"][0] for r in c]
    hi = [r["kendall_tau_ci"][1] for r in c]
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, [r["kendall_tau_mean"] for r in c], color=BLUE, marker="o", ms=2.5, lw=1.2)
    n = ref["changed_case_decomposition"]["n_changed_test"]
    ax.set_xlabel(f"share of the {n} expert changes applied (%)")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Kendall $\\tau_b$ vs\nsource ranking")
    ps.save(fig, str(Path(out) / "fig_dose_cvpr27"))


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def tables(ref, split, ft, fus, out):
    mods = ref["macro_f1"]["models"]
    bacc = {m["name"]: m for m in ref["balanced_accuracy"]["models"]}
    sel = sorted(mods, key=lambda m: -m["score_ref_b"])
    lines = [r"\begin{tabular}{@{}lccccc@{}}\toprule",
             r"Entry & \multicolumn{2}{c}{Source label} & \multicolumn{2}{c}{Expert label} & Bal.\ acc.\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & F1 & rank & F1 & rank & expert\\\midrule"]
    keep = [m for m in sel if m["name"].endswith(("logreg", "fine-tuned"))]
    for m in keep:
        lines.append(f"{m['name'].replace(' / logreg', ' (linear)').replace(' / fine-tuned', ' (fine-tuned)')} & "
                     f"{fmt(m['score_ref_a'])} & {m['rank_ref_a']} & {fmt(m['score_ref_b'])} & {m['rank_ref_b']} & "
                     f"{fmt(bacc[m['name']]['score_ref_b'])}\\\\")
    lines += [r"\bottomrule\end{tabular}"]
    (Path(out) / "zoo_reference.tex").write_text("\n".join(lines) + "\n")


def supp_tables(ref, split, lab, fus, ft, out):
    """Full per-entry tables for the supplement (every executed arm)."""
    bacc = {m["name"]: m for m in ref["balanced_accuracy"]["models"]}
    L = [r"\begin{tabular}{@{}lccccc@{}}\toprule",
         r"Entry & Source F1 & Expert F1 & $\Delta$ & $95\%$ CI of $\Delta$ & Expert bal.\ acc.\\\midrule"]
    for m in sorted(ref["macro_f1"]["models"], key=lambda m: -m["score_ref_b"]):
        lo, hi = m["delta_ci"]
        L.append(f"{m['name']} & {fmt(m['score_ref_a'])} & {fmt(m['score_ref_b'])} & {m['delta']:+.3f} & "
                 f"{lo:+.3f} to {hi:+.3f} & {fmt(bacc[m['name']]['score_ref_b'])}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (Path(out) / "supp_zoo_reference.tex").write_text("\n".join(L) + "\n")
    L = [r"\begin{tabular}{@{}lccccc@{}}\toprule",
         r"Entry & Patient split & Image split & Inflation & SD & Ranks (pat.\,/\,img.)\\\midrule"]
    for n, e in sorted(split["entries"].items(), key=lambda kv: -kv[1]["patient_split"]):
        L.append(f"{n} & {fmt(e['patient_split'])} & {fmt(e['image_split'])} & {e['inflation']:+.3f} & "
                 f"{fmt(e['inflation_sd'])} & {e['rank_patient_split']}\\,/\\,{e['rank_image_split']}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (Path(out) / "supp_split_zoo.tex").write_text("\n".join(L) + "\n")
    pr = lab["confident_learning_pruning"]["per_encoder"]
    L = [r"\begin{tabular}{@{}lcccccc@{}}\toprule",
         r"Encoder & Pruned & \multicolumn{2}{c}{Source F1} & \multicolumn{2}{c}{Expert F1} & Gap $95\%$ CI\\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6} & patients & base & pruned & base & pruned & \\\midrule"]
    for n, v in pr.items():
        lo, hi = v["gap_final_minus_manifest_ci"]
        L.append(f"{n} & {v['n_pruned_patients']} & {fmt(v['manifest_base'])} & {fmt(v['manifest_pruned'])} & "
                 f"{fmt(v['final_base'])} & {fmt(v['final_pruned'])} & {lo:+.3f} to {hi:+.3f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (Path(out) / "supp_pruning.tex").write_text("\n".join(L) + "\n")
    if fus and fus.get("per_encoder"):
        L = [r"\begin{tabular}{@{}lcccc@{}}\toprule",
             r"Encoder & Intraoral & Late fusion & $\Delta$ & $95\%$ CI\\\midrule"]
        for n, v in fus["per_encoder"].items():
            f = v["final"]
            L.append(f"{n} & {fmt(f['intraoral'])} & {fmt(f['late_fusion'])} & {f['delta']:+.3f} & "
                     f"{f['delta_ci'][0]:+.3f} to {f['delta_ci'][1]:+.3f}\\\\")
        L.append(r"\bottomrule\end{tabular}")
        (Path(out) / "supp_fusion.tex").write_text("\n".join(L) + "\n")
    if fus and fus.get("radiograph_tracks"):
        L = [r"\begin{tabular}{@{}llccc@{}}\toprule",
             r"Track & Encoder (linear) & Test patients & Source F1 & Expert F1 ($95\%$ CI)\\\midrule"]
        for k, v in fus["radiograph_tracks"].items():
            m, enc = k.split("/", 1)
            L.append(f"{'OPG' if m == 'opg' else 'Ceph.'} & {enc} & {v['n_test']} & {fmt(v['manifest_macro_f1'])} & "
                     f"{fmt(v['final_macro_f1'])} ({fmt(v['final_ci'][0])}--{fmt(v['final_ci'][1])})\\\\")
        L.append(r"\bottomrule\end{tabular}")
        (Path(out) / "supp_radiograph.tex").write_text("\n".join(L) + "\n")
    if fus and fus.get("radiograph_tracks"):
        rt = fus["radiograph_tracks"]
        encs = [k.split("/", 1)[1] for k in rt if k.startswith("opg/")]
        n_o = rt[f"opg/{encs[0]}"]["n_test"]
        n_c = rt[f"cephalogram/{encs[0]}"]["n_test"]
        best_o = max(encs, key=lambda e: rt[f"opg/{e}"]["final_macro_f1"])
        best_c = max(encs, key=lambda e: rt[f"cephalogram/{e}"]["final_macro_f1"])
        L = [r"\begin{tabular}{@{}lcc@{}}\toprule",
             f"Encoder (linear) & OPG ($n={n_o}$) & Ceph.\\ ($n={n_c}$)\\\\\\midrule"]
        for e in encs:
            o, c = rt[f"opg/{e}"]["final_macro_f1"], rt[f"cephalogram/{e}"]["final_macro_f1"]
            o_s = f"\\textbf{{{fmt(o)}}}" if e == best_o else fmt(o)
            c_s = f"\\textbf{{{fmt(c)}}}" if e == best_c else fmt(c)
            L.append(f"{e} & {o_s} & {c_s}\\\\")
        L.append(r"\bottomrule\end{tabular}")
        (Path(out) / "main_radiograph.tex").write_text("\n".join(L) + "\n")
    if ft:
        names = {"dinov2": "DINOv2-S", "resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-T"}
        L = [r"\begin{tabular}{@{}lccccc@{}}\toprule",
             r"Backbone & Seed & Batch & Source F1 & Expert F1 & Expert bal.\ acc.\\\midrule"]
        for v in ft["runs"].values():
            L.append(f"{names.get(v['arch'], v['arch'])} & {v['seed']} & {v['batch']} & {fmt(v['manifest_macro_f1'])} & "
                     f"{fmt(v['final_macro_f1'])} & {fmt(v['final_balanced_accuracy'])}\\\\")
        L.append(r"\bottomrule\end{tabular}")
        (Path(out) / "supp_finetune.tex").write_text("\n".join(L) + "\n")
    det = lab["label_error_detection"]
    L = [r"\begin{tabular}{@{}lcccc@{}}\toprule",
         r"Encoder (linear) & AUROC & $95\%$ CI & CL precision & CL recall\\\midrule"]
    for n, v in det["per_model"].items():
        enc = n.split(" / ")[0]
        cl = det["confident_learning_flags"].get(enc)
        tail = f"{fmt(cl['precision'])} & {fmt(cl['recall'])}" if (cl and n.endswith("logreg")) else "-- & --"
        L.append(f"{n.replace(' / logreg', '').replace(' / fine-tuned', ' (fine-tuned)')} & {fmt(v['auroc'])} & "
                 f"{fmt(v['auroc_ci'][0])}--{fmt(v['auroc_ci'][1])} & {tail}\\\\")
    L.append(f"Ensemble (all linear) & {fmt(det['ensemble_auroc'])} & {fmt(det['ensemble_auroc_ci'][0])}--"
             f"{fmt(det['ensemble_auroc_ci'][1])} & -- & --\\\\")
    L.append(r"\bottomrule\end{tabular}")
    (Path(out) / "supp_detection.tex").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE.parent / "results"))
    ap.add_argument("--out", default=str(HERE.parent / "figures"))
    ap.add_argument("--tables", default=str(HERE.parent / "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref, lab = load(a.results, "zoo_reference.json"), load(a.results, "label_analysis.json")
    split = load(a.results, "split_zoo.json")
    teaser(split, ref, lab, a.out)
    labels_fig(ref, lab, a.out)
    dose_fig(ref, a.out)
    tables(ref, split, None, None, a.tables)
    fus = load(a.results, "fusion_late.json") if (Path(a.results) / "fusion_late.json").exists() else None
    ftp = Path(a.results) / "finetune_reference.json"
    supp_tables(ref, split, lab, fus, load(a.results, ftp.name) if ftp.exists() else None, a.tables)
    print("[figures] wrote teaser, labels figure, zoo_reference.tex")


if __name__ == "__main__":
    main()
