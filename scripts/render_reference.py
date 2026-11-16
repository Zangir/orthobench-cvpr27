#!/usr/bin/env python3
"""Render audited numbers into TeX and figures only after JSON results exist."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from paper_style import style, comparison_matrix, save

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1] / "paper"
OUT = PAPER / "sections/generated"
COLORS = ["#0072B2", "#E69F00", "#009E73", "#999999"]


def interval(v):
    # "to" keeps negative bounds readable (QA 2026-11-16)
    return f"{v[0]:.3f} to {v[1]:.3f}"


def table_file(name, columns, header, rows):
    """Include a complete tabular to avoid LaTeX file hooks inside an alignment."""
    text = r"\begin{tabular}{" + columns + "}" + r"\toprule" + "\n"
    text += header + r"\\\midrule" + "\n" + "\n".join(rows)
    text += "\n" + r"\bottomrule\end{tabular}" + "\n"
    (OUT / name).write_text(text)


def main():
    style()
    OUT.mkdir(exist_ok=True)
    a = json.loads((ROOT / "results/reference_audit.json").read_text())
    e = json.loads((ROOT / "results/reference_evaluation.json").read_text())
    rows = []
    for key, label in [("primary", "Primary audit"), ("pilot", "Earlier pilot"), ("combined", "Combined")]:
        v = a[key]; g = v["reviewer_agreement"]
        rows.append(f"{label} & {v['n']} & {100*g['raw_agreement']:.1f} & {g['kappa']:.3f} & {interval(g['kappa_ci'])} & {v['senior_n']} & {v['n_excluded']} " + r"\\")
    table_file("audit_batches.tex", "@{}lrrrrrr@{}", r"Batch & Patients & Agreement (\%) & $\kappa$ & $95\%$ CI & Senior & Excluded", rows)
    rows = []
    for model, label in [("resnet50", "ResNet-50"), ("dinov2", "DINOv2 S/14"), ("dinov2_large", "DINOv2 L/14"),
                         ("siglip2", "SigLIP-2 base"), ("biomedclip", "BiomedCLIP")]:
        v = e["metrics"][model + "/intraoral"]
        rows.append(f"{label} & {v['manifest']['macro_f1']:.3f} & {v['consensus']['macro_f1']:.3f} & {interval(v['consensus']['macro_f1_ci'])} & {v['delta_consensus_minus_manifest']:+.3f} " + r"\\")
    table_file("reference_models.tex", "@{}lrrrr@{}", r"Backbone & Manifest & Final reference & Final-reference $95\%$ CI & $\Delta$", rows)
    rows = []
    for name, v in e["metrics"].items():
        if "/majority" in name and name != "dinov2/intraoral/majority":
            continue
        display = name.replace("_", r"\_").replace("cephalogram", "ceph.")
        rows.append(f"{display} & {v['n_train_patients']} & {v['n_test']} & {v['manifest']['macro_f1']:.3f} & {v['consensus']['macro_f1']:.3f} & {interval(v['consensus']['macro_f1_ci'])} & {interval(v['delta_ci'])} " + r"\\")
    table_file("reference_full.tex", "@{}lrrrrll@{}", r"Arm & Train & Test & Manifest & Final & Final $95\%$ CI & Change $95\%$ CI", rows)

    # Reviewer matrices and final label distribution; fonts sized for full text width.
    fig, axes = plt.subplots(1, 3, figsize=(10., 3.), gridspec_kw={"width_ratios": [1.1, 1.1, 1.]})
    labels = ["I", "II", "III", "N/A"]
    for ax, m, xlabel, ylabel in [(axes[0], np.array(a["primary"]["reviewer_agreement"]["matrix"]), "Reviewer 2", "Reviewer 1"),
                                 (axes[1], np.array(a["combined"]["manifest_vs_final_matrix"]), "Final reference", "Manifest")]:
        ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
        for (i, j), val in np.ndenumerate(m):
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val > .5*m.max() else "#222", fontsize=11)
        ax.set_xticks(range(4), labels); ax.set_yticks(range(m.shape[0]), labels[:m.shape[0]])
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(False)
    ax = axes[2];v=a["combined"];counts=[v["final_labels"].get(k, 0) for k in ["Class I","Class II","Class III","Excluded (not assessable)"]]
    bars=ax.bar(range(4),counts,color=COLORS)
    for bar,val in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,val+3,str(val),ha="center",fontsize=11)
    ax.set_xticks(range(4),["I","II","III","Excl."]);ax.set_ylabel("Reviewed patients");ax.set_ylim(0,max(counts)*1.2)
    fig.tight_layout(w_pad=1.4)
    save(fig,str(ROOT / "figures/fig11_reference"))

    # Compact teaser: linked data, measured reviewer agreement, measured split effect.
    ds=json.loads((ROOT/'results/dataset_stats.json').read_text())
    lk=json.loads((ROOT/'results/leakage.json').read_text())
    ft=json.loads((ROOT/'results/finetune.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(10.,2.55),gridspec_kw={"width_ratios":[1.,1.05,1.15]})
    ax=axes[0];ax.axis('off')
    ax.text(0,.93,"Patient-linked cohort",weight='bold',fontsize=12)
    ax.text(0,.70,f"{ds['n_patients']} patients  |  {ds['n_images']:,} images",fontsize=11)
    ax.text(0,.47,"Intraoral  +  OPG  +  Cephalogram",fontsize=10)
    ax.text(0,.24,f"{a['combined']['n']} double-reviewed cases",fontsize=11,weight='bold',color=COLORS[0])
    ax.text(0,.02,f"{a['combined']['n_assessable']} classifiable  |  {a['combined']['n_excluded']} excluded",fontsize=10)
    ax=axes[1];v=a['primary'];g=v['reviewer_agreement']
    ax.barh([1,0],[100*g['raw_agreement'],100*v['changed_fraction']],color=[COLORS[0],'#D55E00'],height=.47)
    for y,value in [(1,100*g['raw_agreement']),(0,100*v['changed_fraction'])]:ax.text(value+2,y,f"{value:.1f}%",va='center',fontsize=11,weight='bold')
    ax.set_yticks([1,0],["Reviewer\nagreement","Reference\nlabel changed"]);ax.set_xlim(0,100);ax.set_xlabel("Primary audit (%)")
    ax=axes[2];names=['Linear','Fine-tuned','1-NN'];keys=['logreg',None,'knn1']
    vals=[];errs=[]
    sg=json.loads((ROOT/'results/significance.json').read_text())['arms']
    for key in ['linear_probe','finetuned','knn1']:
        if key not in sg and key=='finetuned':key=next(k for k in sg if 'fine' in k)
        vals.append(sg[key]['mean_diff']);errs.append(sg[key]['sd_diff'])
    ax.bar(range(3),vals,yerr=errs,color=['#0072B2','#333333','#D55E00'],capsize=3)
    for x,y,err in zip(range(3),vals,errs):ax.text(x,y+err+.006,f"+{y:.3f}",ha='center',fontsize=10,weight='bold')
    ax.set_xticks(range(3),names);ax.set_ylabel("Image-split F1 inflation");ax.set_ylim(0,.21)
    fig.tight_layout(w_pad=1.3)
    save(fig,str(ROOT/'figures/fig0_teaser'))

    cm=json.loads((ROOT/'results/comparison_matrix.json').read_text())
    fig,ax=plt.subplots(figsize=(9.,2.6))
    cells=[["yes" if r[k] is True else "no" if r[k] is False else "n/r" for k in cm['capability_columns']] for r in cm['datasets']]
    comparison_matrix(ax,[r['name'] for r in cm['datasets']],["Linked\nIO+OPG+ceph","Angle\nlabels","Patient split\ndocumented"],cells,ours_row=6,quant_col=[r['n_images'] or 'n/r' for r in cm['datasets']],quant_label='Images')
    for label in ax.texts:
        label.set_fontsize(max(label.get_fontsize(), 11))
    save(fig,str(ROOT/'figures/fig5_comparison'))

    # Size this chart for its actual one-column placement, keeping labels readable.
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    groups = [lk['probes'][k] for k in ('logreg', 'knn5', 'knn1')]
    groups.append(ft['regimes'])
    xs = np.arange(len(groups))
    for offset, arm, color, label in [(-.19, 'image_level', '#D55E00', 'Image split'),
                                     (.19, 'patient_level', '#009E73', 'Patient split')]:
        means = [v[arm]['macro_f1']['mean'] for v in groups]
        errors = [v[arm]['macro_f1']['std'] for v in groups]
        ax.bar(xs + offset, means, width=.36, yerr=errors, color=color, capsize=2, label=label)
        for x, mean, err in zip(xs + offset, means, errors):
            ax.text(x, mean + err + .012, f'{mean:.3f}', ha='center', fontsize=7)
    ax.set_xticks(xs, ['Linear', '5-NN', '1-NN', 'Fine-tuned'], fontsize=7.5)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel('Per-image macro-F1', fontsize=9)
    ax.set_ylim(0, .66)
    ax.legend(loc='upper left', fontsize=7.5, ncols=2, handlelength=1., columnspacing=.8)
    fig.tight_layout(pad=.4)
    save(fig, str(ROOT/'figures/fig1_leakage'))
    catalog = json.loads((ROOT / "scripts/figure_catalog.json").read_text())
    (ROOT / "figures/figures.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print("[reference] TeX tables and figures rendered from persisted JSON")


if __name__ == "__main__":
    main()
