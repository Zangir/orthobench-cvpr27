#!/usr/bin/env python3
"""Render the post-plan review tables (supplementary Sec. L) from results/*.json.

  python scripts/render_review.py   -> ../../paper/sections/generated/review_*.tex (only for results that exist)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT.parents[1] / "paper/sections/generated"


def f(x):
    v = f"{x:.3f}"
    return f"${v}$" if v.startswith("-") else v


def ci(c):
    return f"{f(c[0])} to {f(c[1])}"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def fusion(d):
    names = {"photos": "Photographs only", "concat": "Concatenation", "stack": "Stacking (learned)",
             "weighted": "Modality weights (learned)"}
    feats = {"rad": "RAD-DINO", "dinov2": "DINOv2-S"}
    rows = []
    for fs in ("rad", "dinov2"):
        r = d["feature_sets"][fs]
        for k, nm in names.items():
            m = r["methods"][k]
            dc = r["differences"].get(f"{k}_minus_concat")
            dp = r["differences"].get(f"{k}_minus_photos")
            rows.append(" & ".join([feats[fs] if k == "photos" else "", nm, f"{f(m['mean'])} $\\pm$ {f(m['sd'])}", ci(m["ci"]),
                                    f"{f(dc['mean'])} ({ci(dc['ci'])}) & {dc['splits_won']}/10" if dc else "-- & --",
                                    f"{f(dp['mean'])} ({ci(dp['ci'])})" if dp else "--"]) + "\\\\")
        if fs == "rad":
            rows.append("\\midrule")
    return ("\\begin{tabular}{@{}llccccc@{}}\\toprule\n"
            "Radiographs & Fusion & macro-F1 (mean $\\pm$ SD) & $95\\%$ CI & $-$ concat.\\ ($95\\%$ CI) & splits won & $-$ photos ($95\\%$ CI)\\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def abmil(d):
    rows = []
    for enc in ("DINOv2-S", "SAM ViT-B"):
        e = d["entries"].get(enc)
        if not e or not e.get("complete"):
            continue
        s, a = e["source"], e["adjudicated"]
        rows.append(f"{enc} linear probe & {f(s['probe_macro_f1'])} ({ci(s['probe_macro_f1_ci'])}) & -- & "
                    f"{f(a['probe_macro_f1'])} ({ci(a['probe_macro_f1_ci'])}) & --\\\\")
        rows.append(f"{enc} attention-MIL & {f(s['macro_f1'])} ({ci(s['macro_f1_ci'])}) & {f(s['mil_minus_probe'])} ({ci(s['mil_minus_probe_ci'])}) & "
                    f"{f(a['macro_f1'])} ({ci(a['macro_f1_ci'])}) & {f(a['mil_minus_probe'])} ({ci(a['mil_minus_probe_ci'])})\\\\")
    return ("\\begin{tabular}{@{}lcccc@{}}\\toprule\n"
            " & \\multicolumn{2}{c}{Source label} & \\multicolumn{2}{c}{Adjudicated label}\\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
            "Entry & macro-F1 ($95\\%$ CI) & MIL $-$ probe ($95\\%$ CI) & macro-F1 ($95\\%$ CI) & MIL $-$ probe ($95\\%$ CI)\\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def resnet(d):
    rows = []
    for k, nm in (("protocol", "Protocol (backbone lr $10^{-4}$)"), ("warmup", "Post-plan (lr $10^{-5}$, warmup)")):
        r = d[k]
        for s in r["per_seed"]:
            c = s["pred_class_counts_I_II_III"]
            rows.append(f"{nm if s['seed'] == 0 else ''} & {s['seed']} & {f(s['manifest_macro_f1'])} & {f(s['final_macro_f1'])} & "
                        f"{f(s['final_balanced_accuracy'])} & {c[0]}/{c[1]}/{c[2]}\\\\")
        rows.append(f" & mean & {f(r['source']['macro_f1'])} ({ci(r['source']['macro_f1_ci'])}) & "
                    f"{f(r['adjudicated']['macro_f1'])} ({ci(r['adjudicated']['macro_f1_ci'])}) & -- & --\\\\")
        if k == "protocol":
            rows.append("\\midrule")
    return ("\\begin{tabular}{@{}llcccc@{}}\\toprule\n"
            "Recipe & Seed & Source F1 & Expert F1 & Expert bal.\\ acc. & Predicted I/II/III\\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")


def main():
    for res, fn, out in (("review_learned_fusion.json", fusion, "review_fusion.tex"),
                         ("review_abmil.json", abmil, "review_abmil.tex"),
                         ("review_resnet50_warmup.json", resnet, "review_resnet50.tex")):
        d = load(res)
        if d is None:
            print(f"[skip] {res} missing"); continue
        (OUT / out).write_text(fn(d))
        print(f"[rendered] sections/generated/{out}")


if __name__ == "__main__":
    main()
