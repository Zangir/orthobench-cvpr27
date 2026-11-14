# OrthoBench code (anonymous review snapshot)

Code for "OrthoBench: An Expert-Audited Multimodal Benchmark for Orthodontic Diagnosis"
(anonymous CVPR 2027 submission). **Status: under review.**

This snapshot contains code, the frozen analysis plan and aggregate results. It contains no
clinical images, no patient identifiers and no patient-level records.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn scipy Pillow matplotlib openpyxl PyMuPDF
# feature extraction and fine-tuning (approved data only): pip install -r requirements.txt
```

## Reproduce the paper's tables from the released files (no clinical data needed)

```bash
python scripts/make_figures_exemplar.py --paper out
# -> out/sections/generated/ex_tab1.tex (Tab. 1), ex_tab2.tex (Tab. 2),
#    ex_supp_rank.tex, ex_supp_expert.tex, ex_supp_featdist.tex (supplement Sec. F)
# -> out/figures/ex_fig3..6 and ex_fig_comparison (Figs. 3-6, supplement Fig. S2)
python scripts/make_figures_cvpr27.py --results results --out out/figures --tables out/sections/generated
# -> full leaderboards of supplement Sec. F
```

Figures 1 and 2 of the paper show redacted patient thumbnails and are not reproducible from
the public files; the script skips them.

## Smoke test (synthetic data)

```bash
python sample_data/make_sample.py      # synthetic noise images, no patients
python run_experiment.py smoke
python -m unittest discover -s tests -v
```

## Re-run the experiments (approved data-use agreement only)

With the restricted data root `DR` and the audit workbook `WB`:

```bash
for M in opg cephalogram intraoral; do
  python scripts/extract_zoo.py --data-root "$DR" --modality $M \
    --backbones resnet18,resnet50,convnext_tiny,convnext_large,dinov2,dinov2_base,dinov2_large,rad_dino,siglip2,clip_b16,biomedclip
done
python scripts/run_cvpr27.py split
python scripts/run_cvpr27.py predict --workbook "$WB"
for A in dinov2 resnet50 convnext_tiny; do
  python scripts/finetune_reference.py --workbook "$WB" --arch $A --seeds 0 1 2
done
python scripts/run_cvpr27.py analyze --workbook "$WB"
python scripts/exemplar_checks.py --workbook "$WB"     # 2026-11-14 amendment checks
```

Restricted images are read in place; feature caches and per-patient predictions stay in the
ignored `data/` directory. `cvpr27_plan.json` is the frozen analysis plan.

## Layout

- `orthobench/`: manifests, splits, encoders, probes, metrics, audit recomputation,
  reference evaluation, encoder zoo, leaderboard statistics, fine-tuning.
- `scripts/`: extraction, experiments, tables and figures.
- `results/`: aggregate JSON outputs (no identifiers).
- `sample_data/`: generator for synthetic smoke fixtures.
- `tests/`: unit tests.

## License

Code: MIT (`LICENSE`). Clinical data are not covered by this licence.
