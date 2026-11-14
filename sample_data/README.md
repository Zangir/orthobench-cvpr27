# Synthetic sample data

These are **synthetic, license-clean placeholder images and manifests** with the
same schema as the restricted OrthoBench cohort. They contain **no real patient
data** and exist only so the pipeline (`audit`, `extract`, `dedup`) and the smoke
test can run for anyone without access to the restricted clinical dataset.

Regenerate with:

```bash
python sample_data/make_sample.py
```

The `smoke` command (`python run_experiment.py smoke`) does not even need these
files — it generates synthetic feature vectors in memory. Use this folder to
exercise the manifest/feature/dedup code paths end-to-end on disk.
