# OrthoBench evaluation card

Every reported OrthoBench score states:

| Item | OrthoBench setting |
|---|---|
| Split unit | patient (an image-level split is reported only as a leakage diagnostic) |
| Label version | source and expert-adjudicated, both reported |
| Training pool | unaudited patients only; the audit is held out |
| Leakage screen | cross-patient perceptual-hash quarantine (Hamming distance at most 6) |
| Scoring unit | patient (mean of image probabilities, then argmax) |
| Metric | patient macro-F1 (primary), balanced accuracy (secondary) |
| Uncertainty | 2,000 shared patient-bootstrap draws; paired shifts between label versions |
| Ceiling | expert reviewers and the adjudicated label scored against the source label |
| Release | paired per-patient predictions, with the public dataset release |
