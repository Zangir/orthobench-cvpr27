# OrthoBench

> This folder is the OrthoBench data release (protocol, splits and aggregate results). It is
> hosted next to the code during the anonymous review period and moves to a dataset hub on
> publication.

**Status: under review** (anonymous CVPR 2027 submission).

## Summary

OrthoBench is an expert-audited multimodal benchmark for orthodontic diagnosis. It links
intraoral photograph series, panoramic radiographs (OPG) and cephalograms of each patient
to a patient-level Angle class: 8,299 images of 962 patients
(6,935 photographs, 943 OPGs, 421 cephalograms;
285 patients have all three). Two expert reviewers and a senior
adjudicator re-labelled 240 patients. In the primary audit
the adjudicated label differs from the source label for 75 of
221 classifiable patients (33.9%). The
benchmark scores every model against both label versions under a frozen protocol.

**This repository contains no images and no patient-level records.**

## Contents

| Path | What it is |
|---|---|
| `protocol/cvpr27_plan.json` | Frozen analysis plan, written before any score was computed, with the dated amendment of 2026-11-14 |
| `protocol/evaluation_card.md` | What every reported OrthoBench score must state |
| `protocol/split_protocol.md` | How the patient-level splits and the audit-held-out test set are defined |
| `leaderboard/reference_arm.csv` | 33 entries scored against the source and the adjudicated label on 189 audit-held-out patients, with bootstrap intervals |
| `leaderboard/split_arm.csv` | 30 frozen entries under a patient-level and an image-level split, per seed |
| `leaderboard/radiograph_tracks.csv` | OPG and cephalogram tracks (linear head) |
| `leaderboard/expert_reference.csv` | Expert reviewers and the adjudicated label scored against the source label |
| `results/*.json` | Aggregate result files behind every table and figure of the paper (no identifiers) |
| `LICENSE.md` | Terms for this repository |

## How to obtain the data

The full de-identified dataset (images, labels, splits and site, device and demographic
metadata) will be released publicly under CC BY-NC 4.0 on a public dataset host with the
camera-ready version, after two steps: completion of the manual pixel review for identifiers,
including non-Latin text, and the data custodian's written permission. Until then this folder
holds the protocol and the aggregate results. The code to reproduce every experiment and to
rebuild all tables from the aggregate files here is in the companion code repository.

## Splits

- **Reference arm.** All audited patients are withheld from fitting. Patients in any
  cross-patient perceptual-hash candidate pair (Hamming distance at most 6, any record
  type) are quarantined from fitting and scoring. Models train on the source labels of the
  remaining unaudited patients and predict once for 189 classifiable,
  audit-held-out patients, which are scored against both label versions.
- **Split arm.** The full intraoral cohort is split by image and by patient (class-stratified,
  test fraction 0.3, five seeds) to measure leakage.

Patient-keyed split definitions will be part of the public dataset release at camera-ready.

## Licence and terms

- Clinical images and case-level records: released publicly under CC BY-NC 4.0 on a public
  dataset host with the camera-ready version, after the manual pixel review (including
  non-Latin text) and the data custodian's written permission.
- Protocol and aggregate results in this repository: released under CC BY-NC 4.0
  (see `LICENSE.md`); do not attempt to re-identify any patient.
- Code: MIT, in the companion code repository.

## Citation

```bibtex
@misc{orthobench2027,
  title = {OrthoBench: An Expert-Audited Multimodal Benchmark for Orthodontic Diagnosis},
  year  = {2026},
  note  = {Anonymous CVPR 2027 submission}
}
```

## Contact

Contact details will be added after the anonymous review period.
