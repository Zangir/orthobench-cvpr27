# OrthoBench split protocol

Implemented in `orthobench/reference_eval.py` (`quarantine_candidates`, `split_masks`) and
`orthobench/zoo.py` (`context`, `predict`) of the code repository.

1. **Audit held out.** Every audited patient (primary audit, pilot, exclusions) is removed
   from fitting.
2. **Quarantine.** 64-bit perceptual hashes are computed for every image. Any patient in a
   cross-patient candidate pair with Hamming distance at most 6, in any record type, is
   removed from both fitting and scoring. This is a candidate screen, not identity
   reconciliation.
3. **Test set.** The classifiable patients of the primary audit that survive quarantine.
4. **Training set.** The remaining unaudited, non-quarantined patients, with source labels.
5. **Scoring.** One prediction per test patient (mean of image probabilities), scored
   against the source label and the adjudicated label with shared bootstrap draws.
6. **Split arm.** On the full intraoral cohort with source labels, image-level and
   patient-level class-stratified splits with test fraction 0.3 and seeds 0-4
   (`orthobench/splits.py`), scored per image.

The patient-keyed membership lists will be released with the public dataset at camera-ready.
