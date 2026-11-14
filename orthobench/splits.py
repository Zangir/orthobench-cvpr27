"""Split protocols: patient-level (correct) vs image-level (naive/leaky).

Both are class-stratified with a fixed test fraction. The patient-level split
guarantees all images of a patient fall on one side; the image-level split
shuffles images regardless of patient, which leaks patient identity across the
train/test boundary. Seeds are paired across protocols for a fair comparison.
"""
import numpy as np


def patient_labels(patient_id, label_numeric):
    patients = np.array(sorted(set(patient_id.tolist())))
    plab = {p: int(label_numeric[patient_id == p][0]) for p in patients}
    return patients, plab


def patient_level_split(patient_id, label_numeric, seed, test_frac=0.3, n_classes=3):
    rng = np.random.RandomState(seed)
    patients, plab = patient_labels(patient_id, label_numeric)
    pl = np.array([plab[p] for p in patients])
    tr_p, te_p = set(), set()
    for c in range(n_classes):
        cps = patients[pl == c].copy()
        rng.shuffle(cps)
        k = int(round(len(cps) * (1 - test_frac)))
        tr_p |= set(cps[:k].tolist())
        te_p |= set(cps[k:].tolist())
    tr = np.array([i for i in range(len(patient_id)) if patient_id[i] in tr_p])
    te = np.array([i for i in range(len(patient_id)) if patient_id[i] in te_p])
    return tr, te


def image_level_split(label_numeric, seed, test_frac=0.3, n_classes=3):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in range(n_classes):
        idx = np.where(label_numeric == c)[0].copy()
        rng.shuffle(idx)
        k = int(round(len(idx) * (1 - test_frac)))
        tr += idx[:k].tolist()
        te += idx[k:].tolist()
    return np.array(tr), np.array(te)


def few_shot_patient_split(patient_id, label_numeric, seed, n_per_class,
                           test_frac=0.3, n_classes=3):
    """Patient-level split but cap TRAIN patients at n_per_class per class."""
    rng = np.random.RandomState(seed)
    patients, plab = patient_labels(patient_id, label_numeric)
    pl = np.array([plab[p] for p in patients])
    tr_p, te_p = set(), set()
    for c in range(n_classes):
        cps = patients[pl == c].copy()
        rng.shuffle(cps)
        k = int(round(len(cps) * (1 - test_frac)))
        tr_p |= set(cps[:k][:n_per_class].tolist())
        te_p |= set(cps[k:].tolist())
    tr = np.array([i for i in range(len(patient_id)) if patient_id[i] in tr_p])
    te = np.array([i for i in range(len(patient_id)) if patient_id[i] in te_p])
    return tr, te
