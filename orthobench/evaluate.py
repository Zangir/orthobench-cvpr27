"""Metrics and patient-bootstrap confidence intervals."""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score


def image_metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_f1": [float(x) for x in
                         f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])],
    }


def aggregate_to_patient(prob, test_pid, patient_label):
    """Mean predicted probability across a patient's images -> patient prediction."""
    at, ap = [], []
    for p in np.unique(test_pid):
        m = test_pid == p
        at.append(patient_label[p])
        ap.append(int(prob[m].mean(0).argmax()))
    return np.array(at), np.array(ap)


def patient_bootstrap_ci(true, pred, n_boot=1000, seed=0, alpha=0.05):
    """Bootstrap CI over patients for macro-F1 and accuracy."""
    rng = np.random.RandomState(seed)
    true = np.asarray(true); pred = np.asarray(pred)
    n = len(true)
    f1s, accs = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        # guard: need >=2 classes present for macro-F1 to be meaningful
        f1s.append(f1_score(true[idx], pred[idx], average="macro"))
        accs.append(accuracy_score(true[idx], pred[idx]))
    def ci(a):
        return [float(np.percentile(a, 100 * alpha / 2)),
                float(np.percentile(a, 100 * (1 - alpha / 2)))]
    return {"macro_f1_ci": ci(f1s), "accuracy_ci": ci(accs),
            "macro_f1_mean": float(np.mean(f1s)), "accuracy_mean": float(np.mean(accs))}


def per_class_bootstrap_ci(true, pred, n_classes=3, n_boot=1000, seed=0, alpha=0.05):
    """Bootstrap CI over patients for EACH class F1 separately.

    Macro-F1 intervals hide the fact that almost all of the uncertainty sits on the
    minority class, so the per-class intervals are what a reader needs to judge a
    Class-III claim.
    """
    rng = np.random.RandomState(seed)
    true = np.asarray(true); pred = np.asarray(pred)
    n = len(true)
    cols = [[] for _ in range(n_classes)]
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        f = f1_score(true[idx], pred[idx], average=None,
                     labels=list(range(n_classes)), zero_division=0)
        for c in range(n_classes):
            cols[c].append(float(f[c]))
    return [{"class": c,
             "f1_mean": float(np.mean(cols[c])),
             "f1_ci": [float(np.percentile(cols[c], 100 * alpha / 2)),
                       float(np.percentile(cols[c], 100 * (1 - alpha / 2)))]}
            for c in range(n_classes)]


def confusion(true, pred, n_classes=3):
    """Plain confusion matrix as nested lists (rows = true, cols = predicted)."""
    true = np.asarray(true); pred = np.asarray(pred)
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true, pred):
        m[int(t), int(p)] += 1
    return m.tolist()


def mean_std(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std())}
