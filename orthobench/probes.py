"""Frozen-feature probes: logistic regression and kNN.

These are deliberately simple, lightweight classifiers on frozen features. The
kNN probe directly exploits the nearest-neighbour shortcut, so under an
image-level split it can retrieve another image of the same patient from the
training set -- which is exactly the leakage we quantify.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def _l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def logreg_probe(Xtr, ytr, Xte, class_weight="balanced", C=1.0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=C, class_weight=class_weight)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))


def knn_probe(Xtr, ytr, Xte, k=1, n_classes=3):
    Xtr, Xte = _l2norm(Xtr.astype(np.float64)), _l2norm(Xte.astype(np.float64))
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(Xtr, ytr)
    # return pseudo-probabilities aligned to classes 0..n_classes-1
    proba = np.zeros((len(Xte), n_classes))
    p = clf.predict_proba(Xte)
    for j, c in enumerate(clf.classes_):
        proba[:, int(c)] = p[:, j]
    return proba


def run_probe(kind, Xtr, ytr, Xte, **kw):
    if kind == "logreg":
        return logreg_probe(Xtr, ytr, Xte,
                            class_weight=kw.get("class_weight", "balanced"),
                            C=kw.get("C", 1.0))
    if kind.startswith("knn"):
        k = int(kw.get("k", 1))
        return knn_probe(Xtr, ytr, Xte, k=k, n_classes=kw.get("n_classes", 3))
    raise ValueError(f"unknown probe {kind}")
