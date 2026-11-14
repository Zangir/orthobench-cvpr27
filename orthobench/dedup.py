"""Near-duplicate detection via perceptual hashing.

Exact-content hashing (sha256, done at packaging time) removes byte-identical
copies but not visually near-identical images. We compute 64-bit DCT pHashes and
count near-duplicate pairs at a range of Hamming thresholds, then flag the pairs
that cross patient boundaries -- these are the residual leakage risks a
patient-level split alone does not remove.
"""
import numpy as np
from itertools import combinations


def _popcount64(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)) & np.uint64(0x7f)


def hamming_matrix_pairs(phash, max_dist=10, block=2000):
    """Return dict thresh -> list of (i,j) index pairs with Hamming <= thresh.
    Uses blocked XOR+popcount to stay memory-bounded."""
    ph = phash.astype(np.uint64)
    n = len(ph)
    pairs = {t: [] for t in range(max_dist + 1)}
    for a0 in range(0, n, block):
        a1 = min(a0 + block, n)
        A = ph[a0:a1]
        for b0 in range(a0, n, block):
            b1 = min(b0 + block, n)
            B = ph[b0:b1]
            xor = A[:, None] ^ B[None, :]
            dist = _popcount64(xor)
            for ii in range(a1 - a0):
                gi = a0 + ii
                lo = (gi - b0 + 1) if b0 <= gi else 0
                for jj in range(lo, b1 - b0):
                    gj = b0 + jj
                    d = int(dist[ii, jj])
                    if d <= max_dist:
                        pairs[d].append((gi, gj))
    return pairs


def neardup_report(phash, patient_id, max_dist=10):
    pairs = hamming_matrix_pairs(phash, max_dist=max_dist)
    report = {"max_dist": max_dist, "n_images": int(len(phash)), "by_threshold": {}}
    cum, cum_cross = 0, 0
    for t in range(max_dist + 1):
        at_t = pairs[t]
        cross = sum(1 for i, j in at_t if patient_id[i] != patient_id[j])
        cum += len(at_t); cum_cross += cross
        report["by_threshold"][str(t)] = {
            "pairs_at_exact_dist": len(at_t),
            "cross_patient_at_exact_dist": cross,
            "cum_pairs_leq": cum,
            "cum_cross_patient_leq": cum_cross}
    # nodes involved in any near-dup (<= a moderate threshold of 6)
    involved = set()
    involved_cross = set()
    for t in range(0, 7):
        for i, j in pairs[t]:
            involved.add(i); involved.add(j)
            if patient_id[i] != patient_id[j]:
                involved_cross.add(i); involved_cross.add(j)
    report["images_involved_leq6"] = len(involved)
    report["images_in_cross_patient_neardup_leq6"] = len(involved_cross)
    return report
