"""Checks before training: split leakage, class counts, peak widths and model size.

  python check_data.py
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths

import config as C
import data as D
import models


def train_fwhm_percentiles(n=500, seed=0):
    """FWHM (deg) of peaks in n random training patterns.
    p95 and p99 are used for the dashed lines in Fig. 3."""
    X = np.load(C.X_TRAIN, mmap_mode="r")
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    w = []
    for i in idx:
        x = np.asarray(X[i], dtype=float).reshape(-1)
        r = x.max() - x.min()
        if r <= 0:
            continue
        x = (x - x.min()) / r
        pk, _ = find_peaks(x, prominence=0.05)
        if len(pk):
            w.extend(peak_widths(x, pk, rel_height=0.5)[0] * C.TWO_THETA_STEP)
    w = np.array(w)
    p = np.percentile(w, [5, 25, 50, 75, 95, 99])
    print(f"n peaks {len(w)}")
    print("p5 %.3f  p25 %.3f  median %.3f  p75 %.3f  p95 %.3f  p99 %.3f deg" % tuple(p))
    print("the 1.0 deg threshold lies at percentile %.1f" % (100 * (w < 1.0).mean()))


def main():
    print("ROOT:", C.ROOT)
    print("SCALE:", C.SCALE_MODE, C.SCALE_CONST, "| STRUCTURE_ID_COL:", C.STRUCTURE_ID_COL)

    # 1. no structure in more than one split
    ok = D.check_split_integrity(C.META_TRAIN, C.META_VAL, C.META_TEST)
    assert ok, "structures leak across splits"

    # 2. class distribution
    print()
    print(D.class_distribution(C.META_TRAIN, C.Y_TRAIN).to_string(index=False))
    print()
    print(D.class_distribution(C.META_TEST, C.Y_TEST).to_string(index=False))

    # 3. peak widths in the training data (used to choose BROADEN_FWHM_DEG)
    print()
    print(D.measure_fwhm(C.X_TRAIN, n_samples=500))
    print()
    train_fwhm_percentiles()

    # 4. model size
    print()
    for arch in C.MODELS:
        m = models.Model(arch)
        print(f"{arch:12s} feature {m.feat_shape}  params {models.count_params(m)}")


if __name__ == "__main__":
    main()
