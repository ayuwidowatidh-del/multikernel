"""Rescale the RRUFF test set (Salgado et al., 2023) to max = 100.

  python prepare_rruff.py --x data/rruff/X_rruff_test.npy --y data/rruff/y_rruff_test.npy

The patterns are already on the 5-90 deg grid (8500 points). The output is saved
to config.X_RRUFF and config.Y_RRUFF.
"""
import argparse
import os

import numpy as np

import config as C


def main(x_path, y_path):
    Xr = np.load(x_path).astype(np.float32)
    yr = np.load(y_path).astype(np.int64)
    assert Xr.shape[1] == C.INPUT_LEN, f"length {Xr.shape[1]} != {C.INPUT_LEN}"

    peak = Xr.max(axis=1, keepdims=True)
    assert (peak > 0).all(), "empty pattern found"
    Xr = (Xr / peak * C.SCALE_CONST).astype(np.float32)

    os.makedirs(C.OUT_ROOT, exist_ok=True)
    np.save(C.X_RRUFF, Xr)
    np.save(C.Y_RRUFF, yr)
    print("saved", C.X_RRUFF)

    # compare with the synthetic training data
    Xs = np.load(C.X_TRAIN, mmap_mode="r")
    s = np.asarray(Xs[:200], dtype=np.float32)
    print(f"RRUFF  {Xr.shape} median_max {np.median(Xr.max(1)):7.2f} mean {Xr.mean():6.2f} "
          f"frac_zero {(Xr == 0).mean():.3f}")
    print(f"D4     {Xs.shape} median_max {np.median(s.max(1)):7.2f} mean {s.mean():6.2f} "
          f"frac_zero {(s == 0).mean():.3f}")

    present = sorted(np.unique(yr).tolist())
    missing = [k for k in range(C.NUM_CLASSES) if k not in present]
    print("\nRRUFF classes:", {C.LABEL_NAMES[k]: int((yr == k).sum()) for k in present})
    if missing:
        print("missing:", [C.LABEL_NAMES[k] for k in missing],
              "(macro-F1 on RRUFF uses the classes present)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default=os.path.join("data", "rruff", "X_rruff_test.npy"))
    ap.add_argument("--y", default=os.path.join("data", "rruff", "y_rruff_test.npy"))
    a = ap.parse_args()
    main(a.x, a.y)
