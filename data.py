"""Dataset, scaling and split integrity checks"""
#library
import os
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset
from scipy.signal import find_peaks, peak_widths
import config as C

#intensity scaling
def scale(x):
    if C.SCALE_MODE == "constant":
        return x / C.SCALE_CONST
    lo, hi = x.min(axis=-1, keepdims=True), x.max(axis=-1, keepdims=True)
    return (x - lo) / (hi - lo + 1e-6)

#XRD dataset classes

class XRDDataset(Dataset):
    def __init__(self, x_path, y_path, meta_path=None, transform=None):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path).astype(np.int64)
        self.meta = pd.read_csv(meta_path) if meta_path and os.path.exists(meta_path) else None
        self.transform = transform            
        if len(self.X) != len(self.y):
            raise ValueError(f"X={len(self.X)} vs y={len(self.y)}")

    def __len__(self):
        return len(self.y)

    def structure_ids(self):
        if self.meta is not None and C.STRUCTURE_ID_COL in self.meta.columns:
            return self.meta[C.STRUCTURE_ID_COL].to_numpy()
        print(f"no '{C.STRUCTURE_ID_COL}' column, each pattern is treated as its own structure")
        return np.arange(len(self.y))

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32).reshape(-1)
        if x.shape[0] != C.INPUT_LEN:
            raise ValueError(f"pattern {idx} has length {x.shape[0]}, expected {C.INPUT_LEN}")
        if self.transform is not None:
            x = self.transform(x)
        x = scale(np.ascontiguousarray(x, dtype=np.float32))[None, :]
        return torch.from_numpy(x), torch.tensor(int(self.y[idx]))

#checking split integrity
def check_split_integrity(*meta_paths):
    """Check that no structure appears in more than one split."""
    sets, names = [], []
    for p in meta_paths:
        if not (p and os.path.exists(p)):
            continue
        df = pd.read_csv(p)
        if C.STRUCTURE_ID_COL not in df.columns:
            print(f"SKIP {os.path.basename(p)}: no '{C.STRUCTURE_ID_COL}' column")
            continue
        sets.append(set(df[C.STRUCTURE_ID_COL]))
        names.append(os.path.basename(p))
    leaked = False
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            common = sets[i] & sets[j]
            if common:
                leaked = True
                print(f"LEAK: {len(common)} structures shared by {names[i]} and {names[j]}"
                      f" (e.g. {list(common)[:5]})")
    if sets and not leaked:
        print("Split integrity OK: no structure appears in more than one subset.")
    return not leaked

#class distribution
def class_distribution(meta_path, y_path):
    y = np.load(y_path)
    df = pd.DataFrame({"y":y})
    counts = df["y"].value_counts().sort_index()
    out = pd.DataFrame({"class": [C.LABEL_NAMES[i] for i in counts.index],
                        "patterns": counts.values})
    if meta_path and os.path.exists(meta_path):
        m = pd.read_csv(meta_path)
        if C.STRUCTURE_ID_COL in m.columns:
            m["y"] = y
            out["structures"] = [m[m.y == i][C.STRUCTURE_ID_COL].nunique()
                                             for i in counts.index]
    return out

#Peak FWHM Measurement
def measure_fwhm(x_path, n_samples=500, prominence=0.05, seed=0):
    """FWHM percentiles (deg) of peaks in a random subset of patterns."""
    X = np.load(x_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    widths = []
    for i in idx:
        x = np.asarray(X[i], dtype=float).reshape(-1)
        rng_ = x.max() - x.min()
        if rng_ <= 0:
            continue
        x = (x - x.min()) / rng_
        pk, _ = find_peaks(x, prominence=prominence)
        if len(pk) == 0:
            continue
        widths.extend(peak_widths(x, pk, rel_height=0.5)[0] * C.TWO_THETA_STEP)
    w = np.asarray(widths)
    pct = np.percentile(w, [5, 25, 50, 75, 95]) if len(w) else np.full(5, np.nan)
    return {"n_peaks": len(w), "p5": pct[0], "p25": pct[1], "p50": pct[2],
            "p75": pct[3], "p95": pct[4]}

if __name__ == "__main__":
    check_split_integrity(C.META_TRAIN, C.META_VAL, C.META_TEST)
    print(class_distribution(C.META_TRAIN, C.Y_TRAIN))
    print("FWHM (deg):", measure_fwhm(C.X_TRAIN))