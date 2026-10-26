"""Paired bootstrap of the macro-F1 difference under each perturbation.

  python robustness_stats.py

Same cluster bootstrap as `analyze.py --what stats`, but for the perturbed
test set. Saves robustness_stats.csv in OUT_ROOT.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import analyze
import config as C

CONDS = {
    "clean":        None,
    "broaden_0.5":  analyze.broaden(0.5),
    "broaden_1.0":  analyze.broaden(1.0),
    "broaden_1.5":  analyze.broaden(1.5),
    "broaden_2.0":  analyze.broaden(2.0),
    "noise_snr5":   analyze.add_noise(5),
}
PAIRS = [("npcnn_valid", "mknpcnn"), ("npcnn", "mknpcnn")]
N_BOOT = 10000


def macro(y, p, labels):
    return f1_score(y, p, average="macro", labels=labels, zero_division=0)


def boot(y, pa, pb, clusters, seed, labels):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_of = {c: np.where(clusters == c)[0] for c in uniq}
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draw = rng.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([idx_of[c] for c in draw])
        d[i] = macro(y[ix], pb[ix], labels) - macro(y[ix], pa[ix], labels)
    obs = macro(y, pb, labels) - macro(y, pa, labels)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi


def main():
    rows, preds = [], {}
    for cond, tf in CONDS.items():
        ds = analyze.dataset("test", tf)
        y = ds.y
        labels = sorted(np.unique(y).tolist())
        clusters = ds.structure_ids()
        for arch in C.MODELS:
            for seed in C.SEEDS:
                r = analyze.predict(analyze.load_model(arch, seed), ds)
                preds[(cond, arch, seed)] = r["pred"]
        for base, cand in PAIRS:
            for seed in C.SEEDS:
                obs, lo, hi = boot(y, preds[(cond, base, seed)],
                                   preds[(cond, cand, seed)], clusters, seed, labels)
                rows.append({"condition": cond, "comparison": f"{cand} - {base}",
                             "seed": seed, "diff": obs, "ci_low": lo, "ci_high": hi,
                             "excludes_zero": lo > 0 or hi < 0})
        print("done:", cond, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_ROOT, "robustness_stats.csv"), index=False)
    print(df.to_string(index=False))
    for (cond, cmp_), s in df.groupby(["condition", "comparison"]):
        print(f"{cond:13s} {cmp_:26s} consistent = "
              f"{bool((s['diff'] > 0).all() and s['excludes_zero'].all())}"
              f"  {[round(v, 4) for v in s['diff']]}")


if __name__ == "__main__":
    main()
