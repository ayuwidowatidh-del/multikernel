"""Tables and quick plots from the files written by train.py and analyze.py.

  python summarize_results.py

- Table S2 (training summary per run)       -> table_S2.csv
- validation F1 vs RRUFF F1 per run
- macro-F1 under broadening (mean / std, and per seed at 1.0 deg)
- branch probe plot                          -> branch_probe.png
- broadening plot                            -> robustness_broadening.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config as C


def load_runs():
    rows = []
    for arch in C.MODELS:
        for seed in C.SEEDS:
            p = os.path.join(C.OUT_ROOT, f"{arch}_seed{seed}", "run_info.json")
            if os.path.exists(p):
                rows.append(json.load(open(p)))
    return rows


def table_s2(runs):
    df = pd.DataFrame([{"Model": r["arch"], "Seed": r["seed"],
                        "Epochs": r["epochs_run"],
                        "Best val macro-F1": round(r["best_val_f1"], 3),
                        "Time (min)": r["minutes"],
                        "CNN params": r["params"]["cnn"],
                        "Total params": r["params"]["total"]} for r in runs])
    df = df.sort_values(["Model", "Seed"])
    print("Table S2")
    print(df.to_string(index=False))
    print("\nPyTorch:", runs[-1]["torch"], "| device:", runs[-1]["device"])
    df.to_csv(os.path.join(C.OUT_ROOT, "table_S2.csv"), index=False)


def val_vs_rruff(runs):
    if not os.path.exists(C.Y_RRUFF):
        return
    y = np.load(C.Y_RRUFF)
    labels = sorted(np.unique(y).tolist())
    rows = []
    for r in runs:
        f = os.path.join(C.OUT_ROOT, f"pred_rruff_{r['arch']}_s{r['seed']}.npy")
        if not os.path.exists(f):
            continue
        pred = np.load(f)
        rows.append({"arch": r["arch"], "seed": r["seed"], "epochs": r["epochs_run"],
                     "val_f1": r["best_val_f1"],
                     "rruff_f1": f1_score(y, pred, average="macro", labels=labels,
                                          zero_division=0)})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["arch", "seed"])
    print("\nValidation F1 vs RRUFF F1")
    print(df.to_string(index=False))
    print("\ncorrelation val_f1 vs rruff_f1:", round(df.val_f1.corr(df.rruff_f1), 3))
    print("correlation epochs vs rruff_f1:", round(df.epochs.corr(df.rruff_f1), 3))


def broadening():
    f = os.path.join(C.OUT_ROOT, "robustness.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    b = df[df.condition.str.startswith("broaden")].copy()
    b["fwhm"] = b.condition.str.replace("broaden_", "").astype(float)
    print("\nMacro-F1 under broadening")
    print(b.pivot_table(index="fwhm", columns="arch", values="macro_f1",
                        aggfunc=["mean", "std"]).to_string())
    print("\nbroaden_1.0 per seed")
    print(b[b.fwhm == 1.0].pivot_table(index="seed", columns="arch",
                                       values="macro_f1").to_string())

    ax = b.pivot_table(index="fwhm", columns="arch", values="macro_f1").plot(
        marker="o", figsize=(7, 4))
    ax.set_xlabel("Additional FWHM (deg)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Robustness to peak broadening")
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT_ROOT, "robustness_broadening.png"), dpi=150)
    plt.close()


def probe_plot():
    f = os.path.join(C.OUT_ROOT, "branch_probe.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    piv = df.pivot_table(index="fwhm_deg", columns="branch", values="mean_act")
    piv = piv / piv.max()
    ax = piv.plot(marker="o", logx=True, figsize=(7, 4))
    ax.set_xlabel("Probe peak FWHM (deg)")
    ax.set_ylabel("Mean activation (normalised)")
    ax.set_title("Stage-1 branch response")
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUT_ROOT, "branch_probe.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    runs = load_runs()
    table_s2(runs)
    val_vs_rruff(runs)
    broadening()
    probe_plot()
