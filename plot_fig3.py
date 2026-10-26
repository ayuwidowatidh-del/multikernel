"""Figure 3: macro-F1 under peak broadening, noise and zero-point shift.

  python plot_fig3.py

Reads robustness.csv and saves fig3_robustness.png/.pdf in OUT_ROOT.
The dashed lines in the first panel are the p95 and p99 of the training FWHM.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

# p95 / p99 of the training FWHM (from check_data.py)
TRAIN_P95, TRAIN_P99 = 0.80, 1.44

LABELS = {"npcnn_valid": "NPCNN", "npcnn": "NPCNN (padding-matched)",
          "mknpcnn": "MK-NPCNN"}
COLORS = {"npcnn_valid": "#4363d8", "npcnn": "#9aa0a6", "mknpcnn": "#e6194b"}
ORDER = ["npcnn_valid", "npcnn", "mknpcnn"]


def subset(df, prefix, cast=float):
    d = df[df.condition.str.startswith(prefix)].copy()
    d["x"] = d.condition.str.replace(prefix, "", regex=False).astype(cast)
    return d


def panel(ax, d, xlabel, clean, logx=False, invert=False):
    for arch in ORDER:
        s = d[d.arch == arch]
        if s.empty:
            continue
        g = s.groupby("x").macro_f1
        x = np.array(sorted(s.x.unique()))
        mean, lo, hi = g.mean()[x].values, g.min()[x].values, g.max()[x].values
        if invert:
            x = -x
            o = np.argsort(x)
            x, mean, lo, hi = x[o], mean[o], lo[o], hi[o]
        ax.plot(x, mean, marker="o", ms=4, lw=1.6,
                color=COLORS[arch], label=LABELS.get(arch, arch))
        ax.fill_between(x, lo, hi, color=COLORS[arch], alpha=0.18, lw=0)
    for arch in ORDER:
        v = clean.get(arch)
        if v is not None:
            ax.axhline(v, color=COLORS[arch], ls=":", lw=0.8, alpha=0.5)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, lw=0.5)


def main():
    df = pd.read_csv(os.path.join(C.OUT_ROOT, "robustness.csv"))
    clean = (df[df.condition == "clean"].groupby("arch").macro_f1.mean().to_dict())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    b = subset(df, "broaden_")
    panel(axes[0], b, "Additional peak FWHM (deg)", clean)
    top = axes[0].get_ylim()[1]
    for v, lab in [(TRAIN_P95, "train p95"), (TRAIN_P99, "train p99")]:
        axes[0].axvline(v, color="black", ls="--", lw=0.8, alpha=0.45)
        axes[0].annotate(lab, xy=(v, top), xytext=(3, -4), va="top",
                         textcoords="offset points", fontsize=7, rotation=90,
                         alpha=0.6)
    axes[0].set_title("Peak broadening", fontsize=10)
    axes[0].set_ylabel("Macro-F1")

    n = subset(df, "noise_snr")
    panel(axes[1], n, "Signal-to-noise ratio", clean, logx=True, invert=False)
    ticks = sorted(n.x.unique())
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([f"{t:g}" for t in ticks])
    axes[1].minorticks_off()
    axes[1].invert_xaxis()           # noisier towards the right
    axes[1].set_title("Additive Gaussian noise", fontsize=10)

    s = subset(df, "shift_")
    panel(axes[2], s, "Zero-point shift (deg)", clean)
    axes[2].set_title("Zero-point shift", fontsize=10)

    axes[0].legend(fontsize=8, framealpha=0.85, loc="upper right")
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.text(0.5, 0.02,
             "Dotted lines: unperturbed performance. Bands: range over three seeds.",
             ha="center", fontsize=8, alpha=0.75)
    for ext in ("png", "pdf"):
        p = os.path.join(C.OUT_ROOT, f"fig3_robustness.{ext}")
        plt.savefig(p, dpi=300)
        print("->", p)
    plt.close()


if __name__ == "__main__":
    main()