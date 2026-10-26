"""Compare peak profiles of experimental (RRUFF) and simulated (D4) patterns.

  python profile_match.py --synthetic X_train.npy --experimental X_rruff_scaled.npy

Fits a pseudo-Voigt + linear background to isolated peaks in both sets and
compares the FWHM and eta distributions. eta is also converted to an apparent
crystallite size to compare with GRAIN_NM_RANGE in the generator.

Limitations: instrument and sample broadening are not separated (so the size is
a lower bound), and only isolated peaks are used, which favours high-symmetry
phases.
"""
import argparse

import time

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import ks_2samp

import config as C

K_SCHERRER = 0.9
WAVELENGTH_A = 1.5406          # must match the generator
TT = C.TWO_THETA_START + np.arange(C.INPUT_LEN) * C.TWO_THETA_STEP
S2F = 2.0 * np.sqrt(2.0 * np.log(2.0))


def pseudo_voigt(x, amp, mu, fwhm, eta, b0, b1):
    sig = fwhm / S2F
    gam = fwhm / 2.0
    G = np.exp(-0.5 * ((x - mu) / (sig + 1e-12)) ** 2)
    Lo = 1.0 / (1.0 + ((x - mu) / (gam + 1e-12)) ** 2)
    return amp * ((1 - eta) * G + eta * Lo) + b0 + b1 * (x - mu)


def eta_to_ratio(eta):
    """Invert eta = 1.36603r - 0.47719r^2 + 0.11116r^3 for r = fwhm_L / fwhm."""
    r = np.linspace(0.0, 1.0, 2001)
    e = 1.36603 * r - 0.47719 * r ** 2 + 0.11116 * r ** 3
    return float(np.interp(np.clip(eta, 0.0, e.max()), e, r))


def apparent_size_nm(fwhm_deg, eta, two_theta_deg):
    """Scherrer size from the Lorentzian part only. Lower bound: strain is included."""
    fL = eta_to_ratio(eta) * fwhm_deg
    if fL <= 1e-4:
        return np.inf
    fL_rad = np.radians(fL)
    th = np.radians(two_theta_deg / 2.0)
    return K_SCHERRER * WAVELENGTH_A / (fL_rad * np.cos(th)) / 10.0


def fit_isolated_peaks(y, prominence_frac=0.05, min_sep_deg=0.8, window_mult=4.0,
                       min_r2=0.97, max_peaks=12, max_candidates=40, maxfev=2000):
    """Fit the isolated peaks of one pattern. max_candidates limits the number of
    peaks tried, otherwise crowded low-symmetry patterns take very long."""
    y = np.asarray(y, dtype=float)
    span = y.max() - y.min()
    if span <= 0:
        return []
    pk, props = find_peaks(y, prominence=prominence_frac * span)
    if len(pk) == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:max_candidates]
    sep_pts = min_sep_deg / C.TWO_THETA_STEP
    out = []
    for i in order:
        p = pk[i]
        others = pk[pk != p]
        if len(others) and np.min(np.abs(others - p)) < sep_pts:
            continue                                   # not isolated
        half = int(window_mult * sep_pts / 2)
        lo, hi = max(0, p - half), min(len(y), p + half + 1)
        xw, yw = TT[lo:hi], y[lo:hi]
        if len(xw) < 15:
            continue
        p0 = [yw.max() - yw.min(), TT[p], 0.2, 0.5, yw.min(), 0.0]
        bounds = ([0, TT[p] - 0.5, 0.01, 0.0, -np.inf, -np.inf],
                  [np.inf, TT[p] + 0.5, 5.0, 1.0, np.inf, np.inf])
        try:
            popt, _ = curve_fit(pseudo_voigt, xw, yw, p0=p0, bounds=bounds, maxfev=maxfev)
        except Exception:
            continue
        resid = yw - pseudo_voigt(xw, *popt)
        ss_tot = np.sum((yw - yw.mean()) ** 2)
        r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
        if r2 < min_r2:
            continue
        amp, mu, fwhm, eta = popt[:4]
        out.append({"two_theta": mu, "fwhm_deg": fwhm, "eta": eta, "r2": r2,
                    "size_nm": apparent_size_nm(fwhm, eta, mu)})
        if len(out) >= max_peaks:
            break
    return out


def collect(path, label, n_patterns, seed=0, y_path=None, **fit_kw):
    """Fit peaks for a random subset of patterns, with progress output."""
    X = np.load(path, mmap_mode="r")
    y = np.load(y_path) if y_path else None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_patterns, len(X)), replace=False)
    rows, t0 = [], time.time()
    for n, i in enumerate(idx, 1):
        fits = fit_isolated_peaks(np.asarray(X[i], dtype=float).reshape(-1), **fit_kw)
        for r in fits:
            rows.append({"source": label, "pattern": int(i),
                         "label": int(y[i]) if y is not None else -1, **r})
        if n % 25 == 0 or n == len(idx):
            el = time.time() - t0
            print(f"  [{label}] {n}/{len(idx)} patterns, {len(rows)} peaks, "
                  f"{el:.0f}s, ETA {el / n * (len(idx) - n):.0f}s", flush=True)
    return pd.DataFrame(rows)


def summarise(df, col):
    return (df.groupby("source")[col]
              .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
              .loc[:, ["count", "5%", "25%", "50%", "75%", "95%"]])


def main(args):
    fit_kw = dict(min_sep_deg=args.min_sep_deg, max_candidates=args.max_candidates,
                  min_r2=args.min_r2)
    syn = collect(args.synthetic, "synthetic_D4", args.n_patterns, seed=0,
                  y_path=args.synthetic_y, **fit_kw)
    exp = collect(args.experimental, "experimental_RRUFF", args.n_experimental,
                  seed=1, y_path=args.experimental_y, **fit_kw)
    if syn.empty or exp.empty:
        raise SystemExit("too few fitted peaks, try a lower --min_r2")
    df = pd.concat([syn, exp], ignore_index=True)
    df.to_csv(args.out_csv, index=False)

    for col in ("fwhm_deg", "eta"):
        print(f"\n=== {col} ===")
        print(summarise(df, col).to_string())
        ks = ks_2samp(syn[col], exp[col])
        print(f"KS statistic {ks.statistic:.3f}  p {ks.pvalue:.2e}")

    finite = df[np.isfinite(df.size_nm)]
    print("\n=== apparent crystallite size (nm, lower bound) ===")
    print(summarise(finite, "size_nm").to_string())

    e = finite[finite.source == "experimental_RRUFF"].size_nm
    if len(e) > 20:
        lo, hi = np.percentile(e, [5, 95])
        print(f"\nSuggested GRAIN_NM_RANGE from experimental peaks: "
              f"({lo:.0f}, {hi:.0f})   current: {args.current_range}")

    if (df.label >= 0).any():
        print("\n=== fitted peaks per crystal system ===")
        print(df[df.label >= 0]
              .assign(system=lambda d: d.label.map(C.LABEL_NAMES))
              .pivot_table(index="system", columns="source", values="fwhm_deg",
                           aggfunc="count", observed=False).fillna(0).to_string())

    print("\n=== median FWHM by 2theta band ===")
    df["band"] = pd.cut(df.two_theta, bins=[5, 20, 35, 50, 65, 90])
    print(df.pivot_table(index="band", columns="source", values="fwhm_deg",
                         aggfunc="median", observed=False).to_string())

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        for src, sub in df.groupby("source"):
            ax[0].hist(sub.fwhm_deg, bins=40, alpha=0.55, label=src, density=True)
            ax[1].hist(sub.eta, bins=40, alpha=0.55, label=src, density=True)
            ax[2].scatter(sub.two_theta, sub.fwhm_deg, s=6, alpha=0.4, label=src)
        ax[0].set_xlabel("FWHM (deg)"); ax[0].set_ylabel("density"); ax[0].legend()
        ax[1].set_xlabel("eta (0 = Gaussian, 1 = Lorentzian)"); ax[1].legend()
        ax[2].set_xlabel("2theta (deg)"); ax[2].set_ylabel("FWHM (deg)"); ax[2].legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)
        print(f"\nplot -> {args.plot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", required=True, help="X_train.npy from level D4")
    ap.add_argument("--experimental", required=True, help="RRUFF patterns (.npy, same grid)")
    ap.add_argument("--n_patterns", type=int, default=200, help="simulated patterns")
    ap.add_argument("--n_experimental", type=int, default=908)
    ap.add_argument("--synthetic_y", default=None, help="y_train.npy, for per-class counts")
    ap.add_argument("--experimental_y", default=None)
    ap.add_argument("--min_sep_deg", type=float, default=0.8,
                    help="minimum R^2 of a fitted peak")
    ap.add_argument("--max_candidates", type=int, default=40,
                    help="max. peaks tried per pattern")
    ap.add_argument("--min_r2", type=float, default=0.97)
    ap.add_argument("--out_csv", default="peak_profiles.csv")
    ap.add_argument("--plot", default="peak_profiles.png")
    ap.add_argument("--current_range", default="(15, 500)")
    main(ap.parse_args())