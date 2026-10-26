"""Synthetic XRD pattern generation.

  python generate_data.py            # level D4 (LEVELS_TO_GENERATE)
  python generate_data.py 0 5        # other levels
  python generate_data.py all loo    # all levels + leave-one-out variants
"""
#library
import os
import sys
import json
import time
import hashlib
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

CACHE_NPZ = os.environ.get("XRD_CACHE_NPZ", os.path.join("data", "cache", "cif.npz"))
LABEL_CSV = os.environ.get("XRD_LABEL_CSV", os.path.join("data", "cache", "cif_labels.csv"))
OUT_DIR   = os.environ.get("XRD_OUT_DIR",   os.path.join("data", "datasynth"))

#perturbation level
LADDER = [
    ("DI_ideal",           []),                                          # no perturbation
    ("D0_resolution",      ["res"]),                                     # random Caglioti
    ("D1_instrument",      ["zshift", "displace", "asym"]),              # geometry + axial divergence
    ("D2_microstructure",  ["size", "strain", "texture", "intpert"]),    # sample
    ("D3_composition",     ["impurity"]),                                # secondary phase
    ("D4_measurement",     ["bg", "tilt", "noise"]),                     # full model
]
LEVELS_TO_GENERATE = [len(LADDER) - 1]   # only D4 is used
RUN_LOO = False        # leave-one-out variants of the full level

STORE_FP16 = False     # save X as float16 to halve the file size
TEST_MODE  = False
N_TEST_CIF = 50

#spliting data
SPLIT_MODE = "group_disjoint"
SPLIT_FRAC = (0.70, 0.15, 0.15)

#grid,seed,label
TMIN, TMAX, L = 5.0, 90.0, 8500
TT   = np.linspace(TMIN, TMAX, L).astype(np.float64)
STEP = float((TMAX - TMIN) / (L - 1))
SEED = 42

LABEL_NAMES = {0: "triclinic", 1: "monoclinic", 2: "orthorhombic",
               3: "tetragonal", 4: "trigonal", 5: "hexagonal", 6: "cubic"}

N_SYN_TRAIN = {0: 6, 1: 3, 2: 3, 3: 4, 4: 2, 5: 7, 6: 3}
N_SYN_VAL   = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 1}
N_SYN_TEST  = {0: 2, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 1}
DEFAULT_TRAIN, DEFAULT_VAL, DEFAULT_TEST = 3, 1, 1

#physical parameters
CAGLIOTI_U_RANGE = (0.003, 0.030)
CAGLIOTI_V_RANGE = (-0.015, 0.000)
CAGLIOTI_W_RANGE = (0.002, 0.020)
CAGLIOTI_MIN_FWHM2 = 1e-3      
CAGLIOTI_MAX_TRY   = 100
IDEAL_U, IDEAL_V, IDEAL_W = 0.010, 0.000, 0.005
WAVELENGTH_A   = 1.5406
K_SCHERRER     = 0.9
GRAIN_NM_RANGE = (15.0, 500.0)
GRAIN_LOG_UNIFORM = True
STRAIN_EPS_RANGE = (0.0, 3e-3)
ZERO_SHIFT_RANGE = (-0.05, 0.05)     
GONIO_R_MM       = 240.0             
DISPLACE_S_MM    = (-0.084, 0.084)   
ASYMM_RANGE      = (0.0, 0.40)
MARCH_R_RANGE    = (0.5, 1.7)
POISSON_NOISE    = True
COUNTS_MAX_RANGE = (1e3, 1e5)
NOISE_REL_RANGE  = (0.003, 0.030)
TILT_RANGE = (-0.005, 0.005)
TILT_PROB  = 0.0
BG_RATIO_RANGE  = (0.00, 0.40)
TOPK_PEAKS       = 300
WINDOW_HALF_MULT = 15.0
FINAL_SCALE      = 100.0
IMPURITY_PROB      = 0.85
IMPURITY_N_RANGE   = (1, 6)
IMPURITY_INT_RANGE = (0.05, 0.25)
IMPURITY_TT_RANGE  = (TMIN + 5, TMAX - 5)
INTENSITY_PERT_PROB  = 0.8
INTENSITY_PERT_RANGE = (0.8, 1.2)
HUMP_PROB   = 0.55
HUMP_CENTER = (15.0, 45.0)
HUMP_FWHM   = (8.0, 25.0)
HUMP_AMP    = (0.15, 0.65)
BASELINE_FLOOR = 0.01
TOPK_BEFORE_IMPURITY = True

#components
COMPONENTS = ["res", "zshift", "displace", "asym", "size", "strain",
              "texture", "intpert", "impurity", "bg", "tilt", "noise"]
COMP_ID = {c: i for i, c in enumerate(COMPONENTS)}

COMP_DESC = {
    "res":      "Caglioti U,V,W, instrument resolution (Gaussian)",
    "zshift":   "detector zero shift, constant in 2theta",
    "displace": "sample displacement, d2theta = -2s*cos(theta)/R",
    "asym":     "peak asymmetry (split PV)",
    "size":     "crystallite size (Scherrer), Lorentzian ~ 1/cos(theta)",
    "strain":   "microstrain (Williamson-Hall), Lorentzian ~ tan(theta)",
    "texture":  "preferred orientation (approximation)",
    "intpert":  "per-peak intensity perturbation U[0.8, 1.2]",
    "impurity": "1-6 impurity peaks, 5-25% of max intensity",
    "bg":       "Chebyshev background (order 6) + amorphous hump",
    "tilt":     "linear baseline drift",
    "noise":    "counting noise (Poisson if POISSON_NOISE, else Gaussian)",
}

assert set(sum([c for _, c in LADDER], [])) == set(COMPONENTS), \
    "LADDER must contain exactly all COMPONENTS, without duplicates."

LEVEL_NAMES = [n for n, _ in LADDER]
FULL_LEVEL  = len(LADDER) - 1

def active_at(level, drop=None):
    s = set()
    for i in range(level + 1):
        s.update(LADDER[i][1])
    if drop:
        s.discard(drop)
    return s

#Per-Component Random Number Generation
SPLIT_ID = {"train": 0, "val": 1, "test": 2}

def _h(s):
    return int(hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:15], 16)
    
    
def crng(cid, split, j, comp):
    return np.random.default_rng(
        np.random.SeedSequence([SEED, _h(cid), SPLIT_ID[split], int(j), COMP_ID[comp]]))

#Physics-Based XRD Profile Modeling
_TAN_GRID = np.tan(np.deg2rad(TT / 2.0))

#Instrumental Broadening: Caglioti Model
def draw_caglioti(rng):
    """Draw U, V, W such that FWHM_G^2 > 0 over the whole 2theta range."""
    for _ in range(CAGLIOTI_MAX_TRY):
        U = rng.uniform(*CAGLIOTI_U_RANGE)
        V = rng.uniform(*CAGLIOTI_V_RANGE)
        W = rng.uniform(*CAGLIOTI_W_RANGE)
        if (U * _TAN_GRID ** 2 + V * _TAN_GRID + W).min() > CAGLIOTI_MIN_FWHM2:
            return U, V, W
    # fallback: V=0 is always valid because U, W > 0
    return rng.uniform(*CAGLIOTI_U_RANGE), 0.0, rng.uniform(*CAGLIOTI_W_RANGE)

def fwhm_G_caglioti(tt, U, V, W):
    t = np.tan(np.deg2rad(tt / 2.0))
    return float(np.sqrt(max(U * t ** 2 + V * t + W, 1e-8)))
#Microstructural Broadening
def fwhm_L_size_strain(tt, grain_nm, strain_eps, use_size, use_strain):
    th = np.deg2rad(tt / 2.0)
    fL = 0.0
    if use_size:                                    # Scherrer
        D_A = float(grain_nm) * 10.0
        fL += np.rad2deg(K_SCHERRER * WAVELENGTH_A / D_A) / (np.cos(th) + 1e-12)
    if use_strain:                                  # Williamson-Hall
        fL += np.rad2deg(4.0 * float(strain_eps)) * np.tan(th)
    return max(float(fL), 0.0)

#Cu Kα₂ Doublet and Profile Configuration
WAVELENGTH_A2 = 1.544390      # Cu K-alpha2
KA2_RATIO     = 0.5           # I(Ka2)/I(Ka1)
USE_KA2       = True          # set False for monochromatic (Ka1 only) data
FWHM_CLIP     = (0.04, 4.0)   # deg
NORMALIZE_AREA = True         # I0 = integrated intensity, not peak height


def ka2_position(tt_deg):
    """Ka2 position from Bragg's law (fixed d). Splitting is 0.05 deg at 2th=20 and
    0.28 deg at 2th=90, so high-angle peaks are split."""
    s = (WAVELENGTH_A2 / WAVELENGTH_A) * np.sin(np.deg2rad(tt_deg / 2.0))
    return None if abs(s) > 1.0 else 2.0 * np.degrees(np.arcsin(s))

#Thompson–Cox–Hastings Pseudo-Voigt Profile
def tch(fG, fL):
    f5 = (fG ** 5 + 2.69269 * fG ** 4 * fL + 2.42843 * fG ** 3 * fL ** 2
          + 4.47163 * fG ** 2 * fL ** 3 + 0.07842 * fG * fL ** 4 + fL ** 5)
    f = f5 ** 0.2
    r = fL / (f + 1e-12)
    eta = 1.36603 * r - 0.47719 * r ** 2 + 0.11116 * r ** 3
    return float(np.clip(f, *FWHM_CLIP)), float(np.clip(eta, 0.0, 1.0))

#Sample Displacement
def displacement_shift(tt_deg, s_mm):
    """Delta2theta caused by sample height error. Angle-dependent (~cos theta)."""
    th = np.deg2rad(tt_deg / 2.0)
    return -np.rad2deg(2.0 * float(s_mm) * np.cos(th) / GONIO_R_MM)

#Asymmetric Pseudo-Voigt Peak Profiles
def pv_asymm(x, mu, fwhm, eta, asymm):
    dx = x - mu
    fw = np.where(dx < 0, fwhm * (1.0 - asymm), fwhm * (1.0 + asymm))
    sig = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gam = fw / 2.0
    G = np.exp(-0.5 * (dx / (sig + 1e-12)) ** 2)
    Lo = 1.0 / (1.0 + (dx / (gam + 1e-12)) ** 2)
    prof = (1.0 - eta) * G + eta * Lo
    if not NORMALIZE_AREA:
        return prof
    # I0 is an integrated intensity, so the profile is normalised to unit area
    # (otherwise broad peaks get too much intensity).
    area = ((1.0 - eta) * fwhm * np.sqrt(np.pi / (4.0 * np.log(2.0)))
            + eta * fwhm * np.pi / 2.0)
    return prof / (area + 1e-12)

#Texture Approximation
def texture_factor(tt_deg, march_r, pref_axis):
    """Texture approximated from the 2theta distance (no hkl in the cache)."""
    if abs(march_r - 1.0) < 1e-4:
        return 1.0
    a = np.deg2rad(abs(tt_deg - pref_axis) / 2.0)
    r = float(march_r)
    return float(np.clip((r ** 2 * np.cos(a) ** 2 + np.sin(a) ** 2 / r) ** (-1.5),
                         0.02, 30.0))

CHEB_ORDER = 6
CHEB_DAMP = 0.55
#background modelling
def background_curve(rng):
    x = (TT - (TMIN + TMAX) / 2.0) / ((TMAX - TMIN) / 2.0)
    coeff = rng.normal(0.0, 1.0, size=CHEB_ORDER + 1) * CHEB_DAMP ** np.arange(
        CHEB_ORDER + 1)
    poly = np.polynomial.chebyshev.chebval(x, coeff).astype(np.float64)
    poly -= poly.min()
    if poly.max() > 0:
        poly /= poly.max()
    hump = np.zeros(L)
    if rng.random() < HUMP_PROB:
        c = rng.uniform(*HUMP_CENTER)
        w = rng.uniform(*HUMP_FWHM)
        sig = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        hump = rng.uniform(*HUMP_AMP) * np.exp(-0.5 * ((TT - c) / sig) ** 2)
    bg = poly + hump
    bg -= bg.min()
    if bg.max() > 0:
        bg /= bg.max()
    return bg
#Parameter render
PEAKS_ALL = {}      

def draw_params(tt_peaks, ii_peaks, cid, split, j, act):
    p = {"U": IDEAL_U, "V": IDEAL_V, "W": IDEAL_W,
         "grain_nm": 1e6, "strain_eps": 0.0,     # -> fwhm_L = 0, eta = 0
         "zshift": 0.0, "s_mm": 0.0, "asymm": 0.0}

    ii0 = np.clip(np.asarray(ii_peaks, float), 0, None)
    if ii0.max() > 0:
        ii0 = ii0 / ii0.max()
    tt0 = np.asarray(tt_peaks, float)
    if TOPK_BEFORE_IMPURITY and ii0.size > TOPK_PEAKS:
        k = np.argsort(ii0)[::-1][:TOPK_PEAKS]
        tt0, ii0 = tt0[k], ii0[k]

    if "res" in act:
        p["U"], p["V"], p["W"] = draw_caglioti(crng(cid, split, j, "res"))
    if "zshift" in act:
        p["zshift"] = crng(cid, split, j, "zshift").uniform(*ZERO_SHIFT_RANGE)
    if "displace" in act:
        p["s_mm"] = crng(cid, split, j, "displace").uniform(*DISPLACE_S_MM)
    if "asym" in act:
        p["asymm"] = crng(cid, split, j, "asym").uniform(*ASYMM_RANGE)
    if "size" in act:
        _r = crng(cid, split, j, "size")
        p["grain_nm"] = (float(np.exp(_r.uniform(*np.log(GRAIN_NM_RANGE))))
                         if GRAIN_LOG_UNIFORM else _r.uniform(*GRAIN_NM_RANGE))
    if "strain" in act:
        p["strain_eps"] = crng(cid, split, j, "strain").uniform(*STRAIN_EPS_RANGE)
    if "texture" in act:
        r = crng(cid, split, j, "texture")
        p["march_r"] = r.uniform(*MARCH_R_RANGE)
        p["pref_axis"] = r.uniform(TMIN + 5, TMAX - 5)

    tt, ii = tt0, ii0
    if "impurity" in act:
        r = crng(cid, split, j, "impurity")
        if r.random() <= IMPURITY_PROB:
            n = int(r.integers(IMPURITY_N_RANGE[0], IMPURITY_N_RANGE[1] + 1))
            mx = float(max(ii0.max(), 0.01))
            scale = r.uniform(*IMPURITY_INT_RANGE) * mx
            donor_keys = [k for k in PEAKS_ALL.keys() if k != cid] if PEAKS_ALL else []
            if donor_keys:
                dtt, dii = PEAKS_ALL[donor_keys[int(r.integers(0, len(donor_keys)))]]
                dii = np.clip(np.asarray(dii, float), 0, None)
                if dii.max() > 0:
                    dii = dii / dii.max()
                k = np.argsort(dii)[::-1][:n]
                tt = np.concatenate([tt0, np.asarray(dtt, float)[k]])
                ii = np.concatenate([ii0, dii[k] * scale])
            else:
                tt = np.concatenate([tt0, r.uniform(*IMPURITY_TT_RANGE, size=n)])
                ii = np.concatenate([ii0, r.uniform(*IMPURITY_INT_RANGE, size=n) * mx])
    if not TOPK_BEFORE_IMPURITY and ii.size > TOPK_PEAKS:
        k = np.argsort(ii)[::-1][:TOPK_PEAKS]
        tt, ii = tt[k], ii[k]

    if "intpert" in act:
        r = crng(cid, split, j, "intpert")
        if r.random() <= INTENSITY_PERT_PROB:
            ii = ii * r.uniform(*INTENSITY_PERT_RANGE, size=ii.shape)
    p["tt"], p["ii"] = tt, np.clip(ii, 0, None)

    if "bg" in act:
        r = crng(cid, split, j, "bg")
        p["bg_ratio"] = r.uniform(*BG_RATIO_RANGE)
        p["bg_curve"] = background_curve(r)
    if "noise" in act:
        r = crng(cid, split, j, "noise")
        p["noise_rel"] = r.uniform(*NOISE_REL_RANGE)
        p["counts_max"] = float(np.exp(r.uniform(*np.log(COUNTS_MAX_RANGE))))
        p["noise_seed"] = int(r.integers(0, 2 ** 31 - 1))
        p["noise_vec"] = None if POISSON_NOISE else r.standard_normal(L)
    if "tilt" in act:
        r = crng(cid, split, j, "tilt")
        p["tilt_on"] = bool(r.random() < TILT_PROB)
        p["tilt_slope"] = r.uniform(*TILT_RANGE)
    return p

#Synthetic Pattern Rendering
def render(p, act):
    use_size, use_strain = "size" in act, "strain" in act
    use_tex = "texture" in act
    use_disp = "displace" in act

    y = np.zeros(L)
    for mu0, I0 in zip(p["tt"], p["ii"]):
        mu0 = float(mu0)
        mu = mu0 + p["zshift"]
        if use_disp:
            mu += displacement_shift(mu0, p["s_mm"])

        fG = fwhm_G_caglioti(mu0, p["U"], p["V"], p["W"])
        fL = fwhm_L_size_strain(mu0, p["grain_nm"], p["strain_eps"],
                                use_size, use_strain)
        fwhm, eta = tch(fG, fL)
        po = texture_factor(mu0, p["march_r"], p["pref_axis"]) if use_tex else 1.0

        lines = [(mu, 1.0)]
        if USE_KA2:
            mu2 = ka2_position(mu0)
            if mu2 is not None:
                mu2 = mu2 + p["zshift"]
                if use_disp:
                    mu2 += displacement_shift(mu0, p["s_mm"])
                lines.append((mu2, KA2_RATIO))

        for mu_l, w in lines:
            hw = max(0.3, WINDOW_HALF_MULT * fwhm)
            i0 = max(0, int((mu_l - hw - TMIN) / STEP))
            i1 = min(L - 1, int((mu_l + hw - TMIN) / STEP))
            if i1 <= 0 or i0 >= L:
                continue
            y[i0:i1 + 1] += float(I0) * w * po * pv_asymm(
                TT[i0:i1 + 1], mu_l, fwhm, eta, p["asymm"])

    if y.max() > 0:
        y /= y.max()
    if "bg" in act:
        y = y + float(p["bg_ratio"]) * p["bg_curve"]
    if BASELINE_FLOOR > 0:
        y = y + BASELINE_FLOOR
    if "noise" in act:
        if POISSON_NOISE:
            rngp = np.random.default_rng(p["noise_seed"])
            counts = np.clip(y, 0, None) * p["counts_max"]
            y = rngp.poisson(counts).astype(np.float64) / p["counts_max"]
        else:
            y = np.clip(y + float(p["noise_rel"]) * p["noise_vec"], 0, None)
    if "tilt" in act and p["tilt_on"]:
        y = np.clip(y + float(p["tilt_slope"]) * (TT - TT.mean()), 0, None)
    y = np.clip(y, 0, None)
    if y.max() > 0:
        y = FINAL_SCALE * (y / y.max())
    return y.astype(np.float16 if STORE_FP16 else np.float32)
#  Synthetic Sample Planning
def build_ids(z, df):
    idc = "cif_id" if "cif_id" in df.columns else df.columns[0]
    lbc = "label_int" if "label_int" in df.columns else df.columns[1]
    df[idc] = df[idc].astype(str)
    df[lbc] = pd.to_numeric(df[lbc], errors="coerce")
    df = df.dropna(subset=[lbc])
    ids, by_id = [], {}
    for _, r in df.iterrows():
        cid, k = r[idc], int(r[lbc])
        if 0 <= k <= 6 and f"{cid}__tt" in z.files:
            ids.append(cid)
            by_id[cid] = k
    ids = sorted(set(ids))
    np.random.default_rng(SEED).shuffle(ids)
    return (ids[:N_TEST_CIF] if TEST_MODE else ids), by_id


def plan_samples(ids, by_id):
    plan = []
    if SPLIT_MODE == "shared_cif":
        for cid in ids:
            lab = by_id[cid]
            for s, tbl, d in (("train", N_SYN_TRAIN, DEFAULT_TRAIN),
                              ("val", N_SYN_VAL, DEFAULT_VAL),
                              ("test", N_SYN_TEST, DEFAULT_TEST)):
                for j in range(tbl.get(lab, d)):
                    plan.append((cid, s, j))
    elif SPLIT_MODE == "group_disjoint":
        rng = np.random.default_rng(SEED + 1)
        by_lab = defaultdict(list)
        for cid in ids:
            by_lab[by_id[cid]].append(cid)
        assign = {}
        for lab, cs in by_lab.items():
            cs = sorted(cs)
            rng.shuffle(cs)
            n = len(cs)
            ntr = int(round(SPLIT_FRAC[0] * n))
            nva = min(int(round(SPLIT_FRAC[1] * n)), max(0, n - ntr - 1))
            for c, s in zip(cs, ["train"] * ntr + ["val"] * nva
                            + ["test"] * (n - ntr - nva)):
                assign[c] = s
        for cid in ids:
            s = assign[cid]
            n = N_SYN_TRAIN.get(by_id[cid], DEFAULT_TRAIN) if s == "train" else 1
            for j in range(n):
                plan.append((cid, s, j))
    else:
        raise ValueError(SPLIT_MODE)
    return plan


def leakage_report(plan):
    d = defaultdict(set)
    for cid, s, _ in plan:
        d[s].add(cid)
    ov = len(d["train"] & d["test"])
    return {"split_mode": SPLIT_MODE,
            "n_cif": {k: len(v) for k, v in d.items()},
            "cif_overlap_train_test": ov,
            "test_cif_unseen_in_train": len(d["test"] - d["train"]),
            "warning": (f"{ov} CIFs are in both train and test" if ov
                        else "No CIF overlap.")}

#generate
def generate_variant(name, act, plan, by_id, peaks, desc):
    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    print(f"\n[{name}]  active: {[c for c in COMPONENTS if c in act]}")

    B = {s: {"X": [], "y": [], "m": []} for s in ("train", "val", "test")}
    t0 = time.time()
    for n, (cid, split, j) in enumerate(plan, 1):
        tt_p, ii_p = peaks[cid]
        p = draw_params(tt_p, ii_p, cid, split, j, act)
        B[split]["X"].append(render(p, act))
        B[split]["y"].append(by_id[cid])
        B[split]["m"].append({"pair_key": f"{cid}|{split}|{j}", "cif_id": cid,
                              "label_int": by_id[cid],
                              "label_name": LABEL_NAMES[by_id[cid]],
                              "syn_idx": j, "split": split, "variant": name})
        if n % 20000 == 0:
            el = time.time() - t0
            print(f"    {n}/{len(plan)}  {n/max(el,1e-9):.0f}/s  "
                  f"ETA {(len(plan)-n)*el/max(n,1)/60:.1f}min")

    for split in ("train", "val", "test"):
        if not B[split]["X"]:
            continue
        X = np.stack(B[split]["X"])
        y = np.array(B[split]["y"], dtype=np.int64)
        meta = pd.DataFrame(B[split]["m"])

        # shuffle once with a fixed seed
        idx = np.arange(len(y))
        np.random.default_rng(
            np.random.SeedSequence([SEED, 777, SPLIT_ID[split]])).shuffle(idx)

        # save X in chunks (X[idx] at once needs too much RAM)
        mm = open_memmap(os.path.join(out, f"X_{split}.npy"), mode="w+",
                         dtype=X.dtype, shape=X.shape)
        for k in range(0, len(idx), 2000):
            mm[k:k + 2000] = X[idx[k:k + 2000]]
        mm.flush()
        del mm

        np.save(os.path.join(out, f"y_{split}.npy"), y[idx])
        meta.iloc[idx].reset_index(drop=True).to_csv(
            os.path.join(out, f"meta_{split}.csv"), index=False,
            encoding="utf-8-sig")
        u, c = np.unique(y, return_counts=True)
        print(f"    [{split}] {X.shape} {X.dtype}  "
              f"{ {LABEL_NAMES[int(a)]: int(b) for a, b in zip(u, c)} }")

    # number of unique training spectra (only < total for level DI)
    rows = B["train"]["X"]
    n_tot = len(rows)
    n_uniq = len({hashlib.sha1(np.ascontiguousarray(r, dtype=np.float32)
                               .round(3).tobytes()).digest() for r in rows})
    print(f"    unique(train): {n_uniq}/{n_tot}")

    json.dump({"variant": name, "description": desc,
               "active_components": [c for c in COMPONENTS if c in act],
               "train_samples": int(n_tot),
               "train_unique_spectra": int(n_uniq)},
              open(os.path.join(out, "variant.json"), "w"), indent=2)
    print(f"    {time.time()-t0:.1f}s")


def verify_pairing(names):
    """Check that all variants have the same sample order."""
    ref = ref_nm = None
    for nm in names:
        f = os.path.join(OUT_DIR, nm, "meta_train.csv")
        if not os.path.exists(f):
            return
        k = pd.read_csv(f)["pair_key"].tolist()
        if ref is None:
            ref, ref_nm = k, nm
        elif k != ref:
            print(f"\n  Pairing error: {nm} != {ref_nm}")
            return
    print(f"\n  Pairing OK ({len(names)} variant(s)).")


def main(levels, run_loo):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("XRD synthetic data generator")
    print("=" * 72)
    for i in levels:
        nm, comps = LADDER[i]
        print(f"  {nm:<20} + {', '.join(COMP_DESC[c] for c in comps)}")

    # load peak lists
    z = np.load(CACHE_NPZ, allow_pickle=True)
    ids, by_id = build_ids(z, pd.read_csv(LABEL_CSV))

    peaks = {}
    for cid in ids:
        try:
            tt_p = z[f"{cid}__tt"].astype(float)
            ii_p = z[f"{cid}__ii"].astype(float)
        except Exception:
            continue
        m = (tt_p >= TMIN) & (tt_p <= TMAX)
        if m.sum():
            peaks[cid] = (tt_p[m], ii_p[m])
    ids = [c for c in ids if c in peaks]
    global PEAKS_ALL
    PEAKS_ALL = peaks

    cnt = Counter(by_id[c] for c in ids)
    print(f"\n  CIF: {len(ids)}")
    for k in range(7):
        print(f"    [{k}] {LABEL_NAMES[k]:<13} {cnt.get(k,0):>6}")

    plan = plan_samples(ids, by_id)
    rep = leakage_report(plan)
    print(f"\n  Samples/variant: {len(plan)}   SPLIT_MODE={SPLIT_MODE}")
    print(f"  >> {rep['warning']}")

    made = []
    for i in levels:
        nm, comps = LADDER[i]
        generate_variant(nm, active_at(i), plan, by_id, peaks,
                         "+ " + "; ".join(COMP_DESC[c] for c in comps))
        made.append(nm)

    if run_loo:
        for comp in COMPONENTS:
            nm = f"FULL_minus_{comp}"
            generate_variant(nm, active_at(FULL_LEVEL, drop=comp), plan, by_id,
                             peaks, f"full model without '{comp}' ({COMP_DESC[comp]})")
            made.append(nm)

    # save the settings used
    json.dump({
        "version": "datasynth",
        "variants": made,
        "ladder": [{"level": i, "name": n, "adds": c} for i, (n, c) in enumerate(LADDER)],
        "components": COMP_DESC,
        "settings": {
            "caglioti_min_fwhm2": CAGLIOTI_MIN_FWHM2,
            "topk_before_impurity": TOPK_BEFORE_IMPURITY,
            "baseline_floor": BASELINE_FLOOR,
            "texture": "2theta-based approximation (no hkl in the cache)",
        },
        "split": rep,
    }, open(os.path.join(OUT_DIR, "gen_config.json"), "w"),
        indent=2, ensure_ascii=False)

    verify_pairing(made)
    print(f"\nDONE -> {OUT_DIR}")


if __name__ == "__main__":
    a = [x.lower() for x in sys.argv[1:]]
    loo = RUN_LOO or ("loo" in a)
    nums = [int(x) for x in a if x.isdigit() and int(x) < len(LADDER)]
    if "all" in a:
        lv = list(range(len(LADDER)))
    elif nums:
        lv = sorted(set(nums))
    elif a == ["loo"]:
        lv = []
    else:
        lv = LEVELS_TO_GENERATE
    main(lv, loo)