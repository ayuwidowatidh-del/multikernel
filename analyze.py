""" Analysis of the trained models
python analyze.py --what eval
python analyze.py --what stats
python analyze.py --what ablation
python analyze.py --what probe
python analyze.py --what robust
python analyze.py --what diagnostic
python analyze.py --what viz
"""
#library/function 
import argparse
import hashlib
import itertools
import os
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d, shift as ndshift
from scipy.stats import binomtest
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
import config as C
from data import XRDDataset
from models import Model
#Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#FWHM conversion
FWHM_TO_SIGMA = 1.0 / 2.3548
#Model Loading
def load_model(arch, seed):
    m = Model(arch).to(DEVICE)
    path = os.path.join(C.OUT_ROOT, f"{arch}_seed{seed}", "best_model.pt")
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m
#Present class labels
def present_labels(y):
    """Classes present in y. The RRUFF set has no trigonal patterns, so macro-F1
    there is averaged over 6 classes instead of 7."""
    return sorted(np.unique(y).tolist())
#macro f1 calculation
def macro_f1(y, pred, labels=None):
    return f1_score(y, pred, average="macro", zero_division=0,
                    labels=labels if labels is not None else present_labels(y))
#dataset loading
def dataset(kind, transform=None):
    if kind == "test":
        return XRDDataset(C.X_TEST, C.Y_TEST, C.META_TEST, transform)
    if kind == "rruff":
        return XRDDataset(C.X_RRUFF, C.Y_RRUFF, C.META_RRUFF, transform)
    raise ValueError(kind)
#model prediction and feature extraction
@torch.no_grad()
def predict(model, ds, embeddings=False):
    loader = DataLoader(ds, C.EVAL_BATCH, shuffle=False, num_workers=C.NUM_WORKERS)
    ys, ps, probs, embs = [], [], [], []
    hook = None
    if embeddings:
        hook = model.penultimate.register_forward_hook(
            lambda m, i, o: embs.append(o.detach().cpu().numpy()))
    for x, y in loader:
        out = model(x.to(DEVICE))
        probs.append(torch.softmax(out, 1).cpu().numpy())
        ps.extend(out.argmax(1).cpu().numpy())
        ys.extend(y.numpy())
    if hook is not None:
        hook.remove()
    res = {"y": np.array(ys), "pred": np.array(ps), "prob": np.concatenate(probs)}
    if embeddings:
        res["emb"] = np.concatenate(embs)
    return res
#XRD robustness transformations
'''Three transformations are implemented to evaluate model robustness 
under controlled perturbations of XRD patterns:

Peak broadening (broaden): Applies Gaussian filtering to simulate peak 
broadening. The specified FWHM in degrees is converted to the Gaussian 
standard deviation in bins using the angular step size.

Noise addition (add_noise): Adds Gaussian noise scaled according to the 
input signal-to-noise ratio (SNR). A random seed is provided for reproducibility.

Zero-shift (zero_shift): Shifts the XRD pattern along the two-theta axis 
by a specified angular offset, using linear interpolation and nearest-
boundary handling.'''
def broaden(fwhm_deg):
    sigma = fwhm_deg * FWHM_TO_SIGMA / C.TWO_THETA_STEP
    return lambda x: gaussian_filter1d(x, sigma, mode="nearest")
def add_noise(snr, seed=0):
    # the noise depends on the pattern itself, so every model gets the same noisy input
    def f(x):
        h = int(hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng([seed, h])
        p = float(np.mean(x ** 2))
        return x + rng.normal(0, np.sqrt(p / snr), size=x.shape).astype(np.float32)
    return f
def zero_shift(deg):
    pts = deg / C.TWO_THETA_STEP
    return lambda x: ndshift(x, pts, order=1, mode="nearest")
#statistical analysis
'''Three statistical procedures are implemented to compare 
model performance:

Clustered bootstrap (clustered_bootstrap): Estimates the 95% confidence
 interval (CI) for the difference in macro-F1 between two models by 
 resampling structures as clusters. The class labels are fixed to those 
 present in the full evaluation set.

Structure-level McNemar test (structure_mcnemar): Compares two models 
using one majority-vote prediction per structure and an exact binomial 
test on discordant correctness outcomes.

Holm correction (holm): Applies the Holm–Bonferroni procedure to adjust 
p-values for multiple hypothesis testing, controlling the family-wise 
error rate.'''

def clustered_bootstrap(y, pred_a, pred_b, clusters, n=C.N_BOOTSTRAP, seed=0):
    """95% CI of macro-F1(b) - macro-F1(a), resampling structures (clusters).
    Labels are fixed to the classes in the full set."""
    labels = present_labels(y)
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    index_of = {c: np.where(clusters == c)[0] for c in uniq}
    diffs = np.empty(n)
    for i in range(n):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[c] for c in draw])
        diffs[i] = (macro_f1(y[idx], pred_b[idx], labels)
                    - macro_f1(y[idx], pred_a[idx], labels))
    obs = macro_f1(y, pred_b, labels) - macro_f1(y, pred_a, labels)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": obs, "ci_low": lo, "ci_high": hi, "excludes_zero": lo > 0 or hi < 0}
def structure_mcnemar(y, pred_a, pred_b, clusters):
    """Majority-vote one prediction per structure, then exact McNemar."""
    df = pd.DataFrame({"c": clusters, "y": y, "a": pred_a, "b": pred_b})
    agg = df.groupby("c").agg(y=("y", lambda s: s.mode().iat[0]),
                              a=("a", lambda s: s.mode().iat[0]),
                              b=("b", lambda s: s.mode().iat[0]))
    a_ok, b_ok = agg.a == agg.y, agg.b == agg.y
    n01 = int((~a_ok & b_ok).sum())    # only b correct
    n10 = int((a_ok & ~b_ok).sum())    # only a correct
    if n01 + n10 == 0:
        return {"n_structures": len(agg), "b_only": 0, "a_only": 0, "p": 1.0}
    p = binomtest(n01, n01 + n10, 0.5).pvalue
    return {"n_structures": len(agg), "b_only": n01, "a_only": n10, "p": float(p)}
def holm(pvals):
    order = np.argsort(pvals)
    m, adj = len(pvals), np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj
#analyses
def cmd_eval(args):
    rows, reports = [], {}
    for kind in args.sets:
        ds = dataset(kind)
        labels = present_labels(ds.y)
        if len(labels) < C.NUM_CLASSES:
            print(f"[{kind}] classes present: {[C.LABEL_NAMES[k] for k in labels]} "
                  f"-- macro-F1 averaged over these only")
        for arch, seed in itertools.product(C.MODELS, C.SEEDS):
            r = predict(load_model(arch, seed), ds)
            rows.append({"set": kind, "arch": arch, "seed": seed,
                         "n_classes": len(labels),
                         "acc": float((r["y"] == r["pred"]).mean()),
                         "macro_f1": macro_f1(r["y"], r["pred"], labels),
                         "weighted_f1": f1_score(r["y"], r["pred"], average="weighted",
                                                 zero_division=0)})
            reports[f"{kind}_{arch}_s{seed}"] = classification_report(
                r["y"], r["pred"], digits=4, zero_division=0, labels=labels,
                target_names=[C.LABEL_NAMES[k] for k in labels])
            np.save(os.path.join(C.OUT_ROOT, f"pred_{kind}_{arch}_s{seed}.npy"), r["pred"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_ROOT, "metrics_per_seed.csv"), index=False)
    print(df.groupby(["set", "arch"])[["acc", "macro_f1"]].agg(["mean", "std"]))
    with open(os.path.join(C.OUT_ROOT, "classification_reports.txt"), "w") as f:
        for k, v in reports.items():
            f.write(f"===== {k} =====\n{v}\n")
def cmd_stats(args):
    rows = []
    for kind in args.sets:
        ds = dataset(kind)
        clusters = ds.structure_ids()
        for base, cand in getattr(C, "COMPARISONS", [("npcnn", "mknpcnn")]):
            for seed in C.SEEDS:
                pa = np.load(os.path.join(C.OUT_ROOT, f"pred_{kind}_{base}_s{seed}.npy"))
                pb = np.load(os.path.join(C.OUT_ROOT, f"pred_{kind}_{cand}_s{seed}.npy"))
                y = ds.y
                boot = clustered_bootstrap(y, pa, pb, clusters, seed=seed)
                mc = structure_mcnemar(y, pa, pb, clusters)
                rows.append({"set": kind, "comparison": f"{cand} - {base}",
                             "seed": seed, **boot, **mc})
    df = pd.DataFrame(rows).rename(columns={"diff": "f1_diff"})
    df["diff"] = df["f1_diff"]          # keep both names for readability
    df["p_holm"] = holm(df["p"].to_numpy())
    df.to_csv(os.path.join(C.OUT_ROOT, "statistics.csv"), index=False)
    print(df.to_string(index=False))
    # df.diff and df.set are DataFrame methods, so use df["diff"], df["set"]
    for (kind, comp), sub in df.groupby(["set", "comparison"]):
        consistent = bool((sub["diff"] > 0).all() and sub["excludes_zero"].all())
        print(f"{kind:6s} {comp:24s} consistent improvement = {consistent}"
              f"  (diff per seed: {[round(v, 4) for v in sub['diff']]})")
def cmd_ablation(args):
    """Zero one branch at inference; compare against zeroing random channels."""
    masks = {"full": (1, 1, 1), "no_narrow": (0, 1, 1),
             "no_medium": (1, 0, 1), "no_broad": (1, 1, 0)}
    rows = []
    for kind in args.sets:
        ds = dataset(kind)
        for seed in C.SEEDS:
            model = load_model("mknpcnn", seed)
            for name, mask in masks.items():
                model.cnn.reset_masks()
                model.cnn.set_branch_mask(mask)
                r = predict(model, ds)
                rows.append({"set": kind, "seed": seed, "condition": name,
                             "macro_f1": macro_f1(r["y"], r["pred"])})
            # control: zero a random subset of channels of the same size
            model.cnn.reset_masks()
            n_ch = sum(C.MK_BRANCH_CH)
            rng = np.random.default_rng(seed)
            for rep in range(C.N_RANDOM_MASK_REPEATS):
                for size in set(C.MK_BRANCH_CH):
                    m = torch.ones(n_ch)
                    m[rng.choice(n_ch, size, replace=False)] = 0.0
                    model.cnn.set_channel_mask(m)
                    r = predict(model, ds)
                    rows.append({"set": kind, "seed": seed,
                                 "condition": f"random_{size}ch_rep{rep}",
                                 "macro_f1": macro_f1(r["y"], r["pred"])})
            model.cnn.reset_masks()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_ROOT, "branch_ablation.csv"), index=False)
    print(df[~df.condition.str.startswith("random")]
          .pivot_table(index="condition", columns="seed", values="macro_f1"))
    rnd = df[df.condition.str.startswith("random")]
    print("random-channel control: mean %.4f  std %.4f"
          % (rnd.macro_f1.mean(), rnd.macro_f1.std()))
@torch.no_grad()
def cmd_probe(args):
    """Mean stage-1 branch activation for single Gaussian peaks of varying FWHM."""
    rows = []
    centre = C.INPUT_LEN // 2
    grid = np.arange(C.INPUT_LEN, dtype=np.float32)
    for seed in C.SEEDS:
        model = load_model("mknpcnn", seed)
        block = model.cnn.stages[0]
        for fwhm in C.PROBE_FWHM_DEG:
            sigma = fwhm * FWHM_TO_SIGMA / C.TWO_THETA_STEP
            x = np.exp(-0.5 * ((grid - centre) / sigma) ** 2).astype(np.float32)
            x = torch.from_numpy(x / C.SCALE_CONST)[None, None].to(DEVICE)
            for i, branch in enumerate(block.branches):
                act = torch.relu(branch(x))
                rows.append({"seed": seed, "fwhm_deg": fwhm,
                             "branch": ["narrow", "medium", "broad"][i],
                             "mean_act": float(act.mean()),
                             "max_act": float(act.max())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_ROOT, "branch_probe.csv"), index=False)
    print(df.pivot_table(index="fwhm_deg", columns="branch", values="mean_act"))
def cmd_robust(args):
    conds = ([("clean", None)]
             + [(f"broaden_{v}", broaden(v)) for v in C.BROADEN_FWHM_DEG]
             + [(f"noise_snr{v}", add_noise(v)) for v in C.NOISE_SNR]
             + [(f"shift_{v}", zero_shift(v)) for v in C.SHIFT_DEG])
    rows = []
    for name, tf in conds:
        ds = dataset("test", tf)
        for arch, seed in itertools.product(C.MODELS, C.SEEDS):
            r = predict(load_model(arch, seed), ds)
            rows.append({"condition": name, "arch": arch, "seed": seed,
                         "macro_f1": macro_f1(r["y"], r["pred"])})
    df = pd.DataFrame(rows)
    base = df[df.condition == "clean"].set_index(["arch", "seed"]).macro_f1
    df["rel_drop"] = df.apply(
        lambda r: 1 - r.macro_f1 / base.loc[(r.arch, r.seed)], axis=1)
    df.to_csv(os.path.join(C.OUT_ROOT, "robustness.csv"), index=False)
    print(df.pivot_table(index="condition", columns="arch",
                         values=["macro_f1", "rel_drop"]))
def cmd_diagnostic(args):
    """Logistic regression: is baseline accuracy related to peak width?"""
    from scipy.signal import find_peaks, peak_widths
    import statsmodels.api as sm

    ds = dataset("test")
    clusters = ds.structure_ids()
    feats = []
    for i in range(len(ds)):
        x = np.asarray(ds.X[i], dtype=float).reshape(-1)
        rngx = x.max() - x.min()
        xs = (x - x.min()) / (rngx + 1e-9)
        pk, _ = find_peaks(xs, prominence=0.05)
        w = peak_widths(xs, pk, rel_height=0.5)[0] * C.TWO_THETA_STEP if len(pk) else [np.nan]
        feats.append({"n_peaks": len(pk), "fwhm_med": float(np.nanmedian(w))})
    F = pd.DataFrame(feats)
    for seed in C.SEEDS:
        pred = np.load(os.path.join(C.OUT_ROOT, f"pred_test_npcnn_s{seed}.npy"))
        d = F.assign(correct=(pred == ds.y).astype(int), cluster=clusters).dropna()
        X = sm.add_constant(d[["fwhm_med", "n_peaks"]])
        res = sm.GLM(d.correct, X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d.cluster})
        print(f"--- baseline seed {seed} ---")
        print(res.summary().tables[1])
    F.to_csv(os.path.join(C.OUT_ROOT, "test_pattern_features.csv"), index=False)
#visualisation
def cmd_viz(args):
    """Confusion matrices, per-class F1 and t-SNE/UMAP of the penultimate layer
    (one seed only, for qualitative inspection)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed = args.seed
    for kind in args.sets:
        ds = dataset(kind)
        labels = present_labels(ds.y)
        names = [C.LABEL_NAMES[k] for k in labels]
        res = {}
        for arch in C.MODELS:
            res[arch] = predict(load_model(arch, seed), ds, embeddings=True)

        # ---------- confusion matrix (counts + normalised) ----------
        n = len(C.MODELS)
        fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 9))
        axes = np.atleast_2d(axes)
        for j, arch in enumerate(C.MODELS):
            r = res[arch]
            cm = confusion_matrix(r["y"], r["pred"], labels=labels)
            cmn = cm / (cm.sum(1, keepdims=True) + 1e-9)
            for i, (M, fmt, vmax, tag) in enumerate(
                    [(cm, "d", None, "counts"), (cmn, ".2f", 1.0, "normalised")]):
                ax = axes[i, j]
                im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
                for a_ in range(M.shape[0]):
                    for b_ in range(M.shape[1]):
                        v = M[a_, b_]
                        ax.text(b_, a_, format(v, fmt), ha="center", va="center",
                                fontsize=7,
                                color="white" if v > (M.max() * 0.6) else "black")
                ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45,
                                                                    ha="right", fontsize=7)
                ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
                ax.set_title(f"{arch}, {kind} ({tag})", fontsize=9)
                ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)
        plt.tight_layout()
        p = os.path.join(C.OUT_ROOT, f"confusion_{kind}_seed{seed}.png")
        plt.savefig(p, dpi=200); plt.close()
        print("->", p)

        # ---------- per-class F1, all seeds ----------
        rows = []
        for arch in C.MODELS:
            for s in C.SEEDS:
                pred = np.load(os.path.join(C.OUT_ROOT, f"pred_{kind}_{arch}_s{s}.npy"))
                f1s = f1_score(ds.y, pred, average=None, labels=labels, zero_division=0)
                for k, f in zip(labels, f1s):
                    rows.append({"arch": arch, "seed": s, "class": C.LABEL_NAMES[k],
                                 "support": int((ds.y == k).sum()), "f1": f})
        pc = pd.DataFrame(rows)
        pc.to_csv(os.path.join(C.OUT_ROOT, f"f1_per_class_{kind}.csv"), index=False)
        tab = pc.pivot_table(index=["class", "support"], columns="arch", values="f1",
                             aggfunc=["mean", "std"])
        print(f"\n=== Per-class F1, {kind} ===")
        print(tab.to_string())
        print("Classes with small support have the noisiest F1; read together with the support column.")

        # ---------- embedding projections ----------
        rng = np.random.default_rng(0)
        idx = (rng.choice(len(ds.y), args.max_viz, replace=False)
               if len(ds.y) > args.max_viz else np.arange(len(ds.y)))
        reducers = [("t-SNE", lambda X: TSNE(n_components=2, perplexity=min(40, len(X) // 4),
                                             init="pca", learning_rate="auto",
                                             random_state=0).fit_transform(X))]
        try:
            import umap
            reducers.append(("UMAP", lambda X: umap.UMAP(n_components=2, n_neighbors=30,
                                                         min_dist=0.1, random_state=0
                                                         ).fit_transform(X)))
        except ImportError:
            print("umap-learn not installed, skipping UMAP")

        for rname, fn in reducers:
            fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 10))
            axes = np.atleast_2d(axes)
            for j, arch in enumerate(C.MODELS):
                r = res[arch]
                emb, y, pred = r["emb"][idx], r["y"][idx], r["pred"][idx]
                xy = fn(emb)
                for k in labels:                       # row 1: true label
                    m = y == k
                    axes[0, j].scatter(xy[m, 0], xy[m, 1], s=5, alpha=0.6,
                                       label=C.LABEL_NAMES[k])
                axes[0, j].set_title(f"{arch}, {rname}, {kind} (label)", fontsize=9)
                axes[0, j].legend(markerscale=2.5, fontsize=6)
                ok = y == pred                          # row 2: correct vs wrong
                axes[1, j].scatter(xy[ok, 0], xy[ok, 1], s=5, alpha=0.4,
                                   c="#2ecc71", label="correct")
                axes[1, j].scatter(xy[~ok, 0], xy[~ok, 1], s=12, alpha=0.85,
                                   c="#e74c3c", label="wrong")
                axes[1, j].set_title(f"{arch}, {rname}, {kind} (correct/wrong)", fontsize=9)
                axes[1, j].legend(markerscale=2.0, fontsize=7)
                for ax in (axes[0, j], axes[1, j]):
                    ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            p = os.path.join(C.OUT_ROOT,
                             f"{rname.lower().replace('-', '')}_{kind}_seed{seed}.png")
            plt.savefig(p, dpi=200); plt.close()
            print("->", p)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True,
                    choices=["eval", "stats", "ablation", "probe", "robust",
                             "diagnostic", "viz"])
    ap.add_argument("--sets", nargs="*", default=["test", "rruff"])
    ap.add_argument("--seed", type=int, default=0, help="seed used for viz")
    ap.add_argument("--max_viz", type=int, default=5000)
    args = ap.parse_args()
    {"eval": cmd_eval, "stats": cmd_stats, "ablation": cmd_ablation,
     "probe": cmd_probe, "robust": cmd_robust, "diagnostic": cmd_diagnostic,
     "viz": cmd_viz}[args.what](args)
