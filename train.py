"""Training script: 3 models x 3 seeds."""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import config as C
from data import XRDDataset, check_split_integrity
from models import Model, count_params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    ys, ps, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += float(criterion(out, y)) * y.size(0)
        ps.extend(out.argmax(1).cpu().numpy())
        ys.extend(y.cpu().numpy())
    ys, ps = np.array(ys), np.array(ps)
    return {
        "loss": loss_sum / max(1, len(ys)),
        "acc": float((ys == ps).mean()),
        "f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "precision": float(precision_score(ys, ps, average="macro", zero_division=0)),
        "recall": float(recall_score(ys, ps, average="macro", zero_division=0)),
    }


def run(arch, seed):
    out_dir = os.path.join(C.OUT_ROOT, f"{arch}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "last.pt")
    best_path = os.path.join(out_dir, "best_model.pt")

    set_seed(seed)
    train_ds = XRDDataset(C.X_TRAIN, C.Y_TRAIN, C.META_TRAIN)
    val_ds = XRDDataset(C.X_VAL, C.Y_VAL, C.META_VAL)
    g = torch.Generator().manual_seed(seed)          # identical batch order per seed
    train_loader = DataLoader(train_ds, C.BATCH_SIZE, shuffle=True, generator=g,
                              num_workers=C.NUM_WORKERS)
    val_loader = DataLoader(val_ds, C.EVAL_BATCH, shuffle=False, num_workers=C.NUM_WORKERS)

    model = Model(arch).to(DEVICE)
    params = count_params(model)
    print(f"[{arch} seed{seed}] feature {model.feat_shape} | params {params}")

    criterion = nn.CrossEntropyLoss(label_smoothing=C.LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=C.SCHED_PATIENCE)

    start_epoch, best_f1, wait, history = 1, 0.0, 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_f1, wait = ck["epoch"] + 1, ck["best_f1"], ck["wait"]
        history = ck.get("history", [])
        print(f"  resumed at epoch {start_epoch} (best_f1={best_f1:.4f})")

    t0 = time.time()
    for epoch in range(start_epoch, C.EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            optimizer.step()
            loss_sum += float(loss) * y.size(0)

        train_loss = loss_sum / len(train_ds)
        val = evaluate(model, val_loader, criterion)
        scheduler.step(val["f1"])
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": train_loss, **val})

        # update best_f1 before saving the checkpoint so resuming works correctly
        improved = val["f1"] > best_f1
        if improved:
            best_f1, wait = val["f1"], 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_f1": best_f1, "wait": wait, "history": history}, ckpt_path)

        print(f"  ep {epoch:03d} lr {lr_now:.2e} train {train_loss:.4f} "
              f"val {val['loss']:.4f} acc {val['acc']:.3f} F1 {val['f1']:.4f}"
              f"{' *best' if improved else ''}")

        if wait >= C.PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({"arch": arch, "seed": seed, "best_val_f1": best_f1,
                   "epochs_run": len(history), "params": params,
                   "minutes": round((time.time() - t0) / 60, 1),
                   "device": DEVICE, "torch": torch.__version__}, f, indent=2)
    print(f"[{arch} seed{seed}] best val F1 {best_f1:.4f} "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=C.MODELS + ["all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS)
    args = ap.parse_args()

    os.makedirs(C.OUT_ROOT, exist_ok=True)
    check_split_integrity(C.META_TRAIN, C.META_VAL, C.META_TEST)

    archs = C.MODELS if args.arch == "all" else [args.arch]
    for arch in archs:
        for seed in args.seeds:
            run(arch, seed)