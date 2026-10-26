"""Calculate the XRD peak list of every CIF and save it as input for generate_data.py.

  python build_cif_cache.py

Reads all .cif files in CIF_DIR (AMCSD) and writes to OUT_DIR:
  cif.npz           peak positions and intensities, keys "<cif_id>__tt", "<cif_id>__ii"
  cif_labels.csv    cif_id, label_int (0-6), label_name, formula, n_peaks,
                    space_group, sg_number, lattice parameters, volume
  build_failed.log  CIF files that could not be parsed (if any)

Peaks are calculated with pymatgen (XRDCalculator, Cu Ka1, 2theta 5-90 deg).
The crystal system label comes from the space group number found by pymatgen:
  0 triclinic (1-2), 1 monoclinic (3-15), 2 orthorhombic (16-74),
  3 tetragonal (75-142), 4 trigonal (143-167), 5 hexagonal (168-194),
  6 cubic (195-230)
"""
import glob
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifParser

warnings.filterwarnings("ignore")

#paths
CIF_DIR = os.environ.get("XRD_CIF_DIR", os.path.join("data", "cif"))
OUT_DIR = os.environ.get("XRD_CACHE_DIR", os.path.join("data", "cache"))
CACHE_NPZ_NAME = "cif.npz"
LABEL_CSV_NAME = "cif_labels.csv"
FAIL_LOG_NAME = "build_failed.log"

#xrd settings (same as generate_data.py)
WAVELENGTH = 1.5406                  # Cu K-alpha1
TWO_THETA_RANGE = (5.0, 90.0)

PROGRESS_EVERY = 100
COMPRESS_NPZ = True

LABEL_NAMES = {0: "triclinic", 1: "monoclinic", 2: "orthorhombic",
               3: "tetragonal", 4: "trigonal", 5: "hexagonal", 6: "cubic"}


def sg_number_to_label(sg_no):
    """Crystal system (0-6) from space group number (1-230)."""
    if sg_no <= 2:
        return 0
    if sg_no <= 15:
        return 1
    if sg_no <= 74:
        return 2
    if sg_no <= 142:
        return 3
    if sg_no <= 167:
        return 4
    if sg_no <= 194:
        return 5
    return 6


def process_one(cif_path, xrd_calc):
    structure = CifParser(cif_path).parse_structures(primitive=False)[0]
    pattern = xrd_calc.get_pattern(structure, two_theta_range=TWO_THETA_RANGE)
    sg_sym, sg_no = structure.get_space_group_info()
    return {"tt": np.asarray(pattern.x, dtype=np.float64),
            "ii": np.asarray(pattern.y, dtype=np.float64),
            "structure": structure,
            "sg_sym": str(sg_sym),
            "sg_no": int(sg_no),
            "label_int": sg_number_to_label(int(sg_no))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cif_files = sorted(glob.glob(os.path.join(CIF_DIR, "*.cif")))
    if not cif_files:
        print(f"no .cif files in {CIF_DIR}")
        return

    print(f"CIF folder : {CIF_DIR} ({len(cif_files)} files)")
    print(f"output     : {OUT_DIR}")
    print(f"wavelength : {WAVELENGTH} A, 2theta {TWO_THETA_RANGE[0]}-{TWO_THETA_RANGE[1]} deg")

    xrd_calc = XRDCalculator(wavelength=WAVELENGTH)
    npz_dict, label_rows, failed = {}, [], []

    t0 = time.time()
    for idx, cif_path in enumerate(cif_files, 1):
        cif_id = Path(cif_path).stem
        if idx == 1 or idx % PROGRESS_EVERY == 0 or idx == len(cif_files):
            rate = idx / max(1e-9, time.time() - t0)
            eta = (len(cif_files) - idx) / max(1e-9, rate) / 60.0
            print(f"  [{idx:>5}/{len(cif_files)}] {rate:5.1f} CIF/s  ETA {eta:5.1f} min  "
                  f"ok={len(label_rows)} failed={len(failed)}")
        try:
            r = process_one(cif_path, xrd_calc)
        except Exception as e:
            failed.append((os.path.basename(cif_path), f"{type(e).__name__}: {e}"))
            continue

        npz_dict[f"{cif_id}__tt"] = r["tt"]
        npz_dict[f"{cif_id}__ii"] = r["ii"]
        lat = r["structure"].lattice
        label_rows.append({
            "cif_id": cif_id,
            "label_int": r["label_int"],
            "label_name": LABEL_NAMES[r["label_int"]],
            "formula": r["structure"].composition.reduced_formula,
            "n_peaks": len(r["tt"]),
            "space_group": r["sg_sym"],
            "sg_number": r["sg_no"],
            "a": round(float(lat.a), 4),
            "b": round(float(lat.b), 4),
            "c": round(float(lat.c), 4),
            "alpha": round(float(lat.alpha), 3),
            "beta": round(float(lat.beta), 3),
            "gamma": round(float(lat.gamma), 3),
            "volume": round(float(lat.volume), 3),
        })

    print(f"\nok {len(label_rows)}/{len(cif_files)}, failed {len(failed)}, "
          f"{(time.time() - t0) / 60:.1f} min")

    npz_path = os.path.join(OUT_DIR, CACHE_NPZ_NAME)
    (np.savez_compressed if COMPRESS_NPZ else np.savez)(npz_path, **npz_dict)
    print(f"saved {npz_path} ({os.path.getsize(npz_path) / 1024 / 1024:.1f} MB)")

    labels = pd.DataFrame(label_rows).sort_values("cif_id").reset_index(drop=True)
    labels_path = os.path.join(OUT_DIR, LABEL_CSV_NAME)
    labels.to_csv(labels_path, index=False, encoding="utf-8-sig")
    print(f"saved {labels_path}")

    print("\ncrystal systems:")
    counts = labels["label_int"].value_counts()
    for k in range(7):
        n = int(counts.get(k, 0))
        print(f"  [{k}] {LABEL_NAMES[k]:<13} {n:>6} ({100 * n / max(len(labels), 1):5.1f}%)")

    if failed:
        log_path = os.path.join(OUT_DIR, FAIL_LOG_NAME)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"failed: {len(failed)}\n\n")
            for name, err in failed:
                f.write(f"{name}\n  {err}\n\n")
        print(f"\nfailed files listed in {log_path}")


if __name__ == "__main__":
    main()
