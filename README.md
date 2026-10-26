# MK-NPCNN

Code for crystal system classification (7 classes) from powder XRD patterns
with a multi-kernel version of NPCNN. The models are trained on synthetic
patterns and tested on a synthetic test set and on 908 RRUFF patterns from
Salgado et al. (2023).

Models:
- `npcnn_valid`: original NPCNN from Salgado et al. (no padding)
- `npcnn`: NPCNN with padding and centre cropping
- `mknpcnn`: multi-kernel NPCNN, three kernel sizes per stage

## Files

| File | Description |
|---|---|
| `config.py` | paths and settings |
| `build_cif_cache.py` | peak list of every CIF (pymatgen), only needed to rebuild the Zenodo files |
| `generate_data.py` | synthetic XRD data generation |
| `check_data.py` | split check, class counts, peak widths, model size |
| `prepare_rruff.py` | rescales the RRUFF test set to max = 100 |
| `profile_match.py` | peak profile comparison, RRUFF vs synthetic |
| `models.py` | NPCNN and MK-NPCNN |
| `data.py` | dataset and data checks |
| `train.py` | training |
| `analyze.py` | evaluation, statistics, ablation, robustness, plots |
| `robustness_stats.py` | bootstrap of the F1 difference under perturbation |
| `summarize_results.py` | Table S2 and summary tables |
| `plot_fig3.py` | Figure 3 |

## Data

The data are not included in this repository.

- Peak lists: `cif.npz` and `cif_labels.csv` are available on Zenodo,
  https://doi.org/10.5281/zenodo.23661621. Download both files and put them in
  `data/cache/`. They were calculated from CIF files of the American Mineralogist
  Crystal Structure Database (AMCSD, Downs & Hall-Wallace, 2003), downloaded in
  January 2026, with `build_cif_cache.py` and pymatgen 2025.10.7 (Cu Ka1,
  2theta 5-90 deg). `cif_labels.csv` lists every structure used, with its
  crystal system label. To rebuild the files from the CIFs, put the CIFs in
  `data/cif/` and run `python build_cif_cache.py`.
- Synthetic patterns: generated from the peak lists with `generate_data.py`
  (about 75 min on CPU, several GB on disk).
- RRUFF: 908 experimental patterns (`X_rruff_test.npy`, `y_rruff_test.npy`)
  obtained from Salgado et al. (https://github.com/slerman12/XRDs). The files have
  no metadata (no mineral name or RRUFF ID per pattern), so each pattern is
  treated as one structure in the bootstrap. Put both files in `data/rruff/`.

Paths can also be set with environment variables:

```bash
export XRD_CIF_DIR=/path/to/cif
export XRD_CACHE_DIR=/path/to/cache
export XRD_CACHE_NPZ=/path/to/cif.npz
export XRD_LABEL_CSV=/path/to/cif_labels.csv
export XRD_OUT_DIR=/path/to/datasynth
export MKNPCNN_ROOT=/path/to/datasynth/D4_measurement
```

## How to run

```bash
pip install -r requirements.txt

# data (cif.npz and cif_labels.csv from Zenodo in data/cache/)
python generate_data.py
python check_data.py
python prepare_rruff.py
python profile_match.py \
    --synthetic data/datasynth/D4_measurement/X_train.npy \
    --synthetic_y data/datasynth/D4_measurement/y_train.npy \
    --experimental data/datasynth/D4_measurement/MKNPCNN_study/X_rruff_scaled.npy \
    --experimental_y data/datasynth/D4_measurement/MKNPCNN_study/y_rruff.npy

# training (3 models x 3 seeds)
python train.py --arch all

# analysis
python analyze.py --what eval            # run this first
python analyze.py --what stats
python analyze.py --what ablation --sets test
python analyze.py --what probe
python analyze.py --what robust --sets test
python analyze.py --what diagnostic --sets test
python analyze.py --what viz
python robustness_stats.py

# tables and figures
python summarize_results.py
python plot_fig3.py
```

Outputs are saved in `data/datasynth/D4_measurement/MKNPCNN_study/`.

Tested with Python 3.10 and PyTorch 2.4.1 (CUDA 12.1).

## References

Hapsari, A. W. D. (2026). Calculated XRD peak lists of 20,031 AMCSD mineral
structures [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23661621


Downs, R. T., & Hall-Wallace, M. (2003). The American Mineralogist Crystal
Structure Database. American Mineralogist, 88, 247-250.

Salgado, J. E., Lerman, S., Du, Z., Xu, C., & Abdolrahim, N. (2023). Automated
classification of big X-ray diffraction data using deep learning models.
npj Computational Materials, 9, 222. https://doi.org/10.1038/s41524-023-01164-8
