"""Settings for training and analysis"""
#library
import os
#Dataset and Output Path Configuration
ROOT = os.environ.get("MKNPCNN_ROOT",
                      os.path.join("data", "datasynth", "D4_measurement"))
OUT_ROOT = os.path.join(ROOT, "MKNPCNN_study")

X_TRAIN = os.path.join(ROOT, "X_train.npy")
Y_TRAIN = os.path.join(ROOT, "y_train.npy")
META_TRAIN = os.path.join(ROOT, "meta_train.csv")

X_VAL = os.path.join(ROOT, "X_val.npy")
Y_VAL = os.path.join(ROOT, "y_val.npy")
META_VAL = os.path.join(ROOT, "meta_val.csv")

X_TEST = os.path.join(ROOT, "X_test.npy")
Y_TEST = os.path.join(ROOT, "y_test.npy")
META_TEST = os.path.join(ROOT, "meta_test.csv")

# RRUFF test set after prepare_rruff.py (rescaled to max = 100)
X_RRUFF = os.path.join(OUT_ROOT, "X_rruff_scaled.npy")
Y_RRUFF = os.path.join(OUT_ROOT, "y_rruff.npy")
META_RRUFF = None   # no per-pattern metadata for RRUFF

STRUCTURE_ID_COL = "cif_id"

#XRD Input and Class Configuration
INPUT_LEN = 8500
TWO_THETA_START = 5.0
TWO_THETA_STEP = 0.01
NUM_CLASSES = 7

#scaling const
SCALE_MODE = "constant"
SCALE_CONST = 100.0

#labels
LABEL_NAMES = {0: "triclinic", 1: "monoclinic", 2: "orthorhombic", 3: "tetragonal",
               4: "trigonal", 5: "hexagonal", 6: "cubic"}
LABEL_LIST = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]

#training and evaluation conf

SEEDS = [0, 1, 2]
EPOCHS = 200
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0     
GRAD_CLIP = 1.0
PATIENCE = 20             # early stopping
SCHED_PATIENCE = 5
EVAL_BATCH = 256
NUM_WORKERS = 0

#architecture configuration
DROPOUT_CONV = 0.3
DROPOUT_MLP = 0.5         # value from salgado et al.
CHANNELS = 80

# Baseline kernels/strides (Salgado et al. NPCNN)
BASE_STAGES = [(100, 5), (50, 5), (25, 2)]

# MK-NPCNN kernels per stage, chosen so the parameter count is close to the baseline
MK_STAGES = [((23, 69, 207), 5), ((11, 35, 105), 5), ((5, 17, 53), 2)]
MK_BRANCH_CH = (27, 27, 26)

# npcnn_valid : original NPCNN (Salgado et al.), no padding
# npcnn       : NPCNN with the same padding/crop as MK-NPCNN
# mknpcnn     : proposed model
# 3 models x 3 seeds = 9 runs
MODELS = ["npcnn_valid", "npcnn", "mknpcnn"]

# (baseline, candidate) pairs for the statistical tests
COMPARISONS = [("npcnn_valid", "mknpcnn"), ("npcnn", "mknpcnn"), ("npcnn_valid", "npcnn")]
#analysis
PROBE_FWHM_DEG = [0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2, 2.0]
BROADEN_FWHM_DEG = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
NOISE_SNR = [50, 20, 10, 5]
SHIFT_DEG = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]
N_BOOTSTRAP = 10000
N_RANDOM_MASK_REPEATS = 20