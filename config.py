"""Shared settings. Every notebook imports from here so paths and choices stay identical.

Change a value here (via its owner), never inside your own notebook.
"""

# ---- Paths (same for everyone once the Drive shortcut is added) ----
PROJECT_DIR = "/content/drive/MyDrive/BT4222_Project"
DATA_DIR = f"{PROJECT_DIR}/data"
RAW_DIR = f"{DATA_DIR}/raw"              # original MIND-large files
PROCESSED_DIR = f"{DATA_DIR}/processed"  # parquet outputs of step1
OUTPUT_DIR = f"{PROJECT_DIR}/outputs"    # figures, result tables
SCORES_DIR = f"{DATA_DIR}/scores"   # per-candidate score files, one per model (dev split)

TRAIN_DIR = f"{RAW_DIR}/MINDlarge_train"
DEV_DIR = f"{RAW_DIR}/MINDlarge_dev"
TEST_DIR = f"{RAW_DIR}/MINDlarge_test"

# ---- Splits (from the interim report, Section 2.1) ----
FIT_DATES = ("2019-11-09", "2019-11-13")   # model fitting
TUNE_DATE = "2019-11-14"                   # hyperparameter tuning / model selection
# Final evaluation = the official dev split (15 Nov 2019)

# ---- Modelling choices ----
SEED = 42
NEG_PER_POS = 4          # within-impression negative sampling for training
COLD_THRESHOLD = 10      # article is cold if it has < 10 prior impressions
COLD_REPORT_START = "2019-11-10"  # 9 Nov excluded from cold/warm reporting (counts start at 0)

# ---- Evaluation / re-ranking ----
TOP_K = 10
MMR_LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]   # soft MMR re-ranking; 1.0 = relevance only
MAX_PER_CATEGORY = 4     # optional category cap; use as a soft penalty if kept
SIM_THRESHOLD = 0.85     # DEPRECATED, replaced by MMR_LAMBDAS; remove once P6 confirms nothing uses it
