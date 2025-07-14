"""
Preprocess CHBNIT EDF files into 2s epochs (raw bandpass or Gaussian-filtered).
Usage:
    python -m preprocessing.prepare_data bandpass
    python -m preprocessing.prepare_data gaussian
"""
import os
import sys
import numpy as np
import mne
from preprocessing.utils import parse_summary_file, match_bipolar_channels, epoch_and_label
from scipy.ndimage import gaussian_filter1d

# --- Configuration ---
FILTER_TYPE = sys.argv[1] if len(sys.argv) > 1 else "bandpass"
RAW_ROOT    = "/Volumes/Samsung_T5/chbnit_data"
BASE_SAVE   = "/Volumes/Samsung_T5/project1"
DATA_DIR    = os.path.join(BASE_SAVE, "data")
EPOCH_DIR_BP= os.path.join(DATA_DIR, "epochs_bandpass")
EPOCH_DIR_GA= os.path.join(DATA_DIR, "epochs_gaussian")

# Create directories
os.makedirs(EPOCH_DIR_BP, exist_ok=True)
os.makedirs(EPOCH_DIR_GA, exist_ok=True)

# Process each subject
for subj in sorted(os.listdir(RAW_ROOT)):
    subj_dir = os.path.join(RAW_ROOT, subj)
    if not os.path.isdir(subj_dir):
        continue

    summary_path = os.path.join(subj_dir, f"{subj}-summary.txt")
    if not os.path.isfile(summary_path):
        continue

    annotations = parse_summary_file(summary_path)
    # Choose output base
    if FILTER_TYPE == "bandpass":
        out_base = EPOCH_DIR_BP
    elif FILTER_TYPE == "gaussian":
        out_base = EPOCH_DIR_GA
    else:
        raise ValueError(f"Unknown FILTER_TYPE: {FILTER_TYPE}")

    out_subj = os.path.join(out_base, subj)
    os.makedirs(out_subj, exist_ok=True)

    print(f"Processing subject: {subj} | Mode: {FILTER_TYPE}")
    for fname, segs in annotations.items():
        edf_path = os.path.join(subj_dir, fname)
        if not os.path.isfile(edf_path):
            continue

        # Load raw EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # Apply filter
        if FILTER_TYPE == "bandpass":
            raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
        else:  # gaussian
            data = raw.get_data()
            # sigma chosen for ~1 Hz smoothing; adjust as needed
            filtered = gaussian_filter1d(data, sigma=2, axis=1)
            raw._data = filtered

        # Standardize channels / montage
        raw = match_bipolar_channels(raw)

        # Epoch & label
        X, Y = epoch_and_label(raw, segs,
                               epoch_sec=2.0,
                               overlap_thresh=1.0)
        X = X.astype(np.float32)

        # Save compressed .npz
        base = fname.replace('.edf', '')
        out_file = os.path.join(out_subj, f"{FILTER_TYPE}_epochs_{base}.npz")
        np.savez_compressed(out_file, X=X, Y=Y)

        print(f"  {base}: {len(Y)} epochs → {Y.sum()} seiz, {len(Y)-Y.sum()} non-seiz")
