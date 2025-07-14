import os
import numpy as np
import mne


def parse_summary_file(summary_path):
    """
    Parse CHB-MIT summary text file to extract seizure intervals per EDF file.

    Parameters
    ----------
    summary_path : str
        Path to the summary text file (e.g., chb01-summary.txt).

    Returns
    -------
    seizure_annotations : dict
        Dictionary mapping EDF filenames to lists of (start_sec, end_sec) tuples.
    """
    seizure_annotations = {}
    with open(summary_path, 'r') as f:
        lines = [line.strip() for line in f]

    i = 0
    while i < len(lines):
        line = lines[i]
        # New file entry
        if line.startswith("File Name:"):
            fname = line.split(":", 1)[1].strip()
            seizure_annotations[fname] = []
            i += 1
            # Advance until Number of Seizures
            while i < len(lines) and "Number of Seizures in File:" not in lines[i]:
                i += 1
            if i >= len(lines):
                break
            # Read number of seizures
            n_seizures = int(lines[i].split(":", 1)[1].strip())
            i += 1
            # Read each seizure start/end
            for _ in range(n_seizures):
                # Find start line
                while i < len(lines) and not lines[i].startswith("Seizure Start Time:"):
                    i += 1
                if i >= len(lines):
                    break
                start_sec = int(lines[i].split(":", 1)[1].strip().split()[0])
                i += 1
                # Find end line
                while i < len(lines) and not lines[i].startswith("Seizure End Time:" ):
                    i += 1
                if i >= len(lines):
                    break
                end_sec = int(lines[i].split(":", 1)[1].strip().split()[0])
                seizure_annotations[fname].append((start_sec, end_sec))
                i += 1
        else:
            i += 1
    return seizure_annotations


def match_bipolar_channels(raw):
    """
    Rename CHB-MIT bipolar channels to approximate standard 10-20 montage names,
    then set the standard_1020 montage on the Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with channel names like 'FP1-F7', etc.

    Returns
    -------
    raw : mne.io.Raw
        Raw object with renamed channels and montage applied.
    """
    # Mapping bipolar to approximate 10-20 unipolar
    mapping = {
        'FP1-F7': 'Fp1', 'F7-T7': 'F7', 'T7-P7': 'T7', 'P7-O1': 'P7',
        'FP1-F3': 'F3', 'F3-C3': 'C3', 'C3-P3': 'P3', 'P3-O1': '01',
        'FP2-F4': 'Fp2', 'F4-C4': 'F4', 'C4-P4': 'C4', 'P4-O2': 'P4',
        'FP2-F8': 'F8', 'F8-T8': 'T8', 'T8-P8': 'P8', 'P8-O2': '02',
        'FZ-CZ': 'Fz', 'CZ-PZ': 'Pz',
        'FT9-FT10': 'T9', 'FT10-T8': 'T10'
    }
    # Only rename channels present in raw
    available = set(raw.ch_names)
    rename_dict = {orig: new for orig, new in mapping.items() if orig in available}
    raw.rename_channels(rename_dict)
    # Apply standard montage (ignore missing)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')
    return raw


def epoch_and_label(raw, seizure_times, epoch_sec=2.0, overlap_thresh=1.0):
    """
    Segment raw EEG into non-overlapping epochs and label each epoch.

    Parameters
    ----------
    raw : mne.io.Raw
        ICA-cleaned, filtered Raw object.
    seizure_times : list of tuple
        List of (start_sec, end_sec) pairs for seizures in this recording.
    epoch_sec : float
        Length of each epoch in seconds.
    overlap_thresh : float
        Minimum overlap in seconds between epoch and any seizure to label as seizure.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels, n_times)
    Y : np.ndarray, shape (n_epochs,)
        Binary labels (0 = non-seizure, 1 = seizure).
    """
    sfreq = raw.info['sfreq']
    epoch_samples = int(epoch_sec * sfreq)
    step = epoch_samples  # non-overlapping windows

    data = raw.get_data()  # shape (n_channels, n_times)
    n_channels, n_samples = data.shape

    X = []
    Y = []
    # iterate start indices
    for start in range(0, n_samples - epoch_samples + 1, step):
        stop = start + epoch_samples
        epoch = data[:, start:stop]
        # compute epoch time window in sec
        t_start = start / sfreq
        t_end = stop / sfreq
        # determine label based on overlap
        label = 0
        for s_on, s_off in seizure_times:
            overlap = min(t_end, s_off) - max(t_start, s_on)
            if overlap > overlap_thresh:
                label = 1
                break
        X.append(epoch)
        Y.append(label)
    X = np.stack(X)
    Y = np.array(Y, dtype=int)
    return X, Y

