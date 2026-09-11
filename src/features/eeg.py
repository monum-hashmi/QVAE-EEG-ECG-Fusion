import numpy as np
from scipy.signal import welch

EEG_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

def bandpower(signal, fs, band, nperseg=None):
    """Compute bandpower using Welch PSD.

    signal shape: (n_samples,)
    """
    low, high = band
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return np.trapezoid(psd[mask], freqs[mask])

def extract_eeg_bandpower(eeg_window, fs):
    """Extract channel-wise EEG bandpower.

    eeg_window shape: (n_channels, n_samples)
    returns dict of features.
    """
    eeg_window = np.asarray(eeg_window)
    if eeg_window.ndim != 2:
        raise ValueError("eeg_window must have shape (n_channels, n_samples)")

    features = {}
    for ch in range(eeg_window.shape[0]):
        for band_name, band_range in EEG_BANDS.items():
            features[f"eeg_ch{ch:02d}_{band_name}"] = bandpower(eeg_window[ch], fs, band_range)
    return features
