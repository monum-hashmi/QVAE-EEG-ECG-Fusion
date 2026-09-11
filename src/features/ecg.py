import numpy as np
from scipy.signal import find_peaks

def extract_ecg_hrv(ecg_window, fs):
    """Basic ECG/HRV features from one ECG window.

    This is intentionally simple and transparent. Later, we can replace it with
    neurokit2 or a stronger ECG processing pipeline after dataset audit.
    """
    ecg_window = np.asarray(ecg_window).ravel()
    features = {}

    if len(ecg_window) < fs * 2:
        return {
            "ecg_mean": float(np.mean(ecg_window)),
            "ecg_std": float(np.std(ecg_window)),
            "hr_mean": np.nan,
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "rmssd": np.nan,
            "num_peaks": 0,
        }

    distance = int(0.35 * fs)
    peaks, _ = find_peaks(ecg_window, distance=distance)

    rr = np.diff(peaks) / fs

    features["ecg_mean"] = float(np.mean(ecg_window))
    features["ecg_std"] = float(np.std(ecg_window))
    features["num_peaks"] = int(len(peaks))

    if len(rr) > 1:
        features["rr_mean"] = float(np.mean(rr))
        features["rr_std"] = float(np.std(rr))
        features["rmssd"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        features["hr_mean"] = float(60.0 / np.mean(rr))
    else:
        features["rr_mean"] = np.nan
        features["rr_std"] = np.nan
        features["rmssd"] = np.nan
        features["hr_mean"] = np.nan

    return features
