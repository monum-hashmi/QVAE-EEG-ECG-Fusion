import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.eeg import extract_eeg_bandpower
from src.features.ecg import extract_ecg_hrv


def to_native(obj):
    if hasattr(obj, "_fieldnames"):
        return {field: to_native(getattr(obj, field)) for field in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.size == 1:
                return to_native(obj.item())
            return [to_native(x) for x in obj.flat]
        return obj
    return obj


def get_ci(dct, names, required=True):
    if not isinstance(dct, dict):
        if required:
            raise TypeError(f"Expected dict while looking for {names}, got {type(dct)}")
        return None
    lower_map = {str(k).lower(): k for k in dct.keys()}
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            return dct[key]
    if required:
        raise KeyError(f"Could not find any of {names}. Available keys: {list(dct.keys())}")
    return None


def scalar(x):
    arr = np.asarray(x).squeeze()
    return float(arr)


def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return [to_native(v) for v in x.flat]
        if x.ndim == 3:
            candidates = [(axis, size) for axis, size in enumerate(x.shape) if 5 <= size <= 60]
            if candidates:
                axis = candidates[0][0]
                return [np.take(x, i, axis=axis) for i in range(x.shape[axis])]
        if x.ndim == 2:
            return [x]
    return [x]


def orient_signal(arr, expected_channels=None, name="signal"):
    arr = np.asarray(arr, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} expected 2D array, got shape {arr.shape}")
    if expected_channels is not None:
        if arr.shape[0] == expected_channels:
            return arr
        if arr.shape[1] == expected_channels:
            return arr.T
    if arr.shape[0] <= arr.shape[1]:
        return arr
    return arr.T


def feature_prefix(features, prefix):
    return {f"{prefix}{k}": v for k, v in features.items()}


def extract_ecg_features(ecg_window, fs):
    ecg_window = np.asarray(ecg_window, dtype=float)
    feats = {}
    for ch in range(ecg_window.shape[0]):
        ch_feats = extract_ecg_hrv(ecg_window[ch], fs)
        for k, v in ch_feats.items():
            feats[f"ecg_ch{ch:02d}_{k}"] = v
    return feats


def choose_partner(df, row, mode, rng):
    if mode == "wrong_trial":
        candidates = df[(df.subject_id == row.subject_id) & (df.trial_id != row.trial_id)]
        matched = candidates[candidates.label == row.label]
        if not matched.empty:
            candidates = matched
    elif mode == "cross_subject_label_matched":
        candidates = df[(df.subject_id != row.subject_id) & (df.label == row.label)]
    elif mode == "cross_subject_unrestricted":
        candidates = df[df.subject_id != row.subject_id]
    else:
        raise ValueError(mode)
    if candidates.empty:
        return None
    idx = int(rng.choice(candidates.index.to_numpy()))
    return df.loc[idx]


def fill_numeric(df):
    feature_cols = [c for c in df.columns if c not in ["subject_id", "trial_id", "window_id", "label", "pairing"]]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(med).fillna(0)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True, help="Path to DREAMER.mat")
    parser.add_argument("--out_dir", default="data/dreamer_features")
    parser.add_argument("--label", default="valence", choices=["valence", "arousal", "dominance"])
    parser.add_argument("--threshold", type=float, default=3.0, help="High label if score > threshold")
    parser.add_argument("--drop_neutral", action="store_true", help="Drop score equal to threshold")
    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--step_sec", type=float, default=2.0)
    parser.add_argument("--max_windows_per_trial", type=int, default=10, help="Use -1 for all windows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mat_path = Path(args.mat)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    native = {k: to_native(v) for k, v in mat.items() if not k.startswith("__")}
    dreamer = get_ci(native, ["DREAMER"])
    if isinstance(dreamer, list) and len(dreamer) == 1:
        dreamer = dreamer[0]

    eeg_fs = int(round(scalar(get_ci(dreamer, ["EEG_SamplingRate", "EEGSamplingRate"]))))
    ecg_fs = int(round(scalar(get_ci(dreamer, ["ECG_SamplingRate", "ECGSamplingRate"]))))
    subjects = as_list(get_ci(dreamer, ["Data"]))

    label_field = {
        "valence": ["ScoreValence", "Valence"],
        "arousal": ["ScoreArousal", "Arousal"],
        "dominance": ["ScoreDominance", "Dominance"],
    }[args.label]

    print(f"EEG fs: {eeg_fs}")
    print(f"ECG fs: {ecg_fs}")
    print(f"Subjects found: {len(subjects)}")
    print(f"Target label: {args.label}, high if score > {args.threshold}")

    base_rows = []
    skipped = 0
    eeg_win_len = int(round(args.window_sec * eeg_fs))
    eeg_step = int(round(args.step_sec * eeg_fs))
    ecg_win_len = int(round(args.window_sec * ecg_fs))
    ecg_step = int(round(args.step_sec * ecg_fs))

    for s_idx, subject in enumerate(subjects, start=1):
        if not isinstance(subject, dict):
            print(f"Skipping subject {s_idx}: unexpected type {type(subject)}")
            continue
        eeg_obj = get_ci(subject, ["EEG"])
        ecg_obj = get_ci(subject, ["ECG"])
        eeg_trials = as_list(get_ci(eeg_obj, ["stimuli", "Stimuli"]))
        ecg_trials = as_list(get_ci(ecg_obj, ["stimuli", "Stimuli"]))
        scores = np.asarray(get_ci(subject, label_field), dtype=float).squeeze()
        scores = np.ravel(scores)
        n_trials = min(len(eeg_trials), len(ecg_trials), len(scores))

        for t_idx in range(n_trials):
            score = float(scores[t_idx])
            if args.drop_neutral and score == args.threshold:
                continue
            y = int(score > args.threshold)
            try:
                eeg = orient_signal(eeg_trials[t_idx], expected_channels=14, name="EEG")
                ecg = orient_signal(ecg_trials[t_idx], expected_channels=2, name="ECG")
            except Exception as e:
                print(f"Skipping subject={s_idx}, trial={t_idx+1}: {e}")
                skipped += 1
                continue
            n_eeg_windows = max(0, 1 + (eeg.shape[1] - eeg_win_len) // eeg_step)
            n_ecg_windows = max(0, 1 + (ecg.shape[1] - ecg_win_len) // ecg_step)
            n_windows = min(n_eeg_windows, n_ecg_windows)
            if args.max_windows_per_trial > 0:
                n_windows = min(n_windows, args.max_windows_per_trial)
            for w_idx in range(n_windows):
                es = w_idx * eeg_step
                cs = w_idx * ecg_step
                eeg_window = eeg[:, es:es + eeg_win_len]
                ecg_window = ecg[:, cs:cs + ecg_win_len]
                eeg_feats = feature_prefix(extract_eeg_bandpower(eeg_window, eeg_fs), "eeg_")
                ecg_feats = extract_ecg_features(ecg_window, ecg_fs)
                base_rows.append({
                    "subject_id": s_idx,
                    "trial_id": t_idx + 1,
                    "window_id": w_idx + 1,
                    "label": y,
                    "eeg_feats": eeg_feats,
                    "ecg_feats": ecg_feats,
                })

    if not base_rows:
        raise RuntimeError("No rows extracted. Run 00b_inspect_dreamer_deep.py and send the output.")

    print(f"Base windows extracted: {len(base_rows)}")
    print(f"Skipped trials/windows: {skipped}")

    meta_df = pd.DataFrame([{k: r[k] for k in ["subject_id", "trial_id", "window_id", "label"]} for r in base_rows])
    eeg_df = pd.concat([meta_df, pd.DataFrame([r["eeg_feats"] for r in base_rows])], axis=1)
    eeg_df["pairing"] = "eeg_only"
    eeg_df = eeg_df[["subject_id", "trial_id", "window_id", "label", "pairing"] + [c for c in eeg_df.columns if c.startswith("eeg_")]]

    ecg_df = pd.concat([meta_df, pd.DataFrame([r["ecg_feats"] for r in base_rows])], axis=1)
    ecg_df["pairing"] = "ecg_only"
    ecg_df = ecg_df[["subject_id", "trial_id", "window_id", "label", "pairing"] + [c for c in ecg_df.columns if c.startswith("ecg_")]]

    correct_df = pd.concat([meta_df, pd.DataFrame([r["eeg_feats"] for r in base_rows]), pd.DataFrame([r["ecg_feats"] for r in base_rows])], axis=1)
    correct_df["pairing"] = "correct"
    feature_cols = [c for c in correct_df.columns if c.startswith("eeg_") or c.startswith("ecg_")]
    correct_df = correct_df[["subject_id", "trial_id", "window_id", "label", "pairing"] + feature_cols]

    base_table = meta_df.copy()
    base_table["row_id"] = base_table.index
    rng = np.random.default_rng(args.seed)
    paired_rows = []
    for mode in ["wrong_trial", "cross_subject_label_matched", "cross_subject_unrestricted"]:
        for _, row in base_table.iterrows():
            partner = choose_partner(base_table, row, mode, rng)
            if partner is None:
                continue
            out = {"subject_id": int(row.subject_id), "trial_id": int(row.trial_id), "window_id": int(row.window_id), "label": int(row.label), "pairing": mode}
            out.update(base_rows[int(row.row_id)]["eeg_feats"])
            out.update(base_rows[int(partner.row_id)]["ecg_feats"])
            paired_rows.append(out)
    paired_controls_df = pd.DataFrame(paired_rows)
    paired_all_df = pd.concat([correct_df, paired_controls_df], ignore_index=True)

    eeg_df = fill_numeric(eeg_df)
    ecg_df = fill_numeric(ecg_df)
    correct_df = fill_numeric(correct_df)
    paired_all_df = fill_numeric(paired_all_df)

    base_name = f"dreamer_{args.label}"
    paths = {
        "eeg_only": out_dir / f"{base_name}_eeg_only.csv",
        "ecg_only": out_dir / f"{base_name}_ecg_only.csv",
        "correct": out_dir / f"{base_name}_correct.csv",
        "paired_all": out_dir / f"{base_name}_paired_all.csv",
    }
    eeg_df.to_csv(paths["eeg_only"], index=False)
    ecg_df.to_csv(paths["ecg_only"], index=False)
    correct_df.to_csv(paths["correct"], index=False)
    paired_all_df.to_csv(paths["paired_all"], index=False)

    print("\nSaved:")
    for name, path in paths.items():
        shape = pd.read_csv(path).shape
        print(f"- {name}: {path} | rows={shape[0]} cols={shape[1]}")
    print("\nLabel distribution in correct set:")
    print(correct_df["label"].value_counts().sort_index())
    print("\nPairing distribution in paired_all:")
    print(paired_all_df["pairing"].value_counts())


if __name__ == "__main__":
    main()
