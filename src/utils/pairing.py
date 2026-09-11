import numpy as np
import pandas as pd

def make_wrong_trial_pairs(df, subject_col="subject_id", trial_col="trial_id", random_state=42):
    """Create a metadata table for same-subject wrong-trial pairing."""
    rng = np.random.default_rng(random_state)
    pairs = []

    for subject, sdf in df.groupby(subject_col):
        trials = sdf[trial_col].unique()
        if len(trials) < 2:
            continue

        for _, row in sdf.iterrows():
            candidate_trials = [t for t in trials if t != row[trial_col]]
            paired_trial = rng.choice(candidate_trials)
            candidates = sdf[sdf[trial_col] == paired_trial]
            paired_row = candidates.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

            pairs.append({
                "eeg_row_id": row.name,
                "ecg_row_id": paired_row.name,
                "pairing": "wrong_trial",
            })

    return pd.DataFrame(pairs)

def make_cross_subject_pairs(df, subject_col="subject_id", label_col="label", label_matched=True, random_state=42):
    """Create cross-subject pair mapping.

    If label_matched=True, ECG row is sampled from another subject with same label.
    """
    rng = np.random.default_rng(random_state)
    pairs = []

    for _, row in df.iterrows():
        candidates = df[df[subject_col] != row[subject_col]]
        if label_matched:
            candidates = candidates[candidates[label_col] == row[label_col]]

        if candidates.empty:
            continue

        paired_row = candidates.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        pairs.append({
            "eeg_row_id": row.name,
            "ecg_row_id": paired_row.name,
            "pairing": "cross_subject_label_matched" if label_matched else "cross_subject",
        })

    return pd.DataFrame(pairs)
