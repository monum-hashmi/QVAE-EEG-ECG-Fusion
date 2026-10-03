import pandas as pd
import numpy as np


CSV = "data/dreamer_features/dreamer_valence_paired_all.csv"

TRAIN_SUBJECTS = list(range(1, 19))
TEST_SUBJECTS = list(range(19, 24))


print("=" * 70)
print("DREAMER SUBJECT DISTRIBUTION SHIFT ANALYSIS")
print("=" * 70)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(CSV)

df = df[
    df["pairing"] == "correct"
].reset_index(drop=True)


print("\nDataset:")
print(df.shape)


# ============================================================
# Feature columns
# ============================================================

eeg_cols = [
    c for c in df.columns
    if c.startswith("eeg_")
]

ecg_cols = [
    c for c in df.columns
    if c.startswith("ecg_")
]


print("\nEEG features:", len(eeg_cols))
print("ECG features:", len(ecg_cols))


# ============================================================
# Train / Test
# ============================================================

train_df = df[
    df["subject_id"].isin(TRAIN_SUBJECTS)
].copy()

test_df = df[
    df["subject_id"].isin(TEST_SUBJECTS)
].copy()


print("\nTrain samples:", len(train_df))
print("Test samples:", len(test_df))


# ============================================================
# Subject-level feature statistics
# ============================================================

def analyse_modality(
    feature_cols,
    modality_name
):

    print("\n" + "=" * 70)
    print(f"{modality_name} DISTRIBUTION SHIFT")
    print("=" * 70)


    # --------------------------------------------------------
    # Calculate subject means
    # --------------------------------------------------------

    train_subject_means = (
        train_df
        .groupby("subject_id")[feature_cols]
        .mean()
    )

    test_subject_means = (
        test_df
        .groupby("subject_id")[feature_cols]
        .mean()
    )


    # --------------------------------------------------------
    # Calculate subject standard deviations
    # --------------------------------------------------------

    train_subject_stds = (
        train_df
        .groupby("subject_id")[feature_cols]
        .std()
    )

    test_subject_stds = (
        test_df
        .groupby("subject_id")[feature_cols]
        .std()
    )


    # --------------------------------------------------------
    # Global training statistics
    # --------------------------------------------------------

    train_mean = (
        train_df[feature_cols]
        .mean()
    )

    train_std = (
        train_df[feature_cols]
        .std()
        .replace(0, np.nan)
    )


    # --------------------------------------------------------
    # Test mean compared with training distribution
    # --------------------------------------------------------

    test_mean = (
        test_df[feature_cols]
        .mean()
    )

    standardized_shift = (
        (test_mean - train_mean)
        .abs()
        / train_std
    )


    # --------------------------------------------------------
    # Subject-level train variability
    # --------------------------------------------------------

    train_subject_mean_std = (
        train_subject_means
        .std()
    )


    # --------------------------------------------------------
    # Test subjects relative to training subjects
    # --------------------------------------------------------

    train_subject_mean_avg = (
        train_subject_means
        .mean()
    )


    test_subject_mean_avg = (
        test_subject_means
        .mean()
    )


    between_subject_shift = (
        (test_subject_mean_avg -
         train_subject_mean_avg)
        .abs()
        / train_std
    )


    # ========================================================
    # Largest global shifts
    # ========================================================

    ranking = pd.DataFrame({

        "feature": feature_cols,

        "train_mean":
            train_mean.values,

        "test_mean":
            test_mean.values,

        "train_std":
            train_std.values,

        "standardized_test_shift":
            standardized_shift.values,

        "between_subject_shift":
            between_subject_shift.values

    })


    ranking = ranking.sort_values(
        "standardized_test_shift",
        ascending=False
    )


    print(
        "\nTop 20 features by train/test standardized shift:"
    )

    print(
        ranking.head(20).to_string(
            index=False
        )
    )


    # ========================================================
    # Summary
    # ========================================================

    shifts = standardized_shift.dropna()

    print("\nShift summary:")

    print(
        "Mean standardized shift:",
        shifts.mean()
    )

    print(
        "Median standardized shift:",
        shifts.median()
    )

    print(
        "Maximum standardized shift:",
        shifts.max()
    )

    print(
        "Features with shift > 0.5:",
        int((shifts > 0.5).sum())
    )

    print(
        "Features with shift > 1.0:",
        int((shifts > 1.0).sum())
    )

    print(
        "Features with shift > 2.0:",
        int((shifts > 2.0).sum())
    )


    # ========================================================
    # Subject-level average shift
    # ========================================================

    print(
        "\nAverage feature shift by test subject:"
    )


    for subject in TEST_SUBJECTS:

        subject_mean = (
            test_subject_means
            .loc[subject]
        )

        subject_shift = (
            (subject_mean - train_mean)
            .abs()
            / train_std
        )

        print(
            f"Subject {subject}: "
            f"mean shift = "
            f"{subject_shift.mean():.4f}, "
            f"median = "
            f"{subject_shift.median():.4f}, "
            f"max = "
            f"{subject_shift.max():.4f}"
        )


    return ranking


# ============================================================
# EEG
# ============================================================

eeg_ranking = analyse_modality(
    eeg_cols,
    "EEG"
)


# ============================================================
# ECG
# ============================================================

ecg_ranking = analyse_modality(
    ecg_cols,
    "ECG"
)


# ============================================================
# Save
# ============================================================

import os

os.makedirs(
    "results/subject_distribution_shift",
    exist_ok=True
)


eeg_ranking.to_csv(
    "results/subject_distribution_shift/"
    "eeg_feature_shift.csv",
    index=False
)


ecg_ranking.to_csv(
    "results/subject_distribution_shift/"
    "ecg_feature_shift.csv",
    index=False
)


print("\n" + "=" * 70)
print("DISTRIBUTION SHIFT ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nSaved:"
)

print(
    "results/subject_distribution_shift/"
    "eeg_feature_shift.csv"
)

print(
    "results/subject_distribution_shift/"
    "ecg_feature_shift.csv"
)