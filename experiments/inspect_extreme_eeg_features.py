import pandas as pd
import numpy as np


CSV = "data/dreamer_features/dreamer_valence_paired_all.csv"

TRAIN_SUBJECTS = list(range(1, 19))
TEST_SUBJECTS = list(range(19, 24))


df = pd.read_csv(CSV)

df = df[
    df["pairing"] == "correct"
].reset_index(drop=True)


features = [
    "eeg_eeg_ch02_beta",
    "eeg_eeg_ch01_beta",
    "eeg_eeg_ch11_beta",
    "eeg_eeg_ch03_beta",
    "eeg_eeg_ch00_beta",
    "eeg_eeg_ch11_gamma",
    "eeg_eeg_ch13_beta",
    "eeg_eeg_ch11_alpha",
    "eeg_eeg_ch10_beta",
    "eeg_eeg_ch02_gamma"
]


print("=" * 70)
print("EXTREME EEG FEATURE INSPECTION")
print("=" * 70)


for feature in features:

    print("\n" + "-" * 70)
    print(feature)
    print("-" * 70)

    stats = (
        df.groupby("subject_id")[feature]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max"
            ]
        )
    )

    print(stats.to_string())


    train_values = df[
        df["subject_id"].isin(TRAIN_SUBJECTS)
    ][feature]

    test_values = df[
        df["subject_id"].isin(TEST_SUBJECTS)
    ][feature]


    print("\nTRAIN aggregate:")
    print(
        f"mean={train_values.mean():.4f}, "
        f"std={train_values.std():.4f}, "
        f"min={train_values.min():.4f}, "
        f"max={train_values.max():.4f}"
    )


    print("\nTEST aggregate:")
    print(
        f"mean={test_values.mean():.4f}, "
        f"std={test_values.std():.4f}, "
        f"min={test_values.min():.4f}, "
        f"max={test_values.max():.4f}"
    )


print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)