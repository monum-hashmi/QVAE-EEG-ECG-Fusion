import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CSV = "data/dreamer_features/dreamer_valence_paired_all.csv"
SEED = 42


print("=" * 70)
print("OLD RANDOM-SPLIT LEAKAGE AUDIT")
print("=" * 70)


df = pd.read_csv(CSV)


print("\nFull dataset:")
print("Samples:", len(df))
print("Subjects:", sorted(df["subject_id"].unique()))


# ---------------------------------------------------------
# Reproduce the old dataset filtering behaviour
# ---------------------------------------------------------

print("\nPairing distribution:")
print(df["pairing"].value_counts())


# Old DREAMERDataset did NOT filter pairing.
# Therefore reproduce that exactly.


indices = np.arange(len(df))


train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)


train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()


# ---------------------------------------------------------
# Subject overlap
# ---------------------------------------------------------

train_subjects = set(
    train_df["subject_id"].unique()
)

val_subjects = set(
    val_df["subject_id"].unique()
)

overlap = sorted(
    train_subjects.intersection(val_subjects)
)


print("\n" + "=" * 70)
print("SUBJECT OVERLAP")
print("=" * 70)

print("Training subjects:")
print(sorted(train_subjects))

print("\nValidation subjects:")
print(sorted(val_subjects))

print("\nSubjects appearing in BOTH:")
print(overlap)

print(
    "\nNumber of overlapping subjects:",
    len(overlap)
)


# ---------------------------------------------------------
# Sample counts
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE COUNTS")
print("=" * 70)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# ---------------------------------------------------------
# Overlap per subject
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN / VALIDATION SAMPLES PER SUBJECT")
print("=" * 70)


train_counts = (
    train_df["subject_id"]
    .value_counts()
    .sort_index()
)

val_counts = (
    val_df["subject_id"]
    .value_counts()
    .sort_index()
)


subject_table = pd.DataFrame({
    "train": train_counts,
    "validation": val_counts
}).fillna(0).astype(int)


print(subject_table)


# ---------------------------------------------------------
# Exact subject/trial/window overlap
# ---------------------------------------------------------

keys = [
    "subject_id",
    "trial_id",
    "window_id"
]


train_keys = set(
    map(
        tuple,
        train_df[keys].values
    )
)

val_keys = set(
    map(
        tuple,
        val_df[keys].values
    )
)


exact_overlap = train_keys.intersection(
    val_keys
)


print("\n" + "=" * 70)
print("EXACT SAMPLE OVERLAP")
print("=" * 70)

print(
    "Exact subject/trial/window combinations in both:",
    len(exact_overlap)
)


# ---------------------------------------------------------
# Class distributions
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation:")
print(val_df["label"].value_counts().sort_index())


# ---------------------------------------------------------
# Scaling leakage demonstration
# ---------------------------------------------------------

eeg_cols = [
    c for c in df.columns
    if c.startswith("eeg_")
]

ecg_cols = [
    c for c in df.columns
    if c.startswith("ecg_")
]


# Old behaviour: fit using ALL samples

old_eeg_scaler = StandardScaler()
old_eeg_scaler.fit(
    df[eeg_cols].fillna(0).values
)


old_ecg_scaler = StandardScaler()
old_ecg_scaler.fit(
    df[ecg_cols].fillna(0).values
)


# Correct behaviour: fit TRAIN only

clean_eeg_scaler = StandardScaler()
clean_eeg_scaler.fit(
    train_df[eeg_cols].fillna(0).values
)


clean_ecg_scaler = StandardScaler()
clean_ecg_scaler.fit(
    train_df[ecg_cols].fillna(0).values
)


print("\n" + "=" * 70)
print("SCALER LEAKAGE CHECK")
print("=" * 70)


eeg_mean_difference = np.mean(
    np.abs(
        old_eeg_scaler.mean_
        -
        clean_eeg_scaler.mean_
    )
)


ecg_mean_difference = np.mean(
    np.abs(
        old_ecg_scaler.mean_
        -
        clean_ecg_scaler.mean_
    )
)


print(
    "Mean absolute EEG scaler mean difference:",
    eeg_mean_difference
)

print(
    "Mean absolute ECG scaler mean difference:",
    ecg_mean_difference
)


print("\nCompleted.")