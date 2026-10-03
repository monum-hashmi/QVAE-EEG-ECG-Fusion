import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


CSV = "data/dreamer_features/dreamer_valence_paired_all.csv"


print("=" * 70)
print("DREAMER DATASET DIAGNOSTIC")
print("=" * 70)


df = pd.read_csv(CSV)


# ============================================================
# 1. Basic dataset information
# ============================================================

print("\nDataset shape:")
print(df.shape)


print("\nColumns:")
print(
    list(df.columns)
)


print("\nPairing distribution:")
print(
    df["pairing"].value_counts()
)


# ============================================================
# 2. Keep correct pairings
# ============================================================

df = df[
    df["pairing"] == "correct"
].reset_index(drop=True)


print("\nCorrect-pair dataset:")
print(
    df.shape
)


# ============================================================
# 3. Subject distribution
# ============================================================

print("\nSamples per subject:")

print(
    df.groupby("subject_id").size()
)


# ============================================================
# 4. Label distribution per subject
# ============================================================

print("\nLabel distribution per subject:")

label_table = (
    df.groupby(
        ["subject_id", "label"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    label_table
)


# ============================================================
# 5. Trial distribution
# ============================================================

print("\nTrials per subject:")

print(
    df.groupby("subject_id")["trial_id"]
    .nunique()
)


# ============================================================
# 6. Windows per trial
# ============================================================

windows_per_trial = (
    df.groupby(
        ["subject_id", "trial_id"]
    )["window_id"]
    .nunique()
)

print(
    "\nWindows per trial:"
)

print(
    windows_per_trial.describe()
)


# ============================================================
# 7. Feature columns
# ============================================================

eeg_cols = [
    c for c in df.columns
    if c.startswith("eeg_")
]

ecg_cols = [
    c for c in df.columns
    if c.startswith("ecg_")
]


print(
    "\nNumber of EEG features:",
    len(eeg_cols)
)

print(
    "Number of ECG features:",
    len(ecg_cols)
)


# ============================================================
# 8. Missing values
# ============================================================

print(
    "\nMissing EEG values:",
    df[eeg_cols].isna().sum().sum()
)

print(
    "Missing ECG values:",
    df[ecg_cols].isna().sum().sum()
)


# ============================================================
# 9. Constant / near-constant features
# ============================================================

eeg_variance = (
    df[eeg_cols]
    .fillna(0)
    .var()
)

ecg_variance = (
    df[ecg_cols]
    .fillna(0)
    .var()
)


print(
    "\nConstant EEG features:",
    int((eeg_variance == 0).sum())
)

print(
    "Constant ECG features:",
    int((ecg_variance == 0).sum())
)


# ============================================================
# 10. Subject-independent simple classification
# ============================================================

TRAIN_SUBJECTS = list(range(1, 19))

TEST_SUBJECTS = list(range(19, 24))


train_df = df[
    df.subject_id.isin(TRAIN_SUBJECTS)
].copy()


test_df = df[
    df.subject_id.isin(TEST_SUBJECTS)
].copy()


print(
    "\nTrain samples:",
    len(train_df)
)

print(
    "Test samples:",
    len(test_df)
)


# ============================================================
# 11. EEG logistic regression
# ============================================================

print("\n" + "=" * 70)
print("EEG LOGISTIC REGRESSION")
print("=" * 70)


X_train = (
    train_df[eeg_cols]
    .fillna(0)
    .values
)

X_test = (
    test_df[eeg_cols]
    .fillna(0)
    .values
)


y_train = train_df["label"].values

y_test = test_df["label"].values


eeg_scaler = StandardScaler()

X_train = eeg_scaler.fit_transform(
    X_train
)

X_test = eeg_scaler.transform(
    X_test
)


clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)


clf.fit(
    X_train,
    y_train
)


pred = clf.predict(
    X_test
)


print(
    "\nAccuracy:",
    accuracy_score(
        y_test,
        pred
)
)

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(
        y_test,
        pred
    )
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        pred
    )
)


# ============================================================
# 12. ECG logistic regression
# ============================================================

print("\n" + "=" * 70)
print("ECG LOGISTIC REGRESSION")
print("=" * 70)


X_train = (
    train_df[ecg_cols]
    .fillna(0)
    .values
)

X_test = (
    test_df[ecg_cols]
    .fillna(0)
    .values
)


ecg_scaler = StandardScaler()

X_train = ecg_scaler.fit_transform(
    X_train
)

X_test = ecg_scaler.transform(
    X_test
)


clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)


clf.fit(
    X_train,
    y_train
)


pred = clf.predict(
    X_test
)


print(
    "\nAccuracy:",
    accuracy_score(
        y_test,
        pred
    )
)

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(
        y_test,
        pred
    )
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        pred
    )
)


# ============================================================
# 13. Combined EEG + ECG logistic regression
# ============================================================

print("\n" + "=" * 70)
print("COMBINED EEG + ECG LOGISTIC REGRESSION")
print("=" * 70)


X_train_eeg = (
    train_df[eeg_cols]
    .fillna(0)
    .values
)

X_test_eeg = (
    test_df[eeg_cols]
    .fillna(0)
    .values
)


X_train_ecg = (
    train_df[ecg_cols]
    .fillna(0)
    .values
)

X_test_ecg = (
    test_df[ecg_cols]
    .fillna(0)
    .values
)


eeg_scaler = StandardScaler()

ecg_scaler = StandardScaler()


X_train_eeg = eeg_scaler.fit_transform(
    X_train_eeg
)

X_test_eeg = eeg_scaler.transform(
    X_test_eeg
)


X_train_ecg = ecg_scaler.fit_transform(
    X_train_ecg
)

X_test_ecg = ecg_scaler.transform(
    X_test_ecg
)


X_train = np.concatenate(
    [
        X_train_eeg,
        X_train_ecg
    ],
    axis=1
)


X_test = np.concatenate(
    [
        X_test_eeg,
        X_test_ecg
    ],
    axis=1
)


clf = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)


clf.fit(
    X_train,
    y_train
)


pred = clf.predict(
    X_test
)


print(
    "\nAccuracy:",
    accuracy_score(
        y_test,
        pred
    )
)

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(
        y_test,
        pred
    )
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        pred
    )
)


# ============================================================
# 14. Exact duplicate feature rows
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)


feature_cols = eeg_cols + ecg_cols


duplicate_count = (
    df.duplicated(
        subset=feature_cols
    ).sum()
)


print(
    "Exact duplicate feature rows:",
    duplicate_count
)


# ============================================================
# 15. Duplicate metadata
# ============================================================

print(
    "\nDuplicate subject/trial/window combinations:"
)

metadata_duplicates = (
    df.duplicated(
        subset=[
            "subject_id",
            "trial_id",
            "window_id"
        ]
    ).sum()
)


print(
    metadata_duplicates
)


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)