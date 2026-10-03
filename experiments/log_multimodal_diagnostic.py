import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


CSV = "data/dreamer_features/dreamer_valence_paired_all.csv"

TRAIN_SUBJECTS = list(range(1, 19))
TEST_SUBJECTS = list(range(19, 24))


df = pd.read_csv(CSV)

df = df[
    df["pairing"] == "correct"
].reset_index(drop=True)


eeg_cols = [
    c for c in df.columns
    if c.startswith("eeg_")
]

ecg_cols = [
    c for c in df.columns
    if c.startswith("ecg_")
]


train_df = df[
    df["subject_id"].isin(TRAIN_SUBJECTS)
].copy()

test_df = df[
    df["subject_id"].isin(TEST_SUBJECTS)
].copy()


y_train = train_df["label"].values
y_test = test_df["label"].values


print("=" * 70)
print("LOG-NORMALISED MULTIMODAL DIAGNOSTIC")
print("=" * 70)

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))


# ============================================================
# EEG
# ============================================================

X_train_eeg = train_df[eeg_cols].fillna(0).values
X_test_eeg = test_df[eeg_cols].fillna(0).values


X_train_eeg = np.log1p(
    np.maximum(X_train_eeg, 0)
)

X_test_eeg = np.log1p(
    np.maximum(X_test_eeg, 0)
)


eeg_scaler = StandardScaler()

X_train_eeg = eeg_scaler.fit_transform(
    X_train_eeg
)

X_test_eeg = eeg_scaler.transform(
    X_test_eeg
)


# ============================================================
# ECG
# ============================================================

X_train_ecg = train_df[ecg_cols].fillna(0).values
X_test_ecg = test_df[ecg_cols].fillna(0).values


# ECG already has much smaller distribution shift,
# so only standardise it.

ecg_scaler = StandardScaler()

X_train_ecg = ecg_scaler.fit_transform(
    X_train_ecg
)

X_test_ecg = ecg_scaler.transform(
    X_test_ecg
)


# ============================================================
# Combined EEG + ECG
# ============================================================

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


print("\n" + "=" * 70)
print("LOG1P EEG + STANDARDISED ECG")
print("=" * 70)

print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(
        y_test,
        pred
    )
)

print(
    "Macro-F1:",
    f1_score(
        y_test,
        pred,
        average="macro"
    )
)

print(
    "MCC:",
    matthews_corrcoef(
        y_test,
        pred
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        pred
    )
)


# ============================================================
# Subject-wise
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT-WISE RESULTS")
print("=" * 70)


subject_ids = test_df["subject_id"].values


for subject in TEST_SUBJECTS:

    mask = subject_ids == subject

    true_subject = y_test[mask]
    pred_subject = pred[mask]

    print(
        f"Subject {subject}: "
        f"Accuracy={accuracy_score(true_subject, pred_subject):.4f}, "
        f"Balanced={balanced_accuracy_score(true_subject, pred_subject):.4f}, "
        f"Macro-F1={f1_score(true_subject, pred_subject, average='macro'):.4f}"
    )


print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)