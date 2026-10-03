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
print("LOG-TRANSFORM EEG DIAGNOSTIC")
print("=" * 70)

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))


# ============================================================
# ORIGINAL EEG
# ============================================================

X_train = train_df[eeg_cols].fillna(0).values
X_test = test_df[eeg_cols].fillna(0).values


scaler_original = StandardScaler()

X_train_original = scaler_original.fit_transform(X_train)
X_test_original = scaler_original.transform(X_test)


clf_original = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

clf_original.fit(
    X_train_original,
    y_train
)

pred_original = clf_original.predict(
    X_test_original
)


print("\n" + "=" * 70)
print("ORIGINAL EEG")
print("=" * 70)

print(
    "Accuracy:",
    accuracy_score(y_test, pred_original)
)

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(
        y_test,
        pred_original
    )
)

print(
    "Macro-F1:",
    f1_score(
        y_test,
        pred_original,
        average="macro"
    )
)

print(
    "MCC:",
    matthews_corrcoef(
        y_test,
        pred_original
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        pred_original
    )
)


# ============================================================
# LOG TRANSFORM EEG
# ============================================================

X_train_log = np.log1p(
    np.maximum(X_train, 0)
)

X_test_log = np.log1p(
    np.maximum(X_test, 0)
)


scaler_log = StandardScaler()

X_train_log = scaler_log.fit_transform(
    X_train_log
)

X_test_log = scaler_log.transform(
    X_test_log
)


clf_log = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

clf_log.fit(
    X_train_log,
    y_train
)

pred_log = clf_log.predict(
    X_test_log
)


print("\n" + "=" * 70)
print("LOG1P + STANDARD SCALING EEG")
print("=" * 70)

print(
    "Accuracy:",
    accuracy_score(y_test, pred_log)
)

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(
        y_test,
        pred_log
    )
)

print(
    "Macro-F1:",
    f1_score(
        y_test,
        pred_log,
        average="macro"
    )
)

print(
    "MCC:",
    matthews_corrcoef(
        y_test,
        pred_log
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        pred_log
    )
)


# ============================================================
# SUBJECT-WISE LOG TRANSFORM PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT-WISE LOG TRANSFORM RESULTS")
print("=" * 70)


test_subject_ids = test_df["subject_id"].values


for subject in TEST_SUBJECTS:

    mask = test_subject_ids == subject

    subject_true = y_test[mask]
    subject_pred = pred_log[mask]

    print(
        f"Subject {subject}: "
        f"Accuracy={accuracy_score(subject_true, subject_pred):.4f}, "
        f"Balanced={balanced_accuracy_score(subject_true, subject_pred):.4f}, "
        f"Macro-F1={f1_score(subject_true, subject_pred, average='macro'):.4f}"
    )


print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)