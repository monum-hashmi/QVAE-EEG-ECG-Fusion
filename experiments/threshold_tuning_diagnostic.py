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

# ------------------------------------------------------------
# Three-way subject-independent split
# ------------------------------------------------------------

FIT_SUBJECTS = list(range(1, 15))
THRESHOLD_SUBJECTS = list(range(15, 19))
TEST_SUBJECTS = list(range(19, 24))


print("=" * 70)
print("SUBJECT-INDEPENDENT THRESHOLD TUNING DIAGNOSTIC")
print("=" * 70)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(CSV)

df = df[
    df["pairing"] == "correct"
].reset_index(drop=True)


eeg_cols = [
    c for c in df.columns
    if c.startswith("eeg_")
]


fit_df = df[
    df["subject_id"].isin(FIT_SUBJECTS)
].copy()

threshold_df = df[
    df["subject_id"].isin(THRESHOLD_SUBJECTS)
].copy()

test_df = df[
    df["subject_id"].isin(TEST_SUBJECTS)
].copy()


print("\nSubjects used for model fitting:")
print(FIT_SUBJECTS)

print("\nSubjects used for threshold selection:")
print(THRESHOLD_SUBJECTS)

print("\nUnseen final test subjects:")
print(TEST_SUBJECTS)

print("\nSample counts:")
print("Fit:", len(fit_df))
print("Threshold:", len(threshold_df))
print("Test:", len(test_df))


# ------------------------------------------------------------
# Prepare features
# ------------------------------------------------------------

X_fit = (
    fit_df[eeg_cols]
    .fillna(0)
    .values
)

X_threshold = (
    threshold_df[eeg_cols]
    .fillna(0)
    .values
)

X_test = (
    test_df[eeg_cols]
    .fillna(0)
    .values
)


# ------------------------------------------------------------
# Log1p transformation
# ------------------------------------------------------------

X_fit = np.log1p(
    np.maximum(X_fit, 0)
)

X_threshold = np.log1p(
    np.maximum(X_threshold, 0)
)

X_test = np.log1p(
    np.maximum(X_test, 0)
)


# ------------------------------------------------------------
# Fit scaler ONLY on fitting subjects
# ------------------------------------------------------------

scaler = StandardScaler()

X_fit = scaler.fit_transform(
    X_fit
)

X_threshold = scaler.transform(
    X_threshold
)

X_test = scaler.transform(
    X_test
)


y_fit = fit_df["label"].values
y_threshold = threshold_df["label"].values
y_test = test_df["label"].values


# ------------------------------------------------------------
# Train classifier
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)


clf = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

clf.fit(
    X_fit,
    y_fit
)


# ------------------------------------------------------------
# Probability predictions
# ------------------------------------------------------------

threshold_prob = clf.predict_proba(
    X_threshold
)[:, 1]

test_prob = clf.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# Threshold search
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("THRESHOLD SEARCH")
print("=" * 70)


thresholds = np.arange(
    0.10,
    0.91,
    0.01
)


best_threshold = None
best_balanced_accuracy = -1.0


for threshold in thresholds:

    pred = (
        threshold_prob >= threshold
    ).astype(int)

    balanced = balanced_accuracy_score(
        y_threshold,
        pred
    )

    if balanced > best_balanced_accuracy:

        best_balanced_accuracy = balanced
        best_threshold = threshold


print(
    "Selected threshold:",
    round(best_threshold, 2)
)

print(
    "Threshold-validation Balanced Accuracy:",
    round(best_balanced_accuracy, 4)
)


# ------------------------------------------------------------
# Show threshold validation result
# ------------------------------------------------------------

threshold_pred = (
    threshold_prob >= best_threshold
).astype(int)


print("\nThreshold-validation confusion matrix:")
print(
    confusion_matrix(
        y_threshold,
        threshold_pred
    )
)


print(
    "Threshold-validation Macro-F1:",
    round(
        f1_score(
            y_threshold,
            threshold_pred,
            average="macro"
        ),
        4
    )
)

print(
    "Threshold-validation MCC:",
    round(
        matthews_corrcoef(
            y_threshold,
            threshold_pred
        ),
        4
    )
)


# ------------------------------------------------------------
# Final unseen test evaluation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL UNSEEN TEST")
print("=" * 70)


test_pred_default = (
    test_prob >= 0.50
).astype(int)


test_pred_tuned = (
    test_prob >= best_threshold
).astype(int)


# ------------------------------------------------------------
# Default threshold
# ------------------------------------------------------------

print("\nDEFAULT THRESHOLD = 0.50")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            test_pred_default
        ),
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        balanced_accuracy_score(
            y_test,
            test_pred_default
        ),
        4
    )
)

print(
    "Macro-F1:",
    round(
        f1_score(
            y_test,
            test_pred_default,
            average="macro"
        ),
        4
    )
)

print(
    "MCC:",
    round(
        matthews_corrcoef(
            y_test,
            test_pred_default
        ),
        4
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_pred_default
    )
)


# ------------------------------------------------------------
# Tuned threshold
# ------------------------------------------------------------

print(
    "\nTUNED THRESHOLD =",
    round(best_threshold, 2)
)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            test_pred_tuned
        ),
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        balanced_accuracy_score(
            y_test,
            test_pred_tuned
        ),
        4
    )
)

print(
    "Macro-F1:",
    round(
        f1_score(
            y_test,
            test_pred_tuned,
            average="macro"
        ),
        4
    )
)

print(
    "MCC:",
    round(
        matthews_corrcoef(
            y_test,
            test_pred_tuned
        ),
        4
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_pred_tuned
    )
)


# ------------------------------------------------------------
# Subject-wise final results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SUBJECT-WISE TUNED RESULTS")
print("=" * 70)


for subject_id in TEST_SUBJECTS:

    mask = (
        test_df["subject_id"].values
        == subject_id
    )

    subject_true = y_test[mask]
    subject_pred = test_pred_tuned[mask]

    accuracy = accuracy_score(
        subject_true,
        subject_pred
    )

    balanced = balanced_accuracy_score(
        subject_true,
        subject_pred
    )

    macro_f1 = f1_score(
        subject_true,
        subject_pred,
        average="macro"
    )

    print(
        f"Subject {subject_id}: "
        f"Accuracy={accuracy:.4f}, "
        f"Balanced={balanced:.4f}, "
        f"Macro-F1={macro_f1:.4f}"
    )


print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)