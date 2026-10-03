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

BANDS = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma"
]


print("=" * 70)
print("BAND-WISE EEG NORMALISATION DIAGNOSTIC")
print("=" * 70)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV)

df = df[
    df["pairing"] == "correct"
].reset_index(drop=True)


eeg_cols = [
    c for c in df.columns
    if c.startswith("eeg_")
]


train_df = df[
    df["subject_id"].isin(TRAIN_SUBJECTS)
].copy()

test_df = df[
    df["subject_id"].isin(TEST_SUBJECTS)
].copy()


print("Train samples:", len(train_df))
print("Test samples:", len(test_df))
print("EEG features:", len(eeg_cols))


y_train = train_df["label"].values
y_test = test_df["label"].values


# ============================================================
# Helper
# ============================================================

def evaluate(name, X_train, X_test):

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

    accuracy = accuracy_score(
        y_test,
        pred
    )

    balanced = balanced_accuracy_score(
        y_test,
        pred
    )

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )

    mcc = matthews_corrcoef(
        y_test,
        pred
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Balanced Accuracy:",
        round(balanced, 4)
    )

    print(
        "Macro-F1:",
        round(macro_f1, 4)
    )

    print(
        "MCC:",
        round(mcc, 4)
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            pred
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "mcc": mcc
    }


# ============================================================
# Raw feature matrices
# ============================================================

X_train_raw = (
    train_df[eeg_cols]
    .fillna(0)
    .values
)

X_test_raw = (
    test_df[eeg_cols]
    .fillna(0)
    .values
)


# ============================================================
# 1. Existing best: Log1p + global scaling
# ============================================================

X_train_log = np.log1p(
    np.maximum(X_train_raw, 0)
)

X_test_log = np.log1p(
    np.maximum(X_test_raw, 0)
)


global_scaler = StandardScaler()

X_train_global = global_scaler.fit_transform(
    X_train_log
)

X_test_global = global_scaler.transform(
    X_test_log
)


results_global = evaluate(
    "1. LOG1P + GLOBAL SCALING",
    X_train_global,
    X_test_global
)


# ============================================================
# 2. Log1p + band-wise scaling
# ============================================================

X_train_bandwise = np.zeros_like(
    X_train_log,
    dtype=np.float64
)

X_test_bandwise = np.zeros_like(
    X_test_log,
    dtype=np.float64
)


band_results = {}


for band_index, band in enumerate(BANDS):

    indices = [
        i
        for i, col in enumerate(eeg_cols)
        if col.endswith("_" + band)
    ]

    if len(indices) != 14:
        raise RuntimeError(
            f"Expected 14 features for {band}, "
            f"found {len(indices)}"
        )

    scaler = StandardScaler()

    X_train_bandwise[:, indices] = scaler.fit_transform(
        X_train_log[:, indices]
    )

    X_test_bandwise[:, indices] = scaler.transform(
        X_test_log[:, indices]
    )

    band_results[band] = {
        "features": len(indices),
        "indices": indices
    }


results_bandwise = evaluate(
    "2. LOG1P + BAND-WISE SCALING",
    X_train_bandwise,
    X_test_bandwise
)


# ============================================================
# 3. Band-wise feature statistics
# ============================================================

print("\n" + "=" * 70)
print("BAND-WISE TRAIN/TEST STATISTICS")
print("=" * 70)


for band in BANDS:

    indices = band_results[band]["indices"]

    train_mean = X_train_log[:, indices].mean()
    test_mean = X_test_log[:, indices].mean()

    train_std = X_train_log[:, indices].std()
    test_std = X_test_log[:, indices].std()

    mean_shift = abs(
        test_mean - train_mean
    ) / max(
        train_std,
        1e-8
    )

    print(
        f"{band:>6}: "
        f"train_mean={train_mean:.4f}, "
        f"test_mean={test_mean:.4f}, "
        f"train_std={train_std:.4f}, "
        f"test_std={test_std:.4f}, "
        f"mean_shift={mean_shift:.4f}"
    )


# ============================================================
# 4. Subject-wise band-wise results
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT-WISE BAND-WISE RESULTS")
print("=" * 70)


clf = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

clf.fit(
    X_train_bandwise,
    y_train
)

test_predictions = clf.predict(
    X_test_bandwise
)


for subject_id in TEST_SUBJECTS:

    mask = (
        test_df["subject_id"].values
        == subject_id
    )

    subject_true = y_test[mask]
    subject_pred = test_predictions[mask]

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


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Global scaling: "
    f"Balanced={results_global['balanced_accuracy']:.4f} | "
    f"Macro-F1={results_global['macro_f1']:.4f} | "
    f"MCC={results_global['mcc']:.4f}"
)

print(
    f"Band-wise scaling: "
    f"Balanced={results_bandwise['balanced_accuracy']:.4f} | "
    f"Macro-F1={results_bandwise['macro_f1']:.4f} | "
    f"MCC={results_bandwise['mcc']:.4f}"
)


print("\nCompleted.")