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


print("=" * 70)
print("SUBJECT NORMALISATION DIAGNOSTIC")
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
# Evaluation helper
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
# 1. ORIGINAL EEG
# ============================================================

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


scaler = StandardScaler()

X_train_original = scaler.fit_transform(
    X_train
)

X_test_original = scaler.transform(
    X_test
)


results_original = evaluate(
    "1. ORIGINAL EEG + GLOBAL TRAIN SCALING",
    X_train_original,
    X_test_original
)


# ============================================================
# 2. LOG1P EEG
# ============================================================

X_train_log = np.log1p(
    np.maximum(X_train, 0)
)

X_test_log = np.log1p(
    np.maximum(X_test, 0)
)


scaler = StandardScaler()

X_train_log_scaled = scaler.fit_transform(
    X_train_log
)

X_test_log_scaled = scaler.transform(
    X_test_log
)


results_log = evaluate(
    "2. LOG1P EEG + GLOBAL TRAIN SCALING",
    X_train_log_scaled,
    X_test_log_scaled
)


# ============================================================
# 3. SUBJECT-WISE NORMALISATION
# ============================================================
#
# Important:
# We calculate normalisation separately for each subject.
#
# This is a diagnostic only.
#
# It tells us whether subject-specific amplitude/location
# differences are contributing heavily to the problem.
#
# We do NOT use test labels here.
# ============================================================

def subject_normalise(df_input, columns):

    output = np.zeros(
        (len(df_input), len(columns)),
        dtype=np.float64
    )

    for subject_id, indices in df_input.groupby(
        "subject_id"
    ).groups.items():

        values = (
            df_input.loc[
                indices,
                columns
            ]
            .fillna(0)
            .values
        )

        # Log transform first
        values = np.log1p(
            np.maximum(values, 0)
        )

        mean = values.mean(
            axis=0,
            keepdims=True
        )

        std = values.std(
            axis=0,
            keepdims=True
        )

        # Prevent division by zero
        std[std < 1e-8] = 1.0

        normalized = (
            values - mean
        ) / std

        output[
            df_input.index.get_indexer(indices),
            :
        ] = normalized

    return output


# Reset indices so positional assignment is safe

train_subject_df = train_df.reset_index(
    drop=True
)

test_subject_df = test_df.reset_index(
    drop=True
)


X_train_subject = subject_normalise(
    train_subject_df,
    eeg_cols
)

X_test_subject = subject_normalise(
    test_subject_df,
    eeg_cols
)


results_subject = evaluate(
    "3. LOG1P + SUBJECT-WISE NORMALISATION",
    X_train_subject,
    X_test_subject
)


# ============================================================
# 4. SUBJECT-WISE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT-WISE RESULTS FOR SUBJECT NORMALISATION")
print("=" * 70)


clf = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

clf.fit(
    X_train_subject,
    y_train
)


test_predictions = clf.predict(
    X_test_subject
)


for subject_id in TEST_SUBJECTS:

    mask = (
        test_subject_df["subject_id"].values
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
# 5. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Original EEG:       "
    f"Balanced={results_original['balanced_accuracy']:.4f} | "
    f"Macro-F1={results_original['macro_f1']:.4f} | "
    f"MCC={results_original['mcc']:.4f}"
)

print(
    f"Log1p EEG:          "
    f"Balanced={results_log['balanced_accuracy']:.4f} | "
    f"Macro-F1={results_log['macro_f1']:.4f} | "
    f"MCC={results_log['mcc']:.4f}"
)

print(
    f"Subject-normalised: "
    f"Balanced={results_subject['balanced_accuracy']:.4f} | "
    f"Macro-F1={results_subject['macro_f1']:.4f} | "
    f"MCC={results_subject['mcc']:.4f}"
)


print("\nCompleted.")