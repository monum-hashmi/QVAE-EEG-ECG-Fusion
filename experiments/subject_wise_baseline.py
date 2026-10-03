import sys
import os
import json
import random

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

from src.data.dreamer_subject_dataset import DREAMERSubjectDataset
from src.models.qvae import QVAE


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SUBJECT-WISE CLEAN QVAE BASELINE ANALYSIS")
print("=" * 70)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = (
    "data/dreamer_features/"
    "dreamer_valence_paired_all.csv"
)

TRAIN_SUBJECTS = list(range(1, 19))

TEST_SUBJECTS = [19, 20, 21, 22, 23]

CHECKPOINT = (
    "experiments/qvae_clean_baseline/"
    "best_qvae_clean_seed_42.pth"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training subjects...")

train_dataset = DREAMERSubjectDataset(
    csv_file=CSV_FILE,
    target_subjects=TRAIN_SUBJECTS,
    fit_scalers=True
)

print("Training samples:", len(train_dataset))


# ============================================================
# LOAD UNSEEN TEST DATA
# ============================================================

print("\nLoading unseen test subjects...")

test_dataset = DREAMERSubjectDataset(
    csv_file=CSV_FILE,
    target_subjects=TEST_SUBJECTS,
    eeg_scaler=train_dataset.eeg_scaler,
    ecg_scaler=train_dataset.ecg_scaler,
    fit_scalers=False
)

print("Test samples:", len(test_dataset))


print("\nTraining subjects:")
print(TRAIN_SUBJECTS)

print("\nUnseen test subjects:")
print(TEST_SUBJECTS)


# ============================================================
# TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading clean QVAE checkpoint:")

print(CHECKPOINT)

model = QVAE().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []
all_subjects = []


with torch.no_grad():

    for eeg, ecg, labels in test_loader:

        eeg = eeg.to(DEVICE)
        ecg = ecg.to(DEVICE)

        output = model(
            eeg,
            ecg
        )

        predictions = torch.argmax(
            output["prediction"],
            dim=1
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )


# ============================================================
# SUBJECT IDS
# ============================================================

all_subjects = test_dataset.subject_ids.tolist()


all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_subjects = np.array(
    all_subjects
)


# ============================================================
# GLOBAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL TEST CHECK")
print("=" * 70)

print(
    "Total test samples:",
    len(all_labels)
)

print(
    "True class distribution:",
    np.bincount(all_labels)
)

print(
    "Predicted class distribution:",
    np.bincount(
        all_predictions,
        minlength=2
    )
)

print("\nGlobal Confusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )
)

print(
    "\nGlobal Accuracy:",
    f"{accuracy_score(all_labels, all_predictions) * 100:.2f}%"
)

print(
    "Global Balanced Accuracy:",
    f"{balanced_accuracy_score(all_labels, all_predictions) * 100:.2f}%"
)

print(
    "Global Macro-F1:",
    f"{f1_score(all_labels, all_predictions, average='macro', zero_division=0) * 100:.2f}%"
)

print(
    "Global MCC:",
    f"{matthews_corrcoef(all_labels, all_predictions):.4f}"
)


# ============================================================
# SUBJECT-WISE ANALYSIS
# ============================================================

results = {}


print("\n" + "=" * 70)
print("SUBJECT-WISE RESULTS")
print("=" * 70)


for subject_id in TEST_SUBJECTS:

    mask = (
        all_subjects == subject_id
    )

    y_true = all_labels[mask]

    y_pred = all_predictions[mask]


    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )


    true_distribution = np.bincount(
        y_true,
        minlength=2
    )

    predicted_distribution = np.bincount(
        y_pred,
        minlength=2
    )


    results[str(subject_id)] = {

        "samples": int(len(y_true)),

        "true_class_0": int(
            true_distribution[0]
        ),

        "true_class_1": int(
            true_distribution[1]
        ),

        "predicted_class_0": int(
            predicted_distribution[0]
        ),

        "predicted_class_1": int(
            predicted_distribution[1]
        ),

        "accuracy": float(
            accuracy
        ),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "mcc": float(
            mcc
        ),

        "confusion_matrix": cm.tolist()
    }


    print("\n----------------------------------------")

    print(
        f"Subject {subject_id}"
    )

    print(
        "Samples:",
        len(y_true)
    )

    print(
        "True distribution:",
        true_distribution.tolist()
    )

    print(
        "Predicted distribution:",
        predicted_distribution.tolist()
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1: "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"MCC: {mcc:.4f}"
    )

    print("Confusion Matrix:")

    print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

output_dir = (
    "results/"
    "subject_wise_baseline"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


output_file = (
    output_dir +
    "/subject_wise_seed_42.json"
)


with open(
    output_file,
    "w"
) as f:

    json.dump(
        {
            "experiment":
                "subject_wise_clean_qvae_baseline",

            "seed":
                SEED,

            "train_subjects":
                TRAIN_SUBJECTS,

            "test_subjects":
                TEST_SUBJECTS,

            "checkpoint":
                CHECKPOINT,

            "results":
                results
        },
        f,
        indent=4
    )


print("\n" + "=" * 70)

print(
    "Results saved:"
)

print(
    output_file
)

print("\nCompleted.")