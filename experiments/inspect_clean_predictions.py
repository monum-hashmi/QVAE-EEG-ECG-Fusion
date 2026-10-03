import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.data.dreamer_subject_dataset import DREAMERSubjectDataset
from src.models.qvae import QVAE


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV = "data/dreamer_features/dreamer_valence_paired_all.csv"
MODEL = "results/clean_baseline/clean_baseline_seed_42.pth"

TRAIN_SUBJECTS = list(range(1, 19))
VAL_SUBJECTS = list(range(19, 24))


# -------------------------------------------------
# Fit scalers ONLY on training subjects
# -------------------------------------------------

train_dataset = DREAMERSubjectDataset(
    CSV,
    target_subjects=TRAIN_SUBJECTS,
    fit_scalers=True
)

val_dataset = DREAMERSubjectDataset(
    CSV,
    target_subjects=VAL_SUBJECTS,
    eeg_scaler=train_dataset.eeg_scaler,
    ecg_scaler=train_dataset.ecg_scaler,
    fit_scalers=False
)


loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)


# -------------------------------------------------
# Load model
# -------------------------------------------------

model = QVAE().to(DEVICE)

state = torch.load(
    MODEL,
    map_location=DEVICE
)

model.load_state_dict(state)

model.eval()


# -------------------------------------------------
# Predictions
# -------------------------------------------------

y_true = []
y_pred = []


with torch.no_grad():

    for eeg, ecg, labels in loader:

        eeg = eeg.to(DEVICE)
        ecg = ecg.to(DEVICE)

        output = model(
            eeg,
            ecg
        )

        predictions = torch.argmax(
            output["prediction"],
            dim=1
        ).cpu().numpy()

        y_pred.extend(predictions)
        y_true.extend(labels.numpy())


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# -------------------------------------------------
# Results
# -------------------------------------------------

print("\n==============================")
print("CLEAN BASELINE PREDICTION ANALYSIS")
print("==============================")

print("\nTrue class distribution:")
print(np.bincount(y_true))

print("\nPredicted class distribution:")
print(np.bincount(y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        digits=4
    )
)

print(
    "\nBalanced Accuracy:",
    balanced_accuracy_score(y_true, y_pred)
)

print(
    "MCC:",
    matthews_corrcoef(y_true, y_pred)
)

print("\nCompleted.")
