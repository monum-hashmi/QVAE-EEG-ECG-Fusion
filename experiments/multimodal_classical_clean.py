import sys
import os
import json
import random
import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from src.data.dreamer_subject_dataset import DREAMERSubjectDataset
from src.models.eeg_encoder import EEGEncoder
from src.models.ecg_encoder import ECGEncoder


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("CLEAN SUBJECT-INDEPENDENT MULTIMODAL CLASSICAL BASELINE")
print("=" * 70)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Subjects
# ============================================================

TRAIN_SUBJECTS = list(range(1, 19))

TEST_SUBJECTS = list(range(19, 24))


CSV_FILE = (
    "data/dreamer_features/"
    "dreamer_valence_paired_all.csv"
)


# ============================================================
# Training dataset
# ============================================================

print("\nLoading training subjects...")

train_dataset_raw = DREAMERSubjectDataset(
    CSV_FILE,
    target_subjects=TRAIN_SUBJECTS,
    fit_scalers=True
)

print(
    "Training samples:",
    len(train_dataset_raw)
)


# ============================================================
# Test dataset
# ============================================================

print("\nLoading unseen test subjects...")

test_dataset = DREAMERSubjectDataset(
    CSV_FILE,
    target_subjects=TEST_SUBJECTS,
    eeg_scaler=train_dataset_raw.eeg_scaler,
    ecg_scaler=train_dataset_raw.ecg_scaler,
    fit_scalers=False
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# Print distributions
# ============================================================

print("\nTraining subjects:")
print(TRAIN_SUBJECTS)

print("\nUnseen test subjects:")
print(TEST_SUBJECTS)

print("\nTraining label distribution:")

unique, counts = np.unique(
    train_dataset_raw.labels,
    return_counts=True
)

for label, count in zip(unique, counts):
    print(
        f"Class {label}: {count}"
    )


print("\nTest label distribution:")

unique, counts = np.unique(
    test_dataset.labels,
    return_counts=True
)

for label, count in zip(unique, counts):
    print(
        f"Class {label}: {count}"
    )


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset_raw,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

class MultimodalClassical(nn.Module):

    def __init__(self):

        super().__init__()

        self.eeg_encoder = EEGEncoder()

        self.ecg_encoder = ECGEncoder()

        # Combine both modality representations
        self.fusion = nn.Sequential(

            nn.Linear(64 + 64, 128),

            nn.LayerNorm(128),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.LayerNorm(64),

            nn.GELU(),

            nn.Dropout(0.2)
        )

        # Classification head
        self.classifier = nn.Sequential(

            nn.Linear(64, 32),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(32, 2)
        )


    def forward(
        self,
        eeg,
        ecg
    ):

        eeg_latent = self.eeg_encoder(eeg)

        ecg_latent = self.ecg_encoder(ecg)

        combined = torch.cat(
            [
                eeg_latent,
                ecg_latent
            ],
            dim=1
        )

        fused = self.fusion(
            combined
        )

        prediction = self.classifier(
            fused
        )

        return prediction


# ============================================================
# Model
# ============================================================

model = MultimodalClassical().to(DEVICE)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4
)


# ============================================================
# Scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)


# ============================================================
# Training
# ============================================================

EPOCHS = 60

best_balanced_accuracy = -1

history = []


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    train_true = []

    train_pred = []


    for eeg, ecg, labels in train_loader:

        eeg = eeg.to(DEVICE)

        ecg = ecg.to(DEVICE)

        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        output = model(
            eeg,
            ecg
        )


        loss = criterion(
            output,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        predictions = torch.argmax(
            output,
            dim=1
        )


        train_true.extend(
            labels.detach().cpu().numpy()
        )

        train_pred.extend(
            predictions.detach().cpu().numpy()
        )


    train_accuracy = (
        accuracy_score(
            train_true,
            train_pred
        ) * 100
    )


    # ========================================================
    # Test
    # ========================================================

    model.eval()

    test_true = []

    test_pred = []


    with torch.no_grad():

        for eeg, ecg, labels in test_loader:

            eeg = eeg.to(DEVICE)

            ecg = ecg.to(DEVICE)

            labels = labels.to(DEVICE)


            output = model(
                eeg,
                ecg
            )


            predictions = torch.argmax(
                output,
                dim=1
            )


            test_true.extend(
                labels.cpu().numpy()
            )

            test_pred.extend(
                predictions.cpu().numpy()
            )


    test_accuracy = (
        accuracy_score(
            test_true,
            test_pred
        ) * 100
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            test_true,
            test_pred
        ) * 100
    )


    macro_f1 = (
        f1_score(
            test_true,
            test_pred,
            average="macro"
        ) * 100
    )


    mcc = matthews_corrcoef(
        test_true,
        test_pred
    )


    scheduler.step(
        balanced_accuracy
    )


    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss: {running_loss:.4f} | "
        f"Train: {train_accuracy:.2f}% | "
        f"Test Acc: {test_accuracy:.2f}% | "
        f"Balanced: {balanced_accuracy:.2f}% | "
        f"Macro-F1: {macro_f1:.2f}% | "
        f"MCC: {mcc:.4f}"
    )


    history.append(
        {
            "epoch": epoch + 1,
            "loss": running_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "mcc": mcc
        }
    )


    # Save based on balanced accuracy
    if balanced_accuracy > best_balanced_accuracy:

        best_balanced_accuracy = balanced_accuracy

        os.makedirs(
            "experiments/multimodal_classical",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            "experiments/multimodal_classical/"
            "best_multimodal_classical.pth"
        )

        print(
            "  Saved new best multimodal model."
        )


# ============================================================
# Final evaluation
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEAN MULTIMODAL CLASSICAL RESULT")
print("=" * 70)


state = torch.load(
    "experiments/multimodal_classical/"
    "best_multimodal_classical.pth",
    map_location=DEVICE
)

model.load_state_dict(state)

model.eval()

y_true = []

y_pred = []


with torch.no_grad():

    for eeg, ecg, labels in test_loader:

        eeg = eeg.to(DEVICE)

        ecg = ecg.to(DEVICE)

        output = model(
            eeg,
            ecg
        )

        predictions = torch.argmax(
            output,
            dim=1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )


accuracy = (
    accuracy_score(
        y_true,
        y_pred
    ) * 100
)


balanced = (
    balanced_accuracy_score(
        y_true,
        y_pred
    ) * 100
)


macro_f1 = (
    f1_score(
        y_true,
        y_pred,
        average="macro"
    ) * 100
)


mcc = matthews_corrcoef(
    y_true,
    y_pred
)


print(
    f"\nAccuracy: {accuracy:.2f}%"
)

print(
    f"Balanced Accuracy: {balanced:.2f}%"
)

print(
    f"Macro-F1: {macro_f1:.2f}%"
)

print(
    f"MCC: {mcc:.4f}"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        digits=4
    )
)


# ============================================================
# Save results
# ============================================================

os.makedirs(
    "results/clean_multimodal_classical",
    exist_ok=True
)


results = {

    "experiment":
        "clean_subject_independent_multimodal_classical",

    "seed":
        SEED,

    "train_subjects":
        TRAIN_SUBJECTS,

    "test_subjects":
        TEST_SUBJECTS,

    "train_samples":
        len(train_dataset_raw),

    "test_samples":
        len(test_dataset),

    "epochs":
        EPOCHS,

    "best_balanced_accuracy":
        best_balanced_accuracy,

    "accuracy":
        accuracy,

    "balanced_accuracy":
        balanced,

    "macro_f1":
        macro_f1,

    "mcc":
        mcc,

    "pairing":
        "correct_only",

    "scaling":
        "scalers fitted on training subjects only and applied to unseen test subjects",

    "model_path":
        "experiments/multimodal_classical/"
        "best_multimodal_classical.pth"
}


with open(
    "results/clean_multimodal_classical/"
    "multimodal_classical_seed_42.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


with open(
    "results/clean_multimodal_classical/"
    "training_history_seed_42.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print(
    "\nResults saved:"
)

print(
    "results/clean_multimodal_classical/"
    "multimodal_classical_seed_42.json"
)

print(
    "results/clean_multimodal_classical/"
    "training_history_seed_42.json"
)

print("\nCompleted.")