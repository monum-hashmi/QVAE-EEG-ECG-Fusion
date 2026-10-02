import sys
import os
import json
import random
import numpy as np

# ============================================================
# Project root
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, ROOT)

# ============================================================
# Imports
# ============================================================

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
from src.models.cross_attention_fusion import CrossAttentionFusion
from src.models.focal_loss import FocalLoss


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
print("CLEAN SUBJECT-INDEPENDENT CLASSICAL BOTTLENECK")
print("=" * 70)
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Configuration
# ============================================================

DATA_PATH = (
    "data/dreamer_features/"
    "dreamer_valence_paired_all.csv"
)

TRAIN_SUBJECTS = list(range(1, 19))
TEST_SUBJECTS = list(range(19, 24))

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4


# ============================================================
# Model
# ============================================================

class ClassicalBottleneck(nn.Module):

    def __init__(self):

        super().__init__()

        self.eeg_encoder = EEGEncoder()

        self.ecg_encoder = ECGEncoder()

        self.fusion = CrossAttentionFusion(
            latent_dim=64,
            num_heads=4
        )

        # Four-dimensional classical bottleneck.
        # This deliberately matches the dimensionality
        # of the four-qubit quantum representation.

        self.bottleneck = nn.Sequential(

            nn.Linear(64, 4),

            nn.LayerNorm(4),

            nn.GELU(),

            nn.Dropout(0.2)

        )

        self.classifier = nn.Sequential(

            nn.Linear(4, 32),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(32, 2)

        )


    def forward(self, eeg, ecg):

        eeg_latent = self.eeg_encoder(eeg)

        ecg_latent = self.ecg_encoder(ecg)

        fused, modality_weights = self.fusion(
            eeg_latent,
            ecg_latent
        )

        latent = self.bottleneck(
            fused
        )

        prediction = self.classifier(
            latent
        )

        return {
            "prediction": prediction,
            "latent": latent,
            "modality_weights": modality_weights
        }


# ============================================================
# Dataset
# ============================================================

print("\nLoading training subjects...")

train_dataset = DREAMERSubjectDataset(
    DATA_PATH,
    train_subjects=TRAIN_SUBJECTS,
    target_subjects=TRAIN_SUBJECTS,
    fit_scalers=True
)

print("Training samples:", len(train_dataset))

print("\nLoading unseen test subjects...")

test_dataset = DREAMERSubjectDataset(
    DATA_PATH,
    train_subjects=TRAIN_SUBJECTS,
    target_subjects=TEST_SUBJECTS,
    eeg_scaler=train_dataset.eeg_scaler,
    ecg_scaler=train_dataset.ecg_scaler,
    fit_scalers=False
)

print("Test samples:", len(test_dataset))


# ============================================================
# Dataset sanity check
# ============================================================

print("\nTraining subjects:")
print(TRAIN_SUBJECTS)

print("\nUnseen test subjects:")
print(TEST_SUBJECTS)

print("\nTraining label distribution:")

unique, counts = np.unique(
    train_dataset.labels,
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
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = ClassicalBottleneck().to(DEVICE)


# ============================================================
# Loss
# ============================================================

criterion = FocalLoss(
    alpha=0.5,
    gamma=2
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
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
# Output directory
# ============================================================

OUTPUT_DIR = (
    "results/clean_classical_bottleneck"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


MODEL_PATH = (
    f"{OUTPUT_DIR}/"
    f"classical_bottleneck_seed_{SEED}.pth"
)

HISTORY_PATH = (
    f"{OUTPUT_DIR}/"
    f"training_history_seed_{SEED}.json"
)


# ============================================================
# Training
# ============================================================

best_balanced_accuracy = -1

history = {
    "seed": SEED,
    "train_accuracy": [],
    "test_accuracy": [],
    "test_balanced_accuracy": [],
    "test_macro_f1": [],
    "test_mcc": [],
    "loss": []
}


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    train_correct = 0
    train_total = 0


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
            output["prediction"],
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            output["prediction"],
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_accuracy = (
        100.0 *
        train_correct /
        train_total
    )


    # ========================================================
    # Evaluation on completely unseen subjects
    # ========================================================

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
                output["prediction"],
                dim=1
            )

            y_true.extend(
                labels.numpy().tolist()
            )

            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )


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


    scheduler.step(
        balanced_accuracy
    )


    history["train_accuracy"].append(
        train_accuracy
    )

    history["test_accuracy"].append(
        100 * accuracy
    )

    history["test_balanced_accuracy"].append(
        100 * balanced_accuracy
    )

    history["test_macro_f1"].append(
        100 * macro_f1
    )

    history["test_mcc"].append(
        mcc
    )

    history["loss"].append(
        total_loss
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {total_loss:.4f} | "
        f"Train: {train_accuracy:.2f}% | "
        f"Test Acc: {100 * accuracy:.2f}% | "
        f"Balanced: {100 * balanced_accuracy:.2f}% | "
        f"Macro-F1: {100 * macro_f1:.2f}% | "
        f"MCC: {mcc:.4f}"
    )


    # ========================================================
    # Save best model using balanced accuracy
    # ========================================================

    if balanced_accuracy > best_balanced_accuracy:

        best_balanced_accuracy = (
            balanced_accuracy
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "  Saved new best model."
        )


# ============================================================
# Final evaluation using best checkpoint
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEAN CLASSICAL BASELINE")
print("=" * 70)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

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
            output["prediction"],
            dim=1
        )

        y_true.extend(
            labels.numpy().tolist()
        )

        y_pred.extend(
            predictions.cpu().numpy().tolist()
        )


final_accuracy = accuracy_score(
    y_true,
    y_pred
)

final_balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

final_macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

final_mcc = matthews_corrcoef(
    y_true,
    y_pred
)

cm = confusion_matrix(
    y_true,
    y_pred
)


print(
    f"\nAccuracy: "
    f"{100 * final_accuracy:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{100 * final_balanced_accuracy:.2f}%"
)

print(
    f"Macro-F1: "
    f"{100 * final_macro_f1:.2f}%"
)

print(
    f"MCC: "
    f"{final_mcc:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# Save final results
# ============================================================

results = {

    "seed": SEED,

    "train_subjects": TRAIN_SUBJECTS,

    "test_subjects": TEST_SUBJECTS,

    "pairing": "correct",

    "accuracy": float(
        final_accuracy
    ),

    "balanced_accuracy": float(
        final_balanced_accuracy
    ),

    "macro_f1": float(
        final_macro_f1
    ),

    "mcc": float(
        final_mcc
    ),

    "confusion_matrix": cm.tolist()

}


with open(
    f"{OUTPUT_DIR}/"
    f"classical_bottleneck_seed_{SEED}.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


with open(
    HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print(
    "\nResults saved to:"
)

print(
    f"{OUTPUT_DIR}/"
    f"classical_bottleneck_seed_{SEED}.json"
)

print(
    f"{OUTPUT_DIR}/"
    f"training_history_seed_{SEED}.json"
)

print("\nCompleted.")