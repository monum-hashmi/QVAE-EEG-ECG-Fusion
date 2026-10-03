import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.data.dreamer_subject_dataset import DREAMERSubjectDataset
from src.models.eeg_encoder import EEGEncoder
from src.models.ecg_encoder import ECGEncoder
from src.models.cross_attention_fusion import CrossAttentionFusion
from src.models.focal_loss import FocalLoss


# ============================================================
# Arguments
# ============================================================

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()

SEED = args.seed


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("NO-QUANTUM SUBJECT-INDEPENDENT ABLATION")
print("=" * 70)

print("Seed:", SEED)
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Subjects
# ============================================================

TRAIN_SUBJECTS = list(range(1, 19))
TEST_SUBJECTS = list(range(19, 24))

CSV = "data/dreamer_features/dreamer_valence_paired_all.csv"


# ============================================================
# Dataset
# ============================================================

print("\nLoading training subjects...")

train_dataset = DREAMERSubjectDataset(
    CSV,
    target_subjects=TRAIN_SUBJECTS,
    fit_scalers=True
)

print("Training samples:", len(train_dataset))

print("\nLoading unseen test subjects...")

test_dataset = DREAMERSubjectDataset(
    CSV,
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

print("\nTraining label distribution:")
print(
    np.bincount(
        train_dataset.labels.astype(int)
    )
)

print("\nTest label distribution:")
print(
    np.bincount(
        test_dataset.labels.astype(int)
    )
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE == "cuda")
)


# ============================================================
# No-Quantum Model
# ============================================================

class NoQuantumQVAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.eeg_encoder = EEGEncoder()

        self.ecg_encoder = ECGEncoder()

        self.fusion = CrossAttentionFusion(
            latent_dim=64,
            num_heads=4
        )

        # IMPORTANT:
        # The original model does:
        #
        # fused 64 -> quantum -> 4 -> classifier
        #
        # This ablation removes ONLY the quantum layer.
        #
        # fused 64 -> classifier

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
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

        prediction = self.classifier(
            fused
        )

        return {
            "prediction": prediction,
            "fused_latent": fused,
            "modality_weights": modality_weights
        }


# ============================================================
# Model
# ============================================================

model = NoQuantumQVAE().to(DEVICE)

print("\nModel created.")

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Trainable parameters:", total_parameters)


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
    patience=3
)


# ============================================================
# Mixed Precision
# ============================================================

use_amp = DEVICE == "cuda"

try:
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )
except Exception:
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )


# ============================================================
# Training
# ============================================================

EPOCHS = 40

best_balanced_accuracy = -1.0
best_epoch = 0

train_history = []
test_history = []
loss_history = []
balanced_history = []
macro_f1_history = []
mcc_history = []


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for eeg, ecg, labels in train_loader:

        eeg = eeg.to(
            DEVICE,
            non_blocking=True
        )

        ecg = ecg.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if use_amp:

            with torch.amp.autocast(
                "cuda"
            ):

                output = model(
                    eeg,
                    ecg
                )

                loss = criterion(
                    output["prediction"],
                    labels
                )

        else:

            output = model(
                eeg,
                ecg
            )

            loss = criterion(
                output["prediction"],
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += loss.item()

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
    # Test on COMPLETELY UNSEEN SUBJECTS
    # ========================================================

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for eeg, ecg, labels in test_loader:

            eeg = eeg.to(
                DEVICE,
                non_blocking=True
            )

            ecg = ecg.to(
                DEVICE,
                non_blocking=True
            )

            output = model(
                eeg,
                ecg
            )

            predictions = torch.argmax(
                output["prediction"],
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )


    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )


    test_accuracy = (
        100.0 *
        accuracy_score(
            all_labels,
            all_predictions
        )
    )

    balanced_accuracy = (
        100.0 *
        balanced_accuracy_score(
            all_labels,
            all_predictions
        )
    )

    macro_f1 = (
        100.0 *
        f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    mcc = matthews_corrcoef(
        all_labels,
        all_predictions
    )


    scheduler.step(
        balanced_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    train_history.append(
        train_accuracy
    )

    test_history.append(
        test_accuracy
    )

    loss_history.append(
        running_loss
    )

    balanced_history.append(
        balanced_accuracy
    )

    macro_f1_history.append(
        macro_f1
    )

    mcc_history.append(
        mcc
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {running_loss:.4f} | "
        f"Train: {train_accuracy:.2f}% | "
        f"Test: {test_accuracy:.2f}% | "
        f"Balanced: {balanced_accuracy:.2f}% | "
        f"Macro-F1: {macro_f1:.2f}% | "
        f"MCC: {mcc:.4f} | "
        f"LR: {current_lr:.6f}"
    )


    # ========================================================
    # Save best based on balanced accuracy
    # ========================================================

    if balanced_accuracy > best_balanced_accuracy:

        best_balanced_accuracy = (
            balanced_accuracy
        )

        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            f"experiments/"
            f"no_quantum_ablation/"
            f"best_no_quantum_seed_{SEED}.pth"
        )

        print(
            "  Saved new best no-quantum model."
        )


# ============================================================
# Final Evaluation
# ============================================================

model.load_state_dict(
    torch.load(
        f"experiments/"
        f"no_quantum_ablation/"
        f"best_no_quantum_seed_{SEED}.pth",
        map_location=DEVICE
    )
)

model.eval()

all_predictions = []
all_labels = []

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

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


final_accuracy = (
    100.0 *
    accuracy_score(
        all_labels,
        all_predictions
    )
)

final_balanced = (
    100.0 *
    balanced_accuracy_score(
        all_labels,
        all_predictions
    )
)

final_macro_f1 = (
    100.0 *
    f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )
)

final_mcc = matthews_corrcoef(
    all_labels,
    all_predictions
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 70)
print("FINAL NO-QUANTUM ABLATION")
print("=" * 70)

print(
    f"Accuracy:          {final_accuracy:.2f}%"
)

print(
    f"Balanced Accuracy: {final_balanced:.2f}%"
)

print(
    f"Macro-F1:          {final_macro_f1:.2f}%"
)

print(
    f"MCC:               {final_mcc:.4f}"
)

print(
    f"Best Epoch:        {best_epoch}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nPredicted distribution:")
print(
    np.bincount(
        all_predictions,
        minlength=2
    )
)


# ============================================================
# Save Results
# ============================================================

os.makedirs(
    "results/no_quantum_ablation",
    exist_ok=True
)

os.makedirs(
    "experiments/no_quantum_ablation",
    exist_ok=True
)


results = {
    "experiment":
        "no_quantum_subject_independent_ablation",

    "seed":
        SEED,

    "train_subjects":
        TRAIN_SUBJECTS,

    "test_subjects":
        TEST_SUBJECTS,

    "train_samples":
        len(train_dataset),

    "test_samples":
        len(test_dataset),

    "epochs":
        EPOCHS,

    "best_epoch":
        best_epoch,

    "accuracy":
        final_accuracy,

    "balanced_accuracy":
        final_balanced,

    "macro_f1":
        final_macro_f1,

    "mcc":
        final_mcc,

    "confusion_matrix":
        cm.tolist(),

    "predicted_distribution":
        np.bincount(
            all_predictions,
            minlength=2
        ).tolist(),

    "architecture":
        "EEG encoder + ECG encoder + cross-attention fusion + classifier",

    "quantum_layer":
        "removed",

    "pairing":
        "correct_only",

    "scaling":
        "scalers fitted on training subjects only and applied to unseen test subjects",

    "model_path":
        f"experiments/no_quantum_ablation/"
        f"best_no_quantum_seed_{SEED}.pth"
}


with open(
    f"results/no_quantum_ablation/"
    f"no_quantum_seed_{SEED}.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


history = {
    "seed": SEED,
    "train_accuracy": train_history,
    "test_accuracy": test_history,
    "loss": loss_history,
    "balanced_accuracy": balanced_history,
    "macro_f1": macro_f1_history,
    "mcc": mcc_history,
    "best_epoch": best_epoch,
    "best_balanced_accuracy":
        best_balanced_accuracy
}


with open(
    f"results/no_quantum_ablation/"
    f"training_history_seed_{SEED}.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print("\nResults saved:")
print(
    "results/no_quantum_ablation/"
    f"no_quantum_seed_{SEED}.json"
)

print(
    "results/no_quantum_ablation/"
    f"training_history_seed_{SEED}.json"
)

print("\nCompleted.")