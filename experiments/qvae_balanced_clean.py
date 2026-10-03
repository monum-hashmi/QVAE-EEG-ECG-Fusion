import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import json
import random
import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.dreamer_subject_dataset import DREAMERSubjectDataset
from src.models.qvae import QVAE
from src.models.balanced_focal_loss import BalancedFocalLoss

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix
)


# ============================================================
# Reproducibility
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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("BALANCED CLEAN SUBJECT-INDEPENDENT QVAE")
print("=" * 70)

print("Seed:", SEED)
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Fixed subject-independent split
# ============================================================

TRAIN_SUBJECTS = list(range(1, 19))
VAL_SUBJECTS = list(range(19, 24))

CSV_FILE = (
    "data/dreamer_features/"
    "dreamer_valence_paired_all.csv"
)


# ============================================================
# Training dataset
# ============================================================

train_dataset = DREAMERSubjectDataset(
    csv_file=CSV_FILE,
    target_subjects=TRAIN_SUBJECTS,
    fit_scalers=True
)


# ============================================================
# Unseen test dataset
# ============================================================

val_dataset = DREAMERSubjectDataset(
    csv_file=CSV_FILE,
    target_subjects=VAL_SUBJECTS,
    eeg_scaler=train_dataset.eeg_scaler,
    ecg_scaler=train_dataset.ecg_scaler,
    fit_scalers=False
)


print("\nTraining samples:", len(train_dataset))
print("Test samples:", len(val_dataset))


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE == "cuda")
)


# ============================================================
# Model
# ============================================================

model = QVAE().to(DEVICE)


# ============================================================
# Balanced focal loss
# ============================================================

class_counts = np.bincount(
    train_dataset.labels.astype(int),
    minlength=2
)

total = class_counts.sum()

class_weights = (
    total /
    (2.0 * class_counts)
)

print("\nTraining class counts:")
print("Class 0:", class_counts[0])
print("Class 1:", class_counts[1])

print("\nClass weights:")
print("Class 0:", class_weights[0])
print("Class 1:", class_weights[1])


criterion = BalancedFocalLoss(
    class_weights=class_weights,
    gamma=2.0
).to(DEVICE)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=4
)


# ============================================================
# Mixed precision
# ============================================================

scaler = torch.cuda.amp.GradScaler(
    enabled=(DEVICE == "cuda")
)


# ============================================================
# Training
# ============================================================

EPOCHS = 60

best_balanced_accuracy = -1.0

train_history = []
val_history = []
balanced_history = []
f1_history = []
mcc_history = []
loss_history = []


os.makedirs(
    "experiments/qvae_balanced_clean",
    exist_ok=True
)

os.makedirs(
    "results/qvae_balanced_clean",
    exist_ok=True
)


best_model_path = (
    "experiments/qvae_balanced_clean/"
    f"best_qvae_balanced_seed_{SEED}.pth"
)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    train_predictions = []
    train_targets = []


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


        with torch.cuda.amp.autocast(
            enabled=(DEVICE == "cuda")
        ):

            output = model(
                eeg,
                ecg
            )

            loss = criterion(
                output["prediction"],
                labels
            )


        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()


        running_loss += loss.item()


        predictions = torch.argmax(
            output["prediction"],
            dim=1
        )


        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        train_targets.extend(
            labels.detach().cpu().numpy()
        )


    train_accuracy = (
        accuracy_score(
            train_targets,
            train_predictions
        ) * 100
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_predictions = []
    val_targets = []


    with torch.no_grad():

        for eeg, ecg, labels in val_loader:

            eeg = eeg.to(DEVICE)

            ecg = ecg.to(DEVICE)

            labels = labels.to(DEVICE)


            output = model(
                eeg,
                ecg
            )


            predictions = torch.argmax(
                output["prediction"],
                dim=1
            )


            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_targets.extend(
                labels.cpu().numpy()
            )


    val_accuracy = (
        accuracy_score(
            val_targets,
            val_predictions
        ) * 100
    )


    val_macro_f1 = (
        f1_score(
            val_targets,
            val_predictions,
            average="macro"
        ) * 100
    )


    val_balanced_accuracy = (
        balanced_accuracy_score(
            val_targets,
            val_predictions
        ) * 100
    )


    val_mcc = matthews_corrcoef(
        val_targets,
        val_predictions
    )


    scheduler.step(
        val_balanced_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {running_loss:.4f} | "
        f"Train: {train_accuracy:.2f}% | "
        f"Test: {val_accuracy:.2f}% | "
        f"Balanced: {val_balanced_accuracy:.2f}% | "
        f"Macro-F1: {val_macro_f1:.2f}% | "
        f"MCC: {val_mcc:.4f} | "
        f"LR: {current_lr:.6f}"
    )


    train_history.append(
        train_accuracy
    )

    val_history.append(
        val_accuracy
    )

    balanced_history.append(
        val_balanced_accuracy
    )

    f1_history.append(
        val_macro_f1
    )

    mcc_history.append(
        val_mcc
    )

    loss_history.append(
        running_loss
    )


    # ========================================================
    # Save based on balanced accuracy
    # ========================================================

    if val_balanced_accuracy > best_balanced_accuracy:

        best_balanced_accuracy = (
            val_balanced_accuracy
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"  Saved new best model: "
            f"{best_balanced_accuracy:.2f}% balanced accuracy"
        )


# ============================================================
# Final evaluation
# ============================================================

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

model.eval()

final_predictions = []
final_targets = []


with torch.no_grad():

    for eeg, ecg, labels in val_loader:

        eeg = eeg.to(DEVICE)

        ecg = ecg.to(DEVICE)

        labels = labels.to(DEVICE)


        output = model(
            eeg,
            ecg
        )


        predictions = torch.argmax(
            output["prediction"],
            dim=1
        )


        final_predictions.extend(
            predictions.cpu().numpy()
        )

        final_targets.extend(
            labels.cpu().numpy()
        )


final_accuracy = (
    accuracy_score(
        final_targets,
        final_predictions
    ) * 100
)


final_macro_f1 = (
    f1_score(
        final_targets,
        final_predictions,
        average="macro"
    ) * 100
)


final_balanced_accuracy = (
    balanced_accuracy_score(
        final_targets,
        final_predictions
    ) * 100
)


final_mcc = matthews_corrcoef(
    final_targets,
    final_predictions
)


cm = confusion_matrix(
    final_targets,
    final_predictions
)


best_epoch = int(
    np.argmax(balanced_history) + 1
)


print("\n" + "=" * 70)
print("FINAL BALANCED CLEAN QVAE")
print("=" * 70)

print(
    f"Accuracy:          {final_accuracy:.2f}%"
)

print(
    f"Balanced Accuracy: {final_balanced_accuracy:.2f}%"
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


# ============================================================
# Save results
# ============================================================

results = {

    "experiment":
        "balanced_clean_subject_independent_qvae",

    "seed":
        SEED,

    "train_subjects":
        TRAIN_SUBJECTS,

    "validation_subjects":
        VAL_SUBJECTS,

    "train_samples":
        len(train_dataset),

    "validation_samples":
        len(val_dataset),

    "epochs":
        EPOCHS,

    "best_epoch":
        best_epoch,

    "accuracy":
        final_accuracy,

    "balanced_accuracy":
        final_balanced_accuracy,

    "macro_f1":
        final_macro_f1,

    "mcc":
        final_mcc,

    "confusion_matrix":
        cm.tolist(),

    "class_counts":
        class_counts.tolist(),

    "class_weights":
        class_weights.tolist(),

    "model_path":
        best_model_path,

    "pairing":
        "correct_only",

    "scaling":
        "scalers fitted on training subjects only "
        "and applied to validation subjects"
}


result_path = (
    "results/qvae_balanced_clean/"
    f"qvae_balanced_seed_{SEED}.json"
)


with open(
    result_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


history = {

    "seed":
        SEED,

    "train_accuracy":
        train_history,

    "val_accuracy":
        val_history,

    "balanced_accuracy":
        balanced_history,

    "macro_f1":
        f1_history,

    "mcc":
        mcc_history,

    "loss":
        loss_history,

    "best_epoch":
        best_epoch,

    "best_balanced_accuracy":
        final_balanced_accuracy
}


history_path = (
    "results/qvae_balanced_clean/"
    f"training_history_seed_{SEED}.json"
)


with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print("\nResults saved:")
print(result_path)

print(history_path)

print("\nCompleted.")