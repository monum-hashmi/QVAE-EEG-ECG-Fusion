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
from src.models.focal_loss import FocalLoss

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef
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
print("Clean Subject-Independent QVAE Baseline")
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

print("\nTraining subjects:")
print(TRAIN_SUBJECTS)

print("\nValidation subjects:")
print(VAL_SUBJECTS)


# ============================================================
# First create raw training dataset
# ============================================================

CSV_FILE = (
    "data/dreamer_features/"
    "dreamer_valence_paired_all.csv"
)


# We first load the training subjects with their own scalers.
train_dataset = DREAMERSubjectDataset(
    csv_file=CSV_FILE,
    train_subjects=TRAIN_SUBJECTS,
    target_subjects=TRAIN_SUBJECTS,
    fit_scalers=True
)


# ============================================================
# Validation uses TRAINING scalers
# ============================================================

val_dataset = DREAMERSubjectDataset(
    csv_file=CSV_FILE,
    train_subjects=TRAIN_SUBJECTS,
    target_subjects=VAL_SUBJECTS,
    eeg_scaler=train_dataset.eeg_scaler,
    ecg_scaler=train_dataset.ecg_scaler,
    fit_scalers=False
)


print("\nDataset sizes:")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


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
# Mixed precision
# ============================================================

scaler = torch.cuda.amp.GradScaler(
    enabled=(DEVICE == "cuda")
)


# ============================================================
# Training
# ============================================================

EPOCHS = 40

best_accuracy = -1.0

train_history = []
val_history = []
loss_history = []


os.makedirs(
    "experiments/qvae_clean_baseline",
    exist_ok=True
)

os.makedirs(
    "results/clean_baseline",
    exist_ok=True
)


best_model_path = (
    "experiments/qvae_clean_baseline/"
    f"best_qvae_clean_seed_{SEED}.pth"
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
        val_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {running_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"Macro-F1: {val_macro_f1:.2f}% | "
        f"Balanced Acc: {val_balanced_accuracy:.2f}% | "
        f"MCC: {val_mcc:.4f} | "
        f"LR: {current_lr:.6f}"
    )


    train_history.append(
        train_accuracy
    )

    val_history.append(
        val_accuracy
    )

    loss_history.append(
        running_loss
    )


    # ========================================================
    # Save best model
    # ========================================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            f"results/clean_baseline/clean_baseline_seed_{SEED}.pth"
        )

        print(
            f"  Saved best model: "
            f"{best_accuracy:.2f}%"
        )


# ============================================================
# Final results
# ============================================================

best_epoch = int(
    np.argmax(val_history) + 1
)

best_index = best_epoch - 1


# Re-evaluate best checkpoint
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


results = {

    "experiment": "clean_subject_independent_qvae_baseline",

    "seed": SEED,

    "train_subjects": TRAIN_SUBJECTS,

    "validation_subjects": VAL_SUBJECTS,

    "train_samples": len(train_dataset),

    "validation_samples": len(val_dataset),

    "epochs": EPOCHS,

    "best_epoch": best_epoch,

    "best_validation_accuracy": final_accuracy,

    "macro_f1": final_macro_f1,

    "balanced_accuracy": final_balanced_accuracy,

    "mcc": final_mcc,

    "model_path": best_model_path,

    "pairing": "correct_only",

    "scaling": (
        "scalers fitted on training subjects only "
        "and applied to validation subjects"
    )
}


result_path = (
    "results/clean_baseline/"
    f"clean_baseline_seed_{SEED}.json"
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

    "seed": SEED,

    "train_accuracy": train_history,

    "val_accuracy": val_history,

    "loss": loss_history,

    "best_epoch": best_epoch,

    "best_validation_accuracy": final_accuracy
}


history_path = (
    "results/clean_baseline/"
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


print("\n" + "=" * 70)
print("CLEAN BASELINE COMPLETED")
print("=" * 70)

print(f"Seed:               {SEED}")
print(f"Best epoch:         {best_epoch}")
print(f"Accuracy:           {final_accuracy:.2f}%")
print(f"Macro-F1:           {final_macro_f1:.2f}%")
print(f"Balanced Accuracy:  {final_balanced_accuracy:.2f}%")
print(f"MCC:                {final_mcc:.4f}")

print("\nModel:")
print(best_model_path)

print("\nResults:")
print(result_path)

print("\nHistory:")
print(history_path)
