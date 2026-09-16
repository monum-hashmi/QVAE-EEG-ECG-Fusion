import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from src.data.dreamer_dataset import DREAMERDataset
from src.models.qvae import QVAE


# =========================
# Reproducibility
# =========================

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# =========================
# Device
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))


# =========================
# Dataset
# =========================

dataset = DREAMERDataset(
    "data/dreamer_features/dreamer_valence_paired_all.csv"
)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size


train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)



# =========================
# Model
# =========================

model = QVAE().to(DEVICE)



# =========================
# Loss
# =========================

criterion = nn.CrossEntropyLoss()



# =========================
# Optimizer
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4
)


# =========================
# Scheduler
# =========================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)



# =========================
# Mixed Precision
# =========================

scaler = torch.cuda.amp.GradScaler(
    enabled=(DEVICE == "cuda")
)



# =========================
# Training
# =========================

epochs = 20

best_accuracy = 0


for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0


    for eeg, ecg, labels in train_loader:


        eeg = eeg.to(DEVICE)
        ecg = ecg.to(DEVICE)
        labels = labels.to(DEVICE)



        optimizer.zero_grad()



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



        total_loss += loss.item()



        predictions = torch.argmax(
            output["prediction"],
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)



    train_accuracy = (
        100 * correct / total
    )



    # =====================
    # Validation
    # =====================

    model.eval()


    val_correct = 0
    val_total = 0



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



            val_correct += (
                predictions == labels
            ).sum().item()



            val_total += labels.size(0)



    val_accuracy = (
        100 * val_correct / val_total
    )



    scheduler.step(
        val_accuracy
    )



    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}%"
    )



    # Save best model

    if val_accuracy > best_accuracy:


        best_accuracy = val_accuracy


        torch.save(
            model.state_dict(),
            "best_qvae_model.pth"
        )


        print(
            "Saved best model!"
        )



print("\nTraining Completed")

print(
    "Best Validation Accuracy:",
    best_accuracy
)