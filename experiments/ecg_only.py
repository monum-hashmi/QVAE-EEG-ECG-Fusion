import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from src.data.dreamer_dataset import DREAMERDataset
from src.models.ecg_encoder import ECGEncoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))


# =========================
# ECG ONLY MODEL
# =========================

class ECGOnlyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = ECGEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )


    def forward(self, ecg):

        x = self.encoder(ecg)

        return self.classifier(x)



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
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=32
)



# =========================
# Model
# =========================

model = ECGOnlyModel().to(DEVICE)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)



epochs = 20

best_acc = 0



# =========================
# Training
# =========================

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0


    for eeg, ecg, labels in train_loader:


        ecg = ecg.to(DEVICE)
        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        output = model(ecg)


        loss = criterion(
            output,
            labels
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        pred = torch.argmax(
            output,
            dim=1
        )


        correct += (pred == labels).sum().item()

        total += labels.size(0)



    train_acc = 100 * correct / total



    # Validation

    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for eeg, ecg, labels in val_loader:

            ecg = ecg.to(DEVICE)
            labels = labels.to(DEVICE)


            output = model(ecg)


            pred = torch.argmax(
                output,
                dim=1
            )


            val_correct += (
                pred == labels
            ).sum().item()

            val_total += labels.size(0)



    val_acc = 100 * val_correct / val_total



    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )



    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            "experiments/ecg_only/best_ecg_only.pth"
        )



print("\nBest ECG Only Accuracy:", best_acc)