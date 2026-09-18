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
from src.models.eeg_encoder import EEGEncoder
from src.models.ecg_encoder import ECGEncoder
from src.models.cross_attention_fusion import CrossAttentionFusion


from src.models.focal_loss import FocalLoss



SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Device:", DEVICE)

if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))



# =========================
# Model without Quantum
# =========================

class CrossAttentionNoQuantum(nn.Module):

    def __init__(self):

        super().__init__()


        self.eeg_encoder = EEGEncoder()

        self.ecg_encoder = ECGEncoder()


        self.fusion = CrossAttentionFusion(
            latent_dim=64,
            num_heads=4
        )


        self.classifier = nn.Sequential(

            nn.Linear(64,32),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(32,2)

        )


    def forward(self,eeg,ecg):

        eeg_latent = self.eeg_encoder(eeg)

        ecg_latent = self.ecg_encoder(ecg)


        fused, weights = self.fusion(
            eeg_latent,
            ecg_latent
        )


        prediction = self.classifier(
            fused
        )


        return prediction



# =========================
# Dataset
# =========================

dataset = DREAMERDataset(
    "data/dreamer_features/dreamer_valence_paired_all.csv"
)



train_size = int(0.8 * len(dataset))

val_size = len(dataset)-train_size



train_dataset,val_dataset = random_split(

    dataset,

    [train_size,val_size],

    generator=torch.Generator().manual_seed(SEED)

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
# Training
# =========================

model = CrossAttentionNoQuantum().to(DEVICE)



criterion = FocalLoss(
    alpha=0.5,
    gamma=2
)



optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=0.0005,

    weight_decay=1e-4

)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3

)



epochs = 40


best_accuracy = 0



for epoch in range(epochs):


    model.train()


    total_loss = 0

    correct = 0

    total = 0



    for eeg,ecg,labels in train_loader:


        eeg=eeg.to(DEVICE)

        ecg=ecg.to(DEVICE)

        labels=labels.to(DEVICE)



        optimizer.zero_grad()



        output=model(
            eeg,
            ecg
        )


        loss=criterion(
            output,
            labels
        )


        loss.backward()

        optimizer.step()



        total_loss += loss.item()



        pred=torch.argmax(
            output,
            dim=1
        )


        correct += (
            pred==labels
        ).sum().item()


        total += labels.size(0)



    train_acc=100*correct/total



    model.eval()


    val_correct=0

    val_total=0



    with torch.no_grad():


        for eeg,ecg,labels in val_loader:


            eeg=eeg.to(DEVICE)

            ecg=ecg.to(DEVICE)

            labels=labels.to(DEVICE)



            output=model(
                eeg,
                ecg
            )


            pred=torch.argmax(
                output,
                dim=1
            )


            val_correct += (
                pred==labels
            ).sum().item()


            val_total += labels.size(0)



    val_acc=100*val_correct/val_total


    scheduler.step(val_acc)



    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )



    if val_acc > best_accuracy:

        best_accuracy=val_acc


        torch.save(

            model.state_dict(),

            "experiments/cross_attention_no_quantum/best_no_quantum.pth"

        )

        print("Saved best model!")



print()

print(
    "Best Cross Attention No Quantum Accuracy:",
    best_accuracy
)