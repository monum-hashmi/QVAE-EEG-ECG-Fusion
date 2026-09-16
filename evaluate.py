import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data.dreamer_dataset import DREAMERDataset
from src.models.qvae import QVAE


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


dataset = DREAMERDataset(
    "data/dreamer_features/dreamer_valence_paired_all.csv"
)


train_size = int(0.8 * len(dataset))
val_size = len(dataset)-train_size


_, val_dataset = random_split(
    dataset,
    [train_size,val_size],
    generator=torch.Generator().manual_seed(42)
)


val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


model = QVAE().to(DEVICE)

model.load_state_dict(
    torch.load(
        "best_qvae_model.pth",
        map_location=DEVICE
    )
)


model.eval()


all_preds=[]
all_labels=[]


with torch.no_grad():

    for eeg,ecg,labels in val_loader:

        eeg=eeg.to(DEVICE)
        ecg=ecg.to(DEVICE)

        output=model(
            eeg,
            ecg
        )

        preds=torch.argmax(
            output["prediction"],
            dim=1
        )


        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )



print(
    "Accuracy:",
    accuracy_score(all_labels,all_preds)
)


print(
    "Precision:",
    precision_score(
        all_labels,
        all_preds
    )
)


print(
    "Recall:",
    recall_score(
        all_labels,
        all_preds
    )
)


print(
    "F1:",
    f1_score(
        all_labels,
        all_preds
    )
)


print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        all_labels,
        all_preds
    )
)