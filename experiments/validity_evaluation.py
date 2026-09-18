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
from torch.utils.data import DataLoader

from src.data.dreamer_dataset import DREAMERDataset
from src.models.qvae import QVAE



# =========================
# Device
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))



# =========================
# Load trained QVAE
# =========================

model = QVAE().to(DEVICE)


model.load_state_dict(
    torch.load(
        "best_qvae_model.pth",
        map_location=DEVICE
    )
)


model.eval()


print("Loaded QVAE model")



# =========================
# Evaluation Function
# =========================

def evaluate(dataset_path):


    dataset = DREAMERDataset(
        dataset_path
    )


    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )


    correct = 0

    total = 0



    with torch.no_grad():

        for eeg, ecg, labels in loader:


            eeg = eeg.to(DEVICE)

            ecg = ecg.to(DEVICE)

            labels = labels.to(DEVICE)



            output = model(
                eeg,
                ecg
            )


            prediction = torch.argmax(
                output["prediction"],
                dim=1
            )


            correct += (
                prediction == labels
            ).sum().item()


            total += labels.size(0)



    accuracy = (
        100 * correct / total
    )


    return accuracy



# =========================
# Run Validity Tests
# =========================


tests = {

    "Valid Pairing":
    "data/dreamer_features/dreamer_valence_paired_all.csv",


    "Same Subject Wrong Trial":
    "data/validity_experiments/dreamer_wrong_trial.csv",


    "Cross Subject":
    "data/validity_experiments/dreamer_cross_subject.csv"

}



print("\n========== Validity Evaluation ==========\n")



for name,path in tests.items():

    acc = evaluate(path)

    print(
        f"{name}: {acc:.2f}%"
    )



print(
    "\nEvaluation Completed"
)