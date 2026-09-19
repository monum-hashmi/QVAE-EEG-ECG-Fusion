import sys
import os
import argparse
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from torch.utils.data import DataLoader
from src.data.dreamer_dataset import DREAMERDataset
from src.models.qvae import QVAE


# =========================
# Seed Argument
# =========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()

SEED = args.seed



# =========================
# Device
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))



# =========================
# Load Model
# =========================

model = QVAE().to(DEVICE)


model_path = (
    f"experiments/qvae_cross_attention/"
    f"best_qvae_cross_attention_seed_{SEED}.pth"
)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=DEVICE
    )
)


print(
    "Loaded model:",
    model_path
)


model.eval()



# =========================
# Evaluation Function
# =========================

def evaluate(loader):

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


            predictions = torch.argmax(
                output["prediction"],
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    return (
        100 * correct / total
    )



# =========================
# Load Validity Datasets
# =========================

valid_dataset = DREAMERDataset(
    "data/dreamer_features/dreamer_valence_paired_all.csv"
)


wrong_trial_dataset = DREAMERDataset(
    "data/validity_experiments/dreamer_wrong_trial.csv"
)


cross_subject_dataset = DREAMERDataset(
    "data/validity_experiments/dreamer_cross_subject.csv"
)



valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False
)


wrong_trial_loader = DataLoader(
    wrong_trial_dataset,
    batch_size=32,
    shuffle=False
)


cross_subject_loader = DataLoader(
    cross_subject_dataset,
    batch_size=32,
    shuffle=False
)



# =========================
# Results
# =========================

print(
    "\n========== Validity Evaluation =========="
)


valid_accuracy = evaluate(
    valid_loader
)


wrong_accuracy = evaluate(
    wrong_trial_loader
)


cross_accuracy = evaluate(
    cross_subject_loader
)



print(
    f"Valid Pairing: {valid_accuracy:.2f}%"
)


print(
    f"Same Subject Wrong Trial: {wrong_accuracy:.2f}%"
)


print(
    f"Cross Subject: {cross_accuracy:.2f}%"
)


print(
    "\nEvaluation Completed"
)