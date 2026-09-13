import pandas as pd
import torch
from torch.utils.data import Dataset


class DREAMERDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.eeg_cols = [
            c for c in self.df.columns
            if c.startswith("eeg_")
        ]

        self.ecg_cols = [
            c for c in self.df.columns
            if c.startswith("ecg_")
        ]

        self.eeg = self.df[self.eeg_cols].fillna(0).values
        self.ecg = self.df[self.ecg_cols].fillna(0).values

        self.labels = self.df["label"].values


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        eeg = torch.tensor(
            self.eeg[idx],
            dtype=torch.float32
        )

        ecg = torch.tensor(
            self.ecg[idx],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return eeg, ecg, label