import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class DREAMERSubjectDataset(Dataset):

    def __init__(
        self,
        csv_file,
        target_subjects,
        eeg_scaler=None,
        ecg_scaler=None,
        fit_scalers=False
    ):

        self.df = pd.read_csv(csv_file)

        # Keep only correctly paired samples
        self.df = self.df[
            self.df["pairing"] == "correct"
        ].reset_index(drop=True)

        # Select requested subjects
        if target_subjects is not None:

            self.df = self.df[
                self.df["subject_id"].isin(target_subjects)
            ].reset_index(drop=True)

        # Feature columns
        self.eeg_cols = [
            c for c in self.df.columns
            if c.startswith("eeg_")
        ]

        self.ecg_cols = [
            c for c in self.df.columns
            if c.startswith("ecg_")
        ]

        # Raw features
        eeg = (
            self.df[self.eeg_cols]
            .fillna(0)
            .values
        )

        ecg = (
            self.df[self.ecg_cols]
            .fillna(0)
            .values
        )

        # Scalers
        if eeg_scaler is None:
            eeg_scaler = StandardScaler()

        if ecg_scaler is None:
            ecg_scaler = StandardScaler()

        # Fit only on training subjects
        if fit_scalers:

            self.eeg = eeg_scaler.fit_transform(eeg)
            self.ecg = ecg_scaler.fit_transform(ecg)

        else:

            self.eeg = eeg_scaler.transform(eeg)
            self.ecg = ecg_scaler.transform(ecg)

        self.eeg_scaler = eeg_scaler
        self.ecg_scaler = ecg_scaler

        # Labels
        self.labels = self.df["label"].values

        # Metadata
        self.subject_ids = self.df["subject_id"].values
        self.trial_ids = self.df["trial_id"].values
        self.window_ids = self.df["window_id"].values

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