import torch
import torch.nn as nn


class FusionNetwork(nn.Module):

    def __init__(self, latent_dim=64):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        self.projection = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim * 2)
        )


    def forward(self, eeg_latent, ecg_latent):

        combined = torch.cat(
            [eeg_latent, ecg_latent],
            dim=1
        )

        weights = self.attention(combined)

        eeg_weight = weights[:, 0].unsqueeze(1)
        ecg_weight = weights[:, 1].unsqueeze(1)

        weighted_eeg = eeg_latent * eeg_weight
        weighted_ecg = ecg_latent * ecg_weight

        fused = torch.cat(
            [weighted_eeg, weighted_ecg],
            dim=1
        )

        return self.projection(fused), weights
