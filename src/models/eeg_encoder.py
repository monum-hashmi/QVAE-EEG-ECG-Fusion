import torch
import torch.nn as nn


class EEGEncoder(nn.Module):

    def __init__(self, input_dim=70, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, latent_dim)
        )


    def forward(self, x):
        return self.encoder(x)