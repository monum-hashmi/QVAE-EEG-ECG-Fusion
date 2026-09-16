import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):

    def __init__(
        self,
        latent_dim=64,
        num_heads=4
    ):
        super().__init__()

        self.eeg_to_ecg_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ecg_to_eeg_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True
        )


        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)


        self.fusion_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )


        self.weight_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, 2),
            nn.Softmax(dim=1)
        )


    def forward(
        self,
        eeg_features,
        ecg_features
    ):

        # Add sequence dimension
        eeg = eeg_features.unsqueeze(1)
        ecg = ecg_features.unsqueeze(1)


        # EEG attends to ECG
        eeg_attended, _ = self.eeg_to_ecg_attention(
            eeg,
            ecg,
            ecg
        )


        # ECG attends to EEG
        ecg_attended, _ = self.ecg_to_eeg_attention(
            ecg,
            eeg,
            eeg
        )


        eeg_out = self.norm1(
            eeg + eeg_attended
        ).squeeze(1)


        ecg_out = self.norm2(
            ecg + ecg_attended
        ).squeeze(1)


        combined = torch.cat(
            [
                eeg_out,
                ecg_out
            ],
            dim=1
        )


        fused = self.fusion_layer(
            combined
        )


        weights = self.weight_layer(
            combined
        )


        return fused, weights