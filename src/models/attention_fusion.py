import torch
import torch.nn as nn


class AttentionFusion(nn.Module):

    def __init__(self, dim=64):
        super().__init__()


        self.attention = nn.Sequential(

            nn.Linear(dim*2,64),
            nn.ReLU(),

            nn.Linear(64,2),
            nn.Softmax(dim=1)

        )


        self.output = nn.Sequential(

            nn.Linear(dim*2,128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)

        )


    def forward(self,eeg,ecg):

        combined = torch.cat(
            [eeg,ecg],
            dim=1
        )


        weights = self.attention(
            combined
        )


        eeg_weight = weights[:,0].unsqueeze(1)
        ecg_weight = weights[:,1].unsqueeze(1)


        fused = torch.cat(
            [
                eeg * eeg_weight,
                ecg * ecg_weight
            ],
            dim=1
        )


        return self.output(fused), weights