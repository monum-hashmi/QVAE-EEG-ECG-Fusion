import torch
import torch.nn as nn


class ECGEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(14,128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),


            nn.Linear(128,64),
            nn.LayerNorm(64)

        )


    def forward(self,x):

        return self.encoder(x)