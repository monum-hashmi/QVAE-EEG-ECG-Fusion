import torch
import torch.nn as nn


class EEGEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(70, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),


            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),


            nn.Linear(128,64),
            nn.LayerNorm(64)

        )


    def forward(self,x):

        return self.encoder(x)