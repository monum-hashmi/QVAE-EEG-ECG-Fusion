import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, input_dim=4, num_classes=2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )


    def forward(self, x):
        return self.classifier(x)