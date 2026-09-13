import torch.nn as nn

from .eeg_encoder import EEGEncoder
from .ecg_encoder import ECGEncoder
from .fusion_network import FusionNetwork
from .quantum_layer import QuantumLayer
from .classifier import Classifier


class QVAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.eeg_encoder = EEGEncoder(
            input_dim=70,
            latent_dim=64
        )

        self.ecg_encoder = ECGEncoder(
            input_dim=14,
            latent_dim=64
        )

        self.fusion = FusionNetwork(
            latent_dim=64
        )

        self.quantum = QuantumLayer(
            input_dim=128,
            n_qubits=4
        )

        self.classifier = Classifier(
            input_dim=4,
            num_classes=2
        )


    def forward(self, eeg, ecg):

        eeg_latent = self.eeg_encoder(eeg)

        ecg_latent = self.ecg_encoder(ecg)

        fused, weights = self.fusion(
            eeg_latent,
            ecg_latent
        )

        quantum_latent = self.quantum(
            fused
        )

        prediction = self.classifier(
            quantum_latent
        )

        return {
            "prediction": prediction,
            "quantum_latent": quantum_latent,
            "modality_weights": weights
        }