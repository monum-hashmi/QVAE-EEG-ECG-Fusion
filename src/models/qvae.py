import torch.nn as nn

from .eeg_encoder import EEGEncoder
from .ecg_encoder import ECGEncoder
from .cross_attention_fusion import CrossAttentionFusion
from .quantum_layer import QuantumLayer
from .classifier import Classifier


class QVAE(nn.Module):

    def __init__(self):
        super().__init__()


        # EEG Encoder
        self.eeg_encoder = EEGEncoder()


        # ECG Encoder
        self.ecg_encoder = ECGEncoder()


        # Cross Modal Attention Fusion
        self.fusion = CrossAttentionFusion(
            latent_dim=64,
            num_heads=4
        )


        # Quantum Layer
        # Cross attention output = 64
        self.quantum = QuantumLayer(
            input_dim=64,
            n_qubits=4
        )


        # Classifier
        self.classifier = Classifier(
            input_dim=4,
            num_classes=2
        )


    def forward(self, eeg, ecg):

        eeg_latent = self.eeg_encoder(
            eeg
        )


        ecg_latent = self.ecg_encoder(
            ecg
        )


        fused, modality_weights = self.fusion(
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
            "modality_weights": modality_weights
        }