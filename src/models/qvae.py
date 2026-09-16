import torch.nn as nn

from .eeg_encoder import EEGEncoder
from .ecg_encoder import ECGEncoder
from .attention_fusion import AttentionFusion
from .quantum_layer import QuantumLayer
from .classifier import Classifier


class QVAE(nn.Module):

    def __init__(self):
        super().__init__()

        # =========================
        # EEG Encoder
        # 70 -> 256 -> 128 -> 64
        # =========================

        self.eeg_encoder = EEGEncoder()


        # =========================
        # ECG Encoder
        # 14 -> 128 -> 64
        # =========================

        self.ecg_encoder = ECGEncoder()


        # =========================
        # Attention Fusion
        # 64 + 64 -> 128
        # =========================

        self.fusion = AttentionFusion(
            dim=64
        )


        # =========================
        # Quantum Layer
        # 128 -> 4 qubits
        # =========================

        self.quantum = QuantumLayer(
            input_dim=128,
            n_qubits=4
        )


        # =========================
        # Classification Head
        # 4 -> 2
        # =========================

        self.classifier = Classifier(
            input_dim=4,
            num_classes=2
        )


    def forward(self, eeg, ecg):

        # EEG representation
        eeg_latent = self.eeg_encoder(eeg)


        # ECG representation
        ecg_latent = self.ecg_encoder(ecg)


        # Attention-based multimodal fusion
        fused, modality_weights = self.fusion(
            eeg_latent,
            ecg_latent
        )


        # Quantum latent representation
        quantum_latent = self.quantum(
            fused
        )


        # Final prediction
        prediction = self.classifier(
            quantum_latent
        )


        return {
            "prediction": prediction,
            "quantum_latent": quantum_latent,
            "modality_weights": modality_weights
        }