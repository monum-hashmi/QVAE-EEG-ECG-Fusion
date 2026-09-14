import torch
import torch.nn as nn
import pennylane as qml


class QuantumLayer(nn.Module):

    def __init__(self, input_dim=128, n_qubits=4):
        super().__init__()

        self.n_qubits = n_qubits

        # Classical projection into quantum space
        self.encoder = nn.Linear(
            input_dim,
            n_qubits
        )

        # Trainable quantum parameters
        self.weights = nn.Parameter(
            torch.randn(n_qubits)
        )

        # Quantum simulator runs on CPU
        dev = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):

            # Encoding
            for i in range(n_qubits):
                qml.RY(
                    inputs[i],
                    wires=i
                )

            # Entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(
                    wires=[i, i + 1]
                )

            # Variational layer
            for i in range(n_qubits):
                qml.RY(
                    weights[i],
                    wires=i
                )

            return [
                qml.expval(
                    qml.PauliZ(i)
                )
                for i in range(n_qubits)
            ]

        self.circuit = circuit


    def forward(self, x):

        # Classical projection
        x = self.encoder(x)

        outputs = []

        for sample in x:

            # PennyLane default.qubit is CPU based
            sample_cpu = sample.to("cpu")
            weights_cpu = self.weights.to("cpu")

            quantum_output = self.circuit(
                sample_cpu,
                weights_cpu
            )

            outputs.append(
                torch.stack(quantum_output)
            )

        # Return quantum features back to original device
        return torch.stack(outputs).float().to(x.device)