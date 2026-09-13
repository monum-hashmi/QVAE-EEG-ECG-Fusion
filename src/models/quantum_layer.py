import torch
import torch.nn as nn
import pennylane as qml


class QuantumLayer(nn.Module):

    def __init__(self, input_dim=128, n_qubits=4):
        super().__init__()

        self.n_qubits = n_qubits

        self.encoder = nn.Linear(
            input_dim,
            n_qubits
        )

        self.weights = nn.Parameter(
            torch.randn(n_qubits)
        )

        dev = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):

            for i in range(n_qubits):
                qml.RY(
                    inputs[i],
                    wires=i
                )

            for i in range(n_qubits - 1):
                qml.CNOT(
                    wires=[i, i + 1]
                )

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

        x = self.encoder(x)

        outputs = []

        for sample in x:
            outputs.append(
                torch.stack(
                    self.circuit(
                        sample,
                        self.weights
                    )
                )
            )

        return torch.stack(outputs).float()