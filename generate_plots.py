import matplotlib.pyplot as plt


epochs = list(range(1, 21))

train_acc = [
    60.31, 62.99, 64.09, 65.10, 65.50,
    66.12, 66.47, 67.10, 66.92, 67.01,
    67.13, 67.90, 68.04, 68.18, 68.65,
    68.98, 69.09, 69.61, 69.90, 70.55
]

val_acc = [
    62.05, 64.92, 64.61, 66.43, 66.33,
    66.79, 68.18, 67.96, 66.67, 68.12,
    69.14, 68.87, 67.66, 68.03, 61.75,
    69.17, 70.56, 69.05, 70.62, 70.50
]


plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    train_acc,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_acc,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("QVAE EEG-ECG Fusion Training Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/training_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Training curve saved")