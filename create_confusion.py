import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cm = np.array([
    [1727, 279],
    [694, 612]
])


plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - QVAE EEG-ECG Fusion")

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix saved")