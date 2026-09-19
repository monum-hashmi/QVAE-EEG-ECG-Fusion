import json
import matplotlib.pyplot as plt


history_file = "results/training_history_seed_52.json"

with open(history_file, "r") as f:
    history = json.load(f)


epochs = range(1, len(history["loss"]) + 1)


# ==========================
# Training Loss Curve
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    history["loss"],
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)

plt.savefig(
    "fig_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# ==========================
# Accuracy Curve
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)


plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    "fig_training_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Training graphs created successfully")