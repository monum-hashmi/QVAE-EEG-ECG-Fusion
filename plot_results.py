import json
import matplotlib.pyplot as plt
from pathlib import Path


results_path = Path("results")

seeds = [42, 52, 62]

best_accuracy = []

for seed in seeds:
    file = results_path / f"training_history_seed_{seed}.json"

    with open(file, "r") as f:
        history = json.load(f)

    val_acc = history["val_accuracy"]

    best_accuracy.append(max(val_acc))


# -----------------------------
# Multi Seed Stability Graph
# -----------------------------

plt.figure(figsize=(7,4))

plt.plot(
    seeds,
    best_accuracy,
    marker="o"
)

plt.xlabel("Random Seed")
plt.ylabel("Best Validation Accuracy (%)")
plt.title("Multi-Seed Validation Performance")

plt.grid(True)

plt.savefig(
    "fig_seed_stability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------
# Accuracy Comparison
# -----------------------------

models = [
    "Classical\nBottleneck",
    "MLP\nBottleneck",
    "Cross\nAttention",
    "QIVB\nCross Attention"
]

accuracy = [
    60.57,
    71.44,
    72.19,
    72.28
]


plt.figure(figsize=(7,4))

plt.bar(
    models,
    accuracy
)

plt.ylabel("Validation Accuracy (%)")
plt.title("Comparison of Fusion Architectures")

plt.xticks(rotation=20)

plt.savefig(
    "fig_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Graphs generated successfully!")
print(best_accuracy)