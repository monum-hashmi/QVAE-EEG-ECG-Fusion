import matplotlib.pyplot as plt


conditions = [
    "Valid EEG-ECG\nPairing",
    "Same Subject\nWrong Trial",
    "Cross Subject\nPairing"
]

accuracy = [
    73.91,
    67.95,
    67.15
]


plt.figure(figsize=(8,5))

bars = plt.bar(
    conditions,
    accuracy
)

plt.xlabel("Pairing Condition")
plt.ylabel("Accuracy (%)")

plt.ylim(60, 76)

plt.grid(
    axis="y"
)


# Add values on bars
for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.3,
        f"{value:.2f}%",
        ha="center"
    )


plt.savefig(
    "fig_validity_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Validity graph created successfully")