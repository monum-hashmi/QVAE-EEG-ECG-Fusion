import pandas as pd

CSV = "data/dreamer_features/dreamer_valence_paired_all.csv"

df = pd.read_csv(CSV)

df = df[
    df["pairing"] == "correct"
].reset_index(drop=True)

eeg_cols = [
    c for c in df.columns
    if c.startswith("eeg_")
]

print("=" * 70)
print("EEG FEATURE STRUCTURE")
print("=" * 70)

print("Total EEG features:", len(eeg_cols))

print("\nEEG feature order:")

for i, col in enumerate(eeg_cols):
    print(f"{i:02d}: {col}")

print("\nFeatures grouped by channel:")

channels = {}

for col in eeg_cols:

    parts = col.split("_")

    channel = parts[2]
    band = parts[3]

    channels.setdefault(
        channel,
        []
    ).append(
        band
    )

for channel, bands in channels.items():

    print(
        channel,
        "->",
        bands
    )

print("\nFeatures grouped by band:")

bands = {}

for col in eeg_cols:

    parts = col.split("_")

    band = parts[3]

    bands.setdefault(
        band,
        []
    ).append(
        col
    )

for band, features in bands.items():

    print(
        band,
        "->",
        len(features),
        "features"
    )

print("\nCompleted.")