import pandas as pd
import numpy as np
import os


INPUT = "data/dreamer_features/dreamer_valence_paired_all.csv"

OUTPUT_DIR = "data/validity_experiments"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


df = pd.read_csv(INPUT)


np.random.seed(42)



# ==========================
# Identify columns
# ==========================

eeg_cols = [
    c for c in df.columns
    if c.startswith("eeg_")
]


ecg_cols = [
    c for c in df.columns
    if c.startswith("ecg_")
]


meta_cols = [
    "subject_id",
    "trial_id",
    "window_id",
    "label",
    "pairing"
]



print("EEG features:", len(eeg_cols))
print("ECG features:", len(ecg_cols))



# ==================================================
# 1. Same Subject Wrong Trial
# ==================================================

wrong_trial = []


for subject in df.subject_id.unique():

    subject_data = df[
        df.subject_id == subject
    ].copy()


    trials = subject_data.trial_id.unique()


    for _, row in subject_data.iterrows():

        possible_trials = [
            t for t in trials
            if t != row.trial_id
        ]


        if len(possible_trials) == 0:
            continue


        wrong_trial_id = np.random.choice(
            possible_trials
        )


        ecg_source = subject_data[
            subject_data.trial_id == wrong_trial_id
        ].sample(
            1
        ).iloc[0]


        new_row = row.copy()


        # replace ECG only

        for col in ecg_cols:

            new_row[col] = ecg_source[col]


        new_row["pairing"] = "same_subject_wrong_trial"


        wrong_trial.append(new_row)



wrong_trial_df = pd.DataFrame(
    wrong_trial
)


wrong_trial_df.to_csv(
    f"{OUTPUT_DIR}/dreamer_wrong_trial.csv",
    index=False
)


print(
    "Saved wrong trial dataset:",
    len(wrong_trial_df)
)



# ==================================================
# 2. Cross Subject Pairing
# ==================================================

cross_subject = []


subjects = df.subject_id.unique()


for _, row in df.iterrows():


    possible_subjects = [
        s for s in subjects
        if s != row.subject_id
    ]


    random_subject = np.random.choice(
        possible_subjects
    )


    ecg_source = df[
        df.subject_id == random_subject
    ].sample(
        1
    ).iloc[0]


    new_row = row.copy()


    for col in ecg_cols:

        new_row[col] = ecg_source[col]


    new_row["pairing"] = "cross_subject"


    cross_subject.append(new_row)



cross_subject_df = pd.DataFrame(
    cross_subject
)


cross_subject_df.to_csv(
    f"{OUTPUT_DIR}/dreamer_cross_subject.csv",
    index=False
)


print(
    "Saved cross subject dataset:",
    len(cross_subject_df)
)



print("Completed validity dataset generation.")