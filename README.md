# QVAE EEG--ECG Implementation Starter

This package starts the real implementation phase for the paper:

**Validity-Aware Quantum Variational Autoencoder for Same-Subject EEG--ECG Affective-State Recognition**

## What this package does now

It supports the first real experimental stage:

1. Inspect DREAMER / DEAP files.
2. Build or verify a subject-trial-window feature CSV.
3. Create subject-independent folds.
4. Train classical baselines:
   - Logistic Regression
   - KNN
   - SVM
   - Random Forest
   - XGBoost if installed
5. Evaluate:
   - Accuracy
   - Macro-F1
   - Balanced Accuracy
   - MCC
   - AUROC
6. Generate Overleaf-ready LaTeX result tables.

## What this package does not fake

It does not generate performance numbers unless you run models on actual data.

## Folder structure

```text
data/
    Put dataset files here.
results/
    Outputs will be saved here.
scripts/
    Run these scripts in order.
src/
    Implementation modules.
overleaf/
    LaTeX result table templates.
```

## Installation

Create a fresh Python environment:

```bash
python -m venv .venv
```

Activate it:

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Step 1: Inspect dataset

For DREAMER `.mat` file:

```bash
python scripts/00_inspect_file.py --file data/DREAMER.mat
```

For DEAP `.dat` file:

```bash
python scripts/00_inspect_file.py --file data/s01.dat
```

Send the printed output back before we write the exact DREAMER/DEAP extractor.

## Step 2: Prepare feature CSV

The training scripts expect a CSV like:

```text
subject_id,trial_id,window_id,label,pairing,feature_001,feature_002,...
```

Required columns:

- `subject_id`
- `trial_id`
- `window_id`
- `label`
- `pairing`

Pairing should be one of:

- `eeg_only`
- `ecg_only`
- `correct`
- `wrong_trial`
- `cross_subject`

## Step 3: Create folds

```bash
python scripts/01_make_subject_folds.py --features data/features.csv --out results/folds.json --n_splits 5
```

## Step 4: Train classical baselines

```bash
python scripts/02_train_classical.py --features data/features.csv --folds results/folds.json --out results/classical_results.csv
```

## Step 5: Generate Overleaf table

```bash
python scripts/03_generate_latex_table.py --results results/classical_results.csv --out overleaf/classical_results_table.tex
```

Then copy the generated `.tex` table into your Overleaf Results section.

## Next implementation phase

After dataset inspection confirms exact DREAMER/DEAP structure, we will add:

1. DREAMER extractor
2. DEAP extractor
3. EEG bandpower feature extraction
4. ECG HRV feature extraction
5. CNN-LSTM
6. EEGNet
7. VAE
8. QVAE
9. Correct-pair vs shuffled-pair experiments
