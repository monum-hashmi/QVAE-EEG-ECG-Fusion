# QIVB Cross-Attention Fusion for EEG--ECG Emotion Recognition

Implementation repository for:

**Quantum-Inspired Variational Bottleneck with Cross-Attention Fusion for EEG--ECG Based Emotion Recognition**

This project explores multimodal physiological emotion recognition by combining EEG and ECG representations through adaptive cross-attention fusion and a quantum-inspired variational bottleneck.

The proposed framework learns meaningful EEG-ECG interactions while generating compact latent representations for binary valence classification.

---

## Overview

Physiological emotion recognition is challenging because EEG and ECG represent different aspects of emotional responses:

- **EEG** captures neural activity patterns from the central nervous system.
- **ECG** captures autonomic physiological responses.

Traditional multimodal fusion methods often rely on direct feature concatenation, limiting their ability to model complex relationships between heterogeneous physiological signals.

This project introduces:

1. Independent EEG and ECG feature encoders.
2. Cross-attention based multimodal fusion.
3. Quantum-inspired variational bottleneck representation learning.
4. Validity-aware EEG-ECG pairing evaluation.

---

# Main Contributions

## 1. Cross-Attention EEG-ECG Fusion

The proposed framework uses bidirectional cross-attention to learn interactions between modalities.

The module enables:

- EEG features attending to ECG representations.
- ECG features attending to EEG representations.
- Adaptive weighting of multimodal information.

---

## 2. Quantum-Inspired Variational Bottleneck

The fused representation is compressed into a compact latent space.

The bottleneck aims to:

- Remove redundant information.
- Preserve emotion-discriminative features.
- Improve representation robustness.

---

## 3. Comprehensive Evaluation

The framework is evaluated using:

- Ablation studies.
- Multi-seed robustness analysis.
- Confusion matrix evaluation.
- Validity-aware EEG-ECG pairing experiments.

---

# Repository Structure

```text
QVAE-EEG-ECG-Fusion/

├── data/
│   └── Dataset files

├── experiments/
│   └── QIVB cross-attention experiments

├── results/
│   ├── training_history_seed_42.json
│   ├── training_history_seed_52.json
│   ├── training_history_seed_62.json
│   └── Evaluation outputs

├── src/
│   └── Model implementation

├── scripts/
│   └── Utility scripts

├── create_training_graphs.py
├── create_validity_graph.py
├── plot_results.py

└── README.md
```

---

# Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset

The framework is evaluated on:

## DREAMER Dataset

The dataset contains synchronized:

- EEG recordings
- ECG recordings
- Emotion labels

Task:

```
Binary Valence Classification
```

The preprocessing pipeline creates paired EEG-ECG feature representations for multimodal learning.

---

# Proposed Architecture

```
              EEG Input
                  |
            EEG Encoder
                  |
                  |
                  | 
          Cross-Attention Fusion
                  |
                  |
              ECG Encoder
                  |
              ECG Input


                  |
                  ↓

 Quantum-Inspired Variational Bottleneck

                  |
                  ↓

            Classifier

                  |
                  ↓

        Emotion Classification
```

---

# Training Configuration

The model is implemented using:

- Python
- PyTorch
- CUDA acceleration

Training configuration:

- Optimizer: AdamW
- Mixed precision training
- Multiple random seed evaluation

Random seeds:

```
42
52
62
```

---

# Experimental Results

## Ablation Study

| Model | Validation Accuracy |
|---|---:|
| Classical Bottleneck | 60.57% |
| MLP Bottleneck | 71.44% |
| Cross Attention without Variational Bottleneck | 72.19% |
| QIVB Cross Attention | **72.28%** |

---

## Multi-Seed Robustness

The proposed framework maintains stable performance across multiple training runs.

Average validation accuracy:

```
72.01% ± 0.27%
```

---

## Validity-Aware Pairing Analysis

The model was evaluated under different EEG-ECG pairing conditions.

| Pairing Condition | Accuracy |
|---|---:|
| Valid EEG-ECG Pairing | 73.91% |
| Same Subject Wrong Trial | 67.95% |
| Cross Subject Pairing | 67.15% |

The evaluation investigates whether the model learns meaningful physiological correspondence between EEG and ECG modalities.

---

# Visualization Generation

## Training Curves

Generate training accuracy and loss graphs:

```bash
python create_training_graphs.py
```

Outputs:

```
fig_training_accuracy.png
fig_training_loss.png
```

---

## Validity Analysis Graph

Generate validity comparison:

```bash
python create_validity_graph.py
```

Output:

```
fig_validity_analysis.png
```

---

## Available Visualizations

Generated figures include:

```
fig_accuracy_comparison.png

fig_seed_stability.png

fig_training_accuracy.png

fig_training_loss.png

fig_validity_analysis.png
```

---

# Reproducibility

To reproduce experiments:

1. Prepare DREAMER dataset.
2. Generate EEG-ECG feature representations.
3. Train the QIVB Cross-Attention model.
4. Evaluate using fixed random seeds.
5. Generate result visualizations.

---

# Future Improvements

Planned research directions:

- Raw EEG and ECG signal visualization.
- Cross-attention weight visualization.
- Latent space analysis using t-SNE/UMAP.
- Subject-independent evaluation.
- Transformer-based physiological encoders.
- Larger multimodal physiological datasets.
- Real-time emotion recognition systems.

---



Machine Learning Project
