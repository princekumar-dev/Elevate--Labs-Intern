# Week 4 — Logistic Regression

## Objective
Build a binary classification model and evaluate it with proper classification metrics and threshold tuning.

## Dataset
`cancer-data.csv` — Breast Cancer Wisconsin dataset (diagnosis: Malignant / Benign).

## What Was Done
- Cleaned the data: dropped unnamed/identifier columns and mapped diagnosis to a binary target (M = 1, B = 0)
- Performed an **80/20 stratified train/test split** and standardized features with `StandardScaler`
- Trained a `LogisticRegression` model and obtained predicted probabilities
- Evaluated with **confusion matrix**, **classification report** (precision/recall/F1), and **ROC-AUC**
- **Threshold tuning:** re-evaluated the model with a custom threshold of 0.7 instead of the default 0.5
- Plotted the **ROC curve** and **Precision-Recall curve** side by side

## How to Run
```bash
python logistic_regression.py
```

## Requirements
`pandas`, `numpy`, `scikit-learn`, `matplotlib`

## Outputs
ROC and Precision-Recall curves are saved in the [`screenshots/`](screenshots/) folder.
