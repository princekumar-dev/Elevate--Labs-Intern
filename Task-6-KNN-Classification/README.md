# Week 6 — K-Nearest Neighbors (KNN) Classification

## Objective
Implement a KNN classifier, experiment with different K values, and visualize decision boundaries.

## Dataset
Iris flower dataset (loaded from scikit-learn's built-in `load_iris`; a reference copy is provided as `Iris.csv`). Only the first two features are used so the decision boundary can be visualized in 2D.

## What Was Done
- Split data into **80% train / 20% test** and normalized features with `StandardScaler`
- Trained KNN models for **K = 1, 3, 5, 7, 9, 15** and tracked test accuracy for each
- Selected the best K automatically and retrained the final model
- Evaluated with **accuracy** and a **confusion matrix** (plotted as a heatmap)
- Visualized **decision boundaries** of the final KNN model on the 2D feature space

## How to Run
```bash
python knn_classification.py
```

## Requirements
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

## Outputs
Confusion matrix and decision-boundary plots are saved in the [`screenshots/`](screenshots/) folder.
