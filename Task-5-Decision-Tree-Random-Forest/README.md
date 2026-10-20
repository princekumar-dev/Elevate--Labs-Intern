# Week 5 — Decision Tree & Random Forest

## Objective
Learn tree-based models for classification, understand overfitting, and compare models using cross-validation.

## Dataset
`heart.csv` — Heart disease dataset (target: disease / no disease).

## What Was Done
- Trained a `DecisionTreeClassifier` and visualized the tree (max depth 3)
- **Overfitting check:** compared train vs test accuracy of an unconstrained tree
- Controlled overfitting with `max_depth=5` and `min_samples_split=10`
- Trained a `RandomForestClassifier` (100 trees) and compared test accuracy
- Ranked **feature importances** from the random forest and plotted them
- Evaluated both models with **5-fold cross-validation**

## How to Run
```bash
python decision_tree_random_forest.py
```

## Requirements
`pandas`, `numpy`, `scikit-learn`, `matplotlib`

## Outputs
Tree visualization and feature-importance plots are saved in the [`screenshots/`](screenshots/) folder.
