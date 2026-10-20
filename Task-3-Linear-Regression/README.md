# Week 3 — Linear Regression

## Objective
Build and evaluate a linear regression model to predict a continuous target variable.

## Dataset
`Titanic-Dataset.csv` — used to predict the **Fare** from `Age`, `Pclass`, `SibSp`, and `Parch`.

## What Was Done
- Selected relevant features and dropped rows with missing values
- Split data into **80% train / 20% test** (`random_state=42`)
- Trained a `LinearRegression` model from scikit-learn
- Evaluated using **MAE**, **MSE**, and **R²**
- Printed model coefficients and intercept, with interpretation of each feature's effect
- Plotted **Actual vs Predicted Fare** with the ideal diagonal reference line

## How to Run
```bash
python linear_regression.py
```

## Requirements
`pandas`, `numpy`, `scikit-learn`, `matplotlib`

## Outputs
Actual-vs-predicted plot is saved in the [`screenshots/`](screenshots/) folder.
